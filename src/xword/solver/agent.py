"""The agent loop.

A crossword is not a list of independent trivia questions, and treating it as one
is why a raw language model does badly at it: ask a model 76 clues, take its best
answer for each, and the grid will not agree with itself. What makes this an
*agent* rather than a prompt is the loop it runs around the model:

    ingest -> propose -> fuse -> propagate -> commit -> critique -> repair -> verify

1. **propose**   Ask the model for several ranked answers per clue, with
                 probabilities, in batches.
2. **fuse**      Pool those with a pattern-matching lexicon into one calibrated
                 distribution per entry, explicitly reserving mass for "the answer
                 is not in my list".
3. **propagate** Run loopy belief propagation over the grid so that every entry's
                 opinion is tempered by what its crossings believe.
4. **commit**    Search for the highest-scoring set of real words that fit
                 together, not just the per-clue argmax.
5. **critique**  Find where the committed grid is weak: unresolved conflicts, low
                 posterior margins, entries filled from the lexicon with no clue
                 support, letters BP is unsure of.
6. **repair**    Re-ask the model about exactly those clues, now supplying the
                 crossing letters it has become confident about, and escalating
                 the worst clues to a stronger model with a wordplay-analysis
                 prompt. Then propagate and commit again.

That feedback edge -- step 6 handing partial letters back to step 1 -- is the
whole point. It is how a human solves too: the clue you could not get becomes
easy once three of its letters are filled in.

The loop stops when a round produces nothing worth re-asking, when confidence
crosses a threshold, or when a budget (rounds, API calls, wall clock) runs out.
It always returns a completely filled grid: an unfilled square scores zero, so a
low-confidence guess strictly dominates a blank.

A solve can also be stopped from outside, by a ``cancel`` predicate polled at
the three points that bracket the spending: the top of a round, the moment the
propose pass returns, and after commit. Stopping hands back the best grid found
so far rather than nothing -- the caller asked the agent to stop working, not to
discard work it has already paid Anthropic for.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from xword.config import AgentConfig, estimate_cost
from xword.core.beliefs import SlotBeliefs
from xword.core.grid import GridIndex, index_puzzle
from xword.core.types import (
    LETTER_INDEX,
    WILDCARD,
    AgentEvent,
    Candidate,
    Cell,
    ClueRequest,
    Fill,
    Puzzle,
    Slot,
    SlotOutcome,
    SolveResult,
    SolveStats,
)

# Typing-only. The agent never builds or reads a record -- it only forwards the
# caller's observer to the candidate sources, which is where records are made --
# so importing the type at runtime would buy an import of the web package for a
# signature annotation.
if TYPE_CHECKING:
    from xword.web.trace import LLMCallRecord

# --------------------------------------------------------------------------- #
# Critique
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Weakness:
    """One entry the agent is not happy about, and why.

    ``severity`` is what the repair budget is spent on, highest first. The
    ``reason`` is carried through to the trace so a human reading the log can see
    *why* the agent chose to re-ask a particular clue.
    """

    slot_id: str
    severity: float
    reason: str
    pattern: str

    @property
    def known_letters(self) -> int:
        return sum(1 for ch in self.pattern if ch != WILDCARD)


def _entry_confidence(
    slot: Slot,
    answer: str | None,
    cell_conf: Mapping[Cell, float],
) -> float:
    """Confidence in a whole entry: the geometric mean of its letters'.

    A geometric mean rather than an arithmetic one because one bad letter makes
    the entry wrong -- averaging would let five confident letters hide a sixth
    the agent has no idea about.
    """
    if answer is None:
        return 0.0
    probs = [max(cell_conf.get(cell, 0.0), 1e-9) for cell in slot.cells]
    if not probs:
        return 0.0
    return float(np.exp(np.mean(np.log(probs))))


def _tag_round(source: object, rnd: int) -> None:
    """Tell a candidate source which round it is answering for.

    ``round_hint`` is a plain int attribute on ``LLMCandidateSource``, read only
    when it stamps a per-call trace record. But sources are injectable, and the
    doubles the tests inject are frequently slotted dataclasses where a bare
    assignment raises ``AttributeError``. Failing a solve in order to label a
    trace event would be a poor trade, so a source that cannot hold the hint
    simply does not get one.
    """
    with contextlib.suppress(AttributeError, TypeError):
        source.round_hint = rnd  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #


class CrosswordAgent:
    """Solves one puzzle at a time.

    Construct once and reuse across puzzles: the clue cache, the lexicon, and the
    HTTP connection pool are all worth keeping warm.

    ``cancel`` and ``on_llm_call`` exist for callers that run a solve somewhere
    the user can still reach it -- the web session runner runs one per thread.
    ``cancel`` is asked between phases whether to stop; ``on_llm_call`` receives
    one record per request to the model. Both default to absent, so the CLI and
    the evaluation harness behave exactly as before.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        lexicon: object | None = None,
        llm: object | None = None,
        on_event: Callable[[AgentEvent], None] | None = None,
        cancel: Callable[[], bool] | None = None,
        on_llm_call: Callable[[LLMCallRecord], None] | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self._lexicon = lexicon
        self._llm = llm
        self._llm_injected = llm is not None
        self._hard_llm = None
        self._cache = None
        self._usage_base: dict | None = None
        self._on_event = on_event
        self._cancel = cancel
        self._on_llm_call = on_llm_call

        # Exposed after a solve so the evaluation harness can measure *candidate
        # coverage* -- whether the true answer was ever proposed -- without the
        # agent itself ever seeing the reference solution.
        self.last_beliefs: SlotBeliefs | None = None
        self.last_index: GridIndex | None = None

    # -- lazily built collaborators ---------------------------------------- #

    @property
    def lexicon(self):
        if self._lexicon is None:
            from xword.lexicon.store import Lexicon

            self._lexicon = Lexicon.default() if self.config.use_lexicon else Lexicon.empty()
        return self._lexicon

    def _build_llm(self, model: str, *, batch_size: int):
        from xword.candidates.cache import ClueCache
        from xword.candidates.llm import LLMCandidateSource

        if self._cache is None:
            self._cache = ClueCache(self.config.resolved_cache_path)
        # Every source this agent builds shares the one clue cache (see
        # ``hard_llm``) and now also the one trace observer and the one cancel
        # predicate: a stopped session has to stop the escalation pass too, and
        # a reader of the trace should not need to know which model answered in
        # order to find the call.
        return LLMCandidateSource(
            model=model,
            k=self.config.candidates_per_clue,
            batch_size=batch_size,
            max_concurrency=self.config.max_concurrency,
            temperature=self.config.temperature,
            cache=self._cache,
            on_call=self._on_llm_call,
            cancel=self._cancel,
        )

    @property
    def llm(self):
        if self._llm is None and self._uses_llm:
            self._llm = self._build_llm(self.config.model, batch_size=self.config.batch_size)
        return self._llm

    @property
    def hard_llm(self):
        """A second source on the stronger model, for entries the first pass failed.

        Built separately rather than by swapping a field, because the two share
        one clue cache and the cache key includes the model -- so an escalated
        answer never overwrites the cheap model's cached answer for the same
        clue, and a re-run reuses both.

        When an injected client is in use (tests) or the two models are the same,
        this is just the ordinary source: there is nothing to escalate to.
        """
        if not self._uses_llm:
            return None
        if self._llm_injected or self.config.hard_clue_model == self.config.model:
            return self.llm
        if self._hard_llm is None:
            self._hard_llm = self._build_llm(self.config.hard_clue_model, batch_size=1)
        return self._hard_llm

    @property
    def _uses_llm(self) -> bool:
        return self.config.candidates_per_clue > 0 and self.config.llm_weight > 0

    def _active_sources(self):
        """``(model, source)`` for every LLM source this solve has built.

        Usage has to be summed across both, or a solve that escalated would
        report only the cheap model's tokens and under-state its own cost.
        """
        out = []
        if self._llm is not None:
            out.append((getattr(self._llm, "model", self.config.model), self._llm))
        if self._hard_llm is not None and self._hard_llm is not self._llm:
            out.append((self.config.hard_clue_model, self._hard_llm))
        return out

    def _usage_snapshot(self) -> dict[str, tuple[int, int, int, int, float]]:
        """Cumulative counters for each source, as of right now."""
        snap = {}
        for model, source in self._active_sources():
            usage = getattr(source, "usage", None)
            if usage is None:
                continue
            snap[model] = (
                usage.calls,
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_hits,
                usage.cost_usd,
            )
        return snap

    def _apply_usage(self, stats: SolveStats) -> None:
        """Record usage for *this puzzle only*.

        ``LLMCandidateSource.usage`` accumulates over the lifetime of the
        source, and an agent is meant to be reused across puzzles so the clue
        cache and connection pool stay warm. Reading the counters directly would
        therefore charge every puzzle for all its predecessors too -- the second
        puzzle in a suite would report twice its real cost, the tenth ten times.
        Differencing against a snapshot taken at the start of the solve is what
        keeps per-puzzle cost honest.
        """
        base = self._usage_base or {}
        calls = tokens_in = tokens_out = hits = 0
        cost = 0.0
        for model, source in self._active_sources():
            usage = getattr(source, "usage", None)
            if usage is None:
                continue
            was = base.get(model, (0, 0, 0, 0, 0.0))
            d_calls = usage.calls - was[0]
            d_in = usage.input_tokens - was[1]
            d_out = usage.output_tokens - was[2]
            d_hits = usage.cache_hits - was[3]
            d_cost = usage.cost_usd - was[4]
            calls += d_calls
            tokens_in += d_in
            tokens_out += d_out
            hits += d_hits
            cost += d_cost if d_cost > 0 else estimate_cost(model, d_in, d_out)

        stats.llm_calls = calls
        stats.input_tokens = tokens_in
        stats.output_tokens = tokens_out
        stats.cache_hits = hits
        stats.cost_usd = cost

    # -- events ------------------------------------------------------------ #

    def _emit(
        self,
        trace: list[AgentEvent],
        kind: str,
        rnd: int,
        message: str,
        **data: float | int | str,
    ) -> None:
        event = AgentEvent(kind=kind, round=rnd, message=message, data=data)  # type: ignore[arg-type]
        trace.append(event)
        if self._on_event is not None:
            self._on_event(event)

    # -- cancellation ------------------------------------------------------ #

    def _cancelled(self) -> bool:
        """Whether the caller has asked this solve to stop.

        Polled rather than pushed. The predicate is cheap -- in practice a
        ``threading.Event.is_set`` -- and polling at phase boundaries means no
        phase has to be written to be interruptible half way through.
        """
        return False if self._cancel is None else bool(self._cancel())

    def _emit_stopped(self, trace: list[AgentEvent], rnd: int, stage: str) -> None:
        """Record that a stop request, not a budget or a threshold, ended the loop.

        Worth distinguishing in the trace: a reader who sees the loop end after
        one round needs to know whether the agent decided it was finished or a
        human pressed the button.
        """
        self._emit(
            trace,
            "verify",
            rnd,
            f"stopped by request during {stage}; returning the best grid so far",
            stopped=1,
            stage=stage,
        )

    # -- the loop ---------------------------------------------------------- #

    def solve(self, puzzle: Puzzle) -> SolveResult:
        from xword.solver.beliefs import marginal_pattern, run_bp
        from xword.solver.search import (
            SearchConfig,
            complete_from_marginals,
            repair,
            solve_assignment,
        )

        cfg = self.config
        started = time.perf_counter()
        trace: list[AgentEvent] = []
        stats = SolveStats()

        # Snapshot cumulative usage so this puzzle is charged only its own.
        self._usage_base = self._usage_snapshot()

        index = index_puzzle(puzzle)
        self.last_index = index
        self._emit(
            trace,
            "ingest",
            0,
            f"{puzzle.height}x{puzzle.width} grid, {len(puzzle.slots)} entries, "
            f"{len(index.crossings)} crossings",
            entries=len(puzzle.slots),
            crossings=len(index.crossings),
            open_cells=len(puzzle.open_cells),
        )

        search_config = SearchConfig(
            beam_width=cfg.beam_width,
            lexicon_topk=cfg.lexicon_topk,
            discrepancy_limit=cfg.discrepancy_limit,
            max_seconds=cfg.search_seconds,
            seed=cfg.seed,
        )

        beliefs = SlotBeliefs()
        best_assignment = None
        best_bp = None
        deadline = started + cfg.wall_clock_budget

        for rnd in range(max(1, cfg.max_rounds)):
            # The cheapest possible place to notice: nothing has been spent on
            # this round yet, so a stop between rounds costs nothing at all.
            if self._cancelled():
                self._emit_stopped(trace, rnd, "round start")
                break

            targets = self._targets_for_round(rnd, puzzle, best_assignment, best_bp, trace)
            if rnd > 0 and not targets:
                self._emit(trace, "critique", rnd, "nothing left worth re-asking; stopping")
                break

            # ---- propose ------------------------------------------------- #
            requests = self._build_requests(puzzle, targets, best_bp, marginal_pattern)
            fresh = self._propose(requests, rnd, trace, stats, escalate=rnd > 0)
            beliefs = beliefs.merged_with(fresh) if beliefs.candidates else fresh
            self.last_beliefs = beliefs

            # The merge above is bookkeeping on candidates already paid for, so
            # it happens before this check. Propagate and commit are seconds of
            # further work, and not worth doing for a grid nobody is waiting on.
            if self._cancelled():
                self._emit_stopped(trace, rnd, "propose")
                break

            # ---- propagate ----------------------------------------------- #
            bp = run_bp(
                puzzle,
                index,
                beliefs,
                iterations=cfg.bp_iterations,
                damping=cfg.bp_damping,
                tol=cfg.bp_tolerance,
            )
            self._emit(
                trace,
                "propagate",
                rnd,
                f"belief propagation {'converged' if bp.converged else 'hit the iteration cap'} "
                f"after {bp.iterations} iterations (delta {bp.max_delta:.2e})",
                converged=int(bp.converged),
                iterations=bp.iterations,
            )

            # ---- commit -------------------------------------------------- #
            assignment = solve_assignment(
                puzzle, index, beliefs, bp, lexicon=self.lexicon, config=search_config
            )
            assignment = repair(
                puzzle,
                index,
                beliefs,
                bp,
                assignment,
                lexicon=self.lexicon,
                config=search_config,
            )
            assignment = complete_from_marginals(puzzle, index, bp, assignment)
            self._emit(
                trace,
                "commit",
                rnd,
                f"filled {len(assignment.slot_answers)}/{len(puzzle.slots)} entries "
                f"(score {assignment.score:.1f}, {len(assignment.conflicts)} unresolved)",
                entries_filled=len(assignment.slot_answers),
                conflicts=len(assignment.conflicts),
                score=round(assignment.score, 3),
            )

            if best_assignment is None or self._round_objective(
                puzzle, assignment
            ) > self._round_objective(puzzle, best_assignment):
                best_assignment, best_bp = assignment, bp

            stats.rounds = rnd + 1

            # ---- stop conditions ----------------------------------------- #
            # A stop request outranks the budget tests below: each of those
            # decides whether *more* work is worth doing, and that question is
            # moot once the answer is no more work at all.
            if self._cancelled():
                self._emit_stopped(trace, rnd, "commit")
                break
            confidence = self._grid_confidence(puzzle, best_assignment, best_bp)
            if confidence >= cfg.stop_when_confident and not best_assignment.conflicts:
                self._emit(
                    trace,
                    "verify",
                    rnd,
                    f"confident at {confidence:.3f}; stopping early",
                    confidence=round(confidence, 4),
                )
                break
            if stats.llm_calls >= cfg.max_llm_calls:
                self._emit(trace, "verify", rnd, "API call budget exhausted")
                break
            if time.perf_counter() > deadline:
                self._emit(trace, "verify", rnd, "wall-clock budget exhausted")
                break

        if best_assignment is None or best_bp is None:
            # Every round that gets past ``propose`` reaches ``commit``, so this
            # branch has exactly one route into it: cancellation inside the
            # first propose pass, before any grid existed to keep. That is a
            # user action, and a user action must not surface as an
            # AssertionError, so it gets its own explicit result.
            result = self._empty_result(puzzle, beliefs, stats, trace, started)
        else:
            result = self._finalise(
                puzzle, best_assignment, best_bp, beliefs, stats, trace, started
            )
        self._emit(
            trace,
            "done",
            stats.rounds,
            f"done in {stats.wall_seconds:.1f}s, {stats.llm_calls} API calls, "
            f"${stats.cost_usd:.4f} estimated",
            seconds=round(stats.wall_seconds, 2),
            calls=stats.llm_calls,
        )
        return result

    # -- round planning ---------------------------------------------------- #

    def _targets_for_round(
        self,
        rnd: int,
        puzzle: Puzzle,
        assignment,
        bp,
        trace: list[AgentEvent],
    ) -> list[str] | None:
        """``None`` on the first round (ask about everything), otherwise the
        entries worth re-asking."""
        if rnd == 0 or assignment is None or bp is None:
            return None
        weaknesses = self.critique(puzzle, assignment, bp)
        if not weaknesses:
            return []
        chosen = weaknesses[: self.config.max_repair_slots]
        summary = ", ".join(f"{w.slot_id}({w.reason})" for w in chosen[:6])
        self._emit(
            trace,
            "critique",
            rnd,
            f"{len(weaknesses)} weak entries; re-asking {len(chosen)}: {summary}"
            + ("..." if len(chosen) > 6 else ""),
            weak=len(weaknesses),
            retrying=len(chosen),
        )
        return [w.slot_id for w in chosen]

    def critique(self, puzzle: Puzzle, assignment, bp) -> list[Weakness]:
        """Rank entries by how much the agent distrusts its own answer.

        Four distinct smells, because they call for the same remedy (ask again
        with the letters we now have) but arise differently:

        * an entry the search could not satisfy at all;
        * an entry whose answer no candidate source ever proposed -- it was
          forced in by its crossings, so the clue was never really read;
        * an entry whose belief-propagation margin over the runner-up is thin;
        * an entry whose letters BP itself is unsure about.
        """
        cell_conf = self._cell_confidence(puzzle, assignment, bp)
        weaknesses: list[Weakness] = []
        proposed = {sid: set(self.last_beliefs.answers(sid)) for sid in (self.last_beliefs.slot_ids if self.last_beliefs else ())}

        for slot in puzzle.slots:
            answer = assignment.slot_answers.get(slot.id)
            conf = _entry_confidence(slot, answer, cell_conf)
            margin = bp.slot_margin(slot.id)

            if slot.id in assignment.conflicts or answer is None:
                reason, severity = "unfilled", 1.0
            elif answer not in proposed.get(slot.id, set()):
                reason, severity = "unproposed", 0.9 - 0.5 * conf
            elif conf < self.config.repair_threshold:
                reason, severity = "low-confidence", 0.8 - conf
            elif margin < 0.15:
                reason, severity = "thin-margin", 0.5 - margin
            else:
                continue

            # An entry with no crossing letters yet gains nothing from being
            # re-asked -- the second ask would be identical to the first.
            pattern = self._confident_pattern(slot, cell_conf, assignment)
            if reason != "unfilled" and pattern.count(WILDCARD) == slot.length:
                continue

            weaknesses.append(
                Weakness(slot_id=slot.id, severity=severity, reason=reason, pattern=pattern)
            )

        weaknesses.sort(key=lambda w: (-w.severity, w.slot_id))
        return weaknesses

    def _confident_pattern(self, slot: Slot, cell_conf: Mapping[Cell, float], assignment) -> str:
        """The letters worth telling the model about on the next ask.

        Only letters the agent is genuinely confident in are passed back. Feeding
        a shaky letter to the model as a hard constraint is actively harmful: it
        will loyally produce a wrong answer that fits the wrong letter, and the
        error becomes self-confirming.
        """
        threshold = max(self.config.repair_threshold, 0.75)
        out = []
        for cell in slot.cells:
            letter = assignment.fill.get(cell)
            if letter is not None and cell_conf.get(cell, 0.0) >= threshold:
                out.append(letter)
            else:
                out.append(WILDCARD)
        return "".join(out)

    def _build_requests(
        self,
        puzzle: Puzzle,
        targets: list[str] | None,
        bp,
        marginal_pattern,
    ) -> list[ClueRequest]:
        index = self.last_index or index_puzzle(puzzle)
        wanted = puzzle.slots if targets is None else [
            s for s in puzzle.slots if s.id in set(targets)
        ]
        meta = {k: str(v) for k, v in puzzle.meta.items() if k in {"title", "dow", "difficulty"}}

        requests: list[ClueRequest] = []
        for slot in wanted:
            pattern = None
            if bp is not None:
                pattern = marginal_pattern(bp, slot, threshold=0.9)
                if pattern.count(WILDCARD) == slot.length:
                    pattern = None
            crossing_clues = tuple(
                index.slot_by_id[nid].clue
                for nid in index.neighbours.get(slot.id, ())[:4]
                if index.slot_by_id[nid].clue
            )
            requests.append(
                ClueRequest(
                    slot_id=slot.id,
                    clue=slot.clue,
                    length=slot.length,
                    direction=slot.direction,
                    pattern=pattern,
                    puzzle_meta=meta,
                    crossing_clues=crossing_clues,
                )
            )
        return requests

    # -- generation -------------------------------------------------------- #

    def _propose(
        self,
        requests: Sequence[ClueRequest],
        rnd: int,
        trace: list[AgentEvent],
        stats: SolveStats,
        *,
        escalate: bool,
    ) -> SlotBeliefs:
        from xword.candidates.fusion import FusionConfig, fuse
        from xword.candidates.lexicon_source import LexiconCandidateSource

        cfg = self.config
        per_source: dict[str, dict[str, list[Candidate]]] = {}

        if self._uses_llm and requests:
            hard_mode = escalate and cfg.escalate_hard_clues
            llm = self.hard_llm if hard_mode else self.llm
            # Stamp the round on whichever of the two sources is about to run,
            # so its trace records are attributed to this round rather than to
            # whatever round the source last answered for.
            _tag_round(llm, rnd)
            answers = (
                llm.propose_hard(list(requests)) if hard_mode else llm.propose(list(requests))
            )
            per_source["llm"] = answers

            self._apply_usage(stats)

            found = sum(1 for v in answers.values() if v)
            self._emit(
                trace,
                "propose",
                rnd,
                f"asked {llm.model if hasattr(llm, 'model') else cfg.model} about "
                f"{len(requests)} clue(s)"
                + (" in wordplay-analysis mode" if hard_mode else "")
                + f"; got candidates for {found}",
                asked=len(requests),
                answered=found,
                calls=stats.llm_calls,
            )

        if cfg.use_lexicon and requests:
            # Without the model, the lexicon is the only source, so it must be
            # allowed to propose from an unconstrained pattern even though that
            # is mostly noise -- that is exactly what the lexicon-only ablation
            # is meant to measure.
            source = LexiconCandidateSource(
                self.lexicon,
                limit=cfg.lexicon_topk,
                require_pattern=self._uses_llm,
            )
            per_source["lexicon"] = source.propose(list(requests))

        fusion = FusionConfig(
            source_weights={"llm": cfg.llm_weight, "lexicon": cfg.lexicon_weight},
        )
        beliefs = fuse(per_source, list(requests), config=fusion, lexicon=self.lexicon)
        self._emit(
            trace,
            "fuse",
            rnd,
            f"fused {len(per_source)} source(s) into distributions over "
            f"{len(beliefs.slot_ids)} entries",
            sources=len(per_source),
            entries=len(beliefs.slot_ids),
        )
        return beliefs

    # -- confidence and finalisation --------------------------------------- #

    def _cell_confidence(self, puzzle: Puzzle, assignment, bp) -> dict[Cell, float]:
        """How much belief the final marginals put on the letter actually written.

        Using the marginal of the *committed* letter rather than the max marginal
        is what makes this number honest: when search overrides BP's favourite
        letter to satisfy a crossing word, the confidence reported for that cell
        drops accordingly.
        """
        out: dict[Cell, float] = {}
        for cell in puzzle.open_cells:
            letter = assignment.fill.get(cell)
            marginal = bp.cell_marginals.get(cell)
            if letter is None or marginal is None:
                out[cell] = 0.0
                continue
            idx = LETTER_INDEX.get(letter)
            out[cell] = float(marginal[idx]) if idx is not None else 0.0
        return out

    #: Quantile used for the early-stop test. Low on purpose -- see below.
    WEAKEST_QUANTILE = 0.05

    #: Log-probability charged for an entry left without a word, when comparing
    #: one round against another. Roughly log(6e-6): worse than any real
    #: candidate, so filling an entry badly still beats not filling it.
    UNFILLED_PENALTY = -12.0

    def _round_objective(self, puzzle: Puzzle, assignment) -> float:
        """Compare two rounds on a footing that does not reward blank squares.

        ``Assignment.score`` sums log-probabilities over the entries that got a
        word. Log-probabilities are negative, so an assignment that fills fewer
        entries has fewer negative terms and therefore a *higher* total. Picking
        the maximum of that raw score systematically prefers the emptiest grid,
        which silently discards exactly the repair rounds this loop exists to
        run. Charging every unfilled entry a fixed penalty restores the
        comparison.
        """
        unfilled = max(0, len(puzzle.slots) - len(assignment.slot_answers))
        return assignment.score + self.UNFILLED_PENALTY * unfilled

    def _grid_confidence(self, puzzle: Puzzle, assignment, bp) -> float:
        """How confident the agent is about the *weakest* part of the grid.

        Deliberately not the mean. A mean over ~180 squares is dominated by the
        many the agent is certain about and barely moves when a handful are
        hopeless: on a real Monday puzzle it read 1.000 while two squares were
        wrong, so the early-stop fired before the repair round that would have
        fixed them. Since a crossword is scored all-or-nothing, the quantity
        that decides whether to keep working is the state of the worst squares,
        not the average one.
        """
        conf = self._cell_confidence(puzzle, assignment, bp)
        if not conf:
            return 0.0
        return float(np.quantile(np.fromiter(conf.values(), dtype=float), self.WEAKEST_QUANTILE))

    def _empty_result(
        self,
        puzzle: Puzzle,
        beliefs: SlotBeliefs,
        stats: SolveStats,
        trace: list[AgentEvent],
        started: float,
    ) -> SolveResult:
        """The result for a solve stopped before it ever committed a grid.

        ``_finalise`` cannot serve this case: it reads ``assignment.fill`` and
        ``bp.cell_marginals``, and here there is neither an assignment nor a
        belief-propagation run to read. Handing it invented empty stand-ins
        would put a fake ``Assignment`` into the one code path that is hardest
        to exercise, so this is a separate and deliberately dull path instead.

        Empty rather than absent: every entry is reported with ``answer=None``
        and confidence 0.0, which is what a caller scoring or rendering the
        result should see for a square nobody guessed. Usage is still applied,
        because a session cancelled after its first batch went out did cost
        money and must not report itself as free.
        """
        self._apply_usage(stats)
        stats.wall_seconds = time.perf_counter() - started
        stats.notes.update(
            {
                "cells_filled": 0.0,
                "cells_total": float(len(puzzle.open_cells)),
                "entries_from_crossings": 0.0,
                "mean_cell_confidence": 0.0,
                "bp_converged": 0.0,
                "search_score": 0.0,
                # Not one conflict, because no search ever ran to have one. The
                # flag below is what tells a reader why the grid is blank.
                "unresolved": 0.0,
                "stopped_before_commit": 1.0,
            }
        )

        outcomes = {
            slot.id: SlotOutcome(
                slot_id=slot.id,
                clue=slot.clue,
                answer=None,
                confidence=0.0,
                source="none",
                # Candidates that arrived before the stop are still counted:
                # they are what the round had already bought.
                considered=len(beliefs.answers(slot.id)),
            )
            for slot in puzzle.slots
        }

        return SolveResult(
            puzzle_id=puzzle.id,
            fill=Fill({}),
            cell_confidence={},
            slots=outcomes,
            stats=stats,
            trace=trace,
        )

    def _finalise(
        self,
        puzzle: Puzzle,
        assignment,
        bp,
        beliefs: SlotBeliefs,
        stats: SolveStats,
        trace: list[AgentEvent],
        started: float,
    ) -> SolveResult:
        cell_conf = self._cell_confidence(puzzle, assignment, bp)

        outcomes: dict[str, SlotOutcome] = {}
        for slot in puzzle.slots:
            answer = assignment.fill.answer_for(slot)
            proposed = set(beliefs.answers(slot.id))
            if answer is None:
                source = "none"
            elif answer in proposed:
                source = "proposed"
            else:
                source = "crossings"
            outcomes[slot.id] = SlotOutcome(
                slot_id=slot.id,
                clue=slot.clue,
                answer=answer,
                confidence=_entry_confidence(slot, answer, cell_conf),
                source=source,
                considered=len(proposed),
            )

        stats.wall_seconds = time.perf_counter() - started
        filled = sum(1 for c in puzzle.open_cells if assignment.fill.get(c) is not None)
        stats.notes.update(
            {
                "cells_filled": float(filled),
                "cells_total": float(len(puzzle.open_cells)),
                "entries_from_crossings": float(
                    sum(1 for o in outcomes.values() if o.source == "crossings")
                ),
                "mean_cell_confidence": float(np.mean(list(cell_conf.values())) if cell_conf else 0.0),
                "bp_converged": float(bp.converged),
                "search_score": float(assignment.score),
                "unresolved": float(len(assignment.conflicts)),
            }
        )

        self._emit(
            trace,
            "verify",
            stats.rounds,
            f"grid complete: {filled}/{len(puzzle.open_cells)} squares filled, "
            f"mean confidence {stats.notes['mean_cell_confidence']:.3f}",
            filled=filled,
        )

        return SolveResult(
            puzzle_id=puzzle.id,
            fill=Fill(dict(assignment.fill.letters)),
            cell_confidence=cell_conf,
            slots=outcomes,
            stats=stats,
            trace=trace,
        )


def solve_puzzle(
    puzzle: Puzzle,
    config: AgentConfig | None = None,
    *,
    on_event: Callable[[AgentEvent], None] | None = None,
) -> SolveResult:
    """One-shot convenience wrapper used by the CLI and the tests."""
    return CrosswordAgent(config, on_event=on_event).solve(puzzle)


__all__ = ["CrosswordAgent", "Weakness", "solve_puzzle"]

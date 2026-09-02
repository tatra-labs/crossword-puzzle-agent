"""Discrete search: turning per-letter beliefs into an actual filled grid.

Belief propagation hands the solver marginals -- what each square probably is,
considered one square at a time. A crossword is not solved one square at a
time: every entry has to be a real word, and every crossing has to satisfy two
words at once. This module does that discrete part. It chooses one answer per
entry so that all crossings agree, maximising total log-probability.

The pipeline is three stages, each usable on its own:

``build_pools``
    Per-entry shortlists, merging the fused candidate list (re-priced by the BP
    slot posterior) with lexicon words that BP's own letters already like.
``solve_assignment``
    A beam over entries ordered by BP's confidence, forward-checked against the
    crossings, with a limited-discrepancy retry when the beam cannot fill
    everything.
``repair`` / ``complete_from_marginals``
    Local search that tears out a bad entry with its crossings and re-fills the
    neighbourhood, then a guarantee that no square is left blank.

Two properties are non-negotiable throughout. Everything is under a wall-clock
budget -- this is a real-time system, and a late perfect answer is a failure --
and everything is deterministic for a fixed seed, so an evaluation run can be
reproduced exactly.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from xword.core.beliefs import SlotBeliefs
from xword.core.grid import GridIndex, pattern_from_letters
from xword.core.types import ALPHABET, LETTER_INDEX, WILDCARD, Cell, Fill, Puzzle, Slot
from xword.solver.beliefs import BPResult, marginal_pattern

if TYPE_CHECKING:  # pragma: no cover - typing only
    # Only ever a duck-typed parameter here, so the lexicon machinery is not
    # dragged into solves that never consult a word list.
    from xword.lexicon.store import Lexicon


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Floor applied before taking a log. A candidate BP has driven to zero should
#: be merely terrible, not an exception, because it may still be the only word
#: that fits its crossings.
_EPS = 1e-12

#: log(_EPS): the score of a word we cannot price at all.
_UNSCORED = math.log(_EPS)

#: Score contributed by a cell BP has no opinion about -- uniform over A-Z.
_UNIFORM_LOG = math.log(1.0 / len(ALPHABET))

#: Confidence at which a marginal is treated as a fixed letter when asking the
#: lexicon for matches. Matches the default of ``beliefs.marginal_pattern``.
_MARGINAL_THRESHOLD = 0.9

#: Node cap per limited-discrepancy pass. The pass is a fallback, not the main
#: search, so it gets a bounded slice of the time budget rather than all of it.
_LDS_NODE_BUDGET = 20_000

#: Used when a square has to be guessed and BP has no marginal for it at all.
#: E is the most common letter in crossword fill, so it is the least bad
#: uninformed guess.
_DEFAULT_LETTER = "E"


# --------------------------------------------------------------------------- #
# Configuration and results
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SearchConfig:
    """Knobs for the discrete stage.

    Defaults are sized for a 15x15 daily puzzle: wide enough that the beam
    recovers from a wrong top-1, narrow enough to finish well inside
    ``max_seconds``.
    """

    beam_width: int = 24
    max_candidates_per_slot: int = 40
    lexicon_topk: int = 40
    discrepancy_limit: int = 3
    max_seconds: float = 30.0
    repair_iterations: int = 400
    restarts: int = 3
    seed: int = 0


@dataclass(slots=True)
class Assignment:
    """A choice of answers, plus the grid it implies.

    Invariant: ``conflicts`` lists exactly the entries that did *not* receive a
    word, so ``all_words`` is ``not conflicts`` and ``coverage`` can be read off
    the two collections without carrying the puzzle around.

    ``complete`` is about *squares*, ``all_words`` is about *entries*: a grid
    finished by :func:`complete_from_marginals` is complete but not all-words,
    and that distinction is what the harness reports on.
    """

    fill: Fill
    slot_answers: dict[str, str]
    slot_scores: dict[str, float]
    score: float
    complete: bool
    all_words: bool
    conflicts: tuple[str, ...]

    def coverage(self) -> float:
        """Fraction of entries that were given a real word."""
        total = len(self.slot_answers) + len(self.conflicts)
        if total == 0:
            return 1.0
        return len(self.slot_answers) / total


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #


def _log(value: float) -> float:
    """log with a floor. NaN lands on the floor too, via the failed compare."""
    return math.log(value) if value > _EPS else _UNSCORED


def _slot_margin(bp: BPResult, slot_id: str) -> float:
    """BP's own confidence gap for an entry, 0 when it has no opinion."""
    try:
        value = float(bp.slot_margin(slot_id))
    except (KeyError, IndexError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _top_posterior(bp: BPResult, slot_id: str) -> float:
    posterior = bp.slot_posteriors.get(slot_id)
    if posterior is None or len(posterior) == 0:
        return 0.0
    value = float(np.max(posterior))
    return value if math.isfinite(value) else 0.0


def _letter_logprob(bp: BPResult, slot: Slot, word: str) -> float:
    """How much BP's per-cell marginals like ``word``."""
    total = 0.0
    for cell, char in zip(slot.cells, word, strict=False):
        marginal = bp.cell_marginals.get(cell)
        position = LETTER_INDEX.get(char)
        if marginal is None or position is None or position >= len(marginal):
            total += _UNIFORM_LOG
        else:
            total += _log(float(marginal[position]))
    return total


# --------------------------------------------------------------------------- #
# Pools
# --------------------------------------------------------------------------- #


def build_pools(
    puzzle: Puzzle,
    index: GridIndex,
    beliefs: SlotBeliefs,
    bp: BPResult,
    *,
    lexicon: Lexicon | None = None,
    config: SearchConfig | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """Per-entry shortlists of the words the search is allowed to write.

    Two sources are merged. Fused candidates are re-priced by the BP slot
    posterior rather than by their own prior, so a candidate the crossings have
    already argued against sinks. Lexicon words are scored
    ``log(lexicon_score) + sum_i log(marginal[cell_i][letter_i])``: a
    dictionary word is only attractive when BP's letters already like it, which
    is what stops the lexicon flooding every entry with plausible-looking fill
    that contradicts the grid.

    ``index`` is unused here. It is accepted so that every entry point in this
    module takes the same four positional arguments.
    """
    cfg = config or SearchConfig()
    pools: dict[str, list[tuple[str, float]]] = {}

    for slot in puzzle.slots:
        slot_id = slot.id
        scored: dict[str, float] = {}

        candidates = beliefs.candidates.get(slot_id) or []
        posterior = bp.slot_posteriors.get(slot_id)
        prior = beliefs.prior(slot_id)
        for position, candidate in enumerate(candidates):
            word = candidate.answer.upper()
            if len(word) != slot.length:
                continue
            if posterior is not None and position < len(posterior):
                score = _log(float(posterior[position]))
            elif position < prior.size:
                # No BP opinion for this entry: the fused prior is the next best
                # thing, and is on the same log-probability scale.
                score = _log(float(prior[position]))
            else:
                score = _UNSCORED
            if score > scored.get(word, -math.inf):
                scored[word] = score

        if lexicon is not None and cfg.lexicon_topk > 0:
            pattern = marginal_pattern(bp, slot, _MARGINAL_THRESHOLD)
            for word, weight in lexicon.match(pattern, cfg.lexicon_topk):
                word = word.upper()
                if len(word) != slot.length:
                    continue
                score = _log(float(weight)) + _letter_logprob(bp, slot, word)
                if score > scored.get(word, -math.inf):
                    scored[word] = score

        ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
        pools[slot_id] = ranked[: max(0, cfg.max_candidates_per_slot)]

    return pools


@dataclass(slots=True)
class _Pool:
    """A pool compiled for fast pattern queries.

    ``masks[position][letter]`` is a bitset of the words carrying that letter in
    that position. Answering "does any word still fit?" is then a handful of
    integer ANDs instead of a scan, which is what keeps forward checking cheap
    enough to run on every crossing of every beam expansion.
    """

    words: tuple[str, ...]
    scores: tuple[float, ...]
    masks: tuple[dict[str, int], ...]
    full: int
    by_word: dict[str, int]

    @classmethod
    def build(cls, entries: Sequence[tuple[str, float]]) -> _Pool:
        words = tuple(word for word, _ in entries)
        scores = tuple(float(score) for _, score in entries)
        length = len(words[0]) if words else 0
        masks: tuple[dict[str, int], ...] = tuple({} for _ in range(length))
        for position, word in enumerate(words):
            bit = 1 << position
            for offset, char in enumerate(word):
                masks[offset][char] = masks[offset].get(char, 0) | bit
        return cls(
            words=words,
            scores=scores,
            masks=masks,
            full=(1 << len(words)) - 1,
            by_word={word: i for i, word in enumerate(words)},
        )

    def score_of(self, word: str) -> float | None:
        position = self.by_word.get(word)
        return None if position is None else self.scores[position]


def _pool_objects(pools: Mapping[str, Sequence[tuple[str, float]]]) -> dict[str, _Pool]:
    return {slot_id: _Pool.build(entries) for slot_id, entries in pools.items()}


def _match_mask(pool: _Pool, pattern: str) -> int:
    """Bitset of the pool words consistent with ``pattern``."""
    accumulator = pool.full
    if not accumulator:
        return 0
    for position, char in enumerate(pattern):
        if char == WILDCARD:
            continue
        accumulator &= pool.masks[position].get(char, 0)
        if not accumulator:
            return 0
    return accumulator


def _iter_indices(mask: int, limit: int) -> list[int]:
    """Set bits, lowest first. Pools are score-sorted, so this is best-first."""
    out: list[int] = []
    while mask and len(out) < limit:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


# --------------------------------------------------------------------------- #
# Search state
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _State:
    """A partial assignment.

    The dictionaries are copy-on-write: a successor copies before it writes, so
    a parent can hand its maps straight to any child that changed nothing in
    them.
    """

    letters: dict[Cell, str]
    answers: dict[str, str]
    scores: dict[str, float]
    score: float


def _state_key(state: _State) -> tuple[int, float]:
    """Ranking key: filling an entry always beats leaving a hole, and among
    equally full states the higher total log-probability wins.

    Ordering on the two things separately rather than folding a penalty into the
    score keeps the comparison honest at any grid size -- a fixed hole penalty
    that works on a 5x5 gets swamped by the sum over 78 entries on a 15x15.
    """
    return (-len(state.answers), -state.score)


def _order_slots(puzzle: Puzzle, bp: BPResult, pools: Mapping[str, _Pool]) -> list[Slot]:
    """Entries in decreasing BP confidence, skipping those with nothing to try.

    Filling the entries the network is sure about first is what makes the
    crossings informative for everything that follows; deciding a coin-flip
    entry early just propagates a coin flip.
    """
    usable = [slot for slot in puzzle.slots if (pool := pools.get(slot.id)) and pool.words]
    return sorted(
        usable,
        key=lambda slot: (-(_slot_margin(bp, slot.id) * _top_posterior(bp, slot.id)), slot.id),
    )


def _forward_check(
    index: GridIndex,
    pools: Mapping[str, _Pool],
    slot_id: str,
    letters: Mapping[Cell, str],
    pending: set[str],
) -> bool:
    """True if every still-unassigned crossing entry keeps at least one option."""
    for neighbour_id in index.neighbours.get(slot_id, ()):
        if neighbour_id not in pending:
            continue
        pool = pools.get(neighbour_id)
        if pool is None or not pool.words:
            continue
        neighbour = index.slot_by_id[neighbour_id]
        if not _match_mask(pool, pattern_from_letters(neighbour, letters)):
            return False
    return True


def _place(state: _State, slot: Slot, word: str, score: float) -> _State:
    letters = dict(state.letters)
    for cell, char in zip(slot.cells, word, strict=False):
        letters[cell] = char
    answers = dict(state.answers)
    answers[slot.id] = word
    scores = dict(state.scores)
    scores[slot.id] = score
    return _State(letters, answers, scores, state.score + score)


# --------------------------------------------------------------------------- #
# Beam search
# --------------------------------------------------------------------------- #


def _beam_search(
    order: Sequence[Slot],
    pools: Mapping[str, _Pool],
    index: GridIndex,
    start: _State,
    cfg: SearchConfig,
    deadline: float,
) -> _State:
    """Beam over ``order``, forward-checked, returning the best state reached.

    When no word survives for an entry the state carries on without it rather
    than dying: an unfillable entry is a hole for repair to deal with, and
    killing the beam over one bad pool would throw away every other entry too.
    """
    if not order:
        return start

    width = max(1, cfg.beam_width)
    branch = max(1, cfg.beam_width)
    pending = {slot.id for slot in order}
    beam = [start]

    for slot in order:
        pending.discard(slot.id)
        if time.monotonic() >= deadline:
            break
        pool = pools[slot.id]
        successors: list[_State] = []
        for state in beam:
            mask = _match_mask(pool, pattern_from_letters(slot, state.letters))
            placed = 0
            for position in _iter_indices(mask, branch):
                child = _place(state, slot, pool.words[position], pool.scores[position])
                if not _forward_check(index, pools, slot.id, child.letters, pending):
                    continue
                successors.append(child)
                placed += 1
            if placed == 0:
                successors.append(state)
        if not successors:
            break
        # A stable sort over a deterministically generated list keeps ties
        # resolved the same way on every run.
        successors.sort(key=_state_key)
        beam = successors[:width]

    return beam[0]


# --------------------------------------------------------------------------- #
# Limited-discrepancy fallback
# --------------------------------------------------------------------------- #


def _lds_search(
    order: Sequence[Slot],
    pools: Mapping[str, _Pool],
    index: GridIndex,
    start: _State,
    cfg: SearchConfig,
    deadline: float,
) -> _State | None:
    """Depth-first retry that follows BP's greedy ranking and is allowed to
    deviate from it a bounded number of times.

    The beam is breadth-limited, so a solution needing one unpopular word early
    can be squeezed out of it entirely. Limited discrepancy attacks the same
    space from the other side: commit to the greedy choice everywhere except a
    few places, which is exactly the shape of "BP had one entry wrong". Returns
    a fully assigned state, or ``None`` if it never found one.
    """
    if not order:
        return start

    branch = max(2, cfg.beam_width)
    slot_ids = [slot.id for slot in order]
    # pending_at[d] is everything still unassigned once depth d has been decided.
    pending_at = [set(slot_ids[depth:]) for depth in range(1, len(slot_ids) + 1)]
    budget = _LDS_NODE_BUDGET

    def descend(depth: int, state: _State, allowance: int) -> _State | None:
        nonlocal budget
        if budget <= 0 or time.monotonic() >= deadline:
            return None
        budget -= 1
        if depth == len(order):
            return state
        slot = order[depth]
        pool = pools[slot.id]
        mask = _match_mask(pool, pattern_from_letters(slot, state.letters))
        for rank, position in enumerate(_iter_indices(mask, branch)):
            cost = 0 if rank == 0 else 1
            if cost > allowance:
                break
            child = _place(state, slot, pool.words[position], pool.scores[position])
            if not _forward_check(index, pools, slot.id, child.letters, pending_at[depth]):
                continue
            found = descend(depth + 1, child, allowance - cost)
            if found is not None:
                return found
        return None

    for limit in range(max(0, cfg.discrepancy_limit) + 1):
        budget = _LDS_NODE_BUDGET
        found = descend(0, start, limit)
        if found is not None:
            return found
        if time.monotonic() >= deadline:
            break
    return None


# --------------------------------------------------------------------------- #
# Assignments
# --------------------------------------------------------------------------- #


def _assignment_from_state(puzzle: Puzzle, state: _State) -> Assignment:
    answers = dict(state.answers)
    scores = {slot_id: state.scores[slot_id] for slot_id in answers}
    conflicts = tuple(slot.id for slot in puzzle.slots if slot.id not in answers)
    fill = Fill(dict(state.letters))
    return Assignment(
        fill=fill,
        slot_answers=answers,
        slot_scores=scores,
        score=float(sum(scores.values())),
        complete=fill.is_complete_for(puzzle),
        all_words=not conflicts,
        conflicts=conflicts,
    )


def _state_from_assignment(
    assignment: Assignment, index: GridIndex, pools: Mapping[str, _Pool]
) -> _State:
    """Rebuild a search state from an assignment.

    An answer contradicting a crossing already placed is dropped rather than
    silently overwriting it: repair exists to refill that hole, and a state
    whose letters disagree with its own words would make every score comparison
    meaningless.
    """
    letters: dict[Cell, str] = {}
    answers: dict[str, str] = {}
    scores: dict[str, float] = {}
    for slot_id, word in assignment.slot_answers.items():
        slot = index.slot_by_id.get(slot_id)
        if slot is None or len(word) != slot.length:
            continue
        if any(letters.get(cell, char) != char for cell, char in zip(slot.cells, word, strict=False)):
            continue
        for cell, char in zip(slot.cells, word, strict=False):
            letters[cell] = char
        answers[slot_id] = word
        score = assignment.slot_scores.get(slot_id)
        if score is None:
            pool = pools.get(slot_id)
            score = pool.score_of(word) if pool is not None else None
        scores[slot_id] = float(score) if score is not None else _UNSCORED
    return _State(letters, answers, scores, float(sum(scores.values())))


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def solve_assignment(
    puzzle: Puzzle,
    index: GridIndex,
    beliefs: SlotBeliefs,
    bp: BPResult,
    *,
    lexicon: Lexicon | None = None,
    config: SearchConfig | None = None,
) -> Assignment:
    """Choose one answer per entry so that every crossing agrees.

    Beam first, limited discrepancy only if the beam left a hole. Both stop at
    ``max_seconds`` and return the best state reached so far -- a partial grid
    delivered on time is worth more than a perfect one delivered late.
    """
    cfg = config or SearchConfig()
    deadline = time.monotonic() + max(0.0, cfg.max_seconds)
    pools = _pool_objects(build_pools(puzzle, index, beliefs, bp, lexicon=lexicon, config=cfg))
    order = _order_slots(puzzle, bp, pools)

    start = _State({}, {}, {}, 0.0)
    best = _beam_search(order, pools, index, start, cfg, deadline)

    if (
        len(best.answers) < len(order)
        and cfg.discrepancy_limit > 0
        and time.monotonic() < deadline
    ):
        alternative = _lds_search(order, pools, index, start, cfg, deadline)
        if alternative is not None and _state_key(alternative) < _state_key(best):
            best = alternative

    return _assignment_from_state(puzzle, best)


def repair(
    puzzle: Puzzle,
    index: GridIndex,
    beliefs: SlotBeliefs,
    bp: BPResult,
    assignment: Assignment,
    *,
    lexicon: Lexicon | None = None,
    config: SearchConfig | None = None,
) -> Assignment:
    """Local search around the weakest entries.

    Repeatedly: take the worst entry -- one that got no word at all, else the
    lowest-scoring one -- tear it out together with everything it crosses, and
    re-fill that small neighbourhood from the pools, keeping the result only if
    the total improves. A single bad word poisons up to a dozen crossings, so
    re-deciding the whole neighbourhood at once is the only move that can undo
    it; changing one entry in isolation never can.

    Entries that keep failing go tabu so the loop moves on instead of grinding
    at the same hopeless corner, and ``restarts`` perturbed restarts give the
    hill climb somewhere else to climb from. The returned fill is rebuilt from
    the chosen words alone -- call :func:`complete_from_marginals` afterwards to
    settle whatever is left.
    """
    cfg = config or SearchConfig()
    deadline = time.monotonic() + max(0.0, cfg.max_seconds)
    pools = _pool_objects(build_pools(puzzle, index, beliefs, bp, lexicon=lexicon, config=cfg))
    order = _order_slots(puzzle, bp, pools)

    best = _state_from_assignment(assignment, index, pools)
    if not order:
        return _assignment_from_state(puzzle, best)

    rng = random.Random(cfg.seed)
    restarts = max(1, cfg.restarts)
    per_restart = max(1, cfg.repair_iterations // restarts)

    for attempt in range(restarts):
        if time.monotonic() >= deadline:
            break
        state = best if attempt == 0 else _perturb(best, order, pools, index, cfg, rng, deadline)
        state = _hill_climb(state, order, pools, index, cfg, rng, deadline, per_restart)
        if _state_key(state) < _state_key(best):
            best = state

    return _assignment_from_state(puzzle, best)


def complete_from_marginals(
    puzzle: Puzzle, index: GridIndex, bp: BPResult, assignment: Assignment
) -> Assignment:
    """Fill every remaining square with the argmax of its BP marginal.

    A crossword is scored per square, so a blank is strictly worse than a guess:
    the guess can only gain. ``all_words`` and ``conflicts`` are left alone --
    letters put here are not words, and pretending otherwise would hide a real
    failure from the harness.

    ``index`` is unused; it is accepted for symmetry with the other entry
    points.
    """
    letters = dict(assignment.fill.letters)
    for cell in puzzle.open_cells:
        if cell in letters:
            continue
        marginal = bp.cell_marginals.get(cell)
        if marginal is None or len(marginal) == 0:
            letters[cell] = _DEFAULT_LETTER
        else:
            letters[cell] = ALPHABET[int(np.argmax(np.asarray(marginal)))]

    fill = Fill(letters)
    return Assignment(
        fill=fill,
        slot_answers=dict(assignment.slot_answers),
        slot_scores=dict(assignment.slot_scores),
        score=assignment.score,
        complete=fill.is_complete_for(puzzle),
        all_words=assignment.all_words,
        conflicts=assignment.conflicts,
    )


# --------------------------------------------------------------------------- #
# Repair internals
# --------------------------------------------------------------------------- #


def _neighbourhood(target: str, index: GridIndex, searchable: set[str]) -> set[str]:
    group = {target}
    group.update(n for n in index.neighbours.get(target, ()) if n in searchable)
    return group


def _tear_out(state: _State, group: set[str], index: GridIndex) -> _State:
    """The state with ``group`` removed, keeping every other entry's letters."""
    answers = {sid: word for sid, word in state.answers.items() if sid not in group}
    scores = {sid: state.scores[sid] for sid in answers}
    letters: dict[Cell, str] = {}
    for slot_id, word in answers.items():
        for cell, char in zip(index.slot_by_id[slot_id].cells, word, strict=False):
            letters[cell] = char
    return _State(letters, answers, scores, float(sum(scores.values())))


def _refill(
    state: _State,
    target: str,
    order: Sequence[Slot],
    pools: Mapping[str, _Pool],
    index: GridIndex,
    cfg: SearchConfig,
    deadline: float,
    searchable: set[str],
) -> _State:
    group = _neighbourhood(target, index, searchable)
    base = _tear_out(state, group, index)
    sub_order = [slot for slot in order if slot.id in group]
    return _beam_search(sub_order, pools, index, base, cfg, deadline)


def _pick_target(
    state: _State, slot_ids: Sequence[str], tabu: set[str], rng: random.Random
) -> str | None:
    """The entry most worth re-deciding: a hole first, otherwise the worst word."""
    holes = [
        slot_id
        for slot_id in slot_ids
        if slot_id not in state.answers and slot_id not in tabu
    ]
    if holes:
        return rng.choice(holes)
    live = [slot_id for slot_id in slot_ids if slot_id not in tabu]
    if not live:
        return None
    worst = min(state.scores.get(slot_id, _UNSCORED) for slot_id in live)
    ties = [slot_id for slot_id in live if state.scores.get(slot_id, _UNSCORED) <= worst + 1e-12]
    return rng.choice(ties)


def _hill_climb(
    state: _State,
    order: Sequence[Slot],
    pools: Mapping[str, _Pool],
    index: GridIndex,
    cfg: SearchConfig,
    rng: random.Random,
    deadline: float,
    iterations: int,
) -> _State:
    slot_ids = [slot.id for slot in order]
    searchable = set(slot_ids)
    tabu: set[str] = set()
    for _ in range(iterations):
        if time.monotonic() >= deadline:
            break
        target = _pick_target(state, slot_ids, tabu, rng)
        if target is None:
            break
        candidate = _refill(state, target, order, pools, index, cfg, deadline, searchable)
        if _state_key(candidate) < _state_key(state):
            state = candidate
            tabu.clear()
        else:
            tabu.add(target)
    return state


def _perturb(
    state: _State,
    order: Sequence[Slot],
    pools: Mapping[str, _Pool],
    index: GridIndex,
    cfg: SearchConfig,
    rng: random.Random,
    deadline: float,
) -> _State:
    """Kick the state into a different basin: force one entry to a word it did
    not choose, then re-fill everything that crosses it.

    Re-filling without forcing anything would just reproduce the same local
    optimum, which would make ``restarts`` free and useless.
    """
    slot_ids = [slot.id for slot in order]
    if not slot_ids:
        return state
    searchable = set(slot_ids)
    target = rng.choice(slot_ids)
    slot = index.slot_by_id[target]
    pool = pools[target]

    group = _neighbourhood(target, index, searchable)
    base = _tear_out(state, group, index)
    mask = _match_mask(pool, pattern_from_letters(slot, base.letters))
    current = state.answers.get(target)
    choices = [i for i in _iter_indices(mask, max(2, cfg.beam_width)) if pool.words[i] != current]
    if not choices:
        return state

    picked = rng.choice(choices)
    forced = _place(base, slot, pool.words[picked], pool.scores[picked])
    sub_order = [entry for entry in order if entry.id in group and entry.id != target]
    return _beam_search(sub_order, pools, index, forced, cfg, deadline)


__all__ = [
    "Assignment",
    "SearchConfig",
    "build_pools",
    "complete_from_marginals",
    "repair",
    "solve_assignment",
]

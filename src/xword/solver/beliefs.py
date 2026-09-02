"""Loopy belief propagation over the crossword factor graph.

The grid is read as a factor graph with one variable per open cell (26 letter
values) and one factor per entry. An entry's factor is its candidate
distribution from :class:`~xword.core.beliefs.SlotBeliefs`, extended with an
explicit "none of my candidates" branch that falls back to a background letter
prior. Because across and down entries interlock, the graph is full of short
loops and exact inference is intractable; loopy BP is the standard
approximation, and is the inference step the Berkeley Crossword Solver used to
beat the human field at the 2021 American Crossword Puzzle Tournament -- see
Wallace, Tomlinson, Grebe, Kim, Michaud, Klein and Krishnamurthy, "Automated
Crossword Solving", ACL 2022.

Why the null branch is not optional
-----------------------------------
A factor whose support is exactly its candidate list assigns zero probability
to every other string, so one missing answer makes the whole neighbourhood of
that entry inconsistent and BP spends its iterations spreading that
contradiction. Reserving ``null_mass`` for "the answer is not in my list" turns
a hard contradiction into soft evidence: the entry stops arguing and lets its
crossings decide the letters.

Numerics
--------
Everything that multiplies many small numbers is done in log space, and the
leave-one-out message uses *subtraction* of the position's own log message
rather than division by the message itself. Division is what makes this
algorithm produce NaNs: a message component legitimately reaches 1e-300 on a
long entry, and dividing by it turns a finite weight into ``inf`` and then, one
subtraction later, into ``nan``. Log messages are clamped at ``log(1e-12)`` so
every term stays finite, and every exponential is taken relative to a
per-position maximum so nothing overflows.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from xword.core.beliefs import SlotBeliefs
from xword.core.grid import GridIndex
from xword.core.types import (
    ALPHABET,
    LETTER_INDEX,
    WILDCARD,
    Cell,
    Puzzle,
    Slot,
)

#: Size of the alphabet a cell variable ranges over.
N_LETTERS = 26

#: Floor applied to any probability before a logarithm is taken. Large enough
#: that ``log`` stays comfortably finite (about -27.6), small enough that a
#: message this weak is already a hard veto in practice.
PROB_FLOOR = 1e-12

#: Floor on an entry's null mass. ``SlotBeliefs.set_slot`` already clamps to
#: ``MIN_NULL_MASS``, but beliefs assembled by hand may not, and a factor with
#: an empty candidate list *and* zero null mass would have no support at all.
NULL_FLOOR = 1e-12

#: Relative letter frequencies of English text, used as the background prior
#: ``bg``. This is what the null branch spends its mass on, so it wants to be a
#: real distribution rather than uniform: "not one of my candidates" should
#: still prefer E over Q. Values are percentages; they are renormalised on use.
DEFAULT_LETTER_PRIOR: np.ndarray = np.array(
    [
        8.17, 1.49, 2.78, 4.25, 12.70, 2.23, 2.02, 6.09, 6.97, 0.15,
        0.77, 4.03, 2.41, 6.75, 7.51, 1.93, 0.10, 5.99, 6.33, 9.06,
        2.76, 0.98, 2.36, 0.15, 1.97, 0.07,
    ],
    dtype=np.float64,
)


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class BPResult:
    """The fixed point (or the best approximation of it) that BP reached.

    Attributes
    ----------
    cell_marginals:
        ``Cell -> length-26 array`` summing to 1, over every open cell of the
        puzzle. Cells that belong to no entry get the background prior.
    slot_posteriors:
        ``slot_id -> array`` parallel to ``beliefs.candidates[slot_id]``.
    slot_null:
        ``slot_id -> float``. ``slot_posteriors[s].sum() + slot_null[s] == 1``.
    converged:
        Whether the message update fell below ``tol`` before the iteration cap.
    iterations:
        Sweeps actually performed.
    max_delta:
        The largest single message change on the final sweep.
    """

    cell_marginals: dict[Cell, np.ndarray]
    slot_posteriors: dict[str, np.ndarray]
    slot_null: dict[str, float]
    converged: bool
    iterations: int
    max_delta: float

    # -- per-cell views ----------------------------------------------------- #

    def top_letters(self) -> dict[Cell, tuple[str, float]]:
        """Most likely letter per cell with its marginal probability."""
        out: dict[Cell, tuple[str, float]] = {}
        for cell, marginal in self.cell_marginals.items():
            best = int(np.argmax(marginal))
            out[cell] = (ALPHABET[best], float(marginal[best]))
        return out

    def cell_entropy(self) -> dict[Cell, float]:
        """Marginal entropy per cell, in nats.

        This is the "which squares are shaky" signal: a cell BP is sure about
        sits near 0, a cell it has no opinion on sits near ``log 26 = 3.26``.
        Reported in nats rather than bits so it composes with the log-scores
        used everywhere else.
        """
        out: dict[Cell, float] = {}
        for cell, marginal in self.cell_marginals.items():
            positive = marginal[marginal > 0.0]
            out[cell] = float(-np.sum(positive * np.log(positive)))
        return out

    # -- per-entry views ---------------------------------------------------- #

    def slot_margin(self, slot_id: str) -> float:
        """Gap between the best and second-best candidate posterior.

        Deliberately ignores ``slot_null``: this answers "is one candidate
        clearly ahead of the others", and a caller that also needs "is any
        candidate plausible at all" should read ``slot_null[slot_id]``. An entry
        with a single candidate has no competitor, so its margin is 1.0; an
        entry with no candidates has nothing to compare and returns 0.0.
        """
        posterior = self.slot_posteriors.get(slot_id)
        if posterior is None or posterior.size == 0:
            return 0.0
        if posterior.size == 1:
            return 1.0
        top2 = np.partition(posterior, -2)[-2:]
        return float(abs(top2[1] - top2[0]))


# --------------------------------------------------------------------------- #
# Encoding helpers
# --------------------------------------------------------------------------- #


def letters_matrix(answers: Sequence[str]) -> np.ndarray:
    """Encode equal-length answers as a ``K x L`` matrix of letter indices.

    ``int8`` because the alphabet fits and the search stage keeps a lot of these
    around; callers that index with it should cast to an index dtype first.
    Raises ``ValueError`` on ragged input or on any character outside ``A-Z``,
    since a silently mis-encoded answer would corrupt every message the entry
    sends.
    """
    if len(answers) == 0:
        return np.zeros((0, 0), dtype=np.int8)
    length = len(answers[0])
    out = np.zeros((len(answers), length), dtype=np.int8)
    for row, answer in enumerate(answers):
        if len(answer) != length:
            raise ValueError(
                f"ragged answers: {answer!r} has length {len(answer)}, "
                f"expected {length}"
            )
        for col, char in enumerate(answer):
            index = LETTER_INDEX.get(char)
            if index is None:
                raise ValueError(f"answer {answer!r} contains non-letter {char!r}")
            out[row, col] = index
    return out


def marginal_pattern(result: BPResult, slot: Slot, threshold: float = 0.9) -> str:
    """The letters BP is at least ``threshold`` sure of, ``?`` elsewhere.

    This is the constraint string handed back to the candidate sources on the
    next round, so the default threshold is deliberately strict: a wrong letter
    here poisons the next generation, whereas a ``?`` merely wastes a little of
    the model's attention.
    """
    chars: list[str] = []
    for cell in slot.cells:
        marginal = result.cell_marginals.get(cell)
        if marginal is None or marginal.size != N_LETTERS:
            chars.append(WILDCARD)
            continue
        best = int(np.argmax(marginal))
        chars.append(ALPHABET[best] if marginal[best] >= threshold else WILDCARD)
    return "".join(chars)


# --------------------------------------------------------------------------- #
# Internal factor representation
# --------------------------------------------------------------------------- #


@dataclass(slots=True, frozen=True)
class _Factor:
    """One entry, pre-encoded for the message loop.

    Built once per :func:`run_bp` call so that the per-iteration cost is a
    handful of numpy calls per entry with no Python-level scan over candidates.
    """

    slot_id: str
    start: int  # first port index; an entry's ports are contiguous
    stop: int
    length: int
    letters: np.ndarray  # (K, L) index dtype
    flat_index: np.ndarray  # (K, L) = position * 26 + letter, for bincount
    log_prior: np.ndarray  # (K,), -inf for unusable candidates
    log_null: float


def _normalise_letter_prior(letter_prior: np.ndarray | None) -> np.ndarray:
    if letter_prior is None:
        prior = DEFAULT_LETTER_PRIOR.astype(np.float64, copy=True)
    else:
        prior = np.asarray(letter_prior, dtype=np.float64).reshape(-1)
        if prior.shape != (N_LETTERS,):
            raise ValueError(
                f"letter_prior must have {N_LETTERS} entries, got {prior.shape}"
            )
        if not np.all(np.isfinite(prior)) or np.any(prior < 0.0):
            raise ValueError("letter_prior must be finite and non-negative")
        if prior.sum() <= 0.0:
            raise ValueError("letter_prior sums to zero")
    prior = prior / prior.sum()
    # No letter may be impossible: a zero here makes log(bg) infinite and lets a
    # single confident entry veto a letter everywhere in the grid.
    prior = np.maximum(prior, PROB_FLOOR)
    return prior / prior.sum()


def _cell_order(puzzle: Puzzle, index: GridIndex) -> tuple[list[Cell], dict[Cell, int]]:
    """Every cell that needs a marginal, in row-major order.

    Starts from the puzzle's open cells so that a grid with entry-less squares
    still gets a complete answer, then appends anything the index or a slot
    references that the puzzle does not list as open (malformed, but not a
    reason to crash mid-solve).
    """
    cells: list[Cell] = list(puzzle.open_cells)
    seen = set(cells)
    for cell in sorted(c for c in index.cell_slots if c not in seen):
        seen.add(cell)
        cells.append(cell)
    for slot in index.slots:
        for cell in slot.cells:
            if cell not in seen:
                seen.add(cell)
                cells.append(cell)
    return cells, {cell: i for i, cell in enumerate(cells)}


def _encode_slot(answers: Sequence[str], length: int) -> tuple[np.ndarray, np.ndarray]:
    """``(letters, usable)`` for one entry, tolerating malformed candidates.

    A candidate of the wrong length or with a non-letter is kept in place -- the
    posterior array must stay parallel to ``beliefs.candidates`` -- but marked
    unusable, which becomes a ``-inf`` log prior and therefore exactly zero
    weight in every message.
    """
    count = len(answers)
    letters = np.zeros((count, length), dtype=np.intp)
    usable = np.ones(count, dtype=bool)
    for row, answer in enumerate(answers):
        if len(answer) != length:
            usable[row] = False
            continue
        for col, char in enumerate(answer):
            code = LETTER_INDEX.get(char)
            if code is None:
                usable[row] = False
                break
            letters[row, col] = code
    return letters, usable


def _build_factors(
    index: GridIndex, beliefs: SlotBeliefs, cell_pos: Mapping[Cell, int]
) -> tuple[list[_Factor], np.ndarray]:
    """Encode every entry and lay out the message ports.

    Ports are ``(entry, position)`` pairs numbered contiguously per entry, so a
    factor reads and writes a plain slice of the global message arrays instead
    of a fancy index.
    """
    factors: list[_Factor] = []
    port_cell: list[int] = []
    offset = 0

    for slot in index.slots:
        slot_id = slot.id
        length = slot.length
        answers = [c.answer for c in beliefs.candidates.get(slot_id, ())]
        prior = np.asarray(
            beliefs.priors.get(slot_id, np.zeros(0, dtype=np.float64)),
            dtype=np.float64,
        )
        if prior.shape != (len(answers),):
            raise ValueError(
                f"{slot_id}: {len(answers)} candidates but priors have shape "
                f"{prior.shape}; SlotBeliefs.set_slot keeps these parallel"
            )

        letters, usable = _encode_slot(answers, length)
        keep = usable & (prior > 0.0)
        with np.errstate(divide="ignore"):
            log_prior = np.where(keep, np.log(np.where(keep, prior, 1.0)), -np.inf)

        positions = np.arange(length, dtype=np.intp) * N_LETTERS
        flat_index = letters + positions[None, :]

        null = float(beliefs.null_mass.get(slot_id, 1.0))
        if not math.isfinite(null):
            null = 1.0
        null = min(max(null, NULL_FLOOR), 1.0)

        factors.append(
            _Factor(
                slot_id=slot_id,
                start=offset,
                stop=offset + length,
                length=length,
                letters=letters,
                flat_index=flat_index,
                log_prior=log_prior,
                log_null=math.log(null),
            )
        )
        offset += length
        port_cell.extend(cell_pos[cell] for cell in slot.cells)

    return factors, np.asarray(port_cell, dtype=np.intp)


# --------------------------------------------------------------------------- #
# Message updates
# --------------------------------------------------------------------------- #


def _factor_message(
    factor: _Factor, incoming: np.ndarray, bg: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """One entry's outgoing messages, plus the pieces the posterior needs.

    Returns ``(out, logw, log_null_total)`` where ``out`` is ``L x 26`` with
    normalised rows, ``logw`` is the unnormalised log weight of each candidate
    under *all* the incoming messages, and ``log_null_total`` is the matching
    log weight of the "not in my list" branch.
    """
    length = factor.length
    with np.errstate(divide="ignore"):
        log_in = np.log(np.maximum(incoming, PROB_FLOOR))  # (L, 26)
    # The null branch puts bg on the target cell and asks every other cell how
    # much it likes bg overall.
    bg_dot = np.log(np.maximum(incoming @ bg, PROB_FLOOR))  # (L,)
    bg_dot_total = float(bg_dot.sum())
    log_null_pos = factor.log_null + (bg_dot_total - bg_dot)  # (L,)

    if factor.letters.shape[0]:
        log_at = log_in.ravel()[factor.flat_index]  # (K, L)
        logw = factor.log_prior + log_at.sum(axis=1)  # (K,)
        # Leave-one-out by subtraction, never by division: log_at is clamped, so
        # this is always a finite number minus a finite number.
        leave_out = logw[:, None] - log_at  # (K, L)
        shift = np.maximum(leave_out.max(axis=0), log_null_pos)
        weights = np.exp(leave_out - shift)  # exactly 0 for a dead candidate
        out = np.bincount(
            factor.flat_index.ravel(),
            weights=weights.ravel(),
            minlength=length * N_LETTERS,
        ).reshape(length, N_LETTERS)
    else:
        logw = np.zeros(0, dtype=np.float64)
        shift = log_null_pos
        out = np.zeros((length, N_LETTERS), dtype=np.float64)

    out += np.exp(log_null_pos - shift)[:, None] * bg
    # ``shift`` is finite because ``log_null_pos`` always is, so at least one
    # term above is exp(0) == 1 and no row can sum to zero.
    out /= out.sum(axis=1, keepdims=True)
    return out, logw, factor.log_null + bg_dot_total


def _cell_step(
    s2c: np.ndarray,
    port_cell: np.ndarray,
    n_cells: int,
    log_bg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``(log cell beliefs, cell -> entry messages)``.

    The message a cell sends to one entry is the product of the background prior
    and every *other* entry's opinion; forming the full product once and
    subtracting the one term back out is the same leave-one-out trick used
    inside the factor, for the same reason.
    """
    with np.errstate(divide="ignore"):
        log_s2c = np.log(np.maximum(s2c, PROB_FLOOR))
    belief = np.tile(log_bg, (n_cells, 1))
    np.add.at(belief, port_cell, log_s2c)

    message = belief[port_cell] - log_s2c
    message -= message.max(axis=1, keepdims=True)
    np.exp(message, out=message)
    message /= message.sum(axis=1, keepdims=True)
    return belief, message


def _softmax_rows(log_values: np.ndarray) -> np.ndarray:
    shifted = log_values - log_values.max(axis=1, keepdims=True)
    out = np.exp(shifted)
    return out / out.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run_bp(
    puzzle: Puzzle,
    index: GridIndex,
    beliefs: SlotBeliefs,
    *,
    iterations: int = 60,
    damping: float = 0.5,
    tol: float = 1e-4,
    letter_prior: np.ndarray | None = None,
) -> BPResult:
    """Run loopy BP and return cell marginals and per-entry posteriors.

    ``iterations=0`` is legal and is the ``no-bp`` ablation: the factors are
    evaluated once against uninformative incoming messages, which yields exactly
    the prior marginals (candidate priors plus the null branch's background) and
    no exchange of evidence between entries. That case reports
    ``converged=True`` with ``max_delta=0.0`` because no message was ever sent
    -- vacuously, not because a fixed point was found.

    ``damping`` is the weight kept on the previous message. Crossword factor
    graphs oscillate badly without it: two entries that disagree about a
    crossing will otherwise trade the same contradiction back and forth forever.

    Raises ``ValueError`` if a parameter is out of range or if ``beliefs.priors``
    is not parallel to ``beliefs.candidates``.
    """
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")
    if not 0.0 <= damping < 1.0:
        raise ValueError(f"damping must be in [0, 1), got {damping}")
    if tol < 0.0:
        raise ValueError(f"tol must be >= 0, got {tol}")

    bg = _normalise_letter_prior(letter_prior)
    log_bg = np.log(bg)

    cells, cell_pos = _cell_order(puzzle, index)
    n_cells = len(cells)
    factors, port_cell = _build_factors(index, beliefs, cell_pos)
    n_ports = int(port_cell.shape[0])

    if n_cells == 0:
        return BPResult({}, {}, {}, True, 0, 0.0)

    # Uniform rather than ``bg``: with uniform incoming messages the background
    # term is a constant that cancels, so the opening pass reproduces the
    # candidate priors exactly instead of tilting them by letter frequency.
    # That is what makes ``iterations=0`` mean "priors only".
    c2s_used = np.full((n_ports, N_LETTERS), 1.0 / N_LETTERS, dtype=np.float64)
    s2c = np.empty((n_ports, N_LETTERS), dtype=np.float64)
    for factor in factors:
        s2c[factor.start : factor.stop] = _factor_message(
            factor, c2s_used[factor.start : factor.stop], bg
        )[0]
    belief, c2s = _cell_step(s2c, port_cell, n_cells, log_bg)

    max_delta = 0.0
    performed = 0
    converged = iterations == 0
    scratch = np.empty_like(s2c)

    for step in range(iterations):
        c2s_used = c2s
        for factor in factors:
            scratch[factor.start : factor.stop] = _factor_message(
                factor, c2s_used[factor.start : factor.stop], bg
            )[0]
        if damping > 0.0:
            scratch *= 1.0 - damping
            scratch += damping * s2c
            scratch /= scratch.sum(axis=1, keepdims=True)
        delta = float(np.abs(scratch - s2c).max()) if n_ports else 0.0
        s2c, scratch = scratch, s2c

        belief, c2s = _cell_step(s2c, port_cell, n_cells, log_bg)
        if n_ports:
            delta = max(delta, float(np.abs(c2s - c2s_used).max()))

        performed = step + 1
        max_delta = delta
        if delta < tol:
            converged = True
            break

    marginals = _softmax_rows(belief)
    cell_marginals = {cell: marginals[i] for i, cell in enumerate(cells)}

    # The factor belief uses the same incoming messages that produced the
    # current outgoing ones, which is what keeps the iterations=0 posterior
    # equal to the prior.
    slot_posteriors: dict[str, np.ndarray] = {}
    slot_null: dict[str, float] = {}
    for factor in factors:
        _, logw, log_null_total = _factor_message(
            factor, c2s_used[factor.start : factor.stop], bg
        )
        best = float(logw.max()) if logw.size else -np.inf
        shift = max(best, log_null_total)
        weights = np.exp(logw - shift) if logw.size else np.zeros(0, dtype=np.float64)
        null_weight = math.exp(log_null_total - shift)
        total = float(weights.sum()) + null_weight
        slot_posteriors[factor.slot_id] = weights / total
        slot_null[factor.slot_id] = null_weight / total

    return BPResult(
        cell_marginals=cell_marginals,
        slot_posteriors=slot_posteriors,
        slot_null=slot_null,
        converged=converged,
        iterations=performed,
        max_delta=max_delta,
    )


__all__ = [
    "BPResult",
    "DEFAULT_LETTER_PRIOR",
    "NULL_FLOOR",
    "N_LETTERS",
    "PROB_FLOOR",
    "letters_matrix",
    "marginal_pattern",
    "run_bp",
]

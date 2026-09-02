"""How a crossword solution is scored, aggregated, and triaged.

This module is the backbone of the evaluation methodology, so it tries to be
precise about what each number means and honest about what it does not capture.

Why ``solved`` leads and cell accuracy does not
-----------------------------------------------
The headline metric is ``solved``: every open cell of the puzzle matches the
official answer. It is the number crossword solvers are conventionally compared
on, and the only one that corresponds to what a human means by "I solved it".

Cell accuracy read on its own is actively misleading. A standard 15x15 daily has
roughly 180 open cells, so 95% cell accuracy means about nine wrong letters.
Those nine letters sit in up to eighteen entries, and a puzzle with even one
wrong letter is not solved. A solver can therefore post 95% cell accuracy and a
0% solve rate. The two numbers answer different questions -- "how close is the
grid?" versus "did it actually work?" -- and both are reported here so that
neither can be quoted without the other.

The scored cell set
-------------------
Only open cells that the reference solution actually covers are scored. In a
well-formed puzzle that is every open cell; an open cell belonging to no entry
(which ``grid.validate_puzzle`` flags as a defect) has no gold letter and is
skipped rather than silently counted as wrong.

Confidence, coverage, and abstention
------------------------------------
``mean_confidence`` and ``brier`` are computed over *every* scored cell, with an
unfilled cell contributing confidence 0.0 and correctness 0. That makes the
Brier score a proper scoring rule over the whole grid, but it also means a
solver that writes nothing scores a perfect 0.0 Brier: abstention is free under
this decision. That is exactly why ``coverage`` and ``cell_accuracy`` are
reported next to it, and why the calibration helpers below take explicit
(confidence, correct) pairs so a report can also show the conditional-on-writing
view via :func:`calibration_pairs` with ``filled_only=True``.
"""

from __future__ import annotations

import collections.abc as _abc
import math
import re
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from xword.core.types import WILDCARD, Cell, Puzzle, SolveResult

# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #


def _ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator``, or 0.0 when the denominator is empty.

    Ratios feed averages and JSON reports, so nothing in a dataclass field is
    ever allowed to be NaN -- an undefined ratio is reported as 0.0 and the
    accompanying count (``cells_filled``, ``n``) is what tells a reader the
    number is vacuous. The one deliberate exception is
    :func:`selective_accuracy`, where a gap in the curve is the honest answer.
    """
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _clamp01(value: float) -> float:
    """Coerce a probability into ``[0, 1]``; non-finite input becomes 0.0."""
    x = float(value)
    if not math.isfinite(x):
        return 0.0
    return min(max(x, 0.0), 1.0)


def _percentile_ci(samples: np.ndarray, point: float) -> tuple[float, float]:
    """Two-sided 95% percentile interval, falling back to a degenerate one."""
    if samples.size == 0:
        return (point, point)
    return (
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    )


# --------------------------------------------------------------------------- #
# Per-puzzle score
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PuzzleScore:
    """Everything one puzzle contributes to the evaluation.

    Attributes
    ----------
    cells_total:
        Open cells covered by the reference solution -- the denominator for
        ``cell_accuracy`` and ``coverage``.
    cells_filled:
        Scored cells the solver wrote a letter into. Blanks and ``?`` are not
        letters.
    cells_correct:
        Scored cells whose written letter matches gold. An unfilled cell is
        never correct.
    solved:
        Exact match on every scored cell. False for a puzzle with no scored
        cells: an empty grid is not a solve.
    cell_accuracy:
        ``cells_correct / cells_total``. Unfilled counts as wrong.
    cell_precision:
        ``cells_correct / cells_filled`` -- how trustworthy the letters it *did*
        write are. A solver that fills two cells correctly and abstains
        everywhere else scores 1.0 here and near 0.0 on ``cell_accuracy``; that
        gap is the point of reporting both.
    word_accuracy:
        Entries whose full answer matches gold, over entries with a gold answer.
    coverage:
        ``cells_filled / cells_total``.
    mean_confidence:
        Mean stated confidence over all scored cells, unfilled counting as 0.0.
    brier:
        Mean squared error of that confidence against per-cell correctness. See
        the module docstring for why abstention is free under this definition.
    by_direction:
        Word accuracy keyed by ``"across"`` / ``"down"``. A direction with no
        gold entries is omitted rather than reported as 0.0.
    by_length:
        Word accuracy keyed by gold answer length.
    wrong_entries:
        ``(slot_id, predicted, gold)`` for every entry that is not exactly
        right, in grid order. ``predicted`` is the pattern the fill produced, so
        an incomplete entry shows as e.g. ``"?A??"`` rather than as ``None``.
    """

    puzzle_id: str
    cells_total: int
    cells_filled: int
    cells_correct: int
    words_total: int
    words_correct: int
    solved: bool
    cell_accuracy: float
    cell_precision: float
    word_accuracy: float
    coverage: float
    mean_confidence: float
    brier: float
    by_direction: Mapping[str, float] = field(default_factory=dict)
    by_length: Mapping[int, float] = field(default_factory=dict)
    wrong_entries: tuple[tuple[str, str, str], ...] = ()

    def as_dict(self) -> dict[str, object]:
        """JSON-safe view. ``by_length`` keys become strings."""
        return {
            "puzzle_id": self.puzzle_id,
            "cells_total": self.cells_total,
            "cells_filled": self.cells_filled,
            "cells_correct": self.cells_correct,
            "words_total": self.words_total,
            "words_correct": self.words_correct,
            "solved": self.solved,
            "cell_accuracy": self.cell_accuracy,
            "cell_precision": self.cell_precision,
            "word_accuracy": self.word_accuracy,
            "coverage": self.coverage,
            "mean_confidence": self.mean_confidence,
            "brier": self.brier,
            "by_direction": dict(self.by_direction),
            "by_length": {str(k): v for k, v in self.by_length.items()},
            "wrong_entries": [list(w) for w in self.wrong_entries],
        }


def _normalise_letters(
    puzzle: Puzzle, letters: Mapping[Cell, str]
) -> dict[Cell, str]:
    """Keep only single ``A-Z`` letters written into open cells.

    Solvers hand back partial fills in various shapes -- missing keys, ``?``
    placeholders, lower case from a hand-written test. Normalising here means
    the rest of the module can treat "present in this dict" as "the solver
    committed to a letter", which is what ``coverage`` and ``cell_precision``
    are counting.
    """
    open_cells = set(puzzle.open_cells)
    out: dict[Cell, str] = {}
    for cell, raw in letters.items():
        if cell not in open_cells or not isinstance(raw, str):
            continue
        ch = raw.strip().upper()
        if len(ch) == 1 and "A" <= ch <= "Z":
            out[cell] = ch
    return out


def score_fill(
    puzzle: Puzzle,
    letters: Mapping[Cell, str],
    confidence: Mapping[Cell, float] | None = None,
) -> PuzzleScore:
    """Score a raw cell assignment against the puzzle's reference solution.

    Raises ``ValueError`` (from :meth:`Puzzle.solution_letters`) if the puzzle
    carries no gold answers -- scoring an unlabelled puzzle is a caller bug, not
    a zero.
    """
    gold_cells = puzzle.solution_letters()
    gold_answers = puzzle.solution or {}
    written = _normalise_letters(puzzle, letters)
    conf = confidence or {}

    scored = [c for c in puzzle.open_cells if c in gold_cells]
    cells_total = len(scored)
    cells_filled = 0
    cells_correct = 0
    # Summed with fsum rather than ``+=`` so that a grid of identical
    # confidences reports that confidence back exactly instead of 0.9499...
    probabilities: list[float] = []
    brier_terms: list[float] = []

    for cell in scored:
        got = written.get(cell)
        if got is None:
            probability = 0.0
            correct = False
        else:
            cells_filled += 1
            probability = _clamp01(conf.get(cell, 0.0))
            correct = got == gold_cells[cell]
        if correct:
            cells_correct += 1
        probabilities.append(probability)
        brier_terms.append((probability - (1.0 if correct else 0.0)) ** 2)

    conf_sum = math.fsum(probabilities)
    brier_sum = math.fsum(brier_terms)

    words_total = 0
    words_correct = 0
    dir_total: dict[str, int] = {}
    dir_correct: dict[str, int] = {}
    len_total: dict[int, int] = {}
    len_correct: dict[int, int] = {}
    wrong: list[tuple[str, str, str]] = []

    for slot in puzzle.slots:
        answer = gold_answers.get(slot.id)
        if answer is None:
            continue
        predicted = "".join(written.get(c, WILDCARD) for c in slot.cells)
        ok = predicted == answer
        words_total += 1
        words_correct += int(ok)
        dir_total[slot.direction] = dir_total.get(slot.direction, 0) + 1
        dir_correct[slot.direction] = dir_correct.get(slot.direction, 0) + int(ok)
        size = len(answer)
        len_total[size] = len_total.get(size, 0) + 1
        len_correct[size] = len_correct.get(size, 0) + int(ok)
        if not ok:
            wrong.append((slot.id, predicted, answer))

    return PuzzleScore(
        puzzle_id=puzzle.id,
        cells_total=cells_total,
        cells_filled=cells_filled,
        cells_correct=cells_correct,
        words_total=words_total,
        words_correct=words_correct,
        solved=cells_total > 0 and cells_correct == cells_total,
        cell_accuracy=_ratio(cells_correct, cells_total),
        cell_precision=_ratio(cells_correct, cells_filled),
        word_accuracy=_ratio(words_correct, words_total),
        coverage=_ratio(cells_filled, cells_total),
        mean_confidence=_ratio(conf_sum, cells_total),
        brier=_ratio(brier_sum, cells_total),
        by_direction={
            d: _ratio(dir_correct[d], dir_total[d]) for d in sorted(dir_total)
        },
        by_length={n: _ratio(len_correct[n], len_total[n]) for n in sorted(len_total)},
        wrong_entries=tuple(wrong),
    )


def score_result(puzzle: Puzzle, result: SolveResult) -> PuzzleScore:
    """Score a :class:`SolveResult`, using its per-cell confidence.

    The puzzle id on the result must match the puzzle: a mismatch means the
    harness paired the wrong grid with the wrong run, which would otherwise show
    up as an inexplicably bad score rather than as an error.
    """
    if result.puzzle_id != puzzle.id:
        raise ValueError(
            f"result is for puzzle {result.puzzle_id!r}, not {puzzle.id!r}"
        )
    return score_fill(puzzle, result.fill.letters, result.cell_confidence)


def calibration_pairs(
    puzzle: Puzzle,
    letters: Mapping[Cell, str],
    confidence: Mapping[Cell, float] | None = None,
    *,
    filled_only: bool = True,
) -> tuple[list[float], list[bool]]:
    """Per-cell ``(confidences, correct)`` pairs ready for :func:`calibration`.

    ``filled_only=True`` (the default) answers "when it writes a letter, is its
    stated confidence honest?". ``filled_only=False`` adds the unfilled cells at
    confidence 0.0, matching how ``PuzzleScore.brier`` is computed.
    """
    gold = puzzle.solution_letters()
    written = _normalise_letters(puzzle, letters)
    conf = confidence or {}
    confidences: list[float] = []
    correct: list[bool] = []
    for cell in puzzle.open_cells:
        if cell not in gold:
            continue
        got = written.get(cell)
        if got is None:
            if filled_only:
                continue
            confidences.append(0.0)
            correct.append(False)
            continue
        confidences.append(_clamp01(conf.get(cell, 0.0)))
        correct.append(got == gold[cell])
    return confidences, correct


# --------------------------------------------------------------------------- #
# Suite aggregation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SuiteScore:
    """Aggregate over a set of puzzles.

    ``cell_accuracy`` and ``word_accuracy`` are micro-averages: correct cells
    over all cells in the suite, not the mean of per-puzzle accuracies. That
    weights a 15x15 more than a 5x5 mini, which is the right default when the
    question is "how much of this benchmark did it get right". ``solve_rate``
    and ``mean_coverage`` are per-puzzle means, since both are already
    per-puzzle quantities.

    Both confidence intervals are bootstrap percentile intervals resampled over
    **puzzles, not cells**. Cells inside one puzzle are strongly correlated --
    they share a solver run, a theme, a difficulty, and literal letters at every
    crossing -- so treating ~180 cells as 180 independent samples would report
    an interval several times too tight and turn noise into a finding. The
    puzzle is the unit of independence, so it is the unit of resampling.
    """

    n: int
    solve_rate: float
    cell_accuracy: float
    word_accuracy: float
    mean_coverage: float
    total_cost_usd: float = 0.0
    total_seconds: float = 0.0
    solve_rate_ci: tuple[float, float] = (0.0, 0.0)
    cell_accuracy_ci: tuple[float, float] = (0.0, 0.0)
    by_slice: Mapping[str, SuiteScore] = field(default_factory=dict)

    def with_slices(self, slices: Mapping[str, SuiteScore]) -> SuiteScore:
        """Copy carrying ``slices`` in ``by_slice``."""
        return replace(self, by_slice=dict(slices))

    def as_dict(self) -> dict[str, object]:
        """JSON-safe view, recursing into ``by_slice``."""
        return {
            "n": self.n,
            "solve_rate": self.solve_rate,
            "cell_accuracy": self.cell_accuracy,
            "word_accuracy": self.word_accuracy,
            "mean_coverage": self.mean_coverage,
            "total_cost_usd": self.total_cost_usd,
            "total_seconds": self.total_seconds,
            "solve_rate_ci": list(self.solve_rate_ci),
            "cell_accuracy_ci": list(self.cell_accuracy_ci),
            "by_slice": {k: v.as_dict() for k, v in self.by_slice.items()},
        }


_Totals = Mapping[str, float] | Sequence[float] | None


def _sum_totals(
    values: _Totals, scores: Sequence[PuzzleScore], what: str
) -> float:
    """Sum a per-puzzle quantity given either as a mapping or as a parallel list.

    A mapping is looked up once per score, so a suite that runs the same puzzle
    twice counts that puzzle's mapped cost twice; pass a parallel sequence when
    runs, not puzzles, are the unit.
    """
    if values is None:
        return 0.0
    if isinstance(values, _abc.Mapping):
        return float(sum(float(values.get(s.puzzle_id, 0.0)) for s in scores))
    seq = list(values)
    if len(seq) != len(scores):
        raise ValueError(
            f"{what}: got {len(seq)} values for {len(scores)} puzzles"
        )
    return float(sum(float(v) for v in seq))


#: Cap on the size of one bootstrap index block. Resampling is done in chunks so
#: that a large suite does not allocate a ``bootstrap x n`` index matrix all at
#: once; the chunk size is a deterministic function of ``n`` and ``bootstrap``,
#: so results stay reproducible.
_BOOTSTRAP_CHUNK_ELEMENTS = 2_000_000


def aggregate(
    scores: Sequence[PuzzleScore],
    *,
    costs: _Totals = None,
    seconds: _Totals = None,
    bootstrap: int = 2000,
    seed: int = 0,
) -> SuiteScore:
    """Aggregate per-puzzle scores into a suite-level result.

    ``bootstrap`` replicates are drawn from a ``numpy.random.default_rng(seed)``,
    so the confidence intervals are reproducible for a fixed
    ``(seed, bootstrap, n)``. Pass ``bootstrap=0`` to skip resampling, in which
    case each interval collapses to its point estimate.

    ``costs`` and ``seconds`` are optional per-puzzle totals, either mapping
    ``puzzle_id -> value`` or a sequence parallel to ``scores``.
    """
    scores = list(scores)
    total_cost = _sum_totals(costs, scores, "costs")
    total_seconds = _sum_totals(seconds, scores, "seconds")
    n = len(scores)
    if n == 0:
        return SuiteScore(
            n=0,
            solve_rate=0.0,
            cell_accuracy=0.0,
            word_accuracy=0.0,
            mean_coverage=0.0,
            total_cost_usd=total_cost,
            total_seconds=total_seconds,
            solve_rate_ci=(0.0, 0.0),
            cell_accuracy_ci=(0.0, 0.0),
            by_slice={},
        )

    solved = np.array([1.0 if s.solved else 0.0 for s in scores], dtype=np.float64)
    cell_ok = np.array([s.cells_correct for s in scores], dtype=np.float64)
    cell_all = np.array([s.cells_total for s in scores], dtype=np.float64)
    word_ok = np.array([s.words_correct for s in scores], dtype=np.float64)
    word_all = np.array([s.words_total for s in scores], dtype=np.float64)
    coverage = np.array([s.coverage for s in scores], dtype=np.float64)

    solve_rate = float(solved.mean())
    cell_accuracy = _ratio(cell_ok.sum(), cell_all.sum())
    word_accuracy = _ratio(word_ok.sum(), word_all.sum())
    mean_coverage = float(coverage.mean())

    solve_ci = (solve_rate, solve_rate)
    cell_ci = (cell_accuracy, cell_accuracy)
    if bootstrap and bootstrap > 0:
        rng = np.random.default_rng(seed)
        per_chunk = max(1, _BOOTSTRAP_CHUNK_ELEMENTS // n)
        solve_parts: list[np.ndarray] = []
        cell_parts: list[np.ndarray] = []
        drawn = 0
        while drawn < bootstrap:
            k = min(per_chunk, bootstrap - drawn)
            idx = rng.integers(0, n, size=(k, n))
            solve_parts.append(solved[idx].mean(axis=1))
            num = cell_ok[idx].sum(axis=1)
            den = cell_all[idx].sum(axis=1)
            cell_parts.append(
                np.divide(num, den, out=np.zeros(k, dtype=np.float64), where=den > 0)
            )
            drawn += k
        solve_ci = _percentile_ci(np.concatenate(solve_parts), solve_rate)
        cell_ci = _percentile_ci(np.concatenate(cell_parts), cell_accuracy)

    return SuiteScore(
        n=n,
        solve_rate=solve_rate,
        cell_accuracy=cell_accuracy,
        word_accuracy=word_accuracy,
        mean_coverage=mean_coverage,
        total_cost_usd=total_cost,
        total_seconds=total_seconds,
        solve_rate_ci=solve_ci,
        cell_accuracy_ci=cell_ci,
        by_slice={},
    )


def slice_scores(
    scores: Sequence[PuzzleScore],
    keys: Mapping[str, str],
    *,
    bootstrap: int = 2000,
    seed: int = 0,
    costs: _Totals = None,
    seconds: _Totals = None,
) -> dict[str, SuiteScore]:
    """Aggregate separately per slice label, e.g. day of week.

    ``keys`` maps ``puzzle_id -> label``. A score whose id is not in ``keys``
    belongs to no slice and is dropped: the caller decides what the slicing is,
    and inventing an "other" bucket would quietly pad every table.

    Each slice gets its own bootstrap seed, derived as ``seed + crc32(label)``,
    so a slice's interval does not depend on how many other slices exist or on
    dict ordering, and is stable across runs and platforms.

    ``costs`` and ``seconds``, when given as mappings, are carried into the
    per-slice totals; sequence form is accepted and split by slice membership.
    """
    scores = list(scores)
    cost_seq = _as_per_score(costs, scores, "costs")
    second_seq = _as_per_score(seconds, scores, "seconds")

    buckets: dict[str, list[PuzzleScore]] = {}
    bucket_costs: dict[str, list[float]] = {}
    bucket_seconds: dict[str, list[float]] = {}
    for i, score in enumerate(scores):
        label = keys.get(score.puzzle_id)
        if label is None:
            continue
        buckets.setdefault(label, []).append(score)
        bucket_costs.setdefault(label, []).append(cost_seq[i])
        bucket_seconds.setdefault(label, []).append(second_seq[i])

    out: dict[str, SuiteScore] = {}
    for label in sorted(buckets):
        offset = zlib.crc32(label.encode("utf-8")) % 10_000
        out[label] = aggregate(
            buckets[label],
            costs=bucket_costs[label],
            seconds=bucket_seconds[label],
            bootstrap=bootstrap,
            seed=seed + offset,
        )
    return out


def _as_per_score(
    values: _Totals, scores: Sequence[PuzzleScore], what: str
) -> list[float]:
    """Expand a mapping-or-sequence total into one value per score."""
    if values is None:
        return [0.0] * len(scores)
    if isinstance(values, _abc.Mapping):
        return [float(values.get(s.puzzle_id, 0.0)) for s in scores]
    seq = [float(v) for v in values]
    if len(seq) != len(scores):
        raise ValueError(f"{what}: got {len(seq)} values for {len(scores)} puzzles")
    return seq


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Binned reliability of a confidence estimate.

    Attributes
    ----------
    ece:
        Expected calibration error: the bin-count-weighted mean gap between
        stated confidence and observed accuracy. 0 is perfect; 0.3 means the
        typical claim is off by thirty points.
    mce:
        The worst single non-empty bin's gap. ECE can look respectable while one
        badly broken region of the confidence range does all the damage.
    bins:
        ``(lo, hi, empirical_accuracy, n)`` for every bin, including empty ones,
        so the table has a fixed shape across systems. An empty bin reports
        accuracy 0.0 with ``n == 0`` and contributes to neither ECE nor MCE.
    reliability_points:
        ``(mean_confidence, empirical_accuracy)`` for non-empty bins only --
        the diagram, where the diagonal is perfect calibration.
    """

    ece: float
    mce: float
    bins: tuple[tuple[float, float, float, int], ...] = ()
    reliability_points: tuple[tuple[float, float], ...] = ()

    def as_dict(self) -> dict[str, object]:
        """JSON-safe view."""
        return {
            "ece": self.ece,
            "mce": self.mce,
            "bins": [list(b) for b in self.bins],
            "reliability_points": [list(p) for p in self.reliability_points],
        }


def calibration(
    confidences: Sequence[float], correct: Sequence[bool], *, bins: int = 10
) -> CalibrationReport:
    """Equal-width binned calibration of ``confidences`` against ``correct``.

    Bins are equal *width* rather than equal *count* because the report is read
    against the diagonal of a reliability diagram, where fixed edges keep two
    systems comparable. The cost is that a solver whose confidences pile up near
    1.0 leaves most bins empty; ``n`` per bin is reported so that is visible.

    The top bin is closed on the right so a confidence of exactly 1.0 lands in
    it rather than falling off the end.
    """
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")
    if len(confidences) != len(correct):
        raise ValueError(
            f"got {len(confidences)} confidences for {len(correct)} outcomes"
        )

    edges = [i / bins for i in range(bins + 1)]
    counts = [0] * bins
    hits = [0] * bins
    conf_sums = [0.0] * bins

    for raw, is_ok in zip(confidences, correct, strict=False):
        p = _clamp01(raw)
        which = min(int(p * bins), bins - 1)
        counts[which] += 1
        hits[which] += int(bool(is_ok))
        conf_sums[which] += p

    total = sum(counts)
    rows: list[tuple[float, float, float, int]] = []
    points: list[tuple[float, float]] = []
    ece = 0.0
    mce = 0.0
    for i in range(bins):
        accuracy = _ratio(hits[i], counts[i])
        rows.append((edges[i], edges[i + 1], accuracy, counts[i]))
        if counts[i] == 0:
            continue
        mean_conf = conf_sums[i] / counts[i]
        gap = abs(accuracy - mean_conf)
        ece += (counts[i] / total) * gap
        mce = max(mce, gap)
        points.append((mean_conf, accuracy))

    return CalibrationReport(
        ece=float(ece),
        mce=float(mce),
        bins=tuple(rows),
        reliability_points=tuple(points),
    )


def selective_accuracy(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    thresholds: Sequence[float] = (0.5, 0.7, 0.9, 0.95, 0.99),
) -> list[tuple[float, float, float]]:
    """Risk-coverage curve: ``(threshold, coverage, accuracy above threshold)``.

    An agent that knows what it does not know is far more useful than one that
    is uniformly a bit better: if accuracy at the 0.95 threshold is 0.99 over 40%
    coverage, a human can accept those letters unread and spend their attention
    on the rest. A flat curve -- accuracy barely rising as coverage falls --
    means the confidence signal carries no information, whatever the ECE says.

    Accuracy is ``float('nan')`` where a threshold admits nothing: a gap in the
    curve is the truth there, and plotting a 0.0 would invent a data point.
    """
    if len(confidences) != len(correct):
        raise ValueError(
            f"got {len(confidences)} confidences for {len(correct)} outcomes"
        )
    values = [_clamp01(c) for c in confidences]
    flags = [bool(x) for x in correct]
    n = len(values)

    out: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        t = float(threshold)
        kept = [ok for c, ok in zip(values, flags, strict=False) if c >= t]
        coverage = _ratio(len(kept), n)
        accuracy = float("nan") if not kept else _ratio(sum(kept), len(kept))
        out.append((t, coverage, accuracy))
    return out


# --------------------------------------------------------------------------- #
# Paired system comparison
# --------------------------------------------------------------------------- #


def mcnemar(
    a_solved: Sequence[bool], b_solved: Sequence[bool]
) -> tuple[int, int, float]:
    """Exact McNemar test on paired solve/no-solve outcomes.

    Returns ``(a_only, b_only, p)``: puzzles solved by A but not B, by B but not
    A, and the two-sided exact binomial p-value on those discordant pairs. Pairs
    where both systems agree carry no information about which is better and are
    correctly ignored -- which is why a suite where both solve almost everything
    can leave a large apparent gap statistically unsupported.

    With no discordant pairs the p-value is 1.0: identical outcomes are no
    evidence of a difference.
    """
    if len(a_solved) != len(b_solved):
        raise ValueError(
            f"paired test needs equal lengths, got {len(a_solved)} and {len(b_solved)}"
        )
    a_only = sum(1 for x, y in zip(a_solved, b_solved, strict=False) if bool(x) and not bool(y))
    b_only = sum(1 for x, y in zip(a_solved, b_solved, strict=False) if bool(y) and not bool(x))
    discordant = a_only + b_only
    if discordant == 0:
        return a_only, b_only, 1.0
    return a_only, b_only, _two_sided_binomial(a_only, discordant)


def _two_sided_binomial(successes: int, trials: int) -> float:
    """Exact two-sided binomial p-value at p = 0.5.

    SciPy's ``binomtest`` is used when it is installed, but it is imported
    lazily and has a stdlib fallback: it is a ~100 MB dependency pulled in here
    for one call, which matters when this module is bundled into a serverless
    function whose only use of it is scoring a solved grid.

    The fallback is exact rather than approximate. Under p = 0.5 the
    distribution is symmetric, so doubling the smaller tail is the exact
    two-sided probability -- the case where a normal approximation or a
    general-p implementation would differ does not arise.
    """
    try:
        from scipy.stats import binomtest  # noqa: PLC0415 - optional dependency

        return float(binomtest(successes, trials, 0.5, alternative="two-sided").pvalue)
    except ImportError:
        k = min(successes, trials - successes)
        tail = sum(math.comb(trials, i) for i in range(k + 1)) / (2.0**trials)
        return min(1.0, 2.0 * tail)


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], *, n: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap the mean paired difference ``a - b``.

    Returns ``(mean_diff, ci_lo, ci_hi)`` for a 95% percentile interval.
    Resampling is *paired*: one puzzle index is drawn and both systems' values
    for that puzzle move together, which removes the puzzle-difficulty variance
    that dominates an unpaired comparison. An interval that straddles 0 means
    the suite cannot distinguish the systems.
    """
    if len(a) != len(b):
        raise ValueError(
            f"paired bootstrap needs equal lengths, got {len(a)} and {len(b)}"
        )
    diffs = np.asarray(
        [float(x) - float(y) for x, y in zip(a, b, strict=False)], dtype=np.float64
    )
    if diffs.size == 0:
        return (0.0, 0.0, 0.0)
    mean_diff = float(diffs.mean())
    if n <= 0:
        return (mean_diff, mean_diff, mean_diff)

    size = diffs.size
    rng = np.random.default_rng(seed)
    per_chunk = max(1, _BOOTSTRAP_CHUNK_ELEMENTS // size)
    parts: list[np.ndarray] = []
    drawn = 0
    while drawn < n:
        k = min(per_chunk, n - drawn)
        idx = rng.integers(0, size, size=(k, size))
        parts.append(diffs[idx].mean(axis=1))
        drawn += k
    lo, hi = _percentile_ci(np.concatenate(parts), mean_diff)
    return (mean_diff, lo, hi)


# --------------------------------------------------------------------------- #
# Failure taxonomy
# --------------------------------------------------------------------------- #

FAILURE_CATEGORIES: tuple[str, ...] = (
    "proper-noun",
    "wordplay",
    "abbreviation",
    "foreign",
    "fill-in-blank",
    "multi-word",
    "crosswordese",
    "theme",
    "plural-tense",
    "other",
)

#: Cross-reference and theme markers. A clue that points at another entry cannot
#: be answered from its own text, which is a distinct failure mode from not
#: knowing a word.
_THEME_RE = re.compile(
    r"\b\d+-\s?(?:across|down)\b"
    r"|\bthemes?\b|\bthis puzzle\b|\bpuzzle'?s\b|\bcircled\b"
    r"|\bhint to\b|\bhave in common\b"
    r"|\b(?:starts|ends|beginnings|endings)\s+of\s+(?:the\s+)?(?:answers|entries)\b",
    re.IGNORECASE,
)

#: Two or more underscores is the near-universal fill-in-the-blank marker.
_BLANK_RE = re.compile(r"_{2,}")

_ABBR_RE = re.compile(
    r"\babbr\b|\babbrev\b|\bacronym\b|\binitialism\b|\binitials\b"
    r"|\bfor short\b|\bbriefly\b|\bin brief\b|\bin short\b"
    r"|\b(?:org|grp|sch|assn|dept|agcy|hosp|univ|abbr)\.",
    re.IGNORECASE,
)

_LANGUAGES = (
    "french|spanish|german|italian|latin|greek|russian|japanese|chinese|hebrew"
    "|yiddish|portuguese|arabic|dutch|swedish|norwegian|danish|polish|hawaiian"
    "|gaelic|welsh|korean|hindi|swahili|esperanto|scottish"
)
_FOREIGN_RE = re.compile(
    rf"\b(?:in|en)\s+(?:{_LANGUAGES})\b"
    rf"|\b(?:{_LANGUAGES})\s*:"
    rf"|\b(?:{_LANGUAGES})\s+(?:for|word|words|article|pronoun|friend|farewell"
    rf"|greeting|number|month|yes|no|thanks|here|this|that|love)\b",
    re.IGNORECASE,
)

_WORDPLAY_RE = re.compile(
    r"\bpun\b|\bpunnily\b|\bplayfully\b|\bso to speak\b|\bin a way\b"
    r"|\bof sorts\b|\banagram\b|\bhomophone\b|\bwordplay\b|\bloosely\b",
    re.IGNORECASE,
)

#: Role words that usually introduce a name. Words that are equally at home in a
#: common-noun clue are deliberately absent -- "sea" ("Sea eagle") and "state"
#: ("State of mind") were tried and removed for firing on ordinary vocabulary.
_PROPER_ROLE_RE = re.compile(
    r"\b(?:actor|actress|singer|rapper|author|novelist|poet|painter|director"
    r"|composer|athlete|pitcher|quarterback|boxer|golfer|skater|astronaut"
    r"|president|senator|emperor|czar|king|queen|saint|god|goddess|city"
    r"|capital|country|county|river|lake|mountain|island|desert"
    r"|province|brand|automaker|airline|studio|network|sitcom|film|movie"
    r"|novel|opera|rocker|co-?star)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:1[5-9]\d\d|20\d\d)\b")

#: Words that routinely appear capitalised mid-clue without naming anything.
_NON_NAME_CAPS = frozenset(
    {
        "A", "AN", "THE", "I", "IT", "IS", "OR", "AND", "IN", "OF", "TO", "FOR",
        "ABBR", "VAR", "SLANG", "INFORMAL", "SEE", "E.G.", "EG", "ETC", "MR",
        "MRS", "MS", "DR", "ST",
    }
)

_MULTIWORD_RE = re.compile(
    r"\(\s*\d\s*wds?\.?\s*\)|\btwo words\b|\bthree words\b|\bhyph\b"
    r"|\b\d\s*wds?\.\b",
    re.IGNORECASE,
)

#: Answers so specific to crosswords that missing them is a vocabulary gap in
#: the *puzzle dialect* rather than in English. Hand-curated; deliberately short.
_CROSSWORDESE = frozenset(
    ["ADIT", "AERIE", "AGUE", "ALAI", "ALOE", "AMAH", "ANOA", "ARIA", "ASEA", "ASTA", "ATRI", "AVIA", "EDAM", "EDDA", "ELIA", "EMIR", "ENOL", "EPEE", "EPHA", "ERIE", "ERNE", "ERSE", "ESNE", "ETAL", "ETNA", "ETUI", "EWER", "EYRA", "IBEX", "IDES", "ILIE", "INRE", "IOTA", "IRAE", "ISLE", "ITER", "NENE", "OAST", "OBOE", "ODER", "OLEO", "OLIO", "OLLA", "OMOO", "ONER", "ORCA", "OREO", "ORTS", "OTIC", "OTTO", "RIATA", "SERA", "SLOE", "SMEE", "SNEE", "STOA", "STYE", "SUET", "TEAL", "TSAR", "UKASE", "ULNA", "UNAU", "UREA", "URSA", "UTES", "YSER", "ESAU", "EWES", "ARLO"]
)

_VOWELS = frozenset("AEIOU")

#: Suffix pairs used to decide whether two answers are the same word in
#: different number/tense. Order matters only in that longer suffixes are tried
#: too; every candidate stem is collected, not just the first.
_INFLECTIONS = ("IES", "IEST", "ING", "EST", "ED", "ES", "ER", "EN", "S", "D", "N")

_MIN_STEM = 3


def _stems(word: str) -> set[str]:
    """Every plausible stem of ``word`` under simple English inflection."""
    out = {word}
    for suffix in _INFLECTIONS:
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM:
            stem = word[: -len(suffix)]
            out.add(stem)
            if suffix in ("IES", "IEST"):
                out.add(stem + "Y")
    # Comparatives and superlatives built on a silent E: LATE -> LATER/LATEST.
    if word.endswith("R") and word[:-1].endswith("E") and len(word) - 1 >= _MIN_STEM:
        out.add(word[:-1])
    if word.endswith("ST") and word[:-2].endswith("E") and len(word) - 2 >= _MIN_STEM:
        out.add(word[:-2])
    return out


def _looks_inflectional(predicted: str, gold: str) -> bool:
    """True if the two answers share a stem but differ in number or tense."""
    if not predicted or predicted == gold:
        return False
    if abs(len(predicted) - len(gold)) > 3:
        return False
    return bool(_stems(predicted) & _stems(gold))


def _has_interior_capital(clue: str) -> bool:
    """True if a capitalised token appears anywhere but the first position.

    English sentences capitalise their first word, so an interior capital is the
    cheapest available signal that a clue names something -- a person, a place,
    a title. It over-fires on "Like Swiss cheese" and under-fires on lower-cased
    trade names, which is the accuracy this classifier claims and no more.
    """
    tokens = clue.split()
    for token in tokens[1:]:
        stripped = token.strip("\"'([{.,;:!?)]}-’“”")
        if len(stripped) < 2 or not stripped[0].isupper():
            continue
        if stripped.upper() in _NON_NAME_CAPS:
            continue
        return True
    return False


def classify_failure(clue: str, predicted: str | None, gold: str) -> str:
    """Best-effort triage of *why* an entry was missed. One of
    :data:`FAILURE_CATEGORIES`.

    **This is a heuristic, not ground truth.** It reads surface features of the
    clue text and the shape of the two answers; it has no knowledge of what any
    word means, no name list, and no access to the puzzle's theme. It will call
    "Like Swiss cheese" a proper noun and will miss an unmarked foreign word
    entirely. Its purpose is to turn a list of two hundred misses into a table
    that suggests where to look next, and the evaluation doc must say so
    wherever that table appears -- an error breakdown quoted as measurement
    would be a fabricated result.

    Checks run in this fixed order, first match winning, so the order *is* the
    policy:

    1. ``plural-tense`` -- the prediction is the gold word in another number or
       tense. This is direct evidence about the error itself rather than a guess
       from the clue, so it outranks everything.
    2. ``theme`` -- the clue cross-references another entry or names the theme;
       it was never answerable in isolation.
    3. ``fill-in-blank`` -- an underscore run.
    4. ``abbreviation`` -- an abbreviation marker in the clue, or a gold answer
       with no vowels at all.
    5. ``foreign`` -- an explicit language marker.
    6. ``wordplay`` -- a trailing ``?`` (the standard pun flag) or a pun marker.
    7. ``crosswordese`` -- the gold answer is in the curated crosswordese list.
       This curated, high-precision signal deliberately outranks the fuzzy
       proper-noun test below, which would otherwise claim "Sea eagle" -> ERNE.
    8. ``proper-noun`` -- a role word, a year, or an interior capital.
    9. ``multi-word`` -- an explicit word-count marker, or a very long answer,
       which in a daily is usually a phrase.
    10. ``other``.
    """
    text = (clue or "").strip()
    gold_word = (gold or "").strip().upper()
    guess = (predicted or "").strip().upper()
    if guess and not guess.isalpha():
        guess = ""  # a partial pattern such as "?A??" is not a prediction

    if guess and _looks_inflectional(guess, gold_word):
        return "plural-tense"
    if _THEME_RE.search(text):
        return "theme"
    if _BLANK_RE.search(text):
        return "fill-in-blank"
    if _ABBR_RE.search(text):
        return "abbreviation"
    if gold_word and len(gold_word) >= 2 and not (set(gold_word) & _VOWELS):
        return "abbreviation"
    if _FOREIGN_RE.search(text):
        return "foreign"
    if text.endswith("?") or _WORDPLAY_RE.search(text):
        return "wordplay"
    if gold_word in _CROSSWORDESE:
        return "crosswordese"
    if (
        _PROPER_ROLE_RE.search(text)
        or _YEAR_RE.search(text)
        or _has_interior_capital(text)
    ):
        return "proper-noun"
    if _MULTIWORD_RE.search(text) or len(gold_word) >= 12:
        return "multi-word"
    return "other"


def failure_breakdown(puzzle: Puzzle, score: PuzzleScore) -> dict[str, int]:
    """Count this puzzle's missed entries by :func:`classify_failure` category.

    Every category is present, including zeros, so breakdown tables from
    different runs line up column for column.
    """
    counts = {name: 0 for name in FAILURE_CATEGORIES}
    by_id = puzzle.slot_by_id
    for slot_id, predicted, gold in score.wrong_entries:
        slot = by_id.get(slot_id)
        clue = slot.clue if slot is not None else ""
        counts[classify_failure(clue, predicted, gold)] += 1
    return counts


__all__ = [
    "CalibrationReport",
    "FAILURE_CATEGORIES",
    "PuzzleScore",
    "SuiteScore",
    "aggregate",
    "calibration",
    "calibration_pairs",
    "classify_failure",
    "failure_breakdown",
    "mcnemar",
    "paired_bootstrap",
    "score_fill",
    "score_result",
    "selective_accuracy",
    "slice_scores",
]

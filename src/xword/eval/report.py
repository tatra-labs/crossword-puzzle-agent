"""Turn an :class:`~xword.eval.harness.EvalRun` into something a human reads.

The report is opinionated about order, because the order *is* the argument: a
headline number, then the breakdown that shows where it comes from, then the
ablations that say which part of the system earned it, then calibration (is the
confidence worth anything?), then the failures, then the entries that beat it,
then enough provenance to run the whole thing again.

Two deliberate choices worth knowing about:

* Anything the metrics module reports is preferred, but every number here has a
  local fallback computed from the run records. A report that renders with a
  missing metric is far more useful than one that raises, and the fallbacks are
  exact for the quantities the harness already stores per entry.
* The entry-level sections (calibration, failure taxonomy, hardest misses) read
  ``RunRecord.entries``, not the innards of a ``PuzzleScore``. That keeps this
  module working on a run reloaded from JSON months later.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from random import Random
from typing import Any

from xword.core.types import Cell, Fill, Puzzle
from xword.eval.harness import EntryRecord, EvalRun, RunRecord
from xword.eval.suites import UNKNOWN, order_labels

#: Thresholds for the selective-accuracy table when the metrics module does not
#: impose its own. Deliberately dense near 1.0: the interesting question is how
#: accurate the agent is on the entries it is *most* sure about.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)

#: Bin count for the reliability table.
CALIBRATION_BINS = 10

#: How many missed entries the "hardest" section lists.
HARDEST_N = 10

#: How many grids the HTML report renders.
MAX_GRIDS = 3


# --------------------------------------------------------------------------- #
# Tolerant field access
# --------------------------------------------------------------------------- #


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """First non-``None`` of ``names`` on an object or mapping.

    The metrics types are written against the same spec as this module but not
    by it; probing a couple of plausible field names is cheaper than making the
    whole report brittle to one rename.
    """
    if obj is None:
        return default
    for name in names:
        value = getattr(obj, name, None)
        if value is None and isinstance(obj, Mapping):
            value = obj.get(name)
        if value is not None:
            return value
    return default


def _num(obj: Any, *names: str, default: float | None = None) -> float | None:
    value = _attr(obj, *names)
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pair(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _interval(obj: Any) -> tuple[float, float] | None:
    """A (low, high) confidence interval however the score chose to store it."""
    for name in ("solve_rate_ci", "ci", "solve_rate_interval", "interval"):
        found = _pair(_attr(obj, name))
        if found:
            return found
    low = _num(obj, "ci_low", "ci_lo", "solve_rate_low", "low", "lo")
    high = _num(obj, "ci_high", "ci_hi", "solve_rate_high", "high", "hi")
    if low is not None and high is not None:
        return low, high
    return None


def _mean(values: Sequence[float]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


# --------------------------------------------------------------------------- #
# Local fallbacks for the metric functions
# --------------------------------------------------------------------------- #


def _bootstrap_ci(
    values: Sequence[float], bootstrap: int, seed: int
) -> tuple[float, float] | None:
    """Seeded percentile bootstrap of the mean -- used only when aggregate fails."""
    data = [float(v) for v in values]
    if not data or bootstrap <= 0:
        return None
    rng = Random(seed)
    n = len(data)
    means: list[float] = []
    for _ in range(bootstrap):
        total = 0.0
        for _ in range(n):
            total += data[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo = means[max(0, int(0.025 * (bootstrap - 1)))]
    hi = means[min(bootstrap - 1, int(0.975 * (bootstrap - 1)))]
    return lo, hi


def _reliability(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = CALIBRATION_BINS
) -> list[dict[str, float]]:
    """Equal-width reliability bins.

    Computed here rather than read off ``CalibrationReport`` because the
    contract fixes that type's name, not the shape of its bins; the ECE itself
    is still taken from the metrics module when it is available.
    """
    out: list[dict[str, float]] = []
    for index in range(bins):
        lo = index / bins
        hi = (index + 1) / bins
        picked = [
            (c, bool(k))
            for c, k in zip(confidences, correct)
            if (c >= lo and c < hi) or (index == bins - 1 and c >= lo and c <= 1.0000001)
        ]
        if not picked:
            out.append(
                {"lo": lo, "hi": hi, "n": 0, "confidence": 0.0, "accuracy": 0.0, "gap": 0.0}
            )
            continue
        mean_conf = sum(c for c, _ in picked) / len(picked)
        accuracy = sum(1 for _, k in picked if k) / len(picked)
        out.append(
            {
                "lo": lo,
                "hi": hi,
                "n": len(picked),
                "confidence": mean_conf,
                "accuracy": accuracy,
                "gap": accuracy - mean_conf,
            }
        )
    return out


def _ece_from(bins: Sequence[Mapping[str, float]], total: int) -> float | None:
    if not total:
        return None
    return sum(
        (b["n"] / total) * abs(b["accuracy"] - b["confidence"]) for b in bins if b["n"]
    )


def _selective(
    confidences: Sequence[float],
    correct: Sequence[bool],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[tuple[float, float, float]]:
    total = len(confidences)
    out: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        kept = [bool(k) for c, k in zip(confidences, correct) if c >= threshold]
        coverage = (len(kept) / total) if total else 0.0
        accuracy = (sum(1 for k in kept if k) / len(kept)) if kept else 0.0
        out.append((float(threshold), coverage, accuracy))
    return out


def _mcnemar(a_solved: Sequence[bool], b_solved: Sequence[bool]) -> tuple[int, int, float]:
    """Exact binomial McNemar -- the fallback when metrics is unavailable."""
    n01 = sum(1 for a, b in zip(a_solved, b_solved) if a and not b)
    n10 = sum(1 for a, b in zip(a_solved, b_solved) if b and not a)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return n01, n10, min(1.0, 2.0 * tail)


def _fallback_category(clue: str, predicted: str | None, gold: str | None) -> str:
    if not predicted:
        return "blank"
    if gold and len(predicted) != len(gold):
        return "wrong-length"
    return "wrong-answer"


# --------------------------------------------------------------------------- #
# Pulling entry-level data out of a run
# --------------------------------------------------------------------------- #


def _gradable(records: Iterable[RunRecord]) -> list[EntryRecord]:
    """Entries with a gold answer -- the only ones any metric can speak about."""
    out: list[EntryRecord] = []
    for record in records:
        out.extend(e for e in record.entries if e.gold is not None)
    return out


def _word_accuracy(records: Sequence[RunRecord]) -> float | None:
    entries = _gradable(records)
    if not entries:
        return None
    return sum(1 for e in entries if e.correct) / len(entries)


def _solved_flags(records: Sequence[RunRecord]) -> dict[str, bool]:
    """``puzzle_id -> solved``, preferring the score and falling back to entries."""
    out: dict[str, bool] = {}
    for record in records:
        value = _attr(record.score, "solved", "is_solved", "exact")
        if value is None:
            entries = [e for e in record.entries if e.gold is not None]
            value = bool(entries) and all(e.correct for e in entries)
        out[record.puzzle_id] = bool(value)
    return out


def _score_cell_accuracy(record: RunRecord) -> float | None:
    return _num(record.score, "cell_accuracy", "cell_acc", "letter_accuracy")


def _score_word_accuracy(record: RunRecord) -> float | None:
    value = _num(record.score, "word_accuracy", "word_acc", "entry_accuracy")
    if value is not None:
        return value
    entries = [e for e in record.entries if e.gold is not None]
    if not entries:
        return None
    return sum(1 for e in entries if e.correct) / len(entries)


# --------------------------------------------------------------------------- #
# summarise
# --------------------------------------------------------------------------- #


def _system_row(
    system: str, records: Sequence[RunRecord], *, bootstrap: int, seed: int
) -> dict[str, Any]:
    solved = _solved_flags(records)
    flags = [1.0 if solved.get(r.puzzle_id) else 0.0 for r in records]
    costs = [float(r.cost_usd) for r in records]
    seconds = [float(r.seconds) for r in records]

    suite_score: Any = None
    try:
        from xword.eval.metrics import aggregate

        suite_score = aggregate(
            [r.score for r in records],
            costs=costs,
            seconds=seconds,
            bootstrap=bootstrap,
            seed=seed,
        )
    except Exception:
        suite_score = None

    solve_rate = _num(suite_score, "solve_rate", "solved_rate", "exact_solve_rate")
    if solve_rate is None:
        solve_rate = _mean(flags)
    interval = _interval(suite_score) or _bootstrap_ci(flags, bootstrap, seed)

    cell = _num(suite_score, "cell_accuracy", "cell_acc", "mean_cell_accuracy")
    if cell is None:
        cell = _mean([v for v in (_score_cell_accuracy(r) for r in records) if v is not None])
    word = _num(suite_score, "word_accuracy", "word_acc", "entry_accuracy")
    if word is None:
        word = _mean([v for v in (_score_word_accuracy(r) for r in records) if v is not None])

    cost = _num(suite_score, "cost_per_puzzle", "mean_cost_usd", "cost_usd")
    if cost is None:
        cost = _mean(costs)
    wall = _num(suite_score, "seconds_per_puzzle", "mean_seconds", "wall_seconds")
    if wall is None:
        wall = _mean(seconds)

    return {
        "system": system,
        "n": len(records),
        "errors": sum(1 for r in records if r.error),
        "solve_rate": solve_rate,
        "ci": list(interval) if interval else None,
        "cell_accuracy": cell,
        "word_accuracy": word,
        "cost_usd": cost,
        "seconds": wall,
        "llm_calls": _mean([float(r.llm_calls) for r in records]),
        "candidate_coverage": _mean([float(r.candidate_coverage) for r in records]),
        "total_cost_usd": sum(costs),
    }


def _difficulty_section(
    by_system: Mapping[str, list[RunRecord]], *, bootstrap: int, seed: int
) -> dict[str, Any]:
    labels: list[str] = []
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for system, records in by_system.items():
        solved = _solved_flags(records)
        buckets: dict[str, list[float]] = {}
        for record in records:
            label = record.slices.get("difficulty") or UNKNOWN
            buckets.setdefault(label, []).append(
                1.0 if solved.get(record.puzzle_id) else 0.0
            )
        # Opportunistic: if the metrics module can slice, take its solve rates
        # (they carry its bootstrap machinery); otherwise the local means below
        # are already exact.
        sliced = _try_slice_scores(records, bootstrap=bootstrap, seed=seed)
        for label, flags in buckets.items():
            rate = _mean(flags)
            if sliced and label in sliced:
                got = _num(sliced[label], "solve_rate", "solved_rate")
                if got is not None:
                    rate = got
            table.setdefault(label, {})[system] = {"solve_rate": rate, "n": len(flags)}
            if label not in labels:
                labels.append(label)
    return {"labels": order_labels(labels), "rows": table}


def _try_slice_scores(
    records: Sequence[RunRecord], *, bootstrap: int, seed: int
) -> dict[str, Any] | None:
    try:
        from xword.eval.metrics import slice_scores
    except Exception:
        return None
    scores = [r.score for r in records]
    keys_map = {r.puzzle_id: (r.slices.get("difficulty") or UNKNOWN) for r in records}
    keys_list = [r.slices.get("difficulty") or UNKNOWN for r in records]
    for keys in (keys_map, keys_list):
        try:
            got = slice_scores(scores, keys, bootstrap=bootstrap, seed=seed)
        except Exception:
            continue
        if isinstance(got, Mapping):
            return dict(got)
    return None


def _ablation_section(
    by_system: Mapping[str, list[RunRecord]],
    headline: Mapping[str, Mapping[str, Any]],
    baseline: str,
) -> list[dict[str, Any]]:
    if baseline not in by_system:
        return []
    base_solved = _solved_flags(by_system[baseline])
    rows: list[dict[str, Any]] = []
    for system, records in by_system.items():
        if system == baseline:
            continue
        other_solved = _solved_flags(records)
        shared = sorted(set(base_solved) & set(other_solved))
        a = [base_solved[p] for p in shared]
        b = [other_solved[p] for p in shared]
        try:
            from xword.eval.metrics import mcnemar

            n01, n10, p_value = mcnemar(a, b)
        except Exception:
            n01, n10, p_value = _mcnemar(a, b)
        base_row = headline.get(baseline, {})
        row = headline.get(system, {})
        rows.append(
            {
                "system": system,
                "n": len(shared),
                "delta_solve_rate": _delta(row.get("solve_rate"), base_row.get("solve_rate")),
                "delta_cell_accuracy": _delta(
                    row.get("cell_accuracy"), base_row.get("cell_accuracy")
                ),
                "delta_word_accuracy": _delta(
                    row.get("word_accuracy"), base_row.get("word_accuracy")
                ),
                "delta_cost_usd": _delta(row.get("cost_usd"), base_row.get("cost_usd")),
                "baseline_only": int(n01),
                "system_only": int(n10),
                "p_value": float(p_value),
            }
        )
    return rows


def _delta(value: Any, base: Any) -> float | None:
    if value is None or base is None:
        return None
    return float(value) - float(base)


def _calibration_section(
    by_system: Mapping[str, list[RunRecord]], primary: str
) -> dict[str, Any]:
    per_system: dict[str, Any] = {}
    for system, records in by_system.items():
        entries = _gradable(records)
        confidences = [float(e.confidence) for e in entries]
        correct = [bool(e.correct) for e in entries]
        bins = _reliability(confidences, correct)
        ece = None
        try:
            from xword.eval.metrics import calibration

            report = calibration(confidences, correct, bins=CALIBRATION_BINS)
            ece = _num(report, "ece", "expected_calibration_error")
        except Exception:
            ece = None
        if ece is None:
            ece = _ece_from(bins, len(entries))
        try:
            from xword.eval.metrics import selective_accuracy

            curve = [
                (float(t), float(c), float(a))
                for t, c, a in selective_accuracy(confidences, correct)
            ]
        except Exception:
            curve = _selective(confidences, correct)
        per_system[system] = {
            "ece": ece,
            "n": len(entries),
            "mean_confidence": _mean(confidences),
            "accuracy": _mean([1.0 if c else 0.0 for c in correct]),
            "bins": bins,
            "selective": [list(x) for x in curve],
        }
    return {"per_system": per_system, "primary": primary}


def _failure_section(by_system: Mapping[str, list[RunRecord]]) -> dict[str, Any]:
    try:
        from xword.eval.metrics import FAILURE_CATEGORIES

        known = [str(c) for c in FAILURE_CATEGORIES]
    except Exception:
        known = []
    try:
        from xword.eval.metrics import classify_failure as _classify
    except Exception:
        _classify = None

    counts: dict[str, dict[str, int]] = {}
    seen: list[str] = list(known)
    for system, records in by_system.items():
        bucket: dict[str, int] = {}
        for entry in _gradable(records):
            if entry.correct:
                continue
            category: str
            if _classify is not None:
                try:
                    # predicted stays None for an unanswered entry: the
                    # classifier distinguishes "no guess" from "wrong guess".
                    category = str(
                        _classify(entry.clue, entry.predicted, entry.gold or "")
                    )
                except Exception:
                    category = _fallback_category(entry.clue, entry.predicted, entry.gold)
            else:
                category = _fallback_category(entry.clue, entry.predicted, entry.gold)
            bucket[category] = bucket.get(category, 0) + 1
            if category not in seen:
                seen.append(category)
        counts[system] = bucket
    ordered = [c for c in known if any(counts[s].get(c) for s in counts)]
    extras = sorted(c for c in seen if c not in known)
    return {
        "categories": ordered + extras,
        "counts": counts,
        "totals": {s: sum(b.values()) for s, b in counts.items()},
    }


def _hardest_section(run: EvalRun, primary: str) -> list[dict[str, Any]]:
    """Entries missed by the most systems, confidently-wrong ones first."""
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for record in run.records:
        for entry in record.entries:
            if entry.gold is None:
                continue
            key = (record.puzzle_id, entry.slot_id)
            slot = agg.setdefault(
                key,
                {
                    "puzzle_id": record.puzzle_id,
                    "slot_id": entry.slot_id,
                    "clue": entry.clue,
                    "gold": entry.gold,
                    "predicted": None,
                    "attempts": 0,
                    "missed_by": 0,
                    "confidence": 0.0,
                },
            )
            slot["attempts"] += 1
            if entry.correct:
                continue
            slot["missed_by"] += 1
            slot["confidence"] = max(slot["confidence"], float(entry.confidence))
            if slot["predicted"] is None or record.system == primary:
                slot["predicted"] = entry.predicted
    missed = [v for v in agg.values() if v["missed_by"]]
    missed.sort(
        key=lambda v: (
            -v["missed_by"],
            -v["confidence"],
            v["puzzle_id"],
            v["slot_id"],
        )
    )
    return missed[:HARDEST_N]


def summarise(run: EvalRun, *, bootstrap: int = 2000, seed: int = 0) -> dict:
    """Everything the rendered reports display, as plain JSON-safe data."""
    by_system = run.by_system()
    systems = [s for s in run.systems if s in by_system] or sorted(by_system)
    primary = "full" if "full" in systems else (systems[0] if systems else "")

    headline = {
        system: _system_row(system, by_system[system], bootstrap=bootstrap, seed=seed)
        for system in systems
    }
    config = run.config()
    base_config = config.get("base_config") or {}

    summary: dict[str, Any] = {
        "suite": run.suite,
        "systems": systems,
        "primary": primary,
        "n_puzzles": len(run.puzzle_ids),
        "n_records": len(run.records),
        "errors": sum(1 for r in run.records if r.error),
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "git_sha": run.git_sha,
        "model": str(base_config.get("model", "unknown")),
        "seed": base_config.get("seed", seed),
        "bootstrap": bootstrap,
        "metric_seed": seed,
        "workers": config.get("workers", 1),
        "headline": [headline[s] for s in systems],
        "difficulty": _difficulty_section(by_system, bootstrap=bootstrap, seed=seed),
        "ablations": _ablation_section(by_system, headline, primary),
        "calibration": _calibration_section(by_system, primary),
        "failures": _failure_section(by_system),
        "hardest": _hardest_section(run, primary),
    }
    summary["command"] = _command(summary)
    return summary


def _command(summary: Mapping[str, Any]) -> str:
    """The canonical CLI invocation that reproduces this run."""
    systems = ",".join(summary.get("systems") or ["full"])
    return (
        "PYTHONPATH=src python -m xword.cli eval "
        f"--suite {summary.get('suite', 'bundled')} "
        f"--systems {systems} "
        f"--seed {summary.get('seed', 0)} "
        f"--workers {summary.get('workers', 1)} "
        f"--out reports/{summary.get('suite', 'run')}"
    )


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def _pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.{digits}f}%"


def _signed_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:+.{digits}f}pp"


def _usd(value: Any) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    return f"${value:.4f}" if abs(value) < 1 else f"${value:.2f}"


def _signed_usd(value: Any) -> str:
    if value is None:
        return "n/a"
    return ("+" if float(value) >= 0 else "-") + _usd(abs(float(value)))


def _secs(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f}"


def _ci_text(ci: Any) -> str:
    pair = _pair(ci)
    if not pair:
        return ""
    return f" [{pair[0] * 100:.1f}, {pair[1] * 100:.1f}]"


def _p_text(value: Any) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _clip(text: str, width: int = 44) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|")

    lines = ["| " + " | ".join(cell(h) for h in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(cell(c) for c in row) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


def to_markdown(run: EvalRun, summary: dict | None = None) -> str:
    """Render the full report as markdown."""
    data = summary if summary is not None else summarise(run)
    systems: list[str] = list(data["systems"])
    primary: str = data["primary"]
    out: list[str] = []

    out.append(f"# Crossword evaluation - {data['suite']}")
    out.append("")
    out.append(
        f"{data['n_puzzles']} puzzles x {len(systems)} system(s) = "
        f"{data['n_records']} runs, {data['errors']} error(s). "
        f"Finished {data['finished_at']}."
    )
    out.append("")

    # 1 -- headline ---------------------------------------------------------
    out.append("## 1. Headline")
    out.append("")
    out.append(
        _md_table(
            [
                "System",
                "Solve rate (95% CI)",
                "Cell acc",
                "Word acc",
                "$/puzzle",
                "s/puzzle",
                "n",
                "Errors",
            ],
            [
                [
                    f"`{row['system']}`",
                    _pct(row["solve_rate"]) + _ci_text(row["ci"]),
                    _pct(row["cell_accuracy"]),
                    _pct(row["word_accuracy"]),
                    _usd(row["cost_usd"]),
                    _secs(row["seconds"]),
                    str(row["n"]),
                    str(row["errors"]),
                ]
                for row in data["headline"]
            ],
        )
    )
    out.append("")
    out.append(
        f"CIs are {data['bootstrap']} bootstrap resamples over puzzles "
        f"(seed {data['metric_seed']}); a puzzle counts as solved only when every "
        "gradable entry is right."
    )
    out.append("")

    # 2 -- difficulty -------------------------------------------------------
    out.append("## 2. Difficulty breakdown")
    out.append("")
    diff = data["difficulty"]
    if diff["labels"]:
        out.append("Solve rate by day of week / difficulty label, with n per cell.")
        out.append("")
        rows = []
        for label in diff["labels"]:
            cells = diff["rows"].get(label, {})
            row = [label]
            for system in systems:
                got = cells.get(system)
                row.append(
                    f"{_pct(got['solve_rate'])} ({got['n']})" if got else "-"
                )
            rows.append(row)
        out.append(_md_table(["Difficulty"] + [f"`{s}`" for s in systems], rows))
    else:
        out.append("_No difficulty metadata on this suite._")
    out.append("")

    # 3 -- ablations --------------------------------------------------------
    out.append(f"## 3. Ablation deltas vs `{primary}`")
    out.append("")
    if data["ablations"]:
        out.append(
            _md_table(
                [
                    "System",
                    "d solve rate",
                    "d cell acc",
                    "d word acc",
                    "d $/puzzle",
                    f"`{primary}` only",
                    "System only",
                    "McNemar p",
                    "n",
                ],
                [
                    [
                        f"`{row['system']}`",
                        _signed_pct(row["delta_solve_rate"]),
                        _signed_pct(row["delta_cell_accuracy"]),
                        _signed_pct(row["delta_word_accuracy"]),
                        _signed_usd(row["delta_cost_usd"]),
                        str(row["baseline_only"]),
                        str(row["system_only"]),
                        _p_text(row["p_value"]),
                        str(row["n"]),
                    ]
                    for row in data["ablations"]
                ],
            )
        )
        out.append("")
        out.append(
            f"`{primary}` only / System only are the discordant puzzles McNemar's "
            "exact test is computed on; everything else is uninformative about the "
            "difference."
        )
    else:
        out.append(f"_Only one system (`{primary}`) in this run; nothing to compare._")
    out.append("")

    # 4 -- calibration ------------------------------------------------------
    out.append("## 4. Calibration")
    out.append("")
    calib = data["calibration"]
    out.append(
        _md_table(
            ["System", "ECE", "Mean confidence", "Entry accuracy", "Entries"],
            [
                [
                    f"`{system}`",
                    _pct(calib["per_system"][system]["ece"], 2),
                    _pct(calib["per_system"][system]["mean_confidence"]),
                    _pct(calib["per_system"][system]["accuracy"]),
                    str(calib["per_system"][system]["n"]),
                ]
                for system in systems
            ],
        )
    )
    out.append("")
    detail = calib["per_system"].get(primary, {})
    out.append(f"Reliability, system `{primary}` ({CALIBRATION_BINS} equal-width bins):")
    out.append("")
    out.append(
        _md_table(
            ["Confidence bin", "n", "Mean confidence", "Accuracy", "Gap"],
            [
                [
                    f"{b['lo']:.1f}-{b['hi']:.1f}",
                    str(int(b["n"])),
                    _pct(b["confidence"]) if b["n"] else "-",
                    _pct(b["accuracy"]) if b["n"] else "-",
                    _signed_pct(b["gap"]) if b["n"] else "-",
                ]
                for b in detail.get("bins", [])
            ],
        )
    )
    out.append("")
    out.append(f"Selective accuracy, system `{primary}` (answer only above threshold):")
    out.append("")
    out.append(
        _md_table(
            ["Confidence >=", "Coverage", "Accuracy"],
            [
                [
                    f"{t:.2f}",
                    _pct(coverage),
                    _pct(accuracy) if coverage else "-",
                ]
                for t, coverage, accuracy in detail.get("selective", [])
            ],
        )
    )
    out.append("")

    # 5 -- failures ---------------------------------------------------------
    out.append("## 5. Failure taxonomy")
    out.append("")
    failures = data["failures"]
    if failures["categories"]:
        rows = [
            [category]
            + [str(failures["counts"].get(s, {}).get(category, 0)) for s in systems]
            for category in failures["categories"]
        ]
        rows.append(
            ["**total**"] + [str(failures["totals"].get(s, 0)) for s in systems]
        )
        out.append(_md_table(["Category"] + [f"`{s}`" for s in systems], rows))
    else:
        out.append("_No incorrect entries to classify._")
    out.append("")

    # 6 -- hardest ----------------------------------------------------------
    out.append(f"## 6. Hardest {HARDEST_N} entries missed")
    out.append("")
    if data["hardest"]:
        out.append(
            _md_table(
                ["Puzzle", "Entry", "Clue", "Predicted", "Gold", "Missed by", "Max conf"],
                [
                    [
                        item["puzzle_id"],
                        item["slot_id"],
                        _clip(item["clue"]),
                        item["predicted"] or "_(blank)_",
                        item["gold"] or "",
                        f"{item['missed_by']}/{item['attempts']}",
                        _pct(item["confidence"]),
                    ]
                    for item in data["hardest"]
                ],
            )
        )
        out.append("")
        out.append(
            "Ranked by how many systems missed the entry, then by how confident the "
            "wrong answer was - a confidently wrong entry corrupts its crossings too."
        )
    else:
        out.append("_Nothing was missed._")
    out.append("")

    # 7 -- reproducibility --------------------------------------------------
    out.append("## 7. Reproducibility")
    out.append("")
    out.append(f"- git sha: `{data['git_sha']}`")
    out.append(f"- model: `{data['model']}`")
    out.append(f"- suite: `{data['suite']}` ({data['n_puzzles']} puzzles)")
    out.append(f"- systems: {', '.join('`' + s + '`' for s in systems)}")
    out.append(f"- seed: `{data['seed']}` (metrics seed `{data['metric_seed']}`)")
    out.append(f"- started: {data['started_at']} / finished: {data['finished_at']}")
    out.append("")
    out.append("```")
    out.append(data["command"])
    out.append("```")
    out.append("")

    return "\n".join(out)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

_CSS = """
.xw-report {
  --xw-bg: #ffffff;
  --xw-fg: #17181a;
  --xw-muted: #63666b;
  --xw-line: #e3e5e8;
  --xw-head: #f4f5f7;
  --xw-accent: #3155c6;
  --xw-good: #1c7c3c;
  --xw-bad: #b32222;
  background: var(--xw-bg);
  color: var(--xw-fg);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  padding: 1.5rem;
  max-width: 66rem;
  margin: 0 auto;
}
@media (prefers-color-scheme: dark) {
  .xw-report {
    --xw-bg: #16181c;
    --xw-fg: #e9eaec;
    --xw-muted: #a2a6ad;
    --xw-line: #2c3037;
    --xw-head: #20242a;
    --xw-accent: #8ea6ff;
    --xw-good: #63c98a;
    --xw-bad: #f0736e;
  }
}
.xw-report h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
.xw-report h2 { font-size: 1.1rem; margin: 2rem 0 .5rem; border-bottom: 1px solid var(--xw-line); padding-bottom: .25rem; }
.xw-report h3 { font-size: .95rem; margin: 1.25rem 0 .5rem; color: var(--xw-muted); }
.xw-report p, .xw-report li { color: var(--xw-fg); }
.xw-report .xw-note { color: var(--xw-muted); font-size: .85rem; }
.xw-report table { border-collapse: collapse; width: 100%; margin: .5rem 0 1rem; font-size: .88rem; }
.xw-report th, .xw-report td { border: 1px solid var(--xw-line); padding: .35rem .55rem; text-align: right; }
.xw-report th { background: var(--xw-head); font-weight: 600; }
.xw-report th:first-child, .xw-report td:first-child { text-align: left; }
.xw-report td.xw-text, .xw-report th.xw-text { text-align: left; }
.xw-report code, .xw-report pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.xw-report pre { background: var(--xw-head); border: 1px solid var(--xw-line); padding: .6rem .75rem; overflow-x: auto; border-radius: 4px; }
.xw-report .xw-scroll { overflow-x: auto; }
.xw-report .xw-grids { display: flex; flex-wrap: wrap; gap: 1.25rem; }
.xw-report .xw-grid { border: 1px solid var(--xw-line); border-radius: 4px; padding: .75rem; background: var(--xw-head); }
.xw-report .xw-good { color: var(--xw-good); }
.xw-report .xw-bad { color: var(--xw-bad); }
.xw-report footer { margin-top: 2rem; color: var(--xw-muted); font-size: .85rem; }
"""


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _html_table(
    headers: Sequence[str], rows: Sequence[Sequence[str]], *, text_cols: Sequence[int] = (0,)
) -> str:
    head = "".join(
        f'<th class="xw-text">{h}</th>' if i in text_cols else f"<th>{h}</th>"
        for i, h in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = "".join(
            f'<td class="xw-text">{c}</td>' if i in text_cols else f"<td>{c}</td>"
            for i, c in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        '<div class="xw-scroll"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _puzzle_map(puzzles: Any) -> dict[str, Puzzle]:
    if puzzles is None:
        return {}
    if isinstance(puzzles, Mapping):
        return {str(k): v for k, v in puzzles.items()}
    return {p.id: p for p in puzzles}


def _letters_for(puzzle: Puzzle, entries: Sequence[EntryRecord]) -> dict[Cell, str]:
    fill = Fill({})
    by_id = puzzle.slot_by_id
    for entry in entries:
        slot = by_id.get(entry.slot_id)
        if slot is None or not entry.predicted:
            continue
        if len(entry.predicted) != slot.length:
            continue
        fill = fill.with_slot(slot, entry.predicted)
    return dict(fill.letters)


def _cell_confidence(puzzle: Puzzle, entries: Sequence[EntryRecord]) -> dict[Cell, float]:
    by_id = puzzle.slot_by_id
    out: dict[Cell, float] = {}
    for entry in entries:
        slot = by_id.get(entry.slot_id)
        if slot is None or not entry.predicted:
            continue
        for cell in slot.cells:
            value = float(entry.confidence)
            out[cell] = min(out[cell], value) if cell in out else value
    return out


def _grid_html(run: EvalRun, data: Mapping[str, Any], puzzles: Any) -> str:
    """Render up to :data:`MAX_GRIDS` of the primary system's fills."""
    lookup = _puzzle_map(puzzles)
    if not lookup:
        return ""
    try:
        from xword.io.render import grid_to_html
    except Exception:
        return ""

    primary = data["primary"]
    records = [r for r in run.records if r.system == primary and r.puzzle_id in lookup]
    records.sort(key=lambda r: (-(_score_word_accuracy(r) or 0.0), r.puzzle_id))
    chunks: list[str] = []
    for record in records[:MAX_GRIDS]:
        puzzle = lookup[record.puzzle_id]
        letters = _letters_for(puzzle, record.entries)
        gold = puzzle.solution_letters() if puzzle.has_solution else None
        accuracy = _score_word_accuracy(record)
        title = f"{puzzle.id} - {_pct(accuracy)} entries correct"
        try:
            rendered = grid_to_html(
                puzzle,
                letters,
                confidence=_cell_confidence(puzzle, record.entries),
                gold=gold,
                title=title,
            )
        except Exception:
            continue
        chunks.append(f'<div class="xw-grid">{rendered}</div>')
    if not chunks:
        return ""
    return (
        "<h2>Grids</h2>"
        f'<p class="xw-note">System <code>{_esc(primary)}</code>, best-scoring '
        f"{len(chunks)} puzzle(s).</p>"
        '<div class="xw-grids">' + "".join(chunks) + "</div>"
    )


def to_html(run: EvalRun, summary: dict | None = None, *, puzzles: Any = None) -> str:
    """The same report as HTML, readable in light and dark themes.

    Emitted as a complete document whose body is a single self-contained
    ``<section class="xw-report">`` carrying its own scoped styles, so it can be
    served as-is or lifted out and embedded in another page.
    """
    data = summary if summary is not None else summarise(run)
    systems: list[str] = list(data["systems"])
    primary: str = data["primary"]
    parts: list[str] = []

    parts.append(f"<h1>Crossword evaluation - {_esc(data['suite'])}</h1>")
    parts.append(
        f'<p class="xw-note">{data["n_puzzles"]} puzzles &times; {len(systems)} '
        f'system(s) = {data["n_records"]} runs, {data["errors"]} error(s). '
        f'Finished {_esc(data["finished_at"])}.</p>'
    )

    parts.append("<h2>1. Headline</h2>")
    parts.append(
        _html_table(
            [
                "System",
                "Solve rate (95% CI)",
                "Cell acc",
                "Word acc",
                "$/puzzle",
                "s/puzzle",
                "n",
                "Errors",
            ],
            [
                [
                    f"<code>{_esc(row['system'])}</code>",
                    _pct(row["solve_rate"]) + _esc(_ci_text(row["ci"])),
                    _pct(row["cell_accuracy"]),
                    _pct(row["word_accuracy"]),
                    _usd(row["cost_usd"]),
                    _secs(row["seconds"]),
                    str(row["n"]),
                    str(row["errors"]),
                ]
                for row in data["headline"]
            ],
        )
    )
    parts.append(
        f'<p class="xw-note">CIs are {data["bootstrap"]} bootstrap resamples over '
        f'puzzles (seed {data["metric_seed"]}).</p>'
    )

    parts.append("<h2>2. Difficulty breakdown</h2>")
    diff = data["difficulty"]
    if diff["labels"]:
        rows = []
        for label in diff["labels"]:
            cells = diff["rows"].get(label, {})
            row = [_esc(label)]
            for system in systems:
                got = cells.get(system)
                row.append(f"{_pct(got['solve_rate'])} ({got['n']})" if got else "-")
            rows.append(row)
        parts.append(
            _html_table(
                ["Difficulty"] + [f"<code>{_esc(s)}</code>" for s in systems], rows
            )
        )
    else:
        parts.append('<p class="xw-note">No difficulty metadata on this suite.</p>')

    parts.append(f"<h2>3. Ablation deltas vs <code>{_esc(primary)}</code></h2>")
    if data["ablations"]:
        parts.append(
            _html_table(
                [
                    "System",
                    "&Delta; solve rate",
                    "&Delta; cell acc",
                    "&Delta; word acc",
                    "&Delta; $/puzzle",
                    f"<code>{_esc(primary)}</code> only",
                    "System only",
                    "McNemar p",
                    "n",
                ],
                [
                    [
                        f"<code>{_esc(row['system'])}</code>",
                        _delta_html(row["delta_solve_rate"]),
                        _delta_html(row["delta_cell_accuracy"]),
                        _delta_html(row["delta_word_accuracy"]),
                        _signed_usd(row["delta_cost_usd"]),
                        str(row["baseline_only"]),
                        str(row["system_only"]),
                        _p_text(row["p_value"]),
                        str(row["n"]),
                    ]
                    for row in data["ablations"]
                ],
            )
        )
    else:
        parts.append('<p class="xw-note">Only one system in this run.</p>')

    parts.append("<h2>4. Calibration</h2>")
    calib = data["calibration"]
    parts.append(
        _html_table(
            ["System", "ECE", "Mean confidence", "Entry accuracy", "Entries"],
            [
                [
                    f"<code>{_esc(system)}</code>",
                    _pct(calib["per_system"][system]["ece"], 2),
                    _pct(calib["per_system"][system]["mean_confidence"]),
                    _pct(calib["per_system"][system]["accuracy"]),
                    str(calib["per_system"][system]["n"]),
                ]
                for system in systems
            ],
        )
    )
    detail = calib["per_system"].get(primary, {})
    parts.append(f"<h3>Reliability - <code>{_esc(primary)}</code></h3>")
    parts.append(
        _html_table(
            ["Confidence bin", "n", "Mean confidence", "Accuracy", "Gap"],
            [
                [
                    f"{b['lo']:.1f}-{b['hi']:.1f}",
                    str(int(b["n"])),
                    _pct(b["confidence"]) if b["n"] else "-",
                    _pct(b["accuracy"]) if b["n"] else "-",
                    _delta_html(b["gap"]) if b["n"] else "-",
                ]
                for b in detail.get("bins", [])
            ],
        )
    )
    parts.append(f"<h3>Selective accuracy - <code>{_esc(primary)}</code></h3>")
    parts.append(
        _html_table(
            ["Confidence &ge;", "Coverage", "Accuracy"],
            [
                [
                    f"{t:.2f}",
                    _pct(coverage),
                    _pct(accuracy) if coverage else "-",
                ]
                for t, coverage, accuracy in detail.get("selective", [])
            ],
        )
    )

    parts.append("<h2>5. Failure taxonomy</h2>")
    failures = data["failures"]
    if failures["categories"]:
        rows = [
            [_esc(category)]
            + [str(failures["counts"].get(s, {}).get(category, 0)) for s in systems]
            for category in failures["categories"]
        ]
        rows.append(
            ["<strong>total</strong>"]
            + [str(failures["totals"].get(s, 0)) for s in systems]
        )
        parts.append(
            _html_table(["Category"] + [f"<code>{_esc(s)}</code>" for s in systems], rows)
        )
    else:
        parts.append('<p class="xw-note">No incorrect entries to classify.</p>')

    parts.append(f"<h2>6. Hardest {HARDEST_N} entries missed</h2>")
    if data["hardest"]:
        parts.append(
            _html_table(
                ["Puzzle", "Entry", "Clue", "Predicted", "Gold", "Missed by", "Max conf"],
                [
                    [
                        _esc(item["puzzle_id"]),
                        _esc(item["slot_id"]),
                        _esc(_clip(item["clue"], 60)),
                        f'<span class="xw-bad">{_esc(item["predicted"] or "(blank)")}</span>',
                        f'<span class="xw-good">{_esc(item["gold"] or "")}</span>',
                        f"{item['missed_by']}/{item['attempts']}",
                        _pct(item["confidence"]),
                    ]
                    for item in data["hardest"]
                ],
                text_cols=(0, 1, 2, 3, 4),
            )
        )
    else:
        parts.append('<p class="xw-note">Nothing was missed.</p>')

    grids = _grid_html(run, data, puzzles)
    if grids:
        parts.append(grids)

    parts.append("<h2>7. Reproducibility</h2>")
    parts.append(
        "<ul>"
        f"<li>git sha: <code>{_esc(data['git_sha'])}</code></li>"
        f"<li>model: <code>{_esc(data['model'])}</code></li>"
        f"<li>suite: <code>{_esc(data['suite'])}</code> ({data['n_puzzles']} puzzles)</li>"
        f"<li>systems: {', '.join('<code>' + _esc(s) + '</code>' for s in systems)}</li>"
        f"<li>seed: <code>{_esc(data['seed'])}</code> "
        f"(metrics seed <code>{_esc(data['metric_seed'])}</code>)</li>"
        f"<li>started: {_esc(data['started_at'])} / finished: {_esc(data['finished_at'])}</li>"
        "</ul>"
    )
    parts.append(f"<pre>{_esc(data['command'])}</pre>")

    body = (
        '<section class="xw-report"><style>'
        + _CSS
        + "</style>"
        + "".join(parts)
        + "</section>"
    )
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="color-scheme" content="light dark">\n'
        f"<title>Crossword evaluation - {_esc(data['suite'])}</title>\n"
        "</head>\n<body>\n" + body + "\n</body>\n</html>\n"
    )


def _delta_html(value: Any) -> str:
    if value is None:
        return "n/a"
    css = "xw-good" if float(value) >= 0 else "xw-bad"
    return f'<span class="{css}">{_signed_pct(value)}</span>'


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def write_report(run: EvalRun, out_dir: Path, *, puzzles: Any = None) -> dict[str, Path]:
    """Write markdown, HTML and the machine-readable summary into ``out_dir``.

    The summary JSON is written alongside the prose because every number in the
    report comes from it: a downstream plot or a regression check should read
    that file rather than re-parse the markdown.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarise(run)

    paths = {
        "markdown": out_dir / "report.md",
        "html": out_dir / "report.html",
        "summary": out_dir / "summary.json",
        "run": out_dir / "run.json",
    }
    paths["markdown"].write_text(to_markdown(run, summary), encoding="utf-8")
    paths["html"].write_text(to_html(run, summary, puzzles=puzzles), encoding="utf-8")
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=False), encoding="utf-8"
    )
    paths["run"].write_text(run.to_json(), encoding="utf-8")
    return paths


__all__ = [
    "CALIBRATION_BINS",
    "DEFAULT_THRESHOLDS",
    "HARDEST_N",
    "MAX_GRIDS",
    "summarise",
    "to_html",
    "to_markdown",
    "write_report",
]

"""Run a suite through one or more systems and record what happened.

The harness owns three things the rest of the eval stack depends on:

* **Durability.** Every finished (puzzle, system) pair is appended to a JSON
  Lines file the moment it completes, and a re-run with the same ``out_path``
  skips pairs already in that file. A four-hour paid run that dies at hour
  three must not have to be paid for twice.
* **Isolation.** One puzzle that raises must not lose the other ninety-nine, so
  every unit of work is wrapped and a crash becomes a record with ``error`` set
  rather than a traceback out of ``run_suite``.
* **Self-containment.** A record carries the slice labels and the per-entry
  outcomes it will later be reported on, so a saved run renders into the same
  report months later without the puzzle files or the solver being present.

Scoring itself lives in ``xword.eval.metrics``; it is imported lazily so that a
harness import does not drag in the metric stack (and so this module stays
importable while the eval package is still being assembled).
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import threading
import time
from collections import abc
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from xword import config as xconfig
from xword.config import AgentConfig, config_for_ablation
from xword.core.types import Cell, Fill, Puzzle, SlotOutcome, SolveResult, SolveStats
from xword.eval.suites import SLICE_KINDS, Suite

if TYPE_CHECKING:  # pragma: no cover - typing only
    from xword.eval.metrics import PuzzleScore

SolverFactory = Callable[[AgentConfig], Any]

#: Bumped if the on-disk record shape changes incompatibly.
SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Score (de)serialisation
# --------------------------------------------------------------------------- #


class _ScoreView:
    """Attribute access over a plain dict of score fields.

    A run that is read back from JSON has only dicts. Rather than force the
    report to branch on "real ``PuzzleScore`` or decoded dict", a decoded score
    is wrapped in this, so both are read the same way: ``score.cell_accuracy``.
    It is also the fallback when ``xword.eval.metrics`` is unavailable or its
    scorer raises, which keeps a partial run reportable.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        self._data = dict(data or {})

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(name) from None
        return _wrap_value(value)

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def get(self, name: str, default: Any = None) -> Any:
        return _wrap_value(self._data[name]) if name in self._data else default

    def keys(self):
        return self._data.keys()

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_ScoreView({self._data!r})"


def _wrap_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _ScoreView(value)
    if isinstance(value, (list, tuple)):
        return [_wrap_value(v) for v in value]
    return value


def _jsonable(value: Any) -> Any:
    """Coerce arbitrary score payloads into JSON-safe structures.

    Handles the shapes a metrics module plausibly produces: dataclasses, numpy
    scalars, ``Cell`` keys, sets and paths. Anything else degrades to ``str``
    rather than blowing up a run at write time.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, _ScoreView):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)  # numpy scalars
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    return str(value)


def _score_to_dict(score: Any) -> dict[str, Any]:
    if score is None:
        return {}
    if isinstance(score, _ScoreView):
        return _jsonable(score.to_dict())
    as_dict = getattr(score, "as_dict", None)
    if callable(as_dict):  # a score that documents its own JSON view wins
        try:
            return _jsonable(as_dict())
        except Exception:
            pass
    if dataclasses.is_dataclass(score) and not isinstance(score, type):
        return _jsonable(dataclasses.asdict(score))
    if isinstance(score, Mapping):
        return _jsonable(dict(score))
    data = getattr(score, "__dict__", None)
    if isinstance(data, dict) and data:
        return _jsonable(dict(data))
    out: dict[str, Any] = {}
    for name in dir(score):
        if name.startswith("_"):
            continue
        try:
            value = getattr(score, name)
        except Exception:
            continue
        if callable(value):
            continue
        out[name] = _jsonable(value)
    return out


def _rebuild(cls: type, data: Mapping[str, Any]) -> Any:
    """Reconstruct a dataclass (one level of nesting) from decoded JSON."""
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        kwargs[f.name] = _coerce(hints.get(f.name), data[f.name])
    return cls(**kwargs)


def _coerce(annotation: Any, value: Any) -> Any:
    """Push decoded JSON back towards the declared type.

    JSON has no tuples and no non-string dict keys, so a field declared
    ``tuple[tuple[str, str, str], ...]`` or ``Mapping[int, float]`` comes back
    subtly different from what was written. Restoring the shape here keeps a
    resumed record interchangeable with a freshly computed one.
    """
    if annotation is None or annotation is Any:
        return value
    if dataclasses.is_dataclass(annotation) and isinstance(value, Mapping):
        return _rebuild(annotation, value)

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union or origin is UnionType:
        for candidate in args:
            if candidate is type(None):
                continue
            try:
                return _coerce(candidate, value)
            except Exception:
                continue
        return value

    if isinstance(value, (list, tuple)) and _is_sequence_origin(origin):
        if origin is tuple and args and args[-1] is not Ellipsis and len(args) == len(value):
            items = [_coerce(a, v) for a, v in zip(args, value, strict=False)]
        else:
            inner = args[0] if args and args[0] is not Ellipsis else None
            items = [_coerce(inner, v) for v in value]
        if origin in (tuple, set, frozenset):
            return origin(items)
        return items

    if isinstance(value, Mapping) and _is_mapping_origin(origin):
        key_type = args[0] if args else None
        val_type = args[1] if len(args) > 1 else None
        return {
            _coerce_key(key_type, k): _coerce(val_type, v) for k, v in value.items()
        }

    return value


def _is_sequence_origin(origin: Any) -> bool:
    if origin in (list, tuple, set, frozenset):
        return True
    return isinstance(origin, type) and issubclass(origin, abc.Sequence)


def _is_mapping_origin(origin: Any) -> bool:
    if origin is dict:
        return True
    return isinstance(origin, type) and issubclass(origin, abc.Mapping)


def _coerce_key(annotation: Any, key: Any) -> Any:
    if annotation is int and isinstance(key, str):
        try:
            return int(key)
        except ValueError:
            return key
    return key


def _score_from_dict(data: Mapping[str, Any]) -> Any:
    """Best-effort inverse of :func:`_score_to_dict`.

    Rebuilds a real ``PuzzleScore`` when the class is importable and the fields
    line up, because ``metrics.aggregate`` may reasonably type-check what it is
    handed; falls back to a :class:`_ScoreView` otherwise.
    """
    try:
        from xword.eval.metrics import PuzzleScore as _PuzzleScore
    except Exception:
        return _ScoreView(data)
    if not dataclasses.is_dataclass(_PuzzleScore):
        return _ScoreView(data)
    try:
        return _rebuild(_PuzzleScore, data)
    except Exception:
        return _ScoreView(data)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EntryRecord:
    """What one system answered for one entry of one puzzle.

    The report's calibration, failure-taxonomy and hardest-entry sections all
    need entry-level truth. Deriving it here, from the ``SolveResult`` the
    harness already holds, keeps those sections independent of how a
    ``PuzzleScore`` chooses to store its internals -- and means a saved run
    still reports fully.
    """

    slot_id: str
    clue: str
    predicted: str | None
    gold: str | None
    confidence: float = 0.0
    correct: bool = False
    source: str = ""
    length: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "clue": self.clue,
            "predicted": self.predicted,
            "gold": self.gold,
            "confidence": float(self.confidence),
            "correct": bool(self.correct),
            "source": self.source,
            "length": int(self.length),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EntryRecord:
        return cls(
            slot_id=str(data.get("slot_id", "")),
            clue=str(data.get("clue", "")),
            predicted=data.get("predicted"),
            gold=data.get("gold"),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            correct=bool(data.get("correct", False)),
            source=str(data.get("source", "")),
            length=int(data.get("length", 0) or 0),
        )


@dataclass(slots=True)
class RunRecord:
    """One (puzzle, system) result: the score plus everything it cost."""

    puzzle_id: str
    system: str
    score: PuzzleScore
    seconds: float
    cost_usd: float
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cache_hits: int
    rounds: int
    error: str | None = None
    candidate_coverage: float = 0.0
    entries: tuple[EntryRecord, ...] = ()
    slices: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return (self.puzzle_id, self.system)

    @property
    def solved(self) -> bool:
        value = getattr(self.score, "solved", None)
        if value is None:
            return False
        return bool(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "puzzle_id": self.puzzle_id,
            "system": self.system,
            "score": _score_to_dict(self.score),
            "seconds": float(self.seconds),
            "cost_usd": float(self.cost_usd),
            "llm_calls": int(self.llm_calls),
            "input_tokens": int(self.input_tokens),
            "output_tokens": int(self.output_tokens),
            "cache_hits": int(self.cache_hits),
            "rounds": int(self.rounds),
            "error": self.error,
            "candidate_coverage": float(self.candidate_coverage),
            "entries": [e.to_dict() for e in self.entries],
            "slices": dict(self.slices),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunRecord:
        return cls(
            puzzle_id=str(data.get("puzzle_id", "")),
            system=str(data.get("system", "")),
            score=_score_from_dict(data.get("score") or {}),
            seconds=float(data.get("seconds", 0.0) or 0.0),
            cost_usd=float(data.get("cost_usd", 0.0) or 0.0),
            llm_calls=int(data.get("llm_calls", 0) or 0),
            input_tokens=int(data.get("input_tokens", 0) or 0),
            output_tokens=int(data.get("output_tokens", 0) or 0),
            cache_hits=int(data.get("cache_hits", 0) or 0),
            rounds=int(data.get("rounds", 0) or 0),
            error=data.get("error"),
            candidate_coverage=float(data.get("candidate_coverage", 0.0) or 0.0),
            entries=tuple(
                EntryRecord.from_dict(e) for e in (data.get("entries") or [])
            ),
            slices=dict(data.get("slices") or {}),
        )


@dataclass(slots=True)
class EvalRun:
    """A whole suite x systems run, plus the provenance to reproduce it."""

    suite: str
    systems: tuple[str, ...]
    records: list[RunRecord]
    started_at: str
    finished_at: str
    git_sha: str
    config_json: str

    # -- views ------------------------------------------------------------- #

    def by_system(self) -> dict[str, list[RunRecord]]:
        out: dict[str, list[RunRecord]] = {s: [] for s in self.systems}
        for record in self.records:
            out.setdefault(record.system, []).append(record)
        for records in out.values():
            records.sort(key=lambda r: r.puzzle_id)
        return out

    @property
    def puzzle_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(r.puzzle_id for r in self.records))

    def config(self) -> dict[str, Any]:
        try:
            data = json.loads(self.config_json)
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    # -- serialisation ----------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "suite": self.suite,
            "systems": list(self.systems),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "git_sha": self.git_sha,
            "config_json": self.config_json,
            "records": [r.to_dict() for r in self.records],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> EvalRun:
        data = json.loads(text)
        return cls(
            suite=str(data.get("suite", "")),
            systems=tuple(data.get("systems") or ()),
            records=[RunRecord.from_dict(r) for r in (data.get("records") or [])],
            started_at=str(data.get("started_at", "")),
            finished_at=str(data.get("finished_at", "")),
            git_sha=str(data.get("git_sha", "")),
            config_json=str(data.get("config_json", "{}")),
        )


# --------------------------------------------------------------------------- #
# Provenance helpers
# --------------------------------------------------------------------------- #


def git_sha(cwd: Path | None = None) -> str:
    """``git rev-parse --short HEAD``, or ``"unknown"`` outside a repo."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd or xconfig.PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return "unknown"
    sha = proc.stdout.strip()
    return sha if proc.returncode == 0 and sha else "unknown"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def config_to_json(cfg: AgentConfig, *, systems: Sequence[str] = (), workers: int = 1) -> str:
    payload = {
        "base_config": _jsonable(dataclasses.asdict(cfg)),
        "systems": list(systems),
        "workers": int(workers),
    }
    return json.dumps(payload, sort_keys=True)


# --------------------------------------------------------------------------- #
# Building records
# --------------------------------------------------------------------------- #


def _empty_result(puzzle: Puzzle) -> SolveResult:
    """A result with nothing filled in -- what a crashed solve scores as."""
    return SolveResult(
        puzzle_id=puzzle.id,
        fill=Fill({}),
        cell_confidence={},
        slots={},
        stats=SolveStats(),
    )


def _slot_confidence(result: SolveResult, cells: Sequence[Cell]) -> float:
    """Weakest cell in the entry: an entry is only as sure as its worst square."""
    values = [float(result.cell_confidence.get(c, 0.0)) for c in cells]
    return min(values) if values else 0.0


def entry_records(puzzle: Puzzle, result: SolveResult) -> tuple[EntryRecord, ...]:
    """Per-entry outcomes for one solve, gold included where the puzzle has it."""
    gold: Mapping[str, str] = puzzle.solution or {}
    outcomes: Mapping[str, SlotOutcome] = result.slots or {}
    out: list[EntryRecord] = []
    for slot in puzzle.slots:
        outcome = outcomes.get(slot.id)
        predicted: str | None = None
        confidence = 0.0
        source = ""
        clue = slot.clue
        if outcome is not None:
            predicted = outcome.answer
            confidence = float(outcome.confidence)
            source = outcome.source
            clue = outcome.clue or slot.clue
        if predicted is None:
            predicted = result.fill.answer_for(slot)
            if outcome is None and predicted is not None:
                confidence = _slot_confidence(result, slot.cells)
        want = gold.get(slot.id)
        out.append(
            EntryRecord(
                slot_id=slot.id,
                clue=clue,
                predicted=predicted,
                gold=want,
                confidence=confidence,
                correct=bool(want is not None and predicted == want),
                source=source,
                length=slot.length,
            )
        )
    return tuple(out)


def _score_puzzle(puzzle: Puzzle, result: SolveResult) -> Any:
    """``metrics.score_result``, degrading to a zeroed view if it is unusable."""
    try:
        from xword.eval.metrics import score_result

        return score_result(puzzle, result)
    except Exception:
        entries = entry_records(puzzle, result)
        gradable = [e for e in entries if e.gold is not None]
        correct = sum(1 for e in gradable if e.correct)
        return _ScoreView(
            {
                "puzzle_id": puzzle.id,
                "solved": bool(gradable) and correct == len(gradable),
                "word_accuracy": (correct / len(gradable)) if gradable else 0.0,
                "cell_accuracy": 0.0,
                "scored_by": "harness-fallback",
            }
        )


def _coverage_from(stats: SolveStats) -> float:
    notes = stats.notes or {}
    for key in ("candidate_coverage", "coverage", "gold_coverage"):
        if key in notes:
            try:
                return float(notes[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def _coverage_from_solver(solver: Any, puzzle: Puzzle) -> float:
    """Candidate coverage read off the solver when the stats do not carry it.

    ``CrosswordAgent`` publishes ``last_beliefs`` precisely so the harness can
    ask "was the gold answer ever proposed?" without the solver having seen the
    solution. That number is the ceiling on what any amount of search could have
    reached, which is what separates a generation failure from a search failure,
    so it is worth picking up wherever it is offered.
    """
    beliefs = getattr(solver, "last_beliefs", None)
    if beliefs is None or not puzzle.has_solution:
        return 0.0
    try:
        return float(beliefs.coverage(dict(puzzle.solution or {})))
    except Exception:
        return 0.0


def _cost_of(stats: SolveStats, model: str) -> float:
    """Reported cost, or the published-rate estimate when the solver left it 0."""
    if stats.cost_usd:
        return float(stats.cost_usd)
    if stats.input_tokens or stats.output_tokens:
        return xconfig.estimate_cost(model, stats.input_tokens, stats.output_tokens)
    return 0.0


def make_record(
    puzzle: Puzzle,
    system: str,
    result: SolveResult | None,
    *,
    seconds: float,
    model: str,
    error: str | None = None,
    slices: Mapping[str, str] | None = None,
    coverage: float | None = None,
) -> RunRecord:
    """Assemble the record for one finished (or crashed) unit of work."""
    effective = result if result is not None else _empty_result(puzzle)
    stats = effective.stats
    if coverage is None:
        coverage = _coverage_from(stats)
    return RunRecord(
        puzzle_id=puzzle.id,
        system=system,
        score=_score_puzzle(puzzle, effective),
        seconds=float(seconds),
        cost_usd=_cost_of(stats, model),
        llm_calls=int(stats.llm_calls),
        input_tokens=int(stats.input_tokens),
        output_tokens=int(stats.output_tokens),
        cache_hits=int(stats.cache_hits),
        rounds=int(stats.rounds),
        error=error,
        candidate_coverage=float(coverage),
        entries=entry_records(puzzle, effective),
        slices=dict(slices or {}),
    )


# --------------------------------------------------------------------------- #
# JSON Lines durability
# --------------------------------------------------------------------------- #


def load_records(path: Path) -> list[RunRecord]:
    """Read a records jsonl, skipping any truncated trailing line.

    A run killed mid-write leaves a partial last line; dropping it is exactly
    what makes resume safe, so a decode error here is expected, not an error.
    """
    records: list[RunRecord] = []
    if not Path(path).exists():
        return records
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and data.get("puzzle_id"):
                records.append(RunRecord.from_dict(data))
    return records


def _append_record(path: Path, record: RunRecord, lock: threading.Lock) -> None:
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=False))
            handle.write("\n")
            handle.flush()


def _resolve_paths(out_path: Path | None) -> tuple[Path | None, Path | None]:
    """Split ``out_path`` into (records jsonl, final run json).

    A directory gets the pair inside it; a ``.jsonl`` name is the stream and
    gets a sibling ``.json`` summary; anything else is the summary and gets a
    sibling ``.jsonl`` stream.
    """
    if out_path is None:
        return None, None
    path = Path(out_path)
    if path.is_dir() or (not path.suffix and not path.exists()):
        return path / "records.jsonl", path / "run.json"
    if path.suffix == ".jsonl":
        return path, path.with_suffix(".json")
    return path.with_suffix(".jsonl"), path


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def _default_factory(config: AgentConfig) -> Any:
    """Import the real agent only when nobody supplied a solver.

    ``xword.solver.agent`` is written after (and on top of) this module, so a
    module-level import would make the harness unimportable while the package
    is being assembled -- and untestable with a stub afterwards.
    """
    from xword.solver.agent import CrosswordAgent

    return CrosswordAgent(config)


def run_suite(
    suite: Suite,
    *,
    systems: Sequence[str] = ("full",),
    base_config: AgentConfig | None = None,
    solver_factory: SolverFactory | None = None,
    workers: int = 1,
    progress: bool = True,
    timeout: float | None = None,
    out_path: Path | None = None,
    now: str = "",
    resume: bool = True,
) -> EvalRun:
    """Solve every puzzle in ``suite`` with every named system.

    ``systems`` are ablation names from ``config.ABLATIONS``; each becomes a
    config via ``config.config_for_ablation``. Results stream to
    ``<out_path>.jsonl`` as they land and the final :class:`EvalRun` is written
    to ``<out_path>.json``; with ``resume`` (the default) pairs already present
    in the jsonl are reused instead of re-solved.

    ``timeout`` is a per-unit wall-clock budget, measured from when that unit
    actually starts. A timed-out solve is recorded as an error and abandoned --
    Python cannot kill the thread, so it may keep running in the background
    until it finishes; its result is discarded.
    """
    names = tuple(systems) or ("full",)
    base = base_config or AgentConfig()
    # Fail fast on a typo'd ablation before spending an hour of API budget.
    configs = {name: config_for_ablation(name, base) for name in names}

    started_at = now or utc_now()
    jsonl_path, run_path = _resolve_paths(out_path)
    write_lock = threading.Lock()

    slice_labels = suite.all_slices() if len(suite) else {k: {} for k in SLICE_KINDS}

    def slices_for(puzzle: Puzzle) -> dict[str, str]:
        return {kind: slice_labels[kind].get(puzzle.id, "") for kind in SLICE_KINDS}

    units: list[tuple[Puzzle, str]] = [
        (puzzle, name) for puzzle in suite.puzzles for name in names
    ]

    done: dict[tuple[str, str], RunRecord] = {}
    if resume and jsonl_path is not None:
        for record in load_records(jsonl_path):
            done[record.key] = record

    results: list[RunRecord | None] = [None] * len(units)
    pending: list[int] = []
    for index, (puzzle, name) in enumerate(units):
        cached = done.get((puzzle.id, name))
        if cached is not None:
            results[index] = cached
        else:
            pending.append(index)

    factory = solver_factory or _default_factory
    solvers: dict[str, Any] = {}
    solver_lock = threading.Lock()

    def solver_for(name: str) -> Any:
        # Concurrent runs each get their own solver: nothing in the contract
        # promises an agent is thread-safe. Serial runs reuse one per system so
        # an expensive constructor (lexicon load, cache open) is paid once.
        if workers > 1:
            return factory(configs[name])
        with solver_lock:
            if name not in solvers:
                solvers[name] = factory(configs[name])
            return solvers[name]

    starts: dict[int, float] = {}

    def run_unit(index: int) -> RunRecord:
        puzzle, name = units[index]
        starts[index] = time.perf_counter()
        began = time.perf_counter()
        result: SolveResult | None = None
        error: str | None = None
        solver: Any = None
        try:
            # Constructing the solver is inside the guard too: a missing lexicon
            # or an unreadable cache is exactly the kind of failure that must
            # cost one record, not the whole run.
            solver = solver_for(name)
            result = solver.solve(puzzle)
        except Exception as exc:  # one bad puzzle must not lose the run
            error = f"{type(exc).__name__}: {exc}"
        seconds = time.perf_counter() - began
        coverage = None
        if result is not None:
            coverage = _coverage_from(result.stats) or _coverage_from_solver(
                solver, puzzle
            )
        return make_record(
            puzzle,
            name,
            result,
            seconds=seconds,
            model=configs[name].model,
            error=error,
            slices=slices_for(puzzle),
            coverage=coverage,
        )

    bar = None
    if progress and pending:
        try:
            from tqdm import tqdm  # imported only when a bar is actually wanted

            bar = tqdm(
                total=len(pending),
                desc=f"eval {suite.name}",
                unit="run",
                leave=False,
            )
        except Exception:
            bar = None

    def finish(index: int, record: RunRecord) -> None:
        results[index] = record
        if jsonl_path is not None:
            _append_record(jsonl_path, record, write_lock)
        if bar is not None:
            bar.update(1)

    try:
        if pending and (workers > 1 or timeout is not None):
            executor = ThreadPoolExecutor(max_workers=max(1, workers))
            try:
                futures: dict[int, Future] = {
                    index: executor.submit(run_unit, index) for index in pending
                }
                # Collected in submission order, so the record list is ordered
                # by (puzzle, system) no matter how completion interleaves.
                for index in pending:
                    future = futures[index]
                    record: RunRecord | None = None
                    while record is None:
                        try:
                            record = future.result(timeout=0.2)
                        except FutureTimeout:
                            began = starts.get(index)
                            if (
                                timeout is not None
                                and began is not None
                                and time.perf_counter() - began > timeout
                            ):
                                puzzle, name = units[index]
                                record = make_record(
                                    puzzle,
                                    name,
                                    None,
                                    seconds=time.perf_counter() - began,
                                    model=configs[name].model,
                                    error=f"timeout after {timeout:g}s",
                                    slices=slices_for(puzzle),
                                )
                        except Exception as exc:  # defensive: run_unit catches its own
                            puzzle, name = units[index]
                            record = make_record(
                                puzzle,
                                name,
                                None,
                                seconds=0.0,
                                model=configs[name].model,
                                error=f"{type(exc).__name__}: {exc}",
                                slices=slices_for(puzzle),
                            )
                    finish(index, record)
            finally:
                # Never block on an abandoned, timed-out solve.
                executor.shutdown(wait=False, cancel_futures=True)
        else:
            for index in pending:
                finish(index, run_unit(index))
    finally:
        if bar is not None:
            bar.close()

    run = EvalRun(
        suite=suite.name,
        systems=names,
        records=[r for r in results if r is not None],
        started_at=started_at,
        finished_at=now or utc_now(),
        git_sha=git_sha(),
        config_json=config_to_json(base, systems=names, workers=workers),
    )

    if run_path is not None:
        run_path.parent.mkdir(parents=True, exist_ok=True)
        run_path.write_text(run.to_json(), encoding="utf-8")

    return run


__all__ = [
    "EntryRecord",
    "EvalRun",
    "RunRecord",
    "SCHEMA_VERSION",
    "SolverFactory",
    "config_to_json",
    "entry_records",
    "git_sha",
    "load_records",
    "make_record",
    "run_suite",
    "utc_now",
]

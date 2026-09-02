"""What gets evaluated: named puzzle suites and how they are sampled.

A suite is an ordered, named bag of puzzles plus the keys a report groups by.
It lives apart from the harness because *which* puzzles you evaluate on is the
easiest thing to get quietly wrong: an unstratified sample of NYT dailies is
five sevenths easy days, so a headline solve rate drawn that way says more
about the sampler than about the solver. ``load_suite("nyt:70")`` therefore
balances across the seven days of the week with a seeded RNG, and every
sampling entry point takes a seed so a run can be reproduced exactly.

Nothing here touches the network, and the puzzle readers in ``xword.io`` are
imported lazily so that importing this module costs nothing.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random

from xword import config
from xword.core.types import Puzzle

# --------------------------------------------------------------------------- #
# Slicing vocabulary
# --------------------------------------------------------------------------- #

#: The slice kinds :meth:`Suite.slice_key` understands. The harness stamps all
#: of them onto every record so a saved run can be re-reported later without
#: the puzzle files being present.
SLICE_KINDS: tuple[str, ...] = ("difficulty", "size", "source", "year", "decade")

#: Difficulty is reported in publication order, not alphabetical order: for NYT
#: puzzles the day of week *is* the difficulty axis, and Monday-to-Sunday is the
#: only ordering a reader can interpret at a glance.
DAY_ORDER: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

#: Label used when the metadata simply is not there. Kept explicit, rather than
#: dropping the puzzle from a breakdown, so that missing metadata shows up as a
#: visible row instead of silently shrinking the denominator.
UNKNOWN = "unknown"

_LONG_DAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)

_DAY_ALIASES: dict[str, str] = {}
for _i, _short in enumerate(DAY_ORDER):
    _DAY_ALIASES[_short.lower()] = _short
    _DAY_ALIASES[_LONG_DAYS[_i]] = _short
    _DAY_ALIASES[_LONG_DAYS[_i][:2]] = _short

#: Fallback extension list for :func:`available_suites`, used only when the io
#: package cannot be imported. Counting by extension rather than by parsing
#: keeps the call cheap enough for a CLI list flag.
PUZZLE_SUFFIXES: frozenset[str] = frozenset({".json", ".ipuz", ".puz", ".xwj"})

_YEAR_RE = re.compile(r"(?:19|20)\d{2}")


#: Namespaces a metadata key can arrive under. The native reader keeps its three
#: metadata sources apart by prefix -- a key it does not know from the schema
#: comes back as ``meta.dow`` or ``extra.dow`` -- while the NYT reader passes
#: ``dow`` through verbatim. Looking under all three is what stops a natively
#: stored corpus from silently losing its day-of-week axis and collapsing the
#: stratified sample into one bucket.
_META_NAMESPACES: tuple[str, ...] = ("", "meta.", "extra.")


def _meta_get(meta: Mapping[str, str], *names: str) -> str:
    for name in names:
        for prefix in _META_NAMESPACES:
            value = meta.get(prefix + name)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def normalise_day(value: str) -> str | None:
    """Map any spelling of a weekday to its three-letter form, or ``None``."""
    key = str(value).strip().lower()
    return _DAY_ALIASES.get(key)


def difficulty_label(puzzle: Puzzle) -> str:
    """Day of week when known, else an explicit difficulty tag, else unknown.

    ``dow`` wins over ``difficulty`` because for the NYT the day *is* the
    difficulty scale, and mixing the two vocabularies into one breakdown makes
    the table unreadable.
    """
    meta = puzzle.meta or {}
    raw = _meta_get(meta, "dow", "day", "weekday")
    if raw:
        return normalise_day(raw) or raw
    label = _meta_get(meta, "difficulty")
    if label:
        return normalise_day(label) or label
    return UNKNOWN


def size_label(puzzle: Puzzle) -> str:
    """``WxH``, taken from the geometry rather than from metadata."""
    return f"{puzzle.width}x{puzzle.height}"


def source_label(puzzle: Puzzle) -> str:
    return _meta_get(puzzle.meta or {}, "source", "publisher", "outlet") or UNKNOWN


def year_label(puzzle: Puzzle) -> str:
    """A four-digit year from metadata, a date field, or the puzzle id."""
    meta = puzzle.meta or {}
    raw = _meta_get(meta, "year")
    if _YEAR_RE.fullmatch(raw):
        return raw
    dated = _meta_get(meta, "date", "published", "publication_date", "printDate")
    found = _YEAR_RE.search(dated)
    if found:
        return found.group(0)
    found = _YEAR_RE.search(puzzle.id)
    return found.group(0) if found else UNKNOWN


def decade_label(puzzle: Puzzle) -> str:
    year = year_label(puzzle)
    if not year.isdigit():
        return UNKNOWN
    return f"{int(year) // 10 * 10}s"


_LABELLERS: dict[str, Callable[[Puzzle], str]] = {
    "difficulty": difficulty_label,
    "size": size_label,
    "source": source_label,
    "year": year_label,
    "decade": decade_label,
}


def order_labels(labels: Iterable[str]) -> list[str]:
    """Sort slice labels for display: weekdays first, then the rest, unknown last."""
    seen = list(dict.fromkeys(labels))
    days = [d for d in DAY_ORDER if d in seen]
    rest = sorted(x for x in seen if x not in days and x != UNKNOWN)
    tail = [UNKNOWN] if UNKNOWN in seen else []
    return days + rest + tail


# --------------------------------------------------------------------------- #
# Suite
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Suite:
    """A named, ordered set of puzzles to evaluate on."""

    name: str
    puzzles: tuple[Puzzle, ...]
    description: str = ""

    def __len__(self) -> int:
        return len(self.puzzles)

    def __iter__(self):
        return iter(self.puzzles)

    @property
    def puzzle_ids(self) -> tuple[str, ...]:
        return tuple(p.id for p in self.puzzles)

    @property
    def by_id(self) -> dict[str, Puzzle]:
        return {p.id: p for p in self.puzzles}

    def slice_key(self, kind: str) -> dict[str, str]:
        """``puzzle_id -> label`` for one slicing axis.

        ``kind`` is one of :data:`SLICE_KINDS`.
        """
        labeller = _LABELLERS.get(kind)
        if labeller is None:
            known = ", ".join(SLICE_KINDS)
            raise ValueError(f"unknown slice kind {kind!r}; known: {known}")
        return {p.id: labeller(p) for p in self.puzzles}

    def all_slices(self) -> dict[str, dict[str, str]]:
        """Every axis at once -- what the harness stamps onto each record."""
        return {kind: self.slice_key(kind) for kind in SLICE_KINDS}

    def with_puzzles(self, puzzles: Sequence[Puzzle]) -> Suite:
        return Suite(
            name=self.name, puzzles=tuple(puzzles), description=self.description
        )


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def stratified_sample(
    puzzles: Sequence[Puzzle],
    key: Callable[[Puzzle], str],
    n: int,
    seed: int = 0,
) -> list[Puzzle]:
    """Draw ``n`` puzzles spread as evenly as possible across ``key``.

    Round-robins over the strata, each internally shuffled with a seeded RNG,
    so a small ``n`` still touches every stratum and a thin stratum runs out
    rather than starving the others. The result is sorted by puzzle id so the
    sample -- and therefore the run order -- is reproducible.
    """
    items = list(puzzles)
    if n <= 0:
        return []
    if n >= len(items):
        return sorted(items, key=lambda p: p.id)

    strata: dict[str, list[Puzzle]] = {}
    for puzzle in sorted(items, key=lambda p: p.id):
        strata.setdefault(key(puzzle), []).append(puzzle)

    rng = Random(seed)
    order = sorted(strata)
    pools: dict[str, list[Puzzle]] = {}
    for label in order:
        pool = list(strata[label])
        rng.shuffle(pool)
        pools[label] = pool

    picked: list[Puzzle] = []
    depth = 0
    while len(picked) < n:
        progressed = False
        for label in order:
            pool = pools[label]
            if depth < len(pool):
                picked.append(pool[depth])
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:
            break
        depth += 1

    return sorted(picked, key=lambda p: p.id)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _load_directory(path: Path) -> list[Puzzle]:
    from xword.io import loaders  # lazy: keeps import of this module I/O-free

    return list(loaders.load_directory(path))


def _load_file(path: Path) -> list[Puzzle]:
    from xword.io import loaders

    return [loaders.load_puzzle(path)]


def _known_suffixes() -> frozenset[str]:
    """What the readers will actually open, so a count matches the suite size."""
    try:
        from xword.io.loaders import PUZZLE_SUFFIXES as reader_suffixes

        return frozenset(reader_suffixes)
    except Exception:
        return PUZZLE_SUFFIXES


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    suffixes = _known_suffixes()
    return sum(
        1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in suffixes
    )


def available_suites() -> dict[str, int]:
    """Named suite -> how many puzzle files are on disk for it right now."""
    return {
        "bundled": _count_files(config.BUNDLED_PUZZLE_DIR),
        "nyt": _count_files(config.FETCHED_PUZZLE_DIR),
    }


def load_suite(name_or_path: str, *, limit: int | None = None, seed: int = 0) -> Suite:
    """Resolve a suite name or a filesystem path into a :class:`Suite`.

    ``"bundled"``
        every puzzle in ``config.BUNDLED_PUZZLE_DIR``.
    ``"nyt"`` / ``"nyt:<n>"``
        ``n`` puzzles from ``config.FETCHED_PUZZLE_DIR``, stratified across day
        of week. Without ``n`` (and without ``limit``) the whole directory is
        used, still in a deterministic order.
    anything else
        a path to one puzzle file or to a directory of them.

    ``limit`` truncates deterministically. For the NYT suite it feeds the
    stratified draw rather than chopping off the tail, because chopping would
    reintroduce exactly the day-of-week imbalance the stratification exists to
    remove.
    """
    name = str(name_or_path).strip()

    if name == "bundled":
        directory = config.BUNDLED_PUZZLE_DIR
        if not directory.exists():
            raise FileNotFoundError(f"bundled puzzle directory not found: {directory}")
        puzzles = sorted(_load_directory(directory), key=lambda p: p.id)
        if limit is not None:
            puzzles = puzzles[: max(limit, 0)]
        return Suite(
            name="bundled",
            puzzles=tuple(puzzles),
            description=f"{len(puzzles)} bundled puzzles from {directory}",
        )

    if name == "nyt" or name.startswith("nyt:"):
        # Parsed before the directory is touched, so a typo'd count reports the
        # typo rather than a missing-corpus error.
        requested: int | None = None
        if name.startswith("nyt:"):
            suffix = name.split(":", 1)[1].strip()
            if not suffix.isdigit():
                raise ValueError(f"bad suite spec {name!r}: expected 'nyt:<count>'")
            requested = int(suffix)
        directory = config.FETCHED_PUZZLE_DIR
        if not directory.exists():
            raise FileNotFoundError(
                f"NYT puzzle directory not found: {directory} "
                "(fetch puzzles first, or pass a path)"
            )
        if limit is not None:
            requested = limit if requested is None else min(requested, limit)

        found = _load_directory(directory)
        if requested is None:
            puzzles = sorted(found, key=lambda p: p.id)
            note = "all fetched puzzles"
        else:
            puzzles = stratified_sample(found, difficulty_label, requested, seed=seed)
            note = f"stratified across day of week, seed {seed}"
        return Suite(
            name=name,
            puzzles=tuple(puzzles),
            description=f"{len(puzzles)} NYT puzzles ({note})",
        )

    path = Path(name).expanduser()
    if path.is_dir():
        puzzles = sorted(_load_directory(path), key=lambda p: p.id)
    elif path.is_file():
        puzzles = _load_file(path)
    else:
        raise FileNotFoundError(
            f"no such suite or path: {name!r} "
            f"(known suites: {', '.join(sorted(available_suites()))})"
        )
    if limit is not None:
        puzzles = puzzles[: max(limit, 0)]
    return Suite(
        name=path.name or name,
        puzzles=tuple(puzzles),
        description=f"{len(puzzles)} puzzles from {path}",
    )


__all__ = [
    "DAY_ORDER",
    "PUZZLE_SUFFIXES",
    "SLICE_KINDS",
    "UNKNOWN",
    "Suite",
    "available_suites",
    "decade_label",
    "difficulty_label",
    "load_suite",
    "normalise_day",
    "order_labels",
    "size_label",
    "source_label",
    "stratified_sample",
    "year_label",
]

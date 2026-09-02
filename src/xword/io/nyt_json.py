"""Reader for the ``doshea/nyt_crosswords`` JSON archive.

The archive is a scrape, not a spec: cells are a flat row-major list with ``.``
for black squares, clue text carries its own number as a ``"12. "`` prefix, and
the ``answers`` arrays are a second, sometimes disagreeing, copy of the letters
already in ``grid``. A handful of files are simply broken (short ``grid``,
missing ``answers``). Everything here therefore either derives from ``grid`` --
the one field that is reliable -- or is reported loudly enough that the fetcher
can skip the file.

``meta["dow"]`` (day of week) is the difficulty proxy the evaluation harness
slices on, so it is always present, empty string included.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from xword.core.grid import make_puzzle, number_grid, parse_block_rows
from xword.core.types import BLOCK_CHAR, Cell, Puzzle

#: Black square sentinel used by the archive.
NYT_BLOCK_CHARS: frozenset[str] = frozenset({".", "#"})

#: Clue strings look like ``"17. Attention getter"``. The number is what binds
#: the clue to an entry; the prefix itself is noise and is stripped.
CLUE_PREFIX = re.compile(r"^\s*(\d+)\s*\.\s*(.*)$", re.DOTALL)

#: Metadata fields copied straight through, in the order they appear in meta.
PASSTHROUGH_FIELDS: tuple[str, ...] = (
    "dow",
    "date",
    "title",
    "author",
    "editor",
    "publisher",
    "copyright",
)

_DIRECTIONS: tuple[str, str] = ("across", "down")


class NytFormatError(ValueError):
    """A malformed archive entry. The fetcher treats this as "skip this file"."""

    def __init__(self, puzzle_id: str, message: str, source: str | None = None) -> None:
        self.puzzle_id = puzzle_id
        self.message = message
        self.source = source
        where = puzzle_id if source is None else f"{puzzle_id} [{source}]"
        super().__init__(f"{where}: {message}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _letters_only(text: str) -> str:
    return "".join(ch for ch in text.upper() if "A" <= ch <= "Z")


def _puzzle_id_from_date(date: str) -> tuple[str, str]:
    """``("nyt-2017-01-01", "2017")`` from ``"1/1/2017"``.

    Falls back to a sanitised form of whatever the field held, because an id
    that is merely ugly is still more useful than one that collides.
    """
    parts = [p.strip() for p in date.split("/") if p.strip()]
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        month, day, year = parts
        if len(year) == 2:
            year = ("19" if int(year) >= 50 else "20") + year
        if len(year) == 4:
            return f"nyt-{year}-{int(month):02d}-{int(day):02d}", year
    slug = re.sub(r"[^0-9A-Za-z]+", "-", date).strip("-").lower()
    return (f"nyt-{slug}" if slug else "nyt-unknown"), ""


def _grid_rows(
    payload: Mapping[str, Any], pid: str, rows: int, cols: int
) -> tuple[list[str], int]:
    """Row strings in the canonical ``#``-for-black convention, plus rebus count.

    Rebus squares (several letters in one cell) are collapsed to their first
    letter: the solver's alphabet is one letter per cell, and the collapsed
    entry still crosses correctly for every non-rebus neighbour.
    """
    flat = payload.get("grid")
    if not isinstance(flat, list):
        raise NytFormatError(pid, "'grid' is missing or not a list")
    if len(flat) != rows * cols:
        raise NytFormatError(
            pid, f"'grid' has {len(flat)} cells, expected {rows}x{cols}={rows * cols}"
        )

    rebus = 0
    chars: list[str] = []
    for i, raw in enumerate(flat):
        if raw is None:
            chars.append(BLOCK_CHAR)
            continue
        text = str(raw).strip()
        if text in NYT_BLOCK_CHARS:
            chars.append(BLOCK_CHAR)
            continue
        letters = _letters_only(text)
        if not letters:
            raise NytFormatError(
                pid,
                f"'grid' cell {i} (r{i // cols}c{i % cols}) is {raw!r}, "
                "neither a black square nor a letter",
            )
        if len(letters) > 1:
            rebus += 1
        chars.append(letters[0])
    return ["".join(chars[r * cols : (r + 1) * cols]) for r in range(rows)], rebus


def _clue_numbers(
    payload: Mapping[str, Any], pid: str, direction: str
) -> list[tuple[int, str]]:
    """``[(number, clue text), ...]`` for one direction, prefix stripped."""
    clues = payload.get("clues")
    if not isinstance(clues, Mapping):
        raise NytFormatError(pid, "'clues' is missing or not an object")
    raw = clues.get(direction)
    if not isinstance(raw, list):
        raise NytFormatError(pid, f"'clues.{direction}' is missing or not a list")

    out: list[tuple[int, str]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, str):
            raise NytFormatError(
                pid, f"clues.{direction}[{i}] is {type(item).__name__}, expected a string"
            )
        match = CLUE_PREFIX.match(item)
        if match is None:
            raise NytFormatError(
                pid,
                f"clues.{direction}[{i}] has no '<number>. ' prefix: {item!r}",
            )
        out.append((int(match.group(1)), match.group(2).strip()))
    return out


def _answer_list(
    payload: Mapping[str, Any], pid: str, direction: str, expected: int
) -> list[str]:
    answers = payload.get("answers")
    if not isinstance(answers, Mapping):
        raise NytFormatError(pid, "'answers' is missing or not an object")
    raw = answers.get(direction)
    if not isinstance(raw, list):
        raise NytFormatError(pid, f"'answers.{direction}' is missing or not a list")
    if len(raw) != expected:
        raise NytFormatError(
            pid,
            f"'answers.{direction}' has {len(raw)} entries but "
            f"'clues.{direction}' has {expected}",
        )
    return [_letters_only(str(a)) if isinstance(a, str) else "" for a in raw]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def parse_nyt_json(payload: dict[str, Any], puzzle_id: str | None = None) -> Puzzle:
    """Build a :class:`Puzzle` from one archive entry.

    Raises :class:`NytFormatError` (a ``ValueError``) for the malformed entries
    the archive contains; callers are expected to skip those files.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"nyt payload must be a JSON object, got {type(payload).__name__}"
        )

    date = str(payload.get("date") or "").strip()
    derived_id, year = _puzzle_id_from_date(date)
    pid = puzzle_id or derived_id

    size = payload.get("size")
    if not isinstance(size, Mapping):
        raise NytFormatError(pid, "'size' is missing or not an object")
    try:
        height = int(size["rows"])
        width = int(size["cols"])
    except (KeyError, TypeError, ValueError) as exc:
        raise NytFormatError(pid, f"'size' must hold integer rows/cols ({exc})") from exc
    if height < 1 or width < 1:
        raise NytFormatError(pid, f"'size' is {height}x{width}")

    rows, rebus = _grid_rows(payload, pid, height, width)
    blocks, grid_w, grid_h = parse_block_rows(rows)
    numbers, runs = number_grid(blocks, grid_w, grid_h)

    gridnum_mismatches = _check_gridnums(payload, pid, numbers, height, width)

    across_clues: dict[int, str] = {}
    down_clues: dict[int, str] = {}
    answers: dict[str, list[str]] = {}
    ordered: dict[str, list[int]] = {}
    for direction in _DIRECTIONS:
        pairs = _clue_numbers(payload, pid, direction)
        derived = sorted(n for (n, d) in runs if d == direction)
        got = [n for n, _ in pairs]
        if sorted(got) != derived:
            missing = sorted(set(derived) - set(got))
            extra = sorted(set(got) - set(derived))
            raise NytFormatError(
                pid,
                f"{direction} clue numbers do not match the grid: "
                f"{len(got)} clues for {len(derived)} entries"
                + (f"; no clue for {missing}" if missing else "")
                + (f"; no entry numbered {extra}" if extra else ""),
            )
        target = across_clues if direction == "across" else down_clues
        for number, text in pairs:
            target[number] = text
        ordered[direction] = got
        answers[direction] = _answer_list(payload, pid, direction, len(pairs))

    puzzle = make_puzzle(
        pid,
        rows,
        across_clues=across_clues,
        down_clues=down_clues,
        solution_rows=rows,
        meta={},
    )

    solution = puzzle.solution or {}
    mismatches = 0
    for direction in _DIRECTIONS:
        suffix = "A" if direction == "across" else "D"
        for number, answer in zip(ordered[direction], answers[direction], strict=False):
            if solution.get(f"{number}{suffix}") != answer:
                mismatches += 1

    meta: dict[str, str] = {
        key: str(payload.get(key) or "").strip() for key in PASSTHROUGH_FIELDS
    }
    meta["year"] = year
    meta["source"] = "nyt-json"
    meta["answer_mismatches"] = str(mismatches)
    if rebus:
        meta["rebus_cells"] = str(rebus)
    if gridnum_mismatches:
        meta["gridnum_mismatches"] = str(gridnum_mismatches)
    notepad = str(payload.get("notepad") or "").strip()
    if notepad:
        meta["notepad"] = notepad
    circles = payload.get("circles")
    if isinstance(circles, Sequence) and not isinstance(circles, (str, bytes)):
        circled = sum(1 for c in circles if c)
        if circled:
            meta["circled_cells"] = str(circled)

    # Puzzle is frozen and the mismatch count is only knowable once the answers
    # have been projected onto the entries, so metadata is attached at the end.
    return replace(puzzle, meta=meta)


def _check_gridnums(
    payload: Mapping[str, Any],
    pid: str,
    numbers: Mapping[Any, int],
    height: int,
    width: int,
) -> int:
    """Count disagreements between the file's ``gridnums`` and our numbering.

    Not fatal: clues are bound by the numbering we derive, so a stale
    ``gridnums`` is worth recording but not worth rejecting a puzzle over.
    """
    raw = payload.get("gridnums")
    if not isinstance(raw, list):
        return 0
    if len(raw) != height * width:
        raise NytFormatError(
            pid, f"'gridnums' has {len(raw)} cells, expected {height * width}"
        )
    mismatches = 0
    for i, value in enumerate(raw):
        try:
            given = int(value or 0)
        except (TypeError, ValueError):
            given = 0
        if given != numbers.get(Cell(i // width, i % width), 0):
            mismatches += 1
    return mismatches


def read_nyt_json(path: str | Path) -> Puzzle:
    """Read one archive file (e.g. ``2017/01/01.json``)."""
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{file}: cannot be read ({exc})") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{file}: not valid JSON ({exc})") from exc
    try:
        return parse_nyt_json(payload)
    except NytFormatError as exc:
        raise NytFormatError(exc.puzzle_id, exc.message, source=str(file)) from None


__all__ = [
    "CLUE_PREFIX",
    "NYT_BLOCK_CHARS",
    "NytFormatError",
    "PASSTHROUGH_FIELDS",
    "parse_nyt_json",
    "read_nyt_json",
]

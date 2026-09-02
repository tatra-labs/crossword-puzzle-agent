"""Minimal ipuz v1 crossword reader.

Only the crossword subset is supported, and only the parts that carry meaning
for solving: dimensions, the block/number layout, the solution letters, and the
clues. Presentation keys (``style``, ``showenumerations``, colours) are read
past rather than rejected, because a file that renders with a fancy border is
still a perfectly good puzzle.

ipuz is loose about how a cell may be written -- a bare integer, ``null``, the
block string, or an object wrapping either -- so cell decoding is funnelled
through one place rather than being spread across the two grids.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xword.core.grid import make_puzzle, number_grid, parse_block_rows, validate_puzzle
from xword.core.types import BLOCK_CHAR, Cell, Puzzle

#: ipuz defaults for the two sentinels a puzzle may override.
DEFAULT_BLOCK = "#"
DEFAULT_EMPTY = "0"

#: What an open square looks like in the shape rows handed to ``make_puzzle``.
OPEN_CHAR = "."

_LEADING_INT = re.compile(r"^\s*(\d+)")
_NON_SLUG = re.compile(r"[^0-9a-z]+")
_MAX_EXAMPLES = 8


def _default_id(payload: Mapping[str, Any]) -> str:
    """ipuz has no required id field, so fall back to uniqueid, then the title.

    A title-derived id is slugged because ids end up in filenames and report
    keys, where a puzzle called "Sunday, at Last!" is a nuisance.
    """
    unique = str(payload.get("uniqueid") or "").strip()
    if unique:
        return unique
    title = str(payload.get("title") or "").strip().lower()
    slug = _NON_SLUG.sub("-", title).strip("-")
    return slug or ""


class IpuzFormatError(ValueError):
    """Every problem found in one ipuz payload, reported in one message."""

    def __init__(
        self, puzzle_id: str, problems: list[str], source: str | None = None
    ) -> None:
        self.puzzle_id = puzzle_id
        self.problems = list(problems)
        self.source = source
        where = puzzle_id if source is None else f"{puzzle_id} [{source}]"
        body = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(
            f"{where}: {len(self.problems)} problem(s) in ipuz puzzle:\n{body}"
        )


# --------------------------------------------------------------------------- #
# Cell decoding
# --------------------------------------------------------------------------- #


def _unwrap(raw: object) -> object:
    """Peel the ``{"cell": n, "style": {...}}`` / ``{"value": "A"}`` wrappers."""
    seen = 0
    while isinstance(raw, Mapping) and seen < 4:
        if "cell" in raw:
            raw = raw["cell"]
        elif "value" in raw:
            raw = raw["value"]
        else:
            return None
        seen += 1
    return raw


def _decode_cell(raw: object, block: str, empty: str) -> tuple[bool, int | None]:
    """``(is_block, number)`` for one entry of the ``puzzle`` grid.

    ``null`` counts as a block: in a crossword grid the only thing an
    unspecified square can be is unusable.
    """
    raw = _unwrap(raw)
    if raw is None:
        return True, None
    if isinstance(raw, bool):
        return True, None
    if isinstance(raw, int):
        return False, (raw or None)
    text = str(raw).strip()
    if text == block or text == BLOCK_CHAR:
        return True, None
    if text == empty or text == "":
        return False, None
    if text.isdigit():
        return False, (int(text) or None)
    # Anything else (a letter, a marker like ":") is an unnumbered open square.
    return False, None


def _decode_letter(raw: object, block: str) -> str | None:
    """The solution letter for one square, or ``None`` if there is none."""
    raw = _unwrap(raw)
    if raw is None or isinstance(raw, bool):
        return None
    text = str(raw).strip().upper()
    if text in ("", block.upper(), BLOCK_CHAR, DEFAULT_EMPTY):
        return None
    letters = [ch for ch in text if "A" <= ch <= "Z"]
    # Rebus squares collapse to their first letter; see nyt_json for why.
    return letters[0] if letters else None


def _rows_of(
    raw: object, width: int, height: int, label: str, problems: list[str]
) -> list[list[object]] | None:
    """Normalise a 2-D grid to ``height`` lists of ``width`` raw cells."""
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        problems.append(f"'{label}' must be a list of rows")
        return None
    if len(raw) != height:
        problems.append(f"'{label}' has {len(raw)} rows, expected {height}")
        return None
    out: list[list[object]] = []
    for r, row in enumerate(raw):
        cells: list[object]
        if isinstance(row, str) or isinstance(row, Sequence):
            cells = list(row)
        else:
            problems.append(f"'{label}' row {r} is {type(row).__name__}, expected a list")
            return None
        if len(cells) != width:
            problems.append(
                f"'{label}' row {r} has {len(cells)} cells, expected {width}"
            )
            return None
        out.append(cells)
    return out


# --------------------------------------------------------------------------- #
# Clues
# --------------------------------------------------------------------------- #


def _clue_entries(
    clues: Mapping[str, Any], direction: str, problems: list[str]
) -> dict[int, str]:
    """``{number: text}`` for one direction, matching the key case-insensitively."""
    key = next((k for k in clues if str(k).lower().endswith(direction)), None)
    if key is None:
        problems.append(f"'clues' has no {direction.title()} list")
        return {}
    raw = clues[key]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        problems.append(f"'clues.{key}' must be a list")
        return {}

    out: dict[int, str] = {}
    for i, item in enumerate(raw):
        number: object
        text: object
        if isinstance(item, Mapping):
            number = item.get("number")
            text = item.get("clue")
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            if len(item) < 2:
                problems.append(f"clues.{key}[{i}] needs a number and a clue")
                continue
            number, text = item[0], item[1]
        else:
            problems.append(
                f"clues.{key}[{i}] is {type(item).__name__}, expected [number, text]"
            )
            continue

        match = _LEADING_INT.match(str(number))
        if match is None:
            problems.append(f"clues.{key}[{i}]: {number!r} is not an entry number")
            continue
        parsed = int(match.group(1))
        if not isinstance(text, str) or not text.strip():
            problems.append(f"clues.{key}.{parsed}: clue text is empty")
            continue
        if parsed in out:
            problems.append(f"clues.{key}: two clues for entry {parsed}")
            continue
        out[parsed] = text.strip()
    return out


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def parse_ipuz(payload: dict[str, Any], puzzle_id: str | None = None) -> Puzzle:
    """Build a :class:`Puzzle` from an ipuz v1 crossword payload."""
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"ipuz payload must be a JSON object, got {type(payload).__name__}"
        )

    problems: list[str] = []
    kind = payload.get("kind")
    kinds = [str(k) for k in kind] if isinstance(kind, list) else [str(kind or "")]
    pid = puzzle_id or _default_id(payload) or "ipuz-puzzle"
    if kind is not None and not any("crossword" in k.lower() for k in kinds):
        problems.append(f"kind {kinds} is not an ipuz crossword")

    dims = payload.get("dimensions")
    if not isinstance(dims, Mapping):
        problems.append("'dimensions' is missing or not an object")
        raise IpuzFormatError(pid, problems)
    try:
        width = int(dims["width"])
        height = int(dims["height"])
    except (KeyError, TypeError, ValueError) as exc:
        problems.append(f"'dimensions' must hold integer width/height ({exc})")
        raise IpuzFormatError(pid, problems) from exc
    if width < 1 or height < 1:
        problems.append(f"'dimensions' is {height}x{width}")
        raise IpuzFormatError(pid, problems)

    block = str(payload.get("block") or DEFAULT_BLOCK)
    raw_empty = payload.get("empty")
    empty = str(DEFAULT_EMPTY if raw_empty is None else raw_empty)

    cells = _rows_of(payload.get("puzzle"), width, height, "puzzle", problems)
    if cells is None:
        raise IpuzFormatError(pid, problems)

    rows: list[str] = []
    declared: dict[Cell, int] = {}
    for r, row in enumerate(cells):
        chars: list[str] = []
        for c, raw in enumerate(row):
            is_block, number = _decode_cell(raw, block, empty)
            chars.append(BLOCK_CHAR if is_block else OPEN_CHAR)
            if number:
                declared[Cell(r, c)] = number
        rows.append("".join(chars))

    blocks, grid_w, grid_h = parse_block_rows(rows)
    numbers, runs = number_grid(blocks, grid_w, grid_h)

    # A file that numbers nothing is fine; one that numbers wrongly is not.
    if declared:
        wrong = [
            f"r{cell.row}c{cell.col} says {given}, derived {numbers.get(cell, 0)}"
            for cell, given in sorted(declared.items())
            if numbers.get(cell, 0) != given
        ]
        if wrong:
            shown = "; ".join(wrong[:_MAX_EXAMPLES])
            extra = len(wrong) - _MAX_EXAMPLES
            more = "" if extra <= 0 else f" (+{extra} more)"
            problems.append(f"{len(wrong)} cell number(s) disagree: {shown}{more}")

    clues = payload.get("clues")
    if not isinstance(clues, Mapping):
        problems.append("'clues' is missing or not an object")
        clues = {}
    clue_maps = {
        direction: _clue_entries(clues, direction, problems)
        for direction in ("across", "down")
    }
    for direction, mapping in clue_maps.items():
        derived = {n for (n, d) in runs if d == direction}
        unknown = sorted(set(mapping) - derived)
        if unknown:
            span = f"{min(derived)}..{max(derived)}" if derived else "none"
            problems.append(
                f"clues.{direction}: entry number(s) {unknown} do not exist in this "
                f"grid ({len(derived)} {direction} entries, numbered {span})"
            )

    solution_rows = _solution_rows(payload, cells, rows, block, problems)

    meta = _meta_from_payload(payload, kinds)
    puzzle = make_puzzle(
        pid,
        rows,
        across_clues=clue_maps["across"],
        down_clues=clue_maps["down"],
        solution_rows=solution_rows,
        meta=meta,
    )
    problems.extend(validate_puzzle(puzzle))
    if problems:
        raise IpuzFormatError(pid, problems)
    return puzzle


def _solution_rows(
    payload: Mapping[str, Any],
    cells: list[list[object]],
    shape_rows: list[str],
    block: str,
    problems: list[str],
) -> list[str] | None:
    """Solution letters as row strings, or ``None`` when the file ships none."""
    raw = payload.get("solution")
    if raw is None:
        return None
    height = len(shape_rows)
    width = len(shape_rows[0])
    grid = _rows_of(raw, width, height, "solution", problems)
    if grid is None:
        return None

    out: list[str] = []
    missing: list[str] = []
    for r, row in enumerate(grid):
        chars: list[str] = []
        for c, value in enumerate(row):
            if shape_rows[r][c] == BLOCK_CHAR:
                chars.append(BLOCK_CHAR)
                continue
            letter = _decode_letter(value, block)
            if letter is None:
                missing.append(f"r{r}c{c}")
                chars.append(BLOCK_CHAR)  # placeholder; the file is rejected below
            else:
                chars.append(letter)
        out.append("".join(chars))

    if missing:
        shown = ", ".join(missing[:_MAX_EXAMPLES])
        extra = len(missing) - _MAX_EXAMPLES
        more = "" if extra <= 0 else f" (+{extra} more)"
        problems.append(
            f"'solution' leaves {len(missing)} open square(s) without a letter: "
            f"{shown}{more} (omit 'solution' entirely for an unsolved puzzle)"
        )
        return None
    return out


def _meta_from_payload(payload: Mapping[str, Any], kinds: list[str]) -> dict[str, str]:
    meta: dict[str, str] = {"source": "ipuz"}
    fields = (
        "title", "author", "editor", "publisher", "copyright",
        "date", "notes", "difficulty", "intro",
    )
    for key in fields:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            meta[key] = value.strip()
    if kinds and kinds[0]:
        meta["kind"] = kinds[0]
    return meta


def read_ipuz(path: str | Path) -> Puzzle:
    """Read an ``.ipuz`` file."""
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
        return parse_ipuz(payload, puzzle_id=_default_id(payload) or file.stem)
    except IpuzFormatError as exc:
        raise IpuzFormatError(exc.puzzle_id, exc.problems, source=str(file)) from None


__all__ = [
    "DEFAULT_BLOCK",
    "DEFAULT_EMPTY",
    "IpuzFormatError",
    "OPEN_CHAR",
    "parse_ipuz",
    "read_ipuz",
]

"""Reader and writer for the repo's native puzzle format (``.xwj`` / ``.json``).

Every other format is converted *into* this one, which is why this is the one
reader that refuses to be forgiving: a clue number that no entry claims, or a
row one character short, is a bug in the fixture rather than something to work
around at solve time. All checks therefore run to completion and the failures
are raised together -- a corrupt file should be fully diagnosable from a single
run instead of one exception per edit.

Round-tripping
--------------
``Puzzle`` carries only ``meta: Mapping[str, str]``, so the schema's top-level
scalars (title, author, ...) are stored there under their own names, the nested
``meta`` object is stored with a ``meta.`` prefix, and anything else at top
level with an ``extra.`` prefix. :func:`write_native` puts each group back where
it came from, so read -> write -> read is lossless for schema-conformant files.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xword.core.grid import (
    grid_rows,
    make_puzzle,
    number_grid,
    parse_block_rows,
    validate_puzzle,
)
from xword.core.types import BLOCK_CHAR, WILDCARD, Puzzle

#: Top-level scalar fields named by the schema, kept verbatim in ``Puzzle.meta``.
SCALAR_FIELDS: tuple[str, ...] = (
    "title",
    "author",
    "date",
    "difficulty",
    "source",
    "license",
)

#: Structural keys that are not metadata.
STRUCTURAL_FIELDS: frozenset[str] = frozenset({"id", "grid", "shape", "clues", "meta"})

#: Prefixes that keep the three metadata namespaces apart inside ``Puzzle.meta``.
META_PREFIX = "meta."
EXTRA_PREFIX = "extra."

#: What :func:`write_native` emits for an open square when there is no solution.
OPEN_CHAR = "."

#: Cap on how many same-kind complaints one message lists before summarising.
_MAX_EXAMPLES = 8


class NativeFormatError(ValueError):
    """Every problem found in one native payload, reported in one message."""

    def __init__(
        self, puzzle_id: str, problems: list[str], source: str | None = None
    ) -> None:
        self.puzzle_id = puzzle_id
        self.problems = list(problems)
        self.source = source
        where = puzzle_id if source is None else f"{puzzle_id} [{source}]"
        body = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(
            f"{where}: {len(self.problems)} problem(s) in native puzzle:\n{body}"
        )


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _as_str(value: object, where: str, problems: list[str], strict: bool) -> str:
    """Coerce a metadata value to ``str``; complain when ``strict`` and it is not."""
    if isinstance(value, str):
        return value
    if strict:
        problems.append(f"{where}: expected a string, got {type(value).__name__}")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _rows_from_payload(
    payload: Mapping[str, Any], problems: list[str]
) -> tuple[list[str] | None, bool]:
    """Extract the row strings. Returns ``(rows, rows_are_the_solution)``.

    ``rows`` is ``None`` when the grid is too broken to number, which is the one
    case that stops the remaining checks from running.
    """
    has_grid = payload.get("grid") is not None
    has_shape = payload.get("shape") is not None
    if has_grid and has_shape:
        problems.append("both 'grid' and 'shape' are present; using 'grid'")
    key = "grid" if has_grid else ("shape" if has_shape else None)
    if key is None:
        problems.append("no 'grid' (solution letters) and no 'shape'")
        return None, False

    raw = payload[key]
    if not isinstance(raw, list) or not raw:
        problems.append(f"'{key}' must be a non-empty list of row strings")
        return None, has_grid

    rows: list[str] = []
    bad_type = False
    for i, row in enumerate(raw):
        if not isinstance(row, str):
            problems.append(
                f"'{key}' row {i} is {type(row).__name__}, expected a string"
            )
            bad_type = True
            continue
        rows.append(row.upper())
    if bad_type:
        return None, has_grid

    widths = sorted({len(r) for r in rows})
    if len(widths) > 1:
        offenders = [i for i, r in enumerate(rows) if len(r) != len(rows[0])]
        problems.append(
            f"ragged '{key}': row widths {widths}; rows {offenders[:_MAX_EXAMPLES]} "
            f"differ from row 0 (width {len(rows[0])})"
        )
        return None, has_grid
    if widths[0] == 0:
        problems.append(f"'{key}' rows are empty")
        return None, has_grid

    if has_grid:
        bad = [
            f"r{r}c{c}={ch!r}"
            for r, row in enumerate(rows)
            for c, ch in enumerate(row)
            if ch != BLOCK_CHAR and not ("A" <= ch <= "Z")
        ]
        if bad:
            shown = ", ".join(bad[:_MAX_EXAMPLES])
            extra = len(bad) - _MAX_EXAMPLES
            more = "" if extra <= 0 else f" (+{extra} more)"
            problems.append(
                f"'grid' has {len(bad)} cell(s) that are neither A-Z nor "
                f"{BLOCK_CHAR!r}: {shown}{more}"
            )
    return rows, has_grid


def _clue_maps(
    payload: Mapping[str, Any], problems: list[str]
) -> dict[str, dict[int, str]]:
    """Read ``clues.across`` / ``clues.down`` into ``{number: text}`` maps."""
    clues = payload.get("clues")
    if not isinstance(clues, Mapping):
        problems.append("'clues' must be an object with 'across' and 'down'")
        clues = {}

    out: dict[str, dict[int, str]] = {"across": {}, "down": {}}
    for direction in ("across", "down"):
        raw = clues.get(direction)
        if raw is None:
            problems.append(f"'clues.{direction}' is missing")
            continue
        if not isinstance(raw, Mapping):
            problems.append(
                f"'clues.{direction}' must be an object keyed by entry number, "
                f"got {type(raw).__name__}"
            )
            continue
        for key, text in raw.items():
            try:
                number = int(str(key).strip())
            except ValueError:
                problems.append(
                    f"clues.{direction}: key {key!r} is not an entry number"
                )
                continue
            if number in out[direction]:
                problems.append(f"clues.{direction}: two keys map to entry {number}")
                continue
            if not isinstance(text, str) or not text.strip():
                problems.append(f"clues.{direction}.{number}: clue text is empty")
                continue
            out[direction][number] = text.strip()
    return out


def _meta_from_payload(
    payload: Mapping[str, Any], problems: list[str]
) -> dict[str, str]:
    meta: dict[str, str] = {}
    for key in SCALAR_FIELDS:
        value = payload.get(key)
        if value is not None:
            meta[key] = _as_str(value, f"'{key}'", problems, strict=True)

    nested = payload.get("meta")
    if nested is not None:
        if not isinstance(nested, Mapping):
            problems.append("'meta' must be an object of string values")
        else:
            for key, value in nested.items():
                meta[f"{META_PREFIX}{key}"] = _as_str(
                    value, f"meta.{key}", problems, strict=True
                )

    for key, value in payload.items():
        if key in STRUCTURAL_FIELDS or key in SCALAR_FIELDS:
            continue
        meta[f"{EXTRA_PREFIX}{key}"] = _as_str(
            value, f"'{key}'", problems, strict=False
        )
    return meta


def parse_native(payload: dict[str, Any], puzzle_id: str | None = None) -> Puzzle:
    """Build a :class:`Puzzle` from a native payload, validating hard.

    Raises :class:`NativeFormatError` (a ``ValueError``) listing *every* problem
    found rather than stopping at the first.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"native payload must be a JSON object, got {type(payload).__name__}"
        )

    problems: list[str] = []
    raw_id = puzzle_id if puzzle_id is not None else payload.get("id")
    if isinstance(raw_id, str) and raw_id.strip():
        pid = raw_id.strip()
    else:
        problems.append("missing or non-string 'id'")
        pid = "<unknown>"

    rows, is_solution = _rows_from_payload(payload, problems)
    if rows is None:
        raise NativeFormatError(pid, problems)

    clue_maps = _clue_maps(payload, problems)
    meta = _meta_from_payload(payload, problems)

    blocks, width, height = parse_block_rows(rows)
    _, runs = number_grid(blocks, width, height)
    for direction in ("across", "down"):
        derived = {n for (n, d) in runs if d == direction}
        unknown = sorted(set(clue_maps[direction]) - derived)
        if unknown:
            span = f"{min(derived)}..{max(derived)}" if derived else "none"
            problems.append(
                f"clues.{direction}: entry number(s) {unknown} do not exist in this "
                f"grid ({len(derived)} {direction} entries, numbered {span})"
            )

    puzzle = make_puzzle(
        pid,
        rows,
        across_clues=clue_maps["across"],
        down_clues=clue_maps["down"],
        solution_rows=rows if is_solution else None,
        meta=meta,
    )
    problems.extend(validate_puzzle(puzzle))
    if problems:
        raise NativeFormatError(pid, problems)
    return puzzle


def read_native(path: str | Path) -> Puzzle:
    """Read a native ``.xwj``/``.json`` file."""
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
        return parse_native(payload)
    except NativeFormatError as exc:
        raise NativeFormatError(
            exc.puzzle_id, exc.problems, source=str(file)
        ) from None


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def native_payload(puzzle: Puzzle) -> dict[str, Any]:
    """The JSON-ready dict for ``puzzle``, in canonical key order."""
    meta = dict(puzzle.meta)
    out: dict[str, Any] = {"id": puzzle.id}
    for key in SCALAR_FIELDS:
        if key in meta:
            out[key] = meta[key]

    if puzzle.solution is not None:
        rows = grid_rows(puzzle, puzzle.solution_letters(), blank=WILDCARD)
        if any(WILDCARD in row for row in rows):
            raise ValueError(
                f"{puzzle.id}: solution does not cover every open cell, "
                "so no 'grid' can be written"
            )
        out["grid"] = rows
    else:
        out["shape"] = grid_rows(puzzle, {}, blank=OPEN_CHAR)

    clues: dict[str, dict[str, str]] = {"across": {}, "down": {}}
    for slot in sorted(puzzle.slots, key=lambda s: (s.direction, s.number)):
        clues[slot.direction][str(slot.number)] = slot.clue
    out["clues"] = clues

    nested = {
        key[len(META_PREFIX) :]: value
        for key, value in meta.items()
        if key.startswith(META_PREFIX)
    }
    if nested:
        out["meta"] = nested
    for key, value in meta.items():
        if key.startswith(EXTRA_PREFIX):
            out[key[len(EXTRA_PREFIX) :]] = value
    return out


def write_native(puzzle: Puzzle, path: str | Path) -> None:
    """Write ``puzzle`` as native JSON. Round-trips through :func:`read_native`."""
    file = Path(path)
    if file.parent and not file.parent.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(native_payload(puzzle), indent=2, ensure_ascii=False)
    file.write_text(text + "\n", encoding="utf-8")


__all__ = [
    "EXTRA_PREFIX",
    "META_PREFIX",
    "NativeFormatError",
    "OPEN_CHAR",
    "SCALAR_FIELDS",
    "STRUCTURAL_FIELDS",
    "native_payload",
    "parse_native",
    "read_native",
    "write_native",
]

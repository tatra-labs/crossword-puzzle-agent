"""Grid geometry: numbering, entry extraction, and the crossing index.

This is the one place that knows how a rectangle of black and white squares
turns into the numbered across/down entries a solver reasons about. Everything
downstream consumes :class:`~xword.core.types.Slot` objects and never touches
raw grid characters again.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from xword.core.types import (
    BLOCK_CHAR,
    WILDCARD,
    Cell,
    Crossing,
    Direction,
    Puzzle,
    Slot,
)


class GridError(ValueError):
    """Raised when a grid is malformed or its clues do not line up."""


# --------------------------------------------------------------------------- #
# Numbering
# --------------------------------------------------------------------------- #


def starts_across(blocks: frozenset[Cell], width: int, height: int, cell: Cell) -> bool:
    """True if an across entry begins at ``cell``.

    An across entry starts where the square is open, there is no open square to
    its left, and there is at least one open square to its right (single-letter
    entries are not numbered in standard construction).
    """
    if cell in blocks:
        return False
    left = Cell(cell.row, cell.col - 1)
    right = Cell(cell.row, cell.col + 1)
    has_left = cell.col > 0 and left not in blocks
    has_right = cell.col + 1 < width and right not in blocks
    return not has_left and has_right


def starts_down(blocks: frozenset[Cell], width: int, height: int, cell: Cell) -> bool:
    """True if a down entry begins at ``cell``."""
    if cell in blocks:
        return False
    up = Cell(cell.row - 1, cell.col)
    down = Cell(cell.row + 1, cell.col)
    has_up = cell.row > 0 and up not in blocks
    has_down = cell.row + 1 < height and down not in blocks
    return not has_up and has_down


def number_grid(
    blocks: frozenset[Cell], width: int, height: int
) -> tuple[dict[Cell, int], dict[tuple[int, Direction], tuple[Cell, ...]]]:
    """Assign standard crossword numbers and collect each entry's cells.

    Returns ``(numbers, runs)`` where ``numbers`` maps a starting cell to its
    printed number and ``runs`` maps ``(number, direction)`` to the ordered
    cells of that entry.
    """
    numbers: dict[Cell, int] = {}
    runs: dict[tuple[int, Direction], tuple[Cell, ...]] = {}
    counter = 0

    for row in range(height):
        for col in range(width):
            cell = Cell(row, col)
            across = starts_across(blocks, width, height, cell)
            down = starts_down(blocks, width, height, cell)
            if not (across or down):
                continue
            counter += 1
            numbers[cell] = counter
            if across:
                cells = []
                c = col
                while c < width and Cell(row, c) not in blocks:
                    cells.append(Cell(row, c))
                    c += 1
                runs[(counter, "across")] = tuple(cells)
            if down:
                cells = []
                r = row
                while r < height and Cell(r, col) not in blocks:
                    cells.append(Cell(r, col))
                    r += 1
                runs[(counter, "down")] = tuple(cells)

    return numbers, runs


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


#: The canonical in-repo convention: ``#`` is a black square, anything else is
#: open. Source formats disagree (``.puz`` and the NYT JSON both use ``.`` for
#: black), so their readers pass their own sentinel explicitly rather than
#: having this function guess.
DEFAULT_BLOCK_CHARS: frozenset[str] = frozenset({BLOCK_CHAR})


def parse_block_rows(
    rows: Sequence[str], block_chars: Iterable[str] = DEFAULT_BLOCK_CHARS
) -> tuple[frozenset[Cell], int, int]:
    """Turn a list of row strings into a block set plus dimensions.

    Every character in ``block_chars`` marks a black square; all others mark an
    open square. Defaults to the repo convention of ``#``.
    """
    if not rows:
        raise GridError("grid has no rows")
    width = len(rows[0])
    if width == 0:
        raise GridError("grid rows are empty")
    if any(len(r) != width for r in rows):
        widths = sorted({len(r) for r in rows})
        raise GridError(f"ragged grid: row widths {widths}")
    sentinels = frozenset(block_chars)
    blocks = frozenset(
        Cell(r, c)
        for r, row in enumerate(rows)
        for c, ch in enumerate(row)
        if ch in sentinels
    )
    return blocks, width, len(rows)


def build_slots(
    blocks: frozenset[Cell],
    width: int,
    height: int,
    across_clues: Mapping[int, str] | None = None,
    down_clues: Mapping[int, str] | None = None,
) -> tuple[Slot, ...]:
    """Build every numbered entry, attaching clues by number when supplied."""
    across_clues = across_clues or {}
    down_clues = down_clues or {}
    _, runs = number_grid(blocks, width, height)

    slots: list[Slot] = []
    for (number, direction), cells in sorted(
        runs.items(), key=lambda kv: (kv[0][0], kv[0][1])
    ):
        clues = across_clues if direction == "across" else down_clues
        slots.append(
            Slot(
                number=number,
                direction=direction,
                cells=cells,
                clue=clues.get(number, ""),
            )
        )
    return tuple(slots)


def make_puzzle(
    puzzle_id: str,
    rows: Sequence[str],
    across_clues: Mapping[int, str] | None = None,
    down_clues: Mapping[int, str] | None = None,
    solution_rows: Sequence[str] | None = None,
    meta: Mapping[str, str] | None = None,
    block_chars: Iterable[str] = DEFAULT_BLOCK_CHARS,
) -> Puzzle:
    """Convenience constructor used by readers and tests.

    ``rows`` describes the *shape* only. ``solution_rows``, when given, must be
    the same shape and carries the answer letters.
    """
    blocks, width, height = parse_block_rows(rows, block_chars)
    slots = build_slots(blocks, width, height, across_clues, down_clues)

    solution: dict[str, str] | None = None
    if solution_rows is not None:
        sol_blocks, sol_w, sol_h = parse_block_rows(solution_rows, block_chars)
        if (sol_w, sol_h) != (width, height):
            raise GridError(
                f"solution shape {sol_h}x{sol_w} does not match grid {height}x{width}"
            )
        if sol_blocks != blocks:
            raise GridError("solution black squares do not match the grid")
        letters = {
            Cell(r, c): solution_rows[r][c].upper()
            for r in range(height)
            for c in range(width)
            if Cell(r, c) not in blocks
        }
        solution = {
            slot.id: "".join(letters[cell] for cell in slot.cells) for slot in slots
        }

    return Puzzle(
        id=puzzle_id,
        width=width,
        height=height,
        blocks=blocks,
        slots=slots,
        solution=solution,
        meta=dict(meta or {}),
    )


# --------------------------------------------------------------------------- #
# Crossing index
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class GridIndex:
    """Precomputed adjacency for a puzzle, built once per solve.

    Attributes
    ----------
    slot_by_id:
        Every entry, keyed by id.
    cell_slots:
        For each open cell, the ``(slot_id, position)`` pairs that cover it --
        one across and one down for a fully checked grid, sometimes just one in
        puzzles with unchecked squares.
    crossings:
        The across/down intersections, which are the constraints the solver
        propagates over.
    neighbours:
        For each slot id, the ids of every slot it shares a cell with.
    """

    slot_by_id: Mapping[str, Slot]
    cell_slots: Mapping[Cell, tuple[tuple[str, int], ...]]
    crossings: tuple[Crossing, ...]
    neighbours: Mapping[str, tuple[str, ...]]

    @property
    def slots(self) -> tuple[Slot, ...]:
        return tuple(self.slot_by_id.values())

    def crossings_of(self, slot_id: str) -> tuple[Crossing, ...]:
        return tuple(
            x for x in self.crossings if slot_id in (x.across_id, x.down_id)
        )

    def unchecked_cells(self) -> tuple[Cell, ...]:
        """Cells covered by only one entry -- these carry no crossing evidence."""
        return tuple(c for c, pairs in self.cell_slots.items() if len(pairs) < 2)


def index_puzzle(puzzle: Puzzle) -> GridIndex:
    """Build the :class:`GridIndex` for ``puzzle``."""
    slot_by_id = {s.id: s for s in puzzle.slots}

    cell_slots: dict[Cell, list[tuple[str, int]]] = {}
    for slot in puzzle.slots:
        for position, cell in enumerate(slot.cells):
            cell_slots.setdefault(cell, []).append((slot.id, position))

    crossings: list[Crossing] = []
    for cell, pairs in cell_slots.items():
        across = [(sid, pos) for sid, pos in pairs if slot_by_id[sid].direction == "across"]
        down = [(sid, pos) for sid, pos in pairs if slot_by_id[sid].direction == "down"]
        for a_id, a_pos in across:
            for d_id, d_pos in down:
                crossings.append(
                    Crossing(
                        cell=cell,
                        across_id=a_id,
                        across_index=a_pos,
                        down_id=d_id,
                        down_index=d_pos,
                    )
                )

    neighbours: dict[str, set[str]] = {sid: set() for sid in slot_by_id}
    for pairs in cell_slots.values():
        ids = [sid for sid, _ in pairs]
        for a in ids:
            for b in ids:
                if a != b:
                    neighbours[a].add(b)

    return GridIndex(
        slot_by_id=slot_by_id,
        cell_slots={c: tuple(v) for c, v in cell_slots.items()},
        crossings=tuple(crossings),
        neighbours={k: tuple(sorted(v)) for k, v in neighbours.items()},
    )


# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #


def pattern_from_letters(slot: Slot, letters: Mapping[Cell, str]) -> str:
    """The constraint string for ``slot`` given what is currently believed."""
    return "".join(letters.get(cell, WILDCARD) for cell in slot.cells)


def pattern_matches(pattern: str, word: str) -> bool:
    """True if ``word`` is consistent with ``pattern`` (``?`` matches anything)."""
    if len(pattern) != len(word):
        return False
    return all(p == WILDCARD or p == w for p, w in zip(pattern, word))


def grid_rows(
    puzzle: Puzzle, letters: Mapping[Cell, str], blank: str = WILDCARD
) -> list[str]:
    """Render a fill back to row strings, for printing or round-tripping."""
    out: list[str] = []
    for r in range(puzzle.height):
        row = []
        for c in range(puzzle.width):
            cell = Cell(r, c)
            if cell in puzzle.blocks:
                row.append(BLOCK_CHAR)
            else:
                row.append(letters.get(cell, blank))
        out.append("".join(row))
    return out


def validate_puzzle(puzzle: Puzzle) -> list[str]:
    """Structural sanity checks. Returns a list of human-readable problems."""
    problems: list[str] = []
    index = index_puzzle(puzzle)

    for slot in puzzle.slots:
        if slot.length < 2:
            problems.append(f"{slot.id}: entry shorter than 2 cells")
        if not slot.clue:
            problems.append(f"{slot.id}: missing clue")

    open_cells = set(puzzle.open_cells)
    covered = set(index.cell_slots)
    if open_cells != covered:
        missing = sorted(open_cells - covered)
        if missing:
            problems.append(f"{len(missing)} open cells belong to no entry, e.g. {missing[0]}")

    if puzzle.solution is not None:
        by_id = puzzle.slot_by_id
        for slot_id, answer in puzzle.solution.items():
            slot = by_id.get(slot_id)
            if slot is None:
                problems.append(f"solution references unknown entry {slot_id}")
            elif len(answer) != slot.length:
                problems.append(
                    f"{slot_id}: solution {answer!r} has length {len(answer)}, "
                    f"expected {slot.length}"
                )
        letters = puzzle.solution_letters()
        for crossing in index.crossings:
            a = letters.get(crossing.cell)
            if a is None:
                problems.append(f"solution leaves {crossing.cell} empty")

    return problems


__all__ = [
    "GridError",
    "GridIndex",
    "DEFAULT_BLOCK_CHARS",
    "build_slots",
    "grid_rows",
    "index_puzzle",
    "make_puzzle",
    "number_grid",
    "parse_block_rows",
    "pattern_from_letters",
    "pattern_matches",
    "starts_across",
    "starts_down",
    "validate_puzzle",
]

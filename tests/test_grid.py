"""Tests for :mod:`xword.core.grid`.

The numbering rules are the part of the repo most likely to be silently wrong:
an off-by-one in ``starts_across`` still produces *a* set of entries, just not
the ones the clue list is keyed to. So the geometry tests assert exact slot ids
and exact cell tuples rather than counts.
"""

from __future__ import annotations

import dataclasses

import pytest
from conftest import (
    ASYMMETRIC_SLOT_IDS,
    ASYMMETRIC_SOLUTION_ROWS,
    ASYMMETRIC_UNCHECKED,
    MINI_ROWS,
    MINI_SLOT_IDS,
    MINI_SOLUTION,
    MINI_SOLUTION_ROWS,
    OPEN_SLOT_IDS,
    OPEN_SOLUTION_ROWS,
)

from xword.core.grid import (
    DEFAULT_BLOCK_CHARS,
    GridError,
    build_slots,
    grid_rows,
    index_puzzle,
    make_puzzle,
    number_grid,
    parse_block_rows,
    pattern_from_letters,
    pattern_matches,
    starts_across,
    starts_down,
    validate_puzzle,
)
from xword.core.types import BLOCK_CHAR, WILDCARD, Cell, Puzzle

# --------------------------------------------------------------------------- #
# Numbering
# --------------------------------------------------------------------------- #


def test_mini_numbering_matches_standard_convention(mini_puzzle: Puzzle) -> None:
    assert tuple(s.id for s in mini_puzzle.slots) == MINI_SLOT_IDS


def test_asymmetric_numbering_matches_standard_convention(
    asymmetric_puzzle: Puzzle,
) -> None:
    assert tuple(s.id for s in asymmetric_puzzle.slots) == ASYMMETRIC_SLOT_IDS


def test_open_grid_numbering(open_puzzle: Puzzle) -> None:
    """With no black squares only the top row and the left column start
    entries, so numbering is 1-4 across the top then 5-7 down the side."""
    assert tuple(s.id for s in open_puzzle.slots) == OPEN_SLOT_IDS
    assert all(s.length == 4 for s in open_puzzle.slots)


@pytest.mark.parametrize(
    "slot_id, expected_cells",
    [
        ("1A", (Cell(0, 2), Cell(0, 3), Cell(0, 4))),
        ("1D", (Cell(0, 2), Cell(1, 2), Cell(2, 2), Cell(3, 2), Cell(4, 2))),
        ("2D", (Cell(0, 3), Cell(1, 3), Cell(2, 3), Cell(3, 3))),
        ("3D", (Cell(0, 4), Cell(1, 4), Cell(2, 4))),
        ("4A", (Cell(1, 1), Cell(1, 2), Cell(1, 3), Cell(1, 4))),
        ("5A", (Cell(2, 0), Cell(2, 1), Cell(2, 2), Cell(2, 3), Cell(2, 4))),
        ("5D", (Cell(2, 0), Cell(3, 0), Cell(4, 0))),
        ("7A", (Cell(4, 0), Cell(4, 1), Cell(4, 2))),
    ],
)
def test_mini_slot_cells(
    mini_puzzle: Puzzle, slot_id: str, expected_cells: tuple[Cell, ...]
) -> None:
    assert mini_puzzle.slot_by_id[slot_id].cells == expected_cells


def test_numbers_are_assigned_in_reading_order(mini_puzzle: Puzzle) -> None:
    numbers, _ = number_grid(mini_puzzle.blocks, mini_puzzle.width, mini_puzzle.height)
    starts = sorted(numbers.items(), key=lambda kv: kv[1])
    assert [cell for cell, _ in starts] == sorted(numbers)


def test_a_number_is_shared_by_its_across_and_down_entry(mini_puzzle: Puzzle) -> None:
    by_id = mini_puzzle.slot_by_id
    assert by_id["1A"].start == by_id["1D"].start == Cell(0, 2)


# --------------------------------------------------------------------------- #
# Degenerate shapes
# --------------------------------------------------------------------------- #


def test_grid_with_no_black_squares_has_only_border_starts() -> None:
    blocks, width, height = parse_block_rows(["....", "....", "....", "...."])
    assert blocks == frozenset()
    slots = build_slots(blocks, width, height)
    assert tuple(s.id for s in slots) == OPEN_SLOT_IDS


def test_single_row_grid_has_one_across_and_no_downs() -> None:
    """A 1xN grid can hold no down entry: every down run would be one cell."""
    puzzle = make_puzzle("row", ["....."], across_clues={1: "The whole thing"})
    assert tuple(s.id for s in puzzle.slots) == ("1A",)
    assert puzzle.slots[0].length == 5
    assert validate_puzzle(puzzle) == []


def test_single_column_grid_has_one_down_and_no_acrosses() -> None:
    puzzle = make_puzzle("col", [".", ".", ".", "."], down_clues={1: "Straight down"})
    assert tuple(s.id for s in puzzle.slots) == ("1D",)
    assert puzzle.slots[0].cells == (Cell(0, 0), Cell(1, 0), Cell(2, 0), Cell(3, 0))
    assert validate_puzzle(puzzle) == []


def test_single_cell_grid_produces_no_entries() -> None:
    blocks, width, height = parse_block_rows(["."])
    assert build_slots(blocks, width, height) == ()


@pytest.mark.parametrize(
    "rows, expected_ids",
    [
        # (0,0) heads a 3-cell across run but only a 1-cell down run, and a
        # 1-cell run is not an entry -- so there is no 1D here.
        (["..."], ("1A",)),
        (["...", "#.#", "..."], ("1A", "2D", "3A")),
        # The lone open square at (0,1) is boxed in on both axes.
        (["#.#", "###", "#.#"], ()),
    ],
)
def test_length_one_runs_are_not_numbered(
    rows: list[str], expected_ids: tuple[str, ...]
) -> None:
    blocks, width, height = parse_block_rows(rows)
    slots = build_slots(blocks, width, height)
    assert tuple(s.id for s in slots) == expected_ids
    assert all(s.length >= 2 for s in slots)


@pytest.mark.parametrize(
    "cell, across, down",
    [
        (Cell(0, 0), True, False),  # down run would be length 1
        (Cell(0, 1), False, True),  # across run would be length 1
        (Cell(1, 1), False, False),  # continues 2D, starts nothing
        (Cell(2, 0), True, False),
        (Cell(1, 0), False, False),  # a black square starts nothing
    ],
)
def test_starts_predicates_on_a_pinwheel(cell: Cell, across: bool, down: bool) -> None:
    blocks, width, height = parse_block_rows(["...", "#.#", "..."])
    assert starts_across(blocks, width, height, cell) is across
    assert starts_down(blocks, width, height, cell) is down


# --------------------------------------------------------------------------- #
# Crossing index
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fixture_name", ["mini_puzzle", "asymmetric_puzzle", "open_puzzle"]
)
def test_crossing_count_equals_doubly_covered_cells(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    """Recomputed from the slots rather than trusted from the index, so a bug
    in ``index_puzzle`` cannot validate itself."""
    puzzle: Puzzle = request.getfixturevalue(fixture_name)
    across_cells = {
        c for s in puzzle.slots if s.direction == "across" for c in s.cells
    }
    down_cells = {c for s in puzzle.slots if s.direction == "down" for c in s.cells}
    index = index_puzzle(puzzle)
    assert len(index.crossings) == len(across_cells & down_cells)


def test_crossings_carry_the_right_positions(mini_puzzle: Puzzle) -> None:
    index = index_puzzle(mini_puzzle)
    by_cell = {x.cell: x for x in index.crossings}
    crossing = by_cell[Cell(2, 2)]
    assert (crossing.across_id, crossing.across_index) == ("5A", 2)
    assert (crossing.down_id, crossing.down_index) == ("1D", 2)


def test_unchecked_cells_are_reported(asymmetric_puzzle: Puzzle) -> None:
    index = index_puzzle(asymmetric_puzzle)
    assert tuple(sorted(index.unchecked_cells())) == ASYMMETRIC_UNCHECKED
    for cell in ASYMMETRIC_UNCHECKED:
        assert len(index.cell_slots[cell]) == 1


@pytest.mark.parametrize("fixture_name", ["mini_puzzle", "open_puzzle"])
def test_fully_checked_grids_report_no_unchecked_cells(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    puzzle: Puzzle = request.getfixturevalue(fixture_name)
    assert index_puzzle(puzzle).unchecked_cells() == ()


def test_crossings_of_selects_both_directions(mini_puzzle: Puzzle) -> None:
    index = index_puzzle(mini_puzzle)
    ids = {x.down_id for x in index.crossings_of("5A")}
    assert ids == {"1D", "2D", "3D", "4D", "5D"}
    assert all(x.across_id == "5A" for x in index.crossings_of("5A"))


def test_neighbours_are_symmetric(asymmetric_puzzle: Puzzle) -> None:
    index = index_puzzle(asymmetric_puzzle)
    for slot_id, others in index.neighbours.items():
        for other in others:
            assert slot_id in index.neighbours[other]
    # An entry made entirely of unchecked squares would have no neighbours;
    # this grid has none, so every entry crosses something.
    assert all(index.neighbours.values())


def test_index_covers_every_open_cell(asymmetric_puzzle: Puzzle) -> None:
    index = index_puzzle(asymmetric_puzzle)
    assert set(index.cell_slots) == set(asymmetric_puzzle.open_cells)
    assert index.slots == asymmetric_puzzle.slots


# --------------------------------------------------------------------------- #
# parse_block_rows
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "rows, message",
    [
        ([], "no rows"),
        ((), "no rows"),
        ([""], "empty"),
        (["", ""], "empty"),
        (["...", ".."], "ragged"),
        (["..", "...", ".."], "ragged"),
    ],
)
def test_parse_block_rows_rejects_malformed_grids(rows, message: str) -> None:
    with pytest.raises(GridError, match=message):
        parse_block_rows(rows)


def test_parse_block_rows_uses_the_repo_convention() -> None:
    blocks, width, height = parse_block_rows(["#.", ".#"])
    assert (width, height) == (2, 2)
    assert blocks == frozenset({Cell(0, 0), Cell(1, 1)})
    assert frozenset({BLOCK_CHAR}) == DEFAULT_BLOCK_CHARS


def test_parse_block_rows_honours_a_foreign_sentinel() -> None:
    """``.puz`` and the NYT JSON use ``.`` for black, so readers pass their own
    sentinel instead of the module guessing."""
    blocks, _, _ = parse_block_rows([".AA", "AAA"], block_chars=".")
    assert blocks == frozenset({Cell(0, 0)})


# --------------------------------------------------------------------------- #
# make_puzzle
# --------------------------------------------------------------------------- #


def test_make_puzzle_derives_the_solution_per_entry(mini_puzzle: Puzzle) -> None:
    assert dict(mini_puzzle.solution or {}) == MINI_SOLUTION


def test_make_puzzle_upper_cases_the_solution() -> None:
    puzzle = make_puzzle(
        "lower",
        ["..", ".."],
        across_clues={1: "a", 3: "b"},
        down_clues={1: "c", 2: "d"},
        solution_rows=["at", "to"],
    )
    assert (puzzle.solution or {})["1A"] == "AT"
    assert (puzzle.solution or {})["1D"] == "AT"


@pytest.mark.parametrize(
    "solution_rows, message",
    [
        (["##AD", "#SCA", "SHOR", "PORT", "YEN#"], "does not match grid"),
        (["##ADO", "#SCAN", "SHORE", "PORT#"], "does not match grid"),
        (MINI_SOLUTION_ROWS + ("XXXXX",), "does not match grid"),
        # Right shape, black squares in the wrong places.
        (["#XADO", "#SCAN", "SHORE", "PORT#", "YEN##"], "black squares"),
        (["##ADO", "#SCAN", "SHORE", "PORTX", "YEN##"], "black squares"),
    ],
)
def test_make_puzzle_rejects_a_disagreeing_solution(
    solution_rows: list[str], message: str
) -> None:
    with pytest.raises(GridError, match=message):
        make_puzzle("bad", MINI_ROWS, solution_rows=solution_rows)


def test_make_puzzle_without_a_solution_leaves_it_none() -> None:
    puzzle = make_puzzle("shape-only", MINI_ROWS)
    assert puzzle.solution is None
    assert puzzle.has_solution is False


def test_make_puzzle_attaches_clues_by_number(mini_puzzle: Puzzle) -> None:
    by_id = mini_puzzle.slot_by_id
    assert by_id["1A"].clue == "Fuss"
    assert by_id["1D"].clue == "An oak's beginning"
    assert by_id["1A"].number == by_id["1D"].number == 1


# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "pattern, word, expected",
    [
        ("???", "CAT", True),
        ("CAT", "CAT", True),
        ("C?T", "CAT", True),
        ("C?T", "CUT", True),
        ("C?T", "CAB", False),
        ("CAT", "COT", False),
        ("??", "CAT", False),  # length mismatch beats every wildcard
        ("????", "CAT", False),
        ("", "", True),
        ("", "A", False),
        ("?", "?", True),  # a literal '?' in the word is matched by a wildcard
        ("A", "?", False),
    ],
)
def test_pattern_matches(pattern: str, word: str, expected: bool) -> None:
    assert pattern_matches(pattern, word) is expected


def test_pattern_from_letters_marks_unknown_cells(mini_puzzle: Puzzle) -> None:
    slot = mini_puzzle.slot_by_id["5A"]
    letters = {Cell(2, 0): "S", Cell(2, 4): "E"}
    assert pattern_from_letters(slot, letters) == "S???E"


def test_pattern_from_letters_on_an_empty_fill(mini_puzzle: Puzzle) -> None:
    slot = mini_puzzle.slot_by_id["1D"]
    assert pattern_from_letters(slot, {}) == WILDCARD * slot.length


# --------------------------------------------------------------------------- #
# grid_rows
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fixture_name, rows",
    [
        ("mini_puzzle", MINI_SOLUTION_ROWS),
        ("asymmetric_puzzle", ASYMMETRIC_SOLUTION_ROWS),
        ("open_puzzle", OPEN_SOLUTION_ROWS),
    ],
)
def test_grid_rows_round_trips_the_solution(
    request: pytest.FixtureRequest, fixture_name: str, rows: tuple[str, ...]
) -> None:
    puzzle: Puzzle = request.getfixturevalue(fixture_name)
    assert grid_rows(puzzle, puzzle.solution_letters()) == list(rows)


def test_grid_rows_renders_unknown_cells_as_blank(mini_puzzle: Puzzle) -> None:
    rendered = grid_rows(mini_puzzle, {Cell(0, 2): "A"})
    assert rendered[0] == "##A??"
    assert rendered[4] == "???##"


def test_grid_rows_accepts_a_custom_blank(mini_puzzle: Puzzle) -> None:
    assert grid_rows(mini_puzzle, {}, blank=".")[2] == "....."


# --------------------------------------------------------------------------- #
# validate_puzzle
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fixture_name", ["mini_puzzle", "asymmetric_puzzle", "open_puzzle"]
)
def test_fixtures_validate_clean(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    assert validate_puzzle(request.getfixturevalue(fixture_name)) == []


def test_validate_flags_a_missing_clue(mini_puzzle: Puzzle) -> None:
    slots = tuple(
        dataclasses.replace(s, clue="") if s.id == "4D" else s
        for s in mini_puzzle.slots
    )
    problems = validate_puzzle(dataclasses.replace(mini_puzzle, slots=slots))
    assert problems == ["4D: missing clue"]


def test_validate_flags_a_wrong_length_solution(mini_puzzle: Puzzle) -> None:
    solution = dict(mini_puzzle.solution or {})
    solution["5A"] = "SHORT"[:4]
    problems = validate_puzzle(dataclasses.replace(mini_puzzle, solution=solution))
    assert any(p == "5A: solution 'SHOR' has length 4, expected 5" for p in problems)


def test_validate_flags_an_unknown_slot_id(mini_puzzle: Puzzle) -> None:
    solution = dict(mini_puzzle.solution or {})
    solution["99D"] = "GHOST"
    problems = validate_puzzle(dataclasses.replace(mini_puzzle, solution=solution))
    assert problems == ["solution references unknown entry 99D"]


def test_validate_flags_an_entry_shorter_than_two_cells(mini_puzzle: Puzzle) -> None:
    stub = dataclasses.replace(
        mini_puzzle.slot_by_id["1A"], cells=(Cell(0, 2),), clue="Solo"
    )
    slots = tuple(stub if s.id == "1A" else s for s in mini_puzzle.slots)
    problems = validate_puzzle(dataclasses.replace(mini_puzzle, slots=slots))
    assert "1A: entry shorter than 2 cells" in problems


def test_validate_flags_open_cells_that_belong_to_no_entry() -> None:
    """A boxed-in square is legal geometry but unanswerable, so the harness
    must hear about it before a solver silently ignores it."""
    puzzle = make_puzzle("boxed", ["#.#", "###", "#.#"])
    problems = validate_puzzle(puzzle)
    assert any("belong to no entry" in p for p in problems)


def test_validate_ignores_a_solution_free_puzzle() -> None:
    puzzle = make_puzzle(
        "shape-only", ["..", ".."], across_clues={1: "a", 3: "b"},
        down_clues={1: "c", 2: "d"},
    )
    assert validate_puzzle(puzzle) == []

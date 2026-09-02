"""Tests for :mod:`xword.core.types`.

These types are frozen dataclasses that everything else builds on, so the
tests here concentrate on the two properties downstream code silently relies
on: identifiers are formatted one way and one way only, and nothing that
looks like an update mutates the object it was called on.
"""

from __future__ import annotations

import dataclasses

import pytest
from conftest import MINI_ROWS, MINI_SOLUTION

from xword.core.grid import make_puzzle
from xword.core.types import (
    ALPHABET,
    BLOCK_CHAR,
    DIRECTIONS,
    LETTER_INDEX,
    WILDCARD,
    AgentEvent,
    Candidate,
    Cell,
    ClueRequest,
    Fill,
    Puzzle,
    Slot,
    SlotOutcome,
    SolveStats,
    null_sink,
)

# --------------------------------------------------------------------------- #
# Cell
# --------------------------------------------------------------------------- #


def test_cell_is_hashable_and_ordered() -> None:
    cells = [Cell(1, 0), Cell(0, 2), Cell(0, 1)]
    assert sorted(cells) == [Cell(0, 1), Cell(0, 2), Cell(1, 0)]
    assert len({Cell(0, 0), Cell(0, 0)}) == 1


def test_cell_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Cell(0, 0).row = 3  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Slot
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "number, direction, expected",
    [
        (1, "across", "1A"),
        (1, "down", "1D"),
        (17, "across", "17A"),
        (17, "down", "17D"),
        (120, "across", "120A"),
    ],
)
def test_slot_id_formatting(number: int, direction: str, expected: str) -> None:
    slot = Slot(number=number, direction=direction, cells=(Cell(0, 0), Cell(0, 1)))
    assert slot.id == expected


def test_slot_ids_are_unique_per_direction() -> None:
    cells = (Cell(0, 0), Cell(0, 1))
    ids = {Slot(3, d, cells).id for d in DIRECTIONS}
    assert ids == {"3A", "3D"}


def test_slot_length_and_start() -> None:
    cells = (Cell(2, 1), Cell(3, 1), Cell(4, 1))
    slot = Slot(number=4, direction="down", cells=cells, clue="Downward")
    assert slot.length == 3
    assert slot.start == Cell(2, 1)


@pytest.mark.parametrize(
    "cell, expected",
    [(Cell(2, 1), 0), (Cell(3, 1), 1), (Cell(4, 1), 2)],
)
def test_slot_index_of(cell: Cell, expected: int) -> None:
    slot = Slot(4, "down", (Cell(2, 1), Cell(3, 1), Cell(4, 1)))
    assert slot.index_of(cell) == expected


@pytest.mark.parametrize("foreign", [Cell(0, 0), Cell(2, 2), Cell(-1, 1)])
def test_slot_index_of_rejects_a_foreign_cell(foreign: Cell) -> None:
    slot = Slot(4, "down", (Cell(2, 1), Cell(3, 1), Cell(4, 1)))
    with pytest.raises(ValueError):
        slot.index_of(foreign)


def test_slot_is_frozen() -> None:
    slot = Slot(1, "across", (Cell(0, 0), Cell(0, 1)))
    with pytest.raises(dataclasses.FrozenInstanceError):
        slot.clue = "new clue"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Puzzle
# --------------------------------------------------------------------------- #


def test_solution_letters_projects_every_open_cell(mini_puzzle: Puzzle) -> None:
    letters = mini_puzzle.solution_letters()
    assert set(letters) == set(mini_puzzle.open_cells)
    assert letters[Cell(2, 2)] == "O"
    assert "".join(letters[c] for c in mini_puzzle.slot_by_id["1D"].cells) == "ACORN"


def test_solution_letters_agree_at_every_crossing(mini_puzzle: Puzzle) -> None:
    """Two entries sharing a cell must claim the same letter -- if they did
    not, ``solution_letters`` would quietly return whichever came last."""
    letters = mini_puzzle.solution_letters()
    for slot in mini_puzzle.slots:
        answer = (mini_puzzle.solution or {})[slot.id]
        assert "".join(letters[c] for c in slot.cells) == answer


def test_solution_letters_raises_without_a_solution() -> None:
    puzzle = make_puzzle("no-answers", MINI_ROWS)
    assert puzzle.has_solution is False
    with pytest.raises(ValueError, match="no-answers"):
        puzzle.solution_letters()


def test_solution_letters_skips_unknown_slot_ids(mini_puzzle: Puzzle) -> None:
    """A reader may hand over answers for entries this grid does not have;
    that is a validation problem, not a crash."""
    solution = dict(MINI_SOLUTION) | {"42D": "PHANTOM"}
    puzzle = dataclasses.replace(mini_puzzle, solution=solution)
    assert set(puzzle.solution_letters()) == set(puzzle.open_cells)


@pytest.mark.parametrize(
    "cell, expected_ids",
    [
        (Cell(2, 2), ("1D", "5A")),  # a checked square: one across, one down
        (Cell(0, 2), ("1A", "1D")),
        (Cell(4, 2), ("1D", "7A")),
        (Cell(0, 0), ()),  # a black square belongs to nothing
        (Cell(9, 9), ()),  # off the grid entirely
    ],
)
def test_slots_at(
    mini_puzzle: Puzzle, cell: Cell, expected_ids: tuple[str, ...]
) -> None:
    assert tuple(s.id for s in mini_puzzle.slots_at(cell)) == expected_ids


def test_open_cells_excludes_blocks(mini_puzzle: Puzzle) -> None:
    open_cells = set(mini_puzzle.open_cells)
    assert open_cells.isdisjoint(mini_puzzle.blocks)
    assert len(open_cells) + len(mini_puzzle.blocks) == 5 * 5
    assert mini_puzzle.is_block(Cell(0, 0)) is True
    assert mini_puzzle.is_block(Cell(2, 2)) is False


def test_slot_by_id_covers_every_entry(mini_puzzle: Puzzle) -> None:
    by_id = mini_puzzle.slot_by_id
    assert len(by_id) == len(mini_puzzle.slots)
    assert by_id["4D"].direction == "down"


# --------------------------------------------------------------------------- #
# Fill
# --------------------------------------------------------------------------- #


def test_pattern_for_marks_unknown_cells(mini_puzzle: Puzzle) -> None:
    slot = mini_puzzle.slot_by_id["5A"]
    fill = Fill({Cell(2, 0): "S", Cell(2, 2): "O"})
    assert fill.pattern_for(slot) == "S?O??"


def test_pattern_for_an_empty_fill(mini_puzzle: Puzzle) -> None:
    slot = mini_puzzle.slot_by_id["1D"]
    assert Fill().pattern_for(slot) == WILDCARD * 5


@pytest.mark.parametrize(
    "slot_id, expected",
    [("1A", "ADO"), ("1D", "ACORN"), ("5A", "SHORE"), ("7A", "YEN")],
)
def test_answer_for_a_complete_fill(
    mini_puzzle: Puzzle, perfect_fill: Fill, slot_id: str, expected: str
) -> None:
    assert perfect_fill.answer_for(mini_puzzle.slot_by_id[slot_id]) == expected


def test_answer_for_returns_none_while_incomplete(mini_puzzle: Puzzle) -> None:
    slot = mini_puzzle.slot_by_id["1D"]
    partial = Fill({Cell(0, 2): "A", Cell(1, 2): "C"})
    assert partial.answer_for(slot) is None
    assert partial.pattern_for(slot) == "AC???"


def test_with_slot_leaves_the_original_untouched(mini_puzzle: Puzzle) -> None:
    slot = mini_puzzle.slot_by_id["1A"]
    original = Fill({Cell(0, 2): "X"})
    before = dict(original.letters)

    updated = original.with_slot(slot, "ADO")

    assert updated is not original
    assert dict(original.letters) == before
    assert original.get(Cell(0, 3)) is None
    assert updated.answer_for(slot) == "ADO"


def test_with_slot_overwrites_conflicting_letters(mini_puzzle: Puzzle) -> None:
    slot = mini_puzzle.slot_by_id["1A"]
    fill = Fill({Cell(0, 2): "Z", Cell(0, 3): "Z", Cell(0, 4): "Z"})
    assert fill.with_slot(slot, "ADO").pattern_for(slot) == "ADO"


def test_with_slot_chains_without_mutation(mini_puzzle: Puzzle) -> None:
    by_id = mini_puzzle.slot_by_id
    empty = Fill()
    one = empty.with_slot(by_id["1A"], "ADO")
    two = one.with_slot(by_id["7A"], "YEN")

    assert empty.letters == {}
    assert set(one.letters) == set(by_id["1A"].cells)
    assert set(two.letters) == set(by_id["1A"].cells) | set(by_id["7A"].cells)


def test_is_complete_for(
    mini_puzzle: Puzzle, perfect_fill: Fill, three_wrong_fill: Fill
) -> None:
    assert perfect_fill.is_complete_for(mini_puzzle) is True
    # Wrong letters are still letters: completeness is not correctness.
    assert three_wrong_fill.is_complete_for(mini_puzzle) is True
    assert Fill().is_complete_for(mini_puzzle) is False


def test_three_wrong_fill_differs_in_exactly_three_cells(
    gold_letters: dict[Cell, str],
    three_wrong_fill: Fill,
    three_wrong_cells: tuple[Cell, ...],
) -> None:
    wrong = {
        cell
        for cell, letter in gold_letters.items()
        if three_wrong_fill.get(cell) != letter
    }
    assert wrong == set(three_wrong_cells)


# --------------------------------------------------------------------------- #
# Candidates and requests
# --------------------------------------------------------------------------- #


def test_clue_request_with_pattern_is_a_copy() -> None:
    request = ClueRequest(
        slot_id="1A", clue="Fuss", length=3, direction="across", pattern=None
    )
    narrowed = request.with_pattern("A??")

    assert request.pattern is None
    assert narrowed.pattern == "A??"
    assert narrowed.slot_id == "1A" and narrowed.length == 3
    assert narrowed is not request


def test_clue_request_defaults() -> None:
    request = ClueRequest(slot_id="2D", clue="Dart", length=4, direction="down")
    assert request.pattern is None
    assert request.crossing_clues == ()
    assert dict(request.puzzle_meta) == {}


def test_candidate_defaults_and_equality() -> None:
    first = Candidate(answer="ADO", score=-0.5, source="llm")
    second = Candidate(answer="ADO", score=-0.5, source="llm")
    assert first == second
    assert first.rationale == ""


# --------------------------------------------------------------------------- #
# Reporting types
# --------------------------------------------------------------------------- #


def test_slot_outcome_allows_an_unanswered_entry() -> None:
    outcome = SlotOutcome(
        slot_id="1A", clue="Fuss", answer=None, confidence=0.0, source="none"
    )
    assert outcome.answer is None
    assert outcome.considered == 0


def test_solve_stats_defaults_are_independent() -> None:
    """``notes`` is a mutable default; two SolveStats must not share one."""
    first, second = SolveStats(), SolveStats()
    first.notes["coverage"] = 1.0
    assert second.notes == {}
    assert second.rounds == 0 and second.cost_usd == 0.0


def test_agent_event_is_frozen_and_defaults_empty() -> None:
    event = AgentEvent(kind="propose", round=1, message="asked for candidates")
    assert dict(event.data) == {}
    assert null_sink(event) is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.round = 2  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Module constants
# --------------------------------------------------------------------------- #


def test_alphabet_and_index_agree() -> None:
    assert len(ALPHABET) == 26
    assert ALPHABET[0] == "A" and ALPHABET[-1] == "Z"
    assert all(ALPHABET[i] == ch for ch, i in LETTER_INDEX.items())


def test_sentinels_are_distinct_from_letters() -> None:
    assert WILDCARD not in ALPHABET
    assert BLOCK_CHAR not in ALPHABET
    assert WILDCARD != BLOCK_CHAR

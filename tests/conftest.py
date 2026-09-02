"""Shared fixtures for the whole test suite.

The three puzzle fixtures are deliberately different in shape. ``mini_puzzle``
is an ordinary 5x5 in which every open square is checked by both an across and
a down entry. ``asymmetric_puzzle`` is a non-square grid with unchecked squares
and two-letter entries. ``open_puzzle`` has no black squares at all. Code that
quietly assumes "every cell has exactly one across crossing and one down
crossing" passes on the first two and fails on the second -- which is the whole
point of keeping the second around.

Every puzzle fixture runs :func:`validate_puzzle` on itself before handing the
puzzle over, so a typo in a grid literal here shows up as a fixture error
naming the offending entry rather than as a baffling failure deep inside
whichever test happened to use it.

Grids are written with the repo convention: ``#`` is a black square.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pytest

from xword.core.beliefs import MIN_NULL_MASS, SlotBeliefs
from xword.core.grid import make_puzzle, validate_puzzle
from xword.core.types import ALPHABET, LETTER_INDEX, Candidate, Cell, Fill, Puzzle

# --------------------------------------------------------------------------- #
# Puzzle definitions
#
# Kept as module constants, not buried inside the fixtures, so that a test can
# assert against the literal source rows (round-tripping a fill back to text,
# for instance) without re-deriving them.
# --------------------------------------------------------------------------- #

#: A 5x5 with blocked corners. Numbering runs 1..7 and every open square is
#: crossed, which makes it the default fixture for anything that does not care
#: about degenerate geometry.
MINI_ROWS: tuple[str, ...] = (
    "##...",
    "#....",
    ".....",
    "....#",
    "...##",
)
MINI_SOLUTION_ROWS: tuple[str, ...] = (
    "##ADO",
    "#SCAN",
    "SHORE",
    "PORT#",
    "YEN##",
)
MINI_ACROSS_CLUES: dict[int, str] = {
    1: "Fuss",
    4: "Read quickly",
    5: "Where the surf meets the sand",
    6: "Sailor's destination",
    7: "Tokyo currency",
}
MINI_DOWN_CLUES: dict[int, str] = {
    1: "An oak's beginning",
    2: "Pub game projectile",
    3: "The loneliest number, in song",
    4: "It has a tongue and a sole",
    5: "Undercover operative",
}
#: Slot ids in construction order, i.e. sorted by (number, direction).
MINI_SLOT_IDS: tuple[str, ...] = (
    "1A", "1D", "2D", "3D", "4A", "4D", "5A", "5D", "6A", "7A",
)
MINI_SOLUTION: dict[str, str] = {
    "1A": "ADO",
    "1D": "ACORN",
    "2D": "DART",
    "3D": "ONE",
    "4A": "SCAN",
    "4D": "SHOE",
    "5A": "SHORE",
    "5D": "SPY",
    "6A": "PORT",
    "7A": "YEN",
}

#: 4 rows by 5 columns, no symmetry, with a middle row whose open squares are
#: each covered by a down entry only. Those unchecked squares are exactly the
#: case that breaks solvers written for a fully checked grid.
ASYMMETRIC_ROWS: tuple[str, ...] = (
    ".....",
    ".#.#.",
    ".....",
    "..#..",
)
ASYMMETRIC_SOLUTION_ROWS: tuple[str, ...] = (
    "CHESS",
    "H#E#H",
    "IGLOO",
    "NO#HE",
)
ASYMMETRIC_ACROSS_CLUES: dict[int, str] = {
    1: "Game with rooks and pawns",
    4: "Dome-shaped snow shelter",
    7: "Flat refusal",
    8: "Pronoun for a gentleman",
}
ASYMMETRIC_DOWN_CLUES: dict[int, str] = {
    1: "Feature below the mouth",
    2: "Slippery swimmer",
    3: "Loafer or sneaker",
    5: "Green light word",
    6: '"Now I get it!"',
}
ASYMMETRIC_SLOT_IDS: tuple[str, ...] = (
    "1A", "1D", "2D", "3D", "4A", "5D", "6D", "7A", "8A",
)
#: The squares covered by exactly one entry.
ASYMMETRIC_UNCHECKED: tuple[Cell, ...] = (
    Cell(0, 1),
    Cell(0, 3),
    Cell(1, 0),
    Cell(1, 2),
    Cell(1, 4),
)

#: A 4x4 double word square: every row and every column is an entry, so the
#: crossing graph is complete and there is nowhere for a propagation bug to
#: hide behind a black square.
OPEN_ROWS: tuple[str, ...] = ("....",) * 4
OPEN_SOLUTION_ROWS: tuple[str, ...] = ("BALL", "AREA", "LEAD", "LADY")
OPEN_ACROSS_CLUES: dict[int, str] = {
    1: "Sphere at a bowling alley",
    5: "Region",
    6: "Be out in front",
    7: "Woman",
}
OPEN_DOWN_CLUES: dict[int, str] = {
    1: "Formal dance",
    2: "Square footage, e.g.",
    3: "Pencil filler",
    4: "Titled Englishwoman",
}
OPEN_SLOT_IDS: tuple[str, ...] = ("1A", "1D", "2D", "3D", "4D", "5A", "6A", "7A")

#: Cells that ``three_wrong_fill`` corrupts. Fixed rather than sampled so that
#: an accuracy assertion elsewhere can name the exact squares it expects to be
#: scored wrong.
THREE_WRONG_CELLS: tuple[Cell, ...] = (Cell(0, 2), Cell(2, 0), Cell(4, 1))


def _shift(letter: str) -> str:
    """Next letter of the alphabet, wrapping. Used only to produce a wrong
    letter that is guaranteed to differ from the gold one."""
    return ALPHABET[(LETTER_INDEX[letter] + 1) % len(ALPHABET)]


def _checked(puzzle: Puzzle) -> Puzzle:
    problems = validate_puzzle(puzzle)
    if problems:
        raise AssertionError(f"fixture puzzle {puzzle.id!r} is invalid: {problems}")
    return puzzle


# --------------------------------------------------------------------------- #
# Puzzle fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def mini_puzzle() -> Puzzle:
    """A valid, fully checked 5x5 with real clues and a known solution."""
    return _checked(
        make_puzzle(
            "mini-5x5",
            MINI_ROWS,
            across_clues=MINI_ACROSS_CLUES,
            down_clues=MINI_DOWN_CLUES,
            solution_rows=MINI_SOLUTION_ROWS,
            meta={"title": "Mini", "author": "test suite"},
        )
    )


@pytest.fixture
def asymmetric_puzzle() -> Puzzle:
    """A 4x5 grid with unchecked squares and two-letter entries."""
    return _checked(
        make_puzzle(
            "asym-4x5",
            ASYMMETRIC_ROWS,
            across_clues=ASYMMETRIC_ACROSS_CLUES,
            down_clues=ASYMMETRIC_DOWN_CLUES,
            solution_rows=ASYMMETRIC_SOLUTION_ROWS,
            meta={"title": "Asymmetric"},
        )
    )


@pytest.fixture
def open_puzzle() -> Puzzle:
    """A 4x4 with no black squares at all."""
    return _checked(
        make_puzzle(
            "open-4x4",
            OPEN_ROWS,
            across_clues=OPEN_ACROSS_CLUES,
            down_clues=OPEN_DOWN_CLUES,
            solution_rows=OPEN_SOLUTION_ROWS,
            meta={"title": "Word square"},
        )
    )


# --------------------------------------------------------------------------- #
# Derived fills
# --------------------------------------------------------------------------- #


@pytest.fixture
def gold_letters(mini_puzzle: Puzzle) -> dict[Cell, str]:
    """``mini_puzzle``'s reference answers, projected to per-cell letters."""
    return mini_puzzle.solution_letters()


@pytest.fixture
def perfect_fill(gold_letters: dict[Cell, str]) -> Fill:
    """A complete, entirely correct fill of ``mini_puzzle``."""
    return Fill(dict(gold_letters))


@pytest.fixture
def three_wrong_cells() -> tuple[Cell, ...]:
    """The cells ``three_wrong_fill`` gets wrong."""
    return THREE_WRONG_CELLS


@pytest.fixture
def three_wrong_fill(gold_letters: dict[Cell, str]) -> Fill:
    """A complete fill of ``mini_puzzle`` with exactly three wrong letters."""
    letters = dict(gold_letters)
    for cell in THREE_WRONG_CELLS:
        letters[cell] = _shift(letters[cell])
    return Fill(letters)


# --------------------------------------------------------------------------- #
# Beliefs
# --------------------------------------------------------------------------- #

BeliefSpec = Mapping[str, Sequence[tuple[str, float]]]
BeliefFactory = Callable[..., SlotBeliefs]


@pytest.fixture
def make_beliefs() -> BeliefFactory:
    """Factory building a :class:`SlotBeliefs` from ``{slot_id: [(answer, p)]}``.

    Probabilities are passed through ``set_slot`` rather than written into the
    dataclass fields directly, so a test's numbers get the same normalisation
    the real pipeline applies and cannot accidentally describe a state the
    production code could never produce.
    """

    def _make(
        spec: BeliefSpec,
        *,
        null_mass: float = MIN_NULL_MASS,
        source: str = "test",
        lengths: Mapping[str, int] | None = None,
    ) -> SlotBeliefs:
        beliefs = SlotBeliefs()
        for slot_id, pairs in spec.items():
            candidates = [
                # score is an unnormalised log-score; the floor keeps a zero
                # probability from producing -inf and poisoning later sums.
                Candidate(
                    answer=answer,
                    score=math.log(max(float(prob), 1e-12)),
                    source=source,
                )
                for answer, prob in pairs
            ]
            probabilities = np.array([float(p) for _, p in pairs], dtype=np.float64)
            if lengths is not None and slot_id in lengths:
                length = lengths[slot_id]
            elif candidates:
                length = len(candidates[0].answer)
            else:
                length = 0
            beliefs.set_slot(slot_id, candidates, probabilities, null_mass, length)
        return beliefs

    return _make

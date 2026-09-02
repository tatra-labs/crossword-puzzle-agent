"""Frozen data contracts for the crossword agent.

Every module in this package codes against the types defined here. They are
intentionally small, immutable where practical, and free of I/O so that the
solver, the candidate sources, and the evaluation harness can be tested in
isolation.

Coordinate convention
---------------------
``(row, col)`` with ``row=0`` at the top and ``col=0`` at the left, matching the
way puzzles are printed. A grid is a rectangle of ``height`` rows by ``width``
columns. Blocked ("black") squares are stored as an explicit set rather than as
a sentinel character so that a partially filled grid never has to distinguish
"blocked" from "not yet known".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

# --------------------------------------------------------------------------- #
# Alphabet
# --------------------------------------------------------------------------- #

#: The 26 letters a cell may hold. Rebus squares (multiple letters in one cell)
#: are normalised away at parse time -- see ``xword.io``.
ALPHABET: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LETTER_INDEX: dict[str, int] = {ch: i for i, ch in enumerate(ALPHABET)}

#: Character used in a *pattern* to mean "this cell is not yet known".
WILDCARD = "?"

#: Character used when rendering a blocked square.
BLOCK_CHAR = "#"

Direction = Literal["across", "down"]
DIRECTIONS: tuple[Direction, Direction] = ("across", "down")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True, order=True)
class Cell:
    """A single square of the grid, identified by its position."""

    row: int
    col: int

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"r{self.row}c{self.col}"


@dataclass(frozen=True, slots=True)
class Slot:
    """One numbered entry: the run of cells a single clue answers.

    ``cells`` is ordered from the start of the entry (left-to-right for across
    entries, top-to-bottom for down entries), so ``cells[i]`` holds the *i*-th
    letter of the answer.
    """

    number: int
    direction: Direction
    cells: tuple[Cell, ...]
    clue: str = ""

    @property
    def id(self) -> str:
        """Human-facing identifier, e.g. ``17A`` or ``3D``."""
        suffix = "A" if self.direction == "across" else "D"
        return f"{self.number}{suffix}"

    @property
    def length(self) -> int:
        return len(self.cells)

    @property
    def start(self) -> Cell:
        return self.cells[0]

    def index_of(self, cell: Cell) -> int:
        """Position of ``cell`` within this entry.

        Raises ``ValueError`` if the cell is not part of the slot.
        """
        return self.cells.index(cell)


@dataclass(frozen=True, slots=True)
class Crossing:
    """The single shared cell between an across entry and a down entry."""

    cell: Cell
    across_id: str
    across_index: int
    down_id: str
    down_index: int


# --------------------------------------------------------------------------- #
# Puzzle
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Puzzle:
    """A crossword: geometry, clues, and (optionally) the official answers.

    ``solution`` maps slot id -> answer and is present only for puzzles used in
    evaluation. The solver never reads it; the harness does.
    """

    id: str
    width: int
    height: int
    blocks: frozenset[Cell]
    slots: tuple[Slot, ...]
    solution: Mapping[str, str] | None = None
    meta: Mapping[str, str] = field(default_factory=dict)

    # -- lookups ----------------------------------------------------------- #

    @property
    def slot_by_id(self) -> dict[str, Slot]:
        return {s.id: s for s in self.slots}

    @property
    def open_cells(self) -> tuple[Cell, ...]:
        return tuple(
            Cell(r, c)
            for r in range(self.height)
            for c in range(self.width)
            if Cell(r, c) not in self.blocks
        )

    def is_block(self, cell: Cell) -> bool:
        return cell in self.blocks

    def slots_at(self, cell: Cell) -> tuple[Slot, ...]:
        return tuple(s for s in self.slots if cell in s.cells)

    @property
    def has_solution(self) -> bool:
        return self.solution is not None

    def solution_letters(self) -> dict[Cell, str]:
        """Gold answers projected down to per-cell letters."""
        if self.solution is None:
            raise ValueError(f"puzzle {self.id!r} has no reference solution")
        out: dict[Cell, str] = {}
        by_id = self.slot_by_id
        for slot_id, answer in self.solution.items():
            slot = by_id.get(slot_id)
            if slot is None:
                continue
            for cell, ch in zip(slot.cells, answer):
                out[cell] = ch
        return out


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Candidate:
    """One proposed answer for one slot.

    ``score`` is an unnormalised log-score: larger is better, and only
    differences within a single slot's candidate list are meaningful. Fusion
    turns a set of these into a proper distribution.
    """

    answer: str
    score: float
    source: str
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class ClueRequest:
    """What a candidate source is asked to answer.

    ``pattern`` is a length-``length`` string over ``A-Z`` and ``WILDCARD``
    describing letters already believed known, or ``None`` on the first pass.
    """

    slot_id: str
    clue: str
    length: int
    direction: Direction
    pattern: str | None = None
    puzzle_meta: Mapping[str, str] = field(default_factory=dict)
    crossing_clues: tuple[str, ...] = ()

    def with_pattern(self, pattern: str | None) -> ClueRequest:
        return replace(self, pattern=pattern)


class CandidateSource(Protocol):
    """Anything that can propose answers for clues.

    Implementations must be safe to call with an empty request list and must
    return an entry for every requested ``slot_id`` (possibly an empty list).
    """

    name: str

    def propose(self, requests: Sequence[ClueRequest]) -> dict[str, list[Candidate]]:
        ...


# --------------------------------------------------------------------------- #
# Solutions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Fill:
    """A (possibly partial) assignment of letters to open cells."""

    letters: Mapping[Cell, str] = field(default_factory=dict)

    def get(self, cell: Cell) -> str | None:
        return self.letters.get(cell)

    def pattern_for(self, slot: Slot) -> str:
        return "".join(self.letters.get(c, WILDCARD) for c in slot.cells)

    def answer_for(self, slot: Slot) -> str | None:
        text = self.pattern_for(slot)
        return None if WILDCARD in text else text

    def is_complete_for(self, puzzle: Puzzle) -> bool:
        return all(c in self.letters for c in puzzle.open_cells)

    def with_slot(self, slot: Slot, answer: str) -> Fill:
        merged = dict(self.letters)
        for cell, ch in zip(slot.cells, answer):
            merged[cell] = ch
        return Fill(merged)


@dataclass(frozen=True, slots=True)
class SlotOutcome:
    """Per-entry record of what the agent decided and how sure it was."""

    slot_id: str
    clue: str
    answer: str | None
    confidence: float
    source: str
    considered: int = 0


@dataclass(slots=True)
class SolveStats:
    """Bookkeeping that the evaluation harness reports on."""

    rounds: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hits: int = 0
    wall_seconds: float = 0.0
    cost_usd: float = 0.0
    notes: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class SolveResult:
    """Everything the agent produces for one puzzle."""

    puzzle_id: str
    fill: Fill
    cell_confidence: Mapping[Cell, float]
    slots: Mapping[str, SlotOutcome]
    stats: SolveStats
    trace: list[AgentEvent] = field(default_factory=list)

    def letters(self) -> Mapping[Cell, str]:
        return self.fill.letters


# --------------------------------------------------------------------------- #
# Observability
# --------------------------------------------------------------------------- #

EventKind = Literal[
    "ingest",
    "propose",
    "fuse",
    "propagate",
    "commit",
    "critique",
    "repair",
    "verify",
    "done",
]


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """A single step of the agent loop, emitted for live display and for the
    written trace that accompanies every solve."""

    kind: EventKind
    round: int
    message: str
    data: Mapping[str, float | int | str] = field(default_factory=dict)


class EventSink(Protocol):
    def __call__(self, event: AgentEvent) -> None:
        ...


def null_sink(event: AgentEvent) -> None:  # pragma: no cover - trivial
    """Default sink: discard."""


__all__ = [
    "ALPHABET",
    "LETTER_INDEX",
    "WILDCARD",
    "BLOCK_CHAR",
    "Direction",
    "DIRECTIONS",
    "Cell",
    "Slot",
    "Crossing",
    "Puzzle",
    "Candidate",
    "ClueRequest",
    "CandidateSource",
    "Fill",
    "SlotOutcome",
    "SolveStats",
    "SolveResult",
    "AgentEvent",
    "EventKind",
    "EventSink",
    "null_sink",
]

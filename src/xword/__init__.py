"""crossword-puzzle-agent: an AI agent that solves crossword puzzles.

The short version of how it works::

    from xword import CrosswordAgent, load_puzzle

    puzzle = load_puzzle("data/puzzles/bundled/mini-01.json")
    result = CrosswordAgent().solve(puzzle)
    print(result.fill.answer_for(puzzle.slots[0]))

Submodules are imported lazily so that ``import xword`` stays cheap and so that a
missing optional dependency (or a missing API key) only bites when the relevant
feature is actually used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

if TYPE_CHECKING:  # pragma: no cover
    from xword.config import AgentConfig
    from xword.core.types import Candidate, Cell, Fill, Puzzle, Slot, SolveResult
    from xword.io.loaders import load_puzzle
    from xword.solver.agent import CrosswordAgent, solve_puzzle

_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentConfig": ("xword.config", "AgentConfig"),
    "Candidate": ("xword.core.types", "Candidate"),
    "Cell": ("xword.core.types", "Cell"),
    "CrosswordAgent": ("xword.solver.agent", "CrosswordAgent"),
    "Fill": ("xword.core.types", "Fill"),
    "Puzzle": ("xword.core.types", "Puzzle"),
    "Slot": ("xword.core.types", "Slot"),
    "SolveResult": ("xword.core.types", "SolveResult"),
    "load_puzzle": ("xword.io.loaders", "load_puzzle"),
    "solve_puzzle": ("xword.solver.agent", "solve_puzzle"),
}

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'xword' has no attribute {name!r}") from None
    from importlib import import_module

    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return list(__all__)

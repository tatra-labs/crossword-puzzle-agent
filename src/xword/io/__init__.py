"""Reading and writing puzzles, and showing them to humans.

Re-exports the loader entry points so ``from xword.io import load_puzzle``
works. The format-specific readers stay in their own modules and are imported
lazily by :mod:`xword.io.loaders`, so a malformed optional dependency in one
reader cannot break the others.
"""

from __future__ import annotations

from xword.io.loaders import (
    LOAD_FAILURES,
    detect_format,
    load_directory,
    load_payload,
    load_puzzle,
)

__all__ = [
    "LOAD_FAILURES",
    "detect_format",
    "load_directory",
    "load_payload",
    "load_puzzle",
]

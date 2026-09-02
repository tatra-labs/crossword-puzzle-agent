"""Format detection and the one entry point everything else loads puzzles through.

Callers (the CLI, the fetcher, the evaluation harness) should never have to know
which reader a file needs, so this module sniffs -- by extension first, then by
content, because the archive ships NYT, native and ipuz puzzles all under
``.json``.

Two deliberate choices:

* ``xword.io.puz`` is imported *inside* the dispatch, so a broken or missing
  binary reader cannot stop JSON puzzles from loading.
* :func:`load_directory` never raises for a bad file. A corpus of a thousand
  puzzles with three broken ones should still produce 997 puzzles plus a list of
  what went wrong, which is what :data:`LOAD_FAILURES` is for.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from xword.core.types import Puzzle

LOG = logging.getLogger(__name__)

#: ``(path, error)`` for every file the last :func:`load_directory` call could
#: not read. Cleared -- in place, so ``from ... import LOAD_FAILURES`` keeps
#: working -- at the start of each call.
LOAD_FAILURES: list[tuple[str, str]] = []

Format = Literal["puz", "native", "nyt-json", "ipuz"]

#: Extension -> format, for the cases where the extension is authoritative.
EXTENSION_FORMATS: dict[str, Format] = {
    ".puz": "puz",
    ".ipuz": "ipuz",
    ".xwj": "native",
}

#: Extensions worth trying when scanning a directory with the default pattern.
PUZZLE_SUFFIXES: frozenset[str] = frozenset({".puz", ".ipuz", ".xwj", ".json"})

#: Magic string at offset 2 of every ``.puz`` file.
PUZ_MAGIC = b"ACROSS&DOWN"


def _sniff_payload(payload: object) -> Format:
    """Classify an already-parsed JSON document."""
    if not isinstance(payload, dict):
        raise ValueError("JSON puzzle must be an object at the top level")

    ipuz_shaped = "dimensions" in payload or "puzzle" in payload
    if ("kind" in payload or "version" in payload) and ipuz_shaped:
        return "ipuz"
    if "gridnums" in payload or "dow" in payload:
        return "nyt-json"

    clues = payload.get("clues")
    if isinstance(clues, dict) and clues:
        values = list(clues.values())
        if any(isinstance(v, dict) for v in values):
            return "native"
        if any(isinstance(v, list) for v in values):
            # Both remaining formats key clues by direction; what differs is the
            # element. NYT stores the whole clue as one "12. text" string, ipuz
            # stores [number, text] pairs or {"number": .., "clue": ..} objects.
            items = [item for v in values if isinstance(v, list) for item in v]
            if any(isinstance(item, str) for item in items):
                return "nyt-json"
            return "ipuz"
    if ipuz_shaped:
        return "ipuz"
    if "grid" in payload or "shape" in payload:
        return "native"
    keys = ", ".join(sorted(map(str, payload))[:10]) or "<none>"
    raise ValueError(f"unrecognised JSON puzzle; top-level keys: {keys}")


def detect_format(path: str | Path) -> Format:
    """One of ``"puz"``, ``"native"``, ``"nyt-json"``, ``"ipuz"``.

    Raises ``ValueError`` when the file matches nothing.
    """
    file = Path(path)
    suffix = file.suffix.lower()
    if suffix in EXTENSION_FORMATS:
        return EXTENSION_FORMATS[suffix]

    try:
        with file.open("rb") as handle:
            head = handle.read(4096)
    except OSError as exc:
        raise ValueError(f"{file}: cannot be read ({exc})") from exc
    if PUZ_MAGIC in head[:32]:
        return "puz"

    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{file}: not text and not a .puz file ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{file}: not valid JSON ({exc})") from exc
    try:
        return _sniff_payload(payload)
    except ValueError as exc:
        raise ValueError(f"{file}: {exc}") from None


def load_puzzle(path: str | Path) -> Puzzle:
    """Read any supported puzzle file."""
    file = Path(path)
    fmt = detect_format(file)

    if fmt == "native":
        from xword.io.native import read_native

        return read_native(file)
    if fmt == "nyt-json":
        from xword.io.nyt_json import read_nyt_json

        return read_nyt_json(file)
    if fmt == "ipuz":
        from xword.io.ipuz import read_ipuz

        return read_ipuz(file)
    if fmt == "puz":
        # Imported late and on its own so that an unfinished or broken binary
        # reader only breaks .puz files.
        try:
            from xword.io.puz import read_puz
        except ImportError as exc:
            raise ValueError(f"{file}: .puz support is unavailable ({exc})") from exc
        return read_puz(file)

    raise ValueError(f"{file}: unsupported format {fmt!r}")


def load_directory(path: str | Path, pattern: str = "*") -> list[Puzzle]:
    """Load every puzzle under ``path``, sorted by puzzle id.

    Files that fail to load are skipped, logged at warning level, and recorded
    in :data:`LOAD_FAILURES`, which is reset on every call.

    With the default ``"*"`` pattern only :data:`PUZZLE_SUFFIXES` are tried, so
    a corpus directory can hold a README without every scan reporting it as a
    failure. Pass an explicit pattern to load files with other extensions.

    The scan is **recursive**. Fetched corpora are stored in nested
    ``<year>/<month>/<day>.json`` directories, so a flat scan silently returns
    nothing for them -- which is worse than an error, because the suite loader
    counts files recursively and would report hundreds of puzzles available
    while loading zero.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")

    LOAD_FAILURES.clear()
    loaded: list[tuple[str, str, Puzzle]] = []
    for file in sorted(directory.rglob(pattern)):
        if not file.is_file():
            continue
        if pattern == "*" and file.suffix.lower() not in PUZZLE_SUFFIXES:
            continue
        try:
            puzzle = load_puzzle(file)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the scan
            reason = f"{type(exc).__name__}: {exc}"
            LOAD_FAILURES.append((str(file), reason))
            LOG.warning("skipping %s: %s", file, reason)
            continue
        loaded.append((puzzle.id, str(file), puzzle))

    loaded.sort(key=lambda item: (item[0], item[1]))
    return [puzzle for _, _, puzzle in loaded]


def load_payload(payload: dict[str, Any], puzzle_id: str | None = None) -> Puzzle:
    """Parse an in-memory JSON document, sniffing its format the same way.

    Used by the fetcher, which already has the response body in hand and should
    not have to round-trip it through a file to find out what it is.
    """
    fmt = _sniff_payload(payload)
    if fmt == "native":
        from xword.io.native import parse_native

        return parse_native(payload, puzzle_id)
    if fmt == "nyt-json":
        from xword.io.nyt_json import parse_nyt_json

        return parse_nyt_json(payload, puzzle_id)
    from xword.io.ipuz import parse_ipuz

    return parse_ipuz(payload, puzzle_id)


__all__ = [
    "EXTENSION_FORMATS",
    "LOAD_FAILURES",
    "PUZZLE_SUFFIXES",
    "PUZ_MAGIC",
    "detect_format",
    "load_directory",
    "load_payload",
    "load_puzzle",
]

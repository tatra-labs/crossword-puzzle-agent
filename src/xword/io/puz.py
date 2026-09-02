"""Reader and writer for the Across Lite ``.puz`` binary format.

``.puz`` is what publishers actually ship and what every desktop solver reads,
so it is the widest ingest path this project has. A file is a fixed 52-byte
header, two ``width * height`` character grids (the solution, then the player's
saved state), a run of NUL-terminated strings (title, author, copyright, one
clue per entry, notes), and then optional extension sections.

Header layout, offsets relative to the start of the header::

    0x00  u16   global checksum
    0x02  12    magic b"ACROSS&DOWN\\x00"
    0x0E  u16   CIB (header) checksum
    0x10  8     masked low/high checksums, XORed with b"ICHEATED"
    0x18  4     version string, e.g. b"1.3\\x00\\x00"
    0x1C  u16   reserved
    0x1E  u16   scrambled checksum
    0x20  12    reserved
    0x2C  u8    width
    0x2D  u8    height
    0x2E  u16   number of clues
    0x30  u16   puzzle type
    0x32  u16   scrambled tag

Checksums are deliberately *not* enforced on read. Files in the wild are
routinely re-saved by third-party tools that get the text checksum subtly
wrong, and rejecting them would cost far more puzzles than it would save. Call
:func:`verify_checksums` when the integrity of a particular file matters.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from xword.core.grid import make_puzzle, number_grid, parse_block_rows
from xword.core.types import BLOCK_CHAR, WILDCARD, Cell, Direction, Puzzle


class PuzError(ValueError):
    """Raised when a ``.puz`` payload cannot be understood."""


# --------------------------------------------------------------------------- #
# Format constants
# --------------------------------------------------------------------------- #

MAGIC = b"ACROSS&DOWN\x00"
HEADER_SIZE = 0x34

#: Offsets inside the header.
_OFF_GLOBAL_CKSUM = 0x00
_OFF_MAGIC = 0x02
_OFF_CIB_CKSUM = 0x0E
_OFF_MASKED = 0x10
_OFF_VERSION = 0x18
_OFF_SCRAMBLED_CKSUM = 0x1E
_OFF_CIB = 0x2C  # width, height, n_clues, puzzle_type, scrambled_tag
_CIB_SIZE = 8

#: The eight masked checksum bytes at 0x10 are XORed with this.
_MASK = b"ICHEATED"

#: ``.`` is the standard black square; ``:`` shows up in diagramless puzzles,
#: where Across Lite uses it for a square the solver is not shown.
_BLOCK_GRID_CHARS = frozenset(".:")

#: Value of the 0x32 tag when Across Lite has locked the solution.
SCRAMBLED_TAG = 0x0004

PUZZLE_TYPE_NORMAL = 0x0001

#: GEXT flag bits. Only "circled" carries meaning for the solver.
GEXT_CIRCLED = 0x80

DEFAULT_WRITE_VERSION = "1.3"


# --------------------------------------------------------------------------- #
# Checksums
# --------------------------------------------------------------------------- #


def cksum_region(data: bytes, cksum: int = 0) -> int:
    """The Across Lite 16-bit rotate-and-add checksum over ``data``.

    Seeded with ``cksum`` so regions can be chained in the order the format
    requires.
    """
    for byte in data:
        cksum = (((cksum >> 1) | ((cksum & 1) << 15)) + byte) & 0xFFFF
    return cksum


@dataclass(frozen=True, slots=True)
class PuzChecksums:
    """The five checksums a ``.puz`` file carries, unmasked."""

    cib: int
    solution: int
    state: int
    text: int
    global_: int


def _version_tuple(version: str) -> tuple[int, int]:
    match = re.match(r"(\d+)\.(\d+)", version)
    if not match:
        return (1, 2)
    return (int(match.group(1)), int(match.group(2)))


def _text_cksum(
    title: bytes,
    author: bytes,
    copyright_: bytes,
    clues: Sequence[bytes],
    notes: bytes,
    version: str,
    cksum: int = 0,
) -> int:
    """Checksum over the string block.

    The rules are unintuitive and have to be reproduced exactly: title, author
    and copyright are included *with* their NUL and only when non-empty; clues
    are included *without* their NUL; notes only count from format version 1.3
    onwards.
    """
    for field in (title, author, copyright_):
        if field:
            cksum = cksum_region(field + b"\x00", cksum)
    for clue in clues:
        if clue:
            cksum = cksum_region(clue, cksum)
    if notes and _version_tuple(version) >= (1, 3):
        cksum = cksum_region(notes + b"\x00", cksum)
    return cksum


def _masked_bytes(sums: PuzChecksums) -> bytes:
    """The eight bytes stored at 0x10: low bytes then high bytes, XOR-masked."""
    parts = (sums.cib, sums.solution, sums.state, sums.text)
    low = bytes((value & 0xFF) for value in parts)
    high = bytes(((value >> 8) & 0xFF) for value in parts)
    return bytes(m ^ b for m, b in zip(_MASK, low + high))


def _unmask(masked: bytes) -> tuple[int, int, int, int]:
    plain = bytes(m ^ b for m, b in zip(_MASK, masked))
    values = tuple(plain[i] | (plain[i + 4] << 8) for i in range(4))
    return values  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Public info record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PuzInfo:
    """Everything about a ``.puz`` file that can be read without solving it.

    Available even for scrambled puzzles, which :func:`read_puz_bytes` refuses.
    """

    title: str
    author: str
    copyright: str
    notes: str
    width: int
    height: int
    scrambled: bool
    has_rebus: bool
    circled: tuple[Cell, ...]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Raw:
    """The literal contents of a file, before any interpretation."""

    version: str
    encoding: str
    width: int
    height: int
    n_clues: int
    puzzle_type: int
    scrambled_tag: int
    solution: str
    state: str
    title: str
    author: str
    copyright: str
    notes: str
    clues: tuple[str, ...]
    rebus: dict[Cell, str]
    circled: tuple[Cell, ...]
    extras: dict[str, bytes]
    # The raw bytes are kept so checksums can be recomputed exactly, even when a
    # string only decoded under the replacement fallback.
    header: bytes
    solution_bytes: bytes
    state_bytes: bytes
    title_bytes: bytes
    author_bytes: bytes
    copyright_bytes: bytes
    notes_bytes: bytes
    clue_bytes: tuple[bytes, ...]

    @property
    def scrambled(self) -> bool:
        return self.scrambled_tag == SCRAMBLED_TAG


def _decode(raw: bytes, encoding: str) -> str:
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError:
        # latin-1 cannot fail, so this only rescues a mislabelled v2.0 file.
        return raw.decode("latin-1", errors="replace")


def _read_cstring(data: bytes, pos: int, what: str) -> tuple[bytes, int]:
    end = data.find(b"\x00", pos)
    if end < 0:
        raise PuzError(f"unterminated {what} string at byte {pos}")
    return data[pos:end], end + 1


def _parse_rtbl(text: str) -> dict[int, str]:
    """Turn ``" 0:CAT; 1:DOG;"`` into ``{0: "CAT", 1: "DOG"}``."""
    table: dict[int, str] = {}
    for chunk in text.split(";"):
        if not chunk.strip():
            continue
        key, sep, value = chunk.partition(":")
        if not sep:
            continue
        try:
            table[int(key.strip())] = value
        except ValueError:
            continue
    return table


def _parse_extras(data: bytes, pos: int) -> dict[str, bytes]:
    """Read the trailing extension sections.

    Each is a 4-byte ASCII title, a u16 length, a u16 checksum, ``length``
    bytes, and a NUL. Anything that does not look like a section title ends the
    scan rather than raising: trailing junk after the last section is common.
    """
    extras: dict[str, bytes] = {}
    while pos + 8 <= len(data):
        title = data[pos : pos + 4]
        if not re.fullmatch(rb"[A-Z]{4}", title):
            break
        length, _stored_cksum = struct.unpack_from("<HH", data, pos + 4)
        start = pos + 8
        end = start + length
        if end > len(data):
            break
        extras[title.decode("ascii")] = data[start:end]
        pos = end + 1  # skip the section's trailing NUL
    return extras


def _parse(data: bytes) -> _Raw:
    index = data.find(MAGIC)
    if index < 0:
        raise PuzError("not a .puz file: magic b'ACROSS&DOWN\\x00' not found")
    base = index - 2
    if base < 0:
        raise PuzError("damaged .puz file: magic appears before the checksum field")
    if base + HEADER_SIZE > len(data):
        raise PuzError(
            f"truncated .puz header: need {HEADER_SIZE} bytes at offset {base}, "
            f"file has {len(data) - base}"
        )

    header = data[base : base + HEADER_SIZE]
    version = (
        header[_OFF_VERSION : _OFF_VERSION + 4]
        .split(b"\x00")[0]
        .decode("ascii", errors="replace")
    )
    # v2.0 switched the string encoding to UTF-8; everything earlier is latin-1.
    encoding = "utf-8" if version.startswith("2") else "latin-1"

    width, height, n_clues, puzzle_type, scrambled_tag = struct.unpack_from(
        "<BBHHH", header, _OFF_CIB
    )
    if width == 0 or height == 0:
        raise PuzError(f"degenerate grid: {height}x{width}")

    size = width * height
    grid_start = base + HEADER_SIZE
    if grid_start + 2 * size > len(data):
        raise PuzError(
            f"truncated .puz body: a {height}x{width} puzzle needs {2 * size} bytes "
            f"of grid data, only {max(0, len(data) - grid_start)} remain"
        )
    solution_bytes = data[grid_start : grid_start + size]
    state_bytes = data[grid_start + size : grid_start + 2 * size]

    pos = grid_start + 2 * size
    title_b, pos = _read_cstring(data, pos, "title")
    author_b, pos = _read_cstring(data, pos, "author")
    copyright_b, pos = _read_cstring(data, pos, "copyright")

    clue_bytes: list[bytes] = []
    for i in range(n_clues):
        try:
            clue, pos = _read_cstring(data, pos, f"clue {i + 1}")
        except PuzError as exc:
            raise PuzError(
                f"{exc}: header declares {n_clues} clues but the file ran out "
                f"after {i}"
            ) from None
        clue_bytes.append(clue)

    # Notes are the last string; a few writers omit the terminator at EOF.
    end = data.find(b"\x00", pos)
    if end < 0:
        notes_b, pos = data[pos:], len(data)
    else:
        notes_b, pos = data[pos:end], end + 1

    extras = _parse_extras(data, pos)

    # Grids are ASCII in practice; decoding latin-1 guarantees one char per byte
    # so cell indexing stays valid whatever a publisher put there.
    solution = solution_bytes.decode("latin-1")
    state = state_bytes.decode("latin-1")

    rebus: dict[Cell, str] = {}
    grbs = extras.get("GRBS")
    rtbl = extras.get("RTBL")
    if grbs and rtbl:
        table = _parse_rtbl(_decode(rtbl, encoding))
        for i, key in enumerate(grbs[:size]):
            if key == 0:
                continue
            text = table.get(key - 1)  # GRBS stores the RTBL key plus one
            if text:
                rebus[Cell(i // width, i % width)] = text

    circled: tuple[Cell, ...] = ()
    gext = extras.get("GEXT")
    if gext:
        circled = tuple(
            Cell(i // width, i % width)
            for i, flags in enumerate(gext[:size])
            if flags & GEXT_CIRCLED
        )

    return _Raw(
        version=version,
        encoding=encoding,
        width=width,
        height=height,
        n_clues=n_clues,
        puzzle_type=puzzle_type,
        scrambled_tag=scrambled_tag,
        solution=solution,
        state=state,
        title=_decode(title_b, encoding),
        author=_decode(author_b, encoding),
        copyright=_decode(copyright_b, encoding),
        notes=_decode(notes_b, encoding),
        clues=tuple(_decode(c, encoding) for c in clue_bytes),
        rebus=rebus,
        circled=circled,
        extras=extras,
        header=header,
        solution_bytes=solution_bytes,
        state_bytes=state_bytes,
        title_bytes=title_b,
        author_bytes=author_b,
        copyright_bytes=copyright_b,
        notes_bytes=notes_b,
        clue_bytes=tuple(clue_bytes),
    )


def _computed(raw: _Raw) -> PuzChecksums:
    cib = cksum_region(raw.header[_OFF_CIB : _OFF_CIB + _CIB_SIZE])
    text_args = (
        raw.title_bytes,
        raw.author_bytes,
        raw.copyright_bytes,
        raw.clue_bytes,
        raw.notes_bytes,
        raw.version,
    )
    running = cksum_region(raw.state_bytes, cksum_region(raw.solution_bytes, cib))
    return PuzChecksums(
        cib=cib,
        solution=cksum_region(raw.solution_bytes),
        state=cksum_region(raw.state_bytes),
        text=_text_cksum(*text_args),
        global_=_text_cksum(*text_args, running),
    )


def verify_checksums(data: bytes) -> dict[str, tuple[int, int]]:
    """Recompute every checksum and pair it with the one stored in the file.

    Returns ``name -> (stored, computed)``. The four values behind the mask at
    0x10 are unmasked first, so all of them are directly comparable.
    """
    raw = _parse(data)
    stored_global = struct.unpack_from("<H", raw.header, _OFF_GLOBAL_CKSUM)[0]
    stored_cib = struct.unpack_from("<H", raw.header, _OFF_CIB_CKSUM)[0]
    m_cib, m_sol, m_state, m_text = _unmask(raw.header[_OFF_MASKED : _OFF_MASKED + 8])
    computed = _computed(raw)
    return {
        "global": (stored_global, computed.global_),
        "cib": (stored_cib, computed.cib),
        "cib_masked": (m_cib, computed.cib),
        "solution": (m_sol, computed.solution),
        "state": (m_state, computed.state),
        "text": (m_text, computed.text),
    }


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def puz_info(data: bytes) -> PuzInfo:
    """Metadata for a ``.puz`` payload, including ones we refuse to solve."""
    raw = _parse(data)
    return PuzInfo(
        title=raw.title,
        author=raw.author,
        copyright=raw.copyright,
        notes=raw.notes,
        width=raw.width,
        height=raw.height,
        scrambled=raw.scrambled,
        has_rebus=bool(raw.rebus),
        circled=raw.circled,
    )


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48].strip("-")


def _entry_order(
    blocks: frozenset[Cell], width: int, height: int
) -> list[tuple[int, Direction]]:
    """Entries in the order ``.puz`` stores their clues: by number, across first."""
    _numbers, runs = number_grid(blocks, width, height)
    return sorted(runs, key=lambda key: (key[0], 0 if key[1] == "across" else 1))


def read_puz_bytes(data: bytes, puzzle_id: str | None = None) -> Puzzle:
    """Parse a ``.puz`` payload into a :class:`~xword.core.types.Puzzle`.

    Rebus squares are flattened to their first letter, because the solver's cell
    alphabet is exactly the 26 letters: a rebus entry is therefore scored as if
    the square held that first letter, which is how these puzzles are graded in
    practice anyway. Nothing is lost -- the full rebus map is preserved verbatim
    in ``meta["rebus"]`` as JSON, so a renderer or the harness can put the whole
    string back.

    Raises :class:`PuzError` for a scrambled file: its solution is encrypted and
    this reader will not attempt to recover the key.
    """
    raw = _parse(data)
    if raw.scrambled:
        raise PuzError(
            "the solution in this .puz file is scrambled (scrambled tag 0x0004); "
            "unlock it in Across Lite with the puzzle's key and save it again"
        )

    letters = list(raw.solution)
    for cell, text in raw.rebus.items():
        if text:
            letters[cell.row * raw.width + cell.col] = text[0]

    rows: list[str] = []
    solution_rows: list[str] = []
    solvable = True
    for r in range(raw.height):
        shape: list[str] = []
        answer: list[str] = []
        for c in range(raw.width):
            ch = letters[r * raw.width + c]
            if ch in _BLOCK_GRID_CHARS:
                shape.append(BLOCK_CHAR)
                answer.append(BLOCK_CHAR)
                continue
            shape.append(WILDCARD)
            upper = ch.upper()
            if not ("A" <= upper <= "Z"):
                solvable = False
            answer.append(upper)
        rows.append("".join(shape))
        solution_rows.append("".join(answer))

    blocks, width, height = parse_block_rows(rows)
    entries = _entry_order(blocks, width, height)
    if len(entries) != raw.n_clues:
        raise PuzError(
            f"clue count mismatch: header declares {raw.n_clues} clues, the grid "
            f"yields {len(entries)} entries"
        )

    across_clues: dict[int, str] = {}
    down_clues: dict[int, str] = {}
    for (number, direction), clue in zip(entries, raw.clues):
        target = across_clues if direction == "across" else down_clues
        target[number] = clue

    meta: dict[str, str] = {
        "title": raw.title,
        "author": raw.author,
        "copyright": raw.copyright,
        "notes": raw.notes,
        "source": "puz",
        "version": raw.version,
        "circles": json.dumps([[c.row, c.col] for c in raw.circled]),
    }
    if raw.rebus:
        meta["rebus"] = json.dumps(
            [[cell.row, cell.col, text] for cell, text in sorted(raw.rebus.items())]
        )
    if not solvable:
        # An unfilled or non-alphabetic solution grid still describes a usable
        # puzzle; it just cannot be graded, so no reference solution is attached
        # rather than one full of junk letters.
        meta["solution_status"] = "incomplete"

    if puzzle_id is None:
        puzzle_id = _slug(raw.title) or f"puz-{hashlib.sha1(data).hexdigest()[:8]}"

    return make_puzzle(
        puzzle_id,
        rows,
        across_clues=across_clues,
        down_clues=down_clues,
        solution_rows=solution_rows if solvable else None,
        meta=meta,
    )


def read_puz(path: str | Path) -> Puzzle:
    """Read a ``.puz`` file from disk. The puzzle id defaults to the file stem."""
    file_path = Path(path)
    try:
        data = file_path.read_bytes()
    except OSError as exc:
        raise PuzError(f"cannot read {file_path}: {exc}") from exc
    return read_puz_bytes(data, puzzle_id=file_path.stem)


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def _rebus_from_meta(meta: Mapping[str, str]) -> dict[Cell, str]:
    blob = meta.get("rebus")
    if not blob:
        return {}
    try:
        entries = json.loads(blob)
    except (ValueError, TypeError):
        return {}
    out: dict[Cell, str] = {}
    for item in entries or []:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            row, col, text = item
            out[Cell(int(row), int(col))] = str(text)
    return out


def _circles_from_meta(meta: Mapping[str, str]) -> list[Cell]:
    blob = meta.get("circles")
    if not blob:
        return []
    try:
        entries = json.loads(blob)
    except (ValueError, TypeError):
        return []
    out: list[Cell] = []
    for item in entries or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append(Cell(int(item[0]), int(item[1])))
    return out


def _extra_section(title: str, body: bytes) -> bytes:
    return (
        title.encode("ascii")
        + struct.pack("<HH", len(body), cksum_region(body))
        + body
        + b"\x00"
    )


def write_puz(
    puzzle: Puzzle, path: str | Path, fill: Mapping[Cell, str] | None = None
) -> None:
    """Write ``puzzle`` as a ``.puz`` file that Across Lite -- and
    :func:`read_puz` -- will accept.

    ``fill`` is the *player state* grid, i.e. partial progress. It never touches
    the solution grid, which comes from ``puzzle.solution``; open cells with no
    known answer are written as ``-``, which is what an unsolved square looks
    like in this format.

    Rebus squares and circles recorded in ``puzzle.meta`` are re-emitted as
    GRBS/RTBL and GEXT sections, so a file read by :func:`read_puz` and written
    back keeps them.
    """
    width, height = puzzle.width, puzzle.height
    if not (1 <= width <= 0xFF and 1 <= height <= 0xFF):
        raise PuzError(
            f"the .puz header stores each dimension in one byte; {height}x{width} "
            "does not fit"
        )
    if len(puzzle.slots) > 0xFFFF:
        raise PuzError(
            f"the .puz header stores the clue count in two bytes; "
            f"{len(puzzle.slots)} entries do not fit"
        )
    solution_letters = puzzle.solution_letters() if puzzle.has_solution else {}
    rebus = _rebus_from_meta(puzzle.meta)
    circles = _circles_from_meta(puzzle.meta)
    fill = fill or {}

    solution_chars: list[str] = []
    state_chars: list[str] = []
    for r in range(height):
        for c in range(width):
            cell = Cell(r, c)
            if cell in puzzle.blocks:
                solution_chars.append(".")
                state_chars.append(".")
                continue
            answer = solution_letters.get(cell) or "-"
            solution_chars.append(answer[0].upper() if answer != "-" else "-")
            guess = fill.get(cell) or "-"
            state_chars.append(guess[0].upper() if guess != "-" else "-")

    ordered = sorted(
        puzzle.slots, key=lambda s: (s.number, 0 if s.direction == "across" else 1)
    )
    clue_texts = [slot.clue for slot in ordered]
    strings = [
        puzzle.meta.get("title", ""),
        puzzle.meta.get("author", ""),
        puzzle.meta.get("copyright", ""),
        *clue_texts,
        puzzle.meta.get("notes", ""),
    ]

    version = str(puzzle.meta.get("version") or "").strip()
    if not re.fullmatch(r"\d+\.\d+", version):
        version = DEFAULT_WRITE_VERSION
    encoding = "utf-8" if version.startswith("2") else "latin-1"
    try:
        encoded = [s.encode(encoding) for s in strings]
    except UnicodeEncodeError:
        # Anything outside latin-1 forces the v2.0 UTF-8 flavour of the format.
        version, encoding = "2.0", "utf-8"
        encoded = [s.encode(encoding) for s in strings]

    title_b, author_b, copyright_b = encoded[0], encoded[1], encoded[2]
    clue_bytes = encoded[3 : 3 + len(clue_texts)]
    notes_b = encoded[-1]

    solution_bytes = "".join(solution_chars).encode("latin-1")
    state_bytes = "".join(state_chars).encode("latin-1")

    header = bytearray(HEADER_SIZE)
    header[_OFF_MAGIC : _OFF_MAGIC + len(MAGIC)] = MAGIC
    header[_OFF_VERSION : _OFF_VERSION + 4] = version.encode("ascii").ljust(4, b"\x00")[
        :4
    ]
    struct.pack_into(
        "<BBHHH",
        header,
        _OFF_CIB,
        width,
        height,
        len(clue_texts),
        PUZZLE_TYPE_NORMAL,
        0,
    )

    cib = cksum_region(bytes(header[_OFF_CIB : _OFF_CIB + _CIB_SIZE]))
    text_args = (title_b, author_b, copyright_b, clue_bytes, notes_b, version)
    running = cksum_region(state_bytes, cksum_region(solution_bytes, cib))
    sums = PuzChecksums(
        cib=cib,
        solution=cksum_region(solution_bytes),
        state=cksum_region(state_bytes),
        text=_text_cksum(*text_args),
        global_=_text_cksum(*text_args, running),
    )
    struct.pack_into("<H", header, _OFF_CIB_CKSUM, sums.cib)
    struct.pack_into("<H", header, _OFF_GLOBAL_CKSUM, sums.global_)
    struct.pack_into("<H", header, _OFF_SCRAMBLED_CKSUM, 0)
    header[_OFF_MASKED : _OFF_MASKED + 8] = _masked_bytes(sums)

    body = bytearray(header)
    body += solution_bytes
    body += state_bytes
    for chunk in (title_b, author_b, copyright_b, *clue_bytes, notes_b):
        body += chunk + b"\x00"

    if rebus:
        keys = {cell: i for i, cell in enumerate(sorted(rebus))}
        grbs = bytearray(width * height)
        for cell, key in keys.items():
            if 0 <= cell.row < height and 0 <= cell.col < width:
                grbs[cell.row * width + cell.col] = key + 1
        rtbl = "".join(f"{key:2d}:{rebus[cell]};" for cell, key in keys.items())
        body += _extra_section("GRBS", bytes(grbs))
        body += _extra_section("RTBL", rtbl.encode(encoding))
    if circles:
        gext = bytearray(width * height)
        for cell in circles:
            if 0 <= cell.row < height and 0 <= cell.col < width:
                gext[cell.row * width + cell.col] |= GEXT_CIRCLED
        body += _extra_section("GEXT", bytes(gext))

    out_path = Path(path)
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(body))


__all__ = [
    "GEXT_CIRCLED",
    "HEADER_SIZE",
    "MAGIC",
    "PUZZLE_TYPE_NORMAL",
    "PuzChecksums",
    "PuzError",
    "PuzInfo",
    "SCRAMBLED_TAG",
    "cksum_region",
    "puz_info",
    "read_puz",
    "read_puz_bytes",
    "verify_checksums",
    "write_puz",
]

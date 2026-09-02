"""Pattern lookup over a word list.

The solver asks "which words fit ``?A??E?``" tens of thousands of times for a
single puzzle -- belief propagation wants a per-square letter distribution for
every open cell on every iteration, and the beam search wants one for every
candidate it extends. A linear scan over a 100k-word list costs milliseconds per
query, which is two orders of magnitude too slow at that call volume, so this
module trades memory for latency.

Representation
--------------
Words are bucketed by length; nothing ever compares words of different lengths.
Inside a bucket of ``n`` words of length ``L`` there is one bitset per
``(position, letter)`` pair, stored as ``uint64`` lanes: bit ``j`` of
``bits[i, c]`` is set when word ``j`` has letter ``c`` at position ``i``.
Matching a pattern is a bitwise AND of the bitsets named by the constrained
positions -- ``ceil(n/64)`` machine words of work per constrained position,
independent of how many words actually match -- followed by unpacking the
surviving bits back to word indices.

Words inside a bucket are stored in descending score order, so the ascending bit
indices that fall out of the AND are already the ranking ``match`` promises, and
a ``limit`` can be honoured by walking only the first few set bits.

Memory
------
The bitsets cost ``26 * L * ceil(n / 64) * 8`` bytes per bucket, i.e. about
``3.25 * L * n`` bytes -- roughly 30 bytes per word in a 9-letter bucket, about
11 MB for the whole of a 370k-word dictionary. The per-position letter-code
table that :meth:`letters_at` sums over adds ``L * n`` bytes (another ~3 MB at
that size), and the Python strings plus the word/score dict together cost more
than either. Everything is allocated once, at construction.

The structure is immutable once built, so a query is a pure function of its
pattern; there is nothing to invalidate and nothing is memoised.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from xword.core.types import ALPHABET, LETTER_INDEX, WILDCARD

__all__ = ["PatternIndex"]

_BITS_PER_LANE = 64
_BYTES_PER_LANE = 8
_NATIVE_LITTLE = sys.byteorder == "little"
_HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")

#: Score assumed when the caller supplies no scores. Every word then ties and
#: ranking degrades to alphabetical, which is the honest outcome: an unscored
#: word list carries no opinion about which fill is better.
_UNSCORED = 1.0


# --------------------------------------------------------------------------- #
# Bit plumbing
# --------------------------------------------------------------------------- #
#
# Bit ``j`` of a bitset lives in lane ``j // 64`` at offset ``j % 64``. numpy has
# no uint64 bit-scatter primitive, so the lanes are built with ``packbits`` in
# little bit order -- which lays bit j of a row into byte j//8 at offset j%8 --
# and the byte array is then reinterpreted as uint64. On a little-endian host
# that reinterpretation already puts bit j at offset j%64 of lane j//64; on a
# big-endian host the byte order inside a lane is reversed, so pack and unpack
# both byteswap and remain each other inverse.


def _pack(rows: np.ndarray, lanes: int) -> np.ndarray:
    """Turn a ``(k, n)`` boolean matrix into ``k`` bitsets of ``lanes`` lanes."""
    packed = np.packbits(rows, axis=1, bitorder="little")
    needed = lanes * _BYTES_PER_LANE
    if packed.shape[1] < needed:
        packed = np.pad(packed, ((0, 0), (0, needed - packed.shape[1])))
    out = np.ascontiguousarray(packed).view(np.uint64)
    return out if _NATIVE_LITTLE else out.byteswap()


def _lane_bytes(mask: np.ndarray) -> np.ndarray:
    """The byte view of a bitset, in the order :func:`_pack` wrote it."""
    src = mask if _NATIVE_LITTLE else mask.byteswap()
    return np.ascontiguousarray(src).view(np.uint8)


def _set_bits(mask: np.ndarray) -> np.ndarray:
    """Ascending indices of the set bits of ``mask``."""
    return np.flatnonzero(np.unpackbits(_lane_bytes(mask), bitorder="little"))


def _first_set_bits(mask: np.ndarray, limit: int) -> list[int]:
    """Up to ``limit`` ascending set-bit indices, without unpacking the rest.

    The Python loop pays for itself: the solver almost always wants a bounded
    top-k, and unpacking a 100k-bit mask to keep 60 answers is pure waste.
    """
    out: list[int] = []
    if limit <= 0:
        return out
    for lane in np.flatnonzero(mask).tolist():
        chunk = int(mask[lane])
        base = lane * _BITS_PER_LANE
        while chunk:
            low = chunk & -chunk
            out.append(base + low.bit_length() - 1)
            if len(out) >= limit:
                return out
            chunk ^= low
    return out


def _popcount(mask: np.ndarray) -> int:
    if _HAS_BITWISE_COUNT:
        return int(np.bitwise_count(mask).sum(dtype=np.int64))
    return int(np.unpackbits(_lane_bytes(mask)).sum(dtype=np.int64))


# --------------------------------------------------------------------------- #
# Buckets
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _LengthGroup:
    """Every word of one length, plus the bitsets that index them."""

    words: tuple[str, ...]
    scores: np.ndarray  # float64, (n,), descending
    codes: np.ndarray  # uint8, (L, n): letter index of each word at each position
    bits: np.ndarray  # uint64, (L, 26, lanes)
    full: np.ndarray  # uint64, (lanes,): every word in the bucket

    @property
    def size(self) -> int:
        return len(self.words)

    @property
    def nbytes(self) -> int:
        return int(
            self.scores.nbytes + self.codes.nbytes + self.bits.nbytes + self.full.nbytes
        )


def _build_group(words: list[str], scores: np.ndarray) -> _LengthGroup:
    count = len(words)
    length = len(words[0])
    lanes = (count + _BITS_PER_LANE - 1) // _BITS_PER_LANE

    flat = np.frombuffer("".join(words).encode("ascii"), dtype=np.uint8)
    codes = np.ascontiguousarray((flat.reshape(count, length) - np.uint8(65)).T)

    letters = np.arange(26, dtype=np.uint8)[:, None]
    bits = np.empty((length, 26, lanes), dtype=np.uint64)
    for position in range(length):
        bits[position] = _pack(codes[position][None, :] == letters, lanes)

    full = _pack(np.ones((1, count), dtype=bool), lanes)[0]
    return _LengthGroup(
        words=tuple(words), scores=scores, codes=codes, bits=bits, full=full
    )


# --------------------------------------------------------------------------- #
# The index
# --------------------------------------------------------------------------- #


class PatternIndex:
    """Bitset index answering "which words match this pattern?".

    ``words`` are upper-cased, and anything that is not pure ``A-Z`` is dropped
    rather than mangled -- a list that needs cleaning should go through
    :func:`xword.lexicon.build.normalise_answer` first, where the caller can see
    what happened to it. Duplicates keep their highest score.

    ``scores`` must be parallel to ``words`` and are clamped to ``[0, 1]``.
    """

    __slots__ = ("_groups", "_scores")

    def __init__(
        self, words: Sequence[str], scores: Sequence[float] | None = None
    ) -> None:
        if scores is not None and len(scores) != len(words):
            raise ValueError(
                f"{len(words)} words but {len(scores)} scores; they must be parallel"
            )

        best: dict[str, float] = {}
        for position, raw in enumerate(words):
            word = raw.strip().upper()
            if not word or not (word.isascii() and word.isalpha()):
                continue
            value = _UNSCORED if scores is None else float(scores[position])
            if value != value:  # NaN carries no ranking information
                continue
            value = 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)
            if value > best.get(word, -1.0):
                best[word] = value

        by_length: dict[int, list[str]] = {}
        for word in best:
            by_length.setdefault(len(word), []).append(word)

        groups: dict[int, _LengthGroup] = {}
        for length in sorted(by_length):
            bucket = by_length[length]
            # Descending score, alphabetical to break ties, so that bit order
            # inside a bucket *is* rank order and is stable across runs.
            bucket.sort(key=lambda w: (-best[w], w))
            groups[length] = _build_group(
                bucket, np.array([best[w] for w in bucket], dtype=np.float64)
            )

        self._groups = groups
        self._scores = best

    # -- introspection ----------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._scores)

    def __contains__(self, word: object) -> bool:
        if not isinstance(word, str):
            return False
        return word.strip().upper() in self._scores

    def lengths(self) -> tuple[int, ...]:
        """The word lengths this index holds, ascending."""
        return tuple(sorted(self._groups))

    def size(self, length: int) -> int:
        """How many words of ``length`` are indexed."""
        group = self._groups.get(length)
        return 0 if group is None else group.size

    @property
    def nbytes(self) -> int:
        """Bytes held by the numpy side of the index (see the module docstring)."""
        return sum(group.nbytes for group in self._groups.values())

    # -- matching ---------------------------------------------------------- #

    def _mask_for(self, group: _LengthGroup, pattern: str) -> np.ndarray | None:
        """Bitset of the words in ``group`` matching ``pattern``.

        ``None`` means "nothing can match" (an unusable pattern character). The
        result may be the bucket shared ``full`` mask, so callers must treat it
        as read-only.
        """
        constrained: list[tuple[int, int]] = []
        for position, char in enumerate(pattern):
            if char == WILDCARD:
                continue
            letter = LETTER_INDEX.get(char)
            if letter is None:
                return None
            constrained.append((position, letter))

        if not constrained:
            return group.full

        first_pos, first_letter = constrained[0]
        mask = group.bits[first_pos, first_letter].copy()
        for position, letter in constrained[1:]:
            np.bitwise_and(mask, group.bits[position, letter], out=mask)
        return mask

    def match(self, pattern: str, limit: int | None = None) -> list[tuple[str, float]]:
        """Words consistent with ``pattern``, best score first.

        ``pattern`` is upper-cased for convenience; ``?`` matches any letter. A
        pattern holding a character that is neither ``A-Z`` nor ``?`` matches
        nothing -- a partially filled grid should never produce one, and raising
        would turn a cosmetic bug into a failed solve.
        """
        group = self._groups.get(len(pattern))
        if group is None:
            return []
        mask = self._mask_for(group, pattern.upper())
        if mask is None:
            return []

        if limit is not None and limit < group.size:
            indices = _first_set_bits(mask, limit)
        else:
            indices = _set_bits(mask).tolist()

        words = group.words
        scores = group.scores
        return [(words[j], float(scores[j])) for j in indices]

    def count(self, pattern: str) -> int:
        """How many words match ``pattern``, without materialising them.

        The all-wildcard case is O(1) because the solver asks it for every
        unconstrained entry on every round just to size the search.
        """
        group = self._groups.get(len(pattern))
        if group is None:
            return 0
        text = pattern.upper()
        if text.count(WILDCARD) == len(text):
            return group.size
        mask = self._mask_for(group, text)
        if mask is None:
            return 0
        return _popcount(mask)

    def letters_at(self, pattern: str, position: int) -> dict[str, float]:
        """Summed score mass per letter at ``position`` over the matching words.

        This is the "how constrained is this square" signal: a square whose mass
        sits almost entirely on one letter is effectively decided, and one whose
        mass is spread over a dozen letters is where the solver should spend a
        clue call. Letters come back heaviest first, and a letter reachable only
        through zero-scored words still appears (with mass ``0.0``) so a caller
        asking "what is possible here" does not silently lose it.
        """
        if position < 0 or position >= len(pattern):
            raise IndexError(
                f"position {position} outside pattern of length {len(pattern)}"
            )
        group = self._groups.get(len(pattern))
        if group is None:
            return {}
        mask = self._mask_for(group, pattern.upper())
        if mask is None:
            return {}

        indices = _set_bits(mask)
        if indices.size == 0:
            return {}
        codes = group.codes[position][indices]
        mass = np.bincount(codes, weights=group.scores[indices], minlength=26)
        seen = np.bincount(codes, minlength=26)

        out: dict[str, float] = {}
        for letter in np.argsort(-mass, kind="stable").tolist():
            if seen[letter]:
                out[ALPHABET[letter]] = float(mass[letter])
        return out

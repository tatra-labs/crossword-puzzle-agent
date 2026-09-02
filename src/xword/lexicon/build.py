"""Constructing the scored word list.

Two kinds of source, and they are not equally good. A dictionary tells you a
string is a word; a pile of published puzzles tells you constructors actually
use it, which is the thing the solver needs to know. ``BRAE`` and ``OREO`` are
both in the dictionary and only one of them is real crossword fill, so answers
mined from solved puzzles carry the frequency signal and the dictionary is only
there for coverage. That ranking -- attested fill above dictionary words -- is
the single biggest quality lever in this package, which is why
:func:`build_default_lexicon` weights the puzzle-mined source several times
above the word list.

Everything produced here is upper-case ``A-Z``, length 2 to 23, scored in
``[0, 1]``; see :mod:`xword.lexicon.store` for what the scale means.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from xword import config
from xword.core.grid import make_puzzle
from xword.core.types import Puzzle
from xword.lexicon import store
from xword.lexicon.store import Lexicon, parse_score_line

__all__ = [
    "MAX_ANSWER_LEN",
    "MIN_ANSWER_LEN",
    "build_default_lexicon",
    "build_from_puzzles",
    "build_from_wordlist",
    "merge_scores",
    "normalise_answer",
    "score_by_frequency",
]

#: Grids do not have one-letter entries, and the longest entry in a 23x23
#: Sunday is 23 cells.
MIN_ANSWER_LEN = 2
MAX_ANSWER_LEN = 23

#: Score for a word list entry that arrives with no score of its own. Low on
#: purpose: a bare dictionary word is plausible fill, not good fill, and it
#: should lose to anything a real puzzle has used.
UNRATED_WORDLIST_SCORE = 0.3

#: Floor for an answer seen in at least one published puzzle. Above
#: :data:`UNRATED_WORDLIST_SCORE` because attestation is itself evidence.
PUZZLE_SCORE_FLOOR = 0.4

#: What a single sighting is worth when the corpus is too small to rank
#: anything (every answer seen exactly once).
SINGLE_SIGHTING_SCORE = 0.6

#: How much more :func:`build_default_lexicon` trusts mined answers than a
#: dictionary.
PUZZLE_SOURCE_WEIGHT = 3.0

_PUZZLE_SUFFIXES = frozenset({".json", ".ipuz", ".puz"})


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def normalise_answer(text: str) -> str:
    """Reduce a printed answer to grid letters.

    Grids hold nothing but ``A-Z``, so spaces, punctuation and case all go:
    ``"It's a deal!"`` -> ``"ITSADEAL"``. Accents are decomposed first so that
    ``"café"`` becomes ``"CAFE"`` rather than losing its last letter. Digits are
    dropped too, which is why callers should treat a result whose length no
    longer matches the entry as a rejected answer rather than a repaired one.
    """
    decomposed = unicodedata.normalize("NFKD", text).upper()
    return "".join(ch for ch in decomposed if "A" <= ch <= "Z")


def _acceptable(word: str, min_len: int, max_len: int) -> bool:
    return min_len <= len(word) <= max_len


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


def build_from_wordlist(
    path: str | Path, *, min_len: int = MIN_ANSWER_LEN, max_len: int = MAX_ANSWER_LEN
) -> dict[str, float]:
    """Read a word list into ``word -> score``.

    Handles both shapes in the wild: a plain word per line, and the
    ``WORD;score`` files that constructor tools trade (0-100 or 0-1, see
    :func:`xword.lexicon.store.parse_score_line`). Unscored entries all get
    :data:`UNRATED_WORDLIST_SCORE`, which leaves them tied -- a raw dictionary
    genuinely has no opinion about which of its words is better fill, and
    inventing an ordering here would be inventing evidence.

    Entries are normalised and then length-filtered, so a phrase like
    ``"IT IS A DEAL"`` survives as ``ITISADEAL`` and a stray ``"1990s"`` does
    not survive at all.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out: dict[str, float] = {}
    for line in text.splitlines():
        parsed = parse_score_line(line)
        if parsed is None:
            continue
        raw, score = parsed
        word = normalise_answer(raw)
        if not _acceptable(word, min_len, max_len):
            continue
        value = UNRATED_WORDLIST_SCORE if score is None else score
        if value > out.get(word, -1.0):
            out[word] = value
    return out


def score_by_frequency(counts: Mapping[str, int]) -> dict[str, float]:
    """Map "seen in N puzzles" onto ``[PUZZLE_SCORE_FLOOR, 1.0]``.

    Log rather than linear: the gap between an answer seen once and one seen
    five times says a lot, the gap between 200 and 205 says nothing, and a
    linear scale would push everything except the handful of most-repeated
    answers down into a band the solver cannot rank inside.
    """
    if not counts:
        return {}
    highest = max(counts.values())
    if highest <= 1:
        # No frequency signal to extract; every answer is simply attested.
        return {word: SINGLE_SIGHTING_SCORE for word in counts}
    denominator = math.log1p(highest)
    span = 1.0 - PUZZLE_SCORE_FLOOR
    return {
        word: PUZZLE_SCORE_FLOOR + span * math.log1p(count) / denominator
        for word, count in counts.items()
    }


def build_from_puzzles(puzzles: Iterable[Puzzle]) -> dict[str, float]:
    """Mine answers from solved puzzles, scored by how often they appear.

    An answer that has run in many published puzzles is far better fill than a
    random dictionary word, so this is the source that actually ranks the
    lexicon. Repeats inside one puzzle count once: the signal wanted is "how
    many editors let this through", not "how long is the grid".

    Puzzles with no reference solution contribute nothing.
    """
    counts: Counter[str] = Counter()
    for puzzle in puzzles:
        if puzzle.solution is None:
            continue
        seen: set[str] = set()
        for raw in puzzle.solution.values():
            word = normalise_answer(raw)
            if _acceptable(word, MIN_ANSWER_LEN, MAX_ANSWER_LEN):
                seen.add(word)
        counts.update(seen)
    return score_by_frequency(counts)


def merge_scores(
    *sources: Mapping[str, float], weights: Sequence[float] | None = None
) -> dict[str, float]:
    """Combine several word -> score maps into one.

    A word takes the weighted mean of the scores of the sources that *contain*
    it. Absence is "no opinion", not "score zero": a dictionary that omits
    ``ESAI`` is not evidence that it is bad fill, and the puzzle-mined source is
    deliberately a small subset of the dictionary, so counting its omissions
    against a word would erase exactly the ranking this module exists to build.

    ``weights`` is parallel to ``sources`` and defaults to all ones.
    """
    if weights is None:
        weights = [1.0] * len(sources)
    if len(weights) != len(sources):
        raise ValueError(f"{len(sources)} sources but {len(weights)} weights")

    totals: dict[str, float] = {}
    divisors: dict[str, float] = {}
    for source, weight in zip(sources, weights):
        weight = float(weight)
        if weight <= 0.0:
            continue
        for word, score in source.items():
            totals[word] = totals.get(word, 0.0) + weight * float(score)
            divisors[word] = divisors.get(word, 0.0) + weight

    out: dict[str, float] = {}
    for word, total in totals.items():
        value = total / divisors[word]
        out[word] = 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)
    return out


# --------------------------------------------------------------------------- #
# Reading puzzle directories
# --------------------------------------------------------------------------- #
#
# ``xword.io`` owns puzzle reading, but the lexicon build has to keep working
# before that module lands and for formats it does not cover, so the loader is
# resolved at call time and there is a small JSON fallback behind it. Only the
# answers are wanted here, never the clues or the geometry.

_IO_LOADERS = ("load_puzzle", "read_puzzle", "parse_puzzle", "load_puzzle_file")


def _io_loader():
    """The format-dispatching reader, resolved late.

    Imported from :mod:`xword.io.loaders` rather than from the ``xword.io``
    package: the package re-exports it, but reaching for the module directly
    means this keeps working even if the package's ``__init__`` is trimmed.
    """
    try:
        from xword.io import loaders  # noqa: PLC0415 - optional, resolved late
    except Exception:
        return None
    for name in _IO_LOADERS:
        candidate = getattr(loaders, name, None)
        if callable(candidate):
            return candidate
    return None


def _answers_of(puzzle: Puzzle) -> set[str]:
    if puzzle.solution is None:
        return set()
    return {
        word
        for word in (normalise_answer(a) for a in puzzle.solution.values())
        if _acceptable(word, MIN_ANSWER_LEN, MAX_ANSWER_LEN)
    }


def _answers_from_json(payload: object) -> set[str]:
    """Best-effort answer extraction from a JSON puzzle dump."""
    if isinstance(payload, list):
        out: set[str] = set()
        for item in payload:
            out |= _answers_from_json(item)
        return out
    if not isinstance(payload, dict):
        return set()

    solution = None
    for key in ("solution", "solution_rows", "answers", "fill"):
        if key in payload:
            solution = payload[key]
            break
    if solution is None:
        return set()

    if isinstance(solution, dict):
        # Some schemas group answers by direction -- the NYT archive uses
        # ``{"across": [...], "down": [...]}`` -- so the values may themselves
        # be lists. Flatten one level; stringifying a list here would produce
        # garbage that silently fails the length filter and mine nothing.
        candidates = []
        for value in solution.values():
            if isinstance(value, (list, tuple)):
                candidates.extend(value)
            else:
                candidates.append(value)
    elif isinstance(solution, list) and all(isinstance(x, str) for x in solution):
        rows = [str(x) for x in solution]
        widths = {len(r) for r in rows}
        if len(rows) > 1 and len(widths) == 1:
            # Row strings: run them through the real grid code so that entries
            # are cut the same way the solver would cut them.
            puzzle = make_puzzle("json", rows, solution_rows=rows)
            return _answers_of(puzzle)
        candidates = rows
    else:
        return set()

    return {
        word
        for word in (normalise_answer(str(a)) for a in candidates)
        if _acceptable(word, MIN_ANSWER_LEN, MAX_ANSWER_LEN)
    }


def _answer_sets(puzzle_dirs: Iterable[str | Path]) -> list[set[str]]:
    """One set of answers per puzzle file found under ``puzzle_dirs``.

    Unreadable files are skipped rather than fatal: a lexicon build over a few
    thousand downloaded puzzles should not die on one truncated download.
    """
    loader = _io_loader()
    out: list[set[str]] = []
    for directory in puzzle_dirs:
        root = Path(directory)
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _PUZZLE_SUFFIXES:
                continue
            answers: set[str] = set()
            if loader is not None:
                try:
                    loaded = loader(path)
                except Exception:
                    loaded = None
                if isinstance(loaded, Puzzle):
                    answers = _answers_of(loaded)
                elif isinstance(loaded, (list, tuple)):
                    for item in loaded:
                        if isinstance(item, Puzzle):
                            answers |= _answers_of(item)
            if not answers and path.suffix.lower() == ".json":
                try:
                    answers = _answers_from_json(
                        json.loads(path.read_text(encoding="utf-8", errors="replace"))
                    )
                except Exception:
                    answers = set()
            if answers:
                out.append(answers)
    return out


# --------------------------------------------------------------------------- #
# The build
# --------------------------------------------------------------------------- #


def build_default_lexicon(
    wordlist_path: str | Path | None = None,
    puzzle_dirs: Iterable[str | Path] = (),
    out_path: str | Path | None = None,
) -> Path:
    """Build the lexicon file and return where it was written.

    Mined answers are weighted :data:`PUZZLE_SOURCE_WEIGHT` times the word list,
    so a word both sources know lands near its puzzle-mined score while a
    dictionary-only word stays at :data:`UNRATED_WORDLIST_SCORE`. With no
    sources at all the built-in fallback is written out, so the output path
    always exists afterwards and the caller never has to special-case an empty
    build.
    """
    sources: list[Mapping[str, float]] = []
    weights: list[float] = []

    if wordlist_path is not None:
        sources.append(build_from_wordlist(wordlist_path))
        weights.append(1.0)

    answer_sets = _answer_sets(puzzle_dirs)
    if answer_sets:
        counts: Counter[str] = Counter()
        for answers in answer_sets:
            counts.update(answers)
        sources.append(score_by_frequency(counts))
        weights.append(PUZZLE_SOURCE_WEIGHT)

    if not sources:
        sources.append(dict(store.BUILTIN_FALLBACK))
        weights.append(1.0)

    merged = merge_scores(*sources, weights=weights)
    target = Path(out_path) if out_path is not None else Path(config.DEFAULT_LEXICON_PATH)
    Lexicon(merged).save(target)
    return target

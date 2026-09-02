"""Pattern-driven candidates from the offline word list.

The lexicon knows which strings *fit*; it has no idea what the clue means. So
this source earns its keep in two narrow ways: it rescues entries the LLM
whiffed on once crossings have pinned down enough letters, and it seconds an
LLM answer that is also a real crossword word, which fusion rewards as
independent agreement. What it must never do is outvote a confident LLM answer
on the strength of raw word frequency -- hence the score ceiling below.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from xword.core.grid import pattern_matches
from xword.core.types import WILDCARD, Candidate, ClueRequest


class LexiconLike(Protocol):
    """The slice of ``xword.lexicon.store.Lexicon`` this module needs.

    Declared structurally so that nothing here imports the store at module
    load time; the concrete class is built by another module and is only ever
    touched through these two methods.
    """

    def match(self, pattern: str, limit: int) -> list[tuple[str, float]]:
        ...

    def score(self, word: str) -> float:
        ...


#: Where the *best* lexicon match lands on the shared log-score scale.
#: Candidate scores across the repo are log-probability-like, so this is
#: ``log(0.30)`` -- a mid-confidence guess. A confident LLM answer sits near
#: ``log(0.85) = -0.16`` and still wins comfortably; a hesitant LLM answer near
#: ``log(0.10) = -2.30`` does not, which is exactly the trade we want: when the
#: model is unsure, "it is at least a real word that fits" is the better bet.
MID_CONFIDENCE_SCORE = math.log(0.30)

#: Largest log gap we are willing to express between the best and the worst
#: match in one list. Lexicon weights are frequency-like and span many orders
#: of magnitude; passing that spread through unclipped would make the 30th
#: match numerically dead and, worse, would let a frequency ratio masquerade as
#: semantic evidence.
MAX_SCORE_SPAN = 4.0

#: Floor applied before taking a log, so a zero or negative weight from the
#: store degrades to "worst match" instead of raising.
MIN_LEXICON_SCORE = 1e-12


class LexiconCandidateSource:
    """A :class:`~xword.core.types.CandidateSource` backed by the word list."""

    name = "lexicon"

    __slots__ = ("lexicon", "limit", "require_pattern")

    def __init__(
        self,
        lexicon: LexiconLike | None,
        *,
        limit: int = 60,
        require_pattern: bool = True,
    ) -> None:
        """``lexicon`` may be ``None``, in which case this source proposes
        nothing -- convenient for the ``no-lexicon`` ablation without teaching
        the caller a second code path."""
        self.lexicon = lexicon
        self.limit = limit
        self.require_pattern = require_pattern

    # -- CandidateSource --------------------------------------------------- #

    def propose(self, requests: Sequence[ClueRequest]) -> dict[str, list[Candidate]]:
        """Propose pattern-consistent words for each request.

        With ``require_pattern`` (the default) an unconstrained request gets an
        empty list. A five-letter entry with no known letters matches tens of
        thousands of words: returning the 60 most frequent of them tells fusion
        nothing about *this* clue, but it does hand every entry a full slate of
        confident-looking wrong answers, which then compete with real LLM
        candidates and drag the null mass around. The lexicon only becomes
        informative once crossings have fixed at least one letter, so on the
        first round it stays quiet.
        """
        return {request.slot_id: self._propose_one(request) for request in requests}

    # -- internals --------------------------------------------------------- #

    def _propose_one(self, request: ClueRequest) -> list[Candidate]:
        if self.lexicon is None or self.limit <= 0:
            return []
        pattern = self._pattern_for(request)
        if pattern is None:
            return []

        raw = self.lexicon.match(pattern, self.limit)

        # The store is free to return anything; fusion trusts what we hand it,
        # so filter here rather than downstream.
        best: dict[str, float] = {}
        for word, weight in raw:
            answer = "".join(ch for ch in str(word).upper() if "A" <= ch <= "Z")
            if len(answer) != request.length or not pattern_matches(pattern, answer):
                continue
            value = float(weight)
            if answer not in best or value > best[answer]:
                best[answer] = value
        if not best:
            return []

        logs = {w: math.log(max(v, MIN_LEXICON_SCORE)) for w, v in best.items()}
        top = max(logs.values())
        candidates = [
            Candidate(
                answer=answer,
                score=MID_CONFIDENCE_SCORE + max(value - top, -MAX_SCORE_SPAN),
                source=self.name,
                rationale=f"fits {pattern}",
            )
            for answer, value in logs.items()
        ]
        candidates.sort(key=lambda c: (-c.score, c.answer))
        return candidates[: self.limit]

    def _pattern_for(self, request: ClueRequest) -> str | None:
        """The pattern to query with, or ``None`` if this request is too open."""
        pattern = request.pattern
        if pattern is not None and len(pattern) != request.length:
            # A wrong-length pattern is a caller bug; fall back to the length
            # rather than querying with a constraint we know is inconsistent.
            pattern = None
        known = 0 if pattern is None else sum(ch != WILDCARD for ch in pattern)
        if self.require_pattern and known == 0:
            return None
        return pattern if pattern is not None else WILDCARD * request.length


__all__ = [
    "LexiconCandidateSource",
    "LexiconLike",
    "MAX_SCORE_SPAN",
    "MID_CONFIDENCE_SCORE",
    "MIN_LEXICON_SCORE",
]

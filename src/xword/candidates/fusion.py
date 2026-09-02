"""Fuse several candidate sources into one calibrated distribution per entry.

Each source speaks its own dialect: the LLM emits log-probability-like scores
meaning "how well does this answer the clue", the lexicon emits frequency-like
scores meaning "how ordinary is this word". Fusion puts them on one scale,
rewards the cases where they independently agree, and -- the part that actually
decides whether the solver survives a hard puzzle -- says honestly how likely it
is that the right answer is in none of the lists at all.

Pooling rule
------------
``combined(answer) = logsumexp_s (weight_s + score_{s,answer})``

This is a *mixture* over sources, not a product. Additive weights in log space
are multiplicative confidence factors, and logsumexp is an OR: a source that
never proposed an answer contributes no term instead of vetoing it. A product
pool would let the lexicon -- which cannot know a proper noun -- zero out a
correct LLM answer, which is the opposite of what we want.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np

from xword.core.beliefs import MIN_NULL_MASS, SlotBeliefs
from xword.core.grid import pattern_matches
from xword.core.types import ALPHABET, LETTER_INDEX, WILDCARD, Candidate, ClueRequest


class LexiconLike(Protocol):
    """The slice of ``xword.lexicon.store.Lexicon`` used here.

    Structural, so fusion never imports the store; a plain object carrying
    these methods (or ``None``) is enough.
    """

    def score(self, word: str) -> float:
        ...

    def letters_at(self, pattern: str, position: int) -> dict[str, float]:
        ...


# --------------------------------------------------------------------------- #
# Tuning constants
# --------------------------------------------------------------------------- #

#: Source names beginning with this are treated as "knows what fits" rather
#: than "knows what the clue means". Only the latter feed the LLM confidence
#: terms in :func:`estimate_null_mass`.
LEXICON_SOURCE_PREFIX = "lexicon"

#: How many *distinct* sources must propose an answer before the agreement
#: bonus applies. Two independent lists landing on the same string is real
#: evidence; one list saying it twice is not.
AGREEMENT_MIN_SOURCES = 2

#: Log-bonus for an answer the lexicon recognises as a word. Deliberately much
#: smaller than ``agreement_bonus``: it breaks ties between a real word and a
#: plausible-looking hallucination without ever reordering what the sources
#: actually said.
LEXICON_PLAUSIBILITY_BONUS = 0.15

#: Scale on the "the LLM was not sure" hazard. At 0.6, a flat LLM list (top
#: probability ~0.3) contributes ~0.42 of null mass while a confident one
#: (top ~0.95) contributes ~0.03.
CONFIDENCE_HAZARD_SCALE = 0.6

#: A healthy LLM list has at least this many candidates. Below it the list is
#: too thin to have covered much of the answer space.
SPARSE_CANDIDATE_TARGET = 3

#: Entries up to this length are the bread and butter of both the model and the
#: word list. Beyond it, answers get rarer and more phrasal and both generators
#: cover them worse.
LENGTH_HAZARD_PIVOT = 7
LENGTH_HAZARD_PER_LETTER = 0.03
LENGTH_HAZARD_CAP = 0.35

#: Entry lengths sampled when building an empirical letter prior with no
#: specific length in mind.
PRIOR_SAMPLE_LENGTHS: tuple[int, ...] = (3, 4, 5, 6, 7, 8)

#: How much of the generic English prior to blend into an empirical one. Keeps
#: rare letters off zero: a zero-prior letter is an ``-inf`` message in belief
#: propagation and poisons every entry touching that cell.
LETTER_PRIOR_SMOOTHING = 0.02


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class FusionConfig:
    """Knobs for :func:`fuse`. Defaults are the ones the harness reports on."""

    source_weights: dict[str, float] = field(
        default_factory=lambda: {"llm": 1.0, "lexicon": 0.35}
    )
    temperature: float = 1.0
    max_candidates: int = 40
    null_mass_base: float = 0.02
    null_mass_per_missing: float = 0.25
    length_penalty: float = 6.0
    agreement_bonus: float = 0.5


# --------------------------------------------------------------------------- #
# Letter priors
# --------------------------------------------------------------------------- #

_ENGLISH_LETTER_FREQUENCY: dict[str, float] = {
    "A": 8.12, "B": 1.49, "C": 2.71, "D": 4.32, "E": 12.02, "F": 2.30,
    "G": 2.03, "H": 5.92, "I": 7.31, "J": 0.10, "K": 0.69, "L": 3.98,
    "M": 2.61, "N": 6.95, "O": 7.68, "P": 1.82, "Q": 0.11, "R": 6.02,
    "S": 6.28, "T": 9.10, "U": 2.88, "V": 1.11, "W": 2.09, "X": 0.17,
    "Y": 2.11, "Z": 0.07,
}

#: Background letter distribution, indexed the same way as ``LETTER_INDEX``.
#: Read-only: the solver keeps a reference for the life of a solve, so an
#: accidental in-place update would silently change every later message.
ENGLISH_LETTER_PRIOR: np.ndarray = np.array(
    [_ENGLISH_LETTER_FREQUENCY[ch] for ch in ALPHABET], dtype=np.float64
)
ENGLISH_LETTER_PRIOR /= ENGLISH_LETTER_PRIOR.sum()
ENGLISH_LETTER_PRIOR.flags.writeable = False


def letter_prior(
    lexicon: LexiconLike | None = None, length: int | None = None
) -> np.ndarray:
    """Background distribution over letters for one cell.

    Prefers the lexicon's own empirical frequencies, because crossword fill is
    not English prose -- it is vowel-heavy and stuffed with letters that cross
    well -- and falls back to :data:`ENGLISH_LETTER_PRIOR` when there is no
    lexicon or it cannot answer. ``length`` narrows the estimate to entries of
    that length; without it a spread of common lengths is averaged over every
    position, which is the right weighting for a *per-cell* prior.

    Always returns a fresh, writable, length-26 array summing to 1.
    """
    if lexicon is None:
        return np.array(ENGLISH_LETTER_PRIOR, dtype=np.float64)

    lengths = (length,) if length and length > 0 else PRIOR_SAMPLE_LENGTHS
    total = np.zeros(26, dtype=np.float64)
    for size in lengths:
        pattern = WILDCARD * size
        for position in range(size):
            try:
                dist = lexicon.letters_at(pattern, position)
            except (AttributeError, LookupError, NotImplementedError):
                # A lexicon that cannot answer is a reason to fall back, not a
                # reason to fail the solve.
                return np.array(ENGLISH_LETTER_PRIOR, dtype=np.float64)
            if not dist:
                continue
            column = np.zeros(26, dtype=np.float64)
            for letter, weight in dist.items():
                index = LETTER_INDEX.get(str(letter).upper())
                if index is not None and weight > 0:
                    column[index] += float(weight)
            column_total = float(column.sum())
            if column_total > 0:
                total += column / column_total

    if float(total.sum()) <= 0:
        return np.array(ENGLISH_LETTER_PRIOR, dtype=np.float64)
    empirical = total / total.sum()
    blended = (1.0 - LETTER_PRIOR_SMOOTHING) * empirical + (
        LETTER_PRIOR_SMOOTHING * np.asarray(ENGLISH_LETTER_PRIOR, dtype=np.float64)
    )
    return blended / blended.sum()


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def calibrate(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Re-temper a distribution: ``p ** (1/T)``, renormalised.

    ``T > 1`` flattens (the sources are overconfident), ``T < 1`` sharpens,
    ``T <= 0`` collapses onto the argmax. Done in log space so a long tail of
    tiny probabilities cannot underflow the whole array to zero.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.size == 0:
        return np.zeros(0, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    if total <= 0:
        return np.zeros_like(p)
    p = p / total
    if temperature == 1.0:
        return p
    if temperature <= 0:
        out = np.zeros_like(p)
        out[int(np.argmax(p))] = 1.0
        return out

    with np.errstate(divide="ignore"):
        log_p = np.log(p)
    scaled = log_p / temperature
    scaled = scaled - scaled[np.isfinite(scaled)].max()
    out = np.exp(scaled)
    out[~np.isfinite(log_p)] = 0.0
    out_total = float(out.sum())
    return out / out_total if out_total > 0 else p


# --------------------------------------------------------------------------- #
# Null mass
# --------------------------------------------------------------------------- #


def estimate_null_mass(
    candidates: Sequence[Candidate],
    *,
    request: ClueRequest,
    config: FusionConfig,
) -> float:
    """Probability that the true answer appears in none of the candidate lists.

    This is the solver's humility. Propagation, beam search and the repair loop
    all decide whether to trust an entry or let its crossings overrule it by
    reading this number, so it is estimated from what the *evidence* looks like
    rather than from how confident any single source claimed to be.

    Pass the raw per-source candidates, each still carrying its own ``source``:
    the terms below need to know who said what.

    Independent failure modes, combined as hazards
    (``null = 1 - prod(1 - h)``) so the result is bounded in ``[0, 1)`` by
    construction rather than by clipping a sum that ran off the end:

    ``base``
        ``null_mass_base``. Even a list that looks perfect is sometimes wrong,
        and a slot at exactly zero null mass is unfalsifiable.
    ``missing source``
        ``null_mass_per_missing`` for every source in ``source_weights`` that
        returned nothing here. Silence is not agreement: a source with no
        opinion also had no chance to cover the answer.
    ``low LLM confidence``
        ``CONFIDENCE_HAZARD_SCALE * (1 - top_p)``, ``top_p`` being the softmax
        of the LLM scores at its best answer. Scores are only comparable
        *within* a slot (see ``Candidate``), so peakedness inside the list is
        the only reading of confidence the contract permits.
    ``thin LLM list``
        Grows as the list falls below ``SPARSE_CANDIDATE_TARGET``, reaching
        ``null_mass_per_missing`` at zero -- which is why it only applies when
        the LLM said *something*: an empty LLM is already counted once as a
        missing source.
    ``long entry``
        ``LENGTH_HAZARD_PER_LETTER`` per letter beyond ``LENGTH_HAZARD_PIVOT``,
        capped at ``LENGTH_HAZARD_CAP``.

    Returns a value in ``[MIN_NULL_MASS, 1.0]``; an empty list returns 1.0.
    """
    if not candidates:
        return 1.0

    hazards: list[float] = [float(config.null_mass_base)]

    present = {c.source for c in candidates}
    knowledge_gap_charged = False
    for name, weight in config.source_weights.items():
        if weight <= 0.0:
            continue
        if not any(_source_matches(name, seen) for seen in present):
            hazards.append(float(config.null_mass_per_missing))
            knowledge_gap_charged = knowledge_gap_charged or not _is_lexicon_source(name)

    knowledge = [c for c in candidates if not _is_lexicon_source(c.source)]
    if not knowledge and not knowledge_gap_charged:
        # Nothing that understands clues answered here, and no configured
        # source was charged for it -- the lexicon-only ablation, where the
        # model is switched off rather than silent. A list of words that merely
        # *fit* is not evidence about what the clue means, so it gets the same
        # hazard a silent model would have. Charged in the ``elif`` position so
        # a model that was configured but returned nothing is not billed twice.
        hazards.append(float(config.null_mass_per_missing))
    if knowledge:
        probabilities = _softmax(
            np.array([c.score for c in knowledge], dtype=np.float64)
        )
        top_p = float(probabilities.max())
        hazards.append(CONFIDENCE_HAZARD_SCALE * (1.0 - top_p))
        if len(knowledge) < SPARSE_CANDIDATE_TARGET:
            shortfall = (
                SPARSE_CANDIDATE_TARGET - len(knowledge)
            ) / SPARSE_CANDIDATE_TARGET
            hazards.append(float(config.null_mass_per_missing) * shortfall)

    length = request.length if request.length > 0 else len(candidates[0].answer)
    hazards.append(
        min(
            LENGTH_HAZARD_CAP,
            LENGTH_HAZARD_PER_LETTER * max(0, length - LENGTH_HAZARD_PIVOT),
        )
    )

    survival = 1.0
    for hazard in hazards:
        survival *= 1.0 - min(max(hazard, 0.0), 1.0)
    return min(max(1.0 - survival, MIN_NULL_MASS), 1.0)


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #


def fuse(
    per_source: Mapping[str, Mapping[str, list[Candidate]]],
    requests: Sequence[ClueRequest],
    *,
    config: FusionConfig | None = None,
    lexicon: LexiconLike | None = None,
) -> SlotBeliefs:
    """Pool ``source -> slot_id -> candidates`` into one :class:`SlotBeliefs`.

    Every requested ``slot_id`` appears in the result, including entries no
    source had an opinion about -- those get an empty candidate list and a null
    mass of 1.0, which is how the solver learns to lean entirely on crossings
    there. ``lexicon``, when given, only supplies the small plausibility bonus
    of :data:`LEXICON_PLAUSIBILITY_BONUS`; *generating* lexicon candidates is
    :class:`~xword.candidates.lexicon_source.LexiconCandidateSource`'s job.
    """
    config = config or FusionConfig()
    plausibility = _Plausibility(lexicon)
    beliefs = SlotBeliefs()

    for request in requests:
        evidence = _collect(per_source, request)
        candidates, probabilities, kept_mass = _pool(
            evidence, request, config, plausibility
        )

        null_mass = estimate_null_mass(evidence, request=request, config=config)
        if kept_mass < 1.0:
            # Mass dropped by the ``max_candidates`` cut did not evaporate; it
            # belongs to answers we no longer track, which is exactly what null
            # mass means.
            null_mass = 1.0 - (1.0 - null_mass) * kept_mass

        length = request.length if request.length > 0 else _modal_length(evidence)
        beliefs.set_slot(request.slot_id, candidates, probabilities, null_mass, length)

    return beliefs


def _collect(
    per_source: Mapping[str, Mapping[str, list[Candidate]]], request: ClueRequest
) -> list[Candidate]:
    """Normalised, culled candidates for one slot, tagged with the source *key*.

    The key of ``per_source`` is authoritative for a candidate's identity, not
    the ``source`` field the generator happened to stamp on it: the key is what
    ``source_weights`` is written against, and fusion must not weight an answer
    under a name the caller never registered.
    """
    kept: list[Candidate] = []
    for source, by_slot in per_source.items():
        best: dict[str, Candidate] = {}
        for cand in by_slot.get(request.slot_id) or ():
            answer = _normalise_answer(cand.answer)
            if not answer or not _admissible(answer, request):
                continue
            previous = best.get(answer)
            if previous is None or cand.score > previous.score:
                best[answer] = replace(cand, answer=answer, source=source)
        kept.extend(best.values())
    return kept


def _pool(
    evidence: Sequence[Candidate],
    request: ClueRequest,
    config: FusionConfig,
    plausibility: _Plausibility,
) -> tuple[list[Candidate], np.ndarray, float]:
    """The pool itself. Returns ``(candidates, probabilities, kept_mass)``."""
    if not evidence:
        return [], np.zeros(0, dtype=np.float64), 1.0

    by_answer: dict[str, list[Candidate]] = {}
    for cand in evidence:
        by_answer.setdefault(cand.answer, []).append(cand)

    expected_length = request.length if request.length > 0 else _modal_length(evidence)

    scores: dict[str, float] = {}
    for answer, contributors in by_answer.items():
        combined = _logsumexp(
            [
                _weight_for(c.source, config.source_weights) + c.score
                for c in contributors
            ]
        )
        if len({c.source for c in contributors}) >= AGREEMENT_MIN_SOURCES:
            combined += config.agreement_bonus
        combined += plausibility.bonus(answer)
        if len(answer) != expected_length:
            # Unreachable whenever the request declares a length -- those are
            # culled in ``_admissible``. It matters only for requests that do
            # not, where ranking a wrong-length answer last is all we can do.
            combined -= config.length_penalty
        scores[answer] = combined

    # Ties broken alphabetically so a rerun with the same inputs produces the
    # same order.
    order = sorted(by_answer, key=lambda a: (-scores[a], a))
    full = calibrate(
        _softmax(np.array([scores[a] for a in order], dtype=np.float64)),
        config.temperature,
    )

    limit = config.max_candidates if config.max_candidates > 0 else len(order)
    keep = order[:limit]
    probabilities = full[: len(keep)]
    kept_mass = float(probabilities.sum())

    candidates = [
        Candidate(
            answer=answer,
            score=scores[answer],
            source=_source_label(by_answer[answer], config),
            rationale=_best_rationale(by_answer[answer], config),
        )
        for answer in keep
    ]
    return candidates, probabilities, kept_mass


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _Plausibility:
    """Small, capped "is this even a word" bonus drawn from the lexicon."""

    __slots__ = ("_lexicon",)

    def __init__(self, lexicon: LexiconLike | None) -> None:
        self._lexicon = lexicon

    def bonus(self, answer: str) -> float:
        if self._lexicon is None:
            return 0.0
        try:
            known = float(self._lexicon.score(answer)) > 0.0
        except (AttributeError, LookupError, NotImplementedError):
            # A lexicon that cannot score a word contributes nothing rather
            # than failing the solve.
            return 0.0
        return LEXICON_PLAUSIBILITY_BONUS if known else 0.0


def _normalise_answer(text: str) -> str:
    """Upper-case A-Z only. Models return ``"St. Louis"``; grids hold ``STLOUIS``."""
    return "".join(ch for ch in str(text).upper() if "A" <= ch <= "Z")


def _admissible(answer: str, request: ClueRequest) -> bool:
    if request.length > 0 and len(answer) != request.length:
        return False
    pattern = request.pattern
    if not pattern or len(pattern) != len(answer):
        # A pattern whose length disagrees with the answer is a caller bug rather
        # than evidence against the answer, so it is ignored instead of emptying
        # the entry.
        return True
    return pattern_matches(pattern, answer)


def _source_matches(configured: str, observed: str) -> bool:
    """``"llm"`` in the weight table covers ``"llm-opus"`` and ``"llm:hard"``."""
    return observed == configured or observed.startswith(configured)


def _is_lexicon_source(name: str) -> bool:
    return name.startswith(LEXICON_SOURCE_PREFIX)


def _weight_for(source: str, weights: Mapping[str, float]) -> float:
    """Weight for a source name; the longest matching configured prefix wins.

    An unregistered source gets the *weakest* configured weight: someone who
    plugs in a new generator without declaring it should not have it silently
    outrank the generators that were declared.
    """
    if source in weights:
        return float(weights[source])
    matches = [key for key in weights if _source_matches(key, source)]
    if matches:
        return float(weights[max(matches, key=len)])
    return float(min(weights.values())) if weights else 0.0


def _source_label(contributors: Sequence[Candidate], config: FusionConfig) -> str:
    """``"llm"``, or ``"llm+lexicon"`` when sources agreed. Heaviest first."""
    names = sorted(
        {c.source for c in contributors},
        key=lambda n: (-_weight_for(n, config.source_weights), n),
    )
    return "+".join(names)


def _best_rationale(contributors: Sequence[Candidate], config: FusionConfig) -> str:
    ranked = sorted(
        contributors,
        key=lambda c: -(_weight_for(c.source, config.source_weights) + c.score),
    )
    for cand in ranked:
        if cand.rationale:
            return cand.rationale
    return ""


def _modal_length(evidence: Sequence[Candidate]) -> int:
    """Most common answer length, ties going to the shorter one."""
    if not evidence:
        return 0
    counts = Counter(len(c.answer) for c in evidence)
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return -math.inf
    top = max(values)
    if math.isinf(top):
        return top
    return top + math.log(sum(math.exp(v - top) for v in values))


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return np.zeros(0, dtype=np.float64)
    shifted = np.exp(scores - scores.max())
    total = float(shifted.sum())
    if total <= 0:
        return np.full(scores.shape, 1.0 / scores.size, dtype=np.float64)
    return shifted / total


__all__ = [
    "AGREEMENT_MIN_SOURCES",
    "CONFIDENCE_HAZARD_SCALE",
    "ENGLISH_LETTER_PRIOR",
    "FusionConfig",
    "LENGTH_HAZARD_CAP",
    "LENGTH_HAZARD_PER_LETTER",
    "LENGTH_HAZARD_PIVOT",
    "LETTER_PRIOR_SMOOTHING",
    "LEXICON_PLAUSIBILITY_BONUS",
    "LEXICON_SOURCE_PREFIX",
    "LexiconLike",
    "PRIOR_SAMPLE_LENGTHS",
    "SPARSE_CANDIDATE_TARGET",
    "calibrate",
    "estimate_null_mass",
    "fuse",
    "letter_prior",
]

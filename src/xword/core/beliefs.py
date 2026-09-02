"""The belief container passed between candidate generation and the solver.

Kept in ``core`` rather than in either neighbour because both sides need it:
``xword.candidates.fusion`` builds it, ``xword.solver`` consumes it. Keeping it
here is what stops the two packages importing each other.

A :class:`SlotBeliefs` is a *distribution over answers per entry*, plus an
explicit "none of the above" mass. That last piece matters more than it looks:
a solver that assumes the right answer is always somewhere in its candidate
list will happily force a wrong word into the grid and then corrupt every
crossing entry. Reserving probability for "my list is wrong here" is what lets
belief propagation prefer the crossings' opinion over a confident-but-wrong
guess.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from xword.core.types import Candidate

#: Floor on the probability mass reserved for "the answer is not in my list".
#: Never zero: a source that claims certainty on every clue makes the whole
#: grid brittle.
MIN_NULL_MASS = 1e-3


@dataclass(slots=True)
class SlotBeliefs:
    """Per-entry answer distributions.

    Attributes
    ----------
    candidates:
        ``slot_id -> [Candidate, ...]``, ranked best-first. Every answer is
        upper-case A-Z and has exactly the length of its entry.
    priors:
        ``slot_id -> float array`` parallel to ``candidates[slot_id]``. Each
        array is non-negative and sums to ``1 - null_mass[slot_id]``.
    null_mass:
        ``slot_id -> float`` in ``[MIN_NULL_MASS, 1]``: probability that the
        true answer appears nowhere in the candidate list.
    lengths:
        ``slot_id -> int``, carried so the solver can sanity-check without the
        puzzle in hand.
    """

    candidates: dict[str, list[Candidate]] = field(default_factory=dict)
    priors: dict[str, np.ndarray] = field(default_factory=dict)
    null_mass: dict[str, float] = field(default_factory=dict)
    lengths: dict[str, int] = field(default_factory=dict)

    # -- accessors --------------------------------------------------------- #

    @property
    def slot_ids(self) -> tuple[str, ...]:
        return tuple(self.candidates.keys())

    def answers(self, slot_id: str) -> list[str]:
        return [c.answer for c in self.candidates.get(slot_id, [])]

    def prior(self, slot_id: str) -> np.ndarray:
        got = self.priors.get(slot_id)
        if got is None:
            return np.zeros(0, dtype=np.float64)
        return got

    def top(self, slot_id: str) -> tuple[str, float] | None:
        """Best candidate and its prior probability, or ``None`` if the list is
        empty."""
        cands = self.candidates.get(slot_id) or []
        if not cands:
            return None
        prior = self.prior(slot_id)
        if prior.size == 0:
            return cands[0].answer, 0.0
        best = int(np.argmax(prior))
        return cands[best].answer, float(prior[best])

    def coverage(self, gold: Mapping[str, str]) -> float:
        """Fraction of entries whose gold answer is somewhere in the candidate
        list. This is the ceiling on what any downstream search can achieve, and
        is reported separately from accuracy so a failure can be attributed to
        generation rather than to search."""
        if not gold:
            return 0.0
        hits = sum(
            1 for sid, answer in gold.items() if answer in set(self.answers(sid))
        )
        return hits / len(gold)

    # -- construction ------------------------------------------------------ #

    def set_slot(
        self,
        slot_id: str,
        candidates: list[Candidate],
        probabilities: np.ndarray,
        null_mass: float,
        length: int,
    ) -> None:
        """Install one entry's distribution, normalising defensively."""
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.shape != (len(candidates),):
            raise ValueError(
                f"{slot_id}: {len(candidates)} candidates but "
                f"{probabilities.shape} probabilities"
            )
        null_mass = float(min(max(null_mass, MIN_NULL_MASS), 1.0))
        total = float(probabilities.sum())
        if total > 0:
            probabilities = probabilities * ((1.0 - null_mass) / total)
        else:
            null_mass = 1.0
            probabilities = np.zeros(len(candidates), dtype=np.float64)

        self.candidates[slot_id] = candidates
        self.priors[slot_id] = probabilities
        self.null_mass[slot_id] = null_mass
        self.lengths[slot_id] = length

    def merged_with(self, other: SlotBeliefs) -> SlotBeliefs:
        """``other`` wins where both define a slot. Used when a repair round
        re-generates candidates for a subset of entries."""
        out = SlotBeliefs(
            candidates=dict(self.candidates),
            priors=dict(self.priors),
            null_mass=dict(self.null_mass),
            lengths=dict(self.lengths),
        )
        out.candidates.update(other.candidates)
        out.priors.update(other.priors)
        out.null_mass.update(other.null_mass)
        out.lengths.update(other.lengths)
        return out


__all__ = ["SlotBeliefs", "MIN_NULL_MASS"]

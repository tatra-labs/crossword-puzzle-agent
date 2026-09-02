"""Tests for the :class:`~xword.core.beliefs.SlotBeliefs` container itself.

Only the container -- how it normalises, merges and reports. Whatever produces
the numbers (``xword.candidates.fusion``) is tested separately; the point of
these tests is that no matter how bad a source's numbers are, what reaches the
solver is a real distribution with a non-zero "none of the above" mass.
"""

from __future__ import annotations

import numpy as np
import pytest

from xword.core.beliefs import MIN_NULL_MASS, SlotBeliefs
from xword.core.types import Candidate


def _cands(*answers: str, source: str = "test") -> list[Candidate]:
    return [Candidate(answer=a, score=0.0, source=source) for a in answers]


# --------------------------------------------------------------------------- #
# set_slot: normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "answers, probs, null_mass",
    [
        (("ADO",), [1.0], 0.1),
        (("ADO", "AGO"), [0.5, 0.5], MIN_NULL_MASS),
        (("ADO", "AGO"), [0.5, 0.5], 0.9),
        # Unnormalised input: the caller's numbers only have to be relative.
        (("ADO", "AGO", "APE"), [3.0, 2.0, 1.0], 0.25),
        (("ADO", "AGO", "APE"), [30.0, 20.0, 10.0], 0.25),
        # Tiny masses must not underflow the rescale.
        (("ADO", "AGO"), [1e-9, 1e-12], 0.5),
        # Degenerate but legal: nothing proposed at all.
        ((), [], 0.3),
        # Everything ruled out by the crossings.
        (("ADO", "AGO"), [0.0, 0.0], 0.4),
    ],
)
def test_set_slot_normalises_to_one(
    answers: tuple[str, ...], probs: list[float], null_mass: float
) -> None:
    beliefs = SlotBeliefs()
    beliefs.set_slot(
        "1A", _cands(*answers), np.array(probs, dtype=float), null_mass, 3
    )

    total = float(beliefs.priors["1A"].sum()) + beliefs.null_mass["1A"]
    assert total == pytest.approx(1.0)
    assert np.all(beliefs.priors["1A"] >= 0.0)
    assert beliefs.priors["1A"].shape == (len(answers),)
    assert beliefs.lengths["1A"] == 3


def test_set_slot_preserves_relative_weights() -> None:
    beliefs = SlotBeliefs()
    beliefs.set_slot("1A", _cands("A", "B", "C"), np.array([3.0, 2.0, 1.0]), 0.4, 1)

    prior = beliefs.priors["1A"]
    assert prior[0] / prior[2] == pytest.approx(3.0)
    assert float(prior.sum()) == pytest.approx(0.6)


def test_set_slot_with_all_zero_probabilities_hands_everything_to_null() -> None:
    """A source that ranked nothing above zero is a source with no opinion,
    and pretending otherwise would let the solver commit to arbitrary junk."""
    beliefs = SlotBeliefs()
    beliefs.set_slot("1A", _cands("ADO", "AGO"), np.zeros(2), 0.2, 3)

    assert beliefs.null_mass["1A"] == 1.0
    assert np.array_equal(beliefs.priors["1A"], np.zeros(2))


def test_set_slot_accepts_a_plain_list_of_probabilities() -> None:
    beliefs = SlotBeliefs()
    probs: list[float] = [0.6, 0.4]
    beliefs.set_slot("1A", _cands("ADO", "AGO"), probs, 0.1, 3)  # type: ignore[arg-type]
    assert beliefs.priors["1A"].dtype == np.float64
    assert float(beliefs.priors["1A"].sum()) == pytest.approx(0.9)


@pytest.mark.parametrize(
    "n_candidates, probs",
    [
        (2, [0.5, 0.3, 0.2]),  # too many probabilities
        (3, [0.5, 0.5]),  # too few
        (0, [1.0]),  # none expected, one given
        (1, []),  # one expected, none given
        (2, [[0.5], [0.5]]),  # right count, wrong shape
    ],
)
def test_set_slot_rejects_a_length_mismatch(
    n_candidates: int, probs: list[object]
) -> None:
    beliefs = SlotBeliefs()
    candidates = _cands(*[f"W{i}" for i in range(n_candidates)])
    with pytest.raises(ValueError, match="1A"):
        beliefs.set_slot("1A", candidates, np.array(probs, dtype=float), 0.1, 2)
    assert beliefs.slot_ids == ()


@pytest.mark.parametrize(
    "given, expected",
    [
        (0.0, MIN_NULL_MASS),
        (-1.0, MIN_NULL_MASS),
        (MIN_NULL_MASS / 2, MIN_NULL_MASS),
        (MIN_NULL_MASS, MIN_NULL_MASS),
        (0.5, 0.5),
        (1.0, 1.0),
        (5.0, 1.0),
    ],
)
def test_null_mass_is_clamped(given: float, expected: float) -> None:
    beliefs = SlotBeliefs()
    beliefs.set_slot("1A", _cands("ADO"), np.array([1.0]), given, 3)
    assert beliefs.null_mass["1A"] == pytest.approx(expected)


def test_set_slot_replaces_a_previous_distribution() -> None:
    beliefs = SlotBeliefs()
    beliefs.set_slot("1A", _cands("ADO", "AGO"), np.array([0.5, 0.5]), 0.1, 3)
    beliefs.set_slot("1A", _cands("APE"), np.array([1.0]), 0.2, 3)

    assert beliefs.answers("1A") == ["APE"]
    assert beliefs.priors["1A"].shape == (1,)
    assert beliefs.null_mass["1A"] == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# Accessors
# --------------------------------------------------------------------------- #


def test_top_returns_none_for_an_empty_slot(make_beliefs) -> None:
    beliefs = make_beliefs({"1A": []})
    assert "1A" in beliefs.slot_ids
    assert beliefs.top("1A") is None


def test_top_returns_none_for_an_unknown_slot(make_beliefs) -> None:
    assert make_beliefs({"1A": [("ADO", 1.0)]}).top("9Z") is None


def test_top_picks_the_argmax_not_the_first_entry(make_beliefs) -> None:
    """The candidate list is nominally ranked, but fusion can reorder the
    probabilities without reordering the list, so ``top`` must not assume
    index 0 wins."""
    beliefs = make_beliefs({"1A": [("ADO", 0.1), ("AGO", 0.2), ("APE", 0.7)]})
    answer, prob = beliefs.top("1A")
    assert answer == "APE"
    assert prob == pytest.approx(0.7 * (1.0 - MIN_NULL_MASS))


def test_top_falls_back_to_the_first_candidate_without_priors() -> None:
    """A container assembled by hand may carry candidates but no priors."""
    beliefs = SlotBeliefs(candidates={"1A": _cands("ADO", "AGO")})
    assert beliefs.top("1A") == ("ADO", 0.0)


def test_answers_and_prior_are_safe_on_an_unknown_slot() -> None:
    beliefs = SlotBeliefs()
    assert beliefs.answers("1A") == []
    assert beliefs.prior("1A").shape == (0,)


def test_slot_ids_follow_insertion_order(make_beliefs) -> None:
    beliefs = make_beliefs({"5A": [("SHORE", 1.0)], "1A": [("ADO", 1.0)]})
    assert beliefs.slot_ids == ("5A", "1A")


# --------------------------------------------------------------------------- #
# coverage
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "gold, expected",
    [
        ({"1A": "ADO", "5A": "SHORE"}, 1.0),
        ({"1A": "ADO"}, 1.0),
        ({"1A": "AGO", "5A": "SHORE"}, 1.0),  # second-choice answers still count
        ({"1A": "XXX", "5A": "SHORE"}, 0.5),
        ({"1A": "XXX", "5A": "YYYYY"}, 0.0),
        ({"1A": "ADO", "9Z": "MISSING"}, 0.5),  # slot absent from the beliefs
        ({}, 0.0),  # no gold at all: report 0, do not divide by zero
    ],
)
def test_coverage(make_beliefs, gold: dict[str, str], expected: float) -> None:
    beliefs = make_beliefs(
        {"1A": [("ADO", 0.6), ("AGO", 0.4)], "5A": [("SHORE", 1.0)]}
    )
    assert beliefs.coverage(gold) == pytest.approx(expected)


def test_coverage_ignores_probability_mass(make_beliefs) -> None:
    """Coverage is the ceiling on search, so a candidate that is present but
    hopeless still counts."""
    beliefs = make_beliefs({"1A": [("AGO", 1.0), ("ADO", 0.0)]})
    assert beliefs.coverage({"1A": "ADO"}) == pytest.approx(1.0)


def test_coverage_of_the_real_solution(mini_puzzle, make_beliefs) -> None:
    solution = dict(mini_puzzle.solution or {})
    spec = {sid: [(answer, 1.0)] for sid, answer in solution.items()}
    del spec["6A"]
    beliefs = make_beliefs(spec)
    assert beliefs.coverage(solution) == pytest.approx(9 / 10)


# --------------------------------------------------------------------------- #
# merged_with
# --------------------------------------------------------------------------- #


def test_merged_with_gives_precedence_to_the_argument(make_beliefs) -> None:
    base = make_beliefs({"1A": [("ADO", 1.0)], "5A": [("SHORE", 1.0)]})
    repair = make_beliefs({"1A": [("AGO", 1.0)]}, null_mass=0.4)

    merged = base.merged_with(repair)

    assert merged.answers("1A") == ["AGO"]
    assert merged.answers("5A") == ["SHORE"]
    assert merged.null_mass["1A"] == pytest.approx(0.4)
    assert merged.null_mass["5A"] == pytest.approx(MIN_NULL_MASS)


def test_merged_with_mutates_neither_input(make_beliefs) -> None:
    base = make_beliefs({"1A": [("ADO", 1.0)]})
    repair = make_beliefs({"1A": [("AGO", 1.0)], "5A": [("SHORE", 1.0)]})

    merged = base.merged_with(repair)
    merged.set_slot("7A", _cands("YEN"), np.array([1.0]), 0.1, 3)

    assert base.slot_ids == ("1A",)
    assert base.answers("1A") == ["ADO"]
    assert repair.slot_ids == ("1A", "5A")
    assert repair.answers("1A") == ["AGO"]
    assert merged.slot_ids == ("1A", "5A", "7A")


def test_merged_with_carries_lengths_and_priors(make_beliefs) -> None:
    base = make_beliefs({"1A": [("ADO", 1.0)]}, lengths={"1A": 3})
    repair = make_beliefs({"5A": [("SHORE", 1.0)]}, lengths={"5A": 5})

    merged = base.merged_with(repair)

    assert merged.lengths == {"1A": 3, "5A": 5}
    assert set(merged.priors) == {"1A", "5A"}
    assert float(merged.prior("5A").sum()) == pytest.approx(1.0 - MIN_NULL_MASS)


def test_merging_an_empty_container_is_a_no_op(make_beliefs) -> None:
    base = make_beliefs({"1A": [("ADO", 0.7), ("AGO", 0.3)]})
    merged = base.merged_with(SlotBeliefs())

    assert merged.slot_ids == base.slot_ids
    assert merged.answers("1A") == base.answers("1A")
    assert np.allclose(merged.prior("1A"), base.prior("1A"))


# --------------------------------------------------------------------------- #
# The factory fixture other test modules build on
# --------------------------------------------------------------------------- #


def test_make_beliefs_normalises_through_set_slot(make_beliefs) -> None:
    beliefs = make_beliefs({"1A": [("ADO", 6.0), ("AGO", 4.0)]}, null_mass=0.2)
    assert float(beliefs.prior("1A").sum()) == pytest.approx(0.8)
    assert beliefs.lengths["1A"] == 3


def test_make_beliefs_scores_are_log_probabilities(make_beliefs) -> None:
    beliefs = make_beliefs({"1A": [("ADO", 0.5), ("AGO", 0.25)]})
    scores = [c.score for c in beliefs.candidates["1A"]]
    assert scores[0] > scores[1]
    assert scores[0] == pytest.approx(np.log(0.5))

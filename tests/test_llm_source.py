"""Tests for :mod:`xword.candidates.llm`, the one module that spends money.

Nothing here touches the network. Every test drives either
:class:`~xword.candidates.llm.FakeClient` or a local stub that records the
kwargs it was handed, which is what makes the two things worth asserting about
this module assertable at all:

*The record is the only account of a call.* ``on_call`` fires from inside
``_call``, the choke point every request and every retry passes through, and in
a web session that record is the whole of what the trace panel and
``/api/sessions/{sid}/events`` can ever show. So the tests check the record
against what the client was actually sent -- the prompt as sent, one record per
call, the retries named -- rather than against the arguments that built it.

*The payload has to fit the installed SDK.* ``_request_kwargs`` is a dict handed
straight to ``client.messages.create``, so a parameter the SDK has dropped is
not a warning or a 400 but a local ``TypeError`` that fails every call on the
affected model while leaving the others working. That is how the ``temperature``
regression hid, so one test here compares the payload's keys against the real
signature of the installed ``Messages.create`` for every model the UI offers,
and will keep doing so across SDK upgrades.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from xword.candidates import llm as llm_module
from xword.candidates.cache import ClueCache
from xword.candidates.llm import FakeClient, LLMCandidateSource
from xword.core.types import ClueRequest
from xword.web.trace import LLMCallRecord

#: The models ``public/index.html``'s selector offers. Written out here rather
#: than scraped from the page so this file stays a unit test, but kept in sync
#: deliberately: three of these four were sending a parameter the installed SDK
#: no longer accepts, and the default one was not, which is precisely why a
#: single-model check would have passed through the regression.
OFFERED_MODELS: tuple[str, ...] = (
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-opus-5",
)

ANSWER_BOOK: dict[str, list[tuple[str, float]]] = {
    "Capital of France": [("PARIS", 0.9), ("LYONS", 0.05)],
    "Tokyo currency": [("YEN", 0.95)],
}


def _requests() -> list[ClueRequest]:
    """Two clues the answer book knows, in one batch."""
    return [
        ClueRequest(slot_id="1A", clue="Capital of France", length=5, direction="across"),
        ClueRequest(slot_id="7A", clue="Tokyo currency", length=3, direction="across"),
    ]


def _source(**kwargs: Any) -> LLMCandidateSource:
    """A source with the cache off and retries instant.

    ``cache=None`` by default because most of these tests are about what goes
    over the wire, and a warm cache is precisely the thing that stops anything
    going over it. ``_sleep_for`` is neutered on the instance rather than
    globally: the backoff shape has its own arithmetic and is not what is under
    test here, and paying the real 2.4s of it would make the retry tests the
    slowest in the suite.
    """
    kwargs.setdefault("cache", None)
    kwargs.setdefault("max_concurrency", 1)
    source = LLMCandidateSource(**kwargs)
    source._sleep_for = lambda attempt: 0.0  # type: ignore[method-assign]
    return source


# --------------------------------------------------------------------------- #
# One record per call, describing the request that went out
# --------------------------------------------------------------------------- #


def test_a_record_is_emitted_per_call_carrying_the_prompt_as_sent() -> None:
    """The record's prompt is the payload's, not a re-derivation of it."""
    records: list[LLMCallRecord] = []
    client = FakeClient(ANSWER_BOOK)
    source = _source(model="claude-haiku-4-5", client=client, on_call=records.append)

    results = source.propose(_requests())

    assert [c.answer for c in results["1A"]][:1] == ["PARIS"]
    assert len(client.calls) == 1, "two clues in one batch is one call"
    assert len(records) == 1
    record = records[0]
    # The prompt as sent, byte for byte: the record is read off the payload, so
    # a prompt builder change shows up here rather than being papered over.
    assert record.prompt == client.calls[0]["messages"][0]["content"]
    assert record.system == client.calls[0]["system"][0]["text"]
    assert record.model == "claude-haiku-4-5"
    assert record.kind == "batch"
    assert record.tools == ("submit_answers",)
    assert record.tool_choice == "tool:submit_answers"
    assert record.clue_ids == ("1A", "7A")
    assert (record.attempts, record.error, record.retry_errors) == (1, "", ())
    assert record.output_tokens > 0


def test_each_batch_gets_its_own_record() -> None:
    """Two batches are two records, each naming only its own clues."""
    records: list[LLMCallRecord] = []
    source = _source(
        model="claude-haiku-4-5",
        client=FakeClient(ANSWER_BOOK),
        batch_size=1,
        on_call=records.append,
    )

    source.propose(_requests())

    assert sorted(r.clue_ids for r in records) == [("1A",), ("7A",)]


# --------------------------------------------------------------------------- #
# Retries
# --------------------------------------------------------------------------- #


def test_a_retried_then_succeeded_call_is_one_record_naming_what_failed() -> None:
    """``attempts=3`` with no error, but the two failures are still on record.

    The regression this covers is a silent one: the narration naming the
    exception goes to ``on_event``, which the agent never wires, so before
    ``retry_errors`` existed a web session showed a slow three-attempt call with
    no error chip and nothing anywhere saying what it had been waiting on.
    """
    records: list[LLMCallRecord] = []
    source = _source(
        model="claude-haiku-4-5",
        client=FakeClient(ANSWER_BOOK, fail_times=2),
        on_call=records.append,
    )

    results = source.propose(_requests())

    assert [c.answer for c in results["1A"]][:1] == ["PARIS"], "the third attempt worked"
    assert len(records) == 1, "one logical request is one record, not three"
    record = records[0]
    assert record.attempts == 3
    assert record.error == "", "it succeeded; an error chip here would be a lie"
    assert len(record.retry_errors) == 2
    assert all("ConnectionError" in line for line in record.retry_errors)
    assert all("simulated transient failure" in line for line in record.retry_errors)
    # The trace log stores the dict, so the reason has to survive that too.
    assert record.as_dict()["retry_errors"] == list(record.retry_errors)
    # Accounting is unchanged: two retries, one call, no failure.
    assert (source.usage.calls, source.usage.retries, source.usage.failures) == (1, 2, 0)


def test_a_given_up_call_separates_the_final_failure_from_the_earlier_ones() -> None:
    """``error`` is the outcome; ``retry_errors`` is the history, without overlap."""
    records: list[LLMCallRecord] = []
    source = _source(
        model="claude-haiku-4-5",
        client=FakeClient(ANSWER_BOOK, fail_times=99),
        max_retries=2,
        on_call=records.append,
    )

    results = source.propose(_requests())

    assert results == {"1A": [], "7A": []}, "a dead API costs the slots their candidates"
    assert len(records) == 1
    record = records[0]
    assert record.attempts == 3
    assert "ConnectionError" in record.error
    assert len(record.retry_errors) == 2, "the attempt in `error` is not repeated here"
    assert (source.usage.calls, source.usage.failures) == (0, 1)


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


class _ForbiddenClient:
    """A client that fails the test if anything asks it for a completion."""

    def __init__(self) -> None:
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        raise AssertionError("a cached clue must not reach the API")


def test_a_cache_hit_emits_a_cache_record_and_makes_no_call(tmp_path: Path) -> None:
    """The warm run's trace says why it is empty instead of just being empty."""
    cache = ClueCache(tmp_path / "clue-cache.sqlite")
    try:
        warm = _source(model="claude-haiku-4-5", client=FakeClient(ANSWER_BOOK), cache=cache)
        warm.propose(_requests())

        records: list[LLMCallRecord] = []
        replay = _source(
            model="claude-haiku-4-5",
            client=_ForbiddenClient(),
            cache=cache,
            on_call=records.append,
        )
        results = replay.propose(_requests())
    finally:
        cache.close()

    assert [c.answer for c in results["1A"]][:1] == ["PARIS"]
    assert len(records) == 1, "one record for the whole cached subset, not one per clue"
    record = records[0]
    assert record.cached is True
    assert record.kind == "cache"
    assert sorted(record.clue_ids) == ["1A", "7A"]
    assert (record.prompt, record.system) == ("", ""), "nothing was sent"
    assert (replay.usage.calls, replay.usage.cache_hits) == (0, 2)


# --------------------------------------------------------------------------- #
# Payload conformance
# --------------------------------------------------------------------------- #


def _payload(model: str, *, hard: bool, temperature: float = 0.4) -> dict[str, Any]:
    source = LLMCandidateSource(model=model, temperature=temperature, client=_ForbiddenClient())
    return source._request_kwargs(system="sys", user="1A | len 5 | pat ????? | Clue", hard=hard)


@pytest.mark.parametrize("model", OFFERED_MODELS)
@pytest.mark.parametrize("hard", [False, True])
def test_every_offered_model_sends_only_kwargs_the_installed_sdk_accepts(
    model: str, hard: bool
) -> None:
    """The payload is checked against the SDK that will receive it.

    ``_request_kwargs`` output goes straight into ``client.messages.create``, so
    a key the installed SDK has dropped raises ``TypeError`` before a request is
    made -- no 400, no retry, just every call on that model failing while the
    others work. Asserting against the live signature rather than a hard-coded
    list is what makes this test survive the next SDK release: it fails on the
    version that removes a parameter, not on the version after someone
    remembered to update a fixture.
    """
    from anthropic.resources.messages import Messages

    accepted = set(inspect.signature(Messages.create).parameters)
    unsupported = set(_payload(model, hard=hard)) - accepted
    assert not unsupported, f"{model} would be sent {sorted(unsupported)}"


def test_temperature_is_omitted_when_the_sdk_cannot_take_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probe says no -> no temperature, even for a model that accepts sampling."""
    monkeypatch.setattr(llm_module, "_sdk_accepts_temperature", lambda: False)

    for model in OFFERED_MODELS:
        assert "temperature" not in _payload(model, hard=False)
        assert "temperature" not in _payload(model, hard=True)


def test_temperature_is_sent_only_where_both_the_sdk_and_the_model_allow_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two reasons are independent, so an older SDK keeps working.

    ``pyproject`` allows ``anthropic>=0.40``, where temperature is a real
    parameter that is honoured; the model-family list still has to be respected
    there, which is why this is a capability probe and not a deletion.
    """
    monkeypatch.setattr(llm_module, "_sdk_accepts_temperature", lambda: True)

    assert _payload("claude-sonnet-4-6", hard=False, temperature=0.4)["temperature"] == 0.4
    assert _payload("claude-haiku-4-5", hard=False, temperature=0.4)["temperature"] == 0.4
    # The 4.6+/5 families reject it outright; the SDK's willingness is moot.
    assert "temperature" not in _payload("claude-sonnet-5", hard=False)
    assert "temperature" not in _payload("claude-opus-5", hard=False)


def test_the_probe_answers_from_the_installed_sdk_and_never_raises() -> None:
    """Whatever the answer, it is cached and it matches the real signature."""
    from anthropic.resources.messages import Messages

    expected = "temperature" in inspect.signature(Messages.create).parameters
    assert llm_module._sdk_accepts_temperature() is expected
    assert llm_module._sdk_accepts_temperature() is expected, "second read is cached"


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #


def test_a_cancel_predicate_stops_new_requests() -> None:
    """Nothing is sent, nothing is recorded, and the shape of the result holds."""
    client = FakeClient(ANSWER_BOOK)
    records: list[LLMCallRecord] = []
    source = _source(
        model="claude-haiku-4-5",
        client=client,
        on_call=records.append,
        cancel=lambda: True,
    )

    results = source.propose(_requests())

    assert results == {"1A": [], "7A": []}, "every requested slot still gets an entry"
    assert client.calls == [], "a stopped session must not spend anything"
    assert records == [], "nothing was sent, so there is nothing to show"


def test_a_cancel_mid_pass_keeps_what_arrived_and_issues_nothing_further() -> None:
    """Cancellation suppresses new requests; it does not discard old answers."""
    client = FakeClient(ANSWER_BOOK)
    records: list[LLMCallRecord] = []
    source = _source(
        model="claude-haiku-4-5",
        client=client,
        batch_size=1,
        on_call=records.append,
        # True from the moment the first batch has been issued. With one worker
        # the batches run in submission order, so this stops exactly the second.
        cancel=lambda: len(client.calls) >= 1,
    )

    results = source.propose(_requests())

    assert len(client.calls) == 1
    assert len(records) == 1
    filled = {sid for sid, cands in results.items() if cands}
    assert filled == {"1A"}, "the batch that got through keeps its candidates"
    assert set(results) == {"1A", "7A"}

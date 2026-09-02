"""Tests for :mod:`xword.web.trace` and :mod:`xword.web.sessions`.

These two modules are the only concurrent code in the repo, so the tests here
are shaped by that rather than by the usual "call it and check the return":

*Replay equivalence* is asserted directly. The whole session design rests on
one property -- a subscriber arriving late and reading from cursor 0 sees
exactly what a subscriber that tailed the whole solve saw -- and if it ever
stops holding, reattaching after a puzzle switch silently shows a different
trace from the one that was actually produced. So one test runs both
subscribers against the same log and compares them event for event.

*Every wait is bounded.* ``TraceLog.wait_since`` blocks, which makes a
regression in it the kind of bug that hangs a suite for ever instead of failing
it. So no test here waits without a deadline, the close-while-waiting cases
assert a real upper bound on how long the return took rather than merely that
it happened, and the session fixture cancels and joins every manager it built.

*No test calls the real API.* Two fakes are used, deliberately at different
levels. :class:`_ScriptedAgent` is not a model client at all: it is a stand-in
for the *agent*, which emits chosen events and finishes on cue, and that is
what makes the registry's state machine (queued, running, stopping, terminal)
assertable without any solving. The tests at the bottom instead run the real
:class:`~xword.solver.agent.CrosswordAgent` over
:class:`~xword.candidates.llm.FakeClient`, because the two things worth
checking against the real agent -- that a finished session publishes a
well-formed serialised result, and that a stop inside the very first propose
pass yields an empty result rather than an ``AssertionError`` -- both depend on
the agent's own control flow and cannot be faked at this level without
assuming the answer.
"""

from __future__ import annotations

import itertools
import json
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import pytest
from conftest import MINI_ACROSS_CLUES, MINI_DOWN_CLUES, MINI_ROWS

from xword.candidates.llm import FakeClient, LLMCandidateSource
from xword.config import AgentConfig
from xword.core.grid import grid_rows, make_puzzle
from xword.core.types import (
    AgentEvent,
    Cell,
    Fill,
    Puzzle,
    SlotOutcome,
    SolveResult,
    SolveStats,
)
from xword.solver.agent import CrosswordAgent
from xword.web.sessions import (
    INTERRUPTIBLE_STATES,
    SessionInfo,
    SessionLimit,
    SessionManager,
)
from xword.web.trace import (
    MAX_EVENTS,
    MAX_TEXT_CHARS,
    TERMINAL_STATES,
    LLMCallRecord,
    TraceEvent,
    TraceLog,
)

# --------------------------------------------------------------------------- #
# Timeouts
#
# Two numbers, both generous. ``WAIT`` is the ceiling on anything that is
# expected to happen -- a session finishing, a parked agent noticing a stop --
# and is long enough that a loaded machine cannot fail it on scheduling alone.
# ``PROMPT`` is the much tighter ceiling used where the *speed* of a return is
# itself under test: a ``wait_since`` woken by ``close()`` has to come back at
# once, and asserting only that it eventually returned would pass even for a
# version that had sat out its whole timeout.
# --------------------------------------------------------------------------- #

WAIT = 10.0
PROMPT = 2.0
POLL = 0.005

#: Long enough to outlast Windows' ~16ms ``time.time()`` granularity, so an
#: elapsed-time comparison measures a clock that moved rather than two reads
#: taken inside one tick.
TICK = 0.05


def _wait_for(what: str, predicate: Callable[[], bool], *, timeout: float = WAIT) -> None:
    """Poll until ``predicate`` holds, or fail the test naming what was awaited.

    Polling rather than waiting on a condition, because the things being awaited
    live behind ``SessionManager``'s lock and it exposes no way to block on them
    -- correctly so: a registry that let readers wait for a state change would
    be a registry a reader could hold open.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(POLL)
    raise AssertionError(f"timed out after {timeout:.1f}s waiting for {what}")


def _tail(log: TraceLog, cursor: int = 0, *, timeout: float = WAIT) -> list[TraceEvent]:
    """Read a log the way an SSE subscriber does, and stop when it closes.

    Blocks in ``wait_since``, advances its cursor to the last sequence number it
    saw, and treats "nothing pending and the log is closed" as end of stream --
    which is the loop ``app.py``'s stream endpoint runs.
    """
    collected: list[TraceEvent] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        batch = log.wait_since(cursor, 0.2)
        for event in batch:
            collected.append(event)
            cursor = event.seq
        if not batch and log.closed:
            break
    return collected


# =========================================================================== #
# TraceLog
# =========================================================================== #


def test_sequence_numbers_start_at_one_and_increment() -> None:
    log = TraceLog()
    assert log.cursor == 0

    first = log.append("status", {"state": "queued", "message": "queued"})
    second = log.append("step", {"kind": "ingest", "round": 0, "message": "", "data": {}})

    assert (first.seq, second.seq) == (1, 2)
    assert log.cursor == 2
    assert [e.seq for e in log.since(0)] == [1, 2]


def test_append_returns_the_event_it_stored() -> None:
    log = TraceLog()
    event = log.append("error", {"message": "boom"})

    assert log.since(0) == [event]
    assert event.type == "error"
    assert event.payload == {"message": "boom"}
    assert event.at > 0.0
    assert not log.closed


def test_since_excludes_the_cursor_itself() -> None:
    """A cursor is "the last event I have seen", so including it would show the
    UI every event twice across a reconnect."""
    log = TraceLog()
    for i in range(5):
        log.append("step", {"n": i})

    assert [e.seq for e in log.since(3)] == [4, 5]
    assert [e.seq for e in log.since(4)] == [5]
    assert log.since(5) == []
    assert log.since(99) == []


@pytest.mark.parametrize("cursor", [0, -1, -1000])
def test_since_zero_or_below_is_the_whole_history(cursor: int) -> None:
    log = TraceLog()
    for i in range(4):
        log.append("step", {"n": i})

    assert [e.seq for e in log.since(cursor)] == [1, 2, 3, 4]


def test_snapshot_is_since_zero_as_plain_dicts() -> None:
    log = TraceLog()
    log.append("status", {"state": "running", "message": "solving"})
    snapshot = log.snapshot()

    assert snapshot == [e.as_dict() for e in log.since(0)]
    assert json.loads(json.dumps(snapshot)) == snapshot


def test_wait_since_returns_pending_events_without_waiting() -> None:
    log = TraceLog()
    log.append("step", {"n": 0})
    log.append("step", {"n": 1})

    started = time.monotonic()
    events = log.wait_since(0, WAIT)
    elapsed = time.monotonic() - started

    assert [e.seq for e in events] == [1, 2]
    assert elapsed < PROMPT, f"already-pending events took {elapsed:.2f}s to return"


def test_wait_since_respects_its_timeout_and_returns_empty() -> None:
    log = TraceLog()
    log.append("step", {"n": 0})

    started = time.monotonic()
    events = log.wait_since(1, 0.1)
    elapsed = time.monotonic() - started

    assert events == []
    assert elapsed >= 0.05, f"returned after {elapsed:.3f}s, i.e. it never waited"
    assert elapsed < PROMPT, f"overran a 0.1s timeout by {elapsed:.2f}s"
    assert not log.closed


def test_wait_since_wakes_on_an_append() -> None:
    log = TraceLog()
    writer = threading.Timer(TICK, lambda: log.append("step", {"n": 1}))
    writer.start()
    try:
        started = time.monotonic()
        events = log.wait_since(0, WAIT)
        elapsed = time.monotonic() - started
    finally:
        writer.cancel()

    assert [e.payload["n"] for e in events] == [1]
    assert elapsed < PROMPT, f"took {elapsed:.2f}s to notice an append"


def test_wait_since_returns_promptly_on_a_closed_log() -> None:
    """The regression that would hang the suite rather than fail it.

    ``wait_since`` is handed a timeout far longer than the assertion allows, so
    a version that stopped treating "closed" as a reason not to wait shows up as
    a failure in a couple of seconds instead of as a stuck test run.
    """
    log = TraceLog()
    log.append("step", {"n": 0})
    log.close()

    started = time.monotonic()
    events = log.wait_since(1, 60.0)
    elapsed = time.monotonic() - started

    assert events == []
    assert log.closed
    assert elapsed < PROMPT, f"a closed log kept a subscriber waiting {elapsed:.2f}s"


def test_close_releases_a_subscriber_that_is_already_blocked() -> None:
    """The same property from the other side: closed *while* someone waits."""
    log = TraceLog()
    returned: list[list[TraceEvent]] = []
    elapsed: list[float] = []

    def reader() -> None:
        started = time.monotonic()
        returned.append(log.wait_since(0, 60.0))
        elapsed.append(time.monotonic() - started)

    thread = threading.Thread(target=reader, name="trace-reader", daemon=True)
    thread.start()
    time.sleep(TICK)  # let the reader reach the wait rather than race it
    log.close()
    thread.join(PROMPT)

    assert not thread.is_alive(), "close() did not release a blocked subscriber"
    assert returned == [[]]
    assert elapsed[0] < PROMPT, f"the blocked subscriber took {elapsed[0]:.2f}s to return"


def test_wait_since_on_a_closed_log_still_delivers_the_tail() -> None:
    """Closing does not discard: a subscriber that had fallen behind still gets
    the events it never read, and only then sees the empty list."""
    log = TraceLog()
    log.append("status", {"state": "running", "message": ""})
    log.append("result", {"puzzle": {"id": "mini"}})
    log.append("status", {"state": "done", "message": "done"})
    log.close()

    tail = log.wait_since(1, 60.0)

    assert [e.type for e in tail] == ["result", "status"]
    assert log.wait_since(3, 0.1) == []


def test_max_events_drops_the_oldest_and_counts_them() -> None:
    log = TraceLog(max_events=5)
    for i in range(8):
        log.append("step", {"n": i})

    retained = log.since(0)
    assert log.dropped == 3
    assert [e.payload["n"] for e in retained] == [3, 4, 5, 6, 7]
    # Eviction does not renumber: a cursor held by a subscriber that went away
    # stays meaningful, and ``dropped`` is what explains the gap under it.
    assert [e.seq for e in retained] == [4, 5, 6, 7, 8]
    assert log.cursor == 8
    assert [e.seq for e in log.since(6)] == [7, 8]


def test_the_default_ceiling_is_max_events() -> None:
    log = TraceLog()
    for i in range(MAX_EVENTS + 3):
        log.append("step", {"n": i})

    assert len(log.since(0)) == MAX_EVENTS
    assert log.dropped == 3
    assert log.cursor == MAX_EVENTS + 3


@pytest.mark.parametrize("max_events", [0, -5])
def test_a_log_always_retains_at_least_one_event(max_events: int) -> None:
    """A zero or negative ceiling is a caller bug, and a log that retained
    nothing would report an empty trace for a solve that ran perfectly well."""
    log = TraceLog(max_events=max_events)
    log.append("step", {"n": 0})
    log.append("step", {"n": 1})

    assert [e.payload["n"] for e in log.since(0)] == [1]
    assert log.dropped == 1


# =========================================================================== #
# Replay equivalence -- the property the whole session design rests on
# =========================================================================== #


def test_a_late_subscriber_sees_exactly_what_the_live_tail_saw() -> None:
    """Reattaching is not a special case: it is the only case.

    A live tail and a from-scratch replay are compared event for event,
    *including order*, because "the same set of events" is not the guarantee the
    UI needs -- a trace whose call record precedes the step that caused it is
    wrong even though nothing is missing.
    """
    log = TraceLog()
    live: list[TraceEvent] = []

    def tailer() -> None:
        live.extend(_tail(log))

    subscriber = threading.Thread(target=tailer, name="live-tail", daemon=True)
    subscriber.start()

    for i in range(60):
        log.append("step", {"kind": "propose", "round": i // 15, "n": i})
        if i % 20 == 0:
            # Yield often enough that the tailer really is reading concurrently
            # rather than draining one batch at the end.
            time.sleep(0.001)
    log.append("result", {"puzzle": {"id": "mini-5x5"}})
    log.close()

    subscriber.join(WAIT)
    assert not subscriber.is_alive(), "the live tail never finished"

    replay = log.since(0)
    assert len(live) == 61
    assert [(e.seq, e.type, dict(e.payload)) for e in live] == [
        (e.seq, e.type, dict(e.payload)) for e in replay
    ]
    assert [e.seq for e in replay] == list(range(1, 62))


def test_a_replay_from_a_stale_cursor_resumes_without_a_gap() -> None:
    """What a reconnect after a puzzle switch actually does."""
    log = TraceLog()
    for i in range(10):
        log.append("step", {"n": i})

    first_visit = log.since(0)[:4]
    cursor = first_visit[-1].seq
    for i in range(10, 15):
        log.append("step", {"n": i})

    resumed = log.since(cursor)
    assert [e.payload["n"] for e in first_visit + resumed] == list(range(15))


# =========================================================================== #
# Thread safety
# =========================================================================== #


def test_concurrent_appends_lose_no_sequence_numbers() -> None:
    """One writer is the design, but the agent's batch pool means ``on_call``
    fires from several worker threads at once, so the log has to survive it."""
    log = TraceLog()
    writers = 8
    per_writer = 60
    barrier = threading.Barrier(writers)

    def writer(worker: int) -> None:
        barrier.wait()  # maximise the overlap rather than trickling in
        for i in range(per_writer):
            log.append("llm_call", {"worker": worker, "i": i})

    threads = [
        threading.Thread(target=writer, args=(w,), name=f"appender-{w}", daemon=True)
        for w in range(writers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT)
        assert not thread.is_alive(), "an appender thread never finished"

    events = log.since(0)
    total = writers * per_writer
    assert len(events) == total
    assert [e.seq for e in events] == list(range(1, total + 1))
    assert log.cursor == total
    assert log.dropped == 0
    # Nothing written twice and nothing lost, per writer.
    assert {(e.payload["worker"], e.payload["i"]) for e in events} == {
        (w, i) for w in range(writers) for i in range(per_writer)
    }


def test_a_reader_never_sees_a_half_built_history() -> None:
    """``since`` is called from HTTP handlers while the solve thread appends, so
    what it hands back has to be ordered and gap-free at every instant.

    The writers run until they are told to stop rather than for a fixed count,
    because a fixed count finishes long before the reading thread is next
    scheduled -- the first version of this test read the log exactly zero times
    while anything was writing to it, and would have passed against a log with
    no locking at all.
    """
    log = TraceLog(max_events=400)
    stop = threading.Event()

    def writer(worker: int) -> None:
        i = 0
        # The count is a backstop against a reader that never gets there, not a
        # target: the stop event is what normally ends this.
        while not stop.is_set() and i < 100_000:
            log.append("step", {"worker": worker, "i": i})
            i += 1

    threads = [
        threading.Thread(target=writer, args=(w,), name=f"storm-{w}", daemon=True)
        for w in range(4)
    ]
    for thread in threads:
        thread.start()
    try:
        for _ in range(200):
            seqs = [e.seq for e in log.since(0)]
            assert seqs == sorted(seqs)
            assert len(set(seqs)) == len(seqs)
            if seqs:
                assert seqs == list(range(seqs[0], seqs[-1] + 1)), "a hole in the history"
    finally:
        stop.set()
        for thread in threads:
            thread.join(WAIT)

    assert all(not thread.is_alive() for thread in threads)
    assert log.cursor > 0
    assert log.cursor == len(log.since(0)) + log.dropped


# =========================================================================== #
# LLMCallRecord
# =========================================================================== #


def _record(**overrides: Any) -> LLMCallRecord:
    """A realistic batch-pass record, with whatever a test cares about swapped."""
    fields: dict[str, Any] = {
        "id": "call-1",
        "label": "batch 1/1",
        "kind": "batch",
        "model": "claude-haiku-4-5",
        "round": 0,
        "system": "You are solving a crossword.",
        "prompt": "1A | len 3 | pat ??? | Fuss",
        "tools": ("submit_answers",),
        "tool_choice": "tool:submit_answers",
        "clue_ids": ("1A", "1D"),
        "stop_reason": "tool_use",
        "tool_name": "submit_answers",
        "tool_input": {"answers": [{"slot": "1A", "candidates": [{"answer": "ADO"}]}]},
        "input_tokens": 1500,
        "output_tokens": 200,
        "duration_s": 0.123456,
        "started_at": 1_700_000_000.0,
    }
    fields.update(overrides)
    return LLMCallRecord.build(**fields)


def _cache_record(**overrides: Any) -> LLMCallRecord:
    """What the source emits for clues served from the local clue cache: a
    logged non-call, with no prompt, no tokens and nothing spent."""
    fields: dict[str, Any] = {
        "id": "call-cache",
        "label": "cache hit",
        "kind": "cache",
        "system": "",
        "prompt": "",
        "tools": (),
        "tool_choice": "",
        "stop_reason": "",
        "tool_name": "",
        "tool_input": {},
        "input_tokens": 0,
        "output_tokens": 0,
        "cached": True,
    }
    fields.update(overrides)
    return _record(**fields)


@pytest.mark.parametrize("field_name", ["system", "prompt", "text"])
def test_build_clips_long_text_and_flags_it(field_name: str) -> None:
    record = _record(**{field_name: "x" * (MAX_TEXT_CHARS + 500)})

    assert len(getattr(record, field_name)) == MAX_TEXT_CHARS
    assert record.truncated is True


def test_build_leaves_ordinary_text_alone() -> None:
    record = _record(text="The clue is a pun on 'sole'.")

    assert record.truncated is False
    assert record.prompt == "1A | len 3 | pat ??? | Fuss"
    assert record.text == "The clue is a pun on 'sole'."


def test_build_normalises_the_iterable_fields_to_tuples() -> None:
    record = _record(tools=["submit_answers"], clue_ids=["5A", "5D"])

    assert record.tools == ("submit_answers",)
    assert record.clue_ids == ("5A", "5D")


def test_build_passes_the_accounting_fields_through() -> None:
    record = _record(
        attempts=3,
        cached=True,
        error="ConnectionError: nope",
        cache_read_tokens=900,
        cache_write_tokens=40,
    )

    assert (record.attempts, record.cached) == (3, True)
    assert record.error == "ConnectionError: nope"
    assert (record.cache_read_tokens, record.cache_write_tokens) == (900, 40)


def test_as_dict_round_trips_through_json() -> None:
    """The SSE layer serialises these straight onto the wire, so a field
    ``json.dumps`` chokes on would take the whole stream down mid-solve."""
    record = _record(text="prose from the hard pass", attempts=2)
    payload = record.as_dict()

    assert json.loads(json.dumps(payload)) == payload
    # The container fields have to arrive as JSON types, not as a tuple's repr.
    assert payload["tools"] == ["submit_answers"]
    assert payload["clue_ids"] == ["1A", "1D"]
    assert isinstance(payload["tool_input"], dict)
    assert payload["duration_s"] == pytest.approx(0.123)


def test_as_dict_survives_a_clipped_record() -> None:
    payload = _record(prompt="y" * (MAX_TEXT_CHARS * 2)).as_dict()

    assert payload["truncated"] is True
    assert len(payload["prompt"]) == MAX_TEXT_CHARS
    assert json.loads(json.dumps(payload))["truncated"] is True


# =========================================================================== #
# Session scaffolding: a serialiser, a scripted agent, a managed registry
# =========================================================================== #


def _serialise(puzzle: Puzzle, result: SolveResult) -> dict[str, Any]:
    """The keys ``app.py`` publishes, in the shape the registry reads them.

    A local copy rather than an import of ``app._serialise``: this file tests a
    threading contract, and dragging FastAPI in to do it would make a web
    framework a dependency of that. What matters to :class:`SessionManager` is
    only ``stats`` and ``score``, and both are built here the same way --
    ``score`` present exactly when the puzzle shipped answers, absent otherwise.
    """
    payload: dict[str, Any] = {
        "puzzle": {"id": puzzle.id, "width": puzzle.width, "height": puzzle.height},
        "fill": grid_rows(puzzle, result.fill.letters),
        "confidence": {
            f"{c.row},{c.col}": round(v, 4) for c, v in result.cell_confidence.items()
        },
        "entries": [
            {"id": sid, "clue": o.clue, "answer": o.answer, "source": o.source}
            for sid, o in sorted(result.slots.items())
        ],
        "stats": {
            "rounds": result.stats.rounds,
            "llm_calls": result.stats.llm_calls,
            "input_tokens": result.stats.input_tokens,
            "output_tokens": result.stats.output_tokens,
            "cache_hits": result.stats.cache_hits,
            "wall_seconds": round(result.stats.wall_seconds, 2),
            "cost_usd": round(result.stats.cost_usd, 5),
        },
        "trace": [
            {"kind": e.kind, "round": e.round, "message": e.message} for e in result.trace
        ],
    }
    if puzzle.has_solution:
        from xword.eval.metrics import score_result

        score = score_result(puzzle, result)
        payload["score"] = {
            "solved": score.solved,
            "cells_correct": score.cells_correct,
            "cells_total": score.cells_total,
        }
    return payload


class _ScriptedAgent:
    """A stand-in for :class:`CrosswordAgent` that finishes when it is told to.

    Not a fake model client -- there is one of those, and the tests at the
    bottom of this file use it. This is a fake *agent*, which is what lets the
    registry's own behaviour be pinned down: a solve that parks until it is
    cancelled, one that raises, one whose ``SolveStats`` deliberately disagree
    with its call records. None of that is reachable through a real solve
    without either real money or a lot of luck about timing.
    """

    def __init__(
        self,
        config: AgentConfig,
        *,
        on_event: Callable[[AgentEvent], None],
        cancel: Callable[[], bool],
        on_llm_call: Callable[[LLMCallRecord], None],
        park_until_cancelled: bool = False,
        hold_before_return: threading.Event | None = None,
        fail_with: BaseException | None = None,
        records: Sequence[LLMCallRecord] | None = None,
        stats: SolveStats | None = None,
    ) -> None:
        self.config = config
        self._on_event = on_event
        self._cancel = cancel
        self._on_llm_call = on_llm_call
        self._park = park_until_cancelled
        self._hold = hold_before_return
        self._fail_with = fail_with
        self._records = records
        self._stats = stats

        #: Set once ``solve`` has emitted its opening events, so a test can wait
        #: for "the agent really is solving" instead of guessing at a sleep.
        self.working = threading.Event()
        #: Set once the agent has reached ``hold_before_return`` -- the solve
        #: finished but not yet published, which is the only vantage point from
        #: which the registry's finalisers can be made to race anything.
        self.holding = threading.Event()
        #: What the cancel predicate the registry handed over said, read after
        #: the parking loop. Proves the wiring, not merely the state change.
        self.cancel_seen = False

    def solve(self, puzzle: Puzzle) -> SolveResult:
        self._on_event(
            AgentEvent(
                kind="ingest",
                round=0,
                message=f"{puzzle.height}x{puzzle.width} grid, {len(puzzle.slots)} entries",
                data={"entries": len(puzzle.slots)},
            )
        )
        for record in self._records if self._records is not None else (_record(),):
            self._on_llm_call(record)
        self._on_event(
            AgentEvent(kind="propose", round=0, message="asked about 10 clue(s)", data={})
        )
        self.working.set()

        if self._park:
            # Bounded, so a registry that never delivers the cancel fails the
            # state assertion rather than hanging the suite.
            deadline = time.monotonic() + WAIT
            while not self._cancel() and time.monotonic() < deadline:
                time.sleep(POLL)
        self.cancel_seen = self._cancel()

        if self._fail_with is not None:
            raise self._fail_with

        self._on_event(AgentEvent(kind="done", round=1, message="done", data={}))

        if self._hold is not None:
            # Bounded like the parking loop, and for the same reason: a test
            # that forgot to release this should fail, not hang the suite.
            self.holding.set()
            if not self._hold.wait(WAIT):
                raise TimeoutError("the test never released the held agent")
        return self._result(puzzle, partial=self.cancel_seen)

    def _result(self, puzzle: Puzzle, *, partial: bool) -> SolveResult:
        """A correct grid, or half of one for a solve that was cut short."""
        solution = puzzle.solution or {}
        letters: Mapping[Cell, str] = puzzle.solution_letters() if puzzle.has_solution else {}
        cells = sorted(letters)
        if partial:
            cells = cells[: len(cells) // 2]
        fill = {cell: letters[cell] for cell in cells}
        outcomes = {
            slot.id: SlotOutcome(
                slot_id=slot.id,
                clue=slot.clue,
                answer=None if partial else solution.get(slot.id),
                confidence=0.0 if partial else 0.9,
                source="none" if partial else "llm",
            )
            for slot in puzzle.slots
        }
        return SolveResult(
            puzzle_id=puzzle.id,
            fill=Fill(fill),
            cell_confidence=dict.fromkeys(fill, 0.9),
            slots=outcomes,
            stats=self._stats
            or SolveStats(
                rounds=1,
                llm_calls=1,
                input_tokens=1500,
                output_tokens=200,
                wall_seconds=0.02,
                cost_usd=0.0031,
            ),
            trace=[],
        )


class _ScriptedFactory:
    """The ``agent_factory`` seam, wired to a scripted agent and remembering it.

    ``SessionManager.start`` takes a puzzle and a config, so this factory is the
    only place a test can substitute something that does not spend money. The
    instances are kept because some assertions are about what the agent was
    *handed* -- a cancel predicate that actually flips, for one -- rather than
    about what the registry ended up reporting. ``options`` is mutable so that
    one manager can start a session that parks and then a session that finishes.
    """

    def __init__(self, **options: Any) -> None:
        self.options = options
        self.agents: list[_ScriptedAgent] = []

    def __call__(
        self,
        config: AgentConfig,
        *,
        on_event: Callable[[AgentEvent], None],
        cancel: Callable[[], bool],
        on_llm_call: Callable[[LLMCallRecord], None],
    ) -> _ScriptedAgent:
        agent = _ScriptedAgent(
            config,
            on_event=on_event,
            cancel=cancel,
            on_llm_call=on_llm_call,
            **self.options,
        )
        self.agents.append(agent)
        return agent


ManagerFactory = Callable[..., SessionManager]


@pytest.fixture
def make_manager() -> Iterator[ManagerFactory]:
    """Build managers, and make sure none of them outlives its test.

    Every scripted agent's parking loop watches the cancel predicate, so
    ``shutdown()`` unparks it immediately and the join costs nothing. Without
    this teardown a failed assertion would leave a daemon thread appending to a
    log for the rest of the session.
    """
    built: list[SessionManager] = []

    def _make(agent_factory: Any, **kwargs: Any) -> SessionManager:
        manager = SessionManager(serialise=_serialise, agent_factory=agent_factory, **kwargs)
        built.append(manager)
        return manager

    yield _make

    for manager in built:
        manager.shutdown()


@pytest.fixture
def unsolved_puzzle() -> Puzzle:
    """``mini_puzzle``'s grid with the answers withheld, so no solve of it can
    be graded. The registry has to report that as "no score", not as a zero."""
    return make_puzzle(
        "mini-no-answers",
        MINI_ROWS,
        across_clues=MINI_ACROSS_CLUES,
        down_clues=MINI_DOWN_CLUES,
        meta={"title": "Ungraded"},
    )


def _finished(manager: SessionManager, sid: str) -> SessionInfo:
    """Wait for a session to reach a terminal state, and return its snapshot."""

    def terminal() -> bool:
        info = manager.info(sid)
        return info is not None and info.state in TERMINAL_STATES

    _wait_for(f"session {sid} to finish", terminal)
    info = manager.info(sid)
    assert info is not None
    return info


def _solving(factory: _ScriptedFactory, count: int = 1) -> None:
    """Wait until ``count`` scripted agents are past their opening events."""
    _wait_for(
        f"{count} scripted agent(s) to start solving",
        lambda: len(factory.agents) >= count
        and all(a.working.is_set() for a in factory.agents[:count]),
    )


def _statuses(log: TraceLog) -> list[str]:
    return [str(e.payload["state"]) for e in log.since(0) if e.type == "status"]


def _transition_lock(manager: SessionManager, sid: str) -> threading.Lock:
    """The lock a session's state changes are made under.

    Reached through the registry's private dict, which is the only reach into
    private state anywhere in this file and wants justifying. The property under
    test is a locking property -- a state change and its ``status`` event are one
    step -- and from outside the code that takes a lock, the only way to observe
    it is to hold it: with this held, a ``stop()`` on another thread must have
    written *neither* half. Polling for a moment when the snapshot and the log
    disagree would prove nothing either way, because the window was always
    narrow; what changed is that it is now closed.

    It doubles as a starting gate. Taking it before releasing a held agent parks
    the registry's finaliser on it, so a stop can be lined up behind that and the
    two contend at a known point -- rather than the test issuing a stop into a
    solve that has already finished and calling whatever came out a race.
    """
    with manager._lock:
        return manager._sessions[sid].transition


# =========================================================================== #
# SessionManager: registration and the happy path
# =========================================================================== #


def test_start_returns_a_live_session_describing_the_puzzle(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    """What the POST that created the session hands straight back to the UI.

    The agent is parked, because a scripted solve of a 5x5 can otherwise finish
    inside ``start()`` itself: whether the thread has been scheduled yet is not
    something the caller can know, so with a real solve the answer is "queued or
    running" and asserting that needs a solve that cannot already be over.
    """
    factory = _ScriptedFactory(park_until_cancelled=True)
    manager = make_manager(factory)
    info = manager.start(mini_puzzle, AgentConfig(max_rounds=2))

    assert info.state in INTERRUPTIBLE_STATES
    assert info.id
    assert info.puzzle_id == "mini-5x5"
    assert (info.title, info.size) == ("Mini", "5x5")
    assert (info.entries, info.open_cells) == (10, 19)
    assert info.max_rounds == 2
    assert info.model == AgentConfig().model
    assert info.created_at > 0.0
    assert info.error == ""
    assert info.solved is None
    assert manager.result(info.id) is None, "a result before the solve ran"

    log = manager.log(info.id)
    assert log is not None
    # The session's own creation is the first thing in its trace, so a
    # subscriber reading from cursor 0 does not start mid-story.
    assert log.since(0)[0].payload == {"state": "queued", "message": "queued"}

    _solving(factory)
    assert manager.stop(info.id) is True
    assert _finished(manager, info.id).state == "stopped"


def test_the_log_fills_with_steps_and_call_records(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    """Both granularities in one ordered log, which is what lets the UI show
    cause next to effect."""
    manager = make_manager(_ScriptedFactory())
    info = manager.start(mini_puzzle, AgentConfig())
    assert _finished(manager, info.id).state == "done"

    log = manager.log(info.id)
    assert log is not None
    events = log.since(0)
    types = [e.type for e in events]
    assert types[0] == "status"
    assert "step" in types
    assert "llm_call" in types
    assert [e.seq for e in events] == list(range(1, len(events) + 1))

    step = next(e for e in events if e.type == "step")
    assert set(step.payload) == {"kind", "round", "message", "data"}
    assert step.payload["kind"] == "ingest"
    assert step.payload["round"] == 0

    call = next(e for e in events if e.type == "llm_call")
    assert call.payload["prompt"] == "1A | len 3 | pat ??? | Fuss"
    assert call.payload["tools"] == ["submit_answers"]


def test_the_result_event_lands_before_the_log_closes(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    """Closing first would release every waiting subscriber a moment before the
    result was appended, and each of them would report a solve with no grid."""
    manager = make_manager(_ScriptedFactory())
    info = manager.start(mini_puzzle, AgentConfig())
    log = manager.log(info.id)
    assert log is not None

    _finished(manager, info.id)
    events = log.since(0)

    assert log.closed
    assert [e.type for e in events[-2:]] == ["result", "status"]
    assert events[-1].payload["state"] == "done"
    assert events[-2].payload["stats"]["llm_calls"] == 1


def test_a_subscriber_woken_by_the_close_has_already_seen_the_result(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    """The same ordering asserted from where it matters: inside a subscriber."""
    manager = make_manager(_ScriptedFactory())
    info = manager.start(mini_puzzle, AgentConfig())
    log = manager.log(info.id)
    assert log is not None

    seen = _tail(log)

    assert log.closed
    assert [e.type for e in seen[-2:]] == ["result", "status"]
    assert seen == log.since(0)


def test_the_finished_totals_are_the_authoritative_stats(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    """The running figure is an estimate summed from call records; ``SolveStats``
    supersedes it, so the sidebar must not still be showing the guess."""
    stats = SolveStats(
        rounds=3, llm_calls=7, input_tokens=9_000, output_tokens=800, cost_usd=0.0421
    )
    manager = make_manager(_ScriptedFactory(stats=stats))
    info = manager.start(mini_puzzle, AgentConfig())
    final = _finished(manager, info.id)

    assert final.llm_calls == 7
    assert (final.input_tokens, final.output_tokens) == (9_000, 800)
    assert final.cost_usd == pytest.approx(0.0421)

    result = manager.result(info.id)
    assert result is not None
    assert result["stats"]["rounds"] == 3


def test_progress_and_money_track_the_call_records_while_running(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    """A cached record explains where an answer came from, but it is not a call
    and it cost nothing, so it has to be logged without being counted."""
    live = _record(input_tokens=2_000, output_tokens=300, cache_read_tokens=1_200)
    factory = _ScriptedFactory(
        park_until_cancelled=True, records=(live, _cache_record())
    )
    manager = make_manager(factory)
    info = manager.start(mini_puzzle, AgentConfig())
    _solving(factory)

    running = manager.info(info.id)
    assert running is not None
    assert running.state == "running"
    assert running.llm_calls == 1, "a cached record was counted as an API call"
    assert running.input_tokens == 3_200  # plain plus cache-read tokens
    assert running.output_tokens == 300
    assert running.cost_usd > 0.0
    assert (running.round, running.step) == (0, "propose")
    assert running.started_at > 0.0
    assert running.finished_at == 0.0

    log = manager.log(info.id)
    assert log is not None
    assert [e.type for e in log.since(0)].count("llm_call") == 2

    assert manager.stop(info.id) is True
    _finished(manager, info.id)


def test_a_failed_request_is_logged_without_being_billed(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    """``llm_calls`` counts billed calls, so two kinds of record do not count.

    A cache hit was never a request. A record carrying an ``error`` was a request
    that produced nothing -- it exhausted its retries, or a stop landed before
    the next one -- and the source books it under ``failures``, never under
    ``calls``. Counting attempts instead would make the running figure one
    higher per failure and then let it *drop* onto the authoritative
    ``SolveStats``, which reads as the agent losing a call rather than as an
    estimate being confirmed.
    """
    records = (
        _record(input_tokens=2_000, output_tokens=300),
        _record(
            id="call-2",
            error="ConnectionError: FakeClient: simulated transient failure",
            input_tokens=0,
            output_tokens=0,
        ),
        _cache_record(),
    )
    # The stats a real solve would report for exactly these records: one billed
    # call, and the tokens only that call carried.
    stats = SolveStats(
        rounds=1, llm_calls=1, input_tokens=2_000, output_tokens=300, wall_seconds=0.02
    )
    factory = _ScriptedFactory(park_until_cancelled=True, records=records, stats=stats)
    manager = make_manager(factory)
    info = manager.start(mini_puzzle, AgentConfig())
    _solving(factory)

    running = manager.info(info.id)
    assert running is not None
    assert running.state == "running"
    assert running.llm_calls == 1, "a request that came back with nothing was billed"
    assert (running.input_tokens, running.output_tokens) == (2_000, 300)

    # All three are still in the trace: the counter is about billing, not about
    # what the reader is allowed to see.
    log = manager.log(info.id)
    assert log is not None
    calls = [e.payload for e in log.since(0) if e.type == "llm_call"]
    assert len(calls) == 3
    assert [bool(c["error"]) for c in calls] == [False, True, False]

    # And the scripted stats agree on the same definition, so nothing moves.
    assert manager.stop(info.id) is True
    final = _finished(manager, info.id)
    assert final.llm_calls == running.llm_calls
    assert (final.input_tokens, final.output_tokens) == (2_000, 300)


def test_solved_stays_none_when_the_puzzle_shipped_no_answers(
    unsolved_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    manager = make_manager(_ScriptedFactory())
    info = manager.start(unsolved_puzzle, AgentConfig())
    final = _finished(manager, info.id)

    assert final.state == "done"
    assert final.solved is None
    assert (final.cells_correct, final.cells_total) == (0, 0)
    assert "score" not in (manager.result(info.id) or {})


def test_a_graded_solve_reports_its_score(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    manager = make_manager(_ScriptedFactory())
    info = manager.start(mini_puzzle, AgentConfig())
    final = _finished(manager, info.id)

    assert final.solved is True
    assert (final.cells_correct, final.cells_total) == (19, 19)


def test_list_is_newest_first_and_agrees_with_info(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    manager = make_manager(_ScriptedFactory())
    ids = []
    for _ in range(3):
        info = manager.start(mini_puzzle, AgentConfig())
        ids.append(info.id)
        _finished(manager, info.id)

    listing = manager.list()
    assert [s.id for s in listing] == list(reversed(ids))
    for listed in listing:
        assert listed == manager.info(listed.id)


def test_an_unknown_session_is_nothing_everywhere(make_manager: ManagerFactory) -> None:
    manager = make_manager(_ScriptedFactory())

    assert manager.info("nope") is None
    assert manager.log("nope") is None
    assert manager.result("nope") is None
    assert manager.stop("nope") is False
    assert manager.delete("nope") is False
    assert manager.list() == []
    assert manager.active == 0


def test_the_snapshot_and_the_whole_log_are_json_serialisable(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    """Both go straight down the wire: the snapshot on every poll, the log on
    every stream frame."""
    manager = make_manager(_ScriptedFactory())
    info = manager.start(mini_puzzle, AgentConfig())
    final = _finished(manager, info.id)
    log = manager.log(info.id)
    assert log is not None

    as_dict = final.as_dict()
    assert json.loads(json.dumps(as_dict)) == as_dict
    assert as_dict["terminal"] is True
    assert json.loads(json.dumps(log.snapshot()))[-1]["payload"]["state"] == "done"


def test_a_snapshot_is_a_copy_not_the_registry_s_own_record(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    manager = make_manager(_ScriptedFactory())
    info = manager.start(mini_puzzle, AgentConfig())
    _finished(manager, info.id)

    handed_out = manager.info(info.id)
    assert handed_out is not None
    handed_out.state = "error"
    handed_out.message = "vandalised"

    fresh = manager.info(info.id)
    assert fresh is not None
    assert fresh.state == "done"
    assert fresh.message != "vandalised"


def test_the_cursor_in_a_snapshot_tracks_the_log(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    manager = make_manager(_ScriptedFactory())
    info = manager.start(mini_puzzle, AgentConfig())
    final = _finished(manager, info.id)
    log = manager.log(info.id)
    assert log is not None

    assert final.cursor == log.cursor
    assert log.since(final.cursor) == []


# =========================================================================== #
# SessionManager: stopping, failing, and the clock
# =========================================================================== #


def test_stop_ends_a_running_session_as_stopped(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    factory = _ScriptedFactory(park_until_cancelled=True)
    manager = make_manager(factory)
    info = manager.start(mini_puzzle, AgentConfig())
    _solving(factory)

    assert manager.stop(info.id) is True
    stopping = manager.info(info.id)
    assert stopping is not None
    assert stopping.state == "stopping"

    final = _finished(manager, info.id)
    assert final.state == "stopped"
    assert final.error == ""
    assert factory.agents[0].cancel_seen is True, "the agent never saw the cancel"

    log = manager.log(info.id)
    assert log is not None
    assert _statuses(log) == ["queued", "running", "stopping", "stopped"]
    # A stopped solve still hands back the grid it had reached.
    assert log.since(0)[-2].type == "result"
    assert final.solved is False
    assert 0 < final.cells_correct < final.cells_total


def test_a_second_stop_and_a_stop_after_the_end_report_nothing_changed(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    """The button has to be able to say "there was nothing to stop"."""
    factory = _ScriptedFactory(park_until_cancelled=True)
    manager = make_manager(factory)
    info = manager.start(mini_puzzle, AgentConfig())
    _solving(factory)

    assert manager.stop(info.id) is True
    assert manager.stop(info.id) is False, "a stop already in flight reported success"

    _finished(manager, info.id)
    assert manager.stop(info.id) is False


def test_a_stop_writes_both_halves_of_its_change_or_neither(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    """A stop is one step: the new state and its ``status`` event, together.

    It used to be two. ``stop()`` took the manager lock, wrote ``stopping``,
    dropped the lock and only then appended the event, which left a real gap
    another transition could be made in -- and the solve thread's own finalisers
    made one. Held here from the outside, so that "indivisible" is asserted
    rather than assumed: while the transition lock is held, the stop must have
    moved neither the snapshot nor the log, and once released it must have moved
    both.
    """
    factory = _ScriptedFactory(park_until_cancelled=True)
    manager = make_manager(factory)
    info = manager.start(mini_puzzle, AgentConfig())
    _solving(factory)

    log = manager.log(info.id)
    assert log is not None
    before = log.cursor
    outcome: list[bool] = []
    stopper = threading.Thread(
        target=lambda: outcome.append(manager.stop(info.id)),
        name="stopper",
        daemon=True,
    )

    with _transition_lock(manager, info.id):
        stopper.start()
        stopper.join(TICK * 4)
        assert stopper.is_alive(), "stop() completed in the middle of a transition"
        during = manager.info(info.id)
        assert during is not None
        # Neither half. A state of "stopping" here would be the old bug: the
        # snapshot changed, and a reader taking it has no event to match it to.
        assert during.state == "running", "the state moved without its event"
        assert log.cursor == before, "the event landed without the state"
        assert outcome == []

    stopper.join(WAIT)
    assert not stopper.is_alive(), "stop() never returned"
    assert outcome == [True]

    after = manager.info(info.id)
    assert after is not None
    assert after.state in {"stopping", "stopped"}
    assert _statuses(log)[:3] == ["queued", "running", "stopping"]

    assert _finished(manager, info.id).state == "stopped"
    assert _statuses(log) == ["queued", "running", "stopping", "stopped"]


def test_a_stop_behind_a_finaliser_is_refused_not_logged_late(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    """The review's interleaving A, arranged rather than waited for.

    The agent is held with its solve complete but unpublished, and the
    transition lock is taken *before* it is released, so the registry's
    finaliser parks on that lock before it can publish anything. A stop is then
    queued behind it. Releasing the lock lets the finaliser through first, and
    the stop wakes to find a session that has already ended: it has to be
    refused, leaving the terminal ``status`` the last event on a closed log.
    What the old code could produce instead was a ``stopping`` event appended
    *after* that terminal status and onto the closed log -- a trace whose last
    event contradicts the session and which no subscriber is obliged to have
    read, since it was told the stream ended one event earlier.

    Lock hand-off order is not a language guarantee, so the other resolution --
    the stop getting in first and the solve ending ``stopped`` -- is legitimate
    and asserted too. Both branches say the same thing: the stop's return value
    and the trace agree.
    """
    release = threading.Event()
    factory = _ScriptedFactory(hold_before_return=release)
    manager = make_manager(factory)
    info = manager.start(mini_puzzle, AgentConfig())
    _solving(factory)
    _wait_for("the agent to reach its hold", factory.agents[0].holding.is_set)

    outcome: list[bool] = []
    stopper = threading.Thread(
        target=lambda: outcome.append(manager.stop(info.id)),
        name="stopper",
        daemon=True,
    )
    with _transition_lock(manager, info.id):
        release.set()  # the finaliser now queues on the lock this test holds
        time.sleep(TICK)  # ... and is waiting on it before the stop asks
        stopper.start()
        time.sleep(TICK)

    stopper.join(WAIT)
    assert not stopper.is_alive(), "stop() never returned"
    final = _finished(manager, info.id)

    log = manager.log(info.id)
    assert log is not None
    events = log.since(0)
    statuses = _statuses(log)

    assert log.closed
    assert events[-1].type == "status", f"the log ended on a {events[-1].type} event"
    assert events[-1].payload["state"] == final.state
    assert [s for s in statuses if s in TERMINAL_STATES] == [final.state]

    # The stop either got in ahead of the finaliser or was refused outright. It
    # may not report success without an event, nor leave one behind after the end.
    assert statuses.count("stopping") == (1 if outcome == [True] else 0)
    if outcome == [True]:
        assert statuses == ["queued", "running", "stopping", "stopped"]
    else:
        assert outcome == [False]
        assert statuses == ["queued", "running", "done"]


def test_stop_on_a_finished_session_returns_false(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    manager = make_manager(_ScriptedFactory())
    info = manager.start(mini_puzzle, AgentConfig())
    assert _finished(manager, info.id).state == "done"

    assert manager.stop(info.id) is False

    # And the refused stop did not append a phantom status to a closed log.
    log = manager.log(info.id)
    assert log is not None
    assert _statuses(log) == ["queued", "running", "done"]


def test_a_crashed_solve_becomes_an_error_session_with_its_traceback(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    """A studio whose failure mode is a blank screen is no better than no
    studio, so the exception has to reach the subscriber as data."""
    manager = make_manager(_ScriptedFactory(fail_with=RuntimeError("belief kernel blew up")))
    info = manager.start(mini_puzzle, AgentConfig())
    final = _finished(manager, info.id)

    assert final.state == "error"
    assert final.error == "RuntimeError: belief kernel blew up"
    assert "\n" not in final.error, "the sidebar's error line has to stay one line"
    assert manager.result(info.id) is None

    log = manager.log(info.id)
    assert log is not None
    events = log.since(0)
    assert log.closed
    assert [e.type for e in events[-2:]] == ["error", "status"]
    assert "result" not in [e.type for e in events]
    message = str(events[-2].payload["message"])
    assert message.startswith("RuntimeError: belief kernel blew up")
    assert "Traceback" in message

    # A failed session is finished, so it can be dismissed.
    assert manager.delete(info.id) is True
    assert manager.info(info.id) is None


def test_elapsed_advances_while_running_and_freezes_once_terminal(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    """A solve spends most of a round inside one API batch emitting nothing, so
    a clock derived from the last event looks exactly like a hung session."""
    factory = _ScriptedFactory(park_until_cancelled=True)
    manager = make_manager(factory)
    info = manager.start(mini_puzzle, AgentConfig())
    _solving(factory)

    first = manager.info(info.id)
    time.sleep(TICK)
    second = manager.info(info.id)
    assert first is not None
    assert second is not None
    assert second.elapsed_s > first.elapsed_s, "the clock did not move while running"

    assert manager.stop(info.id) is True
    final = _finished(manager, info.id)
    assert final.finished_at >= final.started_at > 0.0

    time.sleep(TICK)
    later = manager.info(info.id)
    assert later is not None
    assert later.elapsed_s == final.elapsed_s, "a finished session's clock kept running"
    assert later.elapsed_s >= second.elapsed_s


# =========================================================================== #
# SessionManager: the caps
# =========================================================================== #


def test_the_concurrency_cap_raises_rather_than_queueing(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    """A queue would accept the click, start spending minutes later, and bill
    for a solve whoever asked has since navigated away from."""
    factory = _ScriptedFactory(park_until_cancelled=True)
    manager = make_manager(factory, max_concurrent=1)
    first = manager.start(mini_puzzle, AgentConfig())
    _solving(factory)
    assert manager.active == 1

    with pytest.raises(SessionLimit, match="spend guard"):
        manager.start(mini_puzzle, AgentConfig())

    assert len(manager.list()) == 1, "the refused session was registered anyway"
    assert manager.active == 1

    # The slot frees up when the first session finishes, not on a timer.
    assert manager.stop(first.id) is True
    _finished(manager, first.id)
    assert manager.active == 0

    factory.options["park_until_cancelled"] = False
    second = manager.start(mini_puzzle, AgentConfig())
    assert _finished(manager, second.id).state == "done"
    assert len(manager.list()) == 2


def test_delete_refuses_a_running_session(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    """Dropping a live session would orphan a thread that is still spending."""
    factory = _ScriptedFactory(park_until_cancelled=True)
    manager = make_manager(factory)
    info = manager.start(mini_puzzle, AgentConfig())
    _solving(factory)

    assert manager.delete(info.id) is False
    assert manager.info(info.id) is not None
    log = manager.log(info.id)
    assert log is not None
    assert not log.closed, "a refused delete closed the log anyway"

    assert manager.stop(info.id) is True
    _finished(manager, info.id)

    assert manager.delete(info.id) is True
    assert manager.info(info.id) is None
    assert manager.result(info.id) is None
    assert log.closed


def test_retention_evicts_the_oldest_finished_session(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    manager = make_manager(_ScriptedFactory(), max_sessions=2)
    ids: list[str] = []
    logs: list[TraceLog | None] = []
    for _ in range(3):
        info = manager.start(mini_puzzle, AgentConfig())
        ids.append(info.id)
        logs.append(manager.log(info.id))
        _finished(manager, info.id)

    assert [s.id for s in manager.list()] == [ids[2], ids[1]]
    assert manager.info(ids[0]) is None
    assert logs[0] is not None
    assert logs[0].closed, "an evicted session's log was left open"


def test_retention_never_evicts_a_running_session(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    """The cap is deliberately soft: exceeding it beats orphaning a thread that
    is still issuing paid requests."""
    factory = _ScriptedFactory(park_until_cancelled=True)
    manager = make_manager(factory, max_sessions=1)
    live = manager.start(mini_puzzle, AgentConfig())
    _solving(factory)

    # Later sessions finish by themselves, so both are terminal and evictable.
    factory.options["park_until_cancelled"] = False
    quick = manager.start(mini_puzzle, AgentConfig())
    _finished(manager, quick.id)
    third = manager.start(mini_puzzle, AgentConfig())
    _finished(manager, third.id)

    ids = {s.id for s in manager.list()}
    assert live.id in ids, "a running session was evicted"
    assert quick.id not in ids, "the oldest finished session was not evicted"
    assert third.id in ids
    assert len(ids) == 2, "the cap should be exceeded rather than kill a live solve"
    survivor = manager.info(live.id)
    assert survivor is not None
    assert survivor.state == "running"

    assert manager.stop(live.id) is True
    _finished(manager, live.id)


def test_shutdown_cancels_every_session_and_closes_its_log(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    """Without it, Ctrl+C returns the shell prompt while daemon threads carry on
    issuing paid calls until the interpreter finally exits."""
    factory = _ScriptedFactory(park_until_cancelled=True)
    manager = make_manager(factory, max_concurrent=2)
    first = manager.start(mini_puzzle, AgentConfig())
    second = manager.start(mini_puzzle, AgentConfig())
    _solving(factory, count=2)

    manager.shutdown()

    for sid in (first.id, second.id):
        info = manager.info(sid)
        assert info is not None
        assert info.state in TERMINAL_STATES, f"session {sid} outlived the shutdown"
        log = manager.log(sid)
        assert log is not None
        assert log.closed
    assert manager.active == 0
    assert all(agent.cancel_seen for agent in factory.agents)

    names = {f"xword-session-{first.id}", f"xword-session-{second.id}"}
    assert not [t for t in threading.enumerate() if t.name in names]


# =========================================================================== #
# The real agent over a fake client
#
# No key, no network, no money: ``FakeClient`` answers out of a book keyed on
# the clue text it parses back out of the prompt the real builders produced.
# ``use_lexicon=False`` keeps the lexicon file out of it, and injecting the
# source means no clue cache is ever opened, so these run as fast as the
# scripted ones.
# =========================================================================== #


def _answer_book(puzzle: Puzzle) -> dict[str, list[tuple[str, float]]]:
    """``{clue: [(answer, probability)]}`` straight from the puzzle's own key."""
    solution = puzzle.solution or {}
    return {slot.clue: [(solution[slot.id], 0.9)] for slot in puzzle.slots}


def _offline_config(**overrides: Any) -> AgentConfig:
    return AgentConfig(
        use_lexicon=False,
        max_rounds=1,
        candidates_per_clue=5,
        max_concurrency=2,
        **overrides,
    )


class _FakeClientFactory:
    """An ``agent_factory`` building the real agent over a fake client.

    The source is injected rather than built by the agent, which is also what
    keeps the clue cache out of the test: ``_build_llm`` is the only thing that
    opens one. ``on_call`` has to be passed here for the same reason -- the
    agent only forwards its observer to sources it builds itself.

    ``max_retries`` is a constructor argument because ``AgentConfig`` has no
    field for it: the source's default of four means a ``FakeClient`` set to
    fail would back off for seconds and then succeed, and a test about what a
    *failed* request does to the accounting needs it to give up at once.
    """

    def __init__(self, client: FakeClient, *, max_retries: int = 4) -> None:
        self.client = client
        self.max_retries = max_retries
        self.sources: list[LLMCandidateSource] = []

    def __call__(
        self,
        config: AgentConfig,
        *,
        on_event: Callable[[AgentEvent], None],
        cancel: Callable[[], bool],
        on_llm_call: Callable[[LLMCallRecord], None],
    ) -> CrosswordAgent:
        source = LLMCandidateSource(
            model=config.model,
            k=config.candidates_per_clue,
            batch_size=config.batch_size,
            max_concurrency=config.max_concurrency,
            max_retries=self.max_retries,
            cache=None,
            client=self.client,
            on_call=on_llm_call,
            cancel=cancel,
        )
        self.sources.append(source)
        return CrosswordAgent(config, llm=source, on_event=on_event, cancel=cancel)


class _WatchedFactory(_FakeClientFactory):
    """Reads the sidebar's own numbers immediately after every call record.

    A polling thread would sample whatever the scheduler allowed, and a solve
    over a fake client is over in milliseconds -- so a monotonicity assertion
    built on one would usually be asserting nothing. ``info.llm_calls`` and its
    neighbours can only move inside ``_on_llm_call`` and once more in
    ``_succeed``, so a reading taken straight after each record has been
    accounted for is a reading at every point where a drop could appear.

    The snapshot is fetched via ``list()`` rather than by id, because the first
    record can be emitted before ``start()`` has returned one to the test. Only
    one session is ever registered on these managers, so the listing is it.
    """

    def __init__(self, client: FakeClient, *, max_retries: int = 4) -> None:
        super().__init__(client, max_retries=max_retries)
        self.manager: SessionManager | None = None
        self.readings: list[SessionInfo] = []

    def __call__(
        self,
        config: AgentConfig,
        *,
        on_event: Callable[[AgentEvent], None],
        cancel: Callable[[], bool],
        on_llm_call: Callable[[LLMCallRecord], None],
    ) -> CrosswordAgent:
        def watched(record: LLMCallRecord) -> None:
            on_llm_call(record)
            assert self.manager is not None, "the factory was never given its manager"
            self.readings.extend(self.manager.list())

        return super().__call__(
            config, on_event=on_event, cancel=cancel, on_llm_call=watched
        )


class _GatedClient(FakeClient):
    """A fake client that parks inside the request until a test lets it through.

    The only way to be sure a stop lands *while the first propose pass is in
    flight* -- which is the round-0 edge case -- is to hold the request open
    until the stop has been issued. Racing a sleep against the solve thread
    would make the test pass for the wrong reason most of the time.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.entered = threading.Event()
        self.released = threading.Event()

    def _create(self, **kwargs: Any) -> Any:
        self.entered.set()
        if not self.released.wait(WAIT):
            raise TimeoutError("the test never released the gated client")
        return super()._create(**kwargs)


class _CancelAfter:
    """A cancel predicate that flips to ``True`` after ``polls`` reads.

    The agent polls at phase boundaries in a fixed order, so counting reads
    picks a phase deterministically: 0 stops it at the top of round 0, 1 lets
    the propose pass through and stops it on the way out. A sleep, or an event
    set from another thread, would pick whichever phase the scheduler felt like.
    """

    def __init__(self, polls: int) -> None:
        self.limit = polls
        self.polls = 0

    def __call__(self) -> bool:
        self.polls += 1
        return self.polls > self.limit


@pytest.mark.parametrize(
    "polls, stage, expected_calls",
    [
        (0, "round start", 0),
        (1, "propose", 1),
    ],
)
def test_a_cancel_inside_round_zero_yields_an_empty_result(
    mini_puzzle: Puzzle, polls: int, stage: str, expected_calls: int
) -> None:
    """The regression this file partly exists to guard.

    Nothing is committed until the first round reaches ``commit``, so a stop
    before that leaves the agent with no assignment and no belief-propagation
    run to finalise. That case used to trip an ``assert`` -- a user pressing
    Stop surfacing as an ``AssertionError`` on a background thread. It has to be
    a well-formed, blank ``SolveResult`` instead, one that still reports what
    the round had already spent.
    """
    client = FakeClient(_answer_book(mini_puzzle))
    # The source is deliberately left un-cancellable so that the batch really
    # goes out: what is under test here is the agent's own control flow.
    source = LLMCandidateSource(
        model="claude-haiku-4-5",
        k=5,
        batch_size=12,
        max_concurrency=2,
        cache=None,
        client=client,
    )
    trace: list[AgentEvent] = []
    agent = CrosswordAgent(
        _offline_config(), llm=source, on_event=trace.append, cancel=_CancelAfter(polls)
    )

    result = agent.solve(mini_puzzle)

    assert result.puzzle_id == mini_puzzle.id
    assert result.fill.letters == {}
    assert result.cell_confidence == {}
    assert result.stats.rounds == 0
    assert result.stats.llm_calls == expected_calls
    assert len(client.calls) == expected_calls
    assert result.stats.notes["stopped_before_commit"] == 1.0
    assert result.stats.notes["cells_total"] == float(len(mini_puzzle.open_cells))

    # Every entry is reported as unanswered rather than missing, which is what a
    # caller rendering or scoring the result has to be able to read.
    assert set(result.slots) == {slot.id for slot in mini_puzzle.slots}
    assert all(o.answer is None and o.source == "none" for o in result.slots.values())
    if expected_calls:
        assert all(o.considered > 0 for o in result.slots.values())
        assert result.stats.cost_usd > 0.0, "a cancelled round still cost what it spent"

    assert ("verify", stage) in [(e.kind, e.data.get("stage")) for e in trace]
    assert trace[-1].kind == "done"


def test_a_session_solves_the_mini_puzzle_over_a_fake_client(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    """The registry against the real agent, end to end and for free."""
    factory = _FakeClientFactory(FakeClient(_answer_book(mini_puzzle)))
    manager = make_manager(factory)
    info = manager.start(mini_puzzle, _offline_config())
    final = _finished(manager, info.id)

    assert final.state == "done"
    assert final.solved is True
    assert (final.cells_correct, final.cells_total) == (19, 19)
    assert final.llm_calls == 1
    assert final.cost_usd > 0.0
    assert (final.round, final.step) == (1, "done")

    log = manager.log(info.id)
    assert log is not None
    events = log.since(0)
    types = [e.type for e in events]
    assert types[0] == "status"
    assert types[-2:] == ["result", "status"]
    assert types.count("llm_call") == 1

    call = next(e for e in events if e.type == "llm_call")
    assert call.payload["kind"] == "batch"
    assert call.payload["tools"] == ["submit_answers"]
    assert call.payload["tool_name"] == "submit_answers"
    assert "Fuss" in str(call.payload["prompt"]), "the record lost the real prompt"
    assert sorted(call.payload["clue_ids"]) == sorted(s.id for s in mini_puzzle.slots)

    result = manager.result(info.id)
    assert result is not None
    assert json.loads(json.dumps(result)) == result
    assert result["fill"] == grid_rows(mini_puzzle, mini_puzzle.solution_letters())
    assert json.loads(json.dumps(log.snapshot()))  # the whole stream, as SSE sees it


def test_the_live_counters_climb_onto_the_final_stats_and_never_off_them(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    """The real agent, with two requests that come back with nothing.

    ``batch_size=1`` makes one call per clue and ``max_retries=0`` makes the
    first two of them give up instead of backing off, so the solve genuinely
    emits records that were never billed -- which is the case that used to walk
    the sidebar's call count two above the truth and then drop it back at the
    end. Every reading the counters can take is checked, and the last one has to
    *be* the authoritative figure rather than get corrected to it.
    """
    client = FakeClient(_answer_book(mini_puzzle), fail_times=2)
    factory = _WatchedFactory(client, max_retries=0)
    manager = make_manager(factory)
    factory.manager = manager
    info = manager.start(mini_puzzle, _offline_config(batch_size=1))
    final = _finished(manager, info.id)

    log = manager.log(info.id)
    result = manager.result(info.id)
    assert log is not None
    assert result is not None
    calls = [e.payload for e in log.since(0) if e.type == "llm_call"]
    failed = [c for c in calls if c["error"]]
    assert len(failed) == 2, f"the failures never happened: {[c['error'] for c in calls]}"
    assert len(calls) > len(failed), "every request failed, so nothing was billed"

    billed = int(result["stats"]["llm_calls"])
    assert billed == len(calls) - len(failed), "the stats and the records disagree"
    assert final.llm_calls == billed

    readings = [*factory.readings, final]
    assert len(readings) == len(calls) + 1, "a call record went unobserved"
    assert max(r.llm_calls for r in readings) == billed, "the live count overshot"

    for earlier, later in itertools.pairwise(readings):
        assert later.llm_calls >= earlier.llm_calls, "the call count went backwards"
        assert later.input_tokens >= earlier.input_tokens
        assert later.output_tokens >= earlier.output_tokens
        # ``_serialise`` rounds cost to five decimals, so the final reading can
        # sit up to half of that quantum below the running sum it replaces. That
        # is the serialiser's resolution, not drift in the accounting, so the
        # cost series gets exactly that much slack and no more.
        assert later.cost_usd >= earlier.cost_usd - 5e-6, "the cost went backwards"

    assert final.cost_usd == pytest.approx(result["stats"]["cost_usd"])
    assert final.input_tokens == result["stats"]["input_tokens"]
    assert final.output_tokens == result["stats"]["output_tokens"]


def test_a_session_stopped_inside_the_first_propose_pass_still_publishes(
    mini_puzzle: Puzzle, make_manager: ManagerFactory
) -> None:
    """The round-0 edge case through the whole stack.

    The request is held open until the stop has been issued, so the agent is
    guaranteed to notice the cancel on its way out of the very first propose
    pass -- candidates bought, no grid committed. The session has to end
    ``stopped`` with a well-formed result, not ``error`` with an assertion.
    """
    client = _GatedClient(_answer_book(mini_puzzle))
    manager = make_manager(_FakeClientFactory(client))
    info = manager.start(mini_puzzle, _offline_config())
    try:
        _wait_for("the first batch request to go out", client.entered.is_set)
        assert manager.stop(info.id) is True
    finally:
        client.released.set()

    final = _finished(manager, info.id)
    assert final.state == "stopped", f"ended {final.state}: {final.error}"
    assert final.error == ""
    assert final.llm_calls == 1
    assert final.solved is False
    assert final.cells_correct == 0
    assert final.cells_total == len(mini_puzzle.open_cells)

    log = manager.log(info.id)
    assert log is not None
    assert [e.type for e in log.since(0)[-2:]] == ["result", "status"]
    assert log.closed

    result = manager.result(info.id)
    assert result is not None
    assert result["stats"]["rounds"] == 0
    assert result["stats"]["llm_calls"] == 1
    assert all(entry["answer"] is None for entry in result["entries"])
    assert json.loads(json.dumps(result)) == result

"""The session registry: solves that outlive the request that started them.

Why a registry at all
---------------------
``/api/solve/stream`` ties one solve to one HTTP response. The thread that runs
the agent is owned by the request handler and its events go nowhere except down
that socket, so closing the tab leaves a solve that is unreachable and still
spending money, and opening a second puzzle throws the first one's trace away.
The UI this package exists for wants the opposite: start three puzzles, look at
whichever one you like, stop one, come back to another and read its trace from
the beginning. That is only possible if a solve is owned by something which is
not a request. This module is that owner.

Each session holds its own :class:`~xword.web.trace.TraceLog` and subscribers
read it at a cursor, so reattaching is not a special case -- the rationale is
at the top of ``trace.py`` and is not worth a second copy here.

A thread per session, not an asyncio task
-----------------------------------------
The agent is synchronous, and inside one round it fans its clue batches out
over its own thread pool. Awaiting it on the event loop would block every other
request in the process -- including the polls and SSE tails that are the whole
point -- and rewriting the loop as coroutines would buy nothing, because the
work is one blocking SDK call per batch that is already issued concurrently. So
a session is a plain daemon thread, and the async side never touches it: HTTP
handlers only read ``SessionInfo`` snapshots and the log, both of which are safe
to read from anywhere.

Cancellation is cooperative
---------------------------
Nothing here kills a thread. ``stop()`` sets a :class:`threading.Event` whose
``is_set`` the agent is handed as its ``cancel`` predicate, and the agent
returns the best grid it has at its next check. A solve that has already
finished cannot be stopped, and ``stop()`` says ``False`` in that case rather
than pretending otherwise.

In-process, therefore local-only
--------------------------------
This registry is a dictionary in one process's memory. On Vercel a function
cannot outlive its response and the next request may land on a different
instance, so nothing here survives there: durable sessions are a **local**
capability, which is what ``app.py`` reports as ``durable_sessions`` and why the
UI falls back to the legacy single-request stream when it is false. Do not read
this module as a job queue -- there is no persistence, no retry and no
cross-process visibility, by design.

Money
-----
Every running session is issuing real Anthropic requests: roughly $0.007 for a
5x5 and $0.65 for a 15x15. So concurrency is capped, and going over the cap is
an error rather than a queue. A queue would accept the click, start spending
minutes later, and bill for a solve whoever asked for it has since navigated
away from.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from xword.config import AgentConfig, estimate_cost
from xword.core.types import AgentEvent, Puzzle, SolveResult
from xword.web.trace import TERMINAL_STATES, LLMCallRecord, SessionState, TraceLog

# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #

#: Total grace given to in-flight solve threads by :meth:`SessionManager.shutdown`.
#: Short on purpose: this runs on Ctrl+C, where the useful behaviour is to stop
#: spending money now, not to wait out a 15x15's last round.
SHUTDOWN_JOIN_SECONDS = 3.0

#: How much of a failed solve's traceback is kept in the ``error`` event. The
#: tail rather than the head, because the innermost frames are the ones that say
#: what actually broke.
MAX_TRACEBACK_CHARS = 8_000

#: Published multipliers for cached input tokens, mirrored from
#: ``LLMCandidateSource._record_tokens``. Duplicated here only so that the live
#: running cost converges on the ``SolveStats`` figure that replaces it at the
#: end: a sidebar whose number jumps when the solve finishes teaches the reader
#: to distrust both numbers.
CACHE_READ_RATE = 0.1
CACHE_WRITE_RATE = 1.25

#: States a caller can still interrupt. ``stopping`` is absent deliberately --
#: it means a stop is already in flight.
INTERRUPTIBLE_STATES: frozenset[str] = frozenset({"queued", "running"})

Serialiser = Callable[[Puzzle, SolveResult], dict[str, Any]]

#: Builds the object whose ``solve(puzzle)`` a session thread calls. Invoked as
#: ``factory(config, on_event=..., cancel=..., on_llm_call=...)``. It exists so a
#: test can inject an agent wired to a fake client: ``start()`` takes a puzzle
#: and a config, which leaves no other seam for one, and a test that had to
#: build the real agent would be a test that spends money.
AgentFactory = Callable[..., Any]


class SessionLimit(RuntimeError):
    """Raised by :meth:`SessionManager.start` when the concurrency cap is hit."""


# --------------------------------------------------------------------------- #
# Public snapshot
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SessionInfo:
    """Everything the sidebar renders about one session, in one object.

    Progress lives here rather than being derived from the log, because the
    listing is polled far more often than any single trace is read: drawing
    "round 2 - propagate, $0.04" must not cost a replay of every event of every
    session.

    Instances handed out by :class:`SessionManager` are copies. The manager
    mutates its own; a caller holding one of these holds a snapshot, and
    ``elapsed_s`` and ``cursor`` were true at the moment it was taken.
    """

    id: str
    puzzle_id: str
    title: str
    size: str
    entries: int
    open_cells: int
    state: SessionState = "queued"
    created_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    round: int = 0
    step: str = ""
    message: str = ""
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    elapsed_s: float = 0.0
    cursor: int = 0
    error: str = ""
    model: str = ""
    max_rounds: int = 0

    #: ``None`` when the puzzle shipped no answers. Such a solve has no score at
    #: all, and inventing a zero for it would read as a failure rather than as
    #: an absence.
    solved: bool | None = None
    cells_correct: int = 0
    cells_total: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "puzzle_id": self.puzzle_id,
            "title": self.title,
            "size": self.size,
            "entries": self.entries,
            "open_cells": self.open_cells,
            "state": self.state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "round": self.round,
            "step": self.step,
            "message": self.message,
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 5),
            "elapsed_s": round(self.elapsed_s, 2),
            "cursor": self.cursor,
            "error": self.error,
            "model": self.model,
            "max_rounds": self.max_rounds,
            "solved": self.solved,
            "cells_correct": self.cells_correct,
            "cells_total": self.cells_total,
            "terminal": self.state in TERMINAL_STATES,
        }


# --------------------------------------------------------------------------- #
# Internal record
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _Session:
    """One session's whole world: its snapshot, its log and its off switch.

    ``seq`` is a monotonic registration counter and is what orders the listing.
    ``created_at`` cannot do that job: on Windows ``time.time()`` advances in
    ~16ms steps, so two sessions started in the same tick share a timestamp and
    would sort arbitrarily against each other.

    ``transition`` serialises state changes; see
    :meth:`SessionManager._transition_locked` for why one lock could not.
    """

    info: SessionInfo
    puzzle: Puzzle
    config: AgentConfig
    log: TraceLog
    cancel: threading.Event
    transition: threading.Lock = field(default_factory=threading.Lock)
    seq: int = 0
    thread: threading.Thread | None = None
    result: dict[str, Any] | None = None


def _default_agent_factory(
    config: AgentConfig,
    *,
    on_event: Callable[[AgentEvent], None],
    cancel: Callable[[], bool],
    on_llm_call: Callable[[LLMCallRecord], None],
) -> Any:
    """Build the real agent.

    Imported inside the function, like every other solver import on the web
    side, so that merely listing sessions does not drag the solver and its
    belief-propagation kernels into the process.
    """
    from xword.solver.agent import CrosswordAgent

    return CrosswordAgent(
        config,
        on_event=on_event,
        cancel=cancel,
        on_llm_call=on_llm_call,
    )


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #


class SessionManager:
    """Owns the running solves and the logs they write to.

    There are three locks and exactly one order they may be taken in:
    **a session's ``transition`` lock, then the manager lock, then a log's
    lock.** Nothing anywhere takes them in any other order, and nothing takes
    the transition lock while already holding one of the other two, which is
    what makes the order sufficient rather than merely conventional.

    Who takes what: readers (HTTP handlers) take only the manager lock, to copy
    a snapshot. The solve thread takes it to bump progress and releases it
    before appending to the log. State changes -- and only state changes -- take
    the transition lock as well, and hold it across both halves of the change:
    :meth:`_transition_locked` explains why that second lock has to exist.
    Neither the agent nor the serialiser is ever called with any lock held, so a
    session listing never waits on belief propagation.
    """

    def __init__(
        self,
        *,
        serialise: Serialiser,
        max_concurrent: int = 3,
        max_sessions: int = 40,
        agent_factory: AgentFactory | None = None,
    ) -> None:
        self._serialise = serialise
        self._max_concurrent = max(1, int(max_concurrent))
        self._max_sessions = max(1, int(max_sessions))
        self._agent_factory = agent_factory or _default_agent_factory
        self._lock = threading.Lock()
        self._sessions: dict[str, _Session] = {}
        self._seq = 0

    # -- starting ---------------------------------------------------------- #

    def start(self, puzzle: Puzzle, config: AgentConfig) -> SessionInfo:
        """Register a session and put it on its own thread.

        Raises :class:`SessionLimit` when too many solves are already live,
        rather than queueing behind them: see the note on money at the top of
        this module.
        """
        sid = uuid.uuid4().hex[:12]
        now = time.time()
        info = SessionInfo(
            id=sid,
            puzzle_id=puzzle.id,
            title=str(puzzle.meta.get("title", "")) or puzzle.id,
            size=f"{puzzle.height}x{puzzle.width}",
            entries=len(puzzle.slots),
            open_cells=len(puzzle.open_cells),
            state="queued",
            created_at=now,
            message="queued",
            model=config.model,
            max_rounds=config.max_rounds,
        )
        session = _Session(
            info=info,
            puzzle=puzzle,
            config=config,
            log=TraceLog(),
            cancel=threading.Event(),
        )
        # Built before registration and started after it, so that the thread
        # cannot reach a session the registry does not yet know about, and
        # shutdown() cannot find a registered session with no thread to join.
        thread = threading.Thread(
            target=self._run,
            args=(session,),
            name=f"xword-session-{sid}",
            daemon=True,
        )
        session.thread = thread

        with self._lock:
            live = self._active_locked()
            if live >= self._max_concurrent:
                raise SessionLimit(
                    f"{live} solve(s) are already running and this process allows "
                    f"{self._max_concurrent} at once. The cap is a spend guard rather "
                    f"than a performance one: every concurrent solve is issuing real "
                    f"Anthropic requests -- about $0.007 for a 5x5 and $0.65 for a "
                    f"15x15 -- and the agent already fans its own clue batches out "
                    f"over several connections, so another session mostly buys cost. "
                    f"Stop one of the running sessions and try again."
                )
            self._seq += 1
            session.seq = self._seq
            self._sessions[sid] = session
            self._evict_locked()

        # Appended outside the lock, and before the thread runs, so a subscriber
        # reading from cursor 0 sees the session's own creation as the first
        # thing in its trace.
        session.log.append("status", {"state": "queued", "message": info.message})
        thread.start()

        snapshot = self.info(sid)
        assert snapshot is not None  # just registered, and eviction spares live sessions
        return snapshot

    # -- reading ----------------------------------------------------------- #

    def list(self) -> list[SessionInfo]:
        """Every session, newest first."""
        with self._lock:
            ordered = sorted(self._sessions.values(), key=lambda s: s.seq, reverse=True)
            return [self._snapshot_locked(s) for s in ordered]

    def info(self, sid: str) -> SessionInfo | None:
        """A live snapshot: ``elapsed_s`` ticks while running, freezes when done."""
        with self._lock:
            session = self._sessions.get(sid)
            if session is None:
                return None
            return self._snapshot_locked(session)

    def log(self, sid: str) -> TraceLog | None:
        """The session's log, for a subscriber to read at whatever cursor it holds.

        Handed out directly rather than proxied: it is append-only and does its
        own locking, so a reader can neither damage it nor need the manager.
        """
        with self._lock:
            session = self._sessions.get(sid)
            return session.log if session is not None else None

    def result(self, sid: str) -> dict[str, Any] | None:
        """The serialised solve, or ``None`` until there is one."""
        with self._lock:
            session = self._sessions.get(sid)
            return session.result if session is not None else None

    @property
    def active(self) -> int:
        """Sessions that have not finished -- what the concurrency cap counts."""
        with self._lock:
            return self._active_locked()

    # -- stopping and forgetting ------------------------------------------- #

    def stop(self, sid: str) -> bool:
        """Ask a session to stop. ``True`` only if this call changed something.

        An unknown id, a finished session, or a second stop on a session already
        stopping all return ``False``. The UI has to be able to tell "I just
        stopped it" from "there was nothing to stop", and a button that always
        reports success cannot.

        The test-and-set runs under the session's transition lock rather than
        under the manager lock alone, so that this stop is ordered against the
        solve thread's own transitions instead of racing them: without that, a
        stop and a finishing solve interleave into a ``stopping`` event appended
        after the terminal one, and a stop that lands in the queued-to-running
        window is silently overwritten by ``running``.
        """
        with self._lock:
            session = self._sessions.get(sid)
        if session is None:
            return False

        # The manager lock is released above and retaken inside; taking the
        # transition lock while holding it would invert the module's one
        # permitted lock order.
        with session.transition:
            with self._lock:
                if session.info.state not in INTERRUPTIBLE_STATES:
                    return False
                # Flagged in the same critical section as the test, so two
                # concurrent stops cannot both claim to be the one that
                # changed something.
                session.cancel.set()
            return self._transition_locked(
                session,
                "stopping",
                "stop requested; the agent hands back the grid it has at its next "
                "cancellation check",
            )

    def delete(self, sid: str) -> bool:
        """Forget a finished session. Refuses a live one.

        Dropping a running session would leave its thread appending to a log
        nobody can reach and, worse, spending money nobody can now stop. Stop it
        first, then delete it.
        """
        with self._lock:
            session = self._sessions.get(sid)
            if session is None or session.info.state not in TERMINAL_STATES:
                return False
            del self._sessions[sid]
            session.log.close()
        return True

    def shutdown(self) -> None:
        """Cancel everything and wait briefly. For the app's shutdown hook.

        Without it, Ctrl+C returns the shell prompt while daemon threads carry on
        issuing paid API calls until the interpreter finally exits. Logs are
        closed *after* the join, so a solve that finishes inside the grace period
        still delivers its ``result`` to whoever is watching; a thread that
        outlives the grace period dies with the process, being a daemon.
        """
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.cancel.set()
        deadline = time.monotonic() + SHUTDOWN_JOIN_SECONDS
        for session in sessions:
            thread = session.thread
            if thread is None or not thread.is_alive():
                continue
            thread.join(max(0.0, deadline - time.monotonic()))
        for session in sessions:
            session.log.close()

    # -- internals: bookkeeping -------------------------------------------- #

    def _active_locked(self) -> int:
        return sum(1 for s in self._sessions.values() if s.info.state not in TERMINAL_STATES)

    def _snapshot_locked(self, session: _Session) -> SessionInfo:
        """Copy a session's info, with the two fields that are computed on read.

        ``elapsed_s`` has to be derived at call time or it would only advance
        when something happened to be appended -- a solve spends most of a round
        inside one API batch, emitting nothing, and a timer that freezes there
        looks exactly like a hung session.
        """
        snapshot = replace(session.info)
        snapshot.cursor = session.log.cursor
        start = snapshot.started_at or snapshot.created_at
        frozen = snapshot.state in TERMINAL_STATES
        end = (snapshot.finished_at or start) if frozen else time.time()
        snapshot.elapsed_s = max(0.0, end - start)
        return snapshot

    def _evict_locked(self) -> None:
        """Trim the registry to ``max_sessions``, oldest finished session first.

        Only terminal sessions are eligible. If every session is still live the
        cap is simply exceeded until one finishes, which is much the lesser evil:
        ``max_concurrent`` bounds how many that can be anyway, and evicting a
        running session would orphan a thread that is still spending money.
        """
        overflow = len(self._sessions) - self._max_sessions
        if overflow <= 0:
            return
        evictable = sorted(
            (s for s in self._sessions.values() if s.info.state in TERMINAL_STATES),
            key=lambda s: s.seq,
        )
        for session in evictable[:overflow]:
            self._sessions.pop(session.info.id, None)
            session.log.close()

    def _transition_locked(
        self,
        session: _Session,
        state: SessionState,
        message: str,
        *,
        error: str = "",
        finished: bool = False,
    ) -> bool:
        """Move a session's state and tell the log about it, as one step.

        The caller must already hold ``session.transition``; every state change
        in this class does, which is why there is no unlocked variant to reach
        for by mistake.

        Why two locks, since the next reader of this file will want one
        ---------------------------------------------------------------
        The manager lock cannot do this job on its own. It is the lock every
        HTTP reader takes to copy a snapshot, so holding it across
        ``log.append`` would put a session listing behind a log write and break
        the rule that a reader never waits on the writer -- and it would put a
        second lock (the log's) inside the lock a reader holds, which is exactly
        the shape a later deadlock grows out of. Releasing it between the state
        change and the append -- which is what this used to do -- is worse: two
        transitions then interleave freely, and a ``stop()`` racing a finishing
        solve appends its ``stopping`` event *after* the terminal status and
        after ``close()``, leaving a trace whose last event contradicts the
        session and which no subscriber is required to have read.

        So state transitions take a second, per-session lock that only they
        take, and hold it across both halves and across the ``close()`` that
        ends the log. The order is **transition -> manager -> log**, never any
        other; ``stop()`` releases the manager lock before taking the transition
        lock for that reason.

        What this does and does not promise
        -----------------------------------
        Transitions are totally ordered and indivisible with respect to each
        other, and no status event can follow the terminal one -- ``False`` is
        returned, and nothing written, for a session that has already ended.
        Subscribers are told to treat the terminal ``status`` as end-of-stream,
        so an event appended after it is one they cannot be relied on to see.

        It is *not* a claim that a reader sees the state change and the event at
        the same instant: the snapshot is deliberately updated first, because
        appending wakes every subscriber and the first thing a woken subscriber
        does is read ``info()``. The other order shows a client a "done" event
        beside a session that still claims to be running, which is the
        inconsistency that persists; this one closes within microseconds and in
        the safe direction.
        """
        with self._lock:
            info = session.info
            if info.state in TERMINAL_STATES:
                return False
            info.state = state
            info.message = message
            if error:
                info.error = error
            if state == "running" and not info.started_at:
                info.started_at = time.time()
            if finished and not info.finished_at:
                info.finished_at = time.time()
        session.log.append("status", {"state": state, "message": message})
        return True

    # -- internals: the solve thread --------------------------------------- #

    def _run(self, session: _Session) -> None:
        """One session's whole life, on its own thread."""
        try:
            if not self._begin(session):
                return
            agent = self._agent_factory(
                session.config,
                on_event=lambda event: self._on_step(session, event),
                cancel=session.cancel.is_set,
                on_llm_call=lambda record: self._on_llm_call(session, record),
            )
            result = agent.solve(session.puzzle)
            payload = self._serialise(session.puzzle, result)
        except Exception as exc:  # a failure has to reach the subscriber as data
            self._fail(session, exc)
        else:
            self._succeed(session, payload)
        finally:
            self._ensure_terminal(session)

    def _begin(self, session: _Session) -> bool:
        """Take a queued session to ``running``, or straight to ``stopped``.

        ``False`` means a stop had already landed and there is nothing to solve.

        The cancel flag is read and the ``running`` transition made under one
        hold of the transition lock, because the two are a check and an act on
        the same piece of state. Read the flag, let a ``stop()`` run to
        completion, then transition: the session goes stopping -> running, the
        stop reads as ignored, and -- because ``running`` is interruptible again
        -- the next click is accepted as a second stop and logs a duplicate
        ``stopping`` event.
        """
        with session.transition:
            if session.cancel.is_set():
                # Stopped while still queued. Checking here, before the agent
                # exists, is the difference between a cancelled session costing
                # nothing and costing a whole puzzle.
                self._transition_locked(
                    session,
                    "stopped",
                    "stopped before the first API call; nothing was spent",
                    finished=True,
                )
                session.log.close()
                return False

            self._transition_locked(
                session,
                "running",
                f"solving {session.info.puzzle_id} with {session.config.model}",
            )
            return True

    def _on_step(self, session: _Session, event: AgentEvent) -> None:
        """Mirror one ``AgentEvent`` into the snapshot and then into the log."""
        with self._lock:
            info = session.info
            info.round = int(event.round)
            info.step = str(event.kind)
            info.message = event.message
        session.log.append(
            "step",
            {
                "kind": event.kind,
                "round": int(event.round),
                "message": event.message,
                "data": dict(event.data),
            },
        )

    def _on_llm_call(self, session: _Session, record: LLMCallRecord) -> None:
        """Account for one model call as it happens, then log it.

        ``SolveStats`` only exists once the solve is over, so a sidebar fed from
        it shows $0.00 for three minutes and then jumps to the real figure --
        worse than showing nothing, because it looks like the solve was free
        until suddenly it wasn't. These running totals are an estimate summed
        from the call records, and :meth:`_succeed` replaces them with the
        authoritative per-puzzle stats.

        The number means **billed calls**, not attempted ones, and two kinds of
        record are therefore logged without being counted. A record with
        ``cached`` set was answered from the local clue cache, so no request was
        ever made. A record carrying an ``error`` is one whose request produced
        nothing -- it gave up after its retries, or a stop landed before the
        next one -- and ``LLMCandidateSource`` books that under ``usage.failures``
        while ``usage.calls`` advances only after a ``messages.create`` that
        actually returned.

        Which is precisely why "billed" is the definition to pick:
        ``SolveStats.llm_calls`` is a diff of ``usage.calls``, so counting
        attempts here would read one higher per failed request all the way
        through the solve and then be replaced by the lower authoritative figure
        at the end. A counter that walks up and then drops does not read as an
        estimate being corrected, it reads as a bug in the agent -- the exact
        distrust this running total exists to avoid. Nothing is hidden by the
        choice: the failed request is still in the trace with its error, and the
        retries behind a call that did succeed are on its record as ``attempts``.
        """
        with self._lock:
            info = session.info
            if not record.cached and not record.error:
                info.llm_calls += 1
            info.input_tokens += (
                record.input_tokens + record.cache_read_tokens + record.cache_write_tokens
            )
            info.output_tokens += record.output_tokens
            billable_in = record.input_tokens + int(
                round(
                    record.cache_read_tokens * CACHE_READ_RATE
                    + record.cache_write_tokens * CACHE_WRITE_RATE
                )
            )
            info.cost_usd += estimate_cost(
                record.model or info.model, billable_in, record.output_tokens
            )
        session.log.append("llm_call", record.as_dict())

    def _succeed(self, session: _Session, payload: dict[str, Any]) -> None:
        """Publish the finished solve.

        The order is load-bearing: the ``result`` event, then the terminal status
        with ``finished_at`` set, then ``close()``. Closing first would release
        every waiting subscriber before the result had been appended, and each of
        them would conclude the session ended with nothing to show.

        All of it runs under the transition lock, so a ``stop()`` cannot land in
        the middle of the sequence. It either arrives before, and is read below
        as the reason this solve ended ``stopped``, or it arrives after and finds
        a terminal session to decline -- rather than appending a ``stopping``
        event onto a log this method has already closed.
        """
        stats = payload.get("stats") if isinstance(payload, dict) else None
        score = payload.get("score") if isinstance(payload, dict) else None

        seconds = 0.0
        rounds = 0
        calls = 0
        if isinstance(stats, dict):
            seconds = float(stats.get("wall_seconds", 0.0) or 0.0)
            rounds = int(stats.get("rounds", 0) or 0)
            calls = int(stats.get("llm_calls", 0) or 0)

        with session.transition:
            with self._lock:
                info = session.info
                session.result = payload
                if isinstance(stats, dict):
                    # Authoritative, and charged to this puzzle alone; they
                    # supersede the running estimate accumulated from the call
                    # records -- which ``_on_llm_call`` counts on the same
                    # definition, so this is a confirmation and not a jump.
                    info.llm_calls = int(stats.get("llm_calls", info.llm_calls))
                    info.input_tokens = int(stats.get("input_tokens", info.input_tokens))
                    info.output_tokens = int(stats.get("output_tokens", info.output_tokens))
                    info.cost_usd = float(stats.get("cost_usd", info.cost_usd))
                if isinstance(score, dict):
                    info.solved = bool(score.get("solved"))
                    info.cells_correct = int(score.get("cells_correct", 0))
                    info.cells_total = int(score.get("cells_total", 0))
                # No ``score`` key means the puzzle shipped no answers, so
                # ``solved`` stays None rather than claiming a failure nobody
                # graded.

            session.log.append("result", payload)

            if session.cancel.is_set():
                # The stop could have landed anywhere between the first
                # cancellation check and the last, so the honest claim is that a
                # stop was asked for and this is the grid the agent had -- not
                # that the solve was cut short at any particular phase.
                state: SessionState = "stopped"
                message = (
                    f"stopped on request after {rounds} round(s); the result is the best "
                    f"grid the agent had reached, {calls} API call(s) in"
                )
            else:
                state = "done"
                message = (
                    f"done in {seconds:.1f}s, {rounds} round(s), {calls} API call(s)"
                )

            self._transition_locked(session, state, message, finished=True)
            session.log.close()

    def _fail(self, session: _Session, exc: Exception) -> None:
        """Turn a crashed solve into something the UI can show.

        Called from inside the ``except`` block, which is what lets
        ``format_exc`` still see the active exception. The one-line summary goes
        to ``SessionInfo.error`` because that is all the sidebar has room for;
        the traceback goes into the ``error`` event's message, because a studio
        whose failure mode is "something went wrong" is no better than a blank
        screen. Either way it is never merely swallowed.

        Under the transition lock for the same reason as :meth:`_succeed`: the
        ``error`` event, the terminal status and the close are one step, and a
        concurrent ``stop()`` goes either wholly before them or not at all.
        """
        summary = f"{type(exc).__name__}: {exc}"
        detail = traceback.format_exc()[-MAX_TRACEBACK_CHARS:]
        with session.transition:
            session.log.append("error", {"message": f"{summary}\n\n{detail}"})
            self._transition_locked(session, "error", summary, error=summary, finished=True)
            session.log.close()

    def _ensure_terminal(self, session: _Session) -> None:
        """Backstop for a thread that unwound without finishing its session.

        Reachable only through something outside ``Exception`` -- a stray
        ``SystemExit``, an interpreter teardown -- or a bug in the finalisers
        above. Left alone, such a session reads "running" for ever and its
        subscribers block on a log that never closes, so it is marked and closed
        here instead.

        Routed through the transition lock like every other state change, so
        that this backstop cannot itself be the thing that interleaves with a
        ``stop()`` arriving at the same moment.
        """
        with session.transition:
            with self._lock:
                info = session.info
                if info.state in TERMINAL_STATES:
                    return
                summary = info.error or "the solve thread exited without finishing"
            self._transition_locked(session, "error", summary, error=summary, finished=True)
            session.log.close()


__all__ = [
    "CACHE_READ_RATE",
    "CACHE_WRITE_RATE",
    "INTERRUPTIBLE_STATES",
    "MAX_TRACEBACK_CHARS",
    "SHUTDOWN_JOIN_SECONDS",
    "AgentFactory",
    "Serialiser",
    "SessionInfo",
    "SessionLimit",
    "SessionManager",
]

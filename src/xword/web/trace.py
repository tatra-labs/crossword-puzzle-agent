"""The trace event model, and the append-only log the UI reads it from.

Why a log rather than a stream
------------------------------
The obvious design -- push events straight down the HTTP response that started
the solve -- is what the original ``/api/solve/stream`` does, and it cannot
satisfy the thing this package exists for. If the events live only in the
response, then closing the tab, switching to another puzzle, or subscribing a
second later all lose them irrecoverably.

So events go into a per-session log that owns them, and subscribers *read* from
it at a cursor. A subscriber arriving at event 400 asks for everything since 0
and gets the full history followed by the live tail, through exactly the same
code path a subscriber that was there from the start uses. Reattaching is
therefore not a special case: it is the only case.

The log is append-only and monotonically numbered from 1. A cursor is simply
the sequence number of the last event the client has seen, which makes it
durable across reconnects and safe to put in a query string.

What counts as an event
-----------------------
Two granularities, deliberately kept in one ordered log so the UI can show
cause next to effect:

* ``step`` -- the agent's own narration, one per phase per round: the same
  ``AgentEvent`` stream the CLI renders.
* ``llm_call`` -- one per request to the model, carrying the prompt actually
  sent, the tool offered, the tool call that came back, token counts and
  latency. This is the layer the CLI never exposed and the one that makes the
  agent's behaviour auditable rather than merely observable.

Plus ``status`` when a session changes state, and the terminal ``result`` or
``error``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #

#: Prompts are truncated at this many characters before being stored. A clue
#: batch prompt is a few KB; a pathological one should not be able to grow a
#: session's log without bound. Truncation is recorded, never silent.
MAX_TEXT_CHARS = 24_000

#: Hard ceiling on retained events per session. A 15x15 over four rounds
#: produces on the order of 60 events, so this is a runaway backstop rather
#: than a real constraint. When it trips, the *oldest* events are dropped and
#: ``TraceLog.dropped`` says how many, so the UI can say so rather than
#: silently showing a trace with a hole in it.
MAX_EVENTS = 5_000

SessionState = Literal[
    "queued",
    "running",
    "stopping",
    "done",
    "stopped",
    "error",
]

#: States in which no further events will ever be appended.
TERMINAL_STATES: frozenset[str] = frozenset({"done", "stopped", "error"})

EventType = Literal["status", "step", "llm_call", "result", "error"]


def _clip(text: str) -> tuple[str, bool]:
    """Return ``text`` bounded to :data:`MAX_TEXT_CHARS`, and whether it was cut."""
    if len(text) <= MAX_TEXT_CHARS:
        return text, False
    return text[:MAX_TEXT_CHARS], True


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LLMCallRecord:
    """One request to the model, from the request payload to what came back.

    Built at the single choke point every call passes through
    (``LLMCandidateSource._call``), so there is no way for a call to happen
    without being recorded -- including the retries, which are the most
    interesting calls when something is going wrong.
    """

    id: str
    label: str
    kind: str
    """``"batch"`` for the bulk clue pass, ``"hard"`` for the re-ask pass."""

    model: str
    round: int

    # -- what was sent -- #
    system: str
    prompt: str
    tools: tuple[str, ...]
    tool_choice: str
    clue_ids: tuple[str, ...]
    """The entries this call was asking about, for cross-linking in the UI."""

    # -- what came back -- #
    stop_reason: str = ""
    tool_name: str = ""
    tool_input: Mapping[str, Any] = field(default_factory=dict)
    text: str = ""
    """Any non-tool assistant text. The hard pass reasons in prose first."""

    # -- accounting -- #
    started_at: float = 0.0
    duration_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    attempts: int = 1
    cached: bool = False
    """True when the answer came from the local clue cache and no call was made."""

    error: str = ""
    truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "model": self.model,
            "round": self.round,
            "system": self.system,
            "prompt": self.prompt,
            "tools": list(self.tools),
            "tool_choice": self.tool_choice,
            "clue_ids": list(self.clue_ids),
            "stop_reason": self.stop_reason,
            "tool_name": self.tool_name,
            "tool_input": dict(self.tool_input),
            "text": self.text,
            "started_at": self.started_at,
            "duration_s": round(self.duration_s, 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "attempts": self.attempts,
            "cached": self.cached,
            "error": self.error,
            "truncated": self.truncated,
        }

    @classmethod
    def build(
        cls,
        *,
        id: str,
        label: str,
        kind: str,
        model: str,
        round: int,
        system: str,
        prompt: str,
        tools: Iterable[str] = (),
        tool_choice: str = "",
        clue_ids: Iterable[str] = (),
        **rest: Any,
    ) -> LLMCallRecord:
        """Construct with the text fields clipped and truncation flagged."""
        system_text, cut_a = _clip(system)
        prompt_text, cut_b = _clip(prompt)
        body, cut_c = _clip(str(rest.pop("text", "")))
        return cls(
            id=id,
            label=label,
            kind=kind,
            model=model,
            round=round,
            system=system_text,
            prompt=prompt_text,
            tools=tuple(tools),
            tool_choice=tool_choice,
            clue_ids=tuple(clue_ids),
            text=body,
            truncated=cut_a or cut_b or cut_c,
            **rest,
        )


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One numbered entry in a session's log."""

    seq: int
    at: float
    type: EventType
    payload: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "at": self.at,
            "type": self.type,
            "payload": dict(self.payload),
        }


# --------------------------------------------------------------------------- #
# The log
# --------------------------------------------------------------------------- #


class TraceLog:
    """A thread-safe append-only event log with cursor reads and blocking waits.

    One writer (the solve thread) and any number of readers (HTTP subscribers).
    Readers never block the writer: ``append`` takes the lock only long enough
    to push and notify.
    """

    def __init__(self, *, max_events: int = MAX_EVENTS) -> None:
        self._max_events = max(1, int(max_events))
        self._events: list[TraceEvent] = []
        self._seq = 0
        self._dropped = 0
        self._closed = False
        self._cond = threading.Condition()

    # -- writing ------------------------------------------------------------ #

    def append(self, type: EventType, payload: Mapping[str, Any]) -> TraceEvent:
        """Add an event and wake every waiting subscriber."""
        with self._cond:
            self._seq += 1
            event = TraceEvent(seq=self._seq, at=time.time(), type=type, payload=payload)
            self._events.append(event)
            if len(self._events) > self._max_events:
                overflow = len(self._events) - self._max_events
                del self._events[:overflow]
                self._dropped += overflow
            self._cond.notify_all()
            return event

    def close(self) -> None:
        """Mark the log finished. Waiting subscribers return immediately."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    # -- reading ------------------------------------------------------------ #

    @property
    def cursor(self) -> int:
        with self._cond:
            return self._seq

    @property
    def dropped(self) -> int:
        with self._cond:
            return self._dropped

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    def since(self, cursor: int) -> list[TraceEvent]:
        """Every retained event numbered above ``cursor``."""
        with self._cond:
            if cursor <= 0:
                return list(self._events)
            return [e for e in self._events if e.seq > cursor]

    def wait_since(self, cursor: int, timeout: float) -> list[TraceEvent]:
        """Events past ``cursor``, waiting up to ``timeout`` for the first one.

        Returns an empty list on timeout, or when the log is closed and
        exhausted -- the caller distinguishes those with :attr:`closed`.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cond:
            while True:
                pending = [e for e in self._events if e.seq > cursor]
                if pending or self._closed:
                    return pending
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._cond.wait(remaining)

    def snapshot(self) -> list[dict[str, Any]]:
        return [e.as_dict() for e in self.since(0)]


__all__ = [
    "MAX_EVENTS",
    "MAX_TEXT_CHARS",
    "TERMINAL_STATES",
    "EventType",
    "LLMCallRecord",
    "SessionState",
    "TraceEvent",
    "TraceLog",
]

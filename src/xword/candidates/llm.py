"""The LLM candidate source: the part of the agent that actually reads clues.

Shape of the thing
------------------
``propose`` takes a list of :class:`~xword.core.types.ClueRequest` and returns
candidates per slot. In between:

1. **Cache first.** Every request is looked up in the :class:`ClueCache` before
   anything is batched, so a warm run costs nothing and the miss count is an
   honest measure of what a cold run would have cost.
2. **Batch, then fan out.** Misses are packed ``batch_size`` clues to a call --
   the system prompt and puzzle context are the bulk of the input tokens, so
   batching is the single biggest cost lever -- and the calls are issued
   concurrently. Results are merged in request order, never completion order.
3. **Distrust the model.** Every returned answer is re-checked against length,
   alphabet and pattern. The model is good but not obedient, and one wrong-length
   answer that slips through poisons every crossing entry.
4. **Degrade, don't crash.** A dead or rate-limited API costs the affected slots
   their candidates and nothing more; the solver still has the lexicon and the
   crossings.

Probabilities become ``Candidate.score = log(p)``. Downstream fusion works in
log space and normalises per slot, so the absolute scale does not matter, but
using the log keeps a 0.9 and a 0.09 candidate a constant distance apart no
matter which source they came from.

Observability and cancellation
------------------------------
Two optional hooks, both plain callables so this module stays ignorant of
whatever is watching it:

* ``on_call`` receives one :class:`~xword.web.trace.LLMCallRecord` per request
  -- the system and user text actually sent, the tools offered, the tool call
  that came back, tokens, latency, and why any attempt before the last one
  failed. Token counts alone cannot answer "what
  did you ask it and what did it say", which is the only question a trace
  exists to answer. The record is built inside ``_call``, the one point every
  request and every retry passes through, so no code path can issue a call
  without producing one. Batches run concurrently, so the callback is invoked
  from several threads at once and has to be thread-safe.
* ``cancel`` is a predicate consulted before anything new is issued. A stopped
  session must not keep spending credit, but it must also keep the candidates
  that already arrived, so cancellation only ever suppresses new requests: it
  never raises, and a partly-filled result is handed back as it stands.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import math
import random
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from uuid import uuid4

from xword.candidates.cache import ClueCache, cache_key, context_digest
from xword.candidates.prompts import (
    ANSWER_TOOL,
    ANSWER_TOOL_NAME,
    HARD_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_batch_prompt,
    build_hard_prompt,
)
from xword.config import api_key, estimate_cost
from xword.core.grid import pattern_matches
from xword.core.types import Candidate, ClueRequest

# ``xword.web.trace`` imports nothing but the standard library (threading, time,
# dataclasses, typing -- checked, not assumed), so recording calls costs this
# module no dependency on FastAPI or on anything else the web surface needs.
from xword.web.trace import LLMCallRecord

#: Floor applied before taking the log, so a model that returns probability 0
#: yields a very bad score instead of ``-inf``.
MIN_PROBABILITY = 1e-6

#: Output ceiling per call. A 12-clue batch of 10 candidates is well under this;
#: the headroom is for hard-clue calls, which also spend tokens on analysis.
MAX_OUTPUT_TOKENS = 8000

#: HTTP statuses worth retrying. 429 is rate limiting, 529 is Anthropic's
#: "overloaded", the 5xx are transient.
RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

#: Backoff shape: ``BASE * 2**attempt`` seconds, capped, plus up to 25% jitter.
BACKOFF_BASE = 0.75
BACKOFF_CAP = 30.0
BACKOFF_JITTER = 0.25

#: Seed for the jitter RNG. Fixed so a run is reproducible; jitter only needs to
#: decorrelate this process's own concurrent retries, not be unpredictable.
JITTER_SEED = 0x58574F52

#: One failure's text is clipped to this before it enters a call record. An SDK
#: error can carry a whole response body, and ``max_retries`` of those would
#: dominate a session's log while adding nothing: the exception type and the
#: start of its message are what identifies a rate limit, a timeout or a 400.
MAX_ERROR_CHARS = 300

#: Model families that reject ``temperature``/``top_p``/``top_k`` outright (the
#: 4.6+ generation dropped sampling controls). Sending one is a 400, so the
#: configured temperature is silently not sent to these.
#:
#: This list is only half the question. There are now *two* independent reasons
#: the parameter may be unsendable, and both have to be clear before it goes
#: out:
#:
#: 1. **The model rejects it** -- this list. A request carrying a temperature to
#:    one of these families comes back a 400 from the API.
#: 2. **The SDK no longer has it.** Sampling controls were removed from the
#:    Messages endpoint, so ``Messages.create()`` in ``anthropic`` 1.x has no
#:    ``temperature`` parameter at all and passing one raises ``TypeError``
#:    locally, before any request is made. That failed *every* call on a model
#:    not in this list while the default model, being on it, kept working --
#:    which is how the regression stayed invisible until someone picked
#:    ``claude-haiku-4-5`` out of the model selector and watched the solve fall
#:    back to lexicon and crossings.
#:
#: Hence :func:`_sdk_accepts_temperature` rather than deleting the parameter:
#: ``pyproject`` allows ``anthropic>=0.40``, where temperature is real and
#: honoured, so this has to be a capability probe.
_MODELS_WITHOUT_SAMPLING: tuple[str, ...] = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)

#: Reads a clue line back out of a prompt. Used only by :class:`FakeClient`.
_CLUE_LINE_RE = re.compile(
    r"^(?P<slot>\S+) \| len (?P<length>\d+) \| pat (?P<pattern>\S+) \| (?P<clue>.*)$"
)

_NON_ALPHA_RE = re.compile(r"[^A-Z]")


def _supports_sampling(model: str) -> bool:
    return not any(model.startswith(prefix) for prefix in _MODELS_WITHOUT_SAMPLING)


def _failure_text(exc: BaseException) -> str:
    """One failed attempt as a line a trace reader can act on.

    The type is the diagnosis -- ``RateLimitError`` and ``APITimeoutError`` call
    for different responses -- so it leads, and the message follows for the
    detail the type does not carry.
    """
    return f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]


@lru_cache(maxsize=1)
def _sdk_accepts_temperature() -> bool:
    """Whether the installed SDK's ``Messages.create`` still takes ``temperature``.

    Cached, because the answer is a property of the installed package and cannot
    change while the process runs; inspecting a signature per call would put
    reflection on the path of every batch.

    ``anthropic`` is imported here rather than at module scope for the same
    reason :meth:`LLMCandidateSource._get_client` does it: ``import xword`` must
    not pull the SDK in.

    Fails safe. If the probe itself cannot answer -- the SDK is absent, or a
    future version moves the class -- the answer is "no". Omitting a temperature
    costs a solve its sampling setting; sending one the SDK cannot take costs
    the solve every LLM call it was going to make.
    """
    try:
        from anthropic.resources.messages import Messages

        return "temperature" in inspect.signature(Messages.create).parameters
    except Exception:  # noqa: BLE001 - any failure to probe means "do not send"
        return False


# --------------------------------------------------------------------------- #
# Request/response introspection
# --------------------------------------------------------------------------- #
#
# Everything below reads a payload or a response defensively. Both dicts and
# objects turn up here -- the request payload is plain dicts, ``FakeClient``
# returns dataclasses, the real SDK returns pydantic models -- and a tracing
# layer that raised on an unexpected shape would break the solve it exists to
# describe.


def _block_text(block: Any) -> str:
    """The text of one content block, dict-shaped or object-shaped."""
    if isinstance(block, str):
        return block
    if isinstance(block, Mapping):
        if block.get("type", "text") != "text":
            return ""
        return str(block.get("text", "") or "")
    if getattr(block, "type", "text") != "text":
        return ""
    return str(getattr(block, "text", "") or "")


def _content_text(content: Any) -> str:
    """Content as plain text, whether it is a string or a list of blocks.

    Both shapes are live: :meth:`LLMCandidateSource._request_kwargs` sends
    ``system`` as a one-element list so it can carry a cache breakpoint but the
    user turn as a bare string, and a response is a list of mixed blocks.
    Normalising here rather than at each call site means a later change to any
    of those cannot turn a trace into an exception.
    """
    if content is None:
        return ""
    if isinstance(content, (str, Mapping)):
        return _block_text(content)
    if isinstance(content, (list, tuple)):
        return "\n".join(part for part in (_block_text(b) for b in content) if part)
    return str(content)


def _tool_names(tools: Any) -> tuple[str, ...]:
    """The names of the tools offered, in the order they were sent."""
    names: list[str] = []
    for tool in tools or []:
        name = tool.get("name") if isinstance(tool, Mapping) else getattr(tool, "name", None)
        if name:
            names.append(str(name))
    return tuple(names)


def _tool_choice_label(choice: Any) -> str:
    """``tool_choice`` as something a UI can print: ``tool:submit_answers``."""
    if choice is None:
        return ""
    if isinstance(choice, str):
        return choice
    if isinstance(choice, Mapping):
        kind = str(choice.get("type", "") or "")
        name = str(choice.get("name", "") or "")
        return f"tool:{name}" if name else kind
    return str(choice)


def _call_kind(request_kwargs: Mapping[str, Any]) -> str:
    """``"hard"`` for the analysis pass, ``"batch"`` for the bulk pass.

    Derived from the ``tool_choice`` that was actually sent rather than from a
    flag threaded down from ``_run``. The two could drift, and if they ever did
    it is the request that is telling the truth.
    """
    choice = request_kwargs.get("tool_choice")
    if isinstance(choice, Mapping) and choice.get("type") == "auto":
        return "hard"
    return "batch"


def _first_tool_use(message: Any) -> tuple[str, dict[str, Any]]:
    """``(name, input)`` of the first tool call in a response, else ``("", {})``.

    Unlike :meth:`LLMCandidateSource._tool_payload` this does not filter by tool
    name. A call to some other tool, or to none at all, is exactly what a trace
    should show; a record that agreed with the parser instead of with the
    response would hide the one thing worth seeing.
    """
    for block in getattr(message, "content", None) or []:
        if isinstance(block, Mapping):
            if block.get("type") != "tool_use":
                continue
            name = str(block.get("name", "") or "")
            payload: Any = block.get("input")
        else:
            if getattr(block, "type", None) != "tool_use":
                continue
            name = str(getattr(block, "name", "") or "")
            payload = getattr(block, "input", None)
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                payload = None
        return name, dict(payload) if isinstance(payload, Mapping) else {}
    return "", {}


def _usage_int(usage: Any, name: str) -> int:
    """One usage counter, tolerant of a response that does not carry it.

    ``FakeClient`` and older SDK versions omit fields the current API returns,
    and a missing cache-token count has to read as zero rather than raise.
    """
    if usage is None:
        return 0
    value = getattr(usage, name, None)
    if value is None and isinstance(usage, Mapping):
        value = usage.get(name)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# Usage accounting
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class LLMUsage:
    """What one source spent, in tokens, calls and dollars.

    ``cache_hits`` and ``cache_misses`` are deliberately separate from ``calls``:
    a hit costs nothing, and the harness reports cold-cache cost, so conflating
    the two would misstate the number that matters.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retries: int = 0
    failures: int = 0
    cost_usd: float = 0.0
    dropped: int = 0

    def merge(self, other: LLMUsage) -> None:
        """Fold ``other`` into this record, in place."""
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_hits += other.cache_hits
        self.cache_misses += other.cache_misses
        self.retries += other.retries
        self.failures += other.failures
        self.cost_usd += other.cost_usd
        self.dropped += other.dropped


# --------------------------------------------------------------------------- #
# Source
# --------------------------------------------------------------------------- #


class LLMCandidateSource:
    """Candidate generation by asking Claude, with caching and validation.

    Satisfies the :class:`~xword.core.types.CandidateSource` protocol.

    The escalation path is ``propose_hard``, which uses the analysis prompt and
    one call per clue. It runs on this instance's ``model``; to escalate to a
    stronger model, build a second source with ``model=cfg.hard_clue_model``.
    Keeping one model per instance is what keeps the usage and cost figures
    attributable.

    ``on_event`` narrates for humans; ``on_call`` records for machines, one
    :class:`~xword.web.trace.LLMCallRecord` per request. Both are invoked from
    the batch worker threads, so a subscriber must be thread-safe --
    ``TraceLog`` is -- and a subscriber that raises loses its event and nothing
    else.
    """

    name = "llm"

    def __init__(
        self,
        *,
        model: str,
        k: int = 10,
        batch_size: int = 12,
        max_concurrency: int = 8,
        temperature: float = 1.0,
        cache: ClueCache | None = None,
        client: Any = None,
        mode: str = "batch",
        max_retries: int = 4,
        on_event: Callable[[str], None] | None = None,
        on_call: Callable[[LLMCallRecord], None] | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> None:
        self.model = model
        self.k = max(0, int(k))
        self.batch_size = max(1, int(batch_size))
        self.max_concurrency = max(1, int(max_concurrency))
        self.temperature = temperature
        self.cache = cache
        self.mode = mode
        self.max_retries = max(0, int(max_retries))
        self.on_event = on_event
        self.on_call = on_call
        self.cancel = cancel
        self.usage = LLMUsage()

        # Which agent round the calls now being issued belong to. An attribute
        # rather than a parameter: every batch in a round carries the same round
        # number and the batches run concurrently in a thread pool, so threading
        # it through propose -> _run -> worker -> _call would widen four
        # signatures and buy nothing over the agent loop setting it once before
        # each propose pass. It is advisory -- a stale value mislabels a record,
        # it cannot affect a solve.
        self.round_hint: int = 0

        self._client = client
        self._client_lock = threading.Lock()
        self._usage_lock = threading.Lock()
        self._rng = random.Random(JITTER_SEED)
        self._rng_lock = threading.Lock()

    # -- plumbing ---------------------------------------------------------- #

    def _emit(self, message: str) -> None:
        if self.on_event is not None:
            self.on_event(message)

    def _emit_call(self, record: LLMCallRecord) -> None:
        """Hand one call record to the subscriber, whatever it does with it.

        Invoked concurrently from the batch pool, so ``on_call`` has to be
        thread-safe. Exceptions from it are swallowed deliberately: the tracing
        layer is not allowed to be the thing that fails a solve.
        """
        if self.on_call is None:
            return
        with contextlib.suppress(Exception):
            self.on_call(record)

    def _cancelled(self) -> bool:
        """True when the owner has asked for this solve to stop.

        A predicate that raises is read as "keep going". Losing a running solve
        to a broken cancellation check would be a worse failure than the one
        extra call that reading it charitably can cost.
        """
        if self.cancel is None:
            return False
        try:
            return bool(self.cancel())
        except Exception:  # noqa: BLE001 - a broken predicate must not stop work
            return False

    def _get_client(self) -> Any:
        """The Anthropic client, built on first use.

        Built lazily and never at import: a missing key must fail when someone
        actually asks for candidates, not when the CLI is merely loaded.
        """
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            key = api_key()
            if not key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set, so the LLM candidate source "
                    "cannot run. Set it in the environment "
                    "(PowerShell: $env:ANTHROPIC_API_KEY = 'sk-ant-...'; "
                    "bash: export ANTHROPIC_API_KEY=sk-ant-...) or put "
                    "ANTHROPIC_API_KEY=sk-ant-... in a .env file at the project "
                    "root. Run with a lexicon-only ablation to solve offline."
                )
            import anthropic  # imported here so `import xword` stays light

            # max_retries=0: this class does its own backoff so that retries are
            # counted and reported instead of hidden inside the SDK.
            self._client = anthropic.Anthropic(
                api_key=key, max_retries=0, timeout=180.0
            )
            return self._client

    def _record_usage(
        self, *, calls: int = 0, retries: int = 0, failures: int = 0, dropped: int = 0
    ) -> None:
        with self._usage_lock:
            self.usage.calls += calls
            self.usage.retries += retries
            self.usage.failures += failures
            self.usage.dropped += dropped

    def _record_tokens(self, raw_usage: Any) -> None:
        """Fold one response's token counts into the running total.

        Cache-read and cache-creation tokens are counted as input tokens for
        reporting, but priced at their published multipliers (0.1x and 1.25x) so
        the dollar figure does not pretend a cached prefix was free.
        """
        if raw_usage is None:
            return
        in_tok = int(getattr(raw_usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(raw_usage, "output_tokens", 0) or 0)
        cache_read = int(getattr(raw_usage, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(raw_usage, "cache_creation_input_tokens", 0) or 0)
        billable_in = in_tok + int(round(cache_read * 0.1 + cache_write * 1.25))
        cost = estimate_cost(self.model, billable_in, out_tok)
        with self._usage_lock:
            self.usage.input_tokens += in_tok + cache_read + cache_write
            self.usage.output_tokens += out_tok
            self.usage.cost_usd += cost

    # -- retry ------------------------------------------------------------- #

    def _is_retryable(self, exc: BaseException) -> bool:
        """Transient failures only: never retry a 400 that will fail again."""
        if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
            return True
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status in RETRY_STATUSES or status >= 500
        name = type(exc).__name__
        return name in {
            "APIConnectionError",
            "APITimeoutError",
            "RateLimitError",
            "InternalServerError",
        }

    def _sleep_for(self, attempt: int) -> float:
        delay = min(BACKOFF_CAP, BACKOFF_BASE * (2.0**attempt))
        with self._rng_lock:
            jitter = self._rng.uniform(0.0, BACKOFF_JITTER)
        return delay * (1.0 + jitter)

    def _sent_fields(
        self,
        request_kwargs: Mapping[str, Any],
        label: str,
        clue_ids: Sequence[str],
    ) -> dict[str, Any]:
        """The "what was sent" half of a record, read back off the payload.

        Read off the payload rather than off the arguments that built it, so the
        record describes the request that actually went out -- including
        everything :meth:`_request_kwargs` decided on its own: the forced tool,
        the disabled thinking, whether a temperature could be sent at all.

        Every message's text is joined rather than only the first user turn. The
        payload is single-turn today; if that changes the record should grow
        rather than quietly drop a turn.
        """
        messages = request_kwargs.get("messages") or []
        prompt = "\n\n".join(
            text
            for text in (
                _content_text(m.get("content") if isinstance(m, Mapping) else m)
                for m in messages
            )
            if text
        )
        return {
            "id": uuid4().hex[:12],
            "label": label,
            "kind": _call_kind(request_kwargs),
            "model": str(request_kwargs.get("model") or self.model),
            "round": self.round_hint,
            "system": _content_text(request_kwargs.get("system")),
            "prompt": prompt,
            "tools": _tool_names(request_kwargs.get("tools")),
            "tool_choice": _tool_choice_label(request_kwargs.get("tool_choice")),
            "clue_ids": tuple(clue_ids),
        }

    def _call(
        self,
        request_kwargs: dict[str, Any],
        label: str,
        *,
        clue_ids: Sequence[str] = (),
    ) -> Any | None:
        """One API call with exponential backoff; ``None`` once it gives up.

        Also the only place an :class:`~xword.web.trace.LLMCallRecord` is
        produced, and exactly one is produced per invocation: a request that
        failed twice and then succeeded is one record with ``attempts=3``, not
        three records. The UI shows a list of calls, so splitting one logical
        request across three rows would read as three requests, with the
        successful one buried under its own failures. A call that never
        succeeds yields that same single record, carrying the error.

        Which is why the one record has to carry the *reasons* the earlier
        attempts failed, in ``retry_errors``: ``attempts=3`` on its own reports
        that a request was slow without saying what it was waiting on, and the
        narration that does name the exception goes to ``on_event``, which the
        agent does not wire. Losing it there left a web session with no record
        of the failures anywhere -- the retries being exactly the calls a trace
        is worth reading for.

        ``clue_ids`` is passed in because the choke point cannot infer it: the
        prompt has the slot ids in it, but re-parsing them back out to label a
        record would be inventing a second source of truth.
        """
        # No record for this one: nothing was sent, so there is nothing to show,
        # and a stopped session already has a "stopping" status event explaining
        # why its trace ends where it does.
        if self._cancelled():
            return None

        client = self._get_client()
        sent = self._sent_fields(request_kwargs, label, clue_ids)
        started_at = time.time()
        clock = time.perf_counter()
        last: BaseException | None = None
        # Every failure that was retried past, oldest first. ``last`` alone is
        # overwritten each round, so it can only ever describe the final
        # attempt -- and on the path where the call eventually succeeds it
        # describes nothing at all.
        retry_errors: list[str] = []
        cancelled = False
        attempts = 0

        for attempt in range(self.max_retries + 1):
            if attempt and self._cancelled():
                # A retry is a new request, so a stop stops it too.
                cancelled = True
                break
            attempts = attempt + 1
            try:
                message = client.messages.create(**request_kwargs)
            except BaseException as exc:  # noqa: BLE001 - recorded below
                if not self._is_retryable(exc) or attempt == self.max_retries:
                    last = exc
                    break
                last = exc
                retry_errors.append(_failure_text(exc))
                delay = self._sleep_for(attempt)
                self._record_usage(retries=1)
                self._emit(
                    f"{label}: {type(exc).__name__}, retrying in {delay:.1f}s "
                    f"({attempt + 1}/{self.max_retries})"
                )
                time.sleep(delay)
                continue
            self._record_usage(calls=1)
            raw_usage = getattr(message, "usage", None)
            self._record_tokens(raw_usage)
            tool_name, tool_input = _first_tool_use(message)
            self._emit_call(
                LLMCallRecord.build(
                    **sent,
                    stop_reason=str(getattr(message, "stop_reason", "") or ""),
                    tool_name=tool_name,
                    tool_input=tool_input,
                    text=_content_text(getattr(message, "content", None)),
                    started_at=started_at,
                    duration_s=time.perf_counter() - clock,
                    input_tokens=_usage_int(raw_usage, "input_tokens"),
                    output_tokens=_usage_int(raw_usage, "output_tokens"),
                    cache_read_tokens=_usage_int(raw_usage, "cache_read_input_tokens"),
                    cache_write_tokens=_usage_int(raw_usage, "cache_creation_input_tokens"),
                    attempts=attempts,
                    retry_errors=tuple(retry_errors),
                )
            )
            return message

        if cancelled:
            # Not counted as a failure: nothing failed, the session was stopped.
            error = "cancelled: a stop was requested before the retry"
            self._emit(f"{label}: stopped, not retrying")
        else:
            self._record_usage(failures=1)
            # ``error`` is the attempt that ended it; the ones before it are in
            # ``retry_errors``, so the record does not repeat the last failure.
            error = _failure_text(last) if last is not None else "no response"
            self._emit(f"{label}: giving up after {self.max_retries} retries ({last!r})")
        self._emit_call(
            LLMCallRecord.build(
                **sent,
                started_at=started_at,
                duration_s=time.perf_counter() - clock,
                attempts=attempts,
                retry_errors=tuple(retry_errors),
                error=error,
            )
        )
        return None

    # -- request construction ---------------------------------------------- #

    def _request_kwargs(
        self, *, system: str, user: str, hard: bool
    ) -> dict[str, Any]:
        """Build the Messages API payload for one call.

        Two deliberate choices:

        * The batch pass runs with thinking off and the tool call forced. Bulk
          clue answering is recall, not reasoning, and forcing the tool removes
          the failure mode where the model writes an answer in prose.
        * The hard pass leaves thinking on its adaptive default and lets the
          model choose the tool, because the prompt asks it to reason out loud
          first and a forced tool call would cut that off.

        ``temperature`` is the one optional key, and it goes out only when both
        the model and the installed SDK will take it -- see
        :data:`_MODELS_WITHOUT_SAMPLING` for why that is two questions and not
        one. Everything else here is a parameter ``Messages.create`` has had
        throughout the version range ``pyproject`` allows.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            # A cache breakpoint on the system block: every batch in a puzzle
            # shares this prefix. It is a no-op if the prefix is below the
            # model's minimum cacheable size.
            "system": [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user}],
            "tools": [ANSWER_TOOL],
        }
        if hard:
            kwargs["tool_choice"] = {"type": "auto"}
        else:
            kwargs["tool_choice"] = {"type": "tool", "name": ANSWER_TOOL_NAME}
            kwargs["thinking"] = {"type": "disabled"}
        if _supports_sampling(self.model) and _sdk_accepts_temperature():
            kwargs["temperature"] = self.temperature
        return kwargs

    # -- response handling -------------------------------------------------- #

    @staticmethod
    def _tool_payload(message: Any) -> dict[str, Any]:
        """The tool input from a response, or ``{}`` if it never called it."""
        for block in getattr(message, "content", None) or []:
            if getattr(block, "type", None) != "tool_use":
                continue
            if getattr(block, "name", None) != ANSWER_TOOL_NAME:
                continue
            payload = getattr(block, "input", None)
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except ValueError:
                    return {}
            if isinstance(payload, Mapping):
                return dict(payload)
        return {}

    def _clean(
        self, request: ClueRequest, raw: Iterable[Any], rationale: str = ""
    ) -> list[Candidate]:
        """Validate, de-duplicate and score one slot's raw candidates.

        The model ignores instructions occasionally, and a bad answer here is
        worse than no answer: it enters belief propagation with real mass and
        argues for wrong letters in every crossing entry. So everything is
        re-checked locally against the constraints the prompt stated.
        """
        pattern = request.pattern if request.pattern else None
        if pattern is not None and len(pattern) != request.length:
            pattern = None

        best: dict[str, float] = {}
        dropped = 0
        for item in raw or []:
            if not isinstance(item, Mapping):
                dropped += 1
                continue
            answer = _NON_ALPHA_RE.sub("", str(item.get("answer", "")).upper())
            if len(answer) != request.length:
                dropped += 1
                continue
            if pattern is not None and not pattern_matches(pattern.upper(), answer):
                dropped += 1
                continue
            try:
                prob = float(item.get("probability", 0.0))
            except (TypeError, ValueError):
                prob = 0.0
            if not math.isfinite(prob):
                prob = 0.0
            prob = min(max(prob, 0.0), 1.0)
            # A duplicate is not a drop: the model spelled the same answer
            # twice, so keep the more confident reading of it.
            if answer in best:
                best[answer] = max(best[answer], prob)
            else:
                best[answer] = prob

        if dropped:
            self._record_usage(dropped=dropped)

        ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
        if self.k:
            ranked = ranked[: self.k]
        return [
            Candidate(
                answer=answer,
                score=math.log(max(prob, MIN_PROBABILITY)),
                source=self.name,
                rationale=rationale,
            )
            for answer, prob in ranked
        ]

    def _parse_batch(
        self, message: Any, batch: Sequence[ClueRequest]
    ) -> dict[str, list[Candidate]]:
        """Turn one response into validated candidates, keyed by slot id."""
        by_slot = {r.slot_id: r for r in batch}
        payload = self._tool_payload(message)
        out: dict[str, list[Candidate]] = {r.slot_id: [] for r in batch}
        answers = payload.get("answers")
        if not isinstance(answers, list):
            return out
        for entry in answers:
            if not isinstance(entry, Mapping):
                continue
            slot_id = str(entry.get("slot", "")).strip().upper()
            request = by_slot.get(slot_id)
            if request is None:
                # A slot we did not ask about: the model hallucinated an id, or
                # echoed one from another batch. Ignore it rather than let it
                # into the solver.
                continue
            rationale = str(entry.get("analysis", "") or "")[:200]
            out[slot_id] = self._clean(request, entry.get("candidates") or [], rationale)
        return out

    # -- public api --------------------------------------------------------- #

    def propose(self, requests: Sequence[ClueRequest]) -> dict[str, list[Candidate]]:
        """Candidates for every request, batched and issued concurrently."""
        return self._run(requests, hard=False)

    def propose_hard(
        self, requests: Sequence[ClueRequest]
    ) -> dict[str, list[Candidate]]:
        """Candidates for entries the first pass failed on: one call each, with
        the analysis prompt and the crossing clues for context."""
        return self._run(requests, hard=True)

    def _run(
        self, requests: Sequence[ClueRequest], *, hard: bool
    ) -> dict[str, list[Candidate]]:
        unique: dict[str, ClueRequest] = {}
        for request in requests:
            unique[request.slot_id] = request
        results: dict[str, list[Candidate]] = {sid: [] for sid in unique}
        if not unique or self.k == 0:
            return results

        mode = "hard" if hard else self.mode
        pending: list[ClueRequest] = []
        cached_ids: list[str] = []
        for slot_id, request in unique.items():
            hit = None
            if self.cache is not None:
                hit = self.cache.get(
                    cache_key(
                        self.model,
                        request.clue,
                        request.length,
                        request.pattern,
                        mode,
                        context_digest(request.puzzle_meta, request.crossing_clues),
                    )
                )
            if hit is not None:
                results[slot_id] = list(hit)
                cached_ids.append(slot_id)
                with self._usage_lock:
                    self.usage.cache_hits += 1
            else:
                pending.append(request)
                with self._usage_lock:
                    self.usage.cache_misses += 1

        if cached_ids and self.on_call is not None:
            # Without this the trace has an unexplained hole in it: entries that
            # were never asked about, because the answer was already on disk.
            # One record for the whole cached subset rather than one per clue --
            # on a warm run the latter would bury the real calls under dozens of
            # non-calls.
            self._emit_call(
                LLMCallRecord.build(
                    id=uuid4().hex[:12],
                    label="cache hit",
                    kind="cache",
                    model=self.model,
                    round=self.round_hint,
                    system="",
                    prompt="",
                    clue_ids=tuple(cached_ids),
                    started_at=time.time(),
                    cached=True,
                )
            )

        if not pending:
            self._emit(f"{len(unique)} clues, all cached")
            return results

        batches: list[list[ClueRequest]] = (
            [[r] for r in pending]
            if hard
            else [
                pending[i : i + self.batch_size]
                for i in range(0, len(pending), self.batch_size)
            ]
        )
        self._emit(
            f"{len(pending)}/{len(unique)} clues to fetch in {len(batches)} "
            f"{'hard ' if hard else ''}call(s) on {self.model}"
        )

        worker = self._run_hard_batch if hard else self._run_batch
        if len(batches) == 1:
            merged = [worker(batches[0])]
        else:
            with ThreadPoolExecutor(
                max_workers=min(self.max_concurrency, len(batches))
            ) as pool:
                # list(map) keeps submission order, so the merge below is
                # deterministic regardless of which call returns first.
                merged = list(pool.map(worker, batches))

        for batch, produced in zip(batches, merged, strict=True):
            for request in batch:
                candidates = produced.get(request.slot_id, [])
                results[request.slot_id] = candidates
                # Empty lists are not cached: they usually mean a failed call,
                # and baking a failure into the cache would make it permanent.
                if candidates and self.cache is not None:
                    self.cache.put(
                        cache_key(
                            self.model,
                            request.clue,
                            request.length,
                            request.pattern,
                            mode,
                            context_digest(
                                request.puzzle_meta, request.crossing_clues
                            ),
                        ),
                        candidates,
                    )
        return results

    def _run_batch(self, batch: Sequence[ClueRequest]) -> dict[str, list[Candidate]]:
        # Checked here as well as in ``_call``: when a stop arrives most batches
        # of a wide puzzle are still queued in the pool, and this is where they
        # get to exit without building a prompt or spending anything.
        if self._cancelled():
            return {r.slot_id: [] for r in batch}
        meta: Mapping[str, str] = batch[0].puzzle_meta if batch else {}
        kwargs = self._request_kwargs(
            system=SYSTEM_PROMPT,
            user=build_batch_prompt(batch, k=self.k, puzzle_meta=meta),
            hard=False,
        )
        label = f"batch[{batch[0].slot_id}..{batch[-1].slot_id}]"
        message = self._call(kwargs, label, clue_ids=[r.slot_id for r in batch])
        if message is None:
            return {r.slot_id: [] for r in batch}
        return self._parse_batch(message, batch)

    def _run_hard_batch(
        self, batch: Sequence[ClueRequest]
    ) -> dict[str, list[Candidate]]:
        request = batch[0]
        if self._cancelled():
            return {request.slot_id: []}
        kwargs = self._request_kwargs(
            system=HARD_SYSTEM_PROMPT,
            user=build_hard_prompt(
                request, k=self.k, crossing_context=request.crossing_clues
            ),
            hard=True,
        )
        message = self._call(
            kwargs, f"hard[{request.slot_id}]", clue_ids=[request.slot_id]
        )
        if message is None:
            return {request.slot_id: []}
        return self._parse_batch(message, [request])


# --------------------------------------------------------------------------- #
# Offline double
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _FakeUsage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(slots=True)
class _FakeBlock:
    type: str
    name: str
    input: dict[str, Any]


@dataclass(slots=True)
class _FakeMessage:
    content: list[_FakeBlock]
    usage: _FakeUsage
    stop_reason: str = "tool_use"


class _FakeMessages:
    """The ``client.messages`` namespace of :class:`FakeClient`."""

    def __init__(self, owner: FakeClient) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> _FakeMessage:
        return self._owner._create(**kwargs)


class FakeClient:
    """TEST-ONLY stand-in for ``anthropic.Anthropic``.

    Answers from a ``{clue: [(answer, probability), ...]}`` book so the whole
    pipeline -- prompt building, batching, concurrency, validation, caching --
    is exercisable with no network and no key. It parses the clue lines back out
    of the prompt the real builders produced, so a change that breaks the prompt
    format breaks the offline tests too, which is the point.

    ``fail_times`` makes the first N calls raise a retryable error, so the
    backoff and failure accounting can be tested without waiting on a real 429.
    """

    def __init__(
        self,
        answers: Mapping[str, Sequence[tuple[str, float]]],
        *,
        fail_times: int = 0,
        unknown_probability: float = 0.05,
    ) -> None:
        self.answers = {
            " ".join(clue.split()).casefold(): list(items)
            for clue, items in answers.items()
        }
        self.fail_times = int(fail_times)
        self.unknown_probability = unknown_probability
        self.calls: list[dict[str, Any]] = []
        self.prompts: list[str] = []
        self.messages = _FakeMessages(self)
        self._lock = threading.Lock()

    # -- internals ---------------------------------------------------------- #

    def _create(self, **kwargs: Any) -> _FakeMessage:
        with self._lock:
            self.calls.append(kwargs)
            if self.fail_times > 0:
                self.fail_times -= 1
                raise ConnectionError("FakeClient: simulated transient failure")

        prompt = ""
        for message in kwargs.get("messages", []):
            content = message.get("content", "")
            if isinstance(content, str):
                prompt += content
        with self._lock:
            self.prompts.append(prompt)

        answers: list[dict[str, Any]] = []
        for line in prompt.splitlines():
            match = _CLUE_LINE_RE.match(line.strip())
            if match is None:
                continue
            clue = " ".join(match.group("clue").split()).casefold()
            book = self.answers.get(clue)
            if book is None:
                # Unknown clue: return a plausible-looking wrong answer of the
                # right length so validation still has something to chew on.
                book = [("X" * int(match.group("length")), self.unknown_probability)]
            answers.append(
                {
                    "slot": match.group("slot"),
                    "candidates": [
                        {"answer": answer, "probability": prob} for answer, prob in book
                    ],
                }
            )

        payload = {"answers": answers}
        return _FakeMessage(
            content=[
                _FakeBlock(type="tool_use", name=ANSWER_TOOL_NAME, input=payload)
            ],
            usage=_FakeUsage(
                input_tokens=len(prompt) // 4 + 400,
                output_tokens=sum(len(a["candidates"]) for a in answers) * 12,
            ),
        )


__all__ = [
    "FakeClient",
    "LLMCandidateSource",
    "LLMUsage",
    "MAX_OUTPUT_TOKENS",
    "MIN_PROBABILITY",
]

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
"""

from __future__ import annotations

import json
import math
import random
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from xword.candidates.cache import ClueCache, cache_key
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

#: Model families that reject ``temperature``/``top_p``/``top_k`` outright (the
#: 4.6+ generation dropped sampling controls). Sending one is a 400, so the
#: configured temperature is silently not sent to these.
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
        self.usage = LLMUsage()

        self._client = client
        self._client_lock = threading.Lock()
        self._usage_lock = threading.Lock()
        self._rng = random.Random(JITTER_SEED)
        self._rng_lock = threading.Lock()

    # -- plumbing ---------------------------------------------------------- #

    def _emit(self, message: str) -> None:
        if self.on_event is not None:
            self.on_event(message)

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

    def _call(self, request_kwargs: dict[str, Any], label: str) -> Any | None:
        """One API call with exponential backoff; ``None`` once it gives up."""
        client = self._get_client()
        last: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                message = client.messages.create(**request_kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                if not self._is_retryable(exc) or attempt == self.max_retries:
                    last = exc
                    break
                last = exc
                delay = self._sleep_for(attempt)
                self._record_usage(retries=1)
                self._emit(
                    f"{label}: {type(exc).__name__}, retrying in {delay:.1f}s "
                    f"({attempt + 1}/{self.max_retries})"
                )
                time.sleep(delay)
                continue
            self._record_usage(calls=1)
            self._record_tokens(getattr(message, "usage", None))
            return message

        self._record_usage(failures=1)
        self._emit(f"{label}: giving up after {self.max_retries} retries ({last!r})")
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
        if _supports_sampling(self.model):
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
        for slot_id, request in unique.items():
            hit = None
            if self.cache is not None:
                hit = self.cache.get(
                    cache_key(
                        self.model, request.clue, request.length, request.pattern, mode
                    )
                )
            if hit is not None:
                results[slot_id] = list(hit)
                with self._usage_lock:
                    self.usage.cache_hits += 1
            else:
                pending.append(request)
                with self._usage_lock:
                    self.usage.cache_misses += 1

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

        for batch, produced in zip(batches, merged):
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
                        ),
                        candidates,
                    )
        return results

    def _run_batch(self, batch: Sequence[ClueRequest]) -> dict[str, list[Candidate]]:
        meta: Mapping[str, str] = batch[0].puzzle_meta if batch else {}
        kwargs = self._request_kwargs(
            system=SYSTEM_PROMPT,
            user=build_batch_prompt(batch, k=self.k, puzzle_meta=meta),
            hard=False,
        )
        label = f"batch[{batch[0].slot_id}..{batch[-1].slot_id}]"
        message = self._call(kwargs, label)
        if message is None:
            return {r.slot_id: [] for r in batch}
        return self._parse_batch(message, batch)

    def _run_hard_batch(
        self, batch: Sequence[ClueRequest]
    ) -> dict[str, list[Candidate]]:
        request = batch[0]
        kwargs = self._request_kwargs(
            system=HARD_SYSTEM_PROMPT,
            user=build_hard_prompt(
                request, k=self.k, crossing_context=request.crossing_clues
            ),
            hard=True,
        )
        message = self._call(kwargs, f"hard[{request.slot_id}]")
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

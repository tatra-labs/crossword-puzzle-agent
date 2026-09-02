"""Vercel entrypoint: an HTTP surface over the crossword agent.

Why this file exists
--------------------
The project is a CLI and a library. Vercel has nothing to build or serve
without a web entrypoint, which is why deploying the bare repository fails at
the framework-detection step. This module is that entrypoint: Vercel's Python
runtime loads the top-level ``app`` and routes every request to it.

The one constraint that shapes the whole design
-----------------------------------------------
A Vercel Function has a hard wall-clock ceiling -- 10s on Hobby by default, up
to 300s with Fluid Compute enabled. Measured solve times for this agent are
about 11s for a 7x7 and **116-216s for a 15x15**. So:

* Fluid Compute is not optional here. Without it every request but the smallest
  mini is killed mid-solve.
* Even with it, a 21x21 Sunday does not reliably fit, and the API says so
  rather than starting work it cannot finish.
* Every solve is given a wall-clock budget strictly below the function ceiling,
  so the agent returns its best partial grid instead of being killed. The agent
  already honours that budget internally; this just sets it from the deployment
  limit rather than from the library default.

The streaming endpoint exists for the same reason: a 200-second request that
sends nothing looks broken, and the agent's round-by-round trace is the most
interesting thing about it anyway.

Background sessions are a local capability
------------------------------------------
The studio UI wants solves that outlive the request that started them: start a
puzzle, go and read a different one, come back to the trace, stop it from
anywhere. ``SessionManager`` does that with a thread per solve and an
append-only trace log, and it works for exactly as long as the process stays
alive between requests.

A Vercel Function does not. It is frozen once the response is sent, so a thread
started during a request stops making progress, and the next request need not
even reach this instance. So ``durable_sessions`` is false whenever ``VERCEL``
is set: ``POST /api/sessions`` answers 501 there instead of charging for a
trace nobody can read back, and the UI degrades to ``/api/solve/stream``, which
is one solve inside one request. Run the app locally (``uvicorn app:app``) for
the session behaviour.

Filesystem note: everything outside ``/tmp`` is read-only on Vercel, and the
agent writes a SQLite clue cache. ``vercel.json`` points ``XWORD_CACHE_DIR`` at
``/tmp``; this module also passes an explicit cache path so the deployment is
correct even if that env var is missing.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import secrets
import sys
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# The package lives under src/. Vercel installs declared dependencies but does
# not necessarily install *this* project, so make the import work either way.
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Keep every writable path inside /tmp before xword.config is imported: it
# resolves its directories at import time.
os.environ.setdefault("XWORD_CACHE_DIR", "/tmp/xword-cache")
os.environ.setdefault("XWORD_REPORT_DIR", "/tmp/xword-reports")

from fastapi import FastAPI, Header, HTTPException, Request  # noqa: E402
from fastapi.responses import (  # noqa: E402
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel, Field  # noqa: E402

from xword import config as cfg  # noqa: E402
from xword.core.grid import grid_rows, make_puzzle, validate_puzzle  # noqa: E402
from xword.core.types import AgentEvent, Puzzle  # noqa: E402
from xword.web.sessions import SessionInfo, SessionLimit, SessionManager  # noqa: E402

# --------------------------------------------------------------------------- #
# Deployment limits
# --------------------------------------------------------------------------- #

#: Must match ``functions["app.py"].maxDuration`` in vercel.json.
FUNCTION_MAX_SECONDS = int(os.environ.get("XWORD_FUNCTION_MAX_SECONDS", "300"))

#: Head-room left for request parsing, model start-up and response encoding, so
#: the agent stops on its own terms rather than being killed by the platform.
RESERVE_SECONDS = 25

SOLVE_BUDGET = max(5, FUNCTION_MAX_SECONDS - RESERVE_SECONDS)

#: Largest grid accepted. The guard's job is to refuse what cannot work *at
#: all*, not to second-guess the budget: the agent honours ``wall_clock_budget``
#: and returns its best partial grid, so a slow puzzle degrades rather than
#: failing. A standard 15x15 (189 squares, ~78 entries) measured 116-216s and
#: therefore fits inside a 300s function, though not comfortably. A 21x21
#: Sunday is roughly 400 squares and 140 entries and has no chance, so it is
#: rejected before any money is spent on it.
MAX_OPEN_CELLS = int(os.environ.get("XWORD_MAX_OPEN_CELLS", "200"))
MAX_CLUES = int(os.environ.get("XWORD_MAX_CLUES", "90"))

#: Above this, a solve is allowed but flagged in the listing as slower. Set at
#: 55 so the bundled 9x9s and the 11x11 are marked: measured, a 5x5 returns in
#: ~8s and a 7x7 in ~11s, but a 9x9 takes tens of seconds, which is worth
#: warning about before someone clicks and assumes it has hung.
SLOW_OPEN_CELLS = int(os.environ.get("XWORD_SLOW_OPEN_CELLS", "55"))

#: Whether a solve may outlive the request that started it. A session is a
#: thread and a log inside this process, both of which a frozen serverless
#: instance loses, and a follow-up request is not guaranteed to reach the same
#: instance anyway. The env var is a crude platform sniff, but it is the only
#: signal available at import time and the cost of being wrong is one HTTP
#: error and a fallback, not a broken deployment.
DURABLE_SESSIONS = not bool(os.environ.get("VERCEL"))

#: How many solves may run at once. Each is a thread holding an Anthropic key,
#: so this cap is about spend and API rate limits rather than CPU: three
#: concurrent 15x15s is already a few dollars in flight.
MAX_CONCURRENT_SESSIONS = max(1, int(os.environ.get("XWORD_MAX_CONCURRENT_SESSIONS", "3")))


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Take the session threads down with the process.

    A lifespan handler rather than ``@app.on_event("shutdown")``, which is
    deprecated in the installed FastAPI (0.129) and scheduled for removal.
    Without it, a Ctrl-C or a ``--reload`` restart leaves solve threads running
    against a real API key with nothing left to read their traces.
    """
    yield
    manager.shutdown()


app = FastAPI(
    title="crossword-puzzle-agent",
    description="An AI agent that solves crossword puzzles.",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=_lifespan,
)


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #


class SolveRequest(BaseModel):
    """Either a bundled puzzle id, or a grid plus clues supplied inline."""

    puzzle: str | None = Field(
        default=None, description="Bundled puzzle id, e.g. 'mini-01'."
    )
    grid: list[str] | None = Field(
        default=None,
        description="Solution rows ('#' = black square). Enables scoring.",
    )
    shape: list[str] | None = Field(
        default=None,
        description="Shape-only rows ('#' = black square) when no answers are known.",
    )
    across: dict[str, str] | None = Field(
        default=None, description="Across clues keyed by entry number."
    )
    down: dict[str, str] | None = Field(
        default=None, description="Down clues keyed by entry number."
    )
    model: str | None = None
    rounds: int | None = Field(default=None, ge=1, le=6)
    candidates: int | None = Field(default=None, ge=1, le=20)
    seed: int = 0


def _bundled() -> dict[str, Path]:
    return {p.stem: p for p in sorted(cfg.BUNDLED_PUZZLE_DIR.glob("*.json"))}


def _load_request(body: SolveRequest) -> Puzzle:
    """Turn a request into a Puzzle, refusing anything too big to finish."""
    from xword.io.loaders import load_puzzle

    if body.puzzle:
        found = _bundled().get(body.puzzle)
        if found is None:
            raise HTTPException(
                404,
                f"No bundled puzzle {body.puzzle!r}. "
                f"Available: {', '.join(sorted(_bundled()))}",
            )
        puzzle = load_puzzle(found)
    else:
        rows = body.grid or body.shape
        if not rows:
            raise HTTPException(400, "Provide 'puzzle', or 'grid'/'shape' with clues.")
        try:
            puzzle = make_puzzle(
                "submitted",
                rows,
                {int(k): v for k, v in (body.across or {}).items()},
                {int(k): v for k, v in (body.down or {}).items()},
                solution_rows=body.grid,
            )
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, f"Could not read that grid: {exc}") from exc
        problems = [p for p in validate_puzzle(puzzle) if "missing clue" not in p]
        if problems:
            raise HTTPException(400, f"Grid problems: {'; '.join(problems[:5])}")

    open_cells = len(puzzle.open_cells)
    if open_cells > MAX_OPEN_CELLS or len(puzzle.slots) > MAX_CLUES:
        raise HTTPException(
            413,
            f"This puzzle is {puzzle.height}x{puzzle.width} with {open_cells} squares "
            f"and {len(puzzle.slots)} entries. That is too large to finish inside "
            f"this deployment's {FUNCTION_MAX_SECONDS}s function limit, so it is "
            f"refused rather than started and abandoned part-way. The ceiling here "
            f"is {MAX_OPEN_CELLS} squares / {MAX_CLUES} entries, which admits a "
            f"standard 15x15 but not a 21x21 Sunday. Run it locally with no time "
            f"limit instead: xword solve <file>.",
        )
    return puzzle


def _agent_config(body: SolveRequest):
    return cfg.AgentConfig(
        cache_path=Path(os.environ.get("XWORD_CACHE_DIR", "/tmp/xword-cache"))
        / "clue-cache.sqlite",
    ).with_overrides(
        model=body.model,
        max_rounds=body.rounds,
        candidates_per_clue=body.candidates,
        seed=body.seed,
        wall_clock_budget=float(SOLVE_BUDGET),
        search_seconds=min(20.0, SOLVE_BUDGET / 4),
    )


def _serialise(puzzle: Puzzle, result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "puzzle": {
            "id": puzzle.id,
            "width": puzzle.width,
            "height": puzzle.height,
            "title": puzzle.meta.get("title", ""),
            "source": puzzle.meta.get("source", ""),
            "rows": grid_rows(puzzle, {}),
            "numbers": {
                f"{c.row},{c.col}": s.number
                for s in puzzle.slots
                for c in [s.start]
            },
        },
        "fill": grid_rows(puzzle, result.fill.letters),
        "confidence": {
            f"{c.row},{c.col}": round(v, 4)
            for c, v in result.cell_confidence.items()
        },
        "entries": [
            {
                "id": sid,
                "clue": o.clue,
                "answer": o.answer,
                "confidence": round(o.confidence, 4),
                "source": o.source,
                "gold": (puzzle.solution or {}).get(sid),
            }
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
            {"kind": e.kind, "round": e.round, "message": e.message}
            for e in result.trace
        ],
    }
    if puzzle.has_solution:
        from xword.eval.metrics import score_result

        score = score_result(puzzle, result)
        payload["score"] = {
            "solved": score.solved,
            "cells_correct": score.cells_correct,
            "cells_total": score.cells_total,
            "cell_accuracy": round(score.cell_accuracy, 4),
            "words_correct": score.words_correct,
            "words_total": score.words_total,
            "word_accuracy": round(score.word_accuracy, 4),
        }
        payload["gold"] = grid_rows(puzzle, puzzle.solution_letters())
    return payload


#: The session registry, one per process. It is deliberately module-level
#: rather than per-request state: that is the whole point -- a session has to
#: still be there on the *next* request, and it is handed ``_serialise`` so a
#: finished session's ``result`` event is byte-identical to what
#: ``POST /api/solve`` returns for the same puzzle.
manager = SessionManager(serialise=_serialise, max_concurrent=MAX_CONCURRENT_SESSIONS)


def _cap_detail(live: int) -> str:
    """The 429 body for a refused solve, worded like the manager's own sentence.

    One condition should read one way wherever it is hit, and this one is hit
    from three routes. It also has to say that the cap counts *both* kinds of
    solve, because the obvious reading of "3 sessions at once" is that the
    older endpoint is a way around it -- which is exactly the hole this closes.
    """
    return (
        f"{live} solve(s) are already running and this deployment allows "
        f"{MAX_CONCURRENT_SESSIONS} at once. The cap is a spend guard rather than a "
        f"performance one: every concurrent solve is issuing real Anthropic requests "
        f"-- about $0.007 for a 5x5 and $0.65 for a 15x15 -- and it counts background "
        f"sessions and in-request solves together, so POST /api/solve and "
        f"POST /api/solve/stream are not a way around it. Wait for one to finish, or "
        f"stop a running session, and try again."
    )


class _SolveAdmission:
    """Counted admission for the solves that no session registry can see.

    ``SessionManager`` enforces ``max_concurrent`` over its own registry, which
    is where it belongs: a session is a thread the manager owns. But
    ``/api/solve`` and ``/api/solve/stream`` start identical, identically priced
    work without registering anything, so the manager's count is blind to them
    and twenty parallel 15x15s through the older endpoint spend twenty times
    $0.65 while the fourth session is being refused for the same reason.

    This is the missing half, and it is deliberately *added to*
    ``manager.active`` rather than kept as a rival total: every route admits
    against one number, so the two caps cannot drift apart. The manager stays
    the authority on how many sessions are live -- this object only knows about
    the in-request solves, which nothing else does.

    Lock discipline is the manager's, extended by one level: this lock may be
    held while taking the manager's (inside ``manager.active``), never the
    reverse, and it is never held across a solve -- only across the arithmetic.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = 0

    @property
    def live(self) -> int:
        """Paid work in flight: registered sessions plus in-request solves."""
        with self._lock:
            return manager.active + self._running

    def check(self) -> None:
        """Refuse a *session* the combined cap has no room for.

        ``SessionManager.start`` re-applies the cap over its registry under its
        own lock, so this adds only what the registry cannot see. Without it a
        caller could spend the session cap's budget through the legacy routes
        and still be told there was room for another session.
        """
        with self._lock:
            self._refuse_if_full_locked()

    def acquire(self) -> None:
        """Admit one in-request solve, or raise 429. Release it in a ``finally``."""
        with self._lock:
            self._refuse_if_full_locked()
            self._running += 1

    def release(self) -> None:
        with self._lock:
            self._running = max(0, self._running - 1)

    def _refuse_if_full_locked(self) -> None:
        live = manager.active + self._running
        if live >= MAX_CONCURRENT_SESSIONS:
            raise HTTPException(429, _cap_detail(live))


#: One admission gate per process, like the registry it counts alongside.
_admission = _SolveAdmission()


#: Optional shared secret. A deployment of this app is a public URL that spends
#: real money from your Anthropic key on every click, so if this is set the
#: solve endpoints require it (header ``X-Access-Token`` or ``?token=``). Left
#: unset the app is open, which is fine behind Vercel Deployment Protection and
#: risky on a public production domain.
ACCESS_TOKEN = os.environ.get("XWORD_ACCESS_TOKEN", "").strip()


def _require_access(token: str | None) -> None:
    if not ACCESS_TOKEN:
        return
    if not token or not secrets.compare_digest(token, ACCESS_TOKEN):
        raise HTTPException(
            401,
            "This deployment requires an access token, because each solve spends "
            "real Anthropic API credit. Pass it as the X-Access-Token header or a "
            "?token= query parameter.",
        )


def _require_key() -> None:
    if not cfg.api_key():
        raise HTTPException(
            503,
            "ANTHROPIC_API_KEY is not configured on this deployment. Set it with "
            "`vercel env add ANTHROPIC_API_KEY` (or in the project's Environment "
            "Variables settings) and redeploy.",
        )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@app.get("/api/health")
def health() -> JSONResponse:
    """What this deployment can actually do -- the `xword doctor` of the web."""
    from xword.lexicon.store import Lexicon

    lex = Lexicon.default()
    return JSONResponse(
        {
            "ok": True,
            "api_key_configured": bool(cfg.api_key()),
            "access_token_required": bool(ACCESS_TOKEN),
            "bundled_puzzles": sorted(_bundled()),
            "lexicon_entries": len(lex),
            "lexicon_is_fallback": not cfg.DEFAULT_LEXICON_PATH.exists(),
            "model": cfg.DEFAULT_MODEL,
            "function_max_seconds": FUNCTION_MAX_SECONDS,
            "solve_budget_seconds": SOLVE_BUDGET,
            "max_open_cells": MAX_OPEN_CELLS,
            "max_clues": MAX_CLUES,
            # What the UI needs to decide between the session surface and the
            # single-request fallback before the user clicks anything.
            "durable_sessions": DURABLE_SESSIONS,
            "max_concurrent_sessions": MAX_CONCURRENT_SESSIONS,
            "active_sessions": manager.active,
            # Sessions plus the in-request solves the registry never sees.
            # `active_sessions` alone reported 0 while /api/solve/stream burned
            # the deployment's whole budget, which made the bypass invisible in
            # the one place someone would look for it.
            "active_solves": _admission.live,
            # Which Vercel environment this is ("production"/"preview"/
            # "development"), empty when running locally. Worth reporting
            # because the first deployment came up with api_key_configured
            # false while the dashboard showed the key set, and the two
            # questions -- is a key reaching the process, and which
            # environment's variables am I looking at -- could not be told
            # apart from outside.
            "vercel_env": os.environ.get("VERCEL_ENV", ""),
            # Whether the *process environment* carries the key, as distinct
            # from `api_key_configured`, which is true if either the
            # environment or a local .env supplies one. On a deployment there
            # is no .env, so a disagreement here means the platform is not
            # injecting the variable rather than the key being wrong.
            "api_key_in_environment": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "python": sys.version.split()[0],
        }
    )


@app.get("/api/puzzles")
def puzzles() -> JSONResponse:
    """The bundled puzzles, with the sizes that decide what will finish here."""
    from xword.io.loaders import load_puzzle

    out = []
    for pid, path in _bundled().items():
        try:
            p = load_puzzle(path)
        except Exception:  # noqa: BLE001 - a bad fixture must not break the list
            continue
        out.append(
            {
                "id": pid,
                "title": p.meta.get("title", ""),
                "difficulty": p.meta.get("difficulty", ""),
                "size": f"{p.height}x{p.width}",
                "entries": len(p.slots),
                "open_cells": len(p.open_cells),
                "fits_here": len(p.open_cells) <= MAX_OPEN_CELLS
                and len(p.slots) <= MAX_CLUES,
                "slow": len(p.open_cells) > SLOW_OPEN_CELLS,
            }
        )
    # Smallest first, so the default selection in any UI is the one that
    # returns in seconds rather than a 9x9 that takes a minute. Alphabetical
    # order put `maxi-01` at the top, which made the first click the slowest
    # possible experience.
    out.sort(key=lambda p: (p["open_cells"], p["id"]))
    return JSONResponse({"puzzles": out})


def _entries_view(puzzle: Puzzle, direction: str) -> list[dict[str, Any]]:
    """One direction's clue list, numbered order, with each entry's start cell.

    The start cell travels with the clue so the UI can highlight an entry
    without re-deriving the numbering it was already sent.
    """
    return [
        {
            "id": s.id,
            "number": s.number,
            "clue": s.clue,
            "length": s.length,
            "row": s.start.row,
            "col": s.start.col,
        }
        for s in sorted(
            (s for s in puzzle.slots if s.direction == direction),
            key=lambda s: s.number,
        )
    ]


@app.get("/api/puzzles/{pid}")
def puzzle_detail(pid: str) -> JSONResponse:
    """One bundled puzzle in full, so inspecting it costs nothing and no money.

    Everything the grid renderer and the clue list need, and nothing that
    requires a solve. The answers are withheld even when the fixture has them
    -- ``has_solution`` says scoring is possible, and the gold grid arrives
    with a finished solve's result, not before it.
    """
    from xword.io.loaders import load_puzzle

    found = _bundled().get(pid)
    if found is None:
        raise HTTPException(
            404,
            f"No bundled puzzle {pid!r}. Available: {', '.join(sorted(_bundled()))}",
        )
    p = load_puzzle(found)
    open_cells = len(p.open_cells)
    return JSONResponse(
        {
            "id": pid,
            "title": p.meta.get("title", ""),
            "difficulty": p.meta.get("difficulty", ""),
            "size": f"{p.height}x{p.width}",
            "width": p.width,
            "height": p.height,
            "entries": len(p.slots),
            "open_cells": open_cells,
            "fits_here": open_cells <= MAX_OPEN_CELLS and len(p.slots) <= MAX_CLUES,
            "slow": open_cells > SLOW_OPEN_CELLS,
            "has_solution": p.has_solution,
            "shape": grid_rows(p, {}, blank="."),
            "numbers": {f"{s.start.row},{s.start.col}": s.number for s in p.slots},
            "across": _entries_view(p, "across"),
            "down": _entries_view(p, "down"),
        }
    )


@app.post("/api/solve")
def solve(
    body: SolveRequest,
    x_access_token: str | None = Header(default=None),
    token: str | None = None,
) -> JSONResponse:
    """Solve and return the finished grid. Blocking; use /api/solve/stream for progress.

    Subject to the same concurrency cap as a session, and for the same reason:
    the money is spent by the solve, not by the route that started it.
    """
    _require_access(x_access_token or token)
    _require_key()
    puzzle = _load_request(body)
    from xword.solver.agent import CrosswordAgent

    events: list[AgentEvent] = []
    # Admitted after the size guard, so a puzzle that is refused anyway never
    # occupies a slot, and released in a finally so a solve that raises does
    # not leak one and shrink the cap for the life of the process.
    _admission.acquire()
    try:
        agent = CrosswordAgent(_agent_config(body), on_event=events.append)
        result = agent.solve(puzzle)
    finally:
        _admission.release()
    return JSONResponse(_serialise(puzzle, result))


@app.post("/api/solve/stream")
async def solve_stream(
    body: SolveRequest,
    x_access_token: str | None = Header(default=None),
    token: str | None = None,
) -> StreamingResponse:
    """Server-sent events: one per agent step, then the finished grid.

    The solve runs on a worker thread and pushes events through a queue. It has
    to be a thread rather than a task because the agent is synchronous and
    issues its API batches through a thread pool; awaiting it directly would
    block the event loop and starve the very heartbeats this endpoint exists to
    send.

    That thread is also why the concurrency slot is released by the worker and
    not by this handler: the solve has no cancel hook, so it runs to completion
    even after the client hangs up, and a slot handed back at disconnect would
    let one caller hold the deployment's entire spend budget by opening streams
    and dropping them.
    """
    _require_access(x_access_token or token)
    _require_key()
    puzzle = _load_request(body)
    from xword.solver.agent import CrosswordAgent

    channel: queue.Queue[tuple[str, Any]] = queue.Queue()

    def run() -> None:
        try:
            agent = CrosswordAgent(
                _agent_config(body),
                on_event=lambda e: channel.put(
                    ("event", {"kind": e.kind, "round": e.round, "message": e.message})
                ),
            )
            result = agent.solve(puzzle)
            channel.put(("result", _serialise(puzzle, result)))
        except Exception as exc:  # noqa: BLE001 - must reach the client as data
            channel.put(("error", {"message": f"{type(exc).__name__}: {exc}"}))
        finally:
            channel.put(("done", None))
            _admission.release()

    _admission.acquire()
    worker = threading.Thread(target=run, daemon=True)
    try:
        worker.start()
    except RuntimeError:  # the OS refused a thread; do not leak the slot
        _admission.release()
        raise

    async def stream() -> Iterator[str]:
        started = time.monotonic()
        yield _sse("meta", {
            "puzzle_id": puzzle.id,
            "size": f"{puzzle.height}x{puzzle.width}",
            "entries": len(puzzle.slots),
            "budget_seconds": SOLVE_BUDGET,
        })
        while True:
            try:
                kind, data = channel.get_nowait()
            except queue.Empty:
                # A comment frame keeps proxies from closing an idle stream
                # during the long first round.
                if time.monotonic() - started > FUNCTION_MAX_SECONDS:
                    yield _sse("error", {"message": "deployment time limit reached"})
                    return
                yield ": keepalive\n\n"
                await asyncio.sleep(0.5)
                continue
            if kind == "done":
                return
            yield _sse(kind, data)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# --------------------------------------------------------------------------- #
# Sessions
#
# The same solve as /api/solve, except the work is owned by the process instead
# of by one response. Starting is a POST that returns immediately; watching is
# a separate, resumable subscription; and stopping is a third request that may
# come from a page which has since navigated elsewhere.
#
# Every session route requires the access token when one is configured --
# reading included. A trace is not metadata: it carries the verbatim system
# prompt and clue batches sent to Anthropic, the model's answers, the cost, and
# for an inline puzzle the solution its submitter supplied. Leaving the reads
# open let an anonymous visitor list the ids and then replay everything the
# token holder paid for. /api/health, / and the two static files stay open,
# because the smoke test has to work and the page has to load before anyone
# can be asked for a token.
# --------------------------------------------------------------------------- #


def _session_or_404(sid: str) -> SessionInfo:
    info = manager.info(sid)
    if info is None:
        raise HTTPException(
            404,
            f"No session {sid!r}. It was deleted, or this process was restarted -- "
            "sessions live in memory and do not survive a redeploy.",
        )
    return info


@app.post("/api/sessions")
def create_session(
    body: SolveRequest,
    x_access_token: str | None = Header(default=None),
    token: str | None = None,
) -> JSONResponse:
    """Start a solve in the background and return the session immediately.

    Shares ``_load_request`` and ``_agent_config`` with ``/api/solve``, so the
    size guard, the wall-clock budget and the model overrides cannot drift
    between the two ways of starting the same work.

    The durability check comes before the API key check on purpose: on a
    platform that cannot hold a session, a missing key is not the interesting
    problem, and answering 503 would send the UI looking for a configuration
    fix that would not help.
    """
    _require_access(x_access_token or token)
    if not DURABLE_SESSIONS:
        raise HTTPException(
            501,
            "This deployment cannot run background sessions. A serverless function "
            "is frozen once it responds, so the solve would stall the moment this "
            "request returned, and a later poll could land on a different instance "
            "that has never heard of the session. Use POST /api/solve/stream, which "
            "completes a solve inside a single request, or run the app locally "
            "(uvicorn app:app) for the session UI.",
        )
    _require_key()
    puzzle = _load_request(body)
    # The manager re-applies the cap over its own registry under its own lock;
    # this adds the in-request solves the registry cannot see, so a caller
    # cannot spend the session budget through /api/solve and still be told
    # there is room here.
    _admission.check()
    try:
        info = manager.start(puzzle, _agent_config(body))
    except SessionLimit as exc:
        raise HTTPException(429, str(exc)) from exc
    return JSONResponse(info.as_dict())


@app.get("/api/sessions")
def list_sessions(
    x_access_token: str | None = Header(default=None),
    token: str | None = None,
) -> JSONResponse:
    """Every session this process knows about, newest first.

    Token-guarded like the rest: this listing is where an id comes from, and an
    id is all the other reads need.
    """
    _require_access(x_access_token or token)
    return JSONResponse(
        {
            "sessions": [i.as_dict() for i in manager.list()],
            "max_concurrent": MAX_CONCURRENT_SESSIONS,
            "durable": DURABLE_SESSIONS,
        }
    )


@app.get("/api/sessions/{sid}")
def session(
    sid: str,
    x_access_token: str | None = Header(default=None),
    token: str | None = None,
) -> JSONResponse:
    """One session, plus its finished grid once there is one.

    ``result`` is null until the solve completes, which is what lets the UI use
    a single request to restore the whole right-hand pane -- header, progress
    and grid -- when a puzzle is reopened. That grid includes the gold answers
    for a puzzle submitted with a solution, which is one of the reasons the
    token is required here.
    """
    _require_access(x_access_token or token)
    info = _session_or_404(sid)
    return JSONResponse({"session": info.as_dict(), "result": manager.result(sid)})


@app.get("/api/sessions/{sid}/events")
def session_events(
    sid: str,
    cursor: int = 0,
    x_access_token: str | None = Header(default=None),
    token: str | None = None,
) -> JSONResponse:
    """Poll fallback for clients that cannot hold an SSE stream open.

    Non-blocking by design: a long-poll would tie up a worker for as long as a
    round takes, and it buys nothing here, because the cursor already
    guarantees that a client polling every few seconds misses no events.
    """
    _require_access(x_access_token or token)
    info = _session_or_404(sid)
    log = manager.log(sid)
    if log is None:
        raise HTTPException(404, f"Session {sid!r} has no trace log.")
    at = max(0, cursor)
    # `closed` and `dropped` are sampled *before* the slice, and this order is
    # load-bearing: the writer appends `result`, then the terminal `status`,
    # then closes, so reading the flag afterwards can pair `closed: true` with
    # an events array captured before either landed -- and a client that trusts
    # `closed` stops polling with the two events it most wanted still on the
    # server. Sampling first can only err the harmless way round: a log that
    # closes mid-handler reports `closed: false` with the events, and the next
    # poll reports the close with nothing left to carry. Same hazard, same fix
    # and same ordering as the `closed` read in /stream below.
    #
    # `dropped` is sampled on the other side of the slice for the identical
    # reason: a drop evicts the oldest retained events, so a count taken before
    # the slice can miss an eviction that the slice already suffered and tell a
    # client its cursor gap is complete when it is not. The rule is one rule --
    # sample each terminal fact on the side where being wrong is the
    # conservative kind of wrong -- and for `closed` that is before, for
    # `dropped` after.
    closed = log.closed
    events = log.since(at)
    dropped = log.dropped
    # The session state is read *after* the events for the mirror-image reason:
    # the other order can report "running" alongside the result event, and a
    # client that trusts `state` over `closed` would stop one beat too early.
    info = manager.info(sid) or info
    return JSONResponse(
        {
            "events": [e.as_dict() for e in events],
            "cursor": events[-1].seq if events else at,
            "session": info.as_dict(),
            "closed": closed,
            "dropped": dropped,
        }
    )


@app.get("/api/sessions/{sid}/stream")
async def session_stream(
    sid: str,
    request: Request,
    cursor: int = 0,
    x_access_token: str | None = Header(default=None),
    token: str | None = None,
) -> StreamingResponse:
    """Replay the trace from ``cursor``, then tail it live until the log closes.

    Reattaching is not a special case. A client that has seen 40 events asks
    for everything since 40 and gets the gap followed by the tail, through the
    same loop a client joining at 0 runs.

    Three details this has to get right:

    * ``TraceLog.wait_since`` blocks its thread. Called inline it would park
      the event loop for up to a second at a time, stalling every other
      request in the process, so it goes through ``asyncio.to_thread``.
    * A subscriber who closes the tab leaves an async generator waiting on a
      log that may never close. Checking ``request.is_disconnected()`` each
      turn is what stops that becoming a thread leak per abandoned tab.
    * The token, when the deployment has one, arrives either as the
      ``X-Access-Token`` header or as ``?token=``. The query parameter is not
      redundant here: ``EventSource`` cannot set a request header, so it is the
      only way a browser can authenticate this route at all.
    """
    _require_access(x_access_token or token)
    log = manager.log(sid)
    if log is None:
        raise HTTPException(
            404,
            f"No session {sid!r}. It was deleted, or this process was restarted.",
        )

    async def stream() -> AsyncIterator[str]:
        at = max(0, cursor)
        for event in log.since(at):
            at = event.seq
            yield _sse(event.type, event.as_dict())
        while True:
            if await request.is_disconnected():
                return
            # `closed` is sampled BEFORE the wait, and that order is the whole
            # correctness argument of this loop -- the obvious tidy-up, reading
            # it in the `if` below, is the bug it replaced.
            #
            # wait_since returns nothing both on timeout and on a drained
            # closed log, so a flag is the only way to tell those apart. Read
            # after the wait, the flag reports a fact learned later than the
            # last look at the events, and the gap between the two swallows
            # whatever was appended in it. That gap is not a rare one: the
            # writer appends `result`, then the terminal `status`, then closes,
            # so the events lost are exactly the two the subscriber is waiting
            # for, and the client -- told the stream is over -- never
            # reattaches to collect them.
            #
            # Sampled first, the flag can only be stale in the safe direction:
            # a log that closes during the wait looks live for one more turn,
            # and that turn drains it and then declares the close. So there is
            # no interleaving in which this generator says `closed` while the
            # log still holds an event above `at`.
            was_closed = log.closed
            pending = await asyncio.to_thread(log.wait_since, at, 1.0)
            for event in pending:
                at = event.seq
                yield _sse(event.type, event.as_dict())
            if not was_closed:
                if not pending:
                    # Idle: proxies (and Vercel) close a stream that says
                    # nothing.
                    yield ": keepalive\n\n"
                continue
            # The log was already closed when this turn began, so wait_since
            # returned everything above the cursor. Drain once more regardless:
            # TraceLog.append has no closed guard, so a late writer is possible,
            # and re-draining costs one uncontended lock while getting it wrong
            # costs the subscriber the end of its trace.
            for event in log.since(at):
                at = event.seq
                yield _sse(event.type, event.as_dict())
            yield _sse("closed", {"cursor": at, "dropped": log.dropped})
            return

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@app.post("/api/sessions/{sid}/stop")
def stop_session(
    sid: str,
    x_access_token: str | None = Header(default=None),
    token: str | None = None,
) -> JSONResponse:
    """Ask a running solve to give up at its next checkpoint.

    Cancellation is cooperative -- an in-flight model call is not aborted -- so
    ``stopped`` means the request was accepted, not that the thread has already
    finished. The returned session is read after the request, so a UI can show
    ``stopping`` straight away rather than inventing that state itself.
    """
    _require_access(x_access_token or token)
    info = _session_or_404(sid)
    stopped = manager.stop(sid)
    return JSONResponse(
        {"stopped": stopped, "session": (manager.info(sid) or info).as_dict()}
    )


@app.delete("/api/sessions/{sid}")
def delete_session(
    sid: str,
    x_access_token: str | None = Header(default=None),
    token: str | None = None,
) -> JSONResponse:
    """Drop a session and its trace from the registry.

    What happens to a still-running solve is the manager's business; this layer
    only reports whether the id existed, so a double-click on a delete button
    gets a 404 rather than a second, different answer.
    """
    _require_access(x_access_token or token)
    if not manager.delete(sid):
        raise HTTPException(404, f"No session {sid!r}.")
    return JSONResponse({"deleted": True, "id": sid})


# --------------------------------------------------------------------------- #
# Static assets
# --------------------------------------------------------------------------- #

#: The studio's two assets, hardcoded rather than reached through a
#: ``/static/{name}`` path parameter. A parameterised route is one ``..`` away
#: from serving the repository, and two files are not worth the traversal
#: guard that would be needed to make it safe.
_STATIC_FILES: dict[str, str] = {
    "studio.css": "text/css",
    "studio.js": "text/javascript",
}


def _static(name: str) -> Response:
    """Serve one file out of ``public/``, from the function like ``index`` does."""
    path = _ROOT / "public" / name
    if not path.is_file():
        raise HTTPException(404, f"{name} is not part of this build.")
    return Response(path.read_bytes(), media_type=_STATIC_FILES[name])


# Both the root and the /static/ path, deliberately.
#
# Vercel treats a top-level ``public/`` directory as the static output: it
# serves those files from the CDN at the root and hoists them out of the
# function's filesystem. So on a deployment ``/studio.css`` is a cached CDN hit
# that never wakes this function, while ``/static/studio.css`` is a file the
# function can no longer open -- which is exactly how the first deployment
# shipped a page whose CSS and JS both 404'd. The page therefore links the root
# paths, and these routes exist so that the same page works when uvicorn is
# serving it locally, where nothing hoists anything.
@app.get("/studio.css")
@app.get("/static/studio.css")
def studio_css() -> Response:
    return _static("studio.css")


@app.get("/studio.js")
@app.get("/static/studio.js")
def studio_js() -> Response:
    return _static("studio.js")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """The demo page. Served from the function so there is a single deployable."""
    page = _ROOT / "public" / "index.html"
    if not page.exists():
        return HTMLResponse(
            "<h1>crossword-puzzle-agent</h1><p>API is up. See "
            "<a href='/api/docs'>/api/docs</a>.</p>"
        )
    return HTMLResponse(page.read_text(encoding="utf-8"))


__all__ = ["app"]

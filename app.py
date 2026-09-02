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
from collections.abc import Iterator
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

from fastapi import FastAPI, Header, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from xword import config as cfg  # noqa: E402
from xword.core.grid import grid_rows, make_puzzle, validate_puzzle  # noqa: E402
from xword.core.types import AgentEvent, Puzzle  # noqa: E402

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

app = FastAPI(
    title="crossword-puzzle-agent",
    description="An AI agent that solves crossword puzzles.",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
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


@app.post("/api/solve")
def solve(
    body: SolveRequest,
    x_access_token: str | None = Header(default=None),
    token: str | None = None,
) -> JSONResponse:
    """Solve and return the finished grid. Blocking; use /api/solve/stream for progress."""
    _require_access(x_access_token or token)
    _require_key()
    puzzle = _load_request(body)
    from xword.solver.agent import CrosswordAgent

    events: list[AgentEvent] = []
    agent = CrosswordAgent(_agent_config(body), on_event=events.append)
    result = agent.solve(puzzle)
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

    worker = threading.Thread(target=run, daemon=True)
    worker.start()

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

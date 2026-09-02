"""Endpoint tests for the studio's HTTP surface in :mod:`app`.

Why none of these tests builds an agent
---------------------------------------
Every solve spends real Anthropic credit -- about $0.007 for a 5x5 and $0.65
for a 15x15 -- and ``cfg.api_key()`` is configured on a developer machine, so a
test that reached ``POST /api/sessions`` with the real registry installed would
silently buy a puzzle. Three independent guards stop that, and
``test_the_no_network_guards_are_armed`` asserts all three are live rather than
trusting the fixture that installs them:

* ``app.manager`` is replaced with :class:`FakeManager`, so ``start`` records
  the call instead of spawning a solve thread.
* ``xword.solver.agent.CrosswordAgent`` is replaced with a class that raises on
  construction. Both solve endpoints and the session manager import it lazily,
  inside the function, so patching the module attribute catches all of them --
  including the legacy ``/api/solve`` path these tests never call on purpose.
* httpx's real transports raise. ``TestClient`` reaches the app through
  ``ASGITransport``, which is a different class, so this cuts outbound HTTP
  without cutting the tests.

The one test that drives the real :class:`SessionManager` injects a stub agent
through its ``agent_factory`` seam, which exists for exactly this reason.

Why module constants are patched rather than the environment
------------------------------------------------------------
``ACCESS_TOKEN``, ``DURABLE_SESSIONS`` and the size ceilings are read from the
environment once, at import time, into module-level constants. Setting an
environment variable inside a test would therefore change nothing, and editing
the real ``.env`` would change the developer's machine, so the tests patch the
constants themselves.

Why the numbering is re-derived here
------------------------------------
``/api/puzzles/{pid}`` is the only thing the inspect view has to draw a grid
from, and asserting its ``numbers`` against ``xword.core.grid`` -- the code that
produced them -- would pass even if both were wrong together. So the expected
numbering is derived from the returned ``shape`` alone, by the standard
convention, in :func:`_numbering_from_shape`.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

# ``app.py`` lives at the repository root, which is on ``sys.path`` under
# ``python -m pytest`` (the ``-m`` puts the working directory there) but not
# under a bare ``pytest``. Adding it explicitly makes the module importable
# either way, the same way ``app.py`` itself makes ``src/`` importable.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import app as webapp  # noqa: E402  - the module and its FastAPI instance share a name
from xword import config as cfg  # noqa: E402
from xword.core.types import (  # noqa: E402
    AgentEvent,
    Fill,
    SlotOutcome,
    SolveResult,
    SolveStats,
)
from xword.web.sessions import SessionInfo, SessionLimit, SessionManager  # noqa: E402
from xword.web.trace import TERMINAL_STATES, LLMCallRecord, TraceLog  # noqa: E402

BLOCK = "#"

#: Ids read from the data directory rather than from ``app._bundled()``, so a
#: bug that hides a fixture from the listing fails a test instead of quietly
#: shrinking the parametrisation with it.
PUZZLE_IDS: tuple[str, ...] = tuple(
    sorted(p.stem for p in cfg.BUNDLED_PUZZLE_DIR.glob("*.json"))
)

#: ``/api/health`` is consumed by the UI *and* documented in the README, so the
#: pre-existing keys are named here explicitly: a rename is as breaking as a
#: removal, and neither shows up in a test that only counts keys.
LEGACY_HEALTH_KEYS: tuple[str, ...] = (
    "ok",
    "api_key_configured",
    "access_token_required",
    "bundled_puzzles",
    "lexicon_entries",
    "lexicon_is_fallback",
    "model",
    "function_max_seconds",
    "solve_budget_seconds",
    "max_open_cells",
    "max_clues",
    "python",
)

NEW_HEALTH_KEYS: tuple[str, ...] = (
    "durable_sessions",
    "max_concurrent_sessions",
    "active_sessions",
)

#: The frozen ``SessionInfo`` contract, in the order it was frozen in. Asserted
#: as a subset, so a later convenience field (the manager already adds
#: ``terminal``) is compatible while dropping one is not.
SESSION_FIELDS: tuple[str, ...] = (
    "id",
    "puzzle_id",
    "title",
    "size",
    "entries",
    "open_cells",
    "state",
    "created_at",
    "started_at",
    "finished_at",
    "round",
    "step",
    "message",
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "elapsed_s",
    "cursor",
    "error",
    "model",
    "max_rounds",
    "solved",
    "cells_correct",
    "cells_total",
)


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #


class ForbiddenAgent:
    """Stands where ``CrosswordAgent`` stands, and refuses to be built.

    A test that manages to construct this has found a path from an endpoint to
    a paid solve, which is worth a loud failure rather than a slow one.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "a test tried to construct the real CrosswordAgent, which would "
            "spend Anthropic credit"
        )


class FakeManager:
    """A ``SessionManager`` stand-in that records instead of solving.

    It implements only what ``app.py`` calls, and it mimics the two behaviours
    the endpoints lean on: ``stop`` reports whether it changed anything, and
    ``delete`` refuses an id it does not hold. Everything else is a canned
    answer a test sets up with :meth:`add`.
    """

    def __init__(self) -> None:
        self.sessions: dict[str, SessionInfo] = {}
        self.logs: dict[str, TraceLog] = {}
        self.results: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []
        self.started: list[tuple[Any, Any]] = []
        self.stopped: list[str] = []
        self.deleted: list[str] = []
        self.shutdowns = 0
        self.active_value = 0
        self.limit_message: str | None = None

    # -- test-side setup --------------------------------------------------- #

    def add(
        self,
        sid: str,
        *,
        state: str = "running",
        with_log: bool = True,
        result: dict[str, Any] | None = None,
        **fields: Any,
    ) -> SessionInfo:
        info = SessionInfo(
            id=sid,
            puzzle_id=fields.pop("puzzle_id", "mini-01"),
            title=fields.pop("title", "First Steps"),
            size=fields.pop("size", "5x5"),
            entries=fields.pop("entries", 10),
            open_cells=fields.pop("open_cells", 19),
            state=state,  # type: ignore[arg-type]
            created_at=fields.pop("created_at", 1_700_000_000.0),
            **fields,
        )
        self.sessions[sid] = info
        self.order.append(sid)
        if with_log:
            self.logs[sid] = TraceLog()
        if result is not None:
            self.results[sid] = result
        return info

    # -- the surface app.py uses ------------------------------------------- #

    def start(self, puzzle: Any, config: Any) -> SessionInfo:
        if self.limit_message is not None:
            raise SessionLimit(self.limit_message)
        self.started.append((puzzle, config))
        return self.add(
            f"fake{len(self.started)}",
            state="queued",
            puzzle_id=getattr(puzzle, "id", "mini-01"),
            model=getattr(config, "model", "claude-sonnet-5"),
            max_rounds=getattr(config, "max_rounds", 1),
        )

    def list(self) -> list[SessionInfo]:
        return [self.sessions[sid] for sid in self.order if sid in self.sessions]

    def info(self, sid: str) -> SessionInfo | None:
        return self.sessions.get(sid)

    def log(self, sid: str) -> TraceLog | None:
        return self.logs.get(sid)

    def result(self, sid: str) -> dict[str, Any] | None:
        return self.results.get(sid)

    def stop(self, sid: str) -> bool:
        self.stopped.append(sid)
        info = self.sessions.get(sid)
        if info is None or info.state not in {"queued", "running"}:
            return False
        info.state = "stopping"
        info.message = "stop requested"
        return True

    def delete(self, sid: str) -> bool:
        self.deleted.append(sid)
        if sid not in self.sessions:
            return False
        del self.sessions[sid]
        self.logs.pop(sid, None)
        self.results.pop(sid, None)
        return True

    def shutdown(self) -> None:
        self.shutdowns += 1

    @property
    def active(self) -> int:
        return self.active_value


class StubAgent:
    """Narrates a solve without making one, so a session can finish for free.

    It emits one ``AgentEvent`` per phase and one :class:`LLMCallRecord`,
    because the record is the thing the session surface exists to expose and a
    trace without one would not prove the wiring.
    """

    def __init__(self, *, on_event: Any, on_llm_call: Any) -> None:
        self._on_event = on_event
        self._on_llm_call = on_llm_call

    def solve(self, puzzle: Any) -> SolveResult:
        self._on_event(
            AgentEvent(kind="ingest", round=0, message=f"read {puzzle.id}", data={})
        )
        self._on_llm_call(
            LLMCallRecord.build(
                id="call-1",
                label="clue batch 1",
                kind="batch",
                model="stub-model",
                round=0,
                system="You solve crossword clues.",
                prompt="1A | len 3 | pat ??? | Clumsy dolt",
                tools=("submit_answers",),
                tool_choice="tool:submit_answers",
                clue_ids=("1A",),
                stop_reason="tool_use",
                tool_name="submit_answers",
                tool_input={"answers": [{"slot": "1A", "answers": ["OAF"]}]},
                input_tokens=120,
                output_tokens=30,
            )
        )
        self._on_event(
            AgentEvent(kind="commit", round=1, message="committed", data={"cells": 0})
        )
        return SolveResult(
            puzzle_id=puzzle.id,
            fill=Fill({}),
            cell_confidence={},
            slots={
                slot.id: SlotOutcome(slot.id, slot.clue, None, 0.0, "none", 1)
                for slot in puzzle.slots
            },
            stats=SolveStats(
                rounds=1,
                llm_calls=1,
                input_tokens=120,
                output_tokens=30,
                wall_seconds=0.01,
                cost_usd=0.00054,
            ),
            trace=[AgentEvent(kind="done", round=1, message="finished", data={})],
        )


def _stub_agent_factory(
    _config: Any, *, on_event: Any, cancel: Any, on_llm_call: Any
) -> StubAgent:
    """The ``agent_factory`` seam, used with the real manager and no API key."""
    assert callable(cancel)
    return StubAgent(on_event=on_event, on_llm_call=on_llm_call)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _no_real_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make a paid solve, and any outbound HTTP at all, impossible.

    Autouse rather than opt-in: the expensive mistake here is forgetting the
    guard on one test, and the guard costs nothing on the tests that do not
    need it.
    """

    def _forbid_network(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a test opened a real network connection")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _forbid_network)
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", _forbid_network
    )
    monkeypatch.setattr("xword.solver.agent.CrosswordAgent", ForbiddenAgent)


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> FakeManager:
    fake = FakeManager()
    monkeypatch.setattr(webapp, "manager", fake)
    return fake


@pytest.fixture
def client() -> TestClient:
    """A client that does *not* run the lifespan.

    Entering ``TestClient`` as a context manager runs the shutdown hook on
    teardown, which reaches whatever ``app.manager`` is by then and so couples
    every test to fixture teardown order. The hook gets its own test instead.
    """
    return TestClient(webapp.app)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _numbering_from_shape(
    shape: list[str],
) -> tuple[dict[str, int], dict[str, tuple[int, int, int]]]:
    """Re-derive the entry numbering from shape rows alone.

    Returns ``{"row,col": number}`` and ``{slot_id: (row, col, length)}`` under
    the standard convention: a cell starts an entry when the square before it in
    that direction is missing or blocked and the square after it is open, and
    numbers run in reading order.
    """
    height = len(shape)
    width = len(shape[0]) if shape else 0

    def is_open(row: int, col: int) -> bool:
        if not (0 <= row < height and 0 <= col < width):
            return False
        return shape[row][col] != BLOCK

    numbers: dict[str, int] = {}
    entries: dict[str, tuple[int, int, int]] = {}
    number = 0
    for row in range(height):
        for col in range(width):
            if not is_open(row, col):
                continue
            across = not is_open(row, col - 1) and is_open(row, col + 1)
            down = not is_open(row - 1, col) and is_open(row + 1, col)
            if not (across or down):
                continue
            number += 1
            numbers[f"{row},{col}"] = number
            if across:
                length = 0
                while is_open(row, col + length):
                    length += 1
                entries[f"{number}A"] = (row, col, length)
            if down:
                length = 0
                while is_open(row + length, col):
                    length += 1
                entries[f"{number}D"] = (row, col, length)
    return numbers, entries


def _frames(body: str) -> list[tuple[str, Any]]:
    """Split an SSE body into ``(event name, decoded data)``, dropping comments."""
    out: list[tuple[str, Any]] = []
    for block in body.split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        name = ""
        data = ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        out.append((name, json.loads(data)))
    return out


# --------------------------------------------------------------------------- #
# /api/health
# --------------------------------------------------------------------------- #


def test_health_keeps_every_pre_existing_key_and_adds_the_session_ones(
    client: TestClient, manager: FakeManager
) -> None:
    body = client.get("/api/health").json()

    missing = [key for key in LEGACY_HEALTH_KEYS + NEW_HEALTH_KEYS if key not in body]
    assert missing == []

    assert body["ok"] is True
    assert isinstance(body["api_key_configured"], bool)
    assert body["access_token_required"] is bool(webapp.ACCESS_TOKEN)
    assert body["bundled_puzzles"] == list(PUZZLE_IDS)
    assert isinstance(body["lexicon_entries"], int)
    assert isinstance(body["lexicon_is_fallback"], bool)
    assert body["model"] == cfg.DEFAULT_MODEL
    assert body["function_max_seconds"] == webapp.FUNCTION_MAX_SECONDS
    assert body["solve_budget_seconds"] == webapp.SOLVE_BUDGET
    assert body["max_open_cells"] == webapp.MAX_OPEN_CELLS
    assert body["max_clues"] == webapp.MAX_CLUES
    assert body["python"].startswith("3.")

    assert body["durable_sessions"] is webapp.DURABLE_SESSIONS
    assert body["max_concurrent_sessions"] == webapp.MAX_CONCURRENT_SESSIONS
    assert body["active_sessions"] == 0


def test_health_reports_the_live_session_count(
    client: TestClient, manager: FakeManager
) -> None:
    manager.active_value = 2
    assert client.get("/api/health").json()["active_sessions"] == 2


def test_health_reports_a_non_durable_deployment(
    client: TestClient, manager: FakeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(webapp, "DURABLE_SESSIONS", False)
    assert client.get("/api/health").json()["durable_sessions"] is False


# --------------------------------------------------------------------------- #
# /api/puzzles
# --------------------------------------------------------------------------- #


def test_puzzles_are_listed_smallest_first(client: TestClient) -> None:
    """Alphabetical order once put the 11x11 first, which made the default
    click the slowest possible experience."""
    rows = client.get("/api/puzzles").json()["puzzles"]
    assert [(p["open_cells"], p["id"]) for p in rows] == sorted(
        (p["open_cells"], p["id"]) for p in rows
    )
    assert rows[0]["slow"] is False


def test_puzzles_flag_what_fits_and_what_is_slow(client: TestClient) -> None:
    rows = client.get("/api/puzzles").json()["puzzles"]
    assert {p["id"] for p in rows} == set(PUZZLE_IDS)
    for row in rows:
        assert set(row) == {
            "id",
            "title",
            "difficulty",
            "size",
            "entries",
            "open_cells",
            "fits_here",
            "slow",
        }
        assert row["fits_here"] is (
            row["open_cells"] <= webapp.MAX_OPEN_CELLS
            and row["entries"] <= webapp.MAX_CLUES
        )
        assert row["slow"] is (row["open_cells"] > webapp.SLOW_OPEN_CELLS)
    assert any(row["slow"] for row in rows), "no fixture exercises the slow flag"


# --------------------------------------------------------------------------- #
# /api/puzzles/{pid}
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pid", PUZZLE_IDS)
def test_puzzle_detail_shape_is_the_grid_and_nothing_else(
    client: TestClient, pid: str
) -> None:
    body = client.get(f"/api/puzzles/{pid}").json()
    shape = body["shape"]
    assert len(shape) == body["height"]
    assert {len(row) for row in shape} == {body["width"]}
    assert set("".join(shape)) <= {BLOCK, "."}
    assert body["size"] == f"{body['height']}x{body['width']}"
    assert body["open_cells"] == sum(row.count(".") for row in shape)


@pytest.mark.parametrize("pid", PUZZLE_IDS)
def test_puzzle_detail_numbers_land_on_entry_starts(
    client: TestClient, pid: str
) -> None:
    body = client.get(f"/api/puzzles/{pid}").json()
    expected_numbers, _ = _numbering_from_shape(body["shape"])
    assert body["numbers"] == expected_numbers


@pytest.mark.parametrize("pid", PUZZLE_IDS)
def test_puzzle_detail_clue_lists_are_number_sorted_and_complete(
    client: TestClient, pid: str
) -> None:
    body = client.get(f"/api/puzzles/{pid}").json()
    _, expected = _numbering_from_shape(body["shape"])

    seen: dict[str, tuple[int, int, int]] = {}
    for direction, suffix in (("across", "A"), ("down", "D")):
        entries = body[direction]
        assert [e["number"] for e in entries] == sorted(e["number"] for e in entries)
        for entry in entries:
            assert entry["id"] == f"{entry['number']}{suffix}"
            assert entry["clue"].strip(), f"{pid} {entry['id']} has no clue"
            seen[entry["id"]] = (entry["row"], entry["col"], entry["length"])
            assert body["numbers"][f"{entry['row']},{entry['col']}"] == entry["number"]

    assert seen == expected
    assert body["entries"] == len(expected)


def test_puzzle_detail_withholds_the_answers(client: TestClient) -> None:
    """``has_solution`` says scoring is possible; it must not hand the grid over."""
    body = client.get("/api/puzzles/mini-01").json()
    assert body["has_solution"] is True
    assert set(body) == {
        "id",
        "title",
        "difficulty",
        "size",
        "width",
        "height",
        "entries",
        "open_cells",
        "fits_here",
        "slow",
        "has_solution",
        "shape",
        "numbers",
        "across",
        "down",
    }
    assert set("".join(body["shape"])) - {BLOCK, "."} == set()
    assert "OAF" not in json.dumps(body)


def test_puzzle_detail_404_lists_the_available_ids(client: TestClient) -> None:
    response = client.get("/api/puzzles/not-a-puzzle")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "not-a-puzzle" in detail
    for pid in PUZZLE_IDS:
        assert pid in detail


# --------------------------------------------------------------------------- #
# The session routes
# --------------------------------------------------------------------------- #


def test_the_documented_routes_are_all_registered() -> None:
    """Including the two legacy solve routes, which the contract freezes."""
    registered = {
        (route.path, method)
        for route in webapp.app.routes
        for method in getattr(route, "methods", set())
    }
    for path, method in [
        ("/api/health", "GET"),
        ("/api/puzzles", "GET"),
        ("/api/puzzles/{pid}", "GET"),
        ("/api/solve", "POST"),
        ("/api/solve/stream", "POST"),
        ("/api/sessions", "POST"),
        ("/api/sessions", "GET"),
        ("/api/sessions/{sid}", "GET"),
        ("/api/sessions/{sid}/events", "GET"),
        ("/api/sessions/{sid}/stream", "GET"),
        ("/api/sessions/{sid}/stop", "POST"),
        ("/api/sessions/{sid}", "DELETE"),
        ("/static/studio.css", "GET"),
        ("/static/studio.js", "GET"),
        ("/", "GET"),
    ]:
        assert (path, method) in registered


@pytest.mark.parametrize(
    "method, path",
    [
        ("GET", "/api/sessions/ghost"),
        ("GET", "/api/sessions/ghost/events"),
        ("GET", "/api/sessions/ghost/stream"),
        ("POST", "/api/sessions/ghost/stop"),
        ("DELETE", "/api/sessions/ghost"),
    ],
)
def test_an_unknown_session_is_404_on_every_route(
    client: TestClient, manager: FakeManager, method: str, path: str
) -> None:
    manager.add("s1")  # a populated registry, so 404 means "not this id"
    response = client.request(method, path)
    assert response.status_code == 404
    assert "ghost" in response.json()["detail"]


def test_create_session_returns_the_frozen_session_fields(
    client: TestClient, manager: FakeManager
) -> None:
    response = client.post("/api/sessions", json={"puzzle": "mini-01", "rounds": 1})
    assert response.status_code == 200
    body = response.json()
    assert set(SESSION_FIELDS) <= set(body)
    assert body["puzzle_id"] == "mini-01"
    assert body["state"] == "queued"
    assert body["terminal"] is (body["state"] in TERMINAL_STATES)

    puzzle, config = manager.started[0]
    assert puzzle.id == "mini-01"
    assert config.max_rounds == 1
    # The budget comes from _agent_config, shared with /api/solve, and is what
    # keeps a session inside the deployment's function ceiling.
    assert config.wall_clock_budget == float(webapp.SOLVE_BUDGET)


def test_create_session_shares_the_size_guard_with_solve(
    client: TestClient, manager: FakeManager
) -> None:
    response = client.post("/api/sessions", json={"shape": ["." * 15] * 15})
    assert response.status_code == 413
    assert str(webapp.MAX_OPEN_CELLS) in response.json()["detail"]
    assert manager.started == [], "an oversized puzzle must not reach the registry"


def test_create_session_404s_on_an_unknown_puzzle(
    client: TestClient, manager: FakeManager
) -> None:
    response = client.post("/api/sessions", json={"puzzle": "no-such-puzzle"})
    assert response.status_code == 404
    assert manager.started == []


def test_create_session_is_429_when_the_cap_is_reached(
    client: TestClient, manager: FakeManager
) -> None:
    manager.limit_message = "2 solve(s) are already running and this process allows 2"
    response = client.post("/api/sessions", json={"puzzle": "mini-01"})
    assert response.status_code == 429
    # The manager's own sentence, not a paraphrase: the UI shows it verbatim.
    assert response.json()["detail"] == manager.limit_message


def test_create_session_is_503_without_an_api_key(
    client: TestClient, manager: FakeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cfg, "api_key", lambda: None)
    response = client.post("/api/sessions", json={"puzzle": "mini-01"})
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]
    assert manager.started == []


def test_create_session_is_501_where_sessions_cannot_survive_a_response(
    client: TestClient, manager: FakeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(webapp, "DURABLE_SESSIONS", False)
    monkeypatch.setattr(cfg, "api_key", lambda: None)
    response = client.post("/api/sessions", json={"puzzle": "mini-01"})
    # 501 ahead of 503: on a frozen runtime a missing key is not the interesting
    # problem, and 503 would send the UI after a fix that would not help. The
    # message names the endpoint the UI is expected to fall back to.
    assert response.status_code == 501
    assert "/api/solve/stream" in response.json()["detail"]
    assert manager.started == []


@pytest.mark.parametrize(
    "method, path",
    [
        ("POST", "/api/sessions"),
        ("POST", "/api/sessions/s1/stop"),
        ("DELETE", "/api/sessions/s1"),
    ],
)
def test_mutating_routes_require_the_access_token(
    client: TestClient,
    manager: FakeManager,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    path: str,
) -> None:
    # Patched on the module, not in the environment: app.py snapshots the
    # variable into a constant at import time.
    monkeypatch.setattr(webapp, "ACCESS_TOKEN", "opensesame")
    manager.add("s1", state="done")
    response = client.request(method, path, json={"puzzle": "mini-01"})
    assert response.status_code == 401
    assert "X-Access-Token" in response.json()["detail"]
    assert manager.started == []
    assert manager.stopped == []
    assert manager.deleted == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"headers": {"X-Access-Token": "opensesame"}},
        {"params": {"token": "opensesame"}},
    ],
)
def test_create_session_accepts_the_token_in_a_header_or_the_query(
    client: TestClient,
    manager: FakeManager,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
) -> None:
    monkeypatch.setattr(webapp, "ACCESS_TOKEN", "opensesame")
    response = client.post("/api/sessions", json={"puzzle": "mini-01"}, **kwargs)
    assert response.status_code == 200
    assert len(manager.started) == 1


def test_a_wrong_token_is_still_401(
    client: TestClient, manager: FakeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(webapp, "ACCESS_TOKEN", "opensesame")
    response = client.post(
        "/api/sessions",
        json={"puzzle": "mini-01"},
        headers={"X-Access-Token": "opensesamf"},
    )
    assert response.status_code == 401
    assert manager.started == []


def test_the_token_is_checked_before_the_api_key(
    client: TestClient, manager: FakeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unauthenticated caller learns nothing about the deployment's config."""
    monkeypatch.setattr(webapp, "ACCESS_TOKEN", "opensesame")
    monkeypatch.setattr(cfg, "api_key", lambda: None)
    assert client.post("/api/sessions", json={"puzzle": "mini-01"}).status_code == 401


@pytest.mark.parametrize(
    "path",
    [
        "/api/sessions",
        "/api/sessions/s1",
        "/api/sessions/s1/events",
        "/api/sessions/s1/stream",
        "/api/puzzles/mini-01",
    ],
)
def test_reading_never_needs_the_access_token(
    client: TestClient,
    manager: FakeManager,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    """The token guards spending, not information -- and EventSource cannot send
    a header, so requiring it on the stream would push the secret into a URL."""
    monkeypatch.setattr(webapp, "ACCESS_TOKEN", "opensesame")
    manager.add("s1", state="done")
    manager.logs["s1"].close()  # so the stream terminates instead of tailing
    assert client.get(path).status_code == 200


def test_list_sessions_passes_the_registry_order_through(
    client: TestClient, manager: FakeManager
) -> None:
    """The manager already orders newest-first; app.py must not re-sort it."""
    manager.add("newest", state="running")
    manager.add("older", state="done")
    manager.add("oldest", state="error")
    body = client.get("/api/sessions").json()
    assert [s["id"] for s in body["sessions"]] == ["newest", "older", "oldest"]
    assert body["max_concurrent"] == webapp.MAX_CONCURRENT_SESSIONS
    assert body["durable"] is webapp.DURABLE_SESSIONS


def test_session_detail_carries_the_result_once_there_is_one(
    client: TestClient, manager: FakeManager
) -> None:
    manager.add("pending", state="running")
    manager.add("finished", state="done", result={"puzzle": {"id": "mini-01"}})

    pending = client.get("/api/sessions/pending").json()
    assert pending["result"] is None
    assert pending["session"]["id"] == "pending"

    finished = client.get("/api/sessions/finished").json()
    assert finished["result"] == {"puzzle": {"id": "mini-01"}}
    assert finished["session"]["terminal"] is True


def test_stop_says_whether_it_changed_anything(
    client: TestClient, manager: FakeManager
) -> None:
    manager.add("live", state="running")
    manager.add("over", state="done")

    first = client.post("/api/sessions/live/stop").json()
    assert first["stopped"] is True
    # Read after the stop, so the UI does not have to invent the interim state.
    assert first["session"]["state"] == "stopping"

    assert client.post("/api/sessions/live/stop").json()["stopped"] is False
    assert client.post("/api/sessions/over/stop").json()["stopped"] is False


def test_deleting_twice_is_a_404_the_second_time(
    client: TestClient, manager: FakeManager
) -> None:
    manager.add("gone", state="done")
    first = client.delete("/api/sessions/gone")
    assert first.status_code == 200
    assert first.json() == {"deleted": True, "id": "gone"}
    assert client.delete("/api/sessions/gone").status_code == 404


# --------------------------------------------------------------------------- #
# The events poll
# --------------------------------------------------------------------------- #


def test_the_events_cursor_advances_and_never_replays(
    client: TestClient, manager: FakeManager
) -> None:
    manager.add("s1", state="running")
    log = manager.logs["s1"]
    log.append("status", {"state": "queued", "message": "queued"})
    log.append("status", {"state": "running", "message": "solving"})
    log.append("step", {"kind": "ingest", "round": 0, "message": "read", "data": {}})

    seen: list[int] = []
    first = client.get("/api/sessions/s1/events?cursor=0").json()
    seen += [e["seq"] for e in first["events"]]
    assert seen == [1, 2, 3]
    assert first["cursor"] == 3

    drained = client.get(f"/api/sessions/s1/events?cursor={first['cursor']}").json()
    assert drained["events"] == []
    assert drained["cursor"] == 3, "an empty poll echoes the client's own cursor"

    log.append("llm_call", {"id": "call-1"})
    log.append("result", {"puzzle": {"id": "mini-01"}})
    third = client.get(f"/api/sessions/s1/events?cursor={drained['cursor']}").json()
    seen += [e["seq"] for e in third["events"]]
    assert [e["seq"] for e in third["events"]] == [4, 5]
    assert third["cursor"] == 5

    assert seen == sorted(set(seen)) == [1, 2, 3, 4, 5]


def test_the_events_poll_clamps_and_echoes_out_of_range_cursors(
    client: TestClient, manager: FakeManager
) -> None:
    manager.add("s1", state="running")
    manager.logs["s1"].append("status", {"state": "running", "message": "solving"})

    behind = client.get("/api/sessions/s1/events?cursor=-5").json()
    assert [e["seq"] for e in behind["events"]] == [1]
    assert behind["cursor"] == 1

    ahead = client.get("/api/sessions/s1/events?cursor=999").json()
    assert ahead["events"] == []
    assert ahead["cursor"] == 999


def test_the_events_poll_reports_the_log_state_alongside_the_events(
    client: TestClient, manager: FakeManager
) -> None:
    manager.add("s1", state="running")
    log = manager.logs["s1"]
    log.append("step", {"kind": "commit", "round": 1, "message": "done", "data": {}})

    body = client.get("/api/sessions/s1/events").json()
    assert set(body) == {"events", "cursor", "session", "closed", "dropped"}
    assert body["session"]["id"] == "s1"
    assert body["closed"] is False
    assert body["dropped"] == 0
    assert body["events"][0] == {
        "seq": 1,
        "at": body["events"][0]["at"],
        "type": "step",
        "payload": {"kind": "commit", "round": 1, "message": "done", "data": {}},
    }

    log.close()
    assert client.get("/api/sessions/s1/events").json()["closed"] is True


def test_the_events_poll_404s_when_a_session_has_no_log(
    client: TestClient, manager: FakeManager
) -> None:
    manager.add("s1", state="done", with_log=False)
    response = client.get("/api/sessions/s1/events")
    assert response.status_code == 404
    assert "trace log" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# The SSE stream
# --------------------------------------------------------------------------- #


def test_the_stream_replays_a_closed_log_frame_for_frame_and_ends(
    client: TestClient, manager: FakeManager
) -> None:
    """A closed log is the deterministic case, so the frame format is asserted
    byte for byte here rather than parsed loosely."""
    manager.add("s1", state="done")
    log = manager.logs["s1"]
    log.append("status", {"state": "running", "message": "solving"})
    log.append("step", {"kind": "ingest", "round": 0, "message": "read", "data": {}})
    log.close()

    started = time.monotonic()
    response = client.get("/api/sessions/s1/stream?cursor=0")
    elapsed = time.monotonic() - started

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert response.headers["x-accel-buffering"] == "no"

    expected = "".join(
        f"event: {event.type}\ndata: {json.dumps(event.as_dict())}\n\n"
        for event in log.since(0)
    )
    expected += 'event: closed\ndata: {"cursor": 2, "dropped": 0}\n\n'
    assert response.text == expected
    # Half the point of this test is that it returned at all: a closed log has
    # to end the generator, not leave it waiting for events that cannot come.
    assert elapsed < 5.0


def test_the_stream_resumes_from_a_cursor(
    client: TestClient, manager: FakeManager
) -> None:
    manager.add("s1", state="done")
    log = manager.logs["s1"]
    for index in range(3):
        log.append("step", {"kind": "propose", "round": index, "message": "", "data": {}})
    log.close()

    frames = _frames(client.get("/api/sessions/s1/stream?cursor=2").text)
    assert [name for name, _ in frames] == ["step", "closed"]
    assert frames[0][1]["seq"] == 3
    assert frames[-1][1] == {"cursor": 3, "dropped": 0}


def test_the_stream_tails_a_live_log_and_ends_when_it_closes(
    client: TestClient, manager: FakeManager
) -> None:
    manager.add("s1", state="running")
    log = manager.logs["s1"]
    log.append("status", {"state": "running", "message": "solving"})

    def finish() -> None:
        time.sleep(0.05)
        log.append("result", {"puzzle": {"id": "mini-01"}})
        log.close()

    worker = threading.Thread(target=finish, name="test-log-closer", daemon=True)
    worker.start()
    try:
        frames = _frames(client.get("/api/sessions/s1/stream").text)
    finally:
        log.close()  # so a timing surprise cannot leave the generator waiting
        worker.join(timeout=5.0)

    assert [name for name, _ in frames] == ["status", "result", "closed"]
    assert frames[1][1]["payload"] == {"puzzle": {"id": "mini-01"}}


def test_the_sse_frame_helper_formats_one_event() -> None:
    """Unit-level, because a malformed frame is invisible in a passing stream:
    EventSource silently ignores a block it cannot parse."""
    assert webapp._sse("step", {"a": 1}) == 'event: step\ndata: {"a": 1}\n\n'


# --------------------------------------------------------------------------- #
# Static assets and path safety
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name, media_type",
    [("studio.css", "text/css"), ("studio.js", "text/javascript")],
)
def test_the_studio_assets_serve_from_public(
    client: TestClient, name: str, media_type: str
) -> None:
    response = client.get(f"/static/{name}")
    asset = _ROOT / "public" / name
    if not asset.is_file():
        assert response.status_code == 404
        pytest.skip(f"public/{name} is not in this checkout")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(media_type)
    assert response.content == asset.read_bytes()


def test_no_static_route_takes_a_path_parameter() -> None:
    """The traversal defence is structural: there is no name to traverse with.

    ``_static`` trusts its argument -- it indexes ``_STATIC_FILES`` with it --
    so the property that has to hold is that no route can hand it one.
    """
    static_paths = {
        route.path for route in webapp.app.routes if route.path.startswith("/static")
    }
    assert static_paths == {"/static/studio.css", "/static/studio.js"}
    assert all("{" not in path for path in static_paths)
    assert set(webapp._STATIC_FILES) == {"studio.css", "studio.js"}


@pytest.mark.parametrize(
    "path",
    [
        "/static/../app.py",
        "/static/..%2f..%2fapp.py",
        "/static/%2e%2e/app.py",
        "/static/studio.css/../../app.py",
        "/static//app.py",
        "/static/index.html",
        "/static/",
        "/static/studio.css%00.png",
    ],
)
def test_nothing_but_the_two_assets_is_reachable_under_static(
    client: TestClient, path: str
) -> None:
    response = client.get(path)
    assert response.status_code == 404
    # Whatever the router made of the path, it did not read this repository.
    assert "FUNCTION_MAX_SECONDS" not in response.text


def test_a_missing_asset_is_a_404_not_a_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A deployment that shipped without ``public/`` should say so."""
    monkeypatch.setattr(webapp, "_ROOT", tmp_path)
    response = client.get("/static/studio.css")
    assert response.status_code == 404
    assert "studio.css" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# Lifecycle, and the guards these tests depend on
# --------------------------------------------------------------------------- #


def test_the_lifespan_takes_the_registry_down_with_the_app(
    manager: FakeManager,
) -> None:
    with TestClient(webapp.app) as scoped:
        assert scoped.get("/api/health").status_code == 200
    assert manager.shutdowns == 1


def test_the_no_network_guards_are_armed(client: TestClient, manager: FakeManager) -> None:
    """Proves the safety net rather than trusting it: if any of these three
    stops holding, a test could start charging the account."""
    from xword.solver.agent import CrosswordAgent

    with pytest.raises(AssertionError):
        CrosswordAgent(cfg.AgentConfig())

    with pytest.raises(AssertionError):
        httpx.Client().get("http://127.0.0.1:9/should-never-be-reached")

    assert isinstance(webapp.manager, FakeManager)


# --------------------------------------------------------------------------- #
# One pass through the real registry
# --------------------------------------------------------------------------- #


def test_a_session_runs_to_completion_through_the_http_surface(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole surface against the real :class:`SessionManager`.

    The stub agent goes in through the manager's ``agent_factory``, which is the
    only seam there is -- ``start`` takes a puzzle and a config, so there is
    nowhere else to inject a fake client. The assertions are about the endpoints
    rather than about solving: the trace carries the model call, the result
    arrives with the terminal status, and a cursor a client already holds does
    not replay.
    """
    real = SessionManager(
        serialise=webapp._serialise,
        max_concurrent=2,
        agent_factory=_stub_agent_factory,
    )
    monkeypatch.setattr(webapp, "manager", real)
    try:
        created = client.post("/api/sessions", json={"puzzle": "mini-01", "rounds": 1})
        assert created.status_code == 200
        sid = created.json()["id"]

        deadline = time.monotonic() + 15.0
        body: dict[str, Any] = {}
        session: dict[str, Any] = {}
        while time.monotonic() < deadline:
            body = client.get(f"/api/sessions/{sid}").json()
            session = body["session"]
            if session["terminal"]:
                break
            time.sleep(0.02)
        assert session.get("state") == "done", session

        result = body["result"]
        assert result is not None
        assert set(result) >= {
            "puzzle",
            "fill",
            "confidence",
            "entries",
            "stats",
            "trace",
        }
        assert result["score"]["cells_total"] == 19
        assert session["llm_calls"] == 1
        assert session["cells_total"] == 19

        events = client.get(f"/api/sessions/{sid}/events?cursor=0").json()
        types = [e["type"] for e in events["events"]]
        # The orderings that matter to a subscriber, rather than the exact
        # number of status events: the log opens with the session's own
        # creation, the agent's narration and its model calls are interleaved
        # in the order they happened, and the terminal status -- not the
        # result -- is the last thing in the log.
        assert events["events"][0]["payload"]["state"] == "queued"
        assert [t for t in types if t in {"step", "llm_call"}] == [
            "step",
            "llm_call",
            "step",
        ]
        assert types.count("result") == 1
        assert types[-2:] == ["result", "status"]
        assert events["events"][-1]["payload"]["state"] in TERMINAL_STATES
        assert events["closed"] is True

        call = next(e["payload"] for e in events["events"] if e["type"] == "llm_call")
        assert call["prompt"].startswith("1A | len 3")
        assert call["tools"] == ["submit_answers"]
        assert call["tool_name"] == "submit_answers"
        assert call["tool_input"]["answers"][0]["slot"] == "1A"

        cursor = events["cursor"]
        again = client.get(f"/api/sessions/{sid}/events?cursor={cursor}").json()
        assert again["events"] == []

        assert client.post(f"/api/sessions/{sid}/stop").json()["stopped"] is False
        assert client.delete(f"/api/sessions/{sid}").status_code == 200
        assert client.get(f"/api/sessions/{sid}").status_code == 404
    finally:
        real.shutdown()

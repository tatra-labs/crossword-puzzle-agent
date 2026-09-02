"""Tests for the joins between the solver, the session registry and the web client.

Every other test module exercises one layer. These exercise the places where two
layers agree on a name, which is exactly what no single-layer test can see and
what a parallel edit breaks first. Three failures in particular are silent:

* ``CrosswordAgent._build_llm`` passes ``on_call=`` and ``cancel=`` to
  ``LLMCandidateSource`` unconditionally. Rename either parameter and *every*
  real solve dies with a ``TypeError`` at construction -- the CLI and the eval
  harness included -- while every test that injects a candidate source keeps
  passing, because an injected source is never built by that method.
* ``SessionManager``'s default factory calls ``CrosswordAgent`` with three
  keyword arguments. Nothing else in the suite calls it with all three.
* ``public/studio.js`` asks for paths that ``app.py`` has to register, and reads
  keys that ``SessionInfo.as_dict`` and ``_serialise`` have to emit. A UI reading
  ``session.status`` against a payload that says ``state`` fails in the browser
  and nowhere else.

Nothing here calls the Anthropic API: the sources are built but never used, and
the one end-to-end solve goes through ``FakeClient``. ``cache_path`` is pointed
at ``tmp_path`` throughout so no test touches the repository's clue cache.
"""

from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from xword.candidates.llm import FakeClient, LLMCandidateSource
from xword.config import AgentConfig
from xword.core.types import Puzzle
from xword.solver.agent import CrosswordAgent, _tag_round
from xword.web.sessions import SessionInfo, SessionManager, _default_agent_factory
from xword.web.trace import LLMCallRecord

REPO_ROOT = Path(__file__).resolve().parent.parent
STUDIO_JS = REPO_ROOT / "public" / "studio.js"
INDEX_HTML = REPO_ROOT / "public" / "index.html"


@pytest.fixture
def config(tmp_path: Path) -> AgentConfig:
    """An agent config whose clue cache lives in the test's own directory."""
    return AgentConfig(
        use_lexicon=False,
        cache_path=str(tmp_path / "clue-cache.sqlite"),
    )


# --------------------------------------------------------------------------- #
# Solver -> LLM source
# --------------------------------------------------------------------------- #


def test_the_llm_source_accepts_the_hooks_the_agent_passes() -> None:
    """``on_call`` and ``cancel`` are keyword-only and optional.

    Optional matters as much as present: every construction site that predates
    the trace work omits both, so a required parameter would break them all.
    """
    params = inspect.signature(LLMCandidateSource.__init__).parameters
    for name in ("on_call", "cancel"):
        assert name in params, f"LLMCandidateSource lost the {name!r} parameter"
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[name].default is None


def test_the_agent_forwards_both_hooks_to_both_sources(config: AgentConfig) -> None:
    """The escalation source has to be observable and stoppable too.

    ``hard_llm`` is a second, separately constructed source. A stop that only
    reached the batch source would leave the hard pass spending money after the
    user asked it to stop.
    """
    records: list[LLMCallRecord] = []
    # One stored reference: ``records.append`` builds a new bound method on
    # every attribute access, so identity has to be taken against this name.
    observer = records.append
    stopped = False

    def cancel() -> bool:
        return stopped

    agent = CrosswordAgent(
        # Two different models, so hard_llm cannot be the same object as llm.
        AgentConfig(
            model="model-a",
            hard_clue_model="model-b",
            use_lexicon=False,
            cache_path=config.cache_path,
        ),
        cancel=cancel,
        on_llm_call=observer,
    )
    normal, hard = agent.llm, agent.hard_llm

    assert normal is not hard, "the two models should give two distinct sources"
    for label, source in (("batch", normal), ("hard", hard)):
        assert source.on_call is observer, f"{label} source lost the observer"
        assert source.cancel is cancel, f"{label} source lost the cancel predicate"
    # Both sources share one clue cache, so a warm entry found by the batch pass
    # is not paid for again by the escalation pass.
    assert normal.cache is hard.cache


def test_an_agent_with_no_hooks_leaves_them_unset(config: AgentConfig) -> None:
    """The default path must not invent a subscriber or a cancel predicate."""
    agent = CrosswordAgent(config)
    assert agent.llm.on_call is None
    assert agent.llm.cancel is None


def test_round_hint_is_stamped_on_a_real_source(config: AgentConfig) -> None:
    source = CrosswordAgent(config).llm
    assert source.round_hint == 0
    _tag_round(source, 4)
    assert source.round_hint == 4


def test_round_hint_survives_a_source_that_cannot_hold_it() -> None:
    """Losing a trace label must not fail a solve.

    Injected doubles are frequently slotted dataclasses, where the assignment
    raises ``AttributeError``.
    """

    @dataclass(slots=True)
    class Slotted:
        answered: int = 0

    double = Slotted()
    _tag_round(double, 7)  # must not raise
    assert not hasattr(double, "round_hint")


# --------------------------------------------------------------------------- #
# Session registry -> solver
# --------------------------------------------------------------------------- #


def test_the_default_factory_matches_the_agent_signature() -> None:
    """The registry's call into ``CrosswordAgent`` is checked without building one.

    ``bind`` raises ``TypeError`` on an unknown or missing keyword, which is the
    failure this guards: the factory and the constructor are edited in different
    files by different people.
    """
    factory_params = inspect.signature(_default_agent_factory).parameters
    assert set(factory_params) == {"config", "on_event", "cancel", "on_llm_call"}
    for name in ("on_event", "cancel", "on_llm_call"):
        assert factory_params[name].kind is inspect.Parameter.KEYWORD_ONLY

    inspect.signature(CrosswordAgent).bind(
        AgentConfig(),
        on_event=lambda event: None,
        cancel=lambda: False,
        on_llm_call=lambda record: None,
    )


def test_the_factory_really_builds_a_wired_agent(config: AgentConfig) -> None:
    """The signature check above cannot see a factory that drops an argument."""
    records: list[LLMCallRecord] = []
    observer = records.append

    def cancel() -> bool:
        return False

    agent = _default_agent_factory(
        config,
        on_event=lambda event: None,
        cancel=cancel,
        on_llm_call=observer,
    )
    assert agent.llm.cancel is cancel
    assert agent.llm.on_call is observer


def test_the_manager_accepts_the_arguments_app_py_gives_it() -> None:
    """``app.py`` constructs the registry with two keywords and nothing else."""
    inspect.signature(SessionManager).bind(
        serialise=lambda puzzle, result: {},
        max_concurrent=3,
    )
    params = inspect.signature(SessionManager.__init__).parameters
    assert params["agent_factory"].default is None, (
        "agent_factory must stay optional: app.py does not pass one"
    )


# --------------------------------------------------------------------------- #
# Layering
# --------------------------------------------------------------------------- #


def test_the_llm_source_does_not_drag_in_the_session_registry() -> None:
    """``candidates`` may import ``web.trace``, but not the rest of ``web``.

    ``llm.py`` imports ``LLMCallRecord`` at module level, which is only sound
    because ``web/trace.py`` is standard-library-only. If ``trace.py`` ever grew
    an import of ``sessions.py`` (or of ``anthropic``), the solver would start
    paying for the web layer and a circular import would become possible. Run in
    a subprocess because the rest of this suite has already imported everything.
    """
    code = (
        "import sys; import xword.candidates.llm; "
        "print(','.join(sorted(m for m in sys.modules if m.startswith('xword'))))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "SYSTEMROOT": ""},
    )
    assert proc.returncode == 0, proc.stderr
    loaded = set(proc.stdout.strip().split(","))
    assert "xword.web.trace" in loaded, "the record type should be imported eagerly"
    assert "xword.web.sessions" not in loaded
    assert "xword.solver.agent" not in loaded


# --------------------------------------------------------------------------- #
# Server -> browser
#
# The client builds its URLs by concatenation, so the paths cannot be extracted
# from the JavaScript reliably. They are transcribed here instead, and a second
# assertion checks that no ``/api`` literal in the file falls outside the
# transcription -- so a call site added to the client without updating this list
# fails here rather than in a browser.
# --------------------------------------------------------------------------- #

CLIENT_CALLS: tuple[tuple[str, str], ...] = (
    ("GET", "/api/health"),
    ("GET", "/api/puzzles"),
    ("GET", "/api/puzzles/{pid}"),
    ("GET", "/api/sessions"),
    ("POST", "/api/sessions"),
    ("GET", "/api/sessions/{sid}"),
    ("GET", "/api/sessions/{sid}/events"),
    ("GET", "/api/sessions/{sid}/stream"),
    ("POST", "/api/sessions/{sid}/stop"),
    ("DELETE", "/api/sessions/{sid}"),
    ("POST", "/api/solve/stream"),
    ("GET", "/static/studio.css"),
    ("GET", "/static/studio.js"),
)


@pytest.fixture(scope="module")
def registered_routes() -> set[tuple[str, str]]:
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import app as webapp
    finally:
        sys.path.remove(str(REPO_ROOT))
    return {
        (method, route.path)
        for route in webapp.app.routes
        for method in getattr(route, "methods", set()) - {"HEAD", "OPTIONS"}
    }


@pytest.mark.parametrize(("method", "path"), CLIENT_CALLS)
def test_every_client_call_has_a_route(
    method: str, path: str, registered_routes: set[tuple[str, str]]
) -> None:
    assert (method, path) in registered_routes, (
        f"public/studio.js calls {method} {path}, which app.py does not register"
    )


def test_no_client_url_escapes_the_transcribed_list() -> None:
    """Guards the list above against a new fetch nobody added here."""
    source = STUDIO_JS.read_text(encoding="utf-8") + INDEX_HTML.read_text(encoding="utf-8")
    literals = set(re.findall(r"""["'`](/(?:api|static)/[^"'`]*)["'`]""", source))
    known = [path.split("{")[0] for _, path in CLIENT_CALLS]
    # /api/docs is a link in the page, not a call, and is served by FastAPI.
    known.append("/api/docs")
    for literal in sorted(literals):
        stem = literal.split("?")[0]
        assert any(stem.startswith(prefix) or prefix.startswith(stem) for prefix in known), (
            f"{literal} is requested by the client but is not in CLIENT_CALLS"
        )


def test_the_session_snapshot_says_state_not_status() -> None:
    """The whole UI switches on ``session.state``.

    Renaming the field would leave every session pill blank and raise nothing on
    either side.
    """
    snapshot = SessionInfo(
        id="a", puzzle_id="mini-01", title="t", size="5x5", entries=10, open_cells=19
    ).as_dict()
    assert "state" in snapshot
    assert "status" not in snapshot
    source = STUDIO_JS.read_text(encoding="utf-8")
    assert not re.search(r"\b(?:info|session|sess)\.status\b", source)


def test_the_session_snapshot_carries_every_field_the_sidebar_reads() -> None:
    snapshot = SessionInfo(
        id="a", puzzle_id="mini-01", title="t", size="5x5", entries=10, open_cells=19
    ).as_dict()
    for field in (
        "id", "puzzle_id", "title", "size", "entries", "open_cells", "state",
        "round", "step", "message", "llm_calls", "input_tokens", "output_tokens",
        "cost_usd", "elapsed_s", "cursor", "error", "model", "max_rounds",
        "solved", "cells_correct", "cells_total",
    ):
        assert field in snapshot, f"the sidebar reads session.{field}"
    json.dumps(snapshot)  # the snapshot crosses the wire as JSON


def test_a_finished_session_result_carries_what_the_session_view_draws(
    mini_puzzle: Puzzle, tmp_path: Path
) -> None:
    """One real solve, through ``FakeClient``, checked against the UI's reads.

    This is the only place the whole chain runs: agent -> trace record ->
    ``_serialise`` -> the keys ``public/studio.js`` indexes. It costs nothing
    because the client answers from a dictionary.
    """
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import app as webapp
    finally:
        sys.path.remove(str(REPO_ROOT))

    solution = mini_puzzle.solution or {}
    book = {slot.clue: [(solution[slot.id], 0.9)] for slot in mini_puzzle.slots}
    records: list[LLMCallRecord] = []
    source = LLMCandidateSource(
        model="fake",
        client=FakeClient(book),
        cache=None,
        on_call=records.append,
    )
    agent = CrosswordAgent(
        AgentConfig(use_lexicon=False, cache_path=str(tmp_path / "c.sqlite")),
        llm=source,
        on_llm_call=records.append,
    )
    result = agent.solve(mini_puzzle)
    payload: dict[str, Any] = webapp._serialise(mini_puzzle, result)

    assert set(payload) >= {"puzzle", "fill", "confidence", "entries", "stats", "trace"}
    assert set(payload["puzzle"]) >= {"id", "width", "height", "title", "rows", "numbers"}
    assert set(payload["stats"]) >= {
        "rounds", "llm_calls", "input_tokens", "output_tokens", "wall_seconds", "cost_usd"
    }
    assert set(payload["entries"][0]) >= {"id", "clue", "answer", "confidence", "source"}
    # mini_puzzle has a reference solution, so the score block the session view
    # renders must be there.
    assert set(payload["score"]) >= {
        "solved", "cells_correct", "cells_total", "cell_accuracy",
        "words_correct", "words_total",
    }
    json.dumps(payload)

    # The trace panel renders these off each record, and the injected source
    # proves the observer was actually reached.
    assert records, "no LLMCallRecord was emitted for a solve that made calls"
    emitted = records[0].as_dict()
    assert set(emitted) >= {
        "kind", "model", "system", "prompt", "tools", "tool_choice", "tool_name",
        "tool_input", "stop_reason", "input_tokens", "output_tokens", "duration_s",
        "attempts", "error", "cached", "clue_ids", "round", "truncated",
    }
    assert emitted["prompt"], "the prompt the UI displays must not be empty"


# --------------------------------------------------------------------------- #
# Seams the four parallel fixes could not see
#
# Each fix below was made in one file by an agent that could not read the other
# three. What is asserted here is only the agreement between them: the
# credential names the client emits against the ones the server accepts, the two
# enforcement points of one spend cap, a trace field added in one file and
# serialised in another, and the element ids the page reads against the ones it
# defines. A single-file test cannot fail for any of these, which is exactly why
# they are the ones that break.
# --------------------------------------------------------------------------- #


def _web_app() -> Any:
    """``app.py`` as a module. It lives at the repository root, not in ``src``."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import app as webapp
    finally:
        sys.path.remove(str(REPO_ROOT))
    return webapp


def _guarded_routes(webapp: Any) -> list[Any]:
    """Every route whose handler calls ``_require_access``, found by reading it.

    Deliberately not a transcribed list: a route added without a token check is
    the failure this is watching for, and a list would have to be updated by the
    same edit that forgot the check.
    """
    guarded = []
    for route in webapp.app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None or getattr(route, "dependant", None) is None:
            continue
        try:
            source = inspect.getsource(endpoint)
        except OSError:  # pragma: no cover - only if app.py is not on disk
            continue
        if "_require_access(" in source:
            guarded.append(route)
    return guarded


def test_every_guarded_route_takes_the_token_both_ways() -> None:
    """A header *and* a query parameter, on all of them, or the page half-works.

    ``EventSource`` cannot set a request header, so ``?token=`` is the only way
    a browser can authenticate ``/api/sessions/{sid}/stream``; and ``fetch``
    should not have to put a secret in a URL, so the header has to be there too.
    A route that accepts only one of the two is reachable from only half the
    client, which no test of either side alone would notice.
    """
    webapp = _web_app()
    guarded = _guarded_routes(webapp)
    assert len(guarded) >= 9, f"only {len(guarded)} routes check the token"
    for route in guarded:
        headers = {field.alias.lower() for field in route.dependant.header_params}
        query = {field.alias for field in route.dependant.query_params}
        assert "x-access-token" in headers, f"{route.path} cannot take the token as a header"
        assert "token" in query, f"{route.path} cannot take the token as a query parameter"

    paths = {route.path for route in guarded}
    for read in (
        "/api/sessions",
        "/api/sessions/{sid}",
        "/api/sessions/{sid}/events",
        "/api/sessions/{sid}/stream",
    ):
        assert read in paths, f"{read} serves the token holder's prompts and is unguarded"


def test_the_client_emits_the_credential_names_the_server_accepts() -> None:
    """The other half of the same seam, read out of the JavaScript.

    The names are taken from the running app rather than typed here, so renaming
    the header on the server fails this test instead of failing in a browser --
    which is where it would otherwise surface, and where a 401 on an event
    stream is indistinguishable from a stream that simply ended.
    """
    webapp = _web_app()
    stream = next(
        route
        for route in _guarded_routes(webapp)
        if route.path == "/api/sessions/{sid}/stream"
    )
    header = next(field.alias for field in stream.dependant.header_params)
    query = next(field.alias for field in stream.dependant.query_params if field.alias == "token")

    source = STUDIO_JS.read_text(encoding="utf-8")
    # FastAPI lower-cases the alias and HTTP header names are case-insensitive,
    # so the comparison is too -- the page spells it X-Access-Token.
    assert f'"{header}"' in source.lower(), f"the page never sends the {header} header"
    assert f'"{query}="' in source, f"the page never appends ?{query}= for EventSource"
    # The EventSource URL is the one that cannot carry a header, so the query
    # form has to be attached to that URL specifically, not merely defined.
    assert re.search(r"/stream\?cursor=[^;]*tokenParam", source), (
        "the stream URL does not carry the token, so a guarded stream is unreachable"
    )


def test_one_number_caps_every_solve_the_deployment_pays_for() -> None:
    """Sessions and the legacy routes admit against the same limit and the same
    count, because two caps that disagree are worse than one that is too low.

    ``SessionManager`` owns the sessions and ``_SolveAdmission`` owns the
    in-request solves; the property that matters is that neither has a limit or
    a total of its own.
    """
    webapp = _web_app()
    assert webapp.manager._max_concurrent == webapp.MAX_CONCURRENT_SESSIONS, (
        "the registry and the legacy routes are capped by different numbers"
    )
    sources = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    sources += (REPO_ROOT / "src" / "xword" / "web" / "sessions.py").read_text(encoding="utf-8")
    assert sources.count("XWORD_MAX_CONCURRENT_SESSIONS") == 1, (
        "the cap is read from the environment in more than one place"
    )

    # The combined count, exercised: a full registry refuses a legacy solve even
    # though the admission gate itself is holding nothing.
    class _FullRegistry:
        active = webapp.MAX_CONCURRENT_SESSIONS

    admission = webapp._SolveAdmission()
    real, webapp.manager = webapp.manager, _FullRegistry()
    try:
        with pytest.raises(webapp.HTTPException) as caught:
            admission.acquire()
        assert caught.value.status_code == 429
        assert "not a way around it" in caught.value.detail
        assert admission.live == webapp.MAX_CONCURRENT_SESSIONS, "a refusal took a slot"
    finally:
        webapp.manager = real


def _finished_session(manager: SessionManager, sid: str, timeout: float = 20.0) -> SessionInfo:
    """Poll until the session is terminal. There is no condition to wait on from
    outside the manager, and a fake client finishes in milliseconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = manager.info(sid)
        assert info is not None
        if info.state in {"done", "error", "stopped"}:
            return info
        time.sleep(0.01)
    raise AssertionError(f"session {sid} never finished")


def test_the_retry_history_reaches_the_events_payload_as_json(
    mini_puzzle: Puzzle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``retry_errors`` is declared in one file, populated in a second and
    serialised into the wire payload by a third.

    The whole chain runs here: the source retries a real (fake) client, the
    record reaches ``SessionManager._on_llm_call``, and the dict the poll route
    returns is checked with ``allow_nan=False`` -- the flag that turns a value
    ``json.dumps`` emits happily and no browser can parse into a failure here
    rather than in a fetch. The backoff is flattened because this is a test
    about the record, not about how long a retry waits.
    """
    webapp = _web_app()
    monkeypatch.setattr(LLMCandidateSource, "_sleep_for", lambda self, attempt: 0.0)
    solution = mini_puzzle.solution or {}
    book = {slot.clue: [(solution[slot.id], 0.9)] for slot in mini_puzzle.slots}
    client = FakeClient(book, fail_times=2)

    def factory(config: AgentConfig, *, on_event: Any, cancel: Any, on_llm_call: Any) -> Any:
        source = LLMCandidateSource(
            model=config.model,
            k=config.candidates_per_clue,
            batch_size=config.batch_size,
            max_concurrency=1,
            cache=None,
            client=client,
            on_call=on_llm_call,
            cancel=cancel,
        )
        return CrosswordAgent(config, llm=source, on_event=on_event, cancel=cancel)

    manager = SessionManager(serialise=webapp._serialise, agent_factory=factory, max_concurrent=3)
    try:
        started = manager.start(
            mini_puzzle, AgentConfig(use_lexicon=False, max_rounds=1, batch_size=1)
        )
        info = _finished_session(manager, started.id)
        log = manager.log(started.id)
        assert log is not None
        assert info.state == "done", f"the solve did not finish: {info.error}"

        events = log.since(0)
        calls = [event.payload for event in events if event.type == "llm_call"]
        retried = [call for call in calls if call["attempts"] > 1]
        assert retried, "no call was retried, so the field under test was never populated"
        for call in retried:
            assert len(call["retry_errors"]) == call["attempts"] - 1, (
                "a retried call lost the reason one of its attempts failed"
            )
            assert all("ConnectionError" in line for line in call["retry_errors"])
        for call in calls:
            assert isinstance(call["retry_errors"], list), "the field crosses the wire as a list"

        # Exactly the dict app.py's /events handler returns, parsed the way a
        # browser has to parse it.
        payload = {
            "events": [event.as_dict() for event in events],
            "cursor": log.cursor,
            "closed": log.closed,
            "dropped": log.dropped,
            "session": info.as_dict(),
        }
        json.dumps(payload, allow_nan=False)
    finally:
        manager.shutdown()

    # And the panel that exists to show it does read the field.
    assert "retry_errors" in STUDIO_JS.read_text(encoding="utf-8"), (
        "the record carries the retry history and the trace card never shows it"
    )


def test_the_page_only_reads_element_ids_it_also_defines() -> None:
    """``$("x")`` against ``id="x"``, both directions.

    A rename on either side is silent: ``$`` returns null and the next property
    access throws inside an event handler, which the console records and the
    page does not. The reverse direction is checked too, because an id nothing
    reads is usually the other half of a rename.
    """
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = STUDIO_JS.read_text(encoding="utf-8")
    defined = set(re.findall(r'\bid="([^"]+)"', html))
    read = set(re.findall(r'\$\(\s*"([^"]+)"\s*\)', js))
    read |= set(re.findall(r'getElementById\(\s*"([^"]+)"\s*\)', js))
    assert read - defined == set(), "studio.js reads element ids index.html does not define"
    assert defined - read == set(), "index.html defines element ids nothing reads"


def test_the_live_grid_cleans_answers_the_way_the_solver_does() -> None:
    """The provisional-letter path mirrors ``_clean`` by hand, so the two
    normalisations have to be compared as text; there is no shared constant to
    import across the language boundary.

    Drift here does not raise: the grid simply paints letters that never entered
    the solve, in the panel whose whole claim is that it shows the solve.
    """
    from xword.candidates import llm as llm_mod

    js = STUDIO_JS.read_text(encoding="utf-8")
    pattern = re.search(r"const NON_ALPHA = /([^/]+)/g;", js)
    assert pattern, "public/studio.js no longer defines the NON_ALPHA counterpart"
    assert pattern.group(1) == llm_mod._NON_ALPHA_RE.pattern, (
        "the page strips a different character class than LLMCandidateSource._clean"
    )
    # The length rejection is the other half: clipping a wrong-length answer
    # paints a rejected one anyway, which is the shape this replaced.
    assert "answer.length !== slot.cells.length" in js, (
        "the grid no longer rejects the wrong-length answers the solver drops"
    )
    assert "Math.min(slot.cells.length" not in js, "the clip is back"

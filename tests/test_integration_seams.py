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

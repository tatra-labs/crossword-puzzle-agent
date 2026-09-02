"""Central configuration: paths, model choice, budgets, pricing.

One module owns "where things live" and "what knobs exist" so that the CLI, the
solver, and the evaluation harness cannot drift apart. Every knob is settable
three ways, in increasing precedence: built-in default, environment variable,
explicit argument.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

DATA_DIR = Path(os.environ.get("XWORD_DATA_DIR", PROJECT_ROOT / "data"))
BUNDLED_PUZZLE_DIR = DATA_DIR / "puzzles" / "bundled"
FETCHED_PUZZLE_DIR = DATA_DIR / "puzzles" / "nyt"
LEXICON_DIR = DATA_DIR / "lexicon"
DEFAULT_LEXICON_PATH = LEXICON_DIR / "lexicon.txt"
CACHE_DIR = Path(os.environ.get("XWORD_CACHE_DIR", PROJECT_ROOT / ".xword-cache"))
DEFAULT_CACHE_PATH = CACHE_DIR / "clue-cache.sqlite"
REPORT_DIR = Path(os.environ.get("XWORD_REPORT_DIR", PROJECT_ROOT / "reports"))


def ensure_dirs() -> None:
    """Create the writable directories. Safe to call repeatedly."""
    for path in (CACHE_DIR, REPORT_DIR, LEXICON_DIR, FETCHED_PUZZLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


def load_dotenv(path: Path | None = None) -> None:
    """Populate ``os.environ`` from a ``.env`` file without overwriting.

    Deliberately minimal and dependency-free: a ``KEY=value`` per line, with
    optional surrounding quotes, ``#`` comments, and a tolerated ``export``
    prefix. Existing environment variables always win, so a shell export
    overrides the file.
    """
    path = path or (PROJECT_ROOT / ".env")
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def api_key() -> str | None:
    """The Anthropic key, loading ``.env`` on first use."""
    load_dotenv()
    return os.environ.get("ANTHROPIC_API_KEY") or None


# --------------------------------------------------------------------------- #
# Models and pricing
# --------------------------------------------------------------------------- #

DEFAULT_MODEL = os.environ.get("XWORD_MODEL", "claude-sonnet-5")

#: Model used when the agent re-asks a clue it could not resolve. Defaults to
#: the same model as the main pass, so escalation changes the *prompt* (explicit
#: wordplay analysis) but not the price. Set ``XWORD_HARD_MODEL=claude-opus-5``
#: to also escalate to a stronger model: it helps on Friday/Saturday clues and
#: roughly doubles the cost of a solve, so it is opt-in rather than default.
HARD_CLUE_MODEL = os.environ.get("XWORD_HARD_MODEL", DEFAULT_MODEL)

#: USD per million tokens, (input, output), at Anthropic first-party list price.
#: Used only to report an estimated cost alongside accuracy -- the harness
#: treats cost as a first-class axis, so it needs *some* number, but it is a
#: published-rate estimate, not a bill. Rates differ on Bedrock/Vertex.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

#: Used for an unrecognised model id. Deliberately the most expensive current
#: tier: an unknown model should over-estimate cost rather than quietly
#: under-report it in an evaluation table.
FALLBACK_PRICING = (5.0, 25.0)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimated USD for a token count, by published list price."""
    rate_in, rate_out = PRICING.get(model, FALLBACK_PRICING)
    return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000


# --------------------------------------------------------------------------- #
# Solver settings
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class AgentConfig:
    """Every knob the agent loop exposes.

    Defaults are tuned for a standard 15x15 daily puzzle on a warm cache.
    """

    # -- generation
    model: str = DEFAULT_MODEL
    hard_clue_model: str = HARD_CLUE_MODEL
    candidates_per_clue: int = 10
    batch_size: int = 12
    max_concurrency: int = 8
    temperature: float = 1.0
    use_lexicon: bool = True
    lexicon_topk: int = 60
    lexicon_weight: float = 0.35
    llm_weight: float = 1.0

    # -- propagation
    bp_iterations: int = 60
    bp_damping: float = 0.5
    bp_tolerance: float = 1e-4

    # -- search
    beam_width: int = 24
    discrepancy_limit: int = 3
    search_seconds: float = 30.0

    # -- the agent loop
    max_rounds: int = 4
    repair_threshold: float = 0.55
    max_repair_slots: int = 24
    escalate_hard_clues: bool = True

    #: Early-stop threshold, applied to the *weakest* squares of the grid (the
    #: 5th percentile of per-cell confidence), not the mean. A mean is dominated
    #: by the many squares the agent is certain about and stays near 1.0 even
    #: when a few are wrong, which made it stop before repairing them.
    stop_when_confident: float = 0.995

    # -- budgets and plumbing
    max_llm_calls: int = 400
    wall_clock_budget: float = 900.0
    cache_path: Path | None = field(default=None)
    seed: int = 0

    def with_overrides(self, **kwargs: object) -> AgentConfig:
        """Return a copy with the non-``None`` overrides applied."""
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)  # type: ignore[arg-type]

    @property
    def resolved_cache_path(self) -> Path:
        return self.cache_path or DEFAULT_CACHE_PATH


#: Ablation presets. The harness turns these names into configs so that
#: ``docs/EVALUATION.md`` and the code cannot disagree about what an ablation
#: actually changed.
ABLATIONS: dict[str, dict[str, object]] = {
    "full": {},
    "no-bp": {"bp_iterations": 0},
    "no-repair": {"max_rounds": 1, "escalate_hard_clues": False},
    "no-lexicon": {"use_lexicon": False},
    "no-search": {"beam_width": 1, "discrepancy_limit": 0},
    "greedy-llm": {
        "bp_iterations": 0,
        "beam_width": 1,
        "discrepancy_limit": 0,
        "max_rounds": 1,
        "use_lexicon": False,
        "escalate_hard_clues": False,
    },
    "lexicon-only": {
        "candidates_per_clue": 0,
        "llm_weight": 0.0,
        "max_rounds": 1,
        "escalate_hard_clues": False,
    },
    "single-candidate": {"candidates_per_clue": 1},
}


def config_for_ablation(name: str, base: AgentConfig | None = None) -> AgentConfig:
    """Build the config for a named ablation."""
    if name not in ABLATIONS:
        known = ", ".join(sorted(ABLATIONS))
        raise KeyError(f"unknown ablation {name!r}; known: {known}")
    return (base or AgentConfig()).with_overrides(**ABLATIONS[name])


__all__ = [
    "ABLATIONS",
    "AgentConfig",
    "BUNDLED_PUZZLE_DIR",
    "CACHE_DIR",
    "DATA_DIR",
    "DEFAULT_CACHE_PATH",
    "DEFAULT_LEXICON_PATH",
    "DEFAULT_MODEL",
    "FETCHED_PUZZLE_DIR",
    "HARD_CLUE_MODEL",
    "LEXICON_DIR",
    "PRICING",
    "PROJECT_ROOT",
    "REPORT_DIR",
    "api_key",
    "config_for_ablation",
    "ensure_dirs",
    "estimate_cost",
    "load_dotenv",
]

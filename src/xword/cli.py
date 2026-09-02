"""Command line interface.

Everything the project can do is reachable from ``xword <verb>``; the README
teaches the tool through this surface, and the evaluation harness is driven by it
so that a reported number always has an exact command that reproduces it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from xword import config as cfg
from xword.core.types import AgentEvent

app = typer.Typer(
    name="xword",
    help="An AI agent that solves crossword puzzles.",
    no_args_is_help=True,
    add_completion=False,
)
eval_app = typer.Typer(name="eval", help="Measure solution quality.", no_args_is_help=True)
fetch_app = typer.Typer(name="fetch", help="Download evaluation corpora.", no_args_is_help=True)
lexicon_app = typer.Typer(name="lexicon", help="Build and inspect the word list.", no_args_is_help=True)
app.add_typer(eval_app)
app.add_typer(fetch_app)
app.add_typer(lexicon_app)


def _make_console() -> Console:
    """A console that can actually print a crossword on Windows.

    The grid is drawn with box-drawing and superscript characters. On Windows the
    default stdout encoding is still cp1252, and rich falls back to the legacy
    console renderer whenever output is piped -- either one turns a rendered grid
    into a ``UnicodeEncodeError``. Reconfiguring stdout to UTF-8 and opting out of
    the legacy renderer fixes both; ``legacy_windows=False`` is safe because every
    supported Windows 11 terminal handles ANSI.
    """
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                try:
                    reconfigure(encoding="utf-8", errors="replace")
                except (ValueError, OSError):  # pragma: no cover - exotic terminals
                    pass
        return Console(legacy_windows=False)
    return Console()


console = _make_console()


def _build_config(
    model: str | None,
    rounds: int | None,
    candidates: int | None,
    concurrency: int | None,
    seconds: float | None,
    no_lexicon: bool,
    seed: int | None,
) -> cfg.AgentConfig:
    base = cfg.AgentConfig()
    return base.with_overrides(
        model=model,
        max_rounds=rounds,
        candidates_per_clue=candidates,
        max_concurrency=concurrency,
        wall_clock_budget=seconds,
        use_lexicon=False if no_lexicon else None,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# solve
# --------------------------------------------------------------------------- #


@app.command()
def solve(
    path: Path = typer.Argument(..., help="Puzzle file (.puz, .json, .ipuz) or 'demo'."),
    model: str | None = typer.Option(None, "--model", "-m", help="Model for the clue pass."),
    rounds: int | None = typer.Option(None, "--rounds", "-r", help="Max agent rounds."),
    candidates: int | None = typer.Option(None, "--candidates", "-k", help="Candidates per clue."),
    concurrency: int | None = typer.Option(None, "--concurrency", help="Parallel API batches."),
    seconds: float | None = typer.Option(None, "--budget", help="Wall-clock budget in seconds."),
    no_lexicon: bool = typer.Option(False, "--no-lexicon", help="Disable the word list."),
    seed: int | None = typer.Option(None, "--seed", help="Random seed."),
    live: bool = typer.Option(True, "--live/--quiet", help="Stream the agent's reasoning."),
    show_clues: bool = typer.Option(False, "--clues", help="Print the full clue table."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write result JSON here."),
    html: Path | None = typer.Option(None, "--html", help="Write an HTML grid here."),
) -> None:
    """Solve one puzzle and show the filled grid."""
    from xword.io.render import render_clues, render_grid, render_summary, render_trace
    from xword.solver.agent import CrosswordAgent

    puzzle = _resolve_puzzle(path)
    conf = _build_config(model, rounds, candidates, concurrency, seconds, no_lexicon, seed)
    cfg.ensure_dirs()

    console.print(
        Panel.fit(
            f"[bold]{puzzle.meta.get('title') or puzzle.id}[/bold]\n"
            f"{puzzle.height}x{puzzle.width}  ·  {len(puzzle.slots)} entries  ·  "
            f"model {conf.model}",
            title="crossword agent",
            border_style="cyan",
        )
    )

    events: list[AgentEvent] = []

    if live:
        with Live(console=console, refresh_per_second=8, transient=False) as display:

            def on_event(event: AgentEvent) -> None:
                events.append(event)
                display.update(render_trace(events, limit=18))

            agent = CrosswordAgent(conf, on_event=on_event)
            result = agent.solve(puzzle)
    else:
        agent = CrosswordAgent(conf, on_event=events.append)
        result = agent.solve(puzzle)

    gold_letters = puzzle.solution_letters() if puzzle.has_solution else None
    console.print()
    console.print(render_grid(puzzle, result.fill.letters, confidence=result.cell_confidence,
                              gold=gold_letters, title=puzzle.id))
    if show_clues:
        console.print(render_clues(puzzle, result.slots, gold=puzzle.solution))
    console.print(render_summary(puzzle, result, gold=puzzle.solution))

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_result_json(puzzle, result), encoding="utf-8")
        console.print(f"[dim]wrote {out}[/dim]")
    if html is not None:
        from xword.io.render import grid_to_html

        html.parent.mkdir(parents=True, exist_ok=True)
        html.write_text(
            grid_to_html(puzzle, result.fill.letters, confidence=result.cell_confidence,
                         gold=gold_letters, title=puzzle.id),
            encoding="utf-8",
        )
        console.print(f"[dim]wrote {html}[/dim]")

    if puzzle.has_solution:
        from xword.eval.metrics import score_result

        score = score_result(puzzle, result)
        raise typer.Exit(code=0 if score.solved else 1)


@app.command()
def demo(
    puzzle_id: str = typer.Option("mini-01", "--puzzle", "-p", help="Bundled puzzle id."),
    model: str | None = typer.Option(None, "--model", "-m"),
) -> None:
    """Solve a bundled puzzle end to end -- the fastest way to see the agent work."""
    path = cfg.BUNDLED_PUZZLE_DIR / f"{puzzle_id}.json"
    if not path.exists():
        available = sorted(p.stem for p in cfg.BUNDLED_PUZZLE_DIR.glob("*.json"))
        console.print(f"[red]No bundled puzzle {puzzle_id!r}.[/red] Available: {', '.join(available)}")
        raise typer.Exit(code=2)
    solve(path=path, model=model, rounds=None, candidates=None, concurrency=None, seconds=None,
          no_lexicon=False, seed=None, live=True, show_clues=True, out=None, html=None)


@app.command()
def show(path: Path = typer.Argument(..., help="Puzzle to display.")) -> None:
    """Render a puzzle (and its solution, if the file has one) without solving."""
    from xword.io.render import render_clues, render_grid

    puzzle = _resolve_puzzle(path)
    letters = puzzle.solution_letters() if puzzle.has_solution else None
    console.print(render_grid(puzzle, letters, title=puzzle.id))
    console.print(render_clues(puzzle))


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #


@eval_app.command("run")
def eval_run(
    suite: str = typer.Option("bundled", "--suite", "-s", help="'bundled', 'nyt', 'nyt:<n>', or a path."),
    systems: str = typer.Option("full", "--systems", help="Comma-separated ablation names."),
    limit: int | None = typer.Option(None, "--limit", "-n", help="Cap the number of puzzles."),
    workers: int = typer.Option(1, "--workers", "-w", help="Puzzles solved concurrently."),
    model: str | None = typer.Option(None, "--model", "-m"),
    seed: int = typer.Option(0, "--seed"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Report directory."),
) -> None:
    """Run a suite and write a full evaluation report."""
    from xword.eval.harness import run_suite
    from xword.eval.report import write_report
    from xword.eval.suites import load_suite

    cfg.ensure_dirs()
    names = tuple(s.strip() for s in systems.split(",") if s.strip())
    unknown = [n for n in names if n not in cfg.ABLATIONS]
    if unknown:
        console.print(f"[red]Unknown system(s): {', '.join(unknown)}[/red]")
        console.print(f"Known: {', '.join(sorted(cfg.ABLATIONS))}")
        raise typer.Exit(code=2)

    loaded = load_suite(suite, limit=limit, seed=seed)
    if not loaded.puzzles:
        console.print(f"[red]Suite {suite!r} is empty.[/red] Try `xword fetch nyt` first.")
        raise typer.Exit(code=2)

    out_dir = out or cfg.REPORT_DIR
    console.print(f"Running [bold]{len(loaded.puzzles)}[/bold] puzzles x {len(names)} system(s)")
    run = run_suite(
        loaded,
        systems=names,
        base_config=cfg.AgentConfig().with_overrides(model=model, seed=seed),
        workers=workers,
        out_path=out_dir / "runs.jsonl",
    )
    written = write_report(run, out_dir, puzzles=loaded.puzzles)
    for kind, p in written.items():
        console.print(f"[green]wrote[/green] {kind}: {p}")


@eval_app.command("report")
def eval_report(
    run_path: Path = typer.Argument(..., help="An EvalRun JSON written by `eval run`."),
    out: Path | None = typer.Option(None, "--out", "-o"),
) -> None:
    """Regenerate a report from a saved run without re-solving anything."""
    from xword.eval.harness import EvalRun
    from xword.eval.report import write_report

    run = EvalRun.from_json(run_path.read_text(encoding="utf-8"))
    written = write_report(run, out or cfg.REPORT_DIR)
    for kind, p in written.items():
        console.print(f"[green]wrote[/green] {kind}: {p}")


@eval_app.command("suites")
def eval_suites() -> None:
    """List the evaluation suites available on this machine."""
    from xword.eval.suites import available_suites

    table = Table("suite", "puzzles", title="available suites")
    for name, count in sorted(available_suites().items()):
        table.add_row(name, str(count))
    console.print(table)


# --------------------------------------------------------------------------- #
# fetch / lexicon / doctor
# --------------------------------------------------------------------------- #


@fetch_app.command("nyt")
def fetch_nyt(
    per_year: int = typer.Option(14, "--per-year", help="Puzzles sampled per year."),
    years: str | None = typer.Option(None, "--years", help="e.g. 2010-2018 or 2016,2017."),
    seed: int = typer.Option(0, "--seed"),
    workers: int = typer.Option(8, "--workers", "-w"),
) -> None:
    """Download an evaluation corpus of New York Times puzzles.

    The archive is NYT copyright: it lands in a gitignored directory and is used
    for measurement only. It is never redistributed by this project.
    """
    from xword.data_sources import fetch, plan_fetch

    year_list = _parse_years(years)
    plan = plan_fetch(years=year_list, per_year=per_year, seed=seed)
    console.print(f"Fetching up to {len(plan.dates)} puzzles into {plan.dest}")
    report = fetch(plan, workers=workers)
    console.print(
        f"[green]{report.downloaded}[/green] downloaded, "
        f"{report.skipped_existing} already present, {report.failed} unavailable"
    )
    if report.by_dow:
        table = Table("day", "count", title="corpus balance")
        for day, n in report.by_dow.items():
            table.add_row(day, str(n))
        console.print(table)


@fetch_app.command("wordlist")
def fetch_wordlist_cmd() -> None:
    """Download the public-domain English word list used to seed the lexicon."""
    from xword.data_sources import fetch_wordlist

    path = fetch_wordlist()
    console.print(f"[green]wrote[/green] {path}")


@lexicon_app.command("build")
def lexicon_build(
    wordlist: Path | None = typer.Option(None, "--wordlist", help="Source word list."),
    from_puzzles: bool = typer.Option(True, "--from-puzzles/--no-from-puzzles",
                                      help="Mine answers from any fetched corpus."),
    out: Path | None = typer.Option(None, "--out", "-o"),
) -> None:
    """Build the scored word list the solver fills grids from."""
    from xword.lexicon.build import build_default_lexicon

    dirs = [cfg.BUNDLED_PUZZLE_DIR]
    if from_puzzles and cfg.FETCHED_PUZZLE_DIR.exists():
        dirs.append(cfg.FETCHED_PUZZLE_DIR)
    path = build_default_lexicon(wordlist_path=wordlist, puzzle_dirs=dirs, out_path=out)
    from xword.lexicon.store import Lexicon

    console.print(f"[green]wrote[/green] {path} ({len(Lexicon.load(path)):,} entries)")


@lexicon_app.command("match")
def lexicon_match(
    pattern: str = typer.Argument(..., help="Pattern with ? wildcards, e.g. ?RE?O"),
    limit: int = typer.Option(20, "--limit", "-n"),
) -> None:
    """Look up what fits a pattern -- useful for sanity-checking the word list."""
    from xword.lexicon.store import Lexicon

    lex = Lexicon.default()
    table = Table("word", "score", title=f"{pattern.upper()}  ({len(lex):,} entries)")
    for word, score in lex.match(pattern.upper(), limit=limit):
        table.add_row(word, f"{score:.3f}")
    console.print(table)


@app.command()
def doctor() -> None:
    """Check that everything this project needs is present and working."""
    from xword.lexicon.store import Lexicon

    table = Table("check", "status", "detail", title="xword doctor")

    key = cfg.api_key()
    # Never echo any part of the key. `doctor` is the first thing people run,
    # often while screen-sharing or recording, and even a partial key is
    # credential material that does not need to be on screen to answer the
    # only question being asked: is one configured?
    table.add_row(
        "ANTHROPIC_API_KEY",
        "[green]ok[/green]" if key else "[red]missing[/red]",
        f"configured ({len(key)} chars)" if key else "set it in .env (see .env.example)",
    )

    bundled = sorted(cfg.BUNDLED_PUZZLE_DIR.glob("*.json"))
    table.add_row(
        "bundled puzzles",
        "[green]ok[/green]" if bundled else "[red]none[/red]",
        f"{len(bundled)} in {cfg.BUNDLED_PUZZLE_DIR}",
    )

    try:
        lex = Lexicon.default()
        built = cfg.DEFAULT_LEXICON_PATH.exists()
        table.add_row(
            "lexicon",
            "[green]ok[/green]" if built else "[yellow]fallback[/yellow]",
            f"{len(lex):,} entries" + ("" if built else "  (run `xword lexicon build`)"),
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        table.add_row("lexicon", "[red]error[/red]", str(exc))

    try:
        from xword.data_sources import corpus_stats

        stats = corpus_stats()
        total = stats.get("total", 0)
        table.add_row(
            "NYT corpus",
            "[green]ok[/green]" if total else "[yellow]empty[/yellow]",
            f"{total} puzzles" + ("" if total else "  (run `xword fetch nyt`)"),
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        table.add_row("NYT corpus", "[yellow]n/a[/yellow]", str(exc))

    cache = cfg.DEFAULT_CACHE_PATH
    table.add_row(
        "clue cache",
        "[green]ok[/green]" if cache.exists() else "[dim]empty[/dim]",
        f"{cache} ({cache.stat().st_size // 1024} KiB)" if cache.exists() else str(cache),
    )

    table.add_row("python", "[green]ok[/green]", sys.version.split()[0])
    console.print(table)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _resolve_puzzle(path: Path):
    from xword.io.loaders import load_puzzle

    if str(path) == "demo":
        path = cfg.BUNDLED_PUZZLE_DIR / "mini-01.json"
    if not Path(path).exists():
        candidate = cfg.BUNDLED_PUZZLE_DIR / f"{path}.json"
        if candidate.exists():
            path = candidate
        else:
            console.print(f"[red]No such puzzle: {path}[/red]")
            raise typer.Exit(code=2)
    return load_puzzle(path)


def _parse_years(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    years: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            years.extend(range(int(lo), int(hi) + 1))
        elif part:
            years.append(int(part))
    return years or None


def _result_json(puzzle, result) -> str:
    from xword.core.grid import grid_rows

    payload = {
        "puzzle_id": puzzle.id,
        "grid": grid_rows(puzzle, result.fill.letters),
        "entries": {
            sid: {
                "clue": o.clue,
                "answer": o.answer,
                "confidence": round(o.confidence, 4),
                "source": o.source,
            }
            for sid, o in result.slots.items()
        },
        "stats": {
            "rounds": result.stats.rounds,
            "llm_calls": result.stats.llm_calls,
            "input_tokens": result.stats.input_tokens,
            "output_tokens": result.stats.output_tokens,
            "cache_hits": result.stats.cache_hits,
            "wall_seconds": round(result.stats.wall_seconds, 3),
            "cost_usd": round(result.stats.cost_usd, 6),
            "notes": result.stats.notes,
        },
        "trace": [
            {"kind": e.kind, "round": e.round, "message": e.message} for e in result.trace
        ],
    }
    return json.dumps(payload, indent=2)


def main() -> None:  # pragma: no cover - entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

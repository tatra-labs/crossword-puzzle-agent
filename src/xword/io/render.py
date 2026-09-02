"""Every way this project shows a grid to a human.

Terminal output goes through ``rich``; the HTML and SVG exporters exist so the
evaluation harness can drop a real grid into a report page. All of them accept
the same overlays -- a fill, a per-cell confidence map, and a gold map -- so a
caller can move one puzzle between media without reshaping any data.

Colour policy
-------------
Everything here uses mid-tone colours rather than the ANSI brights. A demo gets
recorded on whatever terminal is to hand, and the failure modes are asymmetric:
pure ``yellow`` is illegible on a white background, ``bright_*`` washes out on
white, and the dark ANSI colours disappear on black. Every colour used below
clears roughly 3:1 contrast against *both* #ffffff and #000000, which is why the
"mid" rung of the confidence ramp is amber/olive rather than yellow and the
"high" rung is a mid green rather than ``bright_green``.

Nothing in this module raises on a partially filled grid: a missing letter, a
missing confidence, and a missing gold entry are all just "not known yet".
"""

from __future__ import annotations

import html as _html
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderableType
from rich.measure import Measurement
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from xword.config import DEFAULT_MODEL, estimate_cost
from xword.core.grid import grid_rows
from xword.core.types import (
    AgentEvent,
    Cell,
    Puzzle,
    Slot,
    SlotOutcome,
    SolveResult,
)

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #

#: Grid furniture. Lines sit a shade lighter than the black squares so the
#: lattice reads as drawn-on rather than as part of the blocks.
STYLE_LINE = "grey42"
STYLE_BLOCK = "grey35"
STYLE_NUMBER = "grey50"
STYLE_LETTER = "bold"
STYLE_EMPTY = "grey30"

#: Upper bound of each confidence bucket. Four buckets, not a continuous ramp:
#: a 256-colour terminal cannot render a smooth gradient legibly at
#: one-character sizes, and four steps are what a viewer can actually tell
#: apart at a glance.
CONFIDENCE_STOPS: tuple[float, float, float] = (0.35, 0.60, 0.80)
CONFIDENCE_STYLES: tuple[str, str, str, str] = (
    "red3",  # #d70000 -- 5.4:1 on white, 3.9:1 on black
    "dark_orange3",  # #d75f00 -- the low-mid rung
    "yellow4",  # #808000 -- olive; plain yellow vanishes on white
    "green4",  # #008700 -- 4.7:1 on white, 4.5:1 on black
)
CONFIDENCE_LABELS: tuple[str, str, str, str] = ("low", "fair", "good", "high")

STYLE_CORRECT = "bold green4"
STYLE_WRONG = "bold red3"
STYLE_MISSING = "grey30"

#: Light-theme hexes for the HTML/SVG exporters, with dark-theme replacements.
#: Same four buckets as the terminal ramp, so a written report and a screen
#: recording of the same solve agree with each other.
_HEX_LIGHT = {
    "fg": "#16181d",
    "muted": "#6b7280",
    "line": "#8b939d",
    "cell": "#ffffff",
    "block": "#14171c",
    "panel": "#f5f6f8",
    "c0": "#c62828",
    "c1": "#b45309",
    "c2": "#7d7413",
    "c3": "#2e7d32",
    "ok": "#2e7d32",
    "bad": "#c62828",
    "miss": "#9aa3ad",
}
_HEX_DARK = {
    "fg": "#e8eaed",
    "muted": "#9aa3ad",
    "line": "#4b525b",
    "cell": "#1b1f25",
    "block": "#0a0c0f",
    "panel": "#15181c",
    "c0": "#ef5350",
    "c1": "#f59e0b",
    "c2": "#cbd44a",
    "c3": "#4ade80",
    "ok": "#4ade80",
    "bad": "#ef5350",
    "miss": "#6b7280",
}

_BLOCK_GLYPH = "█"
_BAR_FULL = "█"
_BAR_EMPTY = "░"
_DOT = "·"
_ARROW = "→"

#: Entry numbers are drawn with Unicode superscripts so that a number and a
#: letter can share one terminal row; ``to_text`` is the ASCII-only escape
#: hatch for terminals that cannot render them.
_SUPERSCRIPT = str.maketrans(
    "0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"
)

_KIND_GLYPH: dict[str, str] = {
    "ingest": "▸",
    "propose": "✦",
    "fuse": "◈",
    "propagate": "⇄",
    "commit": "■",
    "critique": "!",
    "repair": "↻",
    "verify": "✓",
    "done": "★",
}
_KIND_STYLE: dict[str, str] = {
    "ingest": "grey50",
    "propose": "cyan3",
    "fuse": "medium_purple2",
    "propagate": "steel_blue1",
    "commit": "green4",
    "critique": "dark_orange3",
    "repair": "yellow4",
    "verify": "green4",
    "done": "bold green4",
}


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #


def confidence_bucket(p: float) -> int:
    """Index into the four-step confidence ramp. NaN and out-of-range clamp."""
    if p != p:  # NaN
        return 0
    for i, stop in enumerate(CONFIDENCE_STOPS):
        if p < stop:
            return i
    return len(CONFIDENCE_STOPS)


def confidence_style(p: float) -> str:
    """Rich style name for a probability in ``[0, 1]``."""
    return CONFIDENCE_STYLES[confidence_bucket(p)]


def _clamp01(p: float) -> float:
    if p != p:
        return 0.0
    return 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)


def _letter_at(letters: Mapping[Cell, str] | None, cell: Cell) -> str | None:
    """One upper-case letter, or ``None``.

    Rebus squares and stray lower case reach here from third-party readers, so
    the value is coerced rather than trusted.
    """
    if not letters:
        return None
    raw = letters.get(cell)
    if raw is None:
        return None
    text = str(raw).strip()
    return text[:1].upper() if text else None


def _cell_width(max_number: int) -> int:
    """Interior width of one square, in characters.

    Three is the house style: with its one-character border a square occupies
    four columns by two rows, which is close to square in a terminal. Jumbo
    puzzles whose numbering reaches three digits get one extra column so the
    number never crowds the letter out.
    """
    return 3 if max_number < 100 else 4


def _compose_cell(
    width: int, number: int | None, letter: str | None
) -> tuple[list[str], int]:
    """Lay out one square's interior; returns its characters and the index the
    letter occupies.

    The number is set superscript from the left and the letter sits centred,
    unless the number would collide with it -- then the letter slides right
    rather than the number being dropped.
    """
    chars = [" "] * width
    letter_ix = width // 2
    sup = "" if not number else str(number).translate(_SUPERSCRIPT)
    sup = sup[: width - 1]  # always leave one column for the letter
    if len(sup) > letter_ix:
        letter_ix = len(sup)
    for i, ch in enumerate(sup):
        chars[i] = ch
    if letter:
        chars[letter_ix] = letter
    return chars, letter_ix


def _is_block(puzzle: Puzzle, row: int, col: int) -> bool:
    """Block test that treats everything outside the grid as open, so the outer
    border never merges into a black square."""
    if not (0 <= row < puzzle.height and 0 <= col < puzzle.width):
        return False
    return Cell(row, col) in puzzle.blocks


def _junction(col: int, width: int, has_above: bool, has_below: bool) -> str:
    if col == 0:
        if has_above and has_below:
            return "├"
        return "┌" if has_below else "└"
    if col == width:
        if has_above and has_below:
            return "┤"
        return "┐" if has_below else "┘"
    if has_above and has_below:
        return "┼"
    return "┬" if has_below else "┴"


def _bar(p: float, width: int = 6, *, percent: bool = True) -> Text:
    """A tiny confidence meter, coloured on the same ramp as the letters."""
    p = _clamp01(p)
    filled = int(round(p * width))
    out = Text(no_wrap=True)
    out.append(_BAR_FULL * filled, confidence_style(p))
    out.append(_BAR_EMPTY * (width - filled), STYLE_EMPTY)
    if percent:
        out.append(f" {int(round(p * 100)):3d}%", STYLE_NUMBER)
    return out


def _fmt_number(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}m {rest:04.1f}s"


def _letters_from_answers(puzzle: Puzzle, answers: Mapping[str, str]) -> dict[Cell, str]:
    """Project slot-keyed answers down to per-cell letters, ignoring entries the
    puzzle does not have."""
    out: dict[Cell, str] = {}
    by_id = puzzle.slot_by_id
    for slot_id, answer in answers.items():
        slot = by_id.get(slot_id)
        if slot is None or not answer:
            continue
        for cell, ch in zip(slot.cells, str(answer).upper(), strict=False):
            out[cell] = ch
    return out


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #


def render_grid(
    puzzle: Puzzle,
    letters: Mapping[Cell, str] | None = None,
    *,
    confidence: Mapping[Cell, float] | None = None,
    gold: Mapping[Cell, str] | None = None,
    numbers: bool = True,
    title: str | None = None,
) -> RenderableType:
    """Draw ``puzzle`` as a real crossword.

    ``gold`` switches on correctness mode and takes precedence over
    ``confidence``: once the answers are known, how sure the agent felt is the
    less interesting fact.
    """
    number_at: dict[Cell, int] = (
        {slot.start: slot.number for slot in puzzle.slots} if numbers else {}
    )
    width = _cell_width(max(number_at.values(), default=0))

    lines: list[Text] = [_rule(puzzle, -1, 0, width)]
    for row in range(puzzle.height):
        lines.append(
            _content_row(puzzle, row, width, number_at, letters, confidence, gold)
        )
        lines.append(_rule(puzzle, row, row + 1, width))

    separator = Text("\n", no_wrap=True, overflow="crop")
    body = separator.join(lines)

    parts: list[RenderableType] = [body]
    if gold is not None:
        parts.append(_gold_legend(puzzle, letters, gold))
    elif confidence is not None:
        parts.append(_confidence_legend())

    group = Group(*parts)
    if title is None:
        return group
    return Panel(
        group,
        title=Text(title, style="bold"),
        title_align="left",
        border_style=STYLE_LINE,
        box=box.ROUNDED,
        padding=(0, 1),
        expand=False,
    )


def _rule(puzzle: Puzzle, above: int, below: int, width: int) -> Text:
    """One horizontal lattice line between rows ``above`` and ``below``.

    Runs of black squares are joined *through* the border, so a region of them
    looks solid instead of being sliced up by grid lines.
    """
    out = Text(no_wrap=True, overflow="crop")
    has_above = above >= 0
    has_below = below < puzzle.height
    for col in range(puzzle.width + 1):
        corner_solid = (
            _is_block(puzzle, above, col - 1)
            and _is_block(puzzle, above, col)
            and _is_block(puzzle, below, col - 1)
            and _is_block(puzzle, below, col)
        )
        if corner_solid:
            out.append(_BLOCK_GLYPH, STYLE_BLOCK)
        else:
            out.append(_junction(col, puzzle.width, has_above, has_below), STYLE_LINE)
        if col < puzzle.width:
            if _is_block(puzzle, above, col) and _is_block(puzzle, below, col):
                out.append(_BLOCK_GLYPH * width, STYLE_BLOCK)
            else:
                out.append("─" * width, STYLE_LINE)
    return out


def _content_row(
    puzzle: Puzzle,
    row: int,
    width: int,
    number_at: Mapping[Cell, int],
    letters: Mapping[Cell, str] | None,
    confidence: Mapping[Cell, float] | None,
    gold: Mapping[Cell, str] | None,
) -> Text:
    out = Text(no_wrap=True, overflow="crop")
    for col in range(puzzle.width):
        if col > 0 and _is_block(puzzle, row, col - 1) and _is_block(puzzle, row, col):
            out.append(_BLOCK_GLYPH, STYLE_BLOCK)
        else:
            out.append("│", STYLE_LINE)

        cell = Cell(row, col)
        if cell in puzzle.blocks:
            out.append(_BLOCK_GLYPH * width, STYLE_BLOCK)
            continue

        letter = _letter_at(letters, cell)
        chars, letter_ix = _compose_cell(width, number_at.get(cell), letter)
        style = _letter_style(cell, letter, confidence, gold)
        for i, ch in enumerate(chars):
            out.append(ch, style if i == letter_ix else STYLE_NUMBER)
    out.append("│", STYLE_LINE)
    return out


def _letter_style(
    cell: Cell,
    letter: str | None,
    confidence: Mapping[Cell, float] | None,
    gold: Mapping[Cell, str] | None,
) -> str:
    if gold is not None:
        if letter is None:
            return STYLE_MISSING
        want = gold.get(cell)
        if want is None:
            return STYLE_LETTER
        return STYLE_CORRECT if letter == str(want)[:1].upper() else STYLE_WRONG
    if letter is None:
        return STYLE_MISSING
    if confidence is not None and cell in confidence:
        return f"bold {confidence_style(float(confidence[cell]))}"
    return STYLE_LETTER


def _confidence_legend() -> Text:
    out = Text(no_wrap=True)
    out.append("confidence  ", STYLE_NUMBER)
    for i, style in enumerate(CONFIDENCE_STYLES):
        low = 0.0 if i == 0 else CONFIDENCE_STOPS[i - 1]
        out.append(_BAR_FULL * 2, style)
        out.append(f" {CONFIDENCE_LABELS[i]} ≥{low:.2f}   ", STYLE_NUMBER)
    return out


def _gold_legend(
    puzzle: Puzzle, letters: Mapping[Cell, str] | None, gold: Mapping[Cell, str]
) -> Text:
    correct = 0
    blank = 0
    unscored = 0  # filled, but the gold map says nothing about this cell
    wrong: list[tuple[Cell, str, str]] = []
    for cell in puzzle.open_cells:
        want = gold.get(cell)
        got = _letter_at(letters, cell)
        if got is None:
            blank += 1
        elif want is None:
            unscored += 1
        elif got == str(want)[:1].upper():
            correct += 1
        else:
            wrong.append((cell, got, str(want)[:1].upper()))

    out = Text()
    out.append(f"{correct} correct", STYLE_CORRECT)
    out.append(f"  {_DOT}  ", STYLE_NUMBER)
    out.append(f"{len(wrong)} wrong", STYLE_WRONG if wrong else STYLE_NUMBER)
    out.append(f"  {_DOT}  ", STYLE_NUMBER)
    out.append(f"{blank} blank", STYLE_MISSING)
    if unscored:
        out.append(f"  {_DOT}  ", STYLE_NUMBER)
        out.append(f"{unscored} unscored", STYLE_MISSING)
    if wrong:
        shown = wrong[:8]
        out.append("\n")
        out.append(
            "  ".join(f"{cell} {got}{_ARROW}{want}" for cell, got, want in shown),
            STYLE_WRONG,
        )
        if len(wrong) > len(shown):
            out.append(f"  +{len(wrong) - len(shown)} more", STYLE_NUMBER)
    return out


# --------------------------------------------------------------------------- #
# Clues
# --------------------------------------------------------------------------- #


def render_clues(
    puzzle: Puzzle,
    outcomes: Mapping[str, SlotOutcome] | None = None,
    *,
    gold: Mapping[str, str] | None = None,
    max_rows: int | None = None,
) -> RenderableType:
    """Across and Down side by side, stacked if the console is too narrow.

    ``max_rows`` caps each column independently -- the two directions are read
    as separate lists, so truncating them as one would be surprising.
    """
    answer_width = _answer_width(puzzle.slots)
    tables = [
        _clue_table(puzzle, direction, heading, outcomes, gold, max_rows, answer_width)
        for direction, heading in (("across", "Across"), ("down", "Down"))
    ]
    # Width one side needs before the clue text starts breaking mid-word.
    fixed = 2 + 4  # the number column and its padding
    if gold is not None:
        fixed += 1 + 2
    if outcomes is not None:
        fixed += answer_width + 2 + _BAR_COLUMN + 2
    return _SideBySide(tables[0], tables[1], min_side=fixed + _MIN_CLUE)


#: Bar column: four blocks plus " 100%".
_BAR_COLUMN = 9
#: Below this much room for the clue itself, stacking beats two columns.
_MIN_CLUE = 18


def _answer_width(slots: Sequence[Slot]) -> int:
    """Fixed width for the answer column: wide enough for the longest entry,
    capped so a jumbo puzzle does not starve the clue column."""
    longest = max((s.length for s in slots), default=0)
    return max(6, min(longest, 15))


@dataclass(frozen=True)
class _SideBySide:
    """Two clue tables, laid out according to how much room there actually is.

    Rich cannot decide this from the outside, because only the console knows its
    own width at print time -- so this defers the choice to render time instead
    of guessing at 80 columns.
    """

    left: Table
    right: Table
    min_side: int

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> Iterator[RenderableType]:
        if options.max_width >= self.min_side * 2 + 2:
            layout = Table.grid(expand=True, padding=(0, 2))
            layout.add_column(ratio=1)
            layout.add_column(ratio=1)
            layout.add_row(self.left, self.right)
            yield layout
        else:
            yield self.left
            yield self.right

    def __rich_measure__(
        self, console: Console, options: ConsoleOptions
    ) -> Measurement:
        return Measurement(min(self.min_side, options.max_width), options.max_width)


def _clue_table(
    puzzle: Puzzle,
    direction: str,
    heading: str,
    outcomes: Mapping[str, SlotOutcome] | None,
    gold: Mapping[str, str] | None,
    max_rows: int | None,
    answer_width: int,
) -> Table:
    slots = [s for s in puzzle.slots if s.direction == direction]
    table = Table(
        box=box.SIMPLE_HEAD,
        expand=True,
        pad_edge=False,
        padding=(0, 1),
        title=heading,
        title_style="bold",
        title_justify="left",
        header_style=STYLE_NUMBER,
    )
    if gold is not None:
        table.add_column("", width=1, no_wrap=True)
    table.add_column("#", justify="right", style=STYLE_NUMBER, no_wrap=True)
    # Only the clue flexes. Everything else is a fixed width, because an answer
    # folded across two lines is much harder to read than a clue that is.
    table.add_column("Clue", ratio=1, min_width=12, overflow="fold")
    if outcomes is not None:
        table.add_column("Answer", width=answer_width, overflow="fold")
        table.add_column("", width=_BAR_COLUMN, no_wrap=True)

    shown = slots if max_rows is None else slots[:max_rows]
    for slot in shown:
        outcome = outcomes.get(slot.id) if outcomes is not None else None
        answer = outcome.answer if outcome is not None else None
        row: list[RenderableType] = []
        if gold is not None:
            row.append(_mark(answer, gold.get(slot.id)))
        row.append(Text(str(slot.number)))
        row.append(
            Text(slot.clue, style="") if slot.clue else Text("—", style=STYLE_MISSING)
        )
        if outcomes is not None:
            row.append(
                Text(answer, style=_answer_style(answer, gold, slot.id))
                if answer
                else Text("—", style=STYLE_MISSING)
            )
            row.append(
                _bar(outcome.confidence if outcome is not None else 0.0, width=4)
            )
        table.add_row(*row)

    hidden = len(slots) - len(shown)
    if hidden > 0:
        row = [Text("")] if gold is not None else []
        row.append(Text(""))
        row.append(Text(f"… {hidden} more", style=STYLE_MISSING))
        while len(row) < len(table.columns):
            row.append(Text(""))
        table.add_row(*row)
    return table


def _answer_style(answer: str | None, gold: Mapping[str, str] | None, slot_id: str) -> str:
    if not answer:
        return STYLE_MISSING
    if gold is None:
        return STYLE_LETTER
    want = gold.get(slot_id)
    if want is None:
        return STYLE_LETTER
    return STYLE_CORRECT if answer.upper() == str(want).upper() else STYLE_WRONG


def _mark(answer: str | None, want: str | None) -> Text:
    if want is None or not answer:
        return Text(_DOT, style=STYLE_MISSING)
    if answer.upper() == str(want).upper():
        return Text("✓", style=STYLE_CORRECT)
    return Text("✗", style=STYLE_WRONG)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #


def render_summary(
    puzzle: Puzzle,
    result: SolveResult,
    gold: Mapping[str, str] | None = None,
) -> RenderableType:
    """The one-panel verdict: how much is right, and what it cost to get there.

    ``gold`` is slot-keyed like :attr:`Puzzle.solution`, and falls back to the
    puzzle's own solution when omitted so evaluation puzzles need no extra
    argument.
    """
    answers = gold if gold is not None else puzzle.solution
    letters = result.fill.letters
    open_cells = puzzle.open_cells

    filled = sum(1 for c in open_cells if _letter_at(letters, c) is not None)
    cells_correct: int | None = None
    words_correct: int | None = None
    words_total = len(puzzle.slots)

    if answers:
        gold_letters = _letters_from_answers(puzzle, answers)
        cells_correct = sum(
            1
            for c in open_cells
            if c in gold_letters and _letter_at(letters, c) == gold_letters[c]
        )
        by_id = puzzle.slot_by_id
        words_correct = sum(
            1
            for slot_id, want in answers.items()
            if slot_id in by_id
            and (result.fill.answer_for(by_id[slot_id]) or "") == str(want).upper()
        )

    solved = (
        bool(answers)
        and cells_correct == len(open_cells)
        and filled == len(open_cells)
    )

    stats = result.stats
    cost = stats.cost_usd
    if not cost and (stats.input_tokens or stats.output_tokens):
        # The solver normally fills this in; estimating here keeps the panel
        # from showing a blank where a number belongs.
        cost = estimate_cost(DEFAULT_MODEL, stats.input_tokens, stats.output_tokens)

    if not answers:
        solved_cell = Text("unknown", style=STYLE_MISSING)
    elif solved:
        solved_cell = Text("yes", style=STYLE_CORRECT)
    else:
        solved_cell = Text("no", style=STYLE_WRONG)

    left: list[tuple[str, RenderableType]] = [
        ("puzzle", Text(puzzle.id, style="bold")),
        (
            "size",
            Text(f"{puzzle.width}×{puzzle.height}  ({len(open_cells)} cells)"),
        ),
        ("source", Text(_source_of(puzzle), style=STYLE_NUMBER)),
        (
            "cells",
            _ratio(cells_correct, len(open_cells), f"{filled}/{len(open_cells)} filled"),
        ),
        ("words", _ratio(words_correct, words_total, f"{words_total} entries")),
        ("solved", solved_cell),
    ]
    right: list[tuple[str, RenderableType]] = [
        ("rounds", Text(str(stats.rounds))),
        ("llm calls", Text(f"{stats.llm_calls:,}")),
        ("tokens", Text(f"{stats.input_tokens:,} in / {stats.output_tokens:,} out")),
        ("cache hits", Text(f"{stats.cache_hits:,}")),
        ("est. cost", Text(f"${cost:,.4f}", style="bold")),
        ("wall time", Text(_fmt_seconds(stats.wall_seconds))),
    ]

    grid = Table.grid(padding=(0, 2))
    for _ in range(2):
        grid.add_column(justify="right", style=STYLE_NUMBER, no_wrap=True)
        grid.add_column(no_wrap=True)
    for (l_label, l_value), (r_label, r_value) in zip(left, right, strict=False):
        grid.add_row(l_label, l_value, r_label, r_value)

    border = (STYLE_CORRECT if solved else "red3") if answers else STYLE_LINE
    return Panel(
        grid,
        title=Text("solve summary", style="bold"),
        title_align="left",
        border_style=border,
        box=box.ROUNDED,
        padding=(0, 1),
        expand=False,
    )


def _source_of(puzzle: Puzzle) -> str:
    meta = puzzle.meta or {}
    for key in ("source", "publisher", "author", "date"):
        value = meta.get(key)
        if value:
            return str(value)
    return "—"


def _ratio(correct: int | None, total: int, fallback: str) -> Text:
    """``n/total (pct)`` coloured on the confidence ramp, or a plain fallback
    when there is nothing to score against."""
    if correct is None or total == 0:
        return Text(fallback, style=STYLE_MISSING)
    fraction = correct / total
    out = Text(no_wrap=True)
    out.append(f"{correct}/{total}", style=f"bold {confidence_style(fraction)}")
    out.append(f"  {fraction * 100:5.1f}%", style=STYLE_NUMBER)
    return out


# --------------------------------------------------------------------------- #
# Trace
# --------------------------------------------------------------------------- #


def render_trace(events: Sequence[AgentEvent], *, limit: int = 40) -> RenderableType:
    """The agent loop as a timeline, oldest first and grouped by round.

    When there are more events than ``limit`` the *tail* survives: the end of a
    solve -- repair, verify, done -- is what a reader is looking for.
    """
    shown = list(events[-limit:]) if limit and len(events) > limit else list(events)
    hidden = len(events) - len(shown)

    lines: list[RenderableType] = []
    if hidden > 0:
        lines.append(Text(f"… {hidden} earlier events hidden", style=STYLE_MISSING))
    if not shown:
        lines.append(Text("(no events recorded)", style=STYLE_MISSING))

    current: int | None = None
    for event in shown:
        if event.round != current:
            current = event.round
            header = Text(no_wrap=True, overflow="crop")
            header.append("── ", STYLE_LINE)
            header.append(f"round {event.round}", "bold")
            header.append(" " + "─" * 24, STYLE_LINE)
            lines.append(header)

        style = _KIND_STYLE.get(event.kind, STYLE_LETTER)
        glyph = _KIND_GLYPH.get(event.kind, _DOT)
        line = Text()
        line.append(f"  {glyph} ", style)
        line.append(f"{event.kind:<11}", style)
        line.append(event.message or "")
        if event.data:
            pairs = list(event.data.items())[:4]
            line.append(
                "   " + " ".join(f"{k}={_fmt_number(v)}" for k, v in pairs),
                STYLE_NUMBER,
            )
        lines.append(line)

    return Group(*lines)


# --------------------------------------------------------------------------- #
# Export: HTML
# --------------------------------------------------------------------------- #


def _css_vars(mapping: Mapping[str, str]) -> str:
    return "".join(f"--xw-{k}:{v};" for k, v in mapping.items())


_HTML_STYLE = (
    "<style>"
    ".xw-wrap{"
    + _css_vars(_HEX_LIGHT)
    + "display:inline-block;color:var(--xw-fg);"
    "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace;"
    "background:var(--xw-panel);padding:.75rem;border-radius:.5rem;line-height:1}"
    "@media (prefers-color-scheme:dark){.xw-wrap{" + _css_vars(_HEX_DARK) + "}}"
    ".xw-title{font-weight:700;font-size:.9rem;margin:0 0 .5rem;letter-spacing:.02em}"
    ".xw-grid{display:grid;gap:1px;background:var(--xw-line);"
    "border:1px solid var(--xw-line);width:max-content}"
    ".xw-cell{position:relative;width:2.1rem;height:2.1rem;background:var(--xw-cell);"
    "display:flex;align-items:center;justify-content:center;"
    "font-size:1.05rem;font-weight:600}"
    ".xw-block{background:var(--xw-block)}"
    ".xw-num{position:absolute;top:1px;left:3px;font-size:.55rem;font-weight:500;"
    "color:var(--xw-muted);line-height:1}"
    ".xw-c0{color:var(--xw-c0)}.xw-c1{color:var(--xw-c1)}"
    ".xw-c2{color:var(--xw-c2)}.xw-c3{color:var(--xw-c3)}"
    ".xw-ok{color:var(--xw-ok)}.xw-bad{color:var(--xw-bad)}"
    ".xw-miss{color:var(--xw-miss)}"
    ".xw-legend{margin-top:.5rem;font-size:.7rem;color:var(--xw-muted)}"
    ".xw-sw{display:inline-block;width:.7rem;height:.7rem;border-radius:2px;"
    "vertical-align:-1px;margin:0 .2rem 0 .6rem}"
    "</style>"
)


def _cell_class(
    cell: Cell,
    letter: str | None,
    confidence: Mapping[Cell, float] | None,
    gold: Mapping[Cell, str] | None,
) -> str:
    """Colour class for one letter. Shared by the HTML and SVG exporters so the
    two cannot drift apart."""
    if gold is not None:
        if letter is None:
            return "xw-miss"
        want = gold.get(cell)
        if want is None:
            return ""
        return "xw-ok" if letter == str(want)[:1].upper() else "xw-bad"
    if letter is None:
        return "xw-miss"
    if confidence is not None and cell in confidence:
        return f"xw-c{confidence_bucket(float(confidence[cell]))}"
    return ""


def grid_to_html(
    puzzle: Puzzle,
    letters: Mapping[Cell, str] | None = None,
    *,
    confidence: Mapping[Cell, float] | None = None,
    gold: Mapping[Cell, str] | None = None,
    title: str = "",
) -> str:
    """A standalone HTML fragment: inline ``<style>`` plus one ``<div>``.

    No ``<html>``/``<head>``, so it can be pasted straight into a report page.
    Colours live in custom properties on the wrapper and are re-declared under
    ``prefers-color-scheme: dark``; repeating the ``<style>`` block once per
    embedded grid is idempotent, which is why it is inlined rather than being
    the caller's problem.
    """
    number_at = {slot.start: slot.number for slot in puzzle.slots}
    cells: list[str] = []
    for row in range(puzzle.height):
        for col in range(puzzle.width):
            cell = Cell(row, col)
            if cell in puzzle.blocks:
                cells.append('<div class="xw-cell xw-block"></div>')
                continue
            letter = _letter_at(letters, cell)
            klass = _cell_class(cell, letter, confidence, gold)
            number = number_at.get(cell)
            inner = f'<span class="xw-num">{number}</span>' if number else ""
            tip = ""
            if gold is not None and letter is not None:
                want = gold.get(cell)
                if want is not None and letter != str(want)[:1].upper():
                    tip = f' title="expected {_html.escape(str(want)[:1].upper())}"'
            body = _html.escape(letter) if letter else ""
            klass_attr = f' class="{klass}"' if klass else ""
            cells.append(
                f'<div class="xw-cell"{tip}>{inner}'
                f"<span{klass_attr}>{body}</span></div>"
            )

    head = f'<div class="xw-title">{_html.escape(title)}</div>' if title else ""
    legend = _html_legend(puzzle, letters, confidence, gold)
    grid = (
        '<div class="xw-grid" style="grid-template-columns:'
        f'repeat({puzzle.width},2.1rem)">' + "".join(cells) + "</div>"
    )
    return f'<div class="xw-wrap">{_HTML_STYLE}{head}{grid}{legend}</div>'


def _html_legend(
    puzzle: Puzzle,
    letters: Mapping[Cell, str] | None,
    confidence: Mapping[Cell, float] | None,
    gold: Mapping[Cell, str] | None,
) -> str:
    if gold is not None:
        correct = wrong = blank = unscored = 0
        for cell in puzzle.open_cells:
            got = _letter_at(letters, cell)
            want = gold.get(cell)
            if got is None:
                blank += 1
            elif want is None:
                unscored += 1
            elif got == str(want)[:1].upper():
                correct += 1
            else:
                wrong += 1
        extra = (
            f' &middot; <span class="xw-miss">{unscored} unscored</span>'
            if unscored
            else ""
        )
        return (
            '<div class="xw-legend">'
            f'<span class="xw-ok">{correct} correct</span> &middot; '
            f'<span class="xw-bad">{wrong} wrong</span> &middot; '
            f'<span class="xw-miss">{blank} blank</span>{extra}</div>'
        )
    if confidence is not None:
        swatches = "".join(
            f'<span class="xw-sw" style="background:var(--xw-c{i})"></span>'
            f"{CONFIDENCE_LABELS[i]}"
            for i in range(len(CONFIDENCE_STYLES))
        )
        return f'<div class="xw-legend">confidence{swatches}</div>'
    return ""


# --------------------------------------------------------------------------- #
# Export: SVG
# --------------------------------------------------------------------------- #


def grid_to_svg(
    puzzle: Puzzle,
    letters: Mapping[Cell, str] | None = None,
    *,
    confidence: Mapping[Cell, float] | None = None,
    gold: Mapping[Cell, str] | None = None,
    cell: int = 34,
) -> str:
    """A standalone ``<svg>`` element, one ``<rect class="xw-cell">`` per square.

    Colours are declared as custom properties inside the SVG's own style block,
    with a dark-scheme override, so one file drops into a light or a dark report
    unchanged.
    """
    size = max(8, int(cell))
    pad = 1
    total_w = puzzle.width * size + pad * 2
    total_h = puzzle.height * size + pad * 2
    number_at = {slot.start: slot.number for slot in puzzle.slots}

    style = (
        "<style>"
        "svg.xw-svg{" + _css_vars(_HEX_LIGHT) + "}"
        "@media (prefers-color-scheme:dark){svg.xw-svg{" + _css_vars(_HEX_DARK) + "}}"
        ".xw-cell{fill:var(--xw-cell);stroke:var(--xw-line);stroke-width:1}"
        ".xw-block{fill:var(--xw-block);stroke:var(--xw-block)}"
        f".xw-num{{fill:var(--xw-muted);font-size:{size * 0.28:.1f}px}}"
        ".xw-ltr{fill:var(--xw-fg);font-weight:600;text-anchor:middle;"
        f"font-size:{size * 0.56:.1f}px}}"
        ".xw-c0{fill:var(--xw-c0)}.xw-c1{fill:var(--xw-c1)}"
        ".xw-c2{fill:var(--xw-c2)}.xw-c3{fill:var(--xw-c3)}"
        ".xw-ok{fill:var(--xw-ok)}.xw-bad{fill:var(--xw-bad)}"
        ".xw-miss{fill:var(--xw-miss)}"
        "</style>"
    )

    rects: list[str] = []
    glyphs: list[str] = []
    for row in range(puzzle.height):
        for col in range(puzzle.width):
            here = Cell(row, col)
            x = pad + col * size
            y = pad + row * size
            blocked = here in puzzle.blocks
            klass = "xw-cell xw-block" if blocked else "xw-cell"
            rects.append(
                f'<rect class="{klass}" x="{x}" y="{y}" '
                f'width="{size}" height="{size}"/>'
            )
            if blocked:
                continue
            number = number_at.get(here)
            if number:
                glyphs.append(
                    f'<text class="xw-num" x="{x + 3}" '
                    f'y="{y + size * 0.30:.1f}">{number}</text>'
                )
            letter = _letter_at(letters, here)
            if letter:
                extra = _cell_class(here, letter, confidence, gold)
                cls = f"xw-ltr {extra}".strip()
                glyphs.append(
                    f'<text class="{cls}" x="{x + size / 2:.1f}" '
                    f'y="{y + size * 0.74:.1f}">{_html.escape(letter)}</text>'
                )

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" class="xw-svg" role="img" '
        f'aria-label="crossword grid {puzzle.width} by {puzzle.height}" '
        f'width="{total_w}" height="{total_h}" viewBox="0 0 {total_w} {total_h}" '
        'font-family="ui-monospace,Menlo,Consolas,monospace">'
        f"{style}{''.join(rects)}{''.join(glyphs)}</svg>"
    )


# --------------------------------------------------------------------------- #
# Export: plain text
# --------------------------------------------------------------------------- #


def to_text(puzzle: Puzzle, letters: Mapping[Cell, str] | None = None) -> str:
    """ASCII only, no colour: for logs, diffs, and terminals that hate Unicode."""
    # Normalise before handing the fill to ``grid_rows``: it concatenates values
    # verbatim, so an un-normalised rebus square would widen its row and shear
    # the whole picture.
    normalised = {
        cell: (_letter_at(letters, cell) or " ") for cell in puzzle.open_cells
    }
    rows = grid_rows(puzzle, normalised, blank=" ")
    rule = "+" + "+".join("---" for _ in range(puzzle.width)) + "+"
    out = [rule]
    for row in rows:
        line = "|"
        for ch in row:
            line += ("###" if ch == "#" else f" {ch.upper()} ") + "|"
        out.append(line)
        out.append(rule)
    return "\n".join(out)


__all__ = [
    "CONFIDENCE_LABELS",
    "CONFIDENCE_STOPS",
    "CONFIDENCE_STYLES",
    "confidence_bucket",
    "confidence_style",
    "grid_to_html",
    "grid_to_svg",
    "render_clues",
    "render_grid",
    "render_summary",
    "render_trace",
    "to_text",
]

"""Prompt text and the answer-tool schema for the LLM candidate source.

Kept in its own module because this is the part of the agent most worth
reviewing and A/B testing: the transport code in ``llm.py`` is mechanical, but
almost all of the accuracy lives in ``SYSTEM_PROMPT``. The model already knows
the vocabulary; what it needs is the conventions of the genre, the fact that
length is a hard constraint, and an explicit instruction to be calibrated
instead of confident.

Everything the model returns comes back through a tool call (``ANSWER_TOOL``)
rather than free text. Parsing prose answers is where this kind of pipeline
usually breaks -- a schema with a required shape moves the failure from "your
regex missed a case" to "the API rejected the call".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from xword.core.types import WILDCARD, ClueRequest

#: Name of the single tool the model is asked to call.
ANSWER_TOOL_NAME = "submit_answers"

#: Line format for one clue inside a prompt. Machine-parsable on purpose: the
#: offline ``FakeClient`` in ``llm.py`` reads the prompt back with this shape,
#: so tests exercise the real prompt builder rather than a stub.
CLUE_LINE = "{slot} | len {length} | pat {pattern} | {clue}"


_CORE_RULES = """\
You are an expert American-style crossword solver. You are given clues from a \
single puzzle and must produce ranked candidate answers for each one.

HOW TO ANSWER
- Answer only by calling the `submit_answers` tool. Do not put answers in prose.
- Produce one entry for every slot id you were given, using the id verbatim
  (17A means 17-Across, 3D means 3-Down).

HARD CONSTRAINTS (a candidate that breaks one of these is discarded unscored)
- Answers are UPPER-CASE LETTERS A-Z only. Strip spaces, punctuation, accents
  and hyphens: "IT'S A DEAL" -> ITSADEAL, "K.O." -> KO, "T-BONE" -> TBONE,
  "MOTHER-IN-LAW" -> MOTHERINLAW, an n-tilde becomes a plain N.
- The answer must be EXACTLY the stated length. Length is a hard constraint, not
  a hint. Count the letters before you commit. A brilliant answer of the wrong
  length is worth nothing here; a merely plausible answer of the right length is
  worth a lot.
- When a pattern is given, every candidate must match it position for position.
  '?' means that square is still unknown, a letter means that square is already
  believed known. Pattern O?E? admits OREO and OMEN; it rules out ERIE and OREOS.

CLUE CONVENTIONS
- An abbreviation marker in the clue means an abbreviated answer: "Abbr.",
  "Org.", "Assn.", "for short", "briefly", "in brief", or an abbreviation used
  inside the clue itself. "Fed. agent" -> TMAN, "Doctors' org." -> AMA.
- A question mark at the end means wordplay, a pun, or a deliberately misleading
  reading. Do not answer it literally: "Bank deposit?" -> SILT.
- A clue written in a foreign language, or naming a foreign place or person,
  wants an answer in that language: "Friend, in France" -> AMI; "Water, in
  Oaxaca" -> AGUA.
- Tense, number and part of speech must match between clue and answer. A plural
  clue takes a plural answer, a past-tense clue a past-tense answer, a gerund an
  -ING answer: "Ran fast" -> SPED (not SPEED), "Cats and dogs" -> PETS.
- "___" is a fill-in-the-blank. Answer only the blank, and make the surrounding
  words come out exactly right: "___ Lisa" -> MONA.
- A clue naming a person may want only part of the name; the length decides.
  "Actor Baldwin" -> ALEC; "Singer Fitzgerald" -> ELLA.
- "e.g.", "for one", "say", "perhaps" mean the answer is an example of the clue
  or the clue is an example of the answer.
- A clue in brackets, or an interjection like "[sigh]", wants a sound or an
  exclamation: PSST, AHEM, OOF.
- "Maker of ...", "... brand", "... competitor" want a proper noun.

CROSSWORDESE
Short entries are drawn from a small, heavily reused vocabulary, and vowel-rich
words are far more likely than obscure dictionary words. When a 3-5 letter clue
is vague, reach for the common fill first: OREO ERIE ETUI ALOE EPEE OLEO ARIA
ANTE ASEA ESNE IDEE OBOE ELAN AERIE ADO AMI ILE ISLE STET SNEE APSE ARETE OLE
ERR EWER ODE ERA ERE ETA EEL EMU IBEX ORCA ANOA OKAPI ASTA EDDA UTE UKE NAE
NEE ALEE AVER AERO ACME EDEN ARLO ONO ESAU ELIA. Only prefer a rare word when
the clue points hard at it.

CALIBRATION
- Return up to the requested number of DIVERSE candidates per clue, best first.
  Diverse means genuinely different answers, not five spellings of one guess and
  not a word plus its own plural. Two good guesses beat ten padded ones.
- Probabilities are honest beliefs and are allowed to sum to well under 1. The
  mass you leave unassigned is your admission that the answer may be none of
  these, and the solver uses exactly that to decide whether to trust you or to
  trust the crossing entries. A calibrated 0.3 is far more useful than a bluffed
  0.95, and an overconfident wrong answer corrupts every entry that crosses it.
- Rough scale: 0.9+ you can name the answer outright; 0.4-0.7 you have the right
  idea and a couple of spellings compete; 0.1-0.3 you are pattern-matching the
  genre. If you truly have no idea, still return your best few at low
  probability -- a weak candidate costs little, an empty list costs a lot.\
"""


SYSTEM_PROMPT: str = _CORE_RULES


HARD_SYSTEM_PROMPT: str = (
    _CORE_RULES
    + """

THIS IS A HARD CLUE
The first pass already failed on this entry, so do not simply repeat a fast
association. Before you call the tool, write out a short analysis in plain text:

1. Clue type: straight definition, fill-in-the-blank, abbreviation, foreign
   language, proper noun, pun or wordplay (question mark), theme entry, or
   something else.
2. Definition part: which words in the clue define the answer.
3. Wordplay part: which words, if any, are doing something other than defining
   -- a pun, a hidden word, a homophone, a category marker, a tense or number
   signal.
4. What the known letters in the pattern imply about the shape of the answer,
   and what the crossing clues suggest about this corner of the puzzle.

Then call `submit_answers` with the candidates that survive that analysis, and
put a one-line summary of the reasoning in the `analysis` field. Reconsider the
obvious answer if it does not fit the length or the pattern: on a hard clue, the
obvious answer is usually the one that already failed."""
)


#: The structured-output contract. A single tool call carries every slot in the
#: batch, so a batch costs one round trip and the response cannot half-parse.
ANSWER_TOOL: dict[str, Any] = {
    "name": ANSWER_TOOL_NAME,
    "description": (
        "Submit ranked candidate answers for every crossword entry you were "
        "asked about. Call this exactly once, covering all requested slot ids."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "description": "One object per requested slot id.",
                "items": {
                    "type": "object",
                    "properties": {
                        "slot": {
                            "type": "string",
                            "description": (
                                "The slot id exactly as given, e.g. '17A' or '3D'."
                            ),
                        },
                        "analysis": {
                            "type": "string",
                            "description": (
                                "Optional one-line note on how the clue works. "
                                "Requested for hard clues, otherwise omit."
                            ),
                        },
                        "candidates": {
                            "type": "array",
                            "description": (
                                "Candidate answers, most likely first. Upper-case "
                                "A-Z only, each exactly the stated length, each "
                                "matching the given pattern."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                        "description": (
                                            "Upper-case letters A-Z, no spaces or "
                                            "punctuation."
                                        ),
                                    },
                                    "probability": {
                                        "type": "number",
                                        "description": (
                                            "Honest probability this is the entry, "
                                            "in [0, 1]. Probabilities within one "
                                            "slot may sum to less than 1."
                                        ),
                                        "minimum": 0,
                                        "maximum": 1,
                                    },
                                },
                                "required": ["answer", "probability"],
                            },
                        },
                    },
                    "required": ["slot", "candidates"],
                },
            }
        },
        "required": ["answers"],
    },
}


# --------------------------------------------------------------------------- #
# Prompt builders
# --------------------------------------------------------------------------- #

#: Meta keys worth showing the model, in the order a solver would want them.
#: Anything else in ``puzzle_meta`` is appended alphabetically.
_META_ORDER: tuple[str, ...] = (
    "source",
    "publication",
    "date",
    "day",
    "weekday",
    "title",
    "author",
    "editor",
    "difficulty",
    "theme",
    "notes",
)


def _pattern_text(request: ClueRequest) -> str:
    """The pattern as the model sees it: all-unknown when nothing is known yet.

    Always rendering a pattern, rather than omitting the field on the first
    pass, keeps every clue line the same shape -- which is what makes the batch
    prompt cheap for the model to read and safe to parse back.
    """
    if request.pattern and len(request.pattern) == request.length:
        return request.pattern.upper()
    return WILDCARD * request.length


def _meta_lines(puzzle_meta: Mapping[str, str]) -> list[str]:
    """Puzzle context, ordered so the same puzzle always renders identically.

    Stable ordering matters twice over: it keeps the prompt prefix cacheable and
    it keeps runs reproducible.
    """
    lines: list[str] = []
    for key in _META_ORDER:
        value = puzzle_meta.get(key)
        if value:
            lines.append(f"{key}: {value}")
    for key in sorted(puzzle_meta):
        if key not in _META_ORDER and puzzle_meta[key]:
            lines.append(f"{key}: {puzzle_meta[key]}")
    return lines


def build_batch_prompt(
    requests: Sequence[ClueRequest], *, k: int, puzzle_meta: Mapping[str, str]
) -> str:
    """One user message covering many clues.

    Batching is the largest cost lever in this pipeline: the system prompt and
    the puzzle context dominate the input tokens and are paid once per call
    rather than once per clue.
    """
    parts: list[str] = []
    meta = _meta_lines(puzzle_meta)
    if meta:
        parts.append("PUZZLE\n" + "\n".join(meta))

    parts.append(
        f"ENTRIES ({len(requests)})\n"
        "Format: slot | len N | pat PATTERN | clue\n"
        "'A' is an across entry, 'D' is a down entry. '?' in a pattern is an "
        "unknown square; a letter is a square already believed known."
    )
    parts.append(
        "\n".join(
            CLUE_LINE.format(
                slot=r.slot_id,
                length=r.length,
                pattern=_pattern_text(r),
                clue=r.clue.strip() or "(no clue text)",
            )
            for r in requests
        )
    )
    parts.append(
        f"Return up to {k} candidates for each of the {len(requests)} entries "
        "above, in one `submit_answers` call. Every candidate must be A-Z only, "
        "exactly the stated length, and consistent with the stated pattern."
    )
    return "\n\n".join(parts)


def build_hard_prompt(
    request: ClueRequest, *, k: int, crossing_context: Sequence[str]
) -> str:
    """One clue, in isolation, with whatever the grid can tell the model.

    The crossing clues are included because a hard entry is often only solvable
    from its neighbourhood: a theme, a language, or a shared pun usually shows up
    in the clues that cross it.
    """
    lines = [
        "ENTRY",
        CLUE_LINE.format(
            slot=request.slot_id,
            length=request.length,
            pattern=_pattern_text(request),
            clue=request.clue.strip() or "(no clue text)",
        ),
        f"Direction: {request.direction}. Length: {request.length} letters.",
    ]

    meta = _meta_lines(request.puzzle_meta)
    if meta:
        lines.append("")
        lines.append("PUZZLE\n" + "\n".join(meta))

    context = [c.strip() for c in crossing_context if c and c.strip()]
    if context:
        lines.append("")
        lines.append(
            "CROSSING ENTRIES (their clues, for theme and register only -- do "
            "not answer them)\n" + "\n".join(f"- {c}" for c in context)
        )

    pattern = _pattern_text(request)
    known = [
        f"position {i + 1} = {ch}" for i, ch in enumerate(pattern) if ch != WILDCARD
    ]
    if known:
        lines.append("")
        lines.append("KNOWN LETTERS: " + ", ".join(known))

    lines.append("")
    lines.append(
        "Work through the analysis steps, then call `submit_answers` once with "
        f"up to {k} candidates for {request.slot_id}, ranked and honestly scored."
    )
    return "\n".join(lines)


__all__ = [
    "ANSWER_TOOL",
    "ANSWER_TOOL_NAME",
    "CLUE_LINE",
    "HARD_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "build_batch_prompt",
    "build_hard_prompt",
]

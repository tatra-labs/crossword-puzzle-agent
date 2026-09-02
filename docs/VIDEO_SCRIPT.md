# One-minute demo video — shot list and script

Target: **60 seconds, hard cap.** Narration below is ~150 words, which lands at ~58s at a
natural 155 wpm. Screen recording at 1920×1080, terminal at ~110×32 with a large font
(16–18pt) so it stays readable when the video is scaled down.

## Before recording

```bash
# 1. Warm the cache so the demo runs fast and identically every take.
xword solve data/puzzles/bundled/midi-01.json --quiet

# 2. Confirm the eval report exists and has real numbers in it.
xword eval run --suite bundled --systems full,greedy-llm --seed 0
open reports/report.html          # keep this tab ready for the 0:45 shot
```

Clear the terminal, set the window title to `crossword-puzzle-agent`, and hide anything
personal (shell prompt with a full path, other tabs, notifications).

---

## Shot list

| Time | Shot | On screen | Narration |
|---|---|---|---|
| 0:00–0:08 | A. Hook | `docs/ARCHITECTURE.md` diagram, or a still of a grid with two conflicting entries highlighted in red | "Ask a language model seventy-six crossword clues and you get seventy-six confident answers that don't fit together. A crossword isn't a quiz — it's a constraint problem." |
| 0:08–0:20 | B. Approach | The loop diagram from `ARCHITECTURE.md`, animated or highlighted stage by stage | "So the agent runs a loop. Claude proposes several ranked answers per clue, with probabilities. Belief propagation reconciles them across every crossing. Search picks the best set of words that actually agree." |
| 0:20–0:32 | C. Live solve, part 1 | `xword solve data/puzzles/bundled/midi-01.json --clues` — the live trace streaming: propose → fuse → propagate → commit | "Here it is solving a puzzle. Watch the trace: it reads the clues, propagates, and commits a grid." |
| 0:32–0:42 | D. The agentic bit | The `critique` line naming weak entries, then the second round | "Then it critiques itself — these entries are shaky — and re-asks only those clues, now handing the model the crossing letters it trusts. That feedback edge is the whole point: it's how a human solves." |
| 0:42–0:50 | E. Result | The filled grid with the confidence heat-map, then the summary panel (cells, words, solved, cost, time) | "Solved, with a confidence score on every square — the agent knows which letters it's unsure about." |
| 0:50–0:58 | F. Evaluation | `reports/report.html` scrolled to the headline table + the ablation delta row | "And it's measured: exact solve rate with bootstrap confidence intervals, ablations against a plain prompt-the-model baseline, and an honest note that public crossword archives are almost certainly in pretraining data." |
| 0:58–1:00 | G. Card | Repo URL on a plain card | *(silent)* |

---

## Narration, clean copy

> Ask a language model seventy-six crossword clues and you get seventy-six confident
> answers that don't fit together. A crossword isn't a quiz — it's a constraint problem.
>
> So the agent runs a loop. Claude proposes several ranked answers per clue, with
> probabilities. Belief propagation reconciles them across every crossing. Search picks the
> best set of words that actually agree.
>
> Here it is solving a puzzle. Watch the trace: it reads the clues, propagates, and commits
> a grid. Then it critiques itself — these entries are shaky — and re-asks only those clues,
> now handing the model the crossing letters it trusts. That feedback edge is the whole
> point: it's how a human solves.
>
> Solved, with a confidence score on every square.
>
> And it's measured: exact solve rate with bootstrap confidence intervals, ablations against
> a plain prompt-the-model baseline, and an honest note that public crossword archives are
> almost certainly in pretraining data.

---

## Notes for the edit

- **Cut, don't speed up, the API waits.** Shot C has real network latency in it. Trim the
  dead frames rather than time-lapsing — a sped-up terminal reads as a cheat.
- **The critique line is the money shot.** If one thing is legible at 480p, make it the
  line naming the weak entries. That is what distinguishes this from a prompt.
- **Show a wrong answer if there is one.** If the run misses an entry, leave it in and let
  the correctness view show it in red. A demo that only shows a perfect run is less
  convincing, not more.
- **Do not claim a headline number you haven't measured on the suite in the shot.** If the
  bundled suite is what's on screen, say "ten puzzles" or don't quote a rate at all.

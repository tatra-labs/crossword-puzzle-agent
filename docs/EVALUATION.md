# Evaluation methodology

How do you know a crossword agent is any good?

The naive answer — "count how many letters it got right" — is close to useless, and
understanding why is the whole design of this evaluation. This document sets out what
gets measured, on what data, with what statistics, and, most importantly, what the
resulting numbers do **not** license you to claim.

Everything here is implemented: `xword eval run` produces every table described below.
Where a measurement is proposed but not yet automated, it is marked
**[proposed]** rather than quietly omitted.

---

## 1. Why letter accuracy is a trap

A 15×15 daily crossword has about 180 white squares. A solver that gets 95% of letters
right sounds excellent. It is not: 95% means roughly nine wrong squares, which means the
puzzle is wrong, and in a tournament it scores zero.

Worse, letter accuracy is *inflated by the structure of English*. Fill in every square
with `E` and you will score around 12% for free. Fill in the highest-frequency letter per
position from a word list and you do better still, having read no clues at all. Any metric
whose floor is that high cannot distinguish good systems from mediocre ones.

So the metric hierarchy runs:

| Tier | Metric | What it answers |
|---|---|---|
| **Primary** | Exact puzzle solve rate | Did it actually solve the crossword? |
| **Primary** | Word (entry) accuracy | How many clues did it genuinely get? |
| Secondary | Cell accuracy | How close was it when it failed? |
| Secondary | Cell precision @ coverage | When it commits to a letter, is it right? |
| Secondary | Calibration (ECE, Brier) | Does it know what it doesn't know? |
| Operational | Cost, latency, API calls | Is it affordable and fast enough? |
| Diagnostic | Candidate coverage | Was the answer ever proposed at all? |

**Exact solve rate is the headline number**, because it is the one that matches what a
solver is for. Everything else exists to explain it.

### The diagnostic that matters most

`candidate_coverage` is the fraction of entries whose true answer appeared *anywhere* in
the candidate list before search ran. It cleanly separates the two ways this agent can
fail:

- **Generation failure** — the answer was never proposed. No amount of better search
  fixes this; the clue-reading model has to improve.
- **Search failure** — the answer was proposed and the solver still didn't pick it. This
  is a fixable inference bug.

Coverage is the ceiling on achievable word accuracy. Reporting accuracy without it tells
you a system is bad but not *where* it is bad. The harness computes it by inspecting the
agent's final beliefs after the solve; the agent itself never sees the reference solution.

---

## 2. Primary metrics, defined precisely

Let a puzzle have open cells $C$ and entries $E$.

- **Solved** — $\mathbb{1}[\forall c \in C: \hat{\ell}_c = \ell^*_c]$. Binary, per puzzle.
- **Solve rate** — mean of *Solved* over the suite. The headline.
- **Cell accuracy** — $|\{c : \hat{\ell}_c = \ell^*_c\}| / |C|$. **An unfilled cell counts
  as wrong.** This is deliberate: a system must not be able to raise its score by
  declining to answer.
- **Cell precision** — correct / *filled*. Read together with coverage, this is the
  abstention story.
- **Word accuracy** — $|\{e : \hat{a}_e = a^*_e\}| / |E|$. Entries are the unit a human
  cares about, and one wrong letter kills an entry, so this is strictly harsher than cell
  accuracy and closer to felt quality.

Rebus squares (multiple letters in one square) are normalised to their first letter at
parse time; puzzles containing them are **flagged in the report**, because scoring them
per-letter is a genuine simplification and hiding it would overstate the result.

---

## 3. Data

### 3.1 Bundled suite (ships with the repo)

Ten original puzzles authored for this project (4× 5×5, 3× 7×7, 2× 9×9, 1× 11×11),
CC0-licensed, with a deliberate spread of clue types — straight definition,
fill-in-the-blank, abbreviation, wordplay, plural/tense agreement, crosswordese. Every one
is verified by `validate_puzzle` in CI.

Its purpose is **smoke-testing and reproducibility**, not headline numbers: ten small
puzzles cannot support a confident claim about anything. It exists so that `git clone &&
xword demo` works with no downloads, and so the failure taxonomy has known examples of
each clue type.

### 3.2 NYT suite (fetched, never committed)

`xword fetch nyt` pulls from the public `doshea/nyt_crosswords` archive (1976–2018,
~15k puzzles). This content is **New York Times copyright**: it lands in a gitignored
directory, is used for measurement only, and is never redistributed by this project. The
fetcher writes a `.gitignore` containing `*` into that directory so it cannot be committed
by accident.

**Sampling is stratified by day of the week.** NYT difficulty rises monotonically Monday
(easiest) through Saturday, with Sunday a larger but Thursday-ish grid. An unstratified
sample is dominated by easy days and inflates the headline number, so `load_suite` draws
equal numbers per weekday with a seeded RNG. Day of week is also the **difficulty slice**
every table breaks down by — it is the closest thing the field has to a free, objective,
publisher-assigned difficulty label.

---

## 4. The contamination problem

**This is the most important section, and the one most crossword-LLM demos skip.**

Every puzzle in the NYT archive was published before 2019. Every one has been reproduced,
discussed, and tabulated across the public web for years. Any modern language model has
very likely seen a large fraction of these exact clue/answer pairs during pre-training.

Therefore: **the solve rate this harness reports on the NYT suite is an upper bound on
performance against fresh puzzles, and should never be quoted as if it were the latter.**

That is not a reason to abandon the corpus — it is the only large, difficulty-labelled,
publicly available crossword corpus, and it makes systems comparable. It is a reason to
measure the size of the effect. Four probes do that:

### Probe A — clue-only baseline (no grid)

Answer every clue independently, with no crossing information, and score word accuracy.
This is what pure recall gets you.

The **structure gap** — full-solve word accuracy minus clue-only word accuracy — is the
value the agent's inference machinery adds on top of whatever the model already knows.
A system whose gap is near zero is a lookup table with extra steps, however good its
headline number.

### Probe B — length-perturbation control

Ask for each clue with a *deliberately wrong* length (±1). A model that reasons from the
clue should either refuse or produce a different word of the requested length. A model
that is reciting a memorised pair will tend to emit the original answer anyway. The rate
at which the original answer reappears at the wrong length is a **memorisation signal**.

### Probe C — clue paraphrase **[proposed]**

Rewrite each clue with a separate model, preserving the answer but changing the surface
form, then re-solve. A large drop indicates dependence on the exact remembered string
rather than on the semantics.

*Caveat, stated up front:* paraphrasing is confounded. A rewritten clue may simply be a
**worse** clue — more ambiguous, or subtly pointing elsewhere — so some of the drop is
added difficulty, not lost memorisation. Interpreting Probe C requires a validation pass
(does a strong independent model still map paraphrase → original answer?), and even then
the number is an upper bound on the memorisation effect. It is reported with that caveat
attached or not at all.

### Probe D — genuinely fresh puzzles

The only clean answer. Two sources:

1. The bundled originals, authored for this project.
2. **[proposed]** A documented protocol for the user to drop in puzzles published *after*
   their model's training cutoff (`xword eval run --suite path/to/fresh/`). Since the
   cutoff moves with each model release, this stays a user-supplied step rather than a
   committed dataset.

Reporting rule: any headline number from the NYT suite is printed alongside its Probe A
structure gap, so a reader always sees recall and reasoning separated.

---

## 5. Ablations

Each ablation isolates one component. Names are defined once, in `config.ABLATIONS`, so
the code and this document cannot drift apart.

| System | Change | Question it answers |
|---|---|---|
| `full` | — | Baseline. |
| `greedy-llm` | Take the model's top answer per clue; no BP, no search, no repair | **What does the agentic scaffolding buy over prompting?** |
| `no-bp` | Skip belief propagation | Is probabilistic inference worth it, or is search enough? |
| `no-search` | Beam width 1, no discrepancies | How much does global optimisation matter? |
| `no-repair` | One round only, no re-asking | Is the feedback loop earning its cost? |
| `no-lexicon` | No word list | How much comes from knowing what *fits* vs. what clues *mean*? |
| `lexicon-only` | No model at all | The pure-CSP floor. |
| `single-candidate` | One candidate per clue | Does ranked uncertainty matter, or just the top guess? |

`greedy-llm` vs `full` is the central claim of the project. If that delta is small, the
agent is not justified and the honest conclusion is to say so.

---

## 6. Statistics

**The unit of analysis is the puzzle, not the cell.** Cells within one puzzle are strongly
correlated — a single wrong crossing entry corrupts five squares at once — so treating
~180 cells as independent samples produces absurdly narrow confidence intervals. Every
interval in the report is bootstrapped over puzzles.

- **Confidence intervals** — 2000-sample seeded bootstrap over puzzles, 95%, percentile
  method. Reported for solve rate and cell accuracy.
- **Paired comparison, solve rate** — McNemar's exact test. Systems run on identical
  puzzles, so the paired form is both more powerful and more appropriate than a two-sample
  proportion test. Reported as (A-only wins, B-only wins, two-sided *p*).
- **Paired comparison, continuous metrics** — paired bootstrap of the mean difference.
- **Multiple comparisons** — with seven ablations against `full`, Holm–Bonferroni
  correction is applied to the family and both raw and adjusted *p* are shown.

### Sample size

For McNemar, power depends on the **discordant** pairs. With $\pi$ the share of discordant
pairs won by the better system:

$$n_{disc} = \left(\frac{z_{\alpha/2} \cdot 0.5 + z_{\beta}\sqrt{\pi(1-\pi)}}{\pi - 0.5}\right)^2$$

At $\alpha=0.05$, 80% power, and $\pi = 0.75$ (the better system wins 3 of every 4
disagreements), that is **≈29 discordant pairs**. If systems disagree on about 20% of
puzzles, the suite needs **≈145 puzzles**.

Practical guidance, stated plainly so nobody over-reads a cheap run:

- **10 puzzles** (bundled) — smoke test. Reports no significance at all.
- **~50 puzzles** — direction-of-effect only; wide intervals.
- **~150 puzzles** — the minimum for a defensible ablation claim.
- **~350 puzzles** (50/weekday) — day-of-week breakdowns become individually meaningful.

The report prints the achieved discordant-pair count next to every *p*-value, so an
underpowered comparison is visible rather than implied.

---

## 7. Calibration and selective accuracy

An agent that is 80% accurate and *knows which 80%* is far more useful than one that is
85% accurate and uniformly overconfident — because the first can flag squares for a human
to check.

- **ECE / MCE** over 10 confidence bins, plus a reliability table.
- **Per-cell Brier score.**
- **Selective accuracy** — accuracy among cells above a confidence threshold, at
  thresholds 0.5/0.7/0.9/0.95/0.99, each with its coverage. The target shape is high
  accuracy at high coverage; a system whose accuracy doesn't rise as the threshold rises
  has confidence estimates that mean nothing.

Confidence for a cell is the belief-propagation marginal of the letter *actually written*
— not the maximum marginal. When search overrides BP's favourite letter to satisfy a
crossing word, the reported confidence for that square correctly drops.

---

## 8. Failure taxonomy

Aggregate numbers say a system failed; they don't say what to fix. Every wrong entry is
bucketed by a rule-based classifier into: proper-noun, wordplay, abbreviation, foreign,
fill-in-blank, multi-word, crosswordese, theme, plural/tense, other.

**This classifier is a heuristic triage aid, not ground truth.** It reads clue surface
features and will mislabel; it is there to point at the biggest bucket, and any claim
resting on it needs a manual read of the underlying entries. The report prints the raw
entries alongside the counts so that read is one glance away.

Theme detection deserves special mention: Thursday NYT puzzles often use a gimmick (rebus
squares, letters reading backwards, entries bending) that this agent has no representation
for. Those failures are *categorically* different from misreading a clue and are reported
separately rather than averaged in.

---

## 9. Reference points

Numbers from published systems, for orientation only — different corpora and protocols
mean these are **not** directly comparable to this project's:

- **Berkeley Crossword Solver** (Wallace et al., ACL 2022) — improved exact puzzle
  accuracy on NYT crosswords from 57% to 82%, with 99.9% letter accuracy on themeless
  puzzles. Neural QA for candidates + loopy belief propagation + local search: the same
  architecture family this project uses.
- **Dr.Fill + Berkeley NLP** at the 2021 American Crossword Puzzle Tournament — recorded
  the top score on puzzles #1–#7 (12,825 points, against 12,810 for the highest-scoring
  human, Erik Agard), the first time a program out-scored the human field.

A human expert baseline is **[proposed]** rather than claimed: timing a competent solver
on the same puzzles is cheap and would make the cost/latency axis interpretable.

---

## 10. Threats to validity

Stated explicitly, because a methodology that lists no weaknesses is not a methodology.

1. **Contamination** (§4) — the dominant threat. NYT numbers are an upper bound.
2. **Rebus/theme normalisation** — puzzles with gimmicks are scored on a simplification
   that favours the solver. Flagged per-puzzle in the report.
3. **Non-determinism** — the model is sampled at temperature > 0, so solve rate is itself
   a random variable. The clue cache makes a *re-run* reproducible, but a genuinely fresh
   run will vary. Repeated-run variance is **[proposed]**: three seeds per configuration
   with the spread reported.
4. **The failure classifier is heuristic** (§8).
5. **Cost is an estimate**, computed from published per-token list prices, not a bill.
6. **Sampling frame** — one publisher, one language, one era. Nothing here licenses a
   claim about cryptic crosswords, other publishers, or other languages.
7. **Paraphrase confounding** (§4, Probe C).
8. **Metric gaming** — solve rate on a fixed public corpus can be over-fitted by tuning
   prompts against it. Suite selection is seeded and declared in the report footer, and
   tuning should use a held-out split; the harness supports `--seed` for exactly this.

---

## 11. Reproducibility

Every report ends with a footer carrying the git SHA, model id, suite name, seed,
timestamp, and **the exact CLI command that reproduces the run**. Results stream to JSON
Lines as they complete, so an interrupted run resumes rather than restarting, and a saved
run can be re-reported without re-solving:

```bash
xword eval run --suite nyt:150 --systems full,greedy-llm,no-bp,no-repair --seed 0
xword eval report reports/run.json
```

The clue cache is keyed on (model, clue, length, pattern, mode), so re-running a suite is
free and byte-identical. **Cache hits are counted separately from misses** — otherwise the
reported cost of a warm re-run would be a lie.

---

## 12. What I would add next

In priority order:

1. **Fresh post-cutoff puzzle set** (Probe D) — the single change that would most improve
   the credibility of every number here.
2. **Repeated-seed variance** — three runs per configuration; report mean ± spread.
3. **Human baseline timings** — makes the latency and cost axes interpretable.
4. **Theme/rebus modelling** — currently the largest *categorical* failure bucket on
   Thursday and Sunday puzzles, and unlike clue errors it is a representational gap.
5. **Cryptic crosswords** — a genuinely different reasoning task and a clean test of
   whether the architecture generalises beyond recall.

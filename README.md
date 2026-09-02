# crossword-puzzle-agent

An AI agent that solves crossword puzzles by combining a language model that **reads
clues** with probabilistic inference and search that make the whole grid **agree with
itself**.

```
┌────────┐   ┌─────────┐   ┌──────┐   ┌───────────┐   ┌────────┐
│ INGEST │──▶│ PROPOSE │──▶│ FUSE │──▶│ PROPAGATE │──▶│ COMMIT │──┐
└────────┘   └─────────┘   └──────┘   └───────────┘   └────────┘  │
                  ▲                                               │
                  │        ┌────────┐   ┌────────────┐            │
                  └────────│ REPAIR │◀──│  CRITIQUE  │◀───────────┘
                           └────────┘   └────────────┘
              re-ask only the shaky clues, now supplying
                  the crossing letters it trusts
```

---

## Why this isn't just a prompt

Hand a language model 76 crossword clues and it returns 76 confident answers that don't
fit together. 17-Across says `OREO`; 5-Down insists the third letter is `A`; both are
stated with total certainty. A crossword is not a quiz — it's ~76 coupled constraints over
~180 shared letters.

So the agent:

1. asks the model for **several ranked answers per clue, with probabilities**, instead of
   one;
2. keeps explicit probability on *"the answer is in none of my candidates"* — the piece
   that stops a confident-but-wrong guess corrupting every entry that crosses it;
3. runs **loopy belief propagation** over the grid so every entry's opinion is tempered by
   what its crossings believe;
4. **searches** for the highest-scoring set of real words that mutually fit;
5. **critiques its own grid**, finds the weak entries, and **re-asks only those clues** —
   now handing the model the crossing letters it has become confident about.

Step 5 is the part that makes it an agent. It's also how humans solve: the clue you
couldn't get becomes easy once three of its letters are filled in.

Full design write-up: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.
Evaluation methodology: **[docs/EVALUATION.md](docs/EVALUATION.md)**.

---

## Does it work?

14 real New York Times puzzles, stratified two per weekday (so Saturdays and
Sundays count as much as Mondays), Claude Sonnet 5, seed 0:

| System | Exact solve rate (95% CI) | Cell acc | Word acc | $/puzzle | s/puzzle |
|---|---|---|---|---|---|
| **`full`** (the agent) | **50.0%** [21.4, 78.6] | 96.8% | 94.1% | $0.65 | 216 |
| `greedy-llm` (prompt the model, take its top answer) | **0.0%** [0.0, 0.0] | 71.2% | 49.3% | $0.04 | 9 |

The gap is the whole point of the project: same model, same clues, and the
scaffolding turns 0 solved puzzles into 7 of 14. McNemar's exact test on the 7
discordant puzzles gives **p = 0.016** (Holm-adjusted 0.016). Word accuracy
nearly doubles, 49.3% → 94.1%.

Solve rate by day of week tracks the difficulty gradient you would expect —
Mon 100%, Wed 100%, Tue/Thu/Sat 50%, Fri 0%, Sun 0% (n=2 each, so these are
directional only).

Confidence is worth reading: expected calibration error 3.8%, and selective
accuracy rises monotonically from 95.5% at 98% coverage to 97.7% at 92%
coverage — the agent's uncertainty points at the squares that are actually
wrong.

Full report, including the reliability table, failure taxonomy and the ten
entries it missed: **[docs/sample-evaluation-report.md](docs/sample-evaluation-report.md)**.

### How to read these numbers honestly

Three caveats, all of which matter:

1. **n = 14.** [docs/EVALUATION.md §6](docs/EVALUATION.md) works out that a
   defensible ablation claim needs ~150 puzzles. The `full` vs `greedy-llm` gap
   is large enough to clear significance anyway; the per-weekday rows are not,
   and the solve-rate CI is 57 points wide. This is a demonstration that the
   harness works on real data, not a benchmark result.
2. **Contamination.** Every puzzle predates 2019 and has been on the public web
   for years, so the model has very likely seen many of these clue/answer pairs
   in pretraining. **These are an upper bound on performance against fresh
   puzzles.** [§4](docs/EVALUATION.md) sets out the probes that size the effect.
3. **Provenance.** Measured at commit `90e493e`. Three later commits fix
   accuracy defects found by an adversarial review of this code — most
   significantly, belief propagation was applying the English letter prior once
   per crossing entry instead of once per square, which cost up to 10 points of
   letter accuracy in a 12-regime sweep and roughly doubled calibration error.
   The shipped code is therefore *better* than the table above, not worse, but
   the table was not re-measured against it. `xword eval run --suite nyt:14
   --systems full,greedy-llm --seed 0` reproduces it on the current code.

---

## Quick start

```bash
git clone https://github.com/tatra-labs/crossword-puzzle-agent
cd crossword-puzzle-agent

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

cp .env.example .env      # then put your Anthropic API key in it
xword doctor              # checks key, puzzles, lexicon, cache
xword demo                # solves a bundled puzzle end to end
```

`xword demo` needs nothing but an API key: ten original, CC0-licensed puzzles ship with
the repo, so a fresh clone works with no downloads.

### Getting an API key

Create one at <https://console.anthropic.com/settings/keys>, then either put it in `.env`:

```
ANTHROPIC_API_KEY="sk-ant-..."
```

or export it in your shell. A shell variable always wins over the file.

---

## Deploying the web demo

There is a FastAPI surface (`app.py`) and a demo page (`public/index.html`) that
stream the agent's trace while it solves. Import the repo in Vercel, set
`ANTHROPIC_API_KEY`, and **enable Fluid Compute** — without it the 10s Hobby
function ceiling kills every solve but the smallest mini, since a 15x15 measures
116-216s.

Full walkthrough, including the timeout table, the cost/access-control warning
and what is degraded versus running locally: **[docs/DEPLOY.md](docs/DEPLOY.md)**.

---

## Using it

### Solve a puzzle

```bash
xword solve data/puzzles/bundled/midi-01.json          # bundled original
xword solve path/to/puzzle.puz --clues                 # Across Lite binary
xword solve puzzle.json --model claude-opus-5 --rounds 5
xword solve puzzle.puz --out result.json --html grid.html
```

Useful flags:

| Flag | Effect |
|---|---|
| `--clues` | print the full clue table with per-entry answers and confidence |
| `--quiet` | no live trace (for scripting) |
| `--model` / `-m` | model for the clue-reading pass |
| `--rounds` / `-r` | max agent rounds (default 4) |
| `--candidates` / `-k` | candidates requested per clue (default 10) |
| `--budget` | wall-clock budget in seconds |
| `--no-lexicon` | solve without the word list |
| `--seed` | make search deterministic |
| `--out` / `--html` | write a result JSON / a rendered grid |

`xword solve` exits `0` when a puzzle with a known solution is solved exactly, `1`
otherwise — so it drops straight into a CI check.

### Look at a puzzle without solving it

```bash
xword show data/puzzles/bundled/maxi-01.json
```

### Evaluate

```bash
# 10 bundled puzzles — a smoke test, not a benchmark
xword eval run --suite bundled --systems full,greedy-llm

# a real measurement (see the corpus note below)
xword fetch nyt --per-year 14
xword eval run --suite nyt:150 --systems full,greedy-llm,no-bp,no-repair --workers 4

# regenerate a report without re-solving anything
xword eval report reports/run.json
```

This writes `reports/report.md` and `reports/report.html`: headline solve rate with
bootstrap confidence intervals, a difficulty breakdown, ablation deltas with McNemar
*p*-values, calibration and selective accuracy, a failure taxonomy, and a reproducibility
footer containing the exact command that produced it.

### Word list

The solver uses a scored word list for pattern-matching (`?A??E?` → what fits). A small
fallback ships in-package; build the full one for materially better fill:

```bash
xword fetch wordlist          # public-domain English word list (Unlicense)
xword lexicon build           # score it, and mine answers from any fetched puzzles
xword lexicon match "?RE?O"   # sanity-check what it knows
```

Mining answers from real puzzles matters more than dictionary size: an entry that has
appeared in many published crosswords is far better fill than a random dictionary word.

---

## Evaluation corpus, and a licensing note

`xword fetch nyt` downloads from the public [`doshea/nyt_crosswords`][archive] archive
(1976–2018) for evaluation.

**That content is New York Times copyright.** It is downloaded by you, onto your machine,
into a gitignored directory (the fetcher writes a `.gitignore` containing `*` into it), and
is never redistributed by this project. The puzzles committed to this repo are the ten
originals under `data/puzzles/bundled/`, released CC0-1.0.

**Contamination warning, stated up front:** every puzzle in that archive predates 2019 and
has been reproduced across the public web for years, so any modern model has likely seen
many of these clue/answer pairs in pretraining. Solve rates measured on it are an **upper
bound** on performance against fresh puzzles. [docs/EVALUATION.md §4](docs/EVALUATION.md)
sets out the probes this project uses to size that effect — including a clue-only baseline
whose gap from the full solve is the honest measure of what the agent's machinery adds on
top of recall.

[archive]: https://github.com/doshea/nyt_crosswords

---

## Supported formats

| Format | Extension | Notes |
|---|---|---|
| Across Lite | `.puz` | binary; read **and** write; rebus and circles preserved in metadata |
| Native | `.json` | this project's own format — see `data/puzzles/bundled/README.md` |
| NYT archive | `.json` | the schema used by `doshea/nyt_crosswords` |
| ipuz | `.ipuz`, `.json` | v1 crossword subset |

`xword solve` sniffs the format by extension and then by content, so `foo.json` in any of
the three JSON schemas just works.

---

## Development

```bash
pytest                       # unit tests; the live-API tests are skipped by default
pytest -m live               # the tests that really call the API (needs a key)
ruff check src tests
```

The whole pipeline is testable offline: `LLMCandidateSource` accepts an injected client and
ships with a `FakeClient` that answers from a dictionary, so fusion, belief propagation,
search, repair, and scoring all run deterministically with no API key.

### Layout

```
src/xword/
  core/       types, grid geometry, the belief container   (no dependencies)
  lexicon/    numpy-bitset pattern index + scored word list
  candidates/ clue answering (Claude), caching, fusion/calibration
  solver/     belief propagation, constrained search, the agent loop
  io/         .puz / ipuz / NYT / native readers, terminal + HTML rendering
  eval/       metrics, suites, harness, reports
  cli.py      everything above, reachable as `xword <verb>`
```

---

## Known limits

- **Themed puzzles.** Thursday and Sunday NYT puzzles often use a gimmick — rebus squares,
  entries reading backwards or bending. The agent has no representation for that and solves
  the grid as if it were themeless. Evaluation reports these failures separately rather
  than averaging them in.
- **Rebus squares** are normalised to their first letter at parse time; the raw rebus map
  is kept in puzzle metadata, and affected puzzles are flagged in the report.
- **Cryptic crosswords** are out of scope — the clue prompt is tuned for American-style
  clues.
- **Loopy belief propagation is approximate.** On densely crossed grids it can settle into
  a confidently wrong fixed point; the local-search repair stage exists partly to escape
  those.

---

## Prior art

This is the same architecture family as the **Berkeley Crossword Solver**
([Wallace et al., ACL 2022](https://aclanthology.org/2022.acl-long.219/)) — neural
candidate generation, loopy belief propagation, local search — which raised exact puzzle
accuracy on NYT crosswords from 57% to 82% and reached 99.9% letter accuracy on themeless
puzzles. Its predecessor **Dr.Fill** (Matt Ginsberg), with the Berkeley NLP group, recorded
the top score at the 2021 American Crossword Puzzle Tournament.

What's different here: the neural QA component is a modern instruction-following LLM asked
for calibrated distributions, and there is an explicit **self-critique step** that decides
which clues to re-ask and what partial letters to reveal when re-asking them.

## License

MIT — see [LICENSE](LICENSE). Bundled puzzles under `data/puzzles/bundled/` are CC0-1.0.

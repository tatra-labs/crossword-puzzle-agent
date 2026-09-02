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

## The studio UI

A three-panel page for driving the agent by hand and reading what it actually
sent the model. Two commands from a fresh clone:

```bash
pip install -e .                       # brings fastapi + uvicorn with it
python -m uvicorn app:app --port 8000
```

Then open <http://127.0.0.1:8000>. There is no build step — no `npm install`, no
bundler; `public/studio.js` is one plain ES module and `public/studio.css` is
plain CSS, both served from `/static/`.

You need `ANTHROPIC_API_KEY` set (in `.env` or the environment) before a solve
will start. Without it the page still loads, lists the puzzles and lets you
inspect them, and the Solve button is disabled and says why. `GET /api/health`
is the same check in JSON.

### What the three panels do

| Panel | Holds |
|---|---|
| **Left** | The ten bundled puzzles, smallest first, then every live and finished session with its state, a ticking clock, and its current round and step. A session keeps running here while you go and look at something else. |
| **Centre** | The puzzle you are inspecting — empty grid, numbering, clues in two columns, hover highlighting both ways — or the session you are watching, with the grid filling in, a stats strip, and right/wrong marking once it finishes. |
| **Right** | The trace. Phase steps as chips; each model call as a card that expands into the system prompt and clue batch **as sent**, the tool offered, the `tool_input` that came back, tokens and duration. |

### A first run

1. `mini-02` is selected by default — a 5×5, about 8 seconds and **$0.007**.
2. Press **Solve**, and watch the right panel. Expand a `batch` card to see the
   prompt and the model's ranked candidates with probabilities.
3. While it runs, click another puzzle. The solve keeps going in the sidebar.
   Click its row to come back — the trace is complete, not just the part that
   happened after you returned.
4. **stop** on a running row ends it at its next checkpoint and keeps the
   partial grid. **dismiss** on a finished row forgets it.

**Every solve spends real credit.** Roughly $0.007 for a 5×5 and ~$0.65 for a
15×15; the button says what the selected puzzle will cost before you press it.

### Knobs

```bash
python -m uvicorn app:app --port 8000 --reload      # reload on edit
XWORD_MAX_CONCURRENT_SESSIONS=1 python -m uvicorn app:app   # tighter spend cap (default 3)
XWORD_ACCESS_TOKEN=<secret> python -m uvicorn app:app       # require a token
XWORD_CACHE_DIR=/tmp/cold python -m uvicorn app:app         # cold clue cache
```

With a token set, every solve **and every session read** requires it — a trace
carries the verbatim prompts, the answers and the cost, so reads are not free to
give away. Open the page as `http://127.0.0.1:8000/?token=<secret>`; there is
deliberately no input field. `/api/health`, the puzzle list and the page itself
stay open so it can load and tell you that.

The clue cache makes a re-solve of the same puzzle nearly instant and free, which
is why a second run reports `0 model calls · $0.00`. Point `XWORD_CACHE_DIR`
somewhere empty to force real calls.

Sessions are threads in the server process, so **they are a local capability**.
`/api/health` reports `durable_sessions`, and where it is false the page falls
back to one solve per request. **[docs/UI.md](docs/UI.md)** covers the panels,
the cap, the access-control story and what is degraded in more detail.

---

## Deploying to Vercel

Import the repo, set `ANTHROPIC_API_KEY`, and **enable Fluid Compute** — without
it the 10s Hobby function ceiling kills every solve but the smallest mini, since
a 15x15 measures 116-216s. `app.py` is the ASGI entrypoint; `requirements.txt`
is deliberately narrower than `pyproject.toml` to keep the bundle small.

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
pytest                       # 390 tests, ~4s, no API key and no network
ruff check src tests app.py
node --check public/studio.js
```

The whole pipeline is testable offline: `LLMCandidateSource` accepts an injected client and
ships with a `FakeClient` that answers from a dictionary, so fusion, belief propagation,
search, repair, the session registry and scoring all run deterministically with no API key.
**No test in the suite calls the real API.** A `live` marker is registered in `pytest.ini`
for tests that would, and `pytest -m live` currently selects nothing — the marker is there
so that such a test cannot be added without opting in.

### Layout

```
src/xword/
  core/       types, grid geometry, the belief container   (no dependencies)
  lexicon/    numpy-bitset pattern index + scored word list
  candidates/ clue answering (Claude), caching, fusion/calibration
  solver/     belief propagation, constrained search, the agent loop
  io/         .puz / ipuz / NYT / native readers, terminal + HTML rendering
  eval/       metrics, suites, harness, reports
  web/        the trace log and the session registry behind the UI
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

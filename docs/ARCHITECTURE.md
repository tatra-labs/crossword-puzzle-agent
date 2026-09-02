# Architecture

## The problem with asking a model 76 questions

A crossword looks like a quiz and isn't. Handing 76 clues to a language model and
taking its best answer for each produces a grid that does not agree with itself: 17-Across
says `OREO`, 5-Down says the third letter is `A`, and both are stated with total
confidence. The clues are not independent questions — they are 76 coupled constraints
over ~180 shared variables.

Two things follow, and they shape everything else:

1. **Uncertainty must survive.** A single best answer per clue throws away exactly the
   information the grid needs. The model has to emit a *distribution*, and that
   distribution has to include "I might be wrong about all of these".
2. **The hard part is global, not local.** Choosing 76 words that mutually agree is a
   weighted constraint-satisfaction problem. Language models are bad at that and search
   algorithms are good at it.

So this system is a **neuro-symbolic loop**: a language model reads clues, a probabilistic
solver reconciles them, and the solver's partial conclusions are fed *back* to the model
as constraints. That last edge is what makes it an agent rather than a pipeline.

This is the same architecture family as the Berkeley Crossword Solver (Wallace et al.,
ACL 2022) — neural candidate generation, loopy belief propagation, local search — with the
neural QA component replaced by a modern instruction-following LLM, and with an explicit
self-critique step driving re-querying.

---

## The loop

```
                    ┌──────────────────────────────────────────────┐
                    │                                              │
                    ▼                                              │
  ┌────────┐   ┌─────────┐   ┌──────┐   ┌───────────┐   ┌────────┐ │
  │ INGEST │──▶│ PROPOSE │──▶│ FUSE │──▶│ PROPAGATE │──▶│ COMMIT │ │
  └────────┘   └─────────┘   └──────┘   └───────────┘   └────────┘ │
   parse .puz   Claude, in    pool +      loopy belief    beam +    │
   → entries    batches, k    calibrate   propagation     LDS +     │
   + crossings  answers w/    + reserve   over the        local     │
                probabilities "none"      factor graph    repair    │
                                                             │      │
                                        ┌────────┐   ┌───────▼────┐ │
                                        │ VERIFY │◀──│  CRITIQUE  │─┘
                                        └────────┘   └────────────┘
                                         complete     which entries
                                         the grid     are weak, and
                                                      why → re-ask
                                                      with the
                                                      letters we now
                                                      trust
```

The feedback edge is the point. It is how humans solve: the clue you couldn't get becomes
easy once three of its letters are filled in. Round 0 asks about every clue blind; each
later round re-asks only about entries the agent distrusts, now supplying the crossing
letters it has become confident about.

---

## Stage by stage

### INGEST — `core/grid.py`, `io/`

A rectangle of black and white squares becomes numbered entries and a crossing index.
`.puz` (Across Lite binary), `.ipuz`, the NYT JSON archive schema, and a native JSON format
all normalise to one `Puzzle`. Rebus squares are folded to their first letter and the raw
rebus map is preserved in metadata — a documented simplification, flagged in evaluation.

`GridIndex` precomputes what the solver needs tens of thousands of times: which entries
cover a cell, which entries cross which, and which cells are *unchecked* (covered by only
one entry — those carry no crossing evidence and the inference must not assume otherwise).

### PROPOSE — `candidates/llm.py`, `candidates/prompts.py`

Clues are batched (a dozen per request — they share the system prompt and puzzle context,
so batching is a large cost saving) and answered through a **tool call with a JSON schema**,
never free-text parsing. Each clue comes back as up to *k* ranked candidates with
probabilities.

The system prompt is where most of the accuracy lives, and it is isolated in its own module
so it can be reviewed and A/B tested. It encodes the conventions a solver learns by
experience: an abbreviation in the clue means an abbreviated answer; a question mark means
wordplay; tense and number must agree; `___` is a fill-in-the-blank; crosswordese is real
and short entries are disproportionately common fill. It also insists on **honest
probabilities** — a calibrated 0.3 is worth far more to the solver downstream than a
bluffed 0.95.

Everything the model returns is then distrusted and validated: uppercased, stripped to
A–Z, dropped if the length is wrong or a supplied pattern is violated, de-duplicated. Every
drop is counted.

Responses are cached in SQLite keyed on (model, clue, length, pattern, mode), so re-runs
are free and byte-identical — with hits counted separately from misses, so a warm re-run
cannot silently misreport its cost.

### FUSE — `candidates/fusion.py`

Model candidates and pattern-matching lexicon candidates are pooled by weighted log-linear
combination, with a bonus when independent sources agree, then softmaxed into a
distribution per entry.

The important part is `estimate_null_mass`: every entry keeps explicit probability on
*"the true answer is in none of my candidates"*, rising when the model's top probability is
low, when it returned few candidates, when the entry is long, or when a source returned
nothing. This is the solver's humility. Without it, a confident-but-wrong answer gets
forced into the grid and corrupts every crossing entry; with it, belief propagation can
prefer the crossings' opinion over the clue-reader's.

### PROPAGATE — `solver/beliefs.py`

The grid is a factor graph: one 26-valued variable per open cell, one factor per entry
whose potential is that entry's candidate distribution plus the null branch. Loopy belief
propagation passes messages until the marginals settle.

It is *loopy* — every crossing creates a cycle, so this is approximate inference, not
exact. It works well here anyway, which is the empirical finding the Berkeley work rests
on.

The implementation detail that matters: candidate weights are computed in log space, and
the leave-one-out message is formed by **subtracting** a position's own log-message rather
than dividing by it. Dividing by a near-zero message is the classic way this algorithm
silently produces `NaN`s across an entire grid.

Output: a letter distribution per cell, a posterior per candidate, and — most useful to
the next stages — per-cell entropy and per-entry margin, i.e. *where the agent is shaky*.

### COMMIT — `solver/search.py`

Marginals are not an answer. Every entry must hold a real word, and all of them
simultaneously. Search picks one word per entry maximising total log-probability:

- **Pools** merge fused candidates with lexicon words matching BP's marginal pattern,
  where a lexicon word scores well only if BP's letters already like it.
- **Beam search** proceeds most-confident-entry-first (filling what you're sure of first is
  what makes the crossings informative), with forward checking: prune any state where an
  unassigned crossing entry has no surviving word.
- **Limited-discrepancy search** retries with a bounded number of deviations from the
  greedy choice when the beam dead-ends.
- **Local repair** tears out the worst entry and its neighbours and re-fills that small
  region, accepting improvements, with seeded restarts.
- **`complete_from_marginals`** guarantees a full grid: any square still empty takes the
  argmax of its marginal. A blank scores zero, so a low-confidence guess strictly dominates.

Everything respects a wall-clock budget and returns the best partial answer rather than
running over. Everything is deterministic for a fixed seed.

### CRITIQUE — `solver/agent.py`

The agent then examines its own grid and ranks entries by how much it distrusts them. Four
distinct smells, all remedied the same way (ask again, with letters):

| Smell | Meaning |
|---|---|
| `unfilled` | search could not satisfy this entry at all |
| `unproposed` | the answer was forced in by crossings — the clue was never really read |
| `low-confidence` | the letters written have thin belief behind them |
| `thin-margin` | the runner-up candidate is nearly as good as the winner |

Entries with no crossing letters yet are skipped: re-asking would produce the identical
question and burn a call for nothing.

### REPAIR

The worst entries are re-asked with a **pattern of only the letters the agent genuinely
trusts** (threshold well above the repair threshold). This restraint is deliberate:
feeding a shaky letter to the model as a hard constraint is actively harmful — it will
loyally produce a wrong answer that fits the wrong letter, and the error becomes
self-confirming.

Re-asked clues use a prompt that demands explicit wordplay analysis first -- what kind of
clue is this, which part is the definition, which part is the wordplay -- before answering.

They can also escalate to a **stronger model**: set `XWORD_HARD_MODEL=claude-opus-5` and
the repair rounds run there instead. This is off by default because it roughly doubles the
cost of a solve, and the numbers in the README were measured without it. The two models
share one clue cache, whose key includes the model, so an escalated answer never
overwrites the cheap model's cached answer for the same clue.

Then propagate and commit again. Repeat until nothing is worth re-asking, confidence
crosses a threshold, or a budget runs out.

### VERIFY

Confidence for each cell is the BP marginal of **the letter actually written** — not the
maximum marginal. When search overrides BP's favourite letter to satisfy a crossing word,
the reported confidence for that square correctly drops. That is what makes the confidence
heat-map worth looking at, and what makes selective accuracy meaningful in evaluation.

---

## Layering

Dependencies point one way, which is what keeps the pieces independently testable:

```
core/{types,grid,beliefs}      ← no dependencies; pure data + geometry
        ▲
lexicon/{index,store,build}    ← numpy bitset pattern index
        ▲
candidates/{llm,fusion,...}    ← generation and calibration
        ▲
solver/{beliefs,search}        ← inference and discrete optimisation
        ▲
solver/agent.py                ← the loop
        ▲
io/, eval/, cli.py             ← boundaries: files, measurement, humans
```

`core/beliefs.py` sits low precisely so that `candidates` and `solver` can both use
`SlotBeliefs` without importing each other.

Two seams exist for testing: `CandidateSource` is a protocol, and `LLMCandidateSource`
accepts an injected client. A `FakeClient` answers from a dictionary, so the entire
pipeline — fusion, BP, search, repair, scoring — runs offline and deterministically with no
API key.

---

## Performance notes

- **Pattern matching** is the inner loop: "which words match `?A??E?`" runs tens of
  thousands of times per puzzle. `lexicon/index.py` answers it with per-(position, letter)
  numpy bitsets over word indices — AND the constrained positions, unpack, done — rather
  than scanning a word list or walking a trie of dicts.
- **BP is vectorised per entry.** Candidate answers become a `K × L` int8 matrix and
  messages are scattered with `np.bincount`, so 60 iterations over a 15×15 grid with 40
  candidates per entry costs a fraction of a second.
- **The API is the bottleneck, not the CPU.** Batching plus a thread pool plus the SQLite
  cache is where the wall-clock and the cost actually go.

---

## Known limits

Stated here rather than discovered later:

- **Themed puzzles.** Thursday and Sunday NYT puzzles often use a gimmick — rebus squares,
  entries reading backwards or bending around corners. The agent has no representation for
  any of that; it solves the grid as if it were themeless. This is the largest
  *categorical* failure bucket, and evaluation reports it separately rather than averaging
  it in.
- **Rebus squares** are normalised to a single letter (§ INGEST).
- **Cryptics** are out of scope. The clue-reading prompt is tuned for American-style
  clues; cryptic wordplay is a different reasoning task and would need its own treatment.
- **Loopy BP is approximate.** On densely-crossed grids it can converge to a confidently
  wrong fixed point; the local-search repair stage exists partly to escape those.

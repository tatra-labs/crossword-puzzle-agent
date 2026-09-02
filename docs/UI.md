# The studio UI

A three-panel page for driving the agent by hand: pick a puzzle, inspect it,
solve it, and read every prompt and tool call the agent made while it worked.
It is served by the same `app.py` as the JSON API, from `public/`.

The point of it is the trace. `xword solve` prints the phase events; this shows
the model requests underneath them — the system prompt, the clue lines, the tool
schema offered, the tool call that came back, tokens and duration per call. That
is the part you need when the agent gets an entry wrong and you want to know
whether the clue was misread or the grid overrode a correct answer.

---

## Run it locally

```bash
python -m uvicorn app:app --port 8000
# then open http://127.0.0.1:8000
```

Nothing else to build — no bundler, no `npm install`. `public/studio.js` is a
plain ES module and `public/studio.css` is plain CSS, both served from two fixed
routes (`/static/studio.js`, `/static/studio.css`).

Set `ANTHROPIC_API_KEY` first, in the environment or in `.env`. Without it the
page still loads, lists puzzles and lets you inspect them; the **Solve** button
is disabled and says why. `GET /api/health` is the same check in JSON.

**Every solve spends real credit** — roughly $0.007 for a 5×5 and ~$0.65 for a
15×15. The button is a spending button. See [the cap](#the-concurrency-cap).

---

## The three panels

| Panel | Holds |
|---|---|
| **Left** | Bundled puzzles, then the live and finished sessions. A health dot at the top carries the deployment's state and the reason in its tooltip. |
| **Centre** | The puzzle being inspected, or the session being watched — grid, clues, stats, entries. |
| **Right** | The trace for whichever session is open: steps and model calls in one ordered log. |

**Left.** Puzzles come in the API's order — smallest first, which is
deliberately not the order that puts the slowest puzzle at the top. Each row
carries its size, entry count, and a marker when the puzzle is too big for this
deployment's limits (it is then unselectable) or merely slow. Below that, every
session newest-first with a state pill, a ticking clock, its current round and
step, and an inline stop or dismiss.

**Centre, inspecting.** The empty grid with its numbering, the clues in two
columns, and hover highlighting that works both ways — a clue lights its
squares, a square lights the clues that cross it. Beside **Solve** are the model,
round and candidate-count settings, and a rough cost/time estimate interpolated
between the two measured anchors in the README. It is labelled rough because it
is: it will be wrong in the middle of the range.

**Centre, watching a session.** The same grid, filling in as the solve
progresses, plus stats and the entries table. Two honesty notes about what you
are looking at:

- While the solve runs, letters are **provisional** — they are the first
  candidate per entry taken from each model call's tool input, drawn dimmed. The
  trace stream carries numbers and strings, not per-cell beliefs, so nothing
  better is available mid-solve.
- Per-cell confidence shading, and right/wrong marking against the reference
  solution, appear only once the `result` event lands.

**Right.** Steps render compactly (phase chip, round, message). Model calls
render as cards: collapsed they show kind, model, duration, tokens, attempts and
any error; expanded they show the prompt (system folded by default), the tool
call with its input, and the result, each block with a copy button. A call served
from the clue cache is drawn as "no API call" rather than as an empty request,
because that is what it is. Filter by steps or model calls; auto-scroll follows
the tail and switches itself off when you scroll up to read something.

---

## Sessions outlive the request

A solve runs on its own thread in the server process and writes to an
append-only trace log. That is what makes the behaviour the UI needs possible:

- Start a solve, navigate to a different puzzle, come back — the session is
  still running and the trace is intact.
- Watch two sessions in sequence without losing either one's history.
- Stop a session from anywhere.

Every subscriber reads the log **from a cursor**, so re-attaching replays what
was missed and then tails live. The page streams over SSE
(`GET /api/sessions/{sid}/stream?cursor=N`) and falls back to polling
(`GET /api/sessions/{sid}/events?cursor=N`) if `EventSource` is unavailable or
the stream keeps failing. Either way the events are the same five types:

| Type | Payload |
|---|---|
| `status` | The session's state and a message. The **terminal** `status` is end-of-stream, not `result`. |
| `step` | One agent phase event: kind, round, message, and numeric data. |
| `llm_call` | One model request: prompts, tools, tool call, tokens, duration, attempts, error. |
| `result` | The finished solve, the same shape `POST /api/solve` returns. |
| `error` | The session died; the message carries the traceback tail. |

The registry keeps 40 sessions and evicts the oldest **finished** ones; a
running session is never evicted.

---

## This is a local capability

Sessions live in the server process, which means:

- **Vercel cannot do this.** A Function is frozen once it responds, so a thread
  started during a request stops making progress the moment the response is
  sent, and the next request need not even reach that instance. So
  `durable_sessions` is false whenever `VERCEL` is set, `POST /api/sessions`
  answers **501** there rather than charging for a trace nobody can read back,
  and the page degrades to `POST /api/solve/stream` — one solve inside one
  request, no session list, and it dies with the response. The trace panel still
  works; there are simply no `llm_call` records in it, because that endpoint
  predates them.
- **One worker only.** `--workers 2` gives each worker its own registry, so a
  session created on one is a 404 on the other. Sharing it would need out-of-process
  state, which is a different design.
- **A restart loses everything.** The registry is memory, not a database.

`GET /api/health` reports `durable_sessions`, `max_concurrent_sessions` and
`active_sessions`, and the page reads the first of those to decide which path to
take. If it is missing, the page assumes false and uses the legacy path.

---

## The concurrency cap

Three solves at once, by default. Over that, `POST /api/sessions` answers **429**
with a message naming the cap; it does not queue.

```bash
XWORD_MAX_CONCURRENT_SESSIONS=1 python -m uvicorn app:app --port 8000
```

The cap exists because **concurrency here is spend, not load.** The work is
almost entirely waiting on the Anthropic API, so the process could happily run
twenty solves at once — and twenty 15×15s is about $13 of credit committed by
twenty clicks, with no way to un-spend it. Refusing the fourth click before any
money moves is the useful failure. It is the same reasoning as the `413` on
oversized puzzles: refused up front rather than started and abandoned part-way.

If you want the page reachable by other people, set `XWORD_ACCESS_TOKEN`. It
guards the three mutating routes — create, stop, delete — as an `X-Access-Token`
header or `?token=`, and leaves the read routes open, since those spend nothing.
There is no per-IP rate limiting in this app.

---

## Stopping a solve

`POST /api/sessions/{sid}/stop`, or the stop button on the session row. What it
actually promises is narrower than it looks:

- It stops **new** requests. Requests already in flight run to completion, and
  **you are billed for them** — their candidates are kept rather than thrown
  away, so a stopped solve returns whatever had arrived.
- Cancellation is checked between phases and before each batch, so a stop lands
  within a round rather than instantly.
- A stop that lands before the first commit returns an **empty grid**. That is
  not a failure: the spend up to that point is still reported in the stats, and
  the result carries a `stopped_before_commit` note explaining the blank.
- A session that finished as `stopped` means only that a stop was requested
  before the agent returned. The registry cannot tell "cut a round short" from
  "the stop arrived as it was finishing anyway".

Stopping a queued session, or one that has already finished, changes nothing and
says so.

---

## Routes the page uses

| Route | For |
|---|---|
| `GET /api/health` | Key, lexicon, limits, durability. |
| `GET /api/puzzles` | The sidebar list. |
| `GET /api/puzzles/{pid}` | The inspect view: shape, numbering, clues with start square and length. Never the answers. |
| `POST /api/sessions` | Start a solve. 429 over the cap, 501 where sessions cannot be held, 503 with no key, 413 on an oversized puzzle. |
| `GET /api/sessions` | The session list, plus `max_concurrent` and `durable`. |
| `GET /api/sessions/{sid}` | One session, with its result once there is one. |
| `GET /api/sessions/{sid}/stream?cursor=N` | Replay from the cursor, then tail live. |
| `GET /api/sessions/{sid}/events?cursor=N` | The same events, non-blocking, as the fallback. |
| `POST /api/sessions/{sid}/stop` | Ask a solve to stop. |
| `DELETE /api/sessions/{sid}` | Dismiss a finished session. Refuses a live one. |
| `POST /api/solve/stream` | The legacy single-request path, used where sessions are unavailable. |

`POST /api/solve` still exists and behaves exactly as it always did; the page
does not use it. See [DEPLOY.md](DEPLOY.md) for the deployed API surface.

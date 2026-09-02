# Deploying to Vercel

## Why the bare repository failed

This project is a CLI and a library: no `vercel.json`, no `api/` directory, no
`requirements.txt`, no web entrypoint, no frontend. Vercel's build step looks for
a framework or a serverless function, finds neither, and stops — so the deploy
never gets as far as running anything.

The fix is a web surface, which now exists:

| File | Purpose |
|---|---|
| `app.py` | The entrypoint. Vercel's Python runtime loads the top-level `app` (a FastAPI ASGI app) and routes every request to it. |
| `public/index.html` | The page: puzzles and sessions on the left, the grid and clues in the middle, the agent's trace on the right. See [UI.md](UI.md). |
| `public/studio.css`, `public/studio.js` | The client, served from `/static/`. No build step: plain CSS and one ES module. |
| `requirements.txt` | Runtime dependencies — deliberately narrower than `pyproject.toml` (see [Bundle size](#bundle-size)). |
| `vercel.json` | `maxDuration`, the `/tmp` cache paths, and `excludeFiles`. |
| `.vercelignore` | Keeps `.env`, the NYT corpus and build artefacts out of the upload. |
| `pyproject.toml` | `[tool.vercel] entrypoint = "app:app"`, so Vercel doesn't guess at the `src/` layout. |

---

## Deploy it

The repository is already on GitHub, so the Git integration is the shortest path:

1. **Vercel → Add New → Project → import `tatra-labs/crossword-puzzle-agent`.**
   Framework detection will say *FastAPI*; leave the build settings alone.
2. **Set the API key.** Project → Settings → Environment Variables:
   ```
   ANTHROPIC_API_KEY = sk-ant-...
   ```
   Nothing solves without it. `/api/health` reports `api_key_configured: false`
   and the UI disables the button rather than failing mid-request.
3. **Turn on Fluid Compute** (Project → Settings → Functions). This is not
   optional — see [Timeouts](#timeouts-the-real-constraint).
4. Deploy. `GET /api/health` is the smoke test.

Or from the CLI:

```bash
npm i -g vercel
vercel login
vercel link
vercel env add ANTHROPIC_API_KEY production
vercel --prod
```

---

## Timeouts: the real constraint

This is the thing that decides whether the deployment is useful.

| Plan | Function ceiling |
|---|---|
| Hobby, default | **10s** |
| Hobby + Fluid Compute | 300s |
| Pro | 60s default, up to 300s (800s+ with Fluid) |

Measured solve times for this agent:

| Puzzle | Squares | Time | Fits in 10s? | Fits in 300s? |
|---|---|---|---|---|
| 5×5 mini | 19 | ~8s | borderline | yes |
| 7×7 midi | 36 | ~11s | no | yes |
| 9×9 / 11×11 | 63–90 | ~20–60s | no | yes |
| 15×15 daily | 189 | **116–216s** | no | yes, but not comfortably |
| 21×21 Sunday | ~358 | minutes | no | **no** |

So: **without Fluid Compute, every request but the smallest mini is killed
mid-solve.** With it, `vercel.json` sets `maxDuration: 300` and `app.py` gives
the agent a 275s budget — 25s of head-room so it stops on its own terms and
returns its best partial grid instead of being killed by the platform.

A 21×21 is refused up front with a `413` and an explanation, rather than
started and abandoned part-way after spending money on it.

**If the deploy fails with a `maxDuration` validation error,** your plan doesn't
allow 300s. Lower it in two places that must agree:

```jsonc
// vercel.json
"functions": { "app.py": { "maxDuration": 60 } },
"env": { "XWORD_FUNCTION_MAX_SECONDS": "60" }
```

At 60s the minis and midis work and 15×15s will usually come back partial. At
10s, only `mini-*` has a chance.

---

## Cost and access control

**Every solve spends real Anthropic credit** — roughly $0.007 for a 5×5, ~$0.65
for a 15×15. A public Vercel URL with your key in its environment is a public
spending endpoint.

Two ways to close that, and you want at least one:

- **Vercel Deployment Protection** (Settings → Deployment Protection) — the
  simplest: requires a Vercel login to reach the deployment at all.
- **A shared secret.** Set `XWORD_ACCESS_TOKEN` and every `/api/solve` and
  `/api/sessions` route requires it as an `X-Access-Token` header or a `?token=`
  parameter — the session **reads** included, because a trace carries the prompts
  sent, the answers returned, the cost, and the solution of an inline puzzle.
  `/api/health`, `/api/puzzles*` and the page and its assets stay open, since the
  page has to load before it can send anything. Open the page as
  `?token=<secret>`; it caches the secret and sends it on every request from
  then on. Unset, the app is open by design.

There is no per-IP rate limiting in this app. If you make it genuinely public,
add Vercel's WAF rate limiting in front of `/api/solve*`.

---

## Bundle size

Python functions get no tree-shaking: everything reachable at build time ships.
Two things keep this lean.

`requirements.txt` is narrower than `pyproject.toml`. The web surface needs the
agent, not the CLI or the statistics stack:

- **dropped `scipy`** (~100 MB) — it was a module-level import in
  `eval/metrics.py` for one call, `binomtest`. It is now imported lazily with an
  exact stdlib fallback (verified identical to SciPy across all 325 cases of
  `n ≤ 24`), so scoring works without it.
- **dropped `typer`, `tqdm`** — CLI only.

Verified by exercising every endpoint and asserting neither `scipy`, `typer` nor
`tqdm` appears in `sys.modules`.

`excludeFiles` in `vercel.json` drops `tests/`, `docs/`, `reports/`, `scripts/`,
the fetched NYT corpus and the built lexicon.

---

## What is degraded compared to running locally

Worth knowing before you judge the hosted demo:

- **The lexicon is the built-in fallback (~3,200 words), not the full ~378,000.**
  The real one is built by `xword lexicon build` from a downloaded word list and
  mined puzzle answers, and both are gitignored — the word list is 3.7 MB of
  third-party data and the mined answers derive from a copyrighted corpus. So
  pattern-fill is weaker here than on your machine. The LLM does most of the
  work, so minis are unaffected; larger grids lose some of the crossing-driven
  recovery. `/api/health` reports `lexicon_is_fallback`.
- **The clue cache is cold on most requests.** It lives in `/tmp`, which is
  per-instance and evaporates; locally it makes re-runs free and byte-identical.
- **No `xword eval`.** The evaluation harness needs the NYT corpus, which is
  never deployed. Evaluation is a local activity.
- **No background sessions**, so no session list, no stopping a solve, and no
  per-call trace: the page runs one solve inside one request instead. This is
  the biggest difference between the hosted demo and running it locally.

---

## Endpoints

| Route | What it does |
|---|---|
| `GET /` | The demo page. |
| `GET /api/health` | Key configured, lexicon size, limits, model — the `xword doctor` of the web. |
| `GET /api/puzzles` | Bundled puzzles with sizes and a `fits_here` flag. |
| `POST /api/solve` | Solve and return the finished grid. Blocking. |
| `POST /api/solve/stream` | Server-sent events: one frame per agent step, then the result. |
| `GET /api/docs` | Generated OpenAPI docs. |
| `POST /api/sessions` and friends | Background solves with a re-subscribable trace. **501 here** — see below. |

The session routes (`/api/sessions*`) are what the UI drives locally, and they
are the one part of this API a Vercel Function cannot serve: it is frozen once
it responds, so a solve on a background thread stops the moment the response is
sent. `durable_sessions` is false whenever `VERCEL` is set, `POST /api/sessions`
answers `501` rather than charging for a trace nobody can read back, and the
page falls back to `/api/solve/stream`. [UI.md](UI.md) has the details.

```bash
curl -s https://<your-deployment>/api/health | jq
curl -s -X POST https://<your-deployment>/api/solve \
  -H 'content-type: application/json' \
  -d '{"puzzle":"mini-01","rounds":2}' | jq '.score'
```

Body accepts either `{"puzzle": "<bundled id>"}` or an inline grid:
`{"grid": ["##OAF", ...], "across": {"1": "..."}, "down": {"1": "..."}}` —
`grid` carries the answers so the response includes a score; use `shape` instead
when you don't know them.

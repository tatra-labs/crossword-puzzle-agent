/*
 * studio.js -- the whole client for the crossword agent studio.
 *
 * Why this file is shaped the way it is
 * ------------------------------------
 * Two requirements drive every decision here, and both of them are about a
 * solve outliving the thing that started it.
 *
 * 1. A session must keep running while the user looks at something else. So
 *    the browser never owns a solve. Starting one is a POST that returns a
 *    session id; watching one is a *subscription* that can be closed and
 *    reopened at will. Switching away closes the EventSource and remembers the
 *    last sequence number seen; coming back reconnects with `?cursor=<seq>`.
 *    Because the server's trace log replays from any cursor, reattaching runs
 *    exactly the same code path as attaching -- there is no "resume" branch.
 *
 * 2. The trace is the product, not a progress bar. `step` events are the
 *    agent's own narration; `llm_call` events carry the prompt actually sent,
 *    the tool offered, the tool call that came back, tokens and latency. They
 *    share one ordered log so cause sits next to effect, and the model calls
 *    expand in place rather than living in a separate inspector.
 *
 * Untrusted text
 * --------------
 * Prompts, clues, tool inputs, model prose and error messages all reach this
 * page from somewhere else, and a prompt is precisely where an angle bracket
 * will turn up. So nothing dynamic is ever assigned through innerHTML: every
 * node is built with `el()` and its text set with textContent. There is no
 * escaping helper in this file on purpose -- a missing escape call is a bug
 * you cannot see, whereas a missing DOM node is one you can.
 *
 * Deployment reality
 * ------------------
 * Background sessions are in-process. `GET /api/health` reports
 * `durable_sessions: false` on a serverless deployment, where a solve cannot
 * survive its own response; there the Sessions list is hidden and Solve routes
 * through the legacy single-request `POST /api/solve/stream`, which is
 * rendered into the same trace panel as a one-off pseudo-session.
 */

// --------------------------------------------------------------------------
// Constants
// --------------------------------------------------------------------------

/** Session states in which no further trace events will ever arrive. */
const TERMINAL = new Set(["done", "stopped", "error"]);

/** Live states: a session in one of these is worth polling for. */
const LIVE = new Set(["queued", "running", "stopping"]);

/** The trace event types this UI knows how to render. */
const EVENT_TYPES = ["status", "step", "llm_call", "result", "error"];

/** Agent phases, in the order the solver visits them. Used for chip colours. */
const PHASES = new Set([
  "ingest", "propose", "fuse", "propagate",
  "commit", "critique", "repair", "verify", "done",
]);

/**
 * Chip colour per call kind. The bulk pass and the re-ask pass are different
 * events with different prompts, so they get different colours rather than
 * both borrowing the "propose" phase's.
 */
const CALL_CHIP = { batch: "propose", hard: "repair", cache: "cache" };

/**
 * Output price per million tokens, mirroring PRICING in src/xword/config.py.
 * Used only to scale the rough cost estimate shown before a solve: the numbers
 * the UI reports *after* one are the agent's own accounting, not these.
 */
const OUTPUT_PRICE = {
  "claude-opus-5": 25.0,
  "claude-sonnet-5": 10.0,
  "claude-sonnet-4-6": 15.0,
  "claude-haiku-4-5": 5.0,
};
const FALLBACK_OUTPUT_PRICE = 25.0;

/** Model ids offered in the picker: the ones config.py knows how to price. */
const MODEL_CHOICES = [
  "claude-sonnet-5",
  "claude-opus-5",
  "claude-sonnet-4-6",
  "claude-haiku-4-5",
];

/**
 * Measured anchors for the pre-solve estimate: a 5x5 (10 entries) and a
 * standard 15x15 (78 entries), Sonnet 5, three rounds, cold cache. Everything
 * between them is linear interpolation, which is wrong in detail and honest
 * about being rough -- the alternative is showing nothing, and then nobody
 * knows a click costs money.
 */
const ANCHOR_SMALL = { entries: 10, usd: 0.007, secs: 8 };
const ANCHOR_LARGE = { entries: 78, usd: 0.65, secs: 216 };

/** Client-side cap on retained events per session, matching the log's intent. */
const MAX_KEEP = 4000;

/** The pseudo-session id used for a legacy single-request solve. */
const LEGACY_SID = "single-request";

const STORE_KEY = "xword.studio.puzzle";

// --------------------------------------------------------------------------
// Tiny DOM helpers
// --------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function add(parent, child) {
  parent.appendChild(child);
  return child;
}

function clear(node) {
  node.replaceChildren();
  return node;
}

const cellKey = (r, c) => r + "," + c;

// --------------------------------------------------------------------------
// State
// --------------------------------------------------------------------------

const state = {
  health: null,
  /** False until /api/health says otherwise: the fallback path always works. */
  durable: false,
  /** Concurrency cap, so the sidebar can say how many slots are in use. */
  maxConcurrent: 0,
  puzzles: [],
  puzzleError: "",
  /** puzzle id -> normalised detail from GET /api/puzzles/{pid}. */
  details: new Map(),
  detailErrors: new Map(),
  selected: null,
  /** session id -> SessionInfo dict, plus a local `_seenAt` stamp. */
  sessions: new Map(),
  /** session ids, newest first, as the API orders them. */
  order: [],
  /** session id -> local subscription record (see track()). */
  logs: new Map(),
  open: null,
  view: "empty",
  filter: "all",
  auto: true,
  heartbeat: null,
  ticks: 0,
  legacyRunning: false,
  lastCard: null,
};

/** The per-session client record: what we have seen and how we are watching. */
function track(sid) {
  let rec = state.logs.get(sid);
  if (!rec) {
    rec = {
      events: [],
      cursor: 0,
      dropped: 0,
      closed: false,
      result: null,
      error: "",
      prov: new Map(),
      es: null,
      retry: 0,
      timer: null,
      poll: null,
    };
    state.logs.set(sid, rec);
  }
  return rec;
}

const infoOf = (sid) => state.sessions.get(sid) || null;

const anyLive = () => {
  for (const info of state.sessions.values()) if (LIVE.has(info.state)) return true;
  return false;
};

// --------------------------------------------------------------------------
// Formatting
// --------------------------------------------------------------------------

function fmtInt(n) {
  return Number(n || 0).toLocaleString();
}

function fmtTokens(n) {
  const v = Number(n || 0);
  if (v >= 10000) return (v / 1000).toFixed(1) + "k";
  return String(v);
}

function fmtElapsed(secs) {
  const s = Math.max(0, Number(secs) || 0);
  if (s < 60) return s.toFixed(s < 10 ? 1 : 0) + "s";
  const m = Math.floor(s / 60);
  return m + "m " + String(Math.floor(s % 60)).padStart(2, "0") + "s";
}

function fmtDuration(secs) {
  const s = Number(secs) || 0;
  return s < 1 ? Math.round(s * 1000) + "ms" : s.toFixed(2) + "s";
}

function fmtUsd(usd) {
  const v = Number(usd) || 0;
  if (v > 0 && v < 0.01) return "$" + v.toFixed(3);
  return "$" + v.toFixed(2);
}

function fmtClock(at) {
  const ms = Number(at) > 1e6 ? Number(at) * 1000 : Date.now();
  try {
    return new Date(ms).toLocaleTimeString([], { hour12: false });
  } catch {
    return "";
  }
}

/**
 * Elapsed time for a session, ticking between polls.
 *
 * The server's `elapsed_s` is a snapshot, and a clock that only moves every
 * two seconds looks broken. Rather than trusting the browser's clock against
 * the server's `started_at` -- which are not the same clock -- this adds the
 * time since *we received* the snapshot.
 */
function elapsedOf(info) {
  const base = Number(info.elapsed_s) || 0;
  if (TERMINAL.has(info.state) || !info._seenAt) return base;
  return base + Math.max(0, (performance.now() - info._seenAt) / 1000);
}

/** A very rough pre-solve estimate. See ANCHOR_SMALL / ANCHOR_LARGE. */
function estimate(entries, rounds, model) {
  const span = ANCHOR_LARGE.entries - ANCHOR_SMALL.entries;
  const t = ((Number(entries) || ANCHOR_SMALL.entries) - ANCHOR_SMALL.entries) / span;
  const scale = (Number(rounds) || 3) / 3;
  const price = OUTPUT_PRICE[model] || FALLBACK_OUTPUT_PRICE;
  const ratio = price / OUTPUT_PRICE["claude-sonnet-5"];
  return {
    usd: Math.max(0.001, ANCHOR_SMALL.usd + t * (ANCHOR_LARGE.usd - ANCHOR_SMALL.usd))
      * scale * ratio,
    secs: Math.max(5, ANCHOR_SMALL.secs + t * (ANCHOR_LARGE.secs - ANCHOR_SMALL.secs)) * scale,
  };
}

function fmtRoughTime(secs) {
  return secs < 90 ? "~" + Math.round(secs) + "s" : "~" + Math.round(secs / 60) + " min";
}

/** Read an error message the server wrote, in preference to inventing one. */
async function serverMessage(res) {
  try {
    const body = await res.json();
    const detail = body.detail ?? body.message ?? body.error;
    if (typeof detail === "string" && detail.trim()) return detail;
    if (detail) return JSON.stringify(detail);
  } catch {
    /* not JSON, or an empty body: fall through to the status line */
  }
  return "HTTP " + res.status + " " + (res.statusText || "");
}

async function getJson(url) {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(await serverMessage(res));
  return res.json();
}

// --------------------------------------------------------------------------
// Puzzle detail
//
// Cells per entry come straight from the API: each clue carries its start
// square, direction and length, which is everything a highlight or a
// provisional letter needs. Re-deriving the numbering in JavaScript would be a
// second implementation of xword.core.grid.number_grid, free to drift from it.
// --------------------------------------------------------------------------

function normaliseDetail(raw) {
  const shape = Array.isArray(raw.shape) ? raw.shape : [];
  const numbers = new Map(Object.entries(raw.numbers || {}));
  const slots = new Map();
  const order = { across: [], down: [] };
  const cellSlots = new Map();

  for (const direction of ["across", "down"]) {
    for (const entry of raw[direction] || []) {
      const cells = [];
      for (let i = 0; i < (entry.length || 0); i++) {
        cells.push(direction === "across"
          ? [entry.row, entry.col + i]
          : [entry.row + i, entry.col]);
      }
      const slot = {
        id: entry.id || String(entry.number) + (direction === "across" ? "A" : "D"),
        number: entry.number,
        direction,
        clue: entry.clue || "",
        length: entry.length || cells.length,
        cells,
      };
      slots.set(slot.id, slot);
      order[direction].push(slot.id);
      for (const [r, c] of cells) {
        const k = cellKey(r, c);
        if (!cellSlots.has(k)) cellSlots.set(k, []);
        cellSlots.get(k).push(slot.id);
      }
    }
  }

  return {
    id: raw.id,
    title: raw.title || "",
    difficulty: raw.difficulty || "",
    size: raw.size || (raw.height + "x" + raw.width),
    width: raw.width || (shape[0] ? shape[0].length : 0),
    height: raw.height || shape.length,
    entries: raw.entries ?? slots.size,
    openCells: raw.open_cells ?? 0,
    fitsHere: raw.fits_here !== false,
    slow: !!raw.slow,
    hasSolution: !!raw.has_solution,
    shape,
    numbers,
    slots,
    order,
    cellSlots,
  };
}

async function ensureDetail(pid) {
  if (!pid) return null;
  if (state.details.has(pid)) return state.details.get(pid);
  try {
    const detail = normaliseDetail(await getJson("/api/puzzles/" + encodeURIComponent(pid)));
    state.details.set(pid, detail);
    state.detailErrors.delete(pid);
    return detail;
  } catch (err) {
    state.detailErrors.set(pid, err.message || String(err));
    return null;
  }
}

// --------------------------------------------------------------------------
// Grid
// --------------------------------------------------------------------------

/**
 * Confidence ramp, carried over from the previous page unchanged.
 *
 * The cell background is tinted rather than the letter recoloured, so a
 * low-confidence square reads as "unsettled" without losing contrast in
 * either theme.
 */
function ramp(p) {
  if (p == null) return "";
  if (p >= 0.95) return "";
  if (p >= 0.8) return "color-mix(in srgb, var(--warn) 12%, transparent)";
  if (p >= 0.5) return "color-mix(in srgb, var(--warn) 26%, transparent)";
  return "color-mix(in srgb, var(--bad) 26%, transparent)";
}

/** Square size that keeps a 15x15 inside the centre column. */
function cellSize(width) {
  if (width <= 7) return 42;
  if (width <= 9) return 36;
  if (width <= 11) return 32;
  return 26;
}

/**
 * Draw a grid into `host`.
 *
 * `opts.letters` maps a cell key to `{ ch, provisional }`; `opts.conf` maps one
 * to a probability; `opts.gold` is the answer grid once a scored solve has
 * finished. `opts.onHover` is called with an array of slot ids (empty on exit).
 */
function drawGrid(host, detail, opts = {}) {
  clear(host);
  if (!detail || !detail.shape.length) {
    add(host, el("p", "legend", "The grid for this puzzle is not available."));
    return new Map();
  }

  const letters = opts.letters || new Map();
  const conf = opts.conf || {};
  const gold = opts.gold || null;
  const tds = new Map();

  const table = el("table", "grid");
  host.style.setProperty("--cell", cellSize(detail.width) + "px");

  detail.shape.forEach((row, r) => {
    const tr = el("tr");
    for (let c = 0; c < detail.width; c++) {
      const td = el("td");
      const ch = row[c];
      if (ch === "#" || ch === undefined) {
        td.className = "block";
        tr.appendChild(td);
        continue;
      }
      const k = cellKey(r, c);
      tds.set(k, td);

      const num = detail.numbers.get(k);
      if (num) add(td, el("span", "num", num));

      const hit = letters.get(k);
      const letter = hit && hit.ch && hit.ch !== "?" ? hit.ch : "";
      if (letter) {
        add(td, document.createTextNode(letter));
        if (hit.provisional) td.classList.add("prov");
      }

      const p = conf[k];
      const tint = ramp(p);
      if (tint) td.style.background = tint;

      const title = [];
      if (p != null) title.push("confidence " + (p * 100).toFixed(1) + "%");
      if (gold && gold[r] && gold[r][c] && gold[r][c] !== "#") {
        const want = gold[r][c];
        if (letter && letter !== want) {
          td.classList.add("wrong");
          td.style.background = "color-mix(in srgb, var(--bad) 30%, transparent)";
          title.unshift("expected " + want);
        } else if (letter) {
          td.style.background = "color-mix(in srgb, var(--good) 16%, transparent)";
        }
      }
      if (hit && hit.provisional) title.push("provisional: the model's first candidate");
      if (title.length) td.title = title.join(" · ");

      if (opts.onHover) {
        const ids = detail.cellSlots.get(k) || [];
        td.addEventListener("mouseenter", () => opts.onHover(ids));
        td.addEventListener("mouseleave", () => opts.onHover([]));
      }
      tr.appendChild(td);
    }
    table.appendChild(tr);
  });

  add(host, table);
  return tds;
}

// --------------------------------------------------------------------------
// Health
// --------------------------------------------------------------------------

function setHealthDot(kind, why) {
  const dot = $("healthDot");
  dot.className = "dot " + kind;
  dot.title = why;
  dot.setAttribute("aria-label", "Deployment status: " + why);
}

async function loadHealth() {
  try {
    state.health = await getJson("/api/health");
  } catch (err) {
    state.durable = false;
    setHealthDot("bad", "Could not reach /api/health: " + (err.message || err));
    $("healthLine").textContent = "The API did not answer. Nothing can be solved from here.";
    return;
  }

  const h = state.health;
  // Absent means false on purpose: assuming durability would point Solve at an
  // endpoint that answers 501, while assuming the fallback always works.
  state.durable = h.durable_sessions === true;
  state.maxConcurrent = Number(h.max_concurrent_sessions) || 0;

  if (h.api_key_configured) {
    setHealthDot("ok", "API reachable, ANTHROPIC_API_KEY configured, model " + h.model + ".");
  } else {
    setHealthDot(
      "warn",
      "API reachable but ANTHROPIC_API_KEY is not set on this deployment, so no "
      + "solve can start. Set it and redeploy.",
    );
  }

  $("healthLine").textContent = h.api_key_configured
    ? "Model " + h.model
      + (state.durable ? " · background sessions on" : " · one solve per request")
    : "ANTHROPIC_API_KEY is not set, so solving is disabled.";

  const bits = [
    "Function limit " + h.function_max_seconds + "s, solve budget " + h.solve_budget_seconds + "s.",
    "Lexicon " + fmtInt(h.lexicon_entries)
      + (h.lexicon_is_fallback ? " (built-in fallback)" : "") + ".",
  ];
  if (state.durable && h.max_concurrent_sessions) {
    bits.push("At most " + h.max_concurrent_sessions + " solves at once.");
  }
  if (h.access_token_required) bits.push("Starting a solve needs an access token.");
  $("deployLine").textContent = bits.join(" ");

  const note = $("ephemeralNote");
  if (!state.durable) {
    note.hidden = false;
    note.textContent =
      "This deployment runs one solve per request: a serverless function is frozen "
      + "once it responds, so a session cannot outlive it. Solving still works, and "
      + "the trace still streams, but there is no session list and no stopping. Run "
      + "the app locally for that.";
  } else {
    note.hidden = true;
  }
}

// --------------------------------------------------------------------------
// Sidebar: puzzles
// --------------------------------------------------------------------------

const puzzleNodes = new Map();

function livePuzzles() {
  const live = new Map();
  for (const info of state.sessions.values()) {
    if (LIVE.has(info.state)) live.set(info.puzzle_id, info.state);
  }
  return live;
}

function renderPuzzles() {
  const list = $("puzzleList");
  const live = livePuzzles();

  if (!state.puzzles.length) {
    clear(list);
    puzzleNodes.clear();
    // "None returned" and "the request failed" are different problems, and
    // saying the first when the second happened sends the reader looking in
    // the wrong place.
    add(list, el("p", "side-empty", state.puzzleError
      ? "Could not load the puzzle list: " + state.puzzleError
      : "No bundled puzzles were returned."));
    return;
  }

  // Built once and then updated in place: rebuilding this list on every poll
  // would steal focus from whichever row the keyboard is on.
  if (!puzzleNodes.size) {
    clear(list);
    for (const p of state.puzzles) {
      const row = el("div", "row");
      const main = el("button", "row-main");
      main.type = "button";
      main.setAttribute("role", "listitem");

      const top = add(main, el("div", "row-top"));
      add(top, el("span", "row-id", p.id));
      add(top, el("span", "badge", p.size));
      const dot = add(top, el("span", "livedot"));
      dot.hidden = true;

      const sub = add(main, el("span", "row-sub"));

      if (!p.fits_here) {
        main.disabled = true;
        main.classList.add("off");
        main.title =
          "Too big for this deployment: it would not finish inside the function's "
          + "time limit, so the API refuses it rather than charging for a solve it "
          + "cannot complete.";
      } else {
        main.addEventListener("click", () => selectPuzzle(p.id));
      }

      add(row, main);
      add(list, row);
      puzzleNodes.set(p.id, { row, main, sub, dot });
    }
  }

  for (const p of state.puzzles) {
    const node = puzzleNodes.get(p.id);
    if (!node) continue;
    const marks = [p.entries + " entries"];
    if (!p.fits_here) marks.push("too big here");
    else if (p.slow) marks.push("slower, ~1 min");
    node.sub.textContent = marks.join(" · ");
    const liveState = live.get(p.id);
    node.dot.hidden = !liveState;
    node.dot.className = "livedot" + (liveState === "stopping" ? " stopping" : "");
    if (liveState) node.dot.title = "a session for this puzzle is " + liveState;
    node.row.classList.toggle("on", state.selected === p.id && state.view !== "session");
  }
}

/** Arrow-key navigation across the puzzle rows. */
function wirePuzzleKeys() {
  $("puzzleList").addEventListener("keydown", (e) => {
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(e.key)) return;
    const rows = Array.from($("puzzleList").querySelectorAll("button.row-main:not([disabled])"));
    if (!rows.length) return;
    const here = rows.indexOf(document.activeElement);
    let next = 0;
    if (e.key === "ArrowDown") next = here < 0 ? 0 : Math.min(rows.length - 1, here + 1);
    else if (e.key === "ArrowUp") next = here < 0 ? rows.length - 1 : Math.max(0, here - 1);
    else if (e.key === "End") next = rows.length - 1;
    rows[next].focus();
    e.preventDefault();
  });
}

// --------------------------------------------------------------------------
// Sidebar: sessions
// --------------------------------------------------------------------------

const sessionNodes = new Map();

function buildSessionRow(sid) {
  const row = el("div", "row");
  const main = el("button", "row-main");
  main.type = "button";
  main.setAttribute("role", "listitem");
  main.addEventListener("click", () => openSession(sid));

  const top = add(main, el("div", "row-top"));
  const id = add(top, el("span", "row-id"));
  const pill = add(top, el("span", "pill"));
  const sub = add(main, el("span", "row-sub"));

  const act = el("button", "row-act");
  act.type = "button";

  add(row, main);
  add(row, act);
  return { row, main, id, pill, sub, act };
}

function updateSessionRow(node, info) {
  node.id.textContent = info.puzzle_id;
  node.pill.textContent = info.state;
  node.pill.className = "pill " + info.state;

  const bits = [fmtElapsed(elapsedOf(info))];
  if (LIVE.has(info.state)) {
    bits.push("round " + info.round + " · " + (info.step || info.state));
  } else if (info.state === "error") {
    bits.push(info.error || info.message || "failed");
  } else if (info.solved === true) bits.push("solved");
  else if (info.solved === false) bits.push("not solved");
  node.sub.textContent = bits.join(" · ");
  node.main.title = info.title
    ? info.title + " · " + info.size + " · " + (info.message || info.state)
    : info.size + " · " + (info.message || info.state);

  const running = LIVE.has(info.state);
  const act = node.act;
  act.hidden = info.id === LEGACY_SID;
  act.textContent = running ? "stop" : "dismiss";
  act.title = running
    ? "Ask this solve to stop at its next checkpoint"
    : "Forget this session and its trace";
  act.setAttribute("aria-label", (running ? "Stop session " : "Dismiss session ") + info.puzzle_id);
  act.onclick = running ? () => stopSession(info.id) : () => deleteSession(info.id);

  node.row.classList.toggle("on", state.open === info.id && state.view === "session");
}

function renderSessions() {
  const sec = $("sessionSec");
  if (!state.durable && !state.sessions.has(LEGACY_SID)) {
    sec.hidden = true;
    return;
  }
  sec.hidden = false;

  const list = $("sessionList");
  const ids = state.order.filter((id) => state.sessions.has(id));

  for (const [id, node] of Array.from(sessionNodes)) {
    if (!ids.includes(id)) {
      node.row.remove();
      sessionNodes.delete(id);
    }
  }

  ids.forEach((sid, i) => {
    let node = sessionNodes.get(sid);
    if (!node) {
      node = buildSessionRow(sid);
      sessionNodes.set(sid, node);
    }
    updateSessionRow(node, state.sessions.get(sid));
    if (list.children[i] !== node.row) list.insertBefore(node.row, list.children[i] || null);
  });

  const live = ids.filter((id) => LIVE.has(state.sessions.get(id).state)).length;
  let count = "";
  if (live) count = live + " running" + (state.maxConcurrent ? " of " + state.maxConcurrent : "");
  else if (ids.length) count = String(ids.length);
  $("sessionCount").textContent = count ? "(" + count + ")" : "";
  $("sessionEmpty").hidden = ids.length > 0;
}

function upsertSession(info) {
  if (!info || !info.id) return null;
  const prev = state.sessions.get(info.id);
  const merged = Object.assign({}, prev || {}, info);
  merged._seenAt = performance.now();
  state.sessions.set(info.id, merged);
  if (!state.order.includes(info.id)) state.order.unshift(info.id);
  return merged;
}

function patchSession(sid, fields) {
  const info = state.sessions.get(sid);
  if (!info) return;
  Object.assign(info, fields);
  info._seenAt = performance.now();
}

// --------------------------------------------------------------------------
// Centre column
// --------------------------------------------------------------------------

function showView(which) {
  state.view = which;
  $("emptyView").hidden = which !== "empty";
  $("inspectView").hidden = which !== "inspect";
  $("sessionView").hidden = which !== "session";
  renderPuzzles();
  renderSessions();
}

function selectPuzzle(pid) {
  state.selected = pid;
  try {
    localStorage.setItem(STORE_KEY, pid);
  } catch {
    /* private mode, or storage disabled: the selection just is not remembered */
  }
  state.open = null;
  detachAll();
  renderTraceHeader();
  rebuildTrace();
  renderInspect();
}

async function renderInspect() {
  const pid = state.selected;
  if (!pid) {
    showView("empty");
    return;
  }
  const row = state.puzzles.find((p) => p.id === pid) || { id: pid };
  showView("inspect");

  $("inspectTitle").textContent = row.title ? row.title : pid;
  const problem = $("inspectProblem");
  problem.hidden = true;

  const detail = await ensureDetail(pid);
  if (state.selected !== pid) return; // the user moved on while we fetched

  const meta = $("inspectMeta");
  clear(meta);
  const parts = [
    pid,
    (detail ? detail.size : row.size) || "",
    (detail ? detail.entries : row.entries) + " entries",
    (detail ? detail.openCells : row.open_cells) + " open squares",
  ];
  const difficulty = detail ? detail.difficulty : row.difficulty;
  if (difficulty) parts.push(difficulty);
  parts.push(
    detail
      ? (detail.hasSolution ? "known solution, so it can be scored" : "no reference solution")
      : "",
  );
  parts.filter(Boolean).forEach((text, i) => {
    if (i) add(meta, el("span", "sep", "·"));
    add(meta, document.createTextNode(text));
  });

  $("solveBtn").disabled = !canSolve();
  updateCostNote();

  if (!detail) {
    problem.hidden = false;
    problem.textContent =
      "Could not load this puzzle's grid and clues: "
      + (state.detailErrors.get(pid) || "unknown error");
    clear($("inspectGrid"));
    clear($("inspectClues"));
    $("inspectGridNote").textContent = "";
    return;
  }

  // The two highlight directions share one map each way: clue -> cells needs
  // the table, cell -> clues needs the clue rows, so both are declared before
  // either is drawn.
  const clueNodes = new Map();
  const tds = drawGrid($("inspectGrid"), detail, {
    onHover: (ids) => highlight(detail, tds, clueNodes, ids),
  });
  $("inspectGridNote").textContent =
    detail.width + " columns × " + detail.height + " rows, "
    + detail.openCells + " open squares. Hover a clue to see where it goes.";

  renderClues(detail, tds, clueNodes);
}

/** Shared highlight: cells for a set of slot ids, plus their clue rows. */
function highlight(detail, tds, clueNodes, ids) {
  for (const td of tds.values()) td.classList.remove("hl");
  for (const node of clueNodes.values()) node.classList.remove("hl");
  for (const id of ids) {
    const slot = detail.slots.get(id);
    if (!slot) continue;
    for (const [r, c] of slot.cells) {
      const td = tds.get(cellKey(r, c));
      if (td) td.classList.add("hl");
    }
    const node = clueNodes.get(id);
    if (node) node.classList.add("hl");
  }
}

function renderClues(detail, tds, nodes) {
  const host = clear($("inspectClues"));
  nodes.clear();

  for (const direction of ["across", "down"]) {
    const col = add(host, el("div", "cluecol"));
    add(col, el("h3", "sub-h", direction === "across" ? "Across" : "Down"));
    for (const id of detail.order[direction]) {
      const slot = detail.slots.get(id);
      const line = el("div", "clue");
      line.tabIndex = 0;
      add(line, el("span", "n", slot.number + (direction === "across" ? "A" : "D")));
      const text = add(line, el("span", "t"));
      add(text, document.createTextNode(slot.clue || "(no clue)"));
      add(text, el("span", "len", " (" + slot.length + ")"));
      const show = () => highlight(detail, tds, nodes, [id]);
      const hide = () => highlight(detail, tds, nodes, []);
      line.addEventListener("mouseenter", show);
      line.addEventListener("mouseleave", hide);
      line.addEventListener("focus", show);
      line.addEventListener("blur", hide);
      add(col, line);
      nodes.set(id, line);
    }
  }
}

function updateCostNote() {
  const pid = state.selected;
  const row = state.puzzles.find((p) => p.id === pid);
  if (!row) return;
  const rounds = Number($("optRounds").value) || 3;
  const model = $("optModel").value;
  const guess = estimate(row.entries, rounds, model);
  $("costNote").textContent =
    "Roughly " + fmtUsd(guess.usd) + " and " + fmtRoughTime(guess.secs) + " of real Anthropic "
    + "credit and wall time for this size at " + rounds + " round" + (rounds === 1 ? "" : "s")
    + ". Rough: measured anchors are a 5x5 at $0.007 / 8s and a 15x15 at $0.65 / 216s on "
    + "claude-sonnet-5. Clues already answered come from the local cache, so a repeat "
    + "is much cheaper.";
}

// -- session view ---------------------------------------------------------- //

/**
 * The header, pill and stats. Separate from the grid and the entry table
 * because a `step` event arrives several times a round and changes only these
 * -- redrawing 225 squares to advance a round counter is wasted work, and it
 * would throw away the grid's hover state while it was at it.
 */
function renderSessionMeta() {
  const sid = state.open;
  const info = infoOf(sid);
  if (!info) return;

  $("sessionTitle").textContent = (info.title || info.puzzle_id) + " · " + info.puzzle_id;
  $("sessionSub").textContent =
    [info.size, info.entries + " entries", info.open_cells + " open squares",
     "model " + info.model, "max " + info.max_rounds + " rounds"].join(" · ");

  const pill = $("sessionPill");
  pill.textContent = info.state;
  pill.className = "pill " + info.state;
  pill.title = info.message || info.state;

  const stop = $("sessionStopBtn");
  stop.hidden = !LIVE.has(info.state) || sid === LEGACY_SID;
  stop.disabled = info.state === "stopping";
  stop.onclick = () => stopSession(sid);

  renderSessionStats(info);

  const rec = track(sid);
  const problem = $("sessionProblem");
  const failure = rec.error || (info.state === "error" ? info.error : "");
  problem.hidden = !failure;
  if (failure) problem.textContent = failure;
}

function renderSessionView() {
  const sid = state.open;
  const info = infoOf(sid);
  if (!info) {
    showView(state.selected ? "inspect" : "empty");
    return;
  }
  showView("session");
  renderSessionMeta();
  renderSessionGrid();
  renderSessionEntries();
}

function renderSessionStats(info) {
  const host = clear($("sessionStats"));
  const tiles = [
    [info.state === "running" ? "round " + info.round : info.state, "state"],
    [fmtElapsed(elapsedOf(info)), "elapsed"],
    [fmtInt(info.llm_calls), "model calls"],
    [fmtInt(info.input_tokens) + " / " + fmtInt(info.output_tokens), "tokens in / out"],
    [fmtUsd(info.cost_usd), "est. cost"],
  ];
  if (info.cells_total) {
    tiles.push([info.cells_correct + "/" + info.cells_total, "squares right"]);
  }
  for (const [value, label] of tiles) {
    const tile = add(host, el("div", "stat"));
    add(tile, el("b", null, value));
    add(tile, el("span", null, label));
  }
}

async function renderSessionGrid() {
  const sid = state.open;
  const info = infoOf(sid);
  if (!info) return;
  const rec = track(sid);
  const detail = await ensureDetail(info.puzzle_id);
  if (state.open !== sid) return;

  const host = $("sessionGrid");
  const note = $("sessionGridNote");
  const result = rec.result;

  if (result && Array.isArray(result.fill)) {
    // The finished (or last committed) grid, straight from the solve result.
    const letters = new Map();
    result.fill.forEach((row, r) => {
      for (let c = 0; c < row.length; c++) {
        if (row[c] !== "#" && row[c] !== "?") letters.set(cellKey(r, c), { ch: row[c] });
      }
    });
    const shaped = detail || {
      shape: result.puzzle.rows,
      width: result.puzzle.width,
      height: result.puzzle.height,
      numbers: new Map(Object.entries(result.puzzle.numbers || {})),
      cellSlots: new Map(),
      openCells: 0,
    };
    drawGrid(host, shaped, { letters, conf: result.confidence || {}, gold: result.gold || null });
    clear(note);
    if (result.score) {
      const s = result.score;
      add(note, el("b", null, s.solved ? "Solved" : "Not solved"));
      add(note, document.createTextNode(
        " · " + s.cells_correct + "/" + s.cells_total + " squares · "
        + s.words_correct + "/" + s.words_total + " entries · "
        + (s.cell_accuracy * 100).toFixed(1) + "% cell accuracy",
      ));
      add(note, el("span", "swatch", "")).style.background =
        "color-mix(in srgb, var(--good) 16%, transparent)";
      add(note, document.createTextNode("right"));
      add(note, el("span", "swatch", "")).style.background =
        "color-mix(in srgb, var(--bad) 30%, transparent)";
      add(note, document.createTextNode("wrong"));
    } else {
      note.textContent = "This puzzle has no reference solution, so the grid is shaded by the "
        + "agent's own confidence rather than marked right or wrong.";
    }
    return;
  }

  if (!detail) {
    clear(host);
    add(host, el("p", "legend", "The grid is not available for this session."));
    note.textContent = "";
    return;
  }

  const letters = new Map();
  for (const [k, ch] of rec.prov) letters.set(k, { ch, provisional: true });
  drawGrid(host, detail, { letters });
  note.textContent = rec.prov.size
    ? letters.size + " squares carry the model's first candidate. They are provisional: "
      + "propagation, search and the critique loop can all still overwrite them."
    : "Waiting for the first batch of answers.";
}

function renderSessionEntries() {
  const sid = state.open;
  const rec = track(sid);
  const table = clear($("sessionEntries"));
  const note = $("entriesNote");
  const result = rec.result;

  if (!result || !Array.isArray(result.entries)) {
    note.textContent = "The entry table arrives with the finished grid. Until then the trace on "
      + "the right is the live view.";
    return;
  }

  const head = add(table, el("thead"));
  const hr = add(head, el("tr"));
  for (const label of ["", "Clue", "Answer", "Conf", "Source"]) {
    add(hr, el("th", null, label));
  }

  const body = add(table, el("tbody"));
  let wrong = 0;
  for (const entry of result.entries) {
    const tr = add(body, el("tr"));
    const first = add(tr, el("td"));
    if (entry.gold != null) {
      const right = entry.answer === entry.gold;
      if (!right) wrong += 1;
      add(first, el("span", right ? "ok" : "no", right ? "✓" : "✗"));
      add(first, document.createTextNode(" "));
    }
    add(first, document.createTextNode(entry.id));
    add(tr, el("td", null, entry.clue || ""));

    const ans = add(tr, el("td", "a"));
    if (entry.gold != null && entry.answer !== entry.gold) {
      const struck = add(ans, el("s", null, entry.answer || "—"));
      struck.title = "the agent's answer";
      add(ans, document.createTextNode(" "));
      add(ans, el("span", "ok", entry.gold));
    } else {
      add(ans, document.createTextNode(entry.answer || "—"));
    }

    add(tr, el("td", "c", Math.round((entry.confidence || 0) * 100) + "%"));
    add(tr, el("td", null, entry.source || ""));
  }
  note.textContent = result.entries.length + " entries"
    + (wrong ? ", " + wrong + " wrong (the gold answer is shown after the strike-through)" : "")
    + ".";
}

// --------------------------------------------------------------------------
// Trace panel
// --------------------------------------------------------------------------

let progScrollUntil = 0;

function renderTraceHeader() {
  const sid = state.open;
  const info = sid ? infoOf(sid) : null;
  $("traceWho").textContent = info
    ? info.puzzle_id + " · " + (sid === LEGACY_SID ? "single request" : sid.slice(0, 8))
    : "no session open";
  syncJump();
  updateTraceCount();
}

function syncJump() {
  $("jumpBtn").hidden = state.auto || !state.open;
}

function updateTraceCount() {
  const rec = state.open ? track(state.open) : null;
  const total = rec ? rec.events.length : 0;
  const shown = rec ? rec.events.filter(passesFilter).length : 0;
  $("traceCount").textContent = state.filter === "all"
    ? total + (total === 1 ? " event" : " events")
    : shown + " of " + total + " events";
}

function passesFilter(ev) {
  if (state.filter === "all") return true;
  return ev.type === state.filter;
}

function setFilter(next) {
  state.filter = next;
  for (const btn of $("traceFilter").querySelectorAll("button")) {
    const on = btn.dataset.filter === next;
    btn.classList.toggle("on", on);
    btn.setAttribute("aria-pressed", on ? "true" : "false");
  }
  rebuildTrace();
}

function setAuto(on) {
  state.auto = on;
  const btn = $("autoBtn");
  btn.setAttribute("aria-pressed", on ? "true" : "false");
  btn.textContent = on ? "Auto-scroll" : "Auto-scroll off";
  syncJump();
  if (on) stickToBottom();
}

/** Follow the tail, but only while the user has not taken the wheel. */
function stickToBottom() {
  if (!state.auto) return;
  const body = $("traceBody");
  progScrollUntil = performance.now() + 150;
  body.scrollTop = body.scrollHeight;
}

function rebuildTrace() {
  const body = clear($("traceBody"));
  // Every card node is about to be discarded, so the Escape target goes too.
  state.lastCard = null;
  const rec = state.open ? track(state.open) : null;
  if (!rec) {
    add(body, el("p", "side-empty", "Open a session to see its prompts, tool calls and steps."));
    updateTraceCount();
    return;
  }
  if (rec.dropped) {
    add(body, el("p", "sys", rec.dropped + " earlier events were dropped from this log."));
  }
  let shown = 0;
  for (const ev of rec.events) {
    if (!passesFilter(ev)) continue;
    add(body, traceRow(ev));
    shown += 1;
  }
  if (!shown) {
    add(body, el("p", "side-empty", rec.events.length
      ? "No events of this kind yet."
      : "Waiting for the first event."));
  }
  updateTraceCount();
  stickToBottom();
}

function appendTraceEvent(ev) {
  if (!passesFilter(ev)) {
    updateTraceCount();
    return;
  }
  const body = $("traceBody");
  const placeholder = body.querySelector(".side-empty");
  if (placeholder) placeholder.remove();
  add(body, traceRow(ev));
  updateTraceCount();
  stickToBottom();
}

function traceRow(ev) {
  if (ev.type === "step") return stepRow(ev);
  if (ev.type === "llm_call") return callCard(ev);
  return systemRow(ev);
}

function stepRow(ev) {
  const p = ev.payload || {};
  const row = el("div", "ev");
  const kind = String(p.kind || "step");
  add(row, el("span", "chip " + (PHASES.has(kind) ? kind : "plain"), kind));
  add(row, el("span", "rnd", "r" + (p.round ?? 0)));
  add(row, el("span", "msg", p.message || ""));
  add(row, el("span", "at", fmtClock(ev.at)));

  const data = p.data && typeof p.data === "object" ? p.data : null;
  if (data) {
    const pairs = Object.entries(data).map(([k, v]) => k + "=" + v);
    if (pairs.length) row.title = pairs.join("  ");
  }
  return row;
}

function systemRow(ev) {
  const p = ev.payload || {};
  const row = el("div", "sys");
  let text = "";
  if (ev.type === "status") {
    text = String(p.state || "") + (p.message ? " — " + p.message : "");
  } else if (ev.type === "error") {
    row.classList.add("bad");
    text = p.message || "the solve failed";
  } else if (ev.type === "result") {
    const stats = p.stats || {};
    const score = p.score;
    text = "result — " + (score ? (score.solved ? "solved, " : "not solved, ") : "")
      + fmtInt(stats.llm_calls) + " model calls, " + fmtUsd(stats.cost_usd)
      + ", " + fmtElapsed(stats.wall_seconds);
  } else {
    text = ev.type;
  }
  add(row, el("span", "at", fmtClock(ev.at)));
  add(row, el("span", "msg", text));
  return row;
}

/**
 * One model call, collapsed to a line and expandable to the whole exchange.
 *
 * A cache hit is deliberately not rendered as a call with empty fields: no
 * request was made, so the card says so and shows which entries were served
 * instead of an empty prompt block.
 */
function callCard(ev) {
  const rec = ev.payload || {};
  const card = el("article", "call");
  const head = el("button", "call-head");
  head.type = "button";
  head.setAttribute("aria-expanded", "false");

  const top = add(head, el("div", "call-top"));
  const cached = !!rec.cached;
  const kind = cached ? "cache" : String(rec.kind || "call");
  add(top, el("span", "chip " + (CALL_CHIP[kind] || "plain"), kind));
  add(top, el("span", "call-label", rec.label || (rec.kind || "model call")));
  if (rec.round != null) add(top, el("span", "rnd", "r" + rec.round));
  if (rec.error) add(top, el("span", "chip bad", "error"));
  add(top, el("span", "caret", "▸"));

  const meta = [];
  if (!cached) meta.push(rec.model || "");
  meta.push(cached ? "served from cache" : fmtDuration(rec.duration_s));
  if (!cached) {
    meta.push(fmtTokens(rec.input_tokens) + " in / " + fmtTokens(rec.output_tokens) + " out");
    if (rec.cache_read_tokens) meta.push(fmtTokens(rec.cache_read_tokens) + " cached in");
  }
  if (rec.attempts > 1) meta.push(rec.attempts + " attempts");
  if ((rec.clue_ids || []).length) meta.push(rec.clue_ids.length + " entries");
  add(head, el("span", "call-meta", meta.filter(Boolean).join(" · ")));

  const bodyBox = el("div", "call-body");
  bodyBox.hidden = true;
  let built = false;

  head.addEventListener("click", () => {
    const open = card.classList.toggle("open");
    if (open && !built) {
      // Built on first expand: a 15x15 solve produces tens of these, and most
      // of them are never opened. Nobody needs a pretty-printed tool_input
      // they did not ask for.
      buildCallBody(bodyBox, rec, cached);
      built = true;
    }
    bodyBox.hidden = !open;
    head.setAttribute("aria-expanded", open ? "true" : "false");
    top.querySelector(".caret").textContent = open ? "▾" : "▸";
    if (open) {
      state.lastCard = card;
      // Opening a card means "I want to read this", which is incompatible with
      // the panel jumping to the tail on the next event.
      if (state.auto) setAuto(false);
    }
  });

  add(card, head);
  add(card, bodyBox);
  return card;
}

function collapseCard(card) {
  if (!card || !card.classList.contains("open")) return false;
  card.classList.remove("open");
  const head = card.querySelector(".call-head");
  const body = card.querySelector(".call-body");
  if (body) body.hidden = true;
  if (head) {
    head.setAttribute("aria-expanded", "false");
    const caret = head.querySelector(".caret");
    if (caret) caret.textContent = "▸";
    head.focus();
  }
  return true;
}

function buildCallBody(host, rec, cached) {
  clear(host);

  if (cached) {
    const sec = add(host, el("section", "sec"));
    add(sec, el("h4", null, "No API call"));
    add(sec, el("p", "call-meta",
      "These answers came from the local SQLite clue cache, keyed by clue, length "
      + "and pattern. Nothing was sent to the model and nothing was charged."));
    if ((rec.clue_ids || []).length) add(sec, slotList(rec.clue_ids));
    if (rec.tool_input && Object.keys(rec.tool_input).length) {
      add(host, jsonBlock("Cached tool input", rec.tool_input));
    }
    return;
  }

  // -- prompt -- //
  const prompt = add(host, el("section", "sec"));
  add(prompt, el("h4", null, "Prompt"));
  if (rec.system) {
    // The system prompt is long and identical across most calls, so it starts
    // folded: what differs between two calls is the user message.
    const fold = add(prompt, el("details", "fold"));
    add(fold, el("summary", null, "System prompt (" + fmtInt(rec.system.length) + " chars)"));
    add(fold, textBlock("system", rec.system));
  }
  if (rec.prompt) add(prompt, textBlock("user message", rec.prompt));
  if (!rec.system && !rec.prompt) {
    add(prompt, el("p", "call-meta", "This record carries no prompt text."));
  }

  // -- tool call -- //
  const tool = add(host, el("section", "sec"));
  add(tool, el("h4", null, "Tool call"));
  const kv = add(tool, el("dl", "kv"));
  const pair = (k, v) => {
    add(kv, el("dt", null, k));
    add(kv, el("dd", null, v));
  };
  pair("offered", (rec.tools || []).join(", ") || "none");
  pair("tool_choice", rec.tool_choice || "none");
  pair("called", rec.tool_name || "(no tool call)");
  if ((rec.clue_ids || []).length) {
    add(kv, el("dt", null, "asked about"));
    add(add(kv, el("dd")), slotList(rec.clue_ids));
  }
  if (rec.tool_input && Object.keys(rec.tool_input).length) {
    add(tool, jsonBlock("tool_input", rec.tool_input));
  } else {
    add(tool, el("p", "call-meta", "The model returned no tool input."));
  }

  // -- result -- //
  const result = add(host, el("section", "sec"));
  add(result, el("h4", null, "Result"));
  const rkv = add(result, el("dl", "kv"));
  add(rkv, el("dt", null, "stop_reason"));
  add(rkv, el("dd", null, rec.stop_reason || "—"));
  add(rkv, el("dt", null, "attempts"));
  add(rkv, el("dd", null, String(rec.attempts ?? 1)));
  if (rec.text) add(result, textBlock("assistant text", rec.text));
  if (rec.error) {
    const box = add(result, el("p", "note err"));
    box.textContent = rec.error;
  }
  if (rec.truncated) {
    add(result, el("p", "call-meta",
      "This record was clipped before storage, so the text above is the beginning "
      + "of what was sent or returned, not all of it."));
  }
}

function slotList(ids) {
  const wrap = el("div", "slots");
  for (const id of ids) add(wrap, el("span", "sid", id));
  return wrap;
}

function blockShell(label, text, cls) {
  const wrap = el("div", "blk");
  const head = add(wrap, el("div", "blk-head"));
  add(head, el("span", "grow", label));
  add(head, copyButton(text));
  const pre = add(wrap, el("pre", cls));
  pre.textContent = text;
  return wrap;
}

function textBlock(label, text) {
  return blockShell(label, String(text), "txt");
}

function jsonBlock(label, value) {
  let text;
  try {
    text = JSON.stringify(value, null, 2);
  } catch {
    text = String(value);
  }
  return blockShell(label, text, "txt json");
}

function copyButton(text) {
  const btn = el("button", "copy", "copy");
  btn.type = "button";
  btn.addEventListener("click", async (e) => {
    e.stopPropagation();
    const done = await copyText(text);
    btn.textContent = done ? "copied" : "copy failed";
    setTimeout(() => { btn.textContent = "copy"; }, 1400);
  });
  return btn;
}

async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall through to the legacy path below */
  }
  try {
    const box = document.createElement("textarea");
    box.value = text;
    box.setAttribute("readonly", "");
    box.style.position = "fixed";
    box.style.opacity = "0";
    document.body.appendChild(box);
    box.select();
    const ok = document.execCommand("copy");
    box.remove();
    return ok;
  } catch {
    return false;
  }
}

function wireTracePanel() {
  $("traceFilter").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-filter]");
    if (btn) setFilter(btn.dataset.filter);
  });

  $("autoBtn").addEventListener("click", () => setAuto(!state.auto));
  $("jumpBtn").addEventListener("click", () => setAuto(true));

  // Scrolling up is how a user says "stop moving this on me", so auto-scroll
  // switches itself off. Our own scrolls are ignored via a short time window:
  // a programmatic scrollTop fires this handler asynchronously, so a flag
  // cleared on the next frame can lose the race.
  $("traceBody").addEventListener("scroll", () => {
    if (performance.now() < progScrollUntil) return;
    const body = $("traceBody");
    const atBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 48;
    if (!atBottom && state.auto) setAuto(false);
  }, { passive: true });
}

// --------------------------------------------------------------------------
// Event ingestion
// --------------------------------------------------------------------------

function ingest(sid, ev) {
  const rec = track(sid);
  if (ev.seq != null) {
    // The cursor is the whole reason a reconnect is not a special case: an
    // event we have already seen is dropped here rather than deduplicated
    // anywhere else.
    if (ev.seq <= rec.cursor) return;
    rec.cursor = ev.seq;
  }
  rec.events.push(ev);
  if (rec.events.length > MAX_KEEP) rec.events.splice(0, rec.events.length - MAX_KEEP);

  const payload = ev.payload || {};
  switch (ev.type) {
    case "status":
      if (payload.state) {
        patchSession(sid, { state: payload.state, message: payload.message || "" });
        if (LIVE.has(payload.state)) startHeartbeat();
        else {
          detach(sid);
          pollSessions();
        }
      }
      break;
    case "step":
      patchSession(sid, {
        round: payload.round ?? 0,
        step: payload.kind || "",
        message: payload.message || "",
      });
      break;
    case "llm_call":
      absorbCall(sid, payload);
      break;
    case "result":
      rec.result = payload;
      break;
    case "error":
      rec.error = payload.message || "the solve failed";
      break;
    default:
      break;
  }

  if (state.open === sid) {
    appendTraceEvent(ev);
    if (ev.type === "llm_call") renderSessionGrid();
    else if (ev.type === "result" || ev.type === "error") renderSessionView();
    else renderSessionMeta();
  }
  if (ev.type === "status" || ev.type === "step") renderSessions();
}

/**
 * Provisional letters, taken from the tool call the model just made.
 *
 * `step` events carry only numbers and prose, so the model's answers are the
 * one place letters appear before the solve finishes. Painting the first
 * candidate of each entry is what makes the grid fill in live; it is drawn
 * dimmed because propagation and search have not weighed in yet.
 */
function absorbCall(sid, call) {
  const info = infoOf(sid);
  const detail = info ? state.details.get(info.puzzle_id) : null;
  if (!detail) return;
  const answers = call && call.tool_input ? call.tool_input.answers : null;
  if (!Array.isArray(answers)) return;

  const store = track(sid).prov;
  for (const item of answers) {
    if (!item || typeof item !== "object") continue;
    const slot = detail.slots.get(String(item.slot || ""));
    const best = Array.isArray(item.candidates) ? item.candidates[0] : null;
    if (!slot || !best || typeof best.answer !== "string") continue;
    const answer = best.answer.toUpperCase();
    const n = Math.min(slot.cells.length, answer.length);
    for (let i = 0; i < n; i++) {
      const [r, c] = slot.cells[i];
      store.set(cellKey(r, c), answer[i]);
    }
  }
}

// --------------------------------------------------------------------------
// Subscription
// --------------------------------------------------------------------------

function detach(sid) {
  const rec = state.logs.get(sid);
  if (!rec) return;
  if (rec.es) {
    rec.es.close();
    rec.es = null;
  }
  if (rec.timer) {
    clearTimeout(rec.timer);
    rec.timer = null;
  }
  if (rec.poll) {
    clearInterval(rec.poll);
    rec.poll = null;
  }
}

function detachAll() {
  for (const sid of state.logs.keys()) detach(sid);
}

function attach(sid) {
  detach(sid);
  const rec = track(sid);
  if (rec.closed) return;
  if (typeof EventSource === "undefined") {
    startEventPoll(sid);
    return;
  }

  const url = "/api/sessions/" + encodeURIComponent(sid)
    + "/stream?cursor=" + encodeURIComponent(rec.cursor);
  let es;
  try {
    es = new EventSource(url);
  } catch {
    startEventPoll(sid);
    return;
  }
  rec.es = es;

  const handle = (name) => (e) => {
    let data = null;
    try {
      data = JSON.parse(e.data);
    } catch {
      return; // a keepalive or a frame we do not understand
    }
    onFrame(sid, name, data);
  };
  for (const type of EVENT_TYPES) es.addEventListener(type, handle(type));
  es.addEventListener("closed", (e) => {
    let data = {};
    try {
      data = JSON.parse(e.data) || {};
    } catch {
      /* the payload is a nicety; the frame itself is the signal */
    }
    rec.closed = true;
    rec.dropped = data.dropped || rec.dropped;
    detach(sid);
    pollSessions();
  });
  es.onmessage = handle(null);

  es.onerror = () => {
    // EventSource reconnects on its own, but always to the URL it was given --
    // which carries a stale cursor and would replay the whole log. Take it
    // over instead: close, and come back from the cursor we have now.
    if (rec.es !== es) return;
    es.close();
    rec.es = null;
    const info = infoOf(sid);
    if (rec.closed || (info && TERMINAL.has(info.state))) return;
    rec.retry += 1;
    if (rec.retry > 6) {
      startEventPoll(sid);
      return;
    }
    const wait = Math.min(15000, 1000 * Math.pow(2, rec.retry - 1));
    rec.timer = setTimeout(() => {
      rec.timer = null;
      if (state.open === sid) attach(sid);
    }, wait);
  };

  es.onopen = () => {
    rec.retry = 0;
  };
}

function onFrame(sid, name, data) {
  let type = name;
  let payload = data;
  let seq = null;
  let at = null;

  if (data && typeof data === "object" && typeof data.type === "string" && "payload" in data) {
    type = data.type;
    payload = data.payload;
    seq = data.seq ?? null;
    at = data.at ?? null;
  }
  if (!EVENT_TYPES.includes(type)) return;
  ingest(sid, { type, payload: payload || {}, seq, at });
}

/**
 * The documented poll fallback, used when EventSource is unavailable or has
 * failed repeatedly. Same cursor, same events, more requests.
 */
function startEventPoll(sid) {
  const rec = track(sid);
  if (rec.poll) return;
  const once = async () => {
    try {
      const data = await getJson("/api/sessions/" + encodeURIComponent(sid)
        + "/events?cursor=" + encodeURIComponent(rec.cursor));
      for (const ev of data.events || []) onFrame(sid, ev.type, ev);
      if (data.session) {
        upsertSession(data.session);
        renderSessions();
        if (state.open === sid) renderSessionView();
      }
      if (data.dropped) rec.dropped = data.dropped;
      if (data.closed) {
        rec.closed = true;
        detach(sid);
      }
    } catch {
      /* a failed poll is not fatal: the next tick tries again */
    }
  };
  rec.poll = setInterval(once, 2500);
  once();
}

// --------------------------------------------------------------------------
// Session lifecycle
// --------------------------------------------------------------------------

function solveBody() {
  const body = { puzzle: state.selected, seed: 0 };
  const model = $("optModel").value;
  if (model) body.model = model;
  const rounds = Number($("optRounds").value);
  if (rounds) body.rounds = rounds;
  const cands = Number($("optCands").value);
  if (cands) body.candidates = cands;
  return body;
}

function inspectProblem(message) {
  const box = $("inspectProblem");
  box.hidden = false;
  box.textContent = message;
}

async function startSolve() {
  if (!state.selected) return;
  const btn = $("solveBtn");
  $("inspectProblem").hidden = true;

  if (!state.durable) {
    startLegacySolve();
    return;
  }

  btn.disabled = true;
  try {
    const res = await fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(solveBody()),
    });
    if (!res.ok) {
      // 429 is the concurrency cap, 503 a missing key, 501 a deployment that
      // cannot hold a session. Each of those messages explains itself far
      // better than anything this page could invent.
      inspectProblem(await serverMessage(res));
      return;
    }
    const info = upsertSession(await res.json());
    startHeartbeat();
    renderSessions();
    openSession(info.id);
  } catch (err) {
    inspectProblem("Could not start a solve: " + (err.message || err));
  } finally {
    btn.disabled = false;
  }
}

async function openSession(sid) {
  detachAll();
  state.open = sid;
  const rec = track(sid);

  let info = infoOf(sid);
  if (!info || (!rec.result && TERMINAL.has(info.state))) {
    try {
      const data = await getJson("/api/sessions/" + encodeURIComponent(sid));
      info = upsertSession(data.session);
      if (data.result) rec.result = data.result;
    } catch (err) {
      if (!info) {
        // Gone: most likely deleted, or the process restarted.
        state.sessions.delete(sid);
        state.order = state.order.filter((id) => id !== sid);
        state.open = null;
        renderSessions();
        renderInspect();
        inspectProblem(err.message || String(err));
        return;
      }
    }
  }
  if (state.open !== sid) return;

  renderTraceHeader();
  rebuildTrace();
  renderSessionView();

  // The geometry the live grid paints into comes from the puzzle detail, not
  // from the trace: `step` events carry numbers and prose, never letters. Load
  // it before subscribing so the first tool call has somewhere to land.
  if (info) await ensureDetail(info.puzzle_id);
  if (state.open !== sid) return;
  renderSessionGrid();

  const watchable = info && !rec.closed && (rec.cursor === 0 || LIVE.has(info.state));
  if (sid !== LEGACY_SID && watchable) attach(sid);
}

function closeSession() {
  detachAll();
  state.open = null;
  renderTraceHeader();
  rebuildTrace();
  renderInspect();
}

async function stopSession(sid) {
  try {
    const res = await fetch(
      "/api/sessions/" + encodeURIComponent(sid) + "/stop", { method: "POST" },
    );
    if (!res.ok) {
      inspectProblem(await serverMessage(res));
      return;
    }
    const data = await res.json();
    if (data.session) upsertSession(data.session);
    renderSessions();
    if (state.open === sid) renderSessionView();
  } catch (err) {
    inspectProblem("Could not stop that session: " + (err.message || err));
  }
}

async function deleteSession(sid) {
  detach(sid);
  try {
    const res = await fetch("/api/sessions/" + encodeURIComponent(sid), { method: "DELETE" });
    if (!res.ok && res.status !== 404) {
      inspectProblem(await serverMessage(res));
      return;
    }
  } catch (err) {
    inspectProblem("Could not dismiss that session: " + (err.message || err));
    return;
  }
  state.sessions.delete(sid);
  state.logs.delete(sid);
  state.order = state.order.filter((id) => id !== sid);
  renderSessions();
  if (state.open === sid) closeSession();
}

// -- the single-request fallback ------------------------------------------- //

/**
 * Solve through POST /api/solve/stream, for a deployment that cannot hold a
 * session. The frames are mapped onto the same trace events the session log
 * produces, so the rest of this file does not know the difference; what it
 * cannot fake is a model-call record, because that endpoint does not send one.
 */
async function startLegacySolve() {
  if (state.legacyRunning) return;
  const pid = state.selected;
  const row = state.puzzles.find((p) => p.id === pid) || {};
  const btn = $("solveBtn");
  state.legacyRunning = true;
  btn.disabled = true;

  state.logs.delete(LEGACY_SID);
  const rec = track(LEGACY_SID);
  let seq = 0;
  const push = (type, payload) => {
    ingest(LEGACY_SID, { type, payload, seq: ++seq, at: Date.now() / 1000 });
  };

  upsertSession({
    id: LEGACY_SID,
    puzzle_id: pid,
    title: row.title || "",
    size: row.size || "",
    entries: row.entries || 0,
    open_cells: row.open_cells || 0,
    state: "running",
    round: 0,
    step: "",
    message: "solving inside one request",
    llm_calls: 0,
    input_tokens: 0,
    output_tokens: 0,
    cost_usd: 0,
    elapsed_s: 0,
    error: "",
    model: (state.health && state.health.model) || "",
    max_rounds: Number($("optRounds").value) || 3,
    solved: null,
    cells_correct: 0,
    cells_total: 0,
  });
  state.order = [LEGACY_SID];
  renderSessions();
  await openSession(LEGACY_SID);
  push("status", {
    state: "running",
    message: "one solve inside one request: no model-call records on this deployment",
  });

  const started = performance.now();
  const tick = setInterval(() => {
    patchSession(LEGACY_SID, { elapsed_s: (performance.now() - started) / 1000 });
    if (state.open === LEGACY_SID) renderSessionStats(infoOf(LEGACY_SID));
    renderSessions();
  }, 1000);

  try {
    const res = await fetch("/api/solve/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(solveBody()),
    });
    if (!res.ok || !res.body) {
      const message = await serverMessage(res);
      patchSession(LEGACY_SID, { state: "error", error: message });
      push("error", { message });
      push("status", { state: "error", message });
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop();
      for (const frame of frames) {
        const name = /^event: (.+)$/m.exec(frame);
        const data = /^data: (.+)$/m.exec(frame);
        if (!name || !data) continue;
        let parsed;
        try {
          parsed = JSON.parse(data[1]);
        } catch {
          continue;
        }
        if (name[1] === "event") push("step", parsed);
        else if (name[1] === "error") {
          patchSession(LEGACY_SID, { state: "error", error: parsed.message || "" });
          push("error", parsed);
        } else if (name[1] === "result") {
          rec.result = parsed;
          const stats = parsed.stats || {};
          patchSession(LEGACY_SID, {
            state: "done",
            llm_calls: stats.llm_calls || 0,
            input_tokens: stats.input_tokens || 0,
            output_tokens: stats.output_tokens || 0,
            cost_usd: stats.cost_usd || 0,
            elapsed_s: stats.wall_seconds || 0,
            round: stats.rounds || 0,
            solved: parsed.score ? parsed.score.solved : null,
            cells_correct: parsed.score ? parsed.score.cells_correct : 0,
            cells_total: parsed.score ? parsed.score.cells_total : 0,
          });
          push("result", parsed);
        }
      }
    }
    const info = infoOf(LEGACY_SID);
    if (info && info.state === "running") {
      patchSession(LEGACY_SID, { state: "done" });
      push("status", { state: "done", message: "stream finished" });
    }
  } catch (err) {
    const message = "Solve failed: " + (err.message || err);
    patchSession(LEGACY_SID, { state: "error", error: message });
    push("error", { message });
  } finally {
    clearInterval(tick);
    state.legacyRunning = false;
    btn.disabled = !canSolve();
    renderSessions();
    if (state.open === LEGACY_SID) renderSessionView();
  }
}

// --------------------------------------------------------------------------
// Polling
//
// The sidebar has to keep ticking while the centre column shows something
// else, and that is the only reason to poll at all -- so the timer runs while
// something is live and stops when nothing is. A tab that comes back to the
// foreground gets one poll, which covers a session started in another tab
// without leaving a timer running forever.
// --------------------------------------------------------------------------

async function pollSessions() {
  if (!state.durable) return;
  let data;
  try {
    data = await getJson("/api/sessions");
  } catch {
    return; // transient: the next tick tries again
  }
  state.durable = data.durable !== false;
  if (data.max_concurrent) state.maxConcurrent = data.max_concurrent;

  const seen = new Set();
  for (const info of data.sessions || []) {
    upsertSession(info);
    seen.add(info.id);
  }
  for (const sid of Array.from(state.sessions.keys())) {
    if (seen.has(sid) || sid === LEGACY_SID) continue;
    // Evicted by the manager, or deleted from another tab. Drop the
    // subscription first: an EventSource left open would keep retrying a 404.
    detach(sid);
    state.sessions.delete(sid);
    state.logs.delete(sid);
    if (state.open === sid) closeSession();
  }
  state.order = (data.sessions || []).map((s) => s.id);
  if (state.sessions.has(LEGACY_SID) && !state.order.includes(LEGACY_SID)) {
    state.order.unshift(LEGACY_SID);
  }

  renderSessions();
  renderPuzzles();
  if (state.open && state.view === "session") renderSessionMeta();
  if (anyLive()) startHeartbeat();
  else stopHeartbeat();
}

function startHeartbeat() {
  if (state.heartbeat) return;
  state.ticks = 0;
  state.heartbeat = setInterval(() => {
    state.ticks += 1;
    // Every second: move the clocks. Every other second: ask the server.
    renderSessions();
    if (state.open && state.view === "session") {
      const info = infoOf(state.open);
      if (info) renderSessionStats(info);
    }
    if (state.ticks % 2 === 0) pollSessions();
  }, 1000);
}

function stopHeartbeat() {
  if (!state.heartbeat) return;
  clearInterval(state.heartbeat);
  state.heartbeat = null;
}

// --------------------------------------------------------------------------
// Options, keyboard, boot
// --------------------------------------------------------------------------

function canSolve() {
  if (!state.selected) return false;
  if (state.legacyRunning) return false;
  if (!state.health || !state.health.api_key_configured) return false;
  const row = state.puzzles.find((p) => p.id === state.selected);
  return !row || row.fits_here !== false;
}

function fillOptions() {
  const model = clear($("optModel"));
  const ids = MODEL_CHOICES.slice();
  const current = (state.health && state.health.model) || MODEL_CHOICES[0];
  if (!ids.includes(current)) ids.unshift(current);
  for (const id of ids) {
    const opt = add(model, el("option", null, id));
    opt.value = id;
    if (id === current) opt.selected = true;
  }

  const rounds = clear($("optRounds"));
  for (let n = 1; n <= 6; n++) {
    const opt = add(rounds, el("option", null, n === 1 ? "1 (no repair)" : String(n)));
    opt.value = String(n);
    if (n === 3) opt.selected = true;
  }

  const cands = clear($("optCands"));
  const auto = add(cands, el("option", null, "default"));
  auto.value = "";
  for (const n of [4, 6, 8, 10, 12, 16]) {
    const opt = add(cands, el("option", null, String(n)));
    opt.value = String(n);
  }

  for (const id of ["optModel", "optRounds", "optCands"]) {
    $(id).addEventListener("change", updateCostNote);
  }
}

function wireKeys() {
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    const focused = document.activeElement;
    const inCard = focused && focused.closest ? focused.closest(".call.open") : null;
    const card = inCard || state.lastCard;
    if (collapseCard(card)) {
      if (card === state.lastCard) state.lastCard = null;
      e.preventDefault();
    }
  });
}

async function boot() {
  fillOptions();
  wirePuzzleKeys();
  wireTracePanel();
  wireKeys();
  setFilter("all");
  setAuto(true);

  $("solveBtn").addEventListener("click", startSolve);
  $("backBtn").addEventListener("click", closeSession);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) pollSessions();
  });

  await loadHealth();
  fillOptions();

  try {
    const data = await getJson("/api/puzzles");
    // Already sorted smallest-first by the API. Re-sorting here would put the
    // slowest puzzle back at the top, which is what that sort exists to avoid.
    state.puzzles = data.puzzles || [];
  } catch (err) {
    state.puzzleError = err.message || String(err);
  }
  renderPuzzles();

  if (state.durable) {
    await pollSessions();
    if (anyLive()) startHeartbeat();
  }

  let restored = null;
  try {
    restored = localStorage.getItem(STORE_KEY);
  } catch {
    /* storage unavailable: start with the first puzzle instead */
  }
  const fits = (id) => {
    const row = state.puzzles.find((p) => p.id === id);
    return row && row.fits_here !== false;
  };
  const first = state.puzzles.find((p) => p.fits_here !== false);
  const pick = restored && fits(restored) ? restored : (first ? first.id : null);
  if (pick) selectPuzzle(pick);
  else showView("empty");

  $("solveBtn").disabled = !canSolve();
  if (state.health && !state.health.api_key_configured) {
    $("solveBtn").title = "ANTHROPIC_API_KEY is not set on this deployment, so no solve can start.";
  }
}

boot();

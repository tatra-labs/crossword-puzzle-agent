#!/usr/bin/env node
/**
 * Record the one-minute demo video from the real studio UI.
 *
 * This is a recording, not a reconstruction. It drives the actual page in
 * headless Chrome against a live server, presses the real Solve button, and
 * captures what the browser paints -- so the grid, the trace, the prompt, the
 * tool call and every number on screen come from a real solve. Frames are
 * stamped with wall-clock time and encoded with those durations, so video time
 * equals real time: nothing is sped up and nothing is cut.
 *
 * The captions are injected into the page rather than burned in afterwards, so
 * they inherit the UI's own tokens and read correctly in either theme.
 *
 * Usage
 * -----
 *   # 1. dependencies (Chrome and ffmpeg must already be on the machine)
 *   npm install puppeteer-core
 *
 *   # 2. a server with a COLD clue cache, so the solve makes real calls --
 *   #    a warm cache answers from SQLite and the trace would show
 *   #    "0 model calls, $0.00", which is true but not what a demo should show
 *   XWORD_CACHE_DIR=/tmp/demo-cache python -m uvicorn app:app --port 8011
 *
 *   # 3. record (spends ~$0.03 of real Anthropic credit on a 7x7)
 *   node scripts/record-demo.js
 *
 * Environment
 * -----------
 *   BASE        server to record        (default http://127.0.0.1:8011)
 *   OUT         output mp4              (default docs/demo.mp4)
 *   GIF         output gif              (default docs/demo.gif; GIF=0 to skip)
 *   GIF_ONLY    rebuild the gif from an existing OUT and record nothing (free)
 *   PUZZLE      bundled id to solve     (default midi-03)
 *   MODEL       model to solve with     (default claude-sonnet-5)
 *   THEME       dark | light            (default dark)
 *   TARGET_MS   recording length        (default 57000, hard cap is 60000)
 *   FPS         capture rate            (default 12; ~17 is the ceiling)
 *   CHROME      browser executable      (default: first one found)
 *   KEEP_FRAMES set to keep the frame directory for inspection
 */
const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

let puppeteer;
try {
  puppeteer = require("puppeteer-core");
} catch {
  console.error("puppeteer-core is not installed. Run: npm install puppeteer-core");
  process.exit(1);
}

const REPO = path.resolve(__dirname, "..");
const BASE = process.env.BASE || "http://127.0.0.1:8011";
const OUT = path.resolve(process.env.OUT || path.join(REPO, "docs", "demo.mp4"));
const GIF = process.env.GIF === "0"
  ? null
  : path.resolve(process.env.GIF || path.join(REPO, "docs", "demo.gif"));
const PUZZLE = process.env.PUZZLE || "midi-03";
const MODEL = process.env.MODEL || "claude-sonnet-5";
const THEME = process.env.THEME || "dark";
const TARGET_MS = Math.min(Number(process.env.TARGET_MS || 57000), 60000);
const FPS = Number(process.env.FPS || 12);
const FRAMES = path.join(REPO, ".demo-frames");

/**
 * When each beat fires, in ms. One source of truth: the beat table below reads
 * it, and so do the GIF's highlight windows -- otherwise moving a beat silently
 * points the GIF at the wrong four seconds.
 */
const AT = {
  title: 0, dropTitle: 4200, sidebar: 4900, clues: 8200, why: 12200, bp: 16200,
  solve: 20400, steps: 23400, calls: 28400, prompt: 32200, toolcall: 37200,
  grid: 42400, result: 47600, sessions: 51800, close: 55000,
};

/**
 * The README embeds a GIF, because GitHub will not reliably play a <video>
 * pointed at a path inside the repository, while an animated GIF renders
 * anywhere markdown does. Windows are [start, end] seconds, derived from the
 * beat times so they cannot drift out of step with the schedule.
 */
function highlightWindows(recordedSeconds) {
  const s = (ms) => ms / 1000;
  return [
    [0.6, 3.8],                                   // title card
    [s(AT.solve) + 2.2, s(AT.solve) + 6.0],       // Solve pressed, trace starts
    [s(AT.prompt) + 0.8, s(AT.prompt) + 4.6],     // the prompt as sent
    [s(AT.toolcall) + 0.8, s(AT.toolcall) + 4.6], // the tool call back
    [s(AT.result) + 0.8, s(AT.result) + 4.4],     // the solved grid and numbers
  ].filter(([a, b]) => a < recordedSeconds && b <= recordedSeconds);
}

/** Two-pass palette GIF. The UI is mostly flat colour, so this stays ~1.3 MB. */
function buildGif(src, dest, recordedSeconds) {
  const windows = highlightWindows(recordedSeconds);
  if (!windows.length) {
    console.log("no highlight window fits in the recording; skipping the gif");
    return;
  }
  const tags = windows.map((_, i) => String.fromCharCode(97 + i));
  const trims = windows
    .map(([a, b], i) => `[0:v]trim=${a}:${b},setpts=PTS-STARTPTS[${tags[i]}]`)
    .join(";");
  const filter =
    `${trims};${tags.map((t) => `[${t}]`).join("")}concat=n=${tags.length}:v=1[v];` +
    "[v]fps=12,scale=1100:-1:flags=lanczos,split[s0][s1];" +
    "[s0]palettegen=max_colors=200:stats_mode=diff[p];" +
    "[s1][p]paletteuse=dither=bayer:bayer_scale=4";
  execFileSync("ffmpeg", ["-y", "-hide_banner", "-loglevel", "error",
                          "-i", src, "-filter_complex", filter, dest],
               { stdio: "inherit" });
  const seconds = windows.reduce((n, [a, b]) => n + (b - a), 0);
  console.log(`wrote ${dest}  ${(fs.statSync(dest).size / 1e6).toFixed(2)} MB  ` +
              `${seconds.toFixed(1)}s of highlights from ${windows.length} windows`);
}

/** Chrome or Edge, wherever this machine keeps it. */
function findChrome() {
  if (process.env.CHROME) return process.env.CHROME;
  const candidates = [
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ];
  const hit = candidates.find((p) => fs.existsSync(p));
  if (!hit) {
    console.error("No Chrome or Edge found. Set CHROME=/path/to/chrome.");
    process.exit(1);
  }
  return hit;
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// --------------------------------------------------------------------------- //
// Overlay: a caption bar, full-screen cards for the open and close, and a
// spotlight that frames whichever panel is being talked about.
// --------------------------------------------------------------------------- //
const OVERLAY_CSS = `
#__ov { position: fixed; inset: 0; z-index: 99999; pointer-events: none; font-family: inherit; }
#__cap { position: absolute; left: 50%; transform: translateX(-50%); bottom: 34px;
         max-width: 1120px; padding: 14px 22px;
         background: color-mix(in srgb, var(--panel) 92%, transparent);
         border: 1px solid var(--line); border-left: 3px solid var(--accent);
         border-radius: 10px; color: var(--ink); font-size: 20px; line-height: 1.45;
         box-shadow: 0 10px 34px rgba(0,0,0,.35); opacity: 0;
         transition: opacity .28s ease; text-align: left; }
#__cap.on { opacity: 1; }
#__cap b { color: var(--accent); font-weight: 600; }
#__cap .k { font-family: var(--mono, monospace); font-size: 18px; color: var(--good); }
#__card { position: absolute; inset: 0; display: flex; flex-direction: column;
          align-items: center; justify-content: center; gap: 18px; background: var(--bg);
          opacity: 0; transition: opacity .4s ease; text-align: center; padding: 0 8%; }
#__card.on { opacity: 1; }
#__card h1 { margin: 0; font-size: 58px; letter-spacing: -.5px; color: var(--ink); font-weight: 650; }
#__card p { margin: 0; font-size: 25px; color: var(--muted); line-height: 1.5; max-width: 980px; }
#__card .pipe { font-family: var(--mono, monospace); font-size: 21px; color: var(--accent); letter-spacing: .5px; }
#__card .foot { font-family: var(--mono, monospace); font-size: 19px; color: var(--muted); }
#__spot { position: absolute; border: 2px solid var(--accent); border-radius: 10px; opacity: 0;
          transition: opacity .25s ease, top .3s ease, left .3s ease, width .3s ease, height .3s ease;
          box-shadow: 0 0 0 4000px rgba(0,0,0,.42); }
#__spot.on { opacity: 1; }
`;

const api = {
  cap: (page, html) => page.evaluate((h) => {
    const c = document.getElementById("__cap");
    if (!h) { c.classList.remove("on"); return; }
    c.innerHTML = h; c.classList.add("on");
  }, html),
  card: (page, html) => page.evaluate((h) => {
    const c = document.getElementById("__card");
    if (!h) { c.classList.remove("on"); c.innerHTML = ""; return; }
    c.innerHTML = h; c.classList.add("on");
  }, html),
  spot: (page, sel) => page.evaluate((s) => {
    const el = document.getElementById("__spot");
    if (!s) { el.classList.remove("on"); return; }
    const t = document.querySelector(s);
    if (!t) { el.classList.remove("on"); return; }
    const r = t.getBoundingClientRect(), pad = 8;
    el.style.top = (r.top - pad) + "px";
    el.style.left = (r.left - pad) + "px";
    el.style.width = (r.width + pad * 2) + "px";
    el.style.height = (r.height + pad * 2) + "px";
    el.classList.add("on");
  }, sel),
};

async function main() {
  // Rebuilding the gif from an existing recording spends nothing, which makes
  // it safe to iterate on the highlight windows.
  if (process.env.GIF_ONLY) {
    if (!fs.existsSync(OUT)) {
      console.error(`GIF_ONLY needs an existing recording at ${OUT}`);
      process.exit(1);
    }
    const dur = Number(execFileSync("ffprobe", ["-v", "error", "-show_entries",
      "format=duration", "-of", "csv=p=0", OUT]).toString().trim());
    if (GIF) buildGif(OUT, GIF, dur);
    else console.log("GIF=0, nothing to do");
    return;
  }

  fs.rmSync(FRAMES, { recursive: true, force: true });
  fs.mkdirSync(FRAMES, { recursive: true });

  const browser = await puppeteer.launch({
    executablePath: findChrome(),
    headless: "new",
    args: ["--no-sandbox", "--disable-dev-shm-usage", "--hide-scrollbars",
           "--force-device-scale-factor=1"],
    defaultViewport: { width: 1600, height: 900, deviceScaleFactor: 1 },
  });
  const page = await browser.newPage();
  await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: THEME }]);
  const pageErrors = [];
  page.on("pageerror", (e) => pageErrors.push(String(e.message)));

  await page.goto(BASE + "/", { waitUntil: "networkidle2", timeout: 30000 });
  await page.waitForSelector("#puzzleList button.row-main", { timeout: 20000 });
  await page.evaluate((css) => {
    const style = document.createElement("style");
    style.textContent = css;
    document.head.appendChild(style);
    const ov = document.createElement("div");
    ov.id = "__ov";
    ov.innerHTML = '<div id="__spot"></div><div id="__cap"></div><div id="__card"></div>';
    document.body.appendChild(ov);
  }, OVERLAY_CSS);

  // ---- actions on the real page ---- //
  const clickPuzzle = (id) => page.evaluate((pid) => {
    Array.from(document.querySelectorAll("#puzzleList .row"))
      .find((r) => r.textContent.includes(pid))
      .querySelector("button.row-main").click();
  }, id);
  const setOpt = (id, val) => page.evaluate((i, v) => {
    const el = document.getElementById(i);
    el.value = v;
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }, id, val);
  const press = (sel) => page.evaluate((s) => document.querySelector(s).click(), sel);
  const filter = (name) => page.evaluate((n) => {
    document.querySelector(`#traceFilter button[data-filter="${n}"]`).click();
  }, name);
  const expandCall = () => page.evaluate(() => {
    const card = Array.from(document.querySelectorAll("#traceBody article.call"))
      .find((c) => !/cache hit/.test(c.textContent));
    if (!card) return false;
    card.querySelector("button.call-head").click();
    card.scrollIntoView({ block: "center" });
    return true;
  });
  const scrollCardTo = (label) => page.evaluate((lab) => {
    const card = document.querySelector("#traceBody article.call.open");
    if (!card) return false;
    const hit = Array.from(card.querySelectorAll("*")).find(
      (n) => n.children.length === 0 && new RegExp(lab, "i").test(n.textContent || ""));
    if (hit) hit.scrollIntoView({ block: "center", behavior: "smooth" });
    return !!hit;
  }, label);
  const readStats = () => page.evaluate(() => {
    // Read the tiles structurally. Each is <div class="stat"><b>value</b>
    // <span>label</span></div> with no whitespace between the two, so
    // textContent on the container yields "done state 6.9s elapsed" run
    // together as "donestate6.9selapsed" -- unreadable in a caption.
    const tiles = Array.from(document.querySelectorAll("#sessionStats .stat"))
      .map((t) => {
        const v = (t.querySelector("b") || {}).textContent || "";
        const l = (t.querySelector("span") || {}).textContent || "";
        return (v + " " + l).trim();
      })
      .filter(Boolean);
    const pill = (document.getElementById("sessionPill") || {}).textContent || "";
    return { stats: tiles.join("  ·  "), pill: pill.trim() };
  });
  /** The puzzle's real shape, so a caption cannot claim the wrong numbers. */
  const readShape = () => page.evaluate(() => {
    const meta = (document.getElementById("inspectMeta") || {}).textContent || "";
    const n = (re) => { const m = meta.match(re); return m ? m[1] : null; };
    return { entries: n(/(\d+)\s+entries/), squares: n(/(\d+)\s+open squares/) };
  });

  // Settle on the chosen puzzle before recording, so the video opens on a
  // finished page rather than on two clicks nobody needs to watch.
  await clickPuzzle(PUZZLE);
  await setOpt("optModel", MODEL);
  await setOpt("optRounds", "2");
  await page.waitForFunction(
    () => (document.getElementById("inspectClues") || {}).textContent.length > 40,
    { timeout: 15000 });
  await wait(600);

  const P = "propose \u2192 fuse \u2192 propagate \u2192 commit \u2192 critique \u2192 repair";
  const beats = [
    [AT.title, "title", () => api.card(page,
      `<h1>Crossword Agent</h1>
       <p>A language model reads the clues. Probabilistic inference and search
          make the whole grid <b>agree with itself</b>.</p>
       <div class="pipe">${P}</div>`)],
    [AT.dropTitle, "drop title", () => api.card(page, "")],
    [AT.sidebar, "sidebar", async () => {
      await api.spot(page, "#puzzleList");
      await api.cap(page, "Ten original puzzles ship with the repo. Pick one and inspect it before spending anything.");
    }],
    [AT.clues, "clues", async () => {
      await api.spot(page, "#inspectClues");
      // Read the counts off the page: hard-coding them means the caption is
      // wrong the moment PUZZLE changes, and a demo that misstates its own
      // numbers is worse than one that shows none.
      const s = await readShape();
      const shape = s.entries && s.squares
        ? `<b>${s.entries} clues over ${s.squares} shared squares</b>`
        : "<b>a few dozen clues over a few dozen shared squares</b>";
      await api.cap(page, `A crossword is not a quiz. ${shape} \u2014 every answer is constrained by the ones crossing it.`);
    }],
    [AT.why, "why", async () => {
      await api.spot(page, null);
      await api.cap(page, "So the agent asks for <b>several ranked answers per clue, with probabilities</b> \u2014 and keeps probability on <i>none of the above</i>.");
    }],
    [AT.bp, "bp", () => api.cap(page,
      "Then <b>loopy belief propagation</b> over the grid tempers every entry by what its crossings believe, and search picks the best set that mutually fits.")],
    [AT.solve, "solve", async () => {
      await api.spot(page, "#solveBtn");
      await api.cap(page, "Press <b>Solve</b>. Real model calls, real credit \u2014 the button says the cost first.");
      await wait(900);
      await press("#solveBtn");
    }],
    [AT.steps, "steps", async () => {
      await api.spot(page, "#traceBody");
      await api.cap(page, "The trace, live. One row per agent step \u2014 <span class='k'>ingest, propose, fuse, propagate, commit</span>.");
    }],
    [AT.calls, "calls", async () => {
      await api.spot(page, null);
      await api.cap(page, "Underneath the steps: <b>every request to the model</b>, which the CLI never showed.");
      await filter("llm_call");
    }],
    [AT.prompt, "prompt", async () => {
      await expandCall();
      await api.cap(page, "The prompt <b>exactly as sent</b> \u2014 each entry with its length and the pattern already believed known.");
    }],
    [AT.toolcall, "toolcall", async () => {
      await scrollCardTo("tool");
      await api.cap(page, "And the <b>tool call back</b>: ranked answers with the model's own probabilities. This is what feeds the inference.");
    }],
    [AT.grid, "grid", async () => {
      await filter("all");
      await api.spot(page, "#sessionGrid");
      await api.cap(page, `The loop is <span class="k">${P}</span> \u2014 re-asking only the shaky clues, now with the crossing letters it trusts.`);
    }],
    [AT.result, "result", async () => {
      await api.spot(page, "#sessionStats");
      const s = await readStats();
      await api.cap(page, `Solved, and every number on screen is measured: <span class="k">${s.stats || s.pill}</span>`);
    }],
    [AT.sessions, "sessions", async () => {
      await api.spot(page, "#sessionList");
      await api.cap(page, "Sessions live on the server \u2014 leave one running, look at another puzzle, come back to a <b>complete</b> trace.");
    }],
    [AT.close, "close", async () => {
      await api.spot(page, null);
      await api.cap(page, "");
      await api.card(page,
        `<h1>Same model, same clues</h1>
         <p>The scaffolding turns <b>0 of 14</b> NYT puzzles solved into <b>7 of 14</b>.
            Word accuracy 49% \u2192 94%.</p>
         <div class="foot">github.com/tatra-labs/crossword-puzzle-agent</div>`);
    }],
  ];

  const t0 = Date.now();
  const shots = [];
  let bi = 0, stop = false;

  const beatRunner = (async () => {
    while (!stop && bi < beats.length) {
      const [at, label, fn] = beats[bi];
      if (Date.now() - t0 >= at) {
        bi++;
        try { await fn(); } catch (e) { console.log(`  beat "${label}" failed: ${e.message}`); }
        console.log(`  ${((Date.now() - t0) / 1000).toFixed(1)}s  ${label}`);
      } else {
        await wait(30);
      }
    }
  })();

  const interval = 1000 / FPS;
  while (Date.now() - t0 < TARGET_MS) {
    const slack = t0 + shots.length * interval - Date.now();
    if (slack > 2) await wait(slack);
    const name = path.join(FRAMES, `f${String(shots.length).padStart(6, "0")}.jpg`);
    try {
      await page.screenshot({ path: name, type: "jpeg", quality: 88 });
      shots.push({ name: path.basename(name), t: Date.now() - t0 });
    } catch { break; }
  }
  stop = true;
  await beatRunner;
  await browser.close();

  const elapsed = (Date.now() - t0) / 1000;
  console.log(`\ncaptured ${shots.length} frames over ${elapsed.toFixed(1)}s ` +
              `(${(shots.length / elapsed).toFixed(1)} fps)`);
  console.log("page errors:", pageErrors.length ? pageErrors.slice(0, 5) : "none");
  if (!shots.length) { console.error("no frames captured"); process.exit(1); }

  // Per-frame durations from real timestamps: video time == wall time.
  const lines = [];
  for (let i = 0; i < shots.length; i++) {
    const next = i + 1 < shots.length ? shots[i + 1].t : shots[i].t + interval;
    lines.push(`file '${shots[i].name}'`);
    lines.push(`duration ${((next - shots[i].t) / 1000).toFixed(4)}`);
  }
  lines.push(`file '${shots[shots.length - 1].name}'`);
  fs.writeFileSync(path.join(FRAMES, "concat.txt"), lines.join("\n") + "\n");

  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  execFileSync("ffmpeg", [
    "-y", "-hide_banner", "-loglevel", "error",
    "-f", "concat", "-safe", "0", "-i", "concat.txt",
    "-fps_mode", "cfr", "-r", "24",
    "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-pix_fmt", "yuv420p",
    "-movflags", "+faststart", OUT,
  ], { cwd: FRAMES, stdio: "inherit" });

  const size = fs.statSync(OUT).size;
  const dur = execFileSync("ffprobe", ["-v", "error", "-show_entries", "format=duration",
                                       "-of", "csv=p=0", OUT]).toString().trim();
  console.log(`wrote ${OUT}  ${(size / 1e6).toFixed(2)} MB  ${Number(dur).toFixed(1)}s`);
  if (Number(dur) > 60) console.log("WARNING: over the one-minute cap");

  if (GIF) buildGif(OUT, GIF, Number(dur));

  if (!process.env.KEEP_FRAMES) fs.rmSync(FRAMES, { recursive: true, force: true });
  else console.log("frames kept in", FRAMES);
}

main().catch((e) => { console.error("ERROR:", e.message); process.exit(1); });

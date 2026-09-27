#!/usr/bin/env node
/** G3, measured in real Chrome: bytes down, cold load, p50 per decision, two widths.
 *
 *   node bench/browser_g3.mjs [--artifact runs/onnx] [--runs 15] [--widths 1440x900,390x844]
 *                            [--out runs/browser_g3.json] [--shots docs/screenshots]
 *                            [--mode both|isolated|plain] [--chrome <path>] [--no-shots]
 *
 * Why a harness and not a browser session: G3's claim is about a *tab* — "the bytes a
 * browser downloads, the time to first answer, the time per decision" — and those three
 * only mean something if they come out of the engine users have, on a cold cache, with
 * the request log next to them. So this starts Chrome headless with a throwaway profile
 * and cache disabled, drives it over CDP, and records:
 *
 *   - `Network.loadingFinished.encodedDataLength` per URL, which is bytes on the wire as
 *     Chrome counted them — the artifact *and* the wasm runtime, in one table, so nobody
 *     can add the model and forget the 14 MB of wasm that runs it;
 *   - the page's own `?report=1` JSON, which is where per-file fetched bytes, cold-load
 *     ms, the p50 over N decisions and the parity witness lines come from (they are the
 *     same checks `node browser/selftest.mjs` gates on);
 *   - a screenshot per width, so the layout claim is a picture and not an adjective.
 *
 * Two modes, because the browser gives two different answers and picking the flattering
 * one would be the usual sin:
 *
 *   **isolated**  the server sends COOP + COEP, so the page is cross-origin-isolated: the
 *     threaded wasm build gets a worker pool and `measureUserAgentSpecificMemory` works.
 *     This is what a deployed myna page with control of its headers would do.
 *   **plain**  no such headers — what a static bucket without them gives you. Single
 *     thread, no memory API, same bytes.
 *
 * The box is measured with the load average at the time of the run (`uptime`), and that
 * number rides in the JSON: a p50 quoted without it is a mood.
 */

import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";

const ROOT = process.cwd();
const CHROME_CANDIDATES = [
  process.env.CHROME,
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "google-chrome",
  "chromium",
].filter(Boolean);

function arg(name, dflt) {
  const i = process.argv.indexOf(`--${name}`);
  if (i > 0 && process.argv[i + 1] && !process.argv[i + 1].startsWith("--")) return process.argv[i + 1];
  return i > 0 ? true : dflt;
}

const ARTIFACT = arg("artifact", "runs/onnx");
const RUNS = Number(arg("runs", 15));
const MODE = arg("mode", "both");
const WIDTHS = String(arg("widths", "1440x900,390x844")).split(",").map((w) => {
  const [width, height] = w.split("x").map(Number);
  return { width, height: height || 900 };
});
const OUT = path.resolve(arg("out", "runs/browser_g3.json"));
const SHOTS = path.resolve(arg("shots", "docs/screenshots"));
const NO_SHOTS = arg("no-shots", false);
const PORT = Number(arg("port", 8019));
const CDP_PORT = Number(arg("cdp-port", 9333));
const DEADLINE_MS = Number(arg("deadline", 240000));
const WASM_PATH = "/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm";

const MIME = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8", ".json": "application/json",
  ".wasm": "application/wasm", ".bin": "application/octet-stream",
  ".onnx": "application/octet-stream", ".png": "image/png",
};

let isolate = false;

function serve() {
  const server = http.createServer((req, res) => {
    if (isolate) {
      res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
      res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
    }
    const rel = decodeURIComponent(req.url.split("?")[0]).replace(/^\/+/, "");
    const file = path.join(ROOT, rel);
    if (!file.startsWith(ROOT + path.sep) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404).end("nope");
      return;
    }
    res.writeHead(200, { "Content-Type": MIME[path.extname(file)] || "application/octet-stream",
                         "Content-Length": fs.statSync(file).size });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise((ok) => server.listen(PORT, "127.0.0.1", () => ok(server)));
}

function launchChrome(bin) {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "myna-g3-"));
  const p = spawn(bin, [
    "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
    `--user-data-dir=${profile}`, `--remote-debugging-port=${CDP_PORT}`,
    "--allow-running-insecure-content", "about:blank",
  ], { stdio: ["ignore", "ignore", "pipe"] });
  return new Promise((ok, no) => {
    let buf = "";
    p.stderr.on("data", (d) => {
      buf += d.toString();
      if (buf.includes("ws://")) ok(p);
      if (buf.includes("Cannot open devtools") || buf.includes("Address already in use")) no(new Error(buf));
    });
    p.on("exit", (c) => no(new Error(`chrome exited ${c}: ${buf.slice(-400)}`)));
    setTimeout(() => no(new Error("chrome never announced its devtools socket")), 30000);
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const num = (v) => (typeof v === "number" ? v : 0);
const ortWebVersion = () => {
  try {
    return JSON.parse(fs.readFileSync(path.join(ROOT, "node_modules/onnxruntime-web/package.json"),
                                      "utf8")).version;
  } catch { return null; }
};

async function devtoolsUrl() {
  for (let i = 0; i < 40; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${CDP_PORT}/json/list`)).json();
      const page = list.find((t) => t.type === "page");
      if (page) return page.webSocketDebuggerUrl;
    } catch { /* not up yet */ }
    await sleep(250);
  }
  throw new Error("no CDP page target");
}

class CDP {
  constructor(ws) {
    this.ws = ws; this.next = 1; this.pending = new Map(); this.handlers = new Map();
    ws.addEventListener("message", (ev) => {
      const m = JSON.parse(ev.data);
      if (m.id && this.pending.has(m.id)) {
        const { ok, no } = this.pending.get(m.id);
        this.pending.delete(m.id);
        m.error ? no(new Error(JSON.stringify(m.error))) : ok(m.result);
      } else if (m.method && this.handlers.has(m.method)) {
        this.handlers.get(m.method).forEach((h) => h(m.params));
      }
    });
  }
  static async connect(url) {
    const ws = new WebSocket(url);
    await new Promise((ok, no) => { ws.addEventListener("open", ok); ws.addEventListener("error", no); });
    return new CDP(ws);
  }
  send(method, params = {}) {
    const id = this.next++;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((ok, no) => this.pending.set(id, { ok, no }));
  }
  on(method, fn) {
    if (!this.handlers.has(method)) this.handlers.set(method, []);
    this.handlers.get(method).push(fn);
  }
}

async function evalJS(cdp, expression) {
  const r = await cdp.send("Runtime.evaluate", { expression, awaitPromise: true,
                                                 returnByValue: true });
  if (r.exceptionDetails) {
    throw new Error(`page threw: ${r.exceptionDetails.exception?.description
      || JSON.stringify(r.exceptionDetails).slice(0, 300)}`);
  }
  return r.result.value;
}

/** One cold-cache load at one width, in one isolation mode. */
async function run(cdp, { width, height, isolated, runs, tag }) {
  isolate = isolated;
  const wire = new Map();
  const seen = new Map();
  const onLoad = (p) => {
    const e = wire.get(p.requestId);
    if (e) wire.set(p.requestId, { ...e, bytes: p.encodedDataLength });
  };
  const onResponse = (p) => {
    seen.set(p.requestId, p.response.url);
    wire.set(p.requestId, { url: p.response.url, status: p.response.status });
  };
  cdp.on("Network.loadingFinished", onLoad);
  cdp.on("Network.responseReceived", onResponse);
  await cdp.send("Emulation.setDeviceMetricsOverride",
                 { width, height, deviceScaleFactor: 2, mobile: width < 700 });
  await cdp.send("Network.clearBrowserCache");
  await cdp.send("Network.setCacheDisabled", { cacheDisabled: true });
  const url = `http://127.0.0.1:${PORT}/browser/index.html?report=1&runs=${runs}`
    + `&artifact=${encodeURIComponent(ARTIFACT)}`;
  const t0 = Date.now();
  await cdp.send("Page.navigate", { url });
  let phase = "";
  for (;;) {
    phase = await evalJS(cdp, "document.getElementById('phase').textContent");
    if (phase === "ready" || phase === "stopped") break;
    if (Date.now() - t0 > DEADLINE_MS) throw new Error(`${tag}: stuck at '${phase}'`);
    await sleep(500);
  }
  const banner = await evalJS(cdp, "document.getElementById('banner').textContent");
  if (phase === "stopped") throw new Error(`${tag}: page reported ${banner}`);
  const report = await evalJS(cdp, "JSON.parse(document.getElementById('report').textContent)");
  const bytesByPath = {};
  for (const { url: u, status, bytes } of wire.values()) {
    const p = new URL(u).pathname;
    bytesByPath[p] = (bytesByPath[p] || 0) + (bytes ?? 0);
    if (status !== 200) bytesByPath[p] = `HTTP ${status} (${bytes ?? 0} B)`;
  }
  let shot = null;
  if (!NO_SHOTS) {
    fs.mkdirSync(SHOTS, { recursive: true });
    shot = path.join(SHOTS, `${tag}.png`);
    const img = await cdp.send("Page.captureScreenshot", { format: "png",
                                                           captureBeyondViewport: true });
    fs.writeFileSync(shot, Buffer.from(img.data, "base64"));
  }
  await cdp.send("Emulation.clearDeviceMetricsOverride");
  return { tag, width, height, cross_origin_isolated: isolated,
           wall_ms: Date.now() - t0, requests: bytesByPath,
           wasm_bytes: num(bytesByPath[WASM_PATH]),
           wire_bytes_total: Object.values(bytesByPath).reduce((a, b) => a + num(b), 0),
           report, screenshot: shot && path.relative(ROOT, shot) };
}

function table(rows) {
  // A row whose page produced no timings must not take the whole table down: the
  // harness's job is to report what each tab did, including a tab that did nothing.
  const span = (a, f) => (a.length ? f(a) : "n/a");
  const head = ["mode", "width", "artifact MiB", "wire MiB", "wasm MiB", "cold ms", "p50 ms",
                "min", "max", "threads", "js MiB", "checks"];
  const cell = (r) => [
    r.cross_origin_isolated ? "isolated" : "plain", `${r.width}x${r.height}`,
    (r.report.transfer.total_bytes / 2 ** 20).toFixed(2),
    (r.wire_bytes_total / 2 ** 20).toFixed(2),
    (r.wasm_bytes / 2 ** 20).toFixed(2),
    r.report.cold_load_ms.toFixed(0), r.report.decision.p50_ms.toFixed(0),
    span(r.report.decision.e2e_ms, (a) => Math.min(...a).toFixed(0)),
    span(r.report.decision.e2e_ms, (a) => Math.max(...a).toFixed(0)),
    r.report.wasm_threads,
    r.report.tab_memory.available ? (r.report.tab_memory.bytes / 2 ** 20).toFixed(1)
      : `n/a (${r.report.tab_memory.reason})`,
    `${r.report.passed}/${r.report.passed + r.report.failed}`,
  ];
  const rowsAll = [head, ...rows.map(cell)];
  const w = head.map((_, i) => Math.max(...rowsAll.map((r) => String(r[i]).length)));
  return rowsAll.map((r) => r.map((c, i) => String(c).padEnd(w[i])).join("  ")).join("\n");
}

async function main() {
  const chrome = CHROME_CANDIDATES.find((c) => (c.startsWith("/")
    ? fs.existsSync(c) : spawnSync(c, ["--version"]).status === 0));
  if (!chrome) throw new Error("no Chrome found: pass --chrome <path>");
  const version = await new Promise((ok) => {
    const p = spawn(chrome, ["--version"], { stdio: ["ignore", "pipe", "ignore"] });
    let s = "";
    p.stdout.on("data", (d) => s += d);
    p.on("close", () => ok(s.trim()));
  });
  const load = spawnSync("uptime", [], { encoding: "utf8" }).stdout.trim();

  const server = await serve();
  const proc = await launchChrome(chrome);
  const cdp = await CDP.connect(await devtoolsUrl());
  await cdp.send("Page.enable");
  await cdp.send("Runtime.enable");
  await cdp.send("Network.enable");

  const modes = MODE === "both" ? [true, false] : [MODE === "isolated"];
  const rows = [];
  for (const isolated of modes) {
    for (const { width, height } of WIDTHS) {
      const tag = `${isolated ? "isolated" : "plain"}-${width}px`;
      process.stdout.write(`run ${tag} ... `);
      const r = await run(cdp, { width, height, isolated, runs: RUNS, tag });
      const d = r.report.decision;
      console.log(`p50 ${d.p50_ms} ms over ${d.runs}, ${(r.wire_bytes_total / 2 ** 20).toFixed(1)} MiB on the wire`);
      rows.push(r);
    }
  }

  const payload = {
    // §9.30: --artifact runs/onnx vs runs/onnx_int8 is the whole int8 row, and
    // --mode/--widths change which of the four rows a reader is looking at.
    cmd: ["node", path.relative(ROOT, process.argv[1]), ...process.argv.slice(2)].join(" "),
    note: "written by bench/browser_g3.mjs; every number is Chrome's or the page's own, "
        + "from a cold cache on this box",
    chrome: version, chrome_headless: "new",
    onnxruntime_web: ortWebVersion(),
    box: { load, platform: `${os.type()} ${os.release()}`, cpus: os.cpus().length,
           mem_gib: +(os.totalmem() / 2 ** 30).toFixed(1) },
    artifact: ARTIFACT, runs: RUNS, expected: "browser/expected.json",
    server: { host: `127.0.0.1:${PORT}`, cache_disabled: true,
              coop_coep: "per mode, recorded on each row" },
    rows,
    summary: rows.map((r) => ({
      mode: r.cross_origin_isolated ? "isolated" : "plain", width: r.width,
      artifact_mib: +(r.report.transfer.total_bytes / 2 ** 20).toFixed(2),
      wire_mib: +(r.wire_bytes_total / 2 ** 20).toFixed(2),
      // the runtime is not the model; a row that reports only the model's bytes is the
      // oldest trick in this genre, so the wasm blob gets its own column
      wasm_runtime_mib: +(r.wasm_bytes / 2 ** 20).toFixed(2),
      cold_load_ms: r.report.cold_load_ms,
      session_create_ms: r.report.transfer.session_create_ms,
      p50_ms: r.report.decision.p50_ms,
      min_ms: r.report.decision.e2e_ms.length ? Math.min(...r.report.decision.e2e_ms) : null,
      max_ms: r.report.decision.e2e_ms.length ? Math.max(...r.report.decision.e2e_ms) : null,
      threads: r.report.wasm_threads,
      peak_tile_mib: r.report.peak_tile_mib,
      tab_memory_mib: r.report.tab_memory.available
        ? +(r.report.tab_memory.bytes / 2 ** 20).toFixed(1) : null,
      tab_memory_note: r.report.tab_memory.available ? null : r.report.tab_memory.reason,
      checks_passed: r.report.passed, checks_failed: r.report.failed,
      weights_requests: Object.keys(r.requests).filter((p) => p.endsWith("weights.bin")).length,
      screenshot: r.screenshot,
    })),
  };
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify(payload, null, 1) + "\n");
  console.log(`\n${table(rows)}\n\nwrote ${path.relative(ROOT, OUT)}`);
  const failed = rows.filter((r) => r.report.failed > 0);
  if (failed.length) console.log(`FAILED checks in ${failed.map((f) => f.tag).join(", ")}`);

  cdp.ws.close();
  proc.kill();
  server.close();
  return failed.length ? 1 : 0;
}

process.exit(await main().catch((e) => { console.error(`\n${e.message}`); return 2; }));

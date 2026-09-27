/** The tab: download the artifact, answer a request, and show the witness.
 *
 * There is no build step and no server logic here. Every number on the page comes from
 * one of three places, and the page says which:
 *
 *   **fetched**  bytes this tab actually pulled, counted per file in `myna.js`;
 *   **measured**  wall-clock this tab observed (cold load, per-decision, per scan call);
 *   **witnessed**  the parity lines from `browser/parity.mjs`, the same module
 *                `browser/selftest.mjs` fails the build on — so a green checkmark in the
 *                browser is produced by the code the gate runs, not a copy of it.
 *
 * `?report=1` adds a machine-readable block with all of it as JSON, for the measurement
 * pass in SPEC §5 P6 6b and for a screenshot to be checkable against.
 *
 * Query parameters: `artifact=runs/onnx` (the profile to load), `expected=browser/
 * expected.json`, `runs=12` (decisions to time), `report=1`, `shape=1067` (a question
 * width to time, which is a shape probe and claims no parity at that width).
 */

import { MynaWeb } from "./myna.js";
import { runChecks } from "./parity.mjs";

const q = new URLSearchParams(location.search);
/** Site-root-relative, because the page is served from the repository root and its own
 *  path is `/browser/index.html` — a relative `runs/onnx` would ask for
 *  `/browser/runs/onnx`. Typing either form in the artifact box works. */
const fromRoot = (p) => (p.startsWith("/") || /^https?:/.test(p) ? p : `/${p}`);
const ARTIFACT = fromRoot(q.get("artifact") || "runs/onnx").replace(/\/$/, "");
const EXPECTED = fromRoot(q.get("expected") || "browser/expected.json");
const RUNS = Number(q.get("runs") || 12);
const SHAPE = q.get("shape") ? Number(q.get("shape")) : null;
const REPORT = q.get("report") === "1";

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const row = (k, v) => {
  const li = el("li");
  li.append(el("span", "k", k), el("span", "v", v));
  return li;
};
const mib = (bytes) => (bytes / 2 ** 20).toFixed(2);
const num = (n, d = 4) => (n === null || n === undefined ? "—" : Number(n).toFixed(d));
const percentile = (sorted, p) => (sorted.length
  ? sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))] : NaN);

/** Everything the page shows, so `?report=1` and the panels read one object. */
const S = { web: null, profile: null, transfer: null, checks: [], questions: [],
            timings: [], decisions: [], shape: null, runtime: null, last: null };

const read = (name) => fetch(`${ARTIFACT}/${name}`).then((r) => {
  if (!r.ok) throw new Error(`${ARTIFACT}/${name}: HTTP ${r.status}`);
  return r.arrayBuffer().then((b) => new Uint8Array(b));
});

function banner(message, kind = "note") {
  const b = $("banner");
  b.className = `banner ${kind}`;
  b.textContent = message;
  b.hidden = false;
}

// --- the runtime, named rather than assumed -----------------------------------------
async function loadOrt() {
  const ort = await import("/node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs");
  ort.env.wasm.wasmPaths = "/node_modules/onnxruntime-web/dist/";
  // A page on a plain static server gets no COOP/COEP headers, so the threaded wasm build
  // cannot spawn workers and runs single-threaded. That is a property of the measurement,
  // not a bug in it, so it is read off the runtime and printed rather than guessed at.
  const isolated = self.crossOriginIsolated === true;
  // Read the thread count the runtime will actually use, rather than assume either the
  // default (which is undefined until the wasm boots) or `hardwareConcurrency` (which is
  // a wish when there is no worker pool).
  const threads = ort.env.wasm.numThreads
    ?? (isolated ? (navigator.hardwareConcurrency ?? 1) : 1);
  ort.env.wasm.numThreads = isolated ? threads : 1;
  return { ort, crossOriginIsolated: isolated, threads: ort.env.wasm.numThreads,
           version: ort.version ?? null,
           build: "ort.wasm.bundle.min.mjs + ort-wasm-simd-threaded.wasm" };
}

// --- panels ---------------------------------------------------------------------------
function renderProfile() {
  const m = S.profile;
  // `opset_actual` is per graph, because dynamo writes what it writes and the exporter
  // reads the number back off each file rather than restating the request.
  const opset = Object.values(m.opset_actual ?? {}).join("/");
  $("profile").replaceChildren(
    row("graphs", `${m.chunk}-token step + ${m.n_questions}×${m.q_len}-token branch, `
        + `scan tile ${m.scan_chunk}, opset ${opset}, ${m.exporter}`),
    row("state", `${m.n_layers} layers × ${m.n_heads} heads × ${m.d_k}×${m.d_v} = `
        + `${mib(m.state_bytes_total)} MiB, the same size whatever the document is`),
    row("head", `pointer probes, d_ptr ${m.d_ptr}, temperature ${m.temperature}, `
        + `abstention floor ${m.abstain_below ?? "none (uncalibrated checkpoint)"}`),
    row("tab peak", `${mib(m.peak_tile_bytes)} MiB — the widest attention tile this `
        + `profile can build`));
}

function renderBytes() {
  const tr = S.transfer;
  const t = $("bytes");
  t.replaceChildren(...Object.entries(tr.per_file).sort((a, b) => b[1] - a[1]).map(([f, n]) => {
    const li = el("li");
    li.append(el("span", "k", f), el("span", "v", `${mib(n)} MiB`));
    return li;
  }));
  const sum = el("li", "sum");
  sum.append(el("span", "k", "fetched total"), el("span", "v", `${mib(tr.total_bytes)} MiB`));
  t.append(sum);
  $("bytes-note").textContent = "weights.bin is fetched once and mounted into both "
    + `sessions. The wasm runtime is not in this table — it is ${S.runtime.build} at `
    + `${S.runtime.threads} thread(s), and G3 is about the model.`;
}

function renderWitness(r, exp) {
  $("witness").replaceChildren(...r.notes.concat(r.fails).map((c) => {
    const li = el("li", c.ok ? "ok" : "bad");
    li.append(el("span", "mark", c.ok ? "pass" : "fail"), el("span", "lbl", c.label),
              el("span", "det", c.detail));
    return li;
  }));
  $("witness-note").textContent = r.fails.length
    ? `${r.fails.length} of ${r.notes.length + r.fails.length} checks FAILED on this `
      + "artifact — the page is showing what the node gate would reject."
    : `${r.notes.length} checks passed against ${exp.ckpt}, the torch engine that wrote `
      + "the expectations file.";
}

function renderQuestions() {
  $("questions").replaceChildren(...S.questions.map((qst) => {
    const a = el("article", "q");
    a.id = `q-${qst.name}`;
    const head = el("header");
    head.append(el("h3", null, qst.name), el("span", "type", qst.type));
    a.append(head, el("p", "instr", qst.instruction), el("div", "answer"),
             el("details", "layout"));
    return a;
  }));
}

/** The token layout the branch receives, with each option's span bracketed and its probe
 *  probability at the span's end. This is the mechanism: no vocabulary readout, just
 *  pointers into the option spans, so the spans are what a reader should see. */
function renderLayout(qst, built, probs) {
  const web = S.web;
  const d = document.querySelector(`#q-${qst.name} .layout`);
  d.replaceChildren(el("summary", null, `question layout · ${built.ids.length} tokens into a `
    + `${web.meta.q_len}-wide graph · [DECIDE] at index ${built.decide}`));
  const line = el("div", "chips");
  const inSpan = (t) => built.spans.findIndex(([s, e]) => t >= s && t < e);
  built.ids.forEach((id, t) => {
    const k = inSpan(t);
    const cls = t === built.decide ? "chip decide"
      : id === web.optId ? "chip opt" : k >= 0 ? "chip tok" : "chip plain";
    line.append(el("span", cls, web.tok.idToToken(id) ?? `#${id}`));
    if (k >= 0 && t === built.spans[k][1] - 1) {
      line.append(el("span", "probe", `${qst.labels[k]} ${probs[k].toFixed(3)}`));
    }
  });
  d.append(line);
}

function answerBlock(qst, a) {
  const box = el("div", "answer");
  const commit = el("div", "commit");
  const verdict = qst.type === "choice" ? a.choice
    : qst.type === "score" ? `${a.level} · score ${num(a.score, 3)}`
      : `${a.noul >= 0.5 ? "Yes" : "No"} · p(Yes) ${num(a.noul, 4)}`;
  commit.append(el("span", a.abstain ? "verdict abstain" : "verdict",
                   a.abstain ? "abstain" : verdict));
  if (a.abstain) commit.append(el("span", "why", a.reason));
  else if (a.confidence !== null && a.confidence !== undefined) {
    commit.append(el("span", "why", `top p ${num(a.confidence, 4)}, `
      + `${num(a.margin, 4)} ahead of the runner-up`));
  }
  box.append(commit);
  const probs = qst.type === "noul" ? [["No", 1 - a.noul], ["Yes", a.noul]]
    : Object.entries(a.probabilities);
  const bars = el("ul", "bars");
  for (const [label, p] of probs.sort((x, y) => y[1] - x[1])) {
    const li = el("li");
    const bar = el("i");
    bar.style.setProperty("--w", `${(p * 100).toFixed(2)}%`);
    li.append(el("span", "lbl", label), bar, el("span", "n", p.toFixed(4)));
    if (a.abstain) li.classList.add("dim");
    bars.append(li);
  }
  box.append(bars);
  return box;
}

function renderScan(last) {
  $("scan").replaceChildren(
    row("document", `${last.tokens} tokens in ${last.calls.length} scan call(s)`),
    row("calls", last.calls.map((c) => `${c.tokens} tok / ${c.ms.toFixed(0)} ms`).join(" · ")),
    row("branch", `${S.questions.length} questions, one call, ${last.branchMs.toFixed(0)} ms`),
    row("end to end", `p50 ${percentile(S.timings, 0.5).toFixed(0)} ms of `
        + `${S.timings.length} runs (min ${S.timings[0].toFixed(0)}, `
        + `max ${S.timings[S.timings.length - 1].toFixed(0)})`));
}

// --- the decision ---------------------------------------------------------------------
async function decide(runs = 1) {
  const web = S.web;
  const text = $("document").value;
  const timings = [];
  let last = null;
  for (let i = 0; i < runs; i++) {
    last = await web.decide(text, S.questions);
    timings.push(+last.e2eMs.toFixed(1));
  }
  timings.sort((a, b) => a - b);
  S.timings = timings;
  S.last = last;

  for (const qst of S.questions) {
    const a = last.answers[qst.name];
    const article = document.querySelector(`#q-${qst.name}`);
    const layout = article.querySelector(".layout");
    article.replaceChildren(article.querySelector("header"),
                            article.querySelector(".instr"), answerBlock(qst, a), layout);
    const built = web.tok.buildQuestion(qst.instruction, qst.labels);
    const probs = qst.type === "noul" ? [1 - a.noul, a.noul]
      : qst.labels.map((l) => a.probabilities[l]);
    renderLayout(qst, built, probs);
  }
  S.decisions = S.questions.map((qst) => {
    const a = last.answers[qst.name];
    return { name: qst.name, type: qst.type, abstain: a.abstain === true,
             verdict: qst.type === "choice" ? a.choice
               : qst.type === "score" ? a.level : (a.noul >= 0.5 ? "Yes" : "No"),
             confidence: a.confidence ?? null, margin: a.margin ?? null };
  });
  renderScan(last);
}

/** Time the graph at a request *width* without claiming parity at it: the option list is
 *  padded with locally written labels until the question reaches `tokens`, which is the
 *  shape `banking77/intent` has (1,067 tokens, 77 options). No upstream text is involved,
 *  and the numbers are latency only — the parity figure for that shape is in
 *  `runs/onnx_parity_widest.json`, measured in Python. */
async function shapeProbe(tokens) {
  const web = S.web;
  const base = S.questions.find((x) => x.type === "choice");
  const labels = base.labels.slice();
  let built = web.tok.buildQuestion(base.instruction, labels);
  while (built.ids.length < tokens) {
    labels.push(`candidate answer number ${labels.length + 1}`);
    built = web.tok.buildQuestion(base.instruction, labels);
  }
  const runs = [];
  for (let i = 0; i < 3; i++) {
    const t0 = performance.now();
    const obs = await web.observe($("document").value);
    await web.ask(obs, [{ ...base, labels }]);
    runs.push(+(performance.now() - t0).toFixed(1));
  }
  S.shape = { target_tokens: tokens, options: labels.length,
              question_tokens: built.ids.length, q_len: web.meta.q_len,
              fits: built.ids.length <= web.meta.q_len, run_ms: runs };
  const li = row("shape probe", `${built.ids.length} tokens × ${labels.length} options in a `
    + `${web.meta.q_len}-wide graph: ${runs.join(" / ")} ms`);
  li.classList.add("probe");
  $("scan").append(li);
}

// --- the machine-readable block -------------------------------------------------------
/** The tab's own estimate of what it is holding, when the browser will say.
 *
 * `measureUserAgentSpecificMemory` exists only on a cross-origin-isolated page (COOP +
 * COEP), which is the same condition the threaded wasm build needs — so a plain static
 * server gets `null` here and the report says so rather than substituting a guess.
 */
async function tabMemory() {
  if (!performance.measureUserAgentSpecificMemory) {
    return { available: false, reason: "API absent in this browser" };
  }
  if (!self.crossOriginIsolated) {
    return { available: false, reason: "page is not crossOriginIsolated (needs COOP+COEP)" };
  }
  try {
    const m = await performance.measureUserAgentSpecificMemory();
    return { available: true, bytes: Math.round(m.bytes),
             // The name is the caveat: this counts the tab's *JS-managed* memory, and the
             // weights handed to onnxruntime live in the wasm heap, which it does not
             // attribute here. So this figure is printed next to the byte it does not
             // include, never instead of it.
             note: "JS-managed memory only; the mounted weights are in the wasm heap",
             breakdown: (m.breakdown || []).map((b) => ({ bytes: Math.round(b.bytes),
                                                          type: b.attribution })) };
  } catch (e) {
    return { available: false, reason: `${e.name}: ${e.message}`, note: null };
  }
}

async function renderReport() {
  const m = S.profile;
  const payload = {
    artifact: ARTIFACT, expected: EXPECTED, ua: navigator.userAgent,
    crossOriginIsolated: S.runtime.crossOriginIsolated, ort_web: S.runtime.version,
    wasm_build: S.runtime.build, wasm_threads: S.runtime.threads,
    profile: { chunk: m.chunk, q_len: m.q_len, n_questions: m.n_questions,
               scan_chunk: m.scan_chunk, n_layers: m.n_layers, d_model: m.d_model,
               d_ptr: m.d_ptr, temperature: m.temperature,
               abstain_below: m.abstain_below, opset: m.opset_actual,
               exporter: m.exporter, peak_tile_bytes: m.peak_tile_bytes,
               state_bytes_total: m.state_bytes_total },
    transfer: S.transfer,
    tab_memory: await tabMemory(),
    peak_tile_mib: +(m.peak_tile_bytes / 2 ** 20).toFixed(2),
    cold_load_ms: S.transfer.cold_ms,
    decision: { tokens: S.last.tokens, calls: S.last.calls.length,
                scan_ms: +S.last.scanMs.toFixed(1),
                branch_ms: +S.last.branchMs.toFixed(1),
                e2e_ms: S.timings, p50_ms: +percentile(S.timings, 0.5).toFixed(1),
                runs: S.timings.length },
    decisions: S.decisions,
    checks: S.checks.map((c) => ({ ok: c.ok, label: c.label, detail: c.detail })),
    passed: S.checks.filter((c) => c.ok).length,
    failed: S.checks.filter((c) => !c.ok).length,
    shape_probe: S.shape,
  };
  $("report").textContent = JSON.stringify(payload, null, 1);
  $("report-panel").hidden = !REPORT;
  document.title = `${payload.failed ? "FAILED" : "PASS"} · myna ${ARTIFACT}`;
  return payload;
}

// --- boot -----------------------------------------------------------------------------
async function boot() {
  const rt = await loadOrt();
  S.runtime = rt;
  $("phase").textContent = `downloading ${ARTIFACT}`;
  const t0 = performance.now();
  S.web = await MynaWeb.load({ ort: rt.ort, read });
  S.transfer = { ...S.web.transfer(), cold_ms: +(performance.now() - t0).toFixed(1),
                 // download and compile are different costs, and a single "cold load"
                 // number hides which one a slower machine pays.
                 session_create_ms: +S.web.loadMs.toFixed(1),
                 mount: "weights.bin fetched once, mounted into both sessions" };
  S.profile = S.web.meta;
  $("threads").textContent = `onnxruntime-web${rt.version ? ` ${rt.version}` : ""} wasm on `
    + `${rt.threads} thread(s)`
    + (rt.crossOriginIsolated ? "" : " (no COOP/COEP here, so no worker pool)");
  renderProfile();
  renderBytes();

  $("phase").textContent = "running the gate";
  const expRes = await fetch(EXPECTED);
  if (!expRes.ok) throw new Error(`${EXPECTED}: HTTP ${expRes.status}`);
  const exp = await expRes.json();
  const r = await runChecks(S.web, exp);
  S.checks = r.notes.concat(r.fails);
  S.questions = exp.questions;
  renderWitness(r, exp);
  renderQuestions();
  $("document").value = exp.document;
  $("ask").disabled = false;

  await decide(RUNS);
  if (SHAPE) await shapeProbe(SHAPE);
  await renderReport();
  // `ready` is the harness's cue to read the report, so it cannot be set before the report
  // exists — an early one once made the harness parse an empty <pre>.
  $("phase").textContent = "ready";
}

$("ask").addEventListener("click", () => {
  $("ask").disabled = true;
  decide(1).then(() => renderReport()).catch((e) => banner(`${e.name}: ${e.message}`, "bad"))
    .finally(() => { $("ask").disabled = false; });
});
$("report-toggle").addEventListener("change", (e) => {
  $("report-panel").hidden = !e.target.checked;
});

boot().catch((e) => {
  banner(`${e.name}: ${e.message}`, "bad");
  $("phase").textContent = "stopped";
}).finally(() => {
  $("booting").hidden = true;
  if (REPORT) $("report").textContent ||= JSON.stringify({ fatal: "boot failed" }, null, 1);
});

/** The browser artifact's gate, runnable without a browser.
 *
 *   node browser/selftest.mjs [--artifact runs/onnx] [--expected browser/expected.json]
 *
 * `myna.onnx_export.parity` checks the exported graphs against torch *in Python*. That
 * proves the graphs are right; it does not prove the page is right. The page adds a
 * JavaScript BPE, a hand-written pointer head, a chained scan in the graph's input
 * layout, and one mounted weight file shared by two sessions — four places a wrong
 * number can appear while every file loads. So this runs the whole JavaScript path under
 * node, against `browser/expected.json`, which `bench/browser_expected.py` writes from
 * the torch engine.
 *
 * The checks live in `browser/parity.mjs`, which `browser/app.js` imports too, so the
 * green lines a user sees in the tab are produced by the same code this gate fails on.
 * What is only measured in the tab is fetch bytes, cold-load and per-decision latency,
 * which `?report=1` prints there.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { MynaWeb } from "./myna.js";
import { runChecks } from "./parity.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));

function arg(name, dflt) {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : dflt;
}

const artifact = path.resolve(arg("artifact", "runs/onnx"));
const expectedPath = path.resolve(arg("expected", path.join(here, "expected.json")));
const repeats = Number(arg("repeats", 12));

const ort = await import(path.join(here, "../node_modules/onnxruntime-web/dist/ort.node.min.mjs"));

const exp = JSON.parse(fs.readFileSync(expectedPath, "utf8"));
const P = exp.profile;

const web = await MynaWeb.load({ ort, read: (n) => fs.readFileSync(path.join(artifact, n)) });

// Every file the artifact directory holds, so the byte check compares the fetch against
// the disk rather than against a number the writer of this file remembered.
const onDisk = Object.fromEntries(fs.readdirSync(artifact)
  .map((f) => [f, fs.statSync(path.join(artifact, f)).size]));

const r = await runChecks(web, exp, { onDisk });

const timings = [];
for (let i = 0; i < repeats; i++) timings.push((await web.decide(exp.document, exp.questions)).e2eMs);
timings.sort((a, b) => a - b);
const p50 = timings[Math.floor(timings.length / 2)];
const last = await web.decide(exp.document, exp.questions);

console.log(`\nprofile      ${artifact}`);
console.log(`             chunk ${P.chunk} x tile ${P.scan_chunk}, ${P.n_questions} questions `
          + `x ${P.q_len} tokens, ${P.n_layers} layers, temperature ${P.temperature}`);
console.log(`cold load    ${web.loadMs.toFixed(0)} ms  (both graphs, one weights.bin mount)`);
console.log(`transfer     ${(r.transfer.total_bytes / 2 ** 20).toFixed(2)} MiB across `
          + `${Object.keys(r.transfer.per_file).length} files`);
console.log(`             ${Object.entries(r.transfer.per_file)
  .map(([f, n]) => `${f} ${(n / 2 ** 20).toFixed(2)}`).join("  ")}`);
console.log(`decision     ${last.tokens} tokens, ${last.calls.length} scan calls, `
          + `p50 ${p50.toFixed(1)} ms over ${repeats} runs on this box`);
console.log(`             scan ${last.scanMs.toFixed(1)} ms + branch ${last.branchMs.toFixed(1)} ms`);

for (const n of r.notes) console.log(`ok    ${n.label}${n.detail ? `: ${n.detail}` : ""}`);
for (const n of r.fails) console.log(`FAIL  ${n.label}${n.detail ? `: ${n.detail}` : ""}`);
if (exp.questions_skipped_as_too_wide.length) {
  console.log(`\nnot asked (over --q-len ${P.q_len}): `
    + exp.questions_skipped_as_too_wide.map((q) => `${q.name} ${q.tokens}`).join(", "));
}
console.log(r.fails.length ? `\n${r.fails.length} check(s) FAILED` : `\n${r.notes.length} checks passed`);
process.exit(r.fails.length ? 1 : 0);

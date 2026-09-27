/** The artifact's self-witness, written once and run in two places.
 *
 * `browser/selftest.mjs` runs this under node as the committed gate; `browser/app.js`
 * runs it in the tab and prints the same lines. That is the point of factoring it out —
 * a page whose green checkmark comes from a different piece of code than the test suite
 * is a page that lies on demand.
 *
 * Every check compares the JavaScript path against `browser/expected.json`, which
 * `bench/browser_expected.py` writes from `myna.engine.Myna`. So the sentence being
 * tested is "the thing a user downloads answers what the thing on the training machine
 * answered", which is the sentence G3 is made of.
 *
 * The bytes check needs the filesystem, so it runs only under the node gate: a tab has no
 * directory to compare against and reports what it fetched instead, which `?report=1`
 * prints and the network log is the witness for.
 */

import { abstainCheck, readout } from "./head.js";

/** Per-layer summary statistics — the same five numbers `state_fingerprint` writes. */
export function fingerprint(tensor) {
  const f = tensor.data;
  let mean = 0, max = -Infinity, min = Infinity, l2 = 0;
  for (const v of f) { mean += v; if (v > max) max = v; if (v < min) min = v; l2 += v * v; }
  mean /= f.length;
  return { n: f.length, mean, max, min, l2: Math.sqrt(l2), tail: Array.from(f.slice(-8)) };
}

export async function runChecks(web, exp, { maxStateRel = 1e-4, maxProbAbs = 1e-4,
                                            maxGateAbs = 1e-6, onDisk = null } = {}) {
  const notes = [], fails = [];
  const check = (label, ok, detail) => {
    (ok ? notes : fails).push({ label, ok, detail: detail ?? "" });
  };
  const ex = (v) => Number(v).toExponential(2);

  // 1. the tokenizer, string by string
  const firstMismatch = [];
  let idMismatch = 0, tokensCompared = 0;
  for (const { text, ids } of exp.tokenizer.strings) {
    const got = web.tok.encode(text);
    tokensCompared += ids.length;
    if (got.length !== ids.length || got.some((v, i) => v !== ids[i])) {
      idMismatch++;
      if (firstMismatch.length < 3) {
        firstMismatch.push(`${JSON.stringify(text.slice(0, 40))}: python ${ids.slice(0, 12)} `
                         + `js ${got.slice(0, 12)}`);
      }
    }
  }
  check(`${exp.tokenizer.strings.length} strings / ${tokensCompared} tokens tokenized identically`,
        idMismatch === 0, idMismatch ? `${idMismatch} strings differ — ${firstMismatch[0]}` : "");

  // 2. the question layout: ids, option spans, and the [DECIDE] index
  const layout = new Map(exp.tokenizer.questions.map((w) => [w.name, w]));
  let qMismatch = 0;
  for (const q of exp.questions) {
    const want = layout.get(q.name);
    const got = web.tok.buildQuestion(q.instruction, q.labels);
    if (JSON.stringify(got.ids) !== JSON.stringify(want.ids)
        || JSON.stringify(got.spans) !== JSON.stringify(want.spans)
        || got.decide !== want.decide) {
      qMismatch++;
    }
  }
  check(`${exp.tokenizer.questions.length} question layouts identical`, qMismatch === 0,
        qMismatch ? `${qMismatch} layouts differ` : "");

  // 3. the chained scan, against the state fingerprint. Relative to each layer's own
  //    scale, which is `onnx_export._rel`'s convention: the state's entries reach ~3e2,
  //    so an absolute 1e-4 there would ask two runtimes to agree bitwise — and dividing a
  //    tail difference by that tail element, which can be ~0, would fail a good artifact.
  const obs = await web.observe(exp.document);
  let stateWorst = 0;
  obs.S.forEach((t, i) => {
    const got = fingerprint(t), want = exp.state[i];
    if (got.n !== want.n) throw new Error(`layer ${i}: ${got.n} entries, expected ${want.n}`);
    const scale = Math.max(Math.abs(want.max), Math.abs(want.min), 1e-12);
    for (const k of ["mean", "max", "min", "l2"]) {
      stateWorst = Math.max(stateWorst, Math.abs(got[k] - want[k]) / scale);
    }
    for (let j = 0; j < 8; j++) {
      stateWorst = Math.max(stateWorst, Math.abs(got.tail[j] - want.tail[j]) / scale);
    }
  });
  check(`chained state over ${obs.tokens} tokens in ${obs.calls.length} calls matches `
        + `${exp.state.length} layer fingerprints (the engine's own chain: ${exp.document_calls} calls)`,
        stateWorst <= maxStateRel && obs.calls.length === exp.document_calls,
        `worst relative ${ex(stateWorst)} against ${maxStateRel}`);

  // 3b. an empty document, which is the one input the chain cannot answer by accident. The
  //     loop's guard is `fed < ids.length || fed === 0`, so the `break` inside it is what
  //     keeps a pasted-empty-textarea request from calling the graph forever.
  const empty = await web.observe("");
  check("an empty document costs one padded scan call rather than hanging the tab",
        empty.calls.length === 1 && empty.tokens === 0,
        `${empty.calls.length} calls for ${empty.tokens} tokens`);

  // 4. the decisions themselves
  const { answers } = await web.ask(obs, exp.questions);
  let probWorst = 0;
  const labelMismatch = [];
  for (const q of exp.questions) {
    const want = exp.engine_answers[q.name], got = answers[q.name];
    if (!want?.probabilities) continue;
    for (const k of Object.keys(want.probabilities)) {
      probWorst = Math.max(probWorst, Math.abs(want.probabilities[k] - (got.probabilities[k] ?? NaN)));
    }
    const pick = (a) => a.choice ?? a.level ?? (a.noul >= 0.5 ? "Yes" : "No");
    if (pick(want) !== pick(got)) labelMismatch.push(`${q.name}: ${pick(want)} vs ${pick(got)}`);
  }
  check(`${Object.keys(exp.engine_answers).length} decisions agree with the engine`,
        probWorst <= maxProbAbs && labelMismatch.length === 0,
        `worst |Δp| ${ex(probWorst)} against ${maxProbAbs}`
        + (labelMismatch.length ? ` — ${labelMismatch.join("; ")}` : ""));

  // 5. the abstention gate, on the engine's own probabilities. The artifact's floor is
  //    null — v0 was never calibrated — so this is the only place the JS gate runs at all,
  //    and it runs at three floors: one where nothing abstains, one that drops the model's
  //    two softest commitments, one that drops all three. A type whose commit branch never
  //    runs has its answer-shaped numbers (`score`'s expectation, `choice`'s label) checked
  //    against `null` on both sides, which passes for the wrong reason — so the coverage
  //    below is asserted, not assumed.
  const ab = exp.abstention;
  const gateBad = [];
  // 1e-6 rather than 1e-9 because one case rebuilds its vector from the single number the
  // engine publishes: `noul` answers carry p(Yes) only — the Brier score is defined on that
  // — so the complement is float64 `1 - p` against float32 `p_no`, which drifts by ~4e-8
  // here. Every mutation this section catches moves a number by 1e-2 or more.
  const near = (w, g) => (typeof w === "number" ? Math.abs(w - (g ?? NaN)) <= maxGateAbs : w === g);
  for (const c of ab.cases) {
    const got = readout(c.type, c.labels, c.probs, c.threshold);
    const keys = ["abstain", "confidence", "margin"].concat(
      c.type === "choice" ? ["choice"]
        : c.type === "score" ? ["score", "level"] : ["noul", "yes"]);
    for (const k of keys) {
      if (!near(c.want[k], got[k])) {
        gateBad.push(`${c.name}@${c.threshold}.${k}: engine ${JSON.stringify(c.want[k])} `
                   + `js ${JSON.stringify(got[k])}`);
      }
    }
  }
  for (const c of ab.gate_cases) {
    const got = abstainCheck(c.probs, c.threshold, c.labels);
    for (const k of ["abstain", "confidence", "margin"]) {
      if (!near(c.want[k], got[k])) {
        gateBad.push(`gate@${c.threshold}.${k}: engine ${JSON.stringify(c.want[k])} `
                   + `js ${JSON.stringify(got[k])}`);
      }
    }
  }
  // `reason` is display text and formats in two languages; the three numbers above are
  // what a caller branches on, so those are what is gated.
  const committed = new Set(ab.cases.filter((c) => !c.want.abstain).map((c) => c.type));
  const uncovered = ["choice", "score", "noul"].filter((t) => !committed.has(t));
  check(`abstention: ${ab.cases.length} answers at the ${ab.thresholds.join("/")} floors and `
        + `${ab.gate_cases.length} gate cases match the engine `
        + `(${ab.cases.filter((c) => c.want.abstain).length} abstain, `
        + `${ab.cases.filter((c) => !c.want.abstain).length} commit)`,
        gateBad.length === 0 && uncovered.length === 0,
        gateBad.slice(0, 4).join("; ")
        + (uncovered.length ? `no committed ${uncovered.join("/")} case: its commit-branch `
                           + `arithmetic is never compared` : ""));

  // 6. the bytes. Only the node gate can compare against the directory — a tab has no
  //    filesystem, so the page reports what it fetched and the network log is the check.
  //    Two properties, not one: that each file arrived as it sits on disk, and that the
  //    printed *total* is those files summed — a total that quietly skips `weights.bin`
  //    would otherwise still print a plausible MiB figure and pass the per-file test.
  const tr = web.transfer();
  if (onDisk) {
    const diffs = Object.entries(tr.per_file).filter(([f, n]) => onDisk[f] !== n);
    const diskTotal = Object.values(onDisk).reduce((a, b) => a + b, 0);
    const totalBad = tr.total_bytes !== diskTotal
      ? `printed ${(tr.total_bytes / 2 ** 20).toFixed(2)} MiB, directory is `
        + `${(diskTotal / 2 ** 20).toFixed(2)} MiB` : "";
    check(`every fetched file is the file on disk, and the total is their sum `
          + `(${(tr.total_bytes / 2 ** 20).toFixed(2)} MiB across ${Object.keys(tr.per_file).length} files)`,
          diffs.length === 0 && !totalBad, [totalBad, ...diffs.map(([f, n]) => `${f} fetched ${n}`)]
          .filter(Boolean).join("; "));
  }

  return { notes, fails, stateWorst, probWorst, transfer: tr,
           tokens: obs.tokens, calls: obs.calls.length };
}

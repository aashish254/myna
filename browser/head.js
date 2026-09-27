/** The pointer head, run from `head.bin` — the last half of the artifact's answer.
 *
 * `myna.onnx_export.export_head` writes wq/wk and their biases as raw little-endian
 * float32 in the field order recorded in `meta.json`, and `pointer_probs` in the same
 * module is the numpy mirror of `Myna._readout`. This file is the third copy of that
 * arithmetic, and the one a user runs, so it is checked against the other two by
 * `browser/expected.json`: the same hidden states in, the same probabilities out.
 *
 * The head stayed outside the ONNX graphs on purpose. Its input is the mean of each
 * option's token span, and the spans are a property of the text the page already has —
 * putting it in the graph would mean re-running a scan every time an option list changed.
 */

function assertLittleEndian() {
  const probe = new Uint8Array(new Uint32Array([0x01020304]).buffer);
  if (probe[0] !== 4) {
    throw new Error("this tab is big-endian, and head.bin is little-endian float32");
  }
}

/** {field: {shape, data}} for every field `meta.head.head_fields` names. */
export function readHead(meta, buffer) {
  assertLittleEndian();
  const view = new DataView(buffer);
  const fields = {};
  for (const name of meta.head.head_fields) {
    const lay = meta.head.head_layout[name];
    if (!lay) throw new Error(`meta.json does not describe head field ${name}`);
    const end = lay.offset + lay.bytes;
    if (end > view.byteLength) {
      throw new Error(`${name} claims bytes ${lay.offset}..${end} of a ${view.byteLength}-byte blob`);
    }
    if (lay.bytes % 4) throw new Error(`${name} is ${lay.bytes} bytes, not a whole float32`);
    fields[name] = { shape: lay.shape, offset: lay.offset,
                     data: new Float32Array(buffer, lay.offset, lay.bytes / 4) };
  }
  return fields;
}

function matvec(field, bias, x, out) {
  const [rows, cols] = field.shape;
  const w = field.data;
  for (let r = 0; r < rows; r++) {
    let acc = bias.data[r];
    const base = r * cols;
    for (let c = 0; c < cols; c++) acc += w[base + c] * x[c];
    out[r] = acc;
  }
}

function softmax(logits) {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  const e = logits.map((v) => Math.exp(v - max));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

/** Probabilities over `spans`, from `h` — a [nTokens, d_model] row-major Float32Array.
 *  Same arithmetic as `Myna._readout`, read off the shipped blob rather than a model. */
export function pointerProbs(head, dPtr, temperature, h, spans, decide, dModel) {
  const q = new Float32Array(dPtr);
  matvec(head.wq_weight, head.wq_bias, h.subarray(decide * dModel, (decide + 1) * dModel), q);
  const keys = new Float32Array(spans.length * dPtr);
  const pooled = new Float32Array(dModel);
  const k = new Float32Array(dPtr);
  spans.forEach(([s, e], o) => {
    pooled.fill(0);
    for (let t = s; t < e; t++) {
      const row = h.subarray(t * dModel, (t + 1) * dModel);
      for (let c = 0; c < dModel; c++) pooled[c] += row[c];
    }
    for (let c = 0; c < dModel; c++) pooled[c] /= (e - s);
    matvec(head.wk_weight, head.wk_bias, pooled, k);
    keys.set(k, o * dPtr);
  });
  const logits = [];
  for (let o = 0; o < spans.length; o++) {
    let acc = 0;
    for (let j = 0; j < dPtr; j++) acc += keys[o * dPtr + j] * q[j];
    logits.push(acc / Math.sqrt(dPtr) / temperature);
  }
  return softmax(logits);
}

/** Mirror of `myna.engine.abstain_check`. Threshold-only, so it cannot see an answer. */
export function abstainCheck(probs, threshold, labels) {
  if (threshold === null || threshold === undefined) {
    return { abstain: false, confidence: null, margin: null, reason: null };
  }
  if (probs.length !== labels.length || !probs.length) {
    throw new Error(`${probs.length} probabilities for ${labels.length} labels`);
  }
  const ordered = probs.slice().sort((a, b) => b - a);
  const margin = ordered[0] - (ordered.length > 1 ? ordered[1] : 0);
  let i = 0;
  for (let k = 1; k < probs.length; k++) if (probs[k] > probs[i]) i = k;
  const conf = probs[i];
  return { abstain: conf < threshold, confidence: conf, margin,
           reason: `${labels[i]} at p=${conf.toFixed(3)}, ${margin.toFixed(3)} ahead of `
                 + `the runner-up, under the ${threshold.toFixed(3)} floor` };
}

/** The typed answer, in the three shapes `Myna._readout` produces — same keys, so the
 *  page's JSON can be compared field by field against the engine's. */
export function readout(type, labels, probs, threshold) {
  const gate = abstainCheck(probs, threshold, labels);
  let i = 0;
  for (let k = 1; k < probs.length; k++) if (probs[k] > probs[i]) i = k;
  const probabilities = Object.fromEntries(labels.map((l, j) => [l, probs[j]]));
  const common = { type, confidence: gate.confidence, abstain: gate.abstain,
                   reason: gate.reason, margin: gate.margin };
  if (type === "choice") {
    return { ...common, choice: gate.abstain ? null : labels[i],
             confidence: probs[i], probabilities };
  }
  if (type === "score") {
    return { ...common,
             score: gate.abstain ? null : probs.reduce((acc, p, j) => acc + j * p, 0),
             level: gate.abstain ? null : labels[i], probabilities,
             legend: Object.fromEntries(labels.map((l, j) => [String(j), l])) };
  }
  if (type !== "noul") throw new Error(`unknown question type ${type}`);
  // noul keeps reporting the raw p(Yes): the Brier score is defined on it.
  return { ...common, noul: probs[1], yes: gate.abstain ? null : probs[1] >= 0.5 };
}

/** The whole engine, in the tab: bytes in, typed decision out.
 *
 * `myna.engine.Myna` does this in Python with torch. This file does it with two ONNX
 * graphs, one mounted weight file, the raw pointer head from `head.bin` and the BPE from
 * `tokenizer.json` — and nothing else, which is the point of P6: no server, no per-request
 * API cost, no KV cache that grows with the document.
 *
 * The two graphs are the shape 6a chose and 6b kept: `state_step.onnx` consumes a fixed
 * 256-token chunk plus the state stack and returns the next state, so a longer document is
 * more *calls*; `question.onnx` resumes from that state and encodes a fixed-width batch of
 * question branches. Both graphs list their weights as external data pointing at
 * `weights.bin`, and onnxruntime-web cannot resolve that name over HTTP — it only accepts a
 * mounted buffer. So `read("weights.bin")` happens once here and the same bytes are handed
 * to both sessions, which is the browser-side half of the sharing the byte budget claims.
 */

import { ByteLevelBPE } from "./bpe.js";
import { readHead, pointerProbs, readout } from "./head.js";

export class MynaWeb {
  /** @param read   name -> bytes, for meta.json / tokenizer.json / *.onnx / weights.bin / head.bin
   *  @param ort    the onnxruntime-web module (imported by the caller so this file runs
   *                unchanged under `node` and in a tab) */
  static async load({ read, ort, profile = "default" }) {
    const bytes = new Map();
    const count = async (name) => {
      const buf = await read(name);
      bytes.set(name, buf.byteLength ?? buf.length);
      return buf;
    };
    const json = async (name) => JSON.parse(new TextDecoder().decode(await count(name)));
    const binary = async (name) => tight(await count(name));
    const meta = await json("meta.json");
    const t0 = now();
    const weights = await binary("weights.bin");
    const opts = { executionProviders: ["cpu"], graphOptimizationLevel: "disabled",
                   externalData: [{ path: "weights.bin", data: weights }] };
    const step = await ort.InferenceSession.create(await binary("state_step.onnx"), opts);
    const question = await ort.InferenceSession.create(await binary("question.onnx"), opts);
    const loadMs = elapsed(t0);
    const tok = ByteLevelBPE.fromJSON(await json("tokenizer.json"));
    const head = readHead(meta, (await binary("head.bin")).buffer);
    return new MynaWeb({ meta, step, question, tok, head, ort, bytes, loadMs, profile });
  }

  constructor({ meta, step, question, tok, head, ort, bytes, loadMs, profile }) {
    this.meta = meta; this.step = step; this.question = question;
    this.tok = tok; this.head = head; this.ort = ort;
    this.fetchedBytes = bytes; this.loadMs = loadMs; this.profile = profile;
    this.L = meta.n_layers;
    this.stateShape = [1, meta.n_heads, meta.d_k, meta.d_v];
    this.stateElems = meta.n_heads * meta.d_k * meta.d_v;
    this.optId = tok.vocab.get("[OPT]");
    this.decId = tok.vocab.get("[DECIDE]");
  }

  /** Total bytes the artifact itself moved over the wire, per file and all together. */
  transfer() {
    const per = Object.fromEntries(this.fetchedBytes);
    const total = [...this.fetchedBytes.values()].reduce((a, b) => a + b, 0);
    return { per_file: per, total_bytes: total,
             // the runtime is not the model, and a table that adds them is a lie of
             // omission: the tab pays for both, so both are printed, apart.
             runtime_excluded: true };
  }

  zeroState() {
    return Array.from({ length: this.L },
      () => new this.ort.Tensor("float32", new Float32Array(this.stateElems), this.stateShape));
  }

  /** Tokenize and scan a document into the fixed-size state. One graph call per 256 tokens. */
  async observe(text) {
    const ids = this.tok.encode(text);
    return this.observeIds(ids);
  }

  async observeIds(ids) {
    const chunk = this.meta.chunk;
    const S = this.zeroState();
    const calls = [];
    for (let fed = 0; fed < ids.length || fed === 0; fed += chunk) {
      const take = ids.slice(fed, fed + chunk);
      const t0 = now();
      const idsT = new BigInt64Array(chunk); idsT.fill(0n);
      take.forEach((v, i) => { idsT[i] = BigInt(v); });
      const mask = new Float32Array(chunk); mask.fill(1, 0, take.length);
      const feeds = { ids: new this.ort.Tensor("int64", idsT, [1, chunk]),
                      mask: new this.ort.Tensor("float32", mask, [1, chunk]),
                      pos_offset: new this.ort.Tensor("int64", BigInt64Array.of(BigInt(fed)), []),
                      ...Object.fromEntries(S.map((s, i) => [`S_in_${i}`, s])) };
      const out = await this.step.run(feeds);
      for (let i = 0; i < this.L; i++) S[i] = out[`S_out_${i}`];
      calls.push({ tokens: take.length, ms: elapsed(t0) });
      if (take.length < chunk) break;
    }
    return { ids, S, calls, tokens: ids.length, scanMs: calls.reduce((a, c) => a + c.ms, 0) };
  }

  /** One decision per question, from a cached observation.
   *  @param questions [{name, type, instruction, labels}] */
  async ask(obs, questions) {
    const N = this.meta.n_questions, Lq = this.meta.q_len;
    if (questions.length > N) {
      throw new Error(`${questions.length} questions exceed the exported width ${N} `
                    + `(re-export with a larger --questions, or split the request)`);
    }
    const built = questions.map((q) => this.tok.buildQuestion(q.instruction, q.labels));
    const tooWide = built.findIndex((b) => b.ids.length > Lq);
    if (tooWide >= 0) {
      throw new Error(`question ${questions[tooWide].name} is ${built[tooWide].ids.length} `
                    + `tokens, over the exported --q-len ${Lq} (use the wide profile)`);
    }
    const qIds = new BigInt64Array(N * Lq);   // 0 is [PAD]
    const qMask = new Float32Array(N * Lq);
    built.forEach((b, m) => {
      b.ids.forEach((v, i) => { qIds[m * Lq + i] = BigInt(v); qMask[m * Lq + i] = 1; });
    });
    const t0 = now();
    const feeds = { q_ids: new this.ort.Tensor("int64", qIds, [N, Lq]),
                    q_mask: new this.ort.Tensor("float32", qMask, [N, Lq]),
                    pos_offset: new this.ort.Tensor("int64", BigInt64Array.of(BigInt(obs.tokens)), []),
                    ...Object.fromEntries(obs.S.map((s, i) => [`S_in_${i}`, s])) };
    const hq = (await this.question.run(feeds)).h_q;
    const branchMs = elapsed(t0);
    const answers = {};
    const dModel = this.meta.d_model, dPtr = this.meta.d_ptr;
    built.forEach((b, m) => {
      const row = hq.data.subarray(m * Lq * dModel, (m + 1) * Lq * dModel);
      const probs = pointerProbs(this.head, dPtr, this.meta.temperature, row, b.spans,
                                 b.decide, dModel);
      answers[questions[m].name] = readout(questions[m].type, questions[m].labels, probs,
                                           this.meta.abstain_below ?? null);
    });
    return { answers, branchMs, tokensPerQuestion: built.map((b) => b.ids.length) };
  }

  /** observe + ask, which is the unit G3 counts: one decision, end to end. */
  async decide(text, questions) {
    const t0 = now();
    const obs = await this.observe(text);
    const res = await this.ask(obs, questions);
    return { ...res, tokens: obs.tokens, calls: obs.calls, scanMs: obs.scanMs,
             branchMs: res.branchMs, e2eMs: elapsed(t0), loadMs: this.loadMs };
  }
}

function now() {
  return globalThis.performance?.now ? globalThis.performance.now() : Date.now();
}

function elapsed(t0) {
  return now() - t0;
}

/** A view over an ArrayBuffer of exactly its own bytes.
 *
 * Node hands out `Buffer`s from a shared pool, so `buf.buffer` can be 8 KiB of memory
 * with the file sitting at some offset inside it — and `head.bin` is read through
 * `new Float32Array(buffer, offset, length)`, which would then be indexing the pool. In
 * a tab `fetch().arrayBuffer()` is already tight, so this copies only when it has to.
 */
function tight(b) {
  const v = b instanceof Uint8Array ? b : new Uint8Array(b);
  return v.byteOffset === 0 && v.buffer.byteLength === v.byteLength
    ? v : new Uint8Array(v);
}

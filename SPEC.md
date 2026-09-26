# Myna — Master Spec

**Status:** authoritative. Supersedes [V2_SPEC.md](V2_SPEC.md) and [PLAN.md](PLAN.md) wherever
their numbers conflict (see [§9 Corrections log](#9-corrections-log-things-we-previously-stated-wrong)).
Every quantity below is tagged **measured** (reproducible from a committed artifact),
**projected** (our estimate, falsifiable by a named experiment), or **gated** (needs access or
approval we do not have yet).

---

## 1. Goal

Build **myna**: an original System-1 typed-decision model that answers `choice` / `score` / `noul`
questions over a growing observation in **one pass, at constant cost**, on hardware people
actually own — a laptop, a phone tab, a browser.

The one-sentence pitch:

> A decision over a 16k-token document, in ~30 ms, in 16 MB of weights, with a probability you
> can trust, on a machine with no network.

### Why this is the right product
The incumbent class of decision models (laya, and decoder-LM routers like kev) re-read the whole
observation per request and operate inside a fixed attention window. That makes three real
workloads expensive or impossible: agent loops that re-decide every turn, documents longer than
the window, and on-device/browser deployment. Myna's architecture removes all three constraints at
once: gated linear attention carries a **fixed-size matrix state** that is exactly continuable, so
the observation is scanned once and then only appended to.

### The bar we hold ourselves to
Fast **and** accurate **and** deployable. Speed alone is not a contribution: a model that is quick
and wrong is worse than slow and wrong, because it is cheap to deploy at scale. Accuracy is a gate
to clear, not a gap to reframe around. Where we lose, we print the loss.

---

## 2. What "done" looks like — target results

### 2.1 Headline table (the artifact the project ships)

| axis | laya typed-decisions | myna target | label |
|---|---|---|---|
| parameters | 421.29M (enc 394.78M + head 26.51M) — *their published figure* | **15.35M measured**, ≤ 32M allowed | measured |
| accuracy, kev decision-v2 **test** | **0.6319** (our own seeded witness, n=546) | **≥ 0.70** gate · 0.75 stretch | projected |
| accuracy, their own typed-decisions bench | 0.766 (2,000 decisions) | not comparable — different suite; do not quote side by side | — |
| p50 latency, 1 question, T4 | **39.5 ms** (their measured) | ≤ 39.5 ms **on a laptop CPU** | measured-ish, needs §5 P4 |
| p50 latency, 50 questions, T4 | **771.3 ms** (their measured) | ≥ 5× faster, same box | projected, gated on cloud access |
| input window | **512 tokens** english / **1024** typed-decisions | 16,384 measured; flat cost | measured |
| retained state | KV-free but re-encodes; window-bounded | **576 KiB fixed, independent of length** | measured |
| calibration ECE | **0.081** after temperature fit (their measured) | **≤ 0.02**; 0.0076 already measured on v0 | measured (v0) |
| browser artifact | ~1.7 GB fp32 / ~420 MB int8 (arithmetic) | **≤ 20 MB int8**, cold load ≤ 3 s | projected, gated on ONNX export |

### 2.2 Acceptance gates — the project is not finished until all pass

| # | gate | pass condition | status |
|---|---|---|---|
| **G1** | Accuracy beats the witness | macro decision-v2 test ≥ 0.70 on the frozen split, per-source table published, ≥ 0.4331 majority floor by ≥ +0.15 | open |
| **G2** | Latency claim survives equal-footing re-measurement | same box, same window, same question count, direct `Agent` call not `Router`; published ratio recomputed from that or withdrawn | open |
| **G3** | Deployment works for real | ONNX browser build answers a live page's decisions; bytes + p50 + cold-load measured in Chrome | open |
| **G4** | Long-context is *correct*, not just cheap | needle-style typed decision ≥ 0.90 at 4k and ≥ 0.85 at 16k state | open |
| **G5** | Useful confidence, with abstention | risk/coverage curve on `calibration.jsonl`; ≥ 0.95 accuracy at ≥ 60% coverage on banking77 + dbpedia14 + trec | open |
| **G6** | No regression on what already worked | synthetic v0 test ≥ 0.94 (was 0.951 measured) | open |
| **G7** | Reproducibility | one command per result, seeds pinned, witness JSONs committed, `pytest` green | open |

### 2.3 Non-goals (deliberate exclusions)
- **No text generation, no vocabulary readout.** The contract is decisions in constant time.
- **No KV cache.** If a mechanism needs the token list at answer time, the architecture failed.
- **No cloud dependency for inference.** Training may use a rented GPU; serving may not.
- **No aggregate-average marketing.** Claims are stratified per source and per stratum.
- **No borrowing a pretrained prior into the trunk** without saying so in the headline; if we
  ever initialize from someone's encoder, the contribution is the readout and we label it as such.

---

## 3. Architecture

### 3.1 Shape as built

```
observation tokens  ──► embedding ──► bidirectional GLA trunk (6 layers)  ──► per-layer matrix state S
                                                                      │
   question text + option descriptions ──► question branch ───────────┘
                                                            │
                              pointer probe (wq, wk over d_ptr=256) over option spans
                                                            │
                                     one logit per option, per question
```

| component | as built (measured) |
|---|---|
| trunk | 6 layers × d_model 384 × 6 heads, d_k = d_v = 64, d_ff 1024 |
| trunk params | 15,156,864 |
| probe params | wq 98,560 + wk 98,560 |
| **total** | **15,353,984** at vocab 4096; ~16.9M at vocab 8192 |
| retained state | 6 × 6 × 64 × 64 × 4 B = **576 KiB**, length-independent |
| scan forms | recurrent (streaming) = chunked (training, chunk 16; 256 under no_grad) = quadratic (reference), pinned equal in float64 |

### 3.2 The five design decisions that are load-bearing
1. **Causal trunk + fully bidirectional question branches.** A causal state is what makes prefix
   caching exact; questions are short enough to afford a backward scan inside the branch.
2. **Questions are branches, not prompt tokens.** Each resumes from the cached state end, so
   cross-question isolation is architectural, not a mask (`test_question_branches_are_isolated`,
   atol 1e-9).
3. **No vocabulary head.** One pointer mechanism over option spans serves all three decision types.
4. **Three scan forms proven equal.** This is the reason the streaming path can be trusted at all;
   it also caught the padding-decay bug that no loss curve would have shown.
5. **Long memory is the prior.** Forget gate initialized to a ≈ 0.99, not sigmoid(0) = 0.5; at 0.5 a
   token's trace decayed to ~1e-6 within 20 steps and the model could never read long inputs.

### 3.3 Known architectural limits (must be fixed, not worked around)
1. **Shared question-tensor batch contract.** A batch currently carries **one** question-tensor set
   (`q_ids [N, Lq]`, not `[B, N, Lq]`). Consequences: `boolq` and `mnli` — where the instruction
   text *is* per-row (the question, the hypothesis) — cannot be batched at all, and thin
   question-sets force short batches. **Fixing this is the single highest-value architecture
   change in the project** (§5 P2).
2. **Mid-state edits cost a full re-encode.** Prefix/suffix block caching is specced but unbuilt
   (Pillar 3 of V2_SPEC).
3. **One matrix state cannot hold verbatim recent detail and distant gist simultaneously.**
   Two-timescale state + learned eviction is specced, unbuilt (Pillar 4).

### 3.4 Code map
`src/myna/`: `trunk.py` (GLA scans) · `model.py` (trunk + probe + `forward_truncated`) ·
`tokenizer.py` (byte-level BPE + `question_tensors`) · `data.py` (synthetic corpus) ·
`real_data.py` (kev suite JSONL adapter) · `engine.py` (streaming sessions) · `rlcd.py`
(strictly-proper scoring + KL leash) · `longctx.py` (needle generator) · `mlx_model.py`
(Apple inference mirror) · `serve.py` (`/v1/sessions`, `/v1/predict`).
`bench/`: streaming bench, laya witness runners, needle eval, MLX bench, result summarizer.
`tests/`: 34 passing, including float64 scan equivalence, engine parity, adapter invariants,
batch-sampling regression guards.

---

## 4. Baselines we are measured against

All numbers below are **measured by us** unless attributed. Our own witness scripts and result
JSONs live in `runs/` (gitignored; the two laya witness JSONs are the ones that matter and must be
committed or reproduced on demand).

### 4.1 laya, on kev decision-v2 (our seeded witness)
`bench/eval_laya_real.py`, n=546, 40 per source, seed 0.
**test 0.6319 · development 0.6392.** Per-source test: agnews 0.950 · trec 0.825 · yelp 0.825 ·
dbpedia14 0.775 · mnli 0.675 · contrastive 0.625 · boolq 0.575 · imdb 0.550 · sst5 0.425 ·
amazon 0.325 · banking77 0.250.

### 4.2 The floors, so "better" means something
Computed from the same splits, pooled per (source, question), n ≥ 30:
**macro majority-class 0.4331 · macro uniform-chance 0.3292.**
Notable: laya beats the constant predictor by **+0.013 on imdb** and **+0.038 on boolq** — i.e. on
its binary tasks laya is statistically indistinguishable from answering the same thing every time.
This is why the accuracy throne is not defended as strongly as a 0.63 average suggests.

### 4.3 laya's own published numbers (from its repo, for context only — different suites)
typed-decisions 0.766 on 2,000 decisions · ECE 0.081 after temperature fit · T4 p50 39.5 ms for 1
question rising ~14.9 ms per extra question (771.3 ms at 50) · checkpoints 421.29M english /
321.91M multilingual. Their `BENCHMARKS.md` states that Jev figures are third-party and never
independently measured — we match that discipline: no competitor number in our tables unless we
ran it or we label whose number it is.

### 4.4 Jev
Browser agent with a dynamic, indexed action space (TypeSafe). Published p50 236–276 ms per
decision per laya's comparison. Relevant to us only as the deployment framing the ecosystem is
currently arguing about: *decisions inside a browser tab, no API cost.* That is myna's best
showing, not its worst.

---

## 5. The plan

Ordered by dependency, not by excitement. Each phase has an exit test; a phase that fails its test
does not advance, it goes in the corrections log (§9).

### P0 — Repo and measurement hygiene *(partly done)*
- [x] Fix duplicate-row batching: `draw_batch()` shortens instead of repeating (`385e06c`).
      Mutation-checked: reverting it fails the regression test.
- [x] Same fix in `rlcd.py`, which would have **crashed** on a short pool.
- [x] Startup diagnostic printing median pool size and share of sets with pool < batch.
- [ ] Fit temperature on the suite's own `calibration.jsonl` (448 records, 568 questions) instead
      of `dev`. Currently we fit on dev, which quietly flatters the dev number — a methodological
      defect on our side.
- [ ] `--device auto` must consider CUDA (it resolves to mps/cpu only, so any cloud box would
      silently train on CPU).
- [ ] Re-derive every V1-B conclusion; numbers in `runs/*.log` predate `385e06c` and are void.
- [ ] Push the repo. **No remote exists** — 4 commits and the entire codebase live on one laptop.
      This is the largest unmanaged risk in the project. User-gated.

### P1 — Data: the accuracy unlock *(tasks #14, #15)*
The suite froze 500 rows per source; the upstream corpora behind them hold ~2M labelled examples,
and decision-v2's manifest lists **all 11 sources as trainable** with `holdout_sources: []` and
explicitly disclaims any decontamination claim.
- [ ] Add `datasets`; pull ~5k upstream train rows per source from the **pinned HF revisions** in
      `manifest.json`.
- [ ] Re-express each through the suite's own question definitions so the eval format matches.
- [ ] Exact-state dedup against `development.jsonl` + `test.jsonl`, using kev's normalized-state
      rule; **print the collision count**. No leak claim without this number.
- [ ] Instruction paraphrase augmentation (≥ 8 phrasings/schema) with the **exact suite wordings
      held out** for eval — this separates "learned the template" from "learned to read".
- [ ] Pilot corpus target: ≥ 55,000 states ≈ **4.1M tokens**, vs 0.26M today (measured).

### P2 — Architecture: per-row question tensors *(the blocker for boolq/mnli)*
- [ ] `q_ids/q_mask/span_mat/opt_valid/decide_idx` gain a leading batch dim, with masks so a batch
      can mix question-sets.
- [ ] Equivalence test: a homogeneous batch scores identically before and after (atol 1e-6).
- [ ] Re-measure memory: the pointer head materializes `[B, N, O, Lq]`, so cost scales with
      options × batch — this is the tensor that OOM'd the M5. Cap options per update in the sampler.

### P3 — Train and validate *(gated on user go, after the M5 crash)*
- [ ] Memory-safe launch: cap sets by option count per update, clear MPS cache between updates,
      start at a few hundred updates, hard-stop if observed s/update rises or free RAM drops.
- [ ] Then the 15.35M-vs-~32M head-to-head on identical data — the size question is unanswerable
      until the data exists, and cheap once it does.
- [ ] Report **stratified** per source: groupable (8/11 sources, 250–300 examples each) vs
      per-row-instruction (boolq, mnli). Aggregate is marketing, strata are evidence.

### P4 — Latency reconciliation *(task #17, gates G2)*
- [ ] Re-run myna vs laya on one box, one process, direct `Agent` call (not `Router`), matched
      state length / question count / dtype / options-per-question.
- [ ] Separate fixed per-call overhead from per-token and per-question cost.
- [ ] Until then the only latency sentence allowed in public: *myna on a laptop CPU has roughly the
      per-decision latency laya reports on a datacenter T4.*

### P5 — Abstention and risk/coverage *(task #16, gates G5 — the flagship scenario)*
- [ ] Confidence threshold tuned on `calibration.jsonl`; emit `abstain` with the reason.
- [ ] Risk-coverage curve in the README table, not just an accuracy number.
- [ ] Fallback seam to a stronger model, with the chosen model labelled per decision. A fast path
      that is uncertain must never be silently load-bearing.

### P6 — Browser/on-device deployment *(task #18, gates G3)*
- [ ] Split ONNX export (trunk + pointer head) mirroring `laya-ts/scripts/export_onnx.py`, with
      torch-vs-ONNX parity ≤ 1e-4.
- [ ] Measured in Chrome via `onnxruntime-web`: download bytes fp32/int8, cold-load ms, p50 per
      decision on a real page.
- [ ] MLX int8 path for Apple, re-measured after any parameter growth.

### P7 — Long-context proof *(gates G4)*
- [ ] Train the 4k truncated-backprop checkpoint (`--long-context --init`), code complete and
      committed.
- [ ] `bench/eval_needle.py` recall-vs-length curve at 1k/4k/8k/16k — the region where laya is
      window-limited to 1024 tokens and is therefore not reading the input, only its prefix.

### P8 — Write it up
- [ ] README + PLAN headline tables from committed artifacts only, each cell labelled
      measured/projected/gated.
- [ ] Reproduction script per table.
- [ ] Public release gate: P0 push + G1–G7 all pass or all explicitly marked not met.

---

## 6. Compute and budget

- **Measured:** 0.26M tokens of training text (3,432 rows × mean 300 chars ÷ 4). 15.35M params ⇒
  **0.017 tokens/param**. Compute-optimal for pretraining (~20 tok/param) would want ~307M tokens.
  This yardstick is a *pretraining* law; supervised fine-tuning is more sample-efficient — but the
  capability we lack, following instructions never seen in training, is exactly the one that
  scales with tokens-per-parameter. That is the honest boundary.
- **Therefore: parameters are not the lever.** 32M would be 2,469× under, 93M would be 7,251×
  under. Scaling params without data doubles the data debt. Growth is affordable on latency and
  gets tested in P3 — after the data.
- **M5 (32 GB, unified):** `--batch 32 --vocab 8192` thrashes swap; a pool+accum run projected
  2.81 s/update but **measured 4.93 s/update** before raising Metal
  `kIOGPUCommandBufferCallbackErrorOutOfMemory` and rebooting the machine. Budget by *option
  count*, not token count. Never run P4/P6/P7 concurrently with P3.
- **Cloud (gated):** blockers are (1) no git remote, (2) `--device auto` never picks CUDA,
  (3) `requires-python >=3.13` vs 3.11/3.12 images, (4) suite path is a local absolute path with a
  trailing space in a parent directory name. Training can move; **measurement cannot** — MLX is
  Apple-only and the on-device claim must be measured on the M5.

---

## 7. Working discipline

1. **A passing count is not proof.** Corroborate from the system under test, and mutation-check
   every gate — a test that cannot fail is not a gate.
2. **A tool's own exit code is the weakest witness.** Verify on-disk and on-process state before
   reporting that anything landed.
3. **Label every number** measured / projected / gated. Never fabricate a competitor multiplier;
   never publish an internal ablation ratio as a competitor comparison.
4. **Stratify, then aggregate.** The aggregate is where claims get made, so it is the last thing
   to trust.
5. **Own corrections in writing**, in this file, with the artifact that proves them (§9 is the
   record; it must never be emptied).
6. **Process safety during training:** never broad-`pkill`; kill stray jobs by exact PID.
7. **Push stays user-gated.** Local commits are fine.

---

## 8. Risks

| risk | severity | mitigation |
|---|---|---|
| 15.35M cannot learn instruction-following well enough to clear G1 | **high** | the P2 per-row contract fix + P1 data are both prerequisites; if G1 still fails, the honest product is abstention (G5), not accuracy |
| Latency advantage evaporates under equal-footing measurement (P4) | **high** | the size, window, calibration and browser-memory claims stand independently; the pitch must survive without the ratio |
| Upstream-data comparison reads as unfair | medium | disclose prominently; publish myna-scratch, myna-trained, laya as three rows |
| Another machine crash mid-run | medium | cloud training, checkpoint every ≤ 50 updates, observed-rate stop rule |
| ONNX export can't express the recurrent scan efficiently | medium | prototype the export before committing to G3; fallback is wasm-only fp16 with a measured caveat |
| Parameter growth quietly voids the speed claims | low | re-run the latency ladder after any size change, in the same commit |

---

## 9. Corrections log — things we previously stated wrong

Kept permanently, because the value of this project's claims is that they survive scrutiny.

1. **"239× faster than laya."** Wrong: 239× is myna-stream vs *myna's own* re-encode path — an
   internal ablation. Real measured comparison vs laya: ~15× at 1k tokens on the same box, rising
   with context length. Fixed in the memory file; must never reappear in the README.
2. **"laya's context ceiling is 8192, so we have 2× it."** Wrong. The laya *checkpoint* windows are
   512 (english) / 1024 (typed-decisions); 8192 was a Router-level input guard. Our long-context
   claim is therefore **categorical, not a ratio**: beyond ~1024 tokens laya is deciding from a
   prefix, not from the document.
3. **"1,146 question-sets, median pool 1, therefore catastrophic interference."** Wrong, and it
   was my own adapter. `_signature` hashed `criteria` **descriptions**, which the suite randomizes
   per row (55 of 77 banking77 descriptions differ between two rows of the same task), shattering
   300 same-schema rows into 300 singleton sets. Grouped instead on **exact instruction text +
   option key set**: **642 sets**, of which 602 are singletons — and 600 of those singletons are
   `boolq` (300) and `mnli` (300), which are genuinely one-set-per-row because their instruction
   text *is* the per-row question/hypothesis. The other nine sources collapse to **40 sets**, with
   amazon/dbpedia14/imdb/trec each a **single** set of 300, and banking77 (249) / sst5 (248) /
   yelp (258) / contrastive (120) / agnews (48) holding 2–22 sets each. So: 8 of 11 sources have
   250–300 genuinely distinct examples under a stable schema, and only boolq + mnli need the P2
   per-row contract fix. (An earlier pass of this analysis quoted "~282 sets"; that came from
   masking quoted spans in the instruction text, which over-collapsed boolq/mnli/agnews. 642 is the
   defensible number and the masking approach is abandoned.)
4. **The batching bug that the above hid.** `rng.choices(pool, k=batch)` samples **with
   replacement**, so a singleton pool produced a batch of N identical rows — loss → ~0, gradient
   ≈ one example. This affected most updates in every V1-B run. Fixed in `385e06c`. Consequence:
   **every accuracy figure in `runs/*.log`, including myna-test 0.2821 and the "below the majority
   floor" line, is void as evidence about myna's ceiling.** They measure my bug.
5. **"myna structurally cannot beat laya on accuracy."** Retracted — it was conclusions 3 and 4.
   The lane is open: laya's own margin over a constant predictor is +0.013 (imdb) and +0.038
   (boolq).
6. **Two mechanisms built for a misdiagnosis.** `--accum-groups` and `--group-sample` were added to
   treat interference. Accumulation did help — partly because it replaced duplicated rows with
   distinct examples. Kept, but the stated reason is corrected.
7. **"The step-600 checkpoint landed."** Said from a background task's exit-0 notification while no
   file existed. Retracted.
8. **"dev-mid 0.692."** Misread a loss line as an accuracy line; actual 0.3413.

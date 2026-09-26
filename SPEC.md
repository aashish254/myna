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
| calibration ECE | **0.081** after temperature fit (their measured) | **≤ 0.02**; 0.0076 on v0 **dev**, whose temperature was fit on those rows — read as optimistic (§9.11) | measured (v0), level disputed |
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

### 3.3 Known architectural limits
1. ~~**Shared question-tensor batch contract.**~~ **Fixed in P2.** A batch can now carry a
   different question set per row (`[B, N, Lq]` and friends, padded on all three varying axes with
   a `has_gold` mask), which is what `boolq` and `mnli` need — their instruction text *is* the
   per-row question, so no regrouping could ever have made them batchable. The shared form is still
   accepted, so the streaming engine, the MLX mirror and serving are unchanged.
   **What remains open is the price, not the shape:** `bench/mem_profile.py` measures 1.6 MiB of
   retained activations per question-token position, so a mixed batch pays for its longest row and
   `--max-q-cells` is the dial that keeps it survivable (§5 P2, §5 P3).
2. **Mid-state edits cost a full re-encode.** Prefix/suffix block caching is specced but unbuilt
   (Pillar 3 of V2_SPEC).
3. **One matrix state cannot hold verbatim recent detail and distant gist simultaneously.**
   Two-timescale state + learned eviction is specced, unbuilt (Pillar 4).

### 3.4 Code map
`src/myna/`: `trunk.py` (GLA scans) · `model.py` (trunk + probe + `forward_truncated`) ·
`tokenizer.py` (byte-level BPE + `question_tensors` / `batch_question_tensors`) · `data.py`
(synthetic corpus) · `real_data.py` (kev suite JSONL adapter, `flatten_groups`) ·
`paraphrase.py` (9 phrasings/core + the reserved held-out index) · `engine.py`
(streaming sessions) · `rlcd.py` (strictly-proper scoring + KL leash) · `longctx.py` (needle
generator) · `mlx_model.py` (Apple inference mirror) · `serve.py` (`/v1/sessions`, `/v1/predict`) ·
`report.py` (the 3g stratified table: per-(source, question) cells, both floors recomputed from the
split, instruction-derived strata, laya's own accuracies joined by cell, G1 verdict).
`bench/`: streaming bench, laya witness runners, needle eval, MLX bench, result summarizer,
`pull_upstream.py` (P1 data pipeline), `mem_profile.py` (per-axis step memory),
`pilot_topology.py` (question-set topology + rows-per-forward under each contract),
`mutation_paraphrase.py` (the P1 gate's mutation battery, 29/29),
`mutation_memory_plan.py` (the P3 sizing/stop/resume gate, 44 mutations),
`mutation_report.py` (the 3g reporting gate: every floor, weight and stratum in the table),
`mutation_kaggle_bundle.py` (the entrypoint + packager gate), `check_python311.py` (the 3.11 witness
that runs the package on a real 3.11 interpreter and exits 2 rather than skipping).
`kaggle/`: `run.py` (one-command entrypoint: corpus discovery, the measured flag set, `--resume` only
when a snapshot exists, tee to `train.log`, `run.json`, non-zero propagation), `package_dataset.py`
(stage the pilot corpus, verify every byte against the corpus, print the upload command, never run
it), `requirements.txt`.
`tests/`: **167 passing + 1 KEV-gated parity test** (skipped here, green wherever `kev` is
installed). The P3 gate is `test_memory_plan.py` (sizing arithmetic, the refusal, the CUDA headroom
read through a monkeypatch, the stop rule both ways, snapshot round-trip, resume),
`test_kaggle_bundle.py` (the entrypoint's refusals and composed command, a real 2-update run through
it, the packager's hash discipline) and `test_report.py` (both floors from the cell's own rows, the
row-weighted cell mean, the instruction-derived strata and their 0.5 boundary, the laya join, the G1
verdict's two halves, and the CLI on the frozen split reproducing §4.2's published floors);
the device-resolution and calibration-split guards, the
batch-sampling regression guards, per-row batch equivalence, float64 scan equivalence, engine parity,
adapter invariants, and the paraphrase table + trainer-wiring pairs (the table's own tests cannot see
the loop that uses it).


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
**macro majority-class 0.4331 · macro uniform-chance 0.3321.**
Both are now recomputed from the shipped rows by `src/myna/report.py` (`python -m myna.report
--suite data/decision-v2-pilot --split test --metrics <run>`) and pinned by a test at 0.433 / 0.332,
so the floor G1 is judged against is derived twice by two independent pieces of code and agrees. The
uniform figure changed from the 0.3292 published here earlier — see §9.16.
Notable: laya beats the constant predictor by **+0.0125 on imdb** (0.550 over a 43/80 floor) and
**+0.0375 on boolq** (0.575 over 43/80) — i.e. on
its binary tasks laya is statistically indistinguishable from answering the same thing every time.
This is why the accuracy throne is not defended as strongly as a 0.63 average suggests.
The same harness also measures, rather than lists, the **9-vs-2 stratum split**: on the test split
boolq and mnli are *per-row-instruction* — the row's own text is the question (80 distinct
instructions over 80 boolq rows, 80 over 116 mnli slots) — while the other nine repeat a handful of
schemas across every row (agnews: 8 distinct instructions over 300 labelled slots, all repeated)
even though the adapter's group signature splits them into 113 near-singleton sets. That distinction
is what makes "groupable vs not" a measured fact instead of a naming exercise; see §9.17.


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
- [x] Fit temperature on the suite's own `calibration.jsonl` (448 records, 568 questions) instead
      of `dev`. We fit it on dev through v0, which quietly flatters the dev number — a
      methodological defect on our side. The counts re-measured *through our adapter*:
      **calibration 448 rows / 568 labelled questions**; dev and test 1,176 rows / 1,440 questions
      each (the pilot dir's copies of the frozen eval splits).
      **Fixed:** `load_suite` now reads the fourth split and
      `train.temperature_source` prefers it, falling back to dev only when it is absent (the
      synthetic corpus ships none). The printout and `metrics.json` carry the split *and the row
      count it fit on*, because a label alone is not a witness — an earlier draft of the line
      printed "calibration" while fitting on dev and only the fitted value gave it away.
      Witnessed end to end: `temperature: 3.0 (fit on calibration: 20 rows in 20 sets)` on a
      20-row sample suite, `(fit on dev)` on the synthetic path. Mutation battery: 8 mutations of
      the new code, all caught — the first pass missed `data.get("calibration", data["dev"])`,
      which is invisible unless the key is *absent* rather than empty, so
      `test_synthetic_data_with_no_calibration_key_reports_dev` now covers exactly that.
      Consequence for the record: **v0's dev metrics carry this defect** (its temperature 3.0 was
      picked on the same rows), so read v0's dev ECE as optimistic. Its *test* metrics do not —
      test rows were never in the fit.
- [x] `--device auto` must consider CUDA (it resolves to mps/cpu only, so any cloud box would
      silently train on CPU). **Fixed:** `train.resolve_device` — MPS, then CUDA, then CPU, and an
      explicit `--device` is never rewritten. Order matters and is tested both ways.
- [ ] Re-derive every V1-B conclusion; numbers in `runs/*.log` predate `385e06c` and are void.
- [ ] Push the repo. **No remote exists** — 30 commits and 45 tracked files live on one laptop.
      This is the largest unmanaged risk in the project. User-gated.

### P1 — Data: the accuracy unlock *(tasks #14 and #15 done)*
The suite froze **300 train rows per source** (contrastive 432, since its generator emits a fixed
policy grid), 80–116 in each of dev/test and 40–48 in calibration — 500–872 rows per source across
all four splits, **6,232 rows total** (counted directly on the pinned checkout; an earlier draft of
this section said "500 rows per source", which conflated the per-source total with the train split,
now §9.10). The upstream corpora behind them hold ~2M labelled examples, and decision-v2's manifest
lists **all 11 sources as trainable** with `holdout_sources: []` and explicitly disclaims any
decontamination claim.

**Done — the pilot slice is on disk** (`bench/pull_upstream.py`, `9e954f9`; corpus in
`data/decision-v2-pilot/`, gitignored except its manifest). Every figure below is printed by that
run and re-read from `upstream_manifest.json`, not transcribed from a log:

| item | target | measured |
|---|---|---|
| train states | ≥ 55,000 | **57,904** = 54,472 upstream + 3,432 frozen-suite train |
| questions | — | **73,804** |
| state tokens (suite's own Qwen2.5 tokenizer) | ≈ 4.1M | **4,246,106** (median 49 · p95 263 · max 402) |
| tokens per parameter | — | **0.28** (was 0.017) |
| rows in eval after the pull | 0 | **0** |
| upstream rows already in frozen train | 0 | **0** |
| repeated states within the pull | 0 | **0** |
| round-trip (re-read files, re-digest) | exact | **exact** |

Per-source `~5k` hits its target for all ten downloaded sources; `contrastive` is generated and
admits **4,472** of 4,800 (the 328 dropped are its collisions with eval + frozen train, below).

- [x] Add `datasets`; pull ~5k upstream train rows per source from the **pinned HF revisions** in
      `manifest.json` (revisions recorded per source in the pull manifest).
- [x] Re-express each through the suite's own question definitions so the eval format matches —
      by *calling* `kev.data.build` / `kev.contrastive.generate` at the pinned suite rather than
      reimplementing the converters, so the format cannot drift. A parity test runs our `admit()`
      against `kev.suite.select_unique` on identical inputs.
- [x] Exact-state dedup against `development.jsonl` + `test.jsonl` (+ `calibration.jsonl`), and
      against the frozen `train.jsonl` we now append to — **printing the collision count**. The
      gate is on *both* keys, because they disagree in both directions:
      `_meta.text_sha256` hashes an upstream *field* (yelp/imdb/amazon hash the full review then
      cut to 220 words → one state, two hashes; contrastive hashes the case sentences but not the
      policy → one hash, two states). Collisions actually found and excluded: boolq 150 eval + 497
      already-in-frozen-train + 726 duplicate states, mnli 44/322/329, banking77 40/300/4,
      contrastive 39/52/73, agnews 40/301/1, sst5 42/300/3, trec 43/307/26, imdb 42/297/16,
      amazon 40/300/10, yelp 40/300/0, dbpedia14 40/300/0.
      **Methodological find:** the frozen suite itself ships 1 `text_sha256` shared between a train
      row and an eval row (their states differ, so it is not a leak — but it is why deduping on
      that key alone would let a real leak through, and it is the number the `--per-source` gate
      would have printed 0 for).
- [x] Instruction paraphrase augmentation with the **exact suite wordings held out** for eval —
      this separates "learned the template" from "learned to read". *(task #15, done)*
      `src/myna/paraphrase.py` ships **9 phrasings for each of the 17 templated cores** (agnews 5,
      contrastive 4, yelp 2, one each for amazon/banking77/dbpedia14/imdb/sst5/trec) plus 9 frames
      each for the two per-row-content sources, where only the frame moves and the row's own
      question/hypothesis is reproduced **verbatim** so no phrasing can shift a gold.
      **Index 8 of 9 is reserved**: training draws 0..7, and dev is scored twice — on the suite's
      exact strings and on that unseen one (`metrics.json:dev_unseen`).
      Coverage, measured on the pilot through our own adapter: **17,112 sets / 19,596 question
      shapes / 57,904 rows / 73,804 labelled slots** paraphrased, **84,936 distinct training
      wordings, 0 of them a dev, test, calibration or train suite string**. The 10,629 distinct
      instruction strings on disk are 10,600 boolq/mnli row content (5,300 + 5,300) and 17 cores.
      An instruction with no table entry raises `UnknownSchema` rather than passing through: a
      silent pass-through would train on the eval wording and void the gate.
      Trainer wiring: `--paraphrase off|on` on **both** batch paths, variants pre-built so the
      `id()`-keyed length cache stays valid, and `worst_case_tokens` pricing every set at its
      **longest** phrasing so `--max-q-cells` remains an upper bound rather than a hope. The run
      prints `paraphrase: N question sets re-worded …` *and* `paraphrase: D phrasing draws reached
      the batches over S steps`, with both numbers in `metrics.json` — the warm-up line alone would
      still print if the loop never consulted the table. `--long-context` refuses `--paraphrase on`
      instead of claiming a gate the needle corpus cannot run.
      Witness: `bench/mutation_paraphrase.py`, **29/29 caught** (§9.12 for the false-green first
      pass, which is what found the two real holes: boolq rows whose question *ends* with one of
      the suite's randomized suffixes were having that suffix edited, and content frames whose
      nine variants all ended the same way).
- [x] Pilot corpus target: ≥ 55,000 states ≈ **4.1M tokens**, vs 0.26M before — **57,904 states /
      4.25M tokens, measured** (§6 recomputed from it).

**What the pilot does *not* fix, measured.** Loading it through our own adapter gives
**17,112 exact-signature question-sets over 57,904 rows: median pool 1, 96% singletons.** The rows
are not unique tasks — grouping on instruction *schema* instead of exact tensors gives 10,668
schemas whose largest holds 5,300 rows (amazon/dbpedia14/imdb/trec 5,300 each; yelp 4,512;
banking77 4,497; sst5 4,490; contrastive families 1,158–1,302). So 16× more data lands in the same
batch-of-1 shape that throttled V1-B: **P2 is now the binding constraint, not an
optimization.** boolq and mnli are the permanent half of it — their instruction text genuinely
*is* the per-row question, so no regrouping can ever make them batchable.

### P2 — Architecture: per-row question tensors *(done; unblocks boolq/mnli)*
- [x] `q_ids/q_mask/span_mat/opt_valid/decide_idx` gain a leading batch dim, with masks so a batch
      can mix question-sets. `tokenizer.batch_question_tensors` builds `[B,N,Lq]`, `[B,N,O,Lq]`,
      `[B,N,O]`, `[B,N]`, padding on all three varying axes; the shared `[N,·]` form is still
      accepted and broadcast, so the engine, MLX mirror and serving path are untouched.
      `train.build_row_batch` pairs it with a `has_gold` mask, and `--row-batch` draws one
      mini-batch across the whole train split instead of one question-set at a time.
- [x] Equivalence test: a homogeneous batch scores identically before and after (atol 1e-6).
      Three directions, in `tests/test_perrow_batch.py`: shared form vs per-row form on a
      homogeneous batch; every row of a mixed batch vs that row scored alone; and the mixed-batch
      loss equals the cell-weighted mean of the single-row losses. **13 mutations of the new code
      were run and all 13 were caught** — including `decide_idx` → zeros and the pooled-span
      mutation, which the invariance tests cannot see by construction and which
      `test_engine_parity` catches against the independent readout implementation.
- [x] Re-measure memory (`bench/mem_profile.py`, one subprocess per config, `ru_maxrss` peak at
      `d_model` 384 / vocab 8192 / batch 8). **The plan's assumption was wrong** (§9.9): the option
      axis is nearly free and the *question branch* is what fills the machine.

| axis | retained activations, measured |
|---|---|
| per question-token position (`B × N × Lq`) | **1.6 MiB** |
| per state-token position (`B × Ls`) | **0.8 MiB** |
| per option cell (`B × N × O`) | **3 KiB** |

  At the pilot's shapes and a 2,048-cell question budget (`--max-q-cells`, the unit
  `draw_row_batch` now enforces), `bench/pilot_topology.py` measures how many rows of each source
  fit in one forward: **boolq 64, mnli 49, imdb 64, contrastive 64, sst5 64, yelp 28, trec 24,
  agnews 23, dbpedia14 37, amazon 53, banking77 4** — against a median pool of **1** row under the
  old shared-set contract (17,112 exact-signature sets, 96% singletons). boolq and mnli go from
  one row per forward to a full batch; banking77 is the exception, because its 77 option
  descriptions make one question 431 tokens long.
- [x] Corollary for P3, stated as arithmetic: a batch of 32 rows at the pilot's p95 state length
      (263 tokens) retains ~6.6 GB in the state scan (32 × 263 × 0.8 MiB), ~10.0 GB at the
      402-token max. A step directly measured at `B=32, Ls=48, N=3, Lq=64` peaked at **11.2 GB**
      of resident memory including weights, gradients and Adam state — which is the M5 crash,
      explained. P3 therefore runs either `--batch ≤ 8` on full states or the truncated-backprop
      path (`forward_truncated`, already built) so only the grad window is retained.

### P3 — Train and validate *(code done and witnessed here; the run itself is `KAGGLE`)*
- [x] **3a** `train.py` runs on CUDA without edits. The device path is witnessed on this box:
      `--device auto` considers CUDA (`59240e1`), an explicit unavailable name fails loudly with the
      list of what the machine has instead of silently downgrading, and `free_device_bytes()` reads
      `torch.cuda.mem_get_info()` — pinned by a monkeypatched-CUDA test that asserts the plan uses
      *free* bytes and not total, and that a `RuntimeError` from the driver degrades to `None`
      ("no headroom reading", `--batch` taken as given). The same flags as the box
      (`--row-batch --max-q-cells 2048 --accum-groups 2 --group-sample uniform --paraphrase on
      --free-gib 10 --mem-safety 0.5 --save-every 1 --stop-factor 1.5`) run 3 CPU updates green.
      The GPU half of the witness is `KAGGLE`.
- [x] **3b** `kaggle/` bundle. `run.py` is the one-command entrypoint: it refuses to start without
      `EXPERIMENT_NAME`, refuses a corpus that is missing a split (naming the file, never falling
      back to the synthetic data), composes the measured flag set, adds `--resume` exactly when
      `model_last.pt` exists, tees the trainer to `runs/<name>/train.log`, writes `run.json`, and
      propagates the trainer's non-zero stop exit. `package_dataset.py` stages the pilot corpus,
      writes `SOURCE_SHA256.json` **from the corpus** plus `dataset-metadata.json`, verifies every
      staged byte, and prints the upload command instead of running it.
      Witness: `16 passed` in `tests/test_kaggle_bundle.py`, including a real 2-update CPU training
      through the entrypoint and a lossy-copy build that exits non-zero. Gate:
      `bench/mutation_kaggle_bundle.py` → **25/25** (first pass 21/25; §9.15).
- [x] **3c** Python 3.11. `requires-python` is `>=3.11` (it was `>=3.13`, which made the Kaggle
      image a *silent fallback*), every file under `src/ tests/ bench/ kaggle/` parses under
      `feature_version=(3, 11)`, and `bench/check_python311.py` runs the whole thing on a real 3.11
      interpreter: prints **`PASS: the package imports, compiles and trains on python 3.11`** on
      3.11.15 / torch 2.6.0, and exits 2 rather than skipping if no 3.11 is found.
- [x] **3d** Memory-safe sizing as a computed startup line. `state_token_p95()` measures p95 over a
      4,000-row sample of the rows the loop will draw (p95, not the mean, because the batch pads to
      its longest row); `rows_that_fit()` divides `safety × free` by `p95 × 0.8 MiB` **after**
      subtracting the question branch (`--max-q-cells × 1.6 MiB` on `--row-batch` steps), both
      prices from `bench/mem_profile.py`. Three outcomes, all printed: the batch fits, it is clamped
      down with the number it was clamped to, or — when the reserve alone eats the budget — the run
      **refuses** and names the knob (`lower --max-q-cells`). No headroom reading is an explicit
      line, not an invented number.
- [x] **3e** Stop rule in the loop: after each update, hard-stop if the last step time exceeds
      1.5× the median of the previous 20, or if measured free bytes fall below 1.25× the projected
      per-step need. The stop writes `model_last.pt`, prints the reason, and the process **exits
      non-zero** — a truncated run that reports success is how a dead job becomes a "result".
      Witnessed by construction: `--stop-factor 1e-4` over 30 steps makes the 21st update a
      violation, and the default factor demonstrably lets a normal run finish.
- [x] **3f** Cadence and resume: `--save-every` defaults to 25 (≤ 50 as the spec demands) and the
      snapshot carries weights, config, temperature, step, **optimizer and scheduler**, so
      `--resume` continues the cosine decay instead of restarting it and does not replay the
      snapshotted update (`start = stored step + 1`). Pinned: a round-trip at step 17 recovers the
      Adam moments and the scheduler's `state_dict()`; a missing snapshot is a named refusal, not a
      `torch.load` traceback.
- [x] All five gates above are mutation-checked by `bench/mutation_memory_plan.py`: **44 mutations,
      all caught**, run inside a scratch copy of `src`+`tests` (the repo's own `pythonpath = ["src"]`
      outranks `PYTHONPATH` — §9.12), with a green-baseline guard and an abort-if-the-first-mutant-
      survives guard so a battery that mutates nothing can never report a clean pass.
- [x] **3g** Stratified reporting harness: per source × question-type, groupable (9/11) vs
      per-row-instruction (boolq, mnli), majority/uniform floors in the same table — inference only,
      so runnable here against v0 and against any Kaggle checkpoint. `src/myna/report.py`
      (`python -m myna.report --suite … --split test --metrics <run dir> [--laya <witness>] [--out j]`).
      Witnessed, in the table itself: the published floors re-derive from the shipped rows
      (**majority 0.433 exactly as §4.2, uniform 0.332** — see §9.16), laya's own per-cell accuracies
      reproduce their margins over the floor (imdb 0.550 over 43/80, boolq 0.575 over the same), the
      9-vs-2 stratum split is *measured* from instruction strings rather than named (agnews: 8 distinct
      instructions over 300 labelled slots while its 116 rows sit in 113 exact-signature sets — §9.17),
      and the six cells whose option counts differ across their sets are printed with their full
      histograms because the chance floor is row-weighted over them. Gate: `bench/mutation_report.py`
      — **54 mutations over `src/myna/report.py`, all caught, exit 0**, run against a frozen repo
      (§9.18 is the process correction that bought that discipline).
- [ ] **3h** 15.35M vs ~32M head-to-head on identical data — `KAGGLE`.
- [ ] **3i** G1 verdict: decision-v2 **test ≥ 0.70** macro, ≥ +0.15 over the 0.4331 majority floor,
      per-source table published — `KAGGLE` result, reported as measured or as a loss.

**Why the sizing formula changed.** The line above formerly read
`free_bytes / (0.8 MiB × state tokens)`, which is the *state scan* alone. A `--row-batch` step holds
the question branch at the same time — that is what the M5 crash was (§9.9) — so the batch the spec
asked for was sized as though half the model were free. The reserve is now subtracted before rows
are counted, and if nothing is left the run refuses. See §9.14.


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

- **Measured:** 0.26M tokens before the pilot (3,432 rows × mean 300 chars ÷ 4) ⇒ **0.017
  tokens/param**. The P1 pilot corpus is **4,246,106 state tokens over 57,904 states** — tokenized
  with the suite's own Qwen2.5-0.5B tokenizer, not the ÷4 estimate — ⇒ **0.28 tokens/param**.
  Compute-optimal for pretraining (~20 tok/param) would want ~307M tokens.
  This yardstick is a *pretraining* law; supervised fine-tuning is more sample-efficient — but the
  capability we lack, following instructions never seen in training, is exactly the one that
  scales with tokens-per-parameter. That is the honest boundary.
- **Therefore: parameters are not the lever.** 32M would still be 1,189× under the pretraining
  yardstick, 93M 3,655× under. Scaling params without data doubles the data debt. Growth is
  affordable on latency and gets tested in P3 — after the data.
- **M5 (32 GB, unified):** `--batch 32 --vocab 8192` thrashes swap; a pool+accum run projected
  2.81 s/update but **measured 4.93 s/update** before raising Metal
  `kIOGPUCommandBufferCallbackErrorOutOfMemory` and rebooting the machine. Budget by the *token*
  axes P2 measured (1.6 MiB per question-token position, 0.8 MiB per state-token position), not by
  option count — §9.9. Never run P4/P6/P7 concurrently with P3.
- **Training runs on Kaggle, not the M5** (user directive, 2026-09-26): the M5 does architecture
  code, data, tests, inference/eval and every Apple-only measurement (MLX, MPS latency, browser
  on-device) — those cannot move. The blockers this loop could clear are cleared: (1) still no git
  remote, so the code travels as the kernel's uploaded source and the data as a dataset
  (`kaggle/package_dataset.py` builds it, prints the upload command, does not run it); (2)
  `requires-python` is `>=3.11` and the package is proven to train on a real 3.11
  (`bench/check_python311.py`); (3) `--device auto` picks CUDA (`59240e1`); (4) the pilot dir ships
  as a dataset, so no suite path with a trailing space reaches the box — `kaggle/run.py` takes
  `--corpus` or globs `/kaggle/input` and refuses a corpus missing a split.
  What remains is not code: the GPU run itself, and whoever holds the credentials (`KAGGLE`).
- **Sizing on the box** is `safety × free − question-branch reserve`, ÷ `p95 state tokens × 0.8 MiB`
  (§5 P3 3d, §9.14) — with `--free-gib` as the escape hatch when the platform reports nothing usable.


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
| 15.35M cannot learn instruction-following well enough to clear G1 | **high** | both prerequisites are now built — P1 data (4.25M tokens) and P2's per-row contract — so this risk is finally testable rather than arguable; if G1 still fails, the honest product is abstention (G5), not accuracy |
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
9. **"The pointer head materializes `[B,N,O,Lq]`, so cost scales with options × batch — that is the
   tensor that OOM'd the M5."** Wrong, and P2 measured it: option pooling is a batched matmul whose
   output is `[B,N,O,d]`, and the whole 8×3×77 option grid costs 2.7 MiB. The step's memory goes to
   the *chunked scans'* `[rows, heads, chunk, chunk, d_k]` terms — 1.6 MiB per question-token
   position, 0.8 MiB per state-token position, 3 KiB per option cell. So the sampler budgets
   question cells (which is what `--max-q-cells` does), and the M5's crash is explained by the state
   axis at `--batch 32`, not by banking77's 77 options. The correction changes the mitigation, not
   the conclusion: budget by *something*, and 32 GB is not a licence for a 32-row long-state batch.
10. **"The suite froze 500 rows per source."** Wrong as written: 500–872 is the per-source total
    across *all four* splits. The train split is 300 rows/source (contrastive 432), dev and test are
    1,176 rows each (80–116/source), calibration 448. The suite ships **6,232 rows**; the pilot's
    57,904 train states are 54,472 upstream + 3,432 of those.
11. **"v0's dev ECE ≤ 0.042."** The scalar behind every v0 dev metric was fit **on dev**, so the
    dev accuracy/calibration numbers were scored on the rows that picked the temperature — in-sample.
    Test metrics are unaffected (test never entered the fit). Fixed for future runs by fitting on the
    suite's `calibration.jsonl`; the v0 figures stand as measured but read dev as optimistic, and
    the RLCD "ECE halved" delta was measured the same in-sample way on both sides, so its *direction*
    is safe and its *level* is not quotable.
12. **"29 mutations, all caught" — said of a battery that mutated nothing.** `bench/mutation_paraphrase.py`'s
    first pass put a patched `src/` on `PYTHONPATH` and reported **0/29 caught**. The tests were not
    weak: `pyproject.toml`'s `pytest pythonpath = ["src"]` is resolved against the rootdir and
    *outranks* `PYTHONPATH`, so every run imported the unmutated tree. A gate whose harness is dead
    reads exactly like a gate that is merely permissive, and the wrong conclusion from 0/29 would
    have been "add more tests". The battery now runs pytest inside a copy of the repo, requires the
    unmutated baseline to be green, and aborts if the *first* mutation survives. Fixed harness:
    **29/29**. Two of the three initial survivors were real holes once the mutations actually landed
    (boolq rows whose question *ends* with a suite suffix had that tail edited; boolq/mnli frames
    whose nine variants all ended the same way), and one was an **equivalent mutant** — every
    `SUITE_SUFFIXES` entry starts with a space, so deleting `.rstrip()` changes no output on this
    data. Kept the call, logged why, and dropped the mutation.
13. **`python -m myna.train --help` crashed.** Present since before P0, found only because the
    paraphrase battery needed a CLI sweep: argparse formats every `help=` string through `%`, and
    `--group-sample`'s measured "~1.4% of updates" is not a legal conversion — `TypeError: %o
    format`. Written `%%`, so the number still prints. Same sweep found `myna.serve` importing
    `uvicorn` *before* `parse_args()`, which made the optional `serve` extra a requirement for
    reading the usage text. Both now pinned by `tests/test_cli_help.py`, since "one command per
    result" (G7) is worth nothing if the command cannot be asked what it does.
14. **"Size the batch as `free_bytes / (0.8 MiB × state tokens)`."** Right price, wrong denominator:
    that is the *state scan* alone, while a `--row-batch` step holds the question branch at the same
    time — 1.6 MiB per question-token cell, from the same `bench/mem_profile.py` measurement, and
    `--max-q-cells` is what bounds it. §9.9 already said the M5 died with both axes live, and the
    P3 sizing bullet still priced one, so the plan would have cheerfully picked a batch that fits
    the scan and then OOM'd in the probe. Fixed: the reserve is subtracted before rows are counted,
    a budget the reserve alone eats is a **refusal that names `--max-q-cells`** rather than a silent
    batch of 1, and the printed plan line says how much of the headroom went to the question branch.
    The rule this breaks is the general one: a per-axis price list is not a budget until the axes
    that are simultaneously live are summed.
16. **"Macro uniform-chance 0.3292."** The 3g harness reproduced §4.2's majority floor *exactly*
    (0.4331) and its uniform floor as **0.3321**. Not a rounding story: a (source, question) cell is
    asked by several question-sets whose option counts **differ**, because kev randomizes distractors
    row to row — agnews/topic appears with 4 options for 104 test rows and 5 for 12, banking77/intent
    with 77 and 78, contrastive/decision with 2, 3 and 4. So `1/options` has no single value per cell,
    and the old figure was one of the available weightings picked implicitly by an ad-hoc pass that
    left no script behind. Three alternatives were measured against it on the same split before
    settling (mean over sets 0.3267, minimum per cell 0.3139, first set read 0.3397 — an ad-hoc probe,
    published here only to show the spread, not as a result): the harness takes the **row-weighted**
    0.3321, prints the six mixed cells with their full
    option histograms so a reader can audit it, and a test pins 0.433/0.332 together. G1 is judged
    against the majority floor, which is unchanged; the correction matters because a +0.15 margin is
    quoted, and a margin against a floor that cannot be recomputed is a claim, not a measurement.
17. **"9 groupable vs 2 per-row" is a fact about instruction strings, not about question-sets.** The
    first pass of `report.strata()` keyed on the adapter's group signature and classified **11 of 11**
    sources as per-row-instruction — it reported the artifact and concluded the suite had no
    structure. The signature *must* include the per-option `criteria` descriptions (kev varies them
    row to row, they are part of the option text, and two rows with different criteria genuinely
    cannot share a question tensor — that is why P2 batches on it), so on the test split agnews's 116
    rows sit in **113** exact-signature sets while sharing only **8** instruction strings, and 0.05 of
    its rows live in a reused set; boolq's 80 rows are 80 sets *and* 80 distinct instructions, which
    is the actual per-row case. Measured on the same rows, the two counts diverge for nine sources
    and coincide for two. Recorded because the same key produced §9.3's "1,146 question-sets, median
    pool 1, therefore interference" story, and because P1's topology line (17,112 pilot question-sets,
    96% singletons, against 10,668 schemas) is the same measurement on the train split — true as a
    *batching* fact, misleading as a taxonomy. The durable rule: a structure claim about a derived
    corpus is checked against the generator's randomization, and a batch key is not a taxonomy.
18. **A mutation battery is a measurement, so nothing may move while it runs.** The first two passes of
    `bench/mutation_report.py` ran while `src/myna/report.py` and `tests/test_report.py` were still
    being strengthened, and because the harness re-copies `src`+`tests` for every mutation, one log
    blended code states: eight survivors and a `BAD-PATTERN` whose pattern matches exactly once in the
    file on disk today, then **53/54** with one survivor ("distinct instructions reported as the set
    count"). All of
    it was real signal about the *older* tests and unusable as a witness for these. Two rules follow:
    freeze the code and the tests for the whole run, and check the reported mutation count against the
    file's list length before quoting it. The clean pass — repo untouched, battery launched once and
    left alone — is **54/54, exit 0**, and it is the only number from this gate in a commit message.
    of `bench/mutation_kaggle_bundle.py` caught **21/25**, and the four survivors were all the same
    species of self-flattery: `WORKING` was only ever exercised through a monkeypatch, so a literal
    `/tmp` default — a checkpoint Kaggle would never save — passed; `check()`'s sha branch was
    probed only by *appending* to a staged file, which changes the size, so a size-only comparison
    survived; and the manifest's provenance plus the build-time verification were never reached,
    because every test staged a copy that had already succeeded. Three tests now pin the literal
    `/kaggle/working`, flip one byte **in place** (same size, different corpus), and make `copy2`
    itself lossy, which is the only way to ask "does the manifest describe the corpus, or the stage?"
    Second pass: **25/25**. The general form: a test that exercises a helper on a path the caller
    never takes is documentation, not a gate — and a mutation battery is the only cheap way to find
    out which paths those are.



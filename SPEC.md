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
| p50 latency, 1 question (87-token state) | **100.7–117.7 ms** measured here on the M5 CPU · 39.5 ms on their T4 (their measured) | **33.1 ms** myna on the same box and process = **3.04×**, and inside their T4 number | measured (§5 P4 4a, `runs/latency_matched.md`) |
| p50 latency, 50 questions | **2,923–3,589 ms** here · 771.3 ms T4 (their measured) | **812 ms = 3.60×** same box; against their *T4* number myna takes 5–13% *longer* (812–870 ms vs 771.3), so the ≥ 5× projection did **not** hold (§9.21) | measured (4a) |
| p50 latency, 1 question, state already scanned | no such API — every call re-reads the state | **14.9 ms = 6.78×** laya's same call | measured (4a) |
| input window | **512 tokens** english / **1024** typed-decisions (`max_len` 1024 + `head_max_len` 256, read from the checkpoint config by the 4a harness) | the *state* is measured to 16,384 tokens — scanned, held at 576 KiB, answered from, at cost that grows only in the scan (§9.22). The *decision* at that length is not: v0's needle accuracy is 0.312 at 1k and 0.062 at 16k against a 0.167 floor (§9.25) | measured (state), **not met** (accuracy, G4) |
| retained state | KV-free but re-encodes; window-bounded | **576 KiB fixed, independent of length** | measured |
| calibration ECE | **0.081** after temperature fit (their measured) | **≤ 0.02**; 0.0076 on v0 **dev**, whose temperature was fit on those rows — read as optimistic (§9.11) | measured (v0), level disputed |
| refusing a decision | nothing in the response shape refuses: `system_one` returns a choice for every question, and a caller can only threshold `confidence` itself | `Myna(abstain_below=t)` withholds the commitment with a measured `reason`; risk/coverage on `calibration.jsonl` runs 0.349 → 0.596 as coverage falls 1.00 → 0.10, and **no rung reaches G5's 0.95** on v0 (§9.24) | measured (5a/5b), gate not met |
| browser artifact | ~1.7 GB fp32 / ~420 MB int8 (arithmetic) | **≤ 20 MB int8**, cold load ≤ 3 s | projected, gated on ONNX export |

The latency cells are one process on one M5 laptop, `device=cpu`, both engines fp32. An absolute p50
from that laptop is a range, not a point — two committed runs of the same harness differ by up to
22% on laya's 50-question row and 14% on myna's, because other sessions share these cores
(§9.23). So each ratio cell is the **minimum across the committed runs** (`latency_matched.md` and
`latency_matched_run1_superseded.md`), taken per row, and every cell names the run it came from.
Quote the ratio, never the millisecond figure.

### 2.2 Acceptance gates — the project is not finished until all pass

| # | gate | pass condition | status |
|---|---|---|---|
| **G1** | Accuracy beats the witness | macro decision-v2 test ≥ 0.70 on the frozen split, per-source table published, ≥ 0.4331 majority floor by ≥ +0.15 | open |
| **G2** | Latency claim survives equal-footing re-measurement | same box, same window, same question count, direct `Agent` call not `Router`; published ratio recomputed from that or withdrawn | **met** (4a/4c): one M5 process, both fp32, `Agent.system_one`, ladder capped inside laya's 1024 window — **3.04×/3.21×/4.12×/3.60×** at 1/5/10/50 questions end-to-end and **6.78×/4.67×/4.75×/3.65×** on the ask-only path, each the minimum of two committed runs, `runs/latency_matched*.md`. G1 is *not* implied: v0's accuracy still fails (§4.2) |
| **G3** | Deployment works for real | ONNX browser build answers a live page's decisions; bytes + p50 + cold-load measured in Chrome | open |
| **G4** | Long-context is *correct*, not just cheap | needle-style typed decision ≥ 0.90 at 4k and ≥ 0.85 at 16k state | **not met, and not yet measurable on this box.** v0's needle curve (`runs/needle_myna-v0.md`) reads 0.188 at 128 tokens against a 0.167 floor — at chance on the shortest rung — so the longer rows are decay of nothing, and the harness now says so itself rather than printing a table (§9.25). The 16k *state* is measured and cheap (§9.22); the 16k *decision* needs the long-context checkpoint, which is 7b / `KAGGLE` |
| **G5** | Useful confidence, with abstention | risk/coverage curve on `calibration.jsonl`; ≥ 0.95 accuracy at ≥ 60% coverage on banking77 + dbpedia14 + trec | **harness met, gate not met.** Abstention ships: `Myna(ckpt, abstain_below=t)` withholds the commitment and prints the measured reason, the fallback seam labels every answer with the engine that committed, and the curve is a committed artifact (`runs/risk_coverage.md`, 448 rows / 568 questions). Accuracy on v0 climbs **0.349 → 0.535** as coverage falls 1.00 → 0.20 — the confidence ranks the answers, on a checkpoint that cannot do the task — and setting the 0.693 floor on the engine abstains on **227** questions where the curve withheld **227**. But **no rung reaches 0.95** (max 0.596, at 10% coverage) and G5's own three sources sit at 0.000 / 0.025 / 0.150, so the pass needs the `KAGGLE` checkpoint (§9.24) |
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
(streaming sessions, and the abstention floor: `abstain_check`, `policy` echo) · `fallback.py`
(the `Decider` seam, the `DECIDERS` registry, `laya_spec`, and the per-decision `model` label) ·
`rlcd.py` (strictly-proper scoring + KL leash) · `longctx.py` (needle
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
that runs the package on a real 3.11 interpreter and exits 2 rather than skipping),
`bench_latency_matched.py` (the 4a/4b harness: one process, both engines, matched inputs, `flock`ed
output) and its gate `mutation_latency_matched.py` (34 mutations), `risk_coverage.py` (the 5b
risk/coverage curve, its floor re-run and the G5 verdict) and its gate `mutation_p5.py`
(49 mutations across the engine's commitment, the seam's labels, the transport and the curve),
`eval_needle.py` (the 7a recall ladder — `--lengths` so the published ladder *is* the command that
ran it, and `baseline_verdict`, which refuses to let a curve that starts at the uniform floor be
read as decay) and its gate `mutation_longctx.py` (20 mutations).
`kaggle/`: `run.py` (one-command entrypoint: corpus discovery, the measured flag set, `--resume` only
when a snapshot exists, tee to `train.log`, `run.json`, non-zero propagation), `package_dataset.py`
(stage the pilot corpus, verify every byte against the corpus, print the upload command, never run
it), `requirements.txt`.
`tests/`: **264 passing + 1 KEV-gated parity test** (skipped here, green wherever `kev` is
installed). The P5 gate is `test_abstain.py` (the flag against the distribution the same answer
prints, monotonicity in the floor, noul's confident "no", the reason's measured numbers, the
policy echo), `test_fallback.py` (only the abstained questions reach the secondary, the label
follows the decider rather than a literal, both-refused reported, the laya schema shim held equal
to `bench/eval_laya_real.py`'s) and `test_risk_coverage.py` (a synthetic oracle whose curve the
test knows in advance: informative, uninformative and anti-calibrated heads, both G5 boundaries
landed on exactly, and the floor's own row belonging to the kept set on both sides of the
comparison). The P3 gate is `test_memory_plan.py` (sizing arithmetic, the refusal, the CUDA headroom
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
JSONs live in `runs/`, and the small tables in it (`.json`/`.md`/`.log`) are committed — checkpoints
and `*.pt` dumps are not (G7's "witness JSONs committed" is satisfied by the ignore rule, not by an
intent; an earlier draft of this section said the opposite and meant it).

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

**Their T4 ladder does not transfer to this box, so 4a measured laya here rather than citing it.**
Same checkpoint (revision `55cf4c4`, the one their server-CPU table used), `Agent.system_one`, fp32,
weights fp32 with autocast off, `max_len` 1024 / `head_max_len` 256, 421.29M params, 8 intra-op
threads, two committed runs (p50, `run3`/`run1`): **100.7 / 117.7 ms** for one question, 272.0 / 305.5
at 5, 506.3 / 574.9 at 10, 3,589.5 / 2,923.4 at 50 — 2.5–3.0× their T4 figure at one question and
3.8–4.7× at 50, i.e. **57–71 ms per extra question here against their ~14.9 ms**. Both vendors'
latency is now from the same silicon; the T4 column stays labelled as theirs and is never used as
myna's denominator, which is exactly why §2.1's 50-question row had to be restated (§9.21). A third
run of the same command, whose artifact the current one overwrote, read within 0.4% of these at one
question and 32% below at 50 — this column is machine-state dependent and the ratio is the
publishable quantity (§9.23).

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
- [x] Re-run myna vs laya on one box, one process, direct `Agent` call (not `Router`), matched
      state length / question count / dtype / options-per-question.
      **4a witness:** `bench/bench_latency_matched.py`, one M5 process, device `cpu` for both, both
      weight dtypes *reported* rather than assumed (`torch.float32` each, laya's autocast off), the
      same `str` state object and the same question `dict` handed to both engines (3-option `choice`
      alternating with `noul`, exactly laya's own `bench_latency.py` ladder), laya called as
      `Agent.system_one` on checkpoint `55cf4c4`. Ladder capped at 1024 tokens so **no row compares
      myna reading a document against laya reading a prefix of it** — the harness derives
      `usage.input_tokens / questions` per row and would label a plateau `laya_truncated`; no row
      was labelled. Reproduce with
      `PYTHONPATH="<laya checkout>" .venv/bin/python -u -m bench.bench_latency_matched` (without
      laya importable the harness runs the myna half and **withdraws the ratio**, which is the
      §9.1 behaviour, not a failure). Artifacts `runs/latency_matched.{json,md}` (run 3) and
      `runs/latency_matched_run1_superseded.{json,md}` (run 1), both committed. Ratio, recomputed:
      **3.04× / 3.21× / 4.12× / 3.60×** end-to-end at 1 / 5 / 10 / 50 questions, and
      **6.78× / 4.67× / 4.75× / 3.65×** against laya's same call from myna's ask-only path — each
      cell the **worse of the two runs**, because the two runs disagree by up to 22% in absolute p50
      while other sessions share these cores, and a busy laptop must not get to pick the flattering
      number (§9.23). The internal drift check re-measured the first shape at +1.9% (myna) and
      +12.6% (laya) in run 3, −0.7% / +1.8% in run 1 — that spread is why the absolute column is not
      the claim.
      Gate: `tests/test_latency_matched.py` **26 tests**, `bench/mutation_latency_matched.py`
      **34/34 mutations caught, exit 0** on a frozen repo (`runs/mutation_latency_matched.log`). The
      battery's first pass scored 30/31 and the
      survivor was an uncovered branch (`fit_cost` guarded the question axis against a zero-variance
      column but not the state axis), now pinned both ways; the three newest mutations are the
      single-instance lock, which exists because a duplicate harness run was caught mid-measurement.
- [x] Separate fixed per-call overhead from per-token and per-question cost.
      **4b witness:** least squares on the ladder, `t = fixed + per_state_token·L + per_question·Q`,
      run 3 with run 1 in brackets.
      myna **observe**: 360.4 [548.5] µs/state-token, R² 0.9979 [0.9996] — with a *negative*
      intercept, because within a 256-token chunk the scan is quadratic: doubling the state costs
      2.85× [3.40×] at 66→133 tokens and 2.82× [2.85×] at 133→265, then stops compounding once the
      state spans several chunks (1.93× [1.72×] at 265→530, 1.97× [2.20×] at 530→1060; §9.22, and
      the reason myna's ~33 ms decision figure is a *short-state* result).
      myna **ask**: 10.12 [9.29] ms fixed + 9.59 [12.15] ms/question with a state slope of −10.3
      [−4.0] µs/token, i.e. zero within run noise — the fixed-size state claim, measured, on both
      runs. myna **end-to-end**: 4.82 [1.89] ms + 368.4 [528.2] µs/token + 8.94 [9.74] ms/question,
      R² 0.9915 [0.9794].
      **laya's additive fit gives R² 0.7356 [0.7297]**, which is the structural point: its cost is
      state × questions (one sequence carrying the whole state, per question), so the ladder's
      530-token rows read 289 [507] ms at 1 question and 2,563 [3,151] ms at 10.
- [x] Until then the only latency sentence allowed in public: *myna on a laptop CPU has roughly the
      per-decision latency laya reports on a datacenter T4.*
      **4c: superseded by measurement, partly.** The first half of that sentence is now *stronger
      than the constraint and still conservative*: at a 1-question / 87-token decision myna measures
      **33.1 ms** on the M5 CPU (run 1: 38.3 ms), inside laya's published T4 39.5 ms on **both** runs,
      and **3.04×** faster than laya measured on this same box. The T4 margin stays context only —
      cross-silicon, never a denominator (§4.3) — so what G2 actually carries is the same-box ratio.
      What does **not** survive is §2.1's other projection (**≥ 5× at 50 questions, §9.21**): myna's
      50-question p50 is 870 ms (run 1: 812 ms) against laya's 771.3 ms on a T4, i.e. myna takes
      13% (5%) *longer* than laya took on datacenter silicon, and against laya on this box the ratio
      is 3.60× — with a per-extra-question marginal of 17.1 ms (15.8) against laya's 71.2 (57.3), so
      even the *slope* ratio never reached 5× here.
      Also withdrawn with it: the README's 33×–239× long-context column, which compared myna
      reading a document against laya truncating it (**§9.20**).
      And the sub-40 ms figure is a *1-question, 87-token* result: at 50 questions the state is
      irrelevant and the question branches are the whole cost (ask-only 787 ms, run 1: 801, §9.22).
      Quote a shape, not a number.

### P5 — Abstention and risk/coverage *(task #16, gates G5 — the flagship scenario)*
- [x] **5a** Confidence threshold tuned on `calibration.jsonl`; emit `abstain` with the reason.
      `Myna(ckpt, device, abstain_below=t)`: the default is `None`, which never abstains — the latency
      and parity paths must answer every question, because an engine that can quietly stop answering
      would invalidate those measurements (§4.3, `bench_latency_matched.require_full_answers`). With a
      floor set, the committed field goes `None` (`choice` / `level` + `score` / `yes`) and every
      answer carries `confidence`, `margin` (the runner-up gap), `abstain` and a `reason` quoting the
      measured numbers, so a caller that ignores the flag fails at the seam instead of shipping a
      label the model refused to commit to. `ask()` echoes `policy.abstain_below` and
      `policy.abstained`; `myna.serve --abstain-below` and `/v1/health` carry it, because the HTTP
      surface is where a fast path could go silent on the way out. noul is the trap this caught in
      draft: its `confidence` is the probability of the *side committed to*, `max(p, 1-p)`, not
      `p(Yes)` — pricing it the other way abstains on the model's surest "no"s, and
      `test_a_confident_no_is_not_an_abstention` pins it by sharpening the real head
      (`temperature=1e-3`) rather than a dictionary. **Witness:** `tests/test_abstain.py` (19) +
      `tests/test_serve.py` (6, four of them the abstention path over HTTP and the CLI flag
      that feeds it).
- [x] **5b** Risk/coverage curve in the README table, not just an accuracy number.
      `bench/risk_coverage.py` — one pass over a labelled split, ranking each question by its
      committed-side probability, with the per-source cut G5 is written against. Two things make the
      table mean something rather than being a plot: the ranking quantity is **recomputed** from the
      printed distribution and cross-checked against the engine's own `confidence` field per answer
      (a divergence raises), and the chosen floor is **re-run on the engine**, which must abstain on
      exactly the count the curve withheld. Measured on v0 / `calibration.jsonl` (448 rows, 568
      questions): accuracy **0.349** at full coverage rising to **0.596** at 10% — the confidence
      carries information on a checkpoint that cannot do the task — and the 0.693 floor re-run gives
      **227 engine abstentions against 227 curve rows below the floor** (§2.2, §9.24). **Witness:**
      `runs/risk_coverage.{md,json,log}`, `tests/test_risk_coverage.py` (21 tests against a synthetic
      oracle, including both G5 boundaries landed on exactly).
- [x] **5c** Fallback seam to a stronger model, with the chosen model labelled per decision. A fast
      path that is uncertain must never be silently load-bearing. `src/myna/fallback.py`:
      `Decider` is the seam, `DECIDERS` the registry (`myna`, `laya`), and `Fallback.predict` re-asks
      the secondary **only** the abstained questions, labels every answer with the engine that
      committed to it, and reports `routing.still_abstained` when both refuse rather than dropping the
      row. `LayaDecider` wraps `Agent.system_one`, not `Router` — §9.1's equal-footing rule applies to
      the fallback path too, since a router's label would name the router rather than the thing that
      answered. **Witness:** `tests/test_fallback.py` (15).
- [ ] **5d** G5's pass condition measured on banking77 + dbpedia14 + trec — needs the real-trained
      checkpoint, so `KAGGLE`-gated. The harness is provably correct against a synthetic oracle first
      and mutation-checked second: **49/49 mutations caught** (`bench/mutation_p5.py`,
      `runs/mutation_p5.log`) across the engine's commitment, the seam's labels, the transport and
      the curve's boundaries. v0's own numbers are in §2.2 and they are a `NOT MET`, not a pass: its
      full-coverage accuracy on the gate's three sources is 0.000 / 0.025 / 0.150, which is *under*
      both floors of those cells on this split (uniform 0.013 / 0.071 / 0.167, majority-class
      0.075 / 0.125 / 0.300 — both recomputed from the shipped rows by `myna.report.cell_stats`, not
      quoted from the class counts). Abstention cannot rescue accuracy that is below guessing, and
      the artifact says so in the same breath as the curve.

### P6 — Browser/on-device deployment *(task #18, gates G3)*
- [ ] Split ONNX export (trunk + pointer head) mirroring `laya-ts/scripts/export_onnx.py`, with
      torch-vs-ONNX parity ≤ 1e-4.
- [ ] Measured in Chrome via `onnxruntime-web`: download bytes fp32/int8, cold-load ms, p50 per
      decision on a real page.
- [ ] MLX int8 path for Apple, re-measured after any parameter growth.

### P7 — Long-context proof *(gates G4)*
- [x] **7a** `bench/eval_needle.py` recall-vs-length curve, swept 128 → 16,384 on the checkpoint
      this box has (`runs/myna-v0`, `--lengths 128 1024 4096 8192 16384 --n 16`, seed 0):
      **0.188 / 0.312 / 0.312 / 0.125 / 0.062** against a 0.167 uniform floor. The measured answer
      is that G4 cannot be judged from this run — v0 is at chance on the *shortest* rung, so there
      is no recall to decay, and the harness now refuses its own curve in print
      (`baseline_verdict` → "G4: NOT MEASURED here") instead of publishing a table that reads as
      long-context failure. This is the correction §9.25 records: the 16k number §2.1 carried was
      a *state and cost* measurement (§9.22), never a reading one. Witnesses:
      `runs/needle_myna-v0.{md,json,log}`; the gate is `tests/test_longctx.py` (13 — the generator's
      evidence-uniqueness and length target, the scorer checked against an exactly recomputed share,
      needle positions that actually spread, and the baseline guard's own boundary) and
      `bench/mutation_longctx.py` (**20/20 caught**, `runs/mutation_longctx.log`). Two things this
      pass also fixed in the harness rather than the model: `--lengths` so the published ladder is
      the command that ran it (the old ladder stopped at 8k, one rung short of the claim), and a JSON
      witness beside the markdown (G7). The guard's first battery pass scored 19/20, and the
      survivor was mine rather than the tests': I had written `>= chance + margin - 1e-9` to spare a
      float-equality annoyance, which made "exclusive versus inclusive" untestable by construction.
      The tolerance is gone and the boundary is now checked on exactly representable values
      (0.25 + 0.125 = 0.375), which is the only honest way to claim a boundary is inclusive.
- [ ] **7b** Train the 4k truncated-backprop checkpoint (`--long-context --init`) — code complete,
      committed and tested; `KAGGLE`. G4's numbers come from re-running 7a against *that* checkpoint,
      and only if its 128-token rung clears the floor by the margin the guard requires.

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
   **This entry's own remedy was wrong too** — no artifact ever produced "~15× at 1k tokens", and the
   number that would have, laya's ~500 ms plateau, is its truncation window (§9.20). The lesson is
   not "the ratio was inflated"; it is that a correction which replaces one unmeasured competitor
   number with another, while the README that carried the first stays published, is not a correction.
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
19. **"The Kaggle bundle is verified at build time" was a claim about a code path no test reached.** The
    first battery of `bench/mutation_kaggle_bundle.py` caught **21/25**, and the four survivors were all the same
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
20. **The README's streaming table outlived the correction that named it.** §9.1 flagged the 239×
    mislabel and the table stayed up, with its laya column intact, until 4a. The column-by-column
    failures, which §9.1 did not see:
    * The 2.1×–239.3× "myna speedup" column is **myna against itself** — `myna-full / myna-stream`,
      both myna, from `bench/bench_stream.py`. It sits in a table headed "Measured on an Apple M5"
      with a laya column beside it, and the prose around it ("33× faster than laya", "the streaming
      column is flat by design… laya pays a ~500 ms floor") reads it as a competitor ratio. It never
      was one; the artifact `runs/bench_stream.md` names its own column `speedup` and computes it
      from two myna modes.
    * The "~500 ms floor above 512 tokens" is **truncation, not a plateau**. laya's typed-decisions
      checkpoint has `max_len` 1024 (`head_max_len` 256), and `system_one`'s own docstring says it
      truncates silently beyond the window — so every row from 1,024 to 8,192 tokens fed it the same
      ~1k-token input and cost the same ~510 ms. Its cost stopped growing because its *input*
      stopped growing. 4a's `laya_tokens_per_row` exists precisely to catch this shape.
    * "myna answers a 16k-token observation ~17× faster than laya answers an 8k one" divides myna
      reading 16,384 tokens by laya reading the first ~1,024 of them. Two systems doing different
      amounts of work is not a comparison, and no re-labelling saves it.
    * The laya half also went through `Router`, not `Agent.system_one`, and through a local
      `laya_questions()` shim that *deleted* the `criteria` of every `noul` question — so options per
      question were not matched either, on the one axis G2's own row names.
    The rule this buys: a column header has to name both operands, and a plateau in a competitor's
    cost curve must be explained by that competitor's *input* before it is published as that
    competitor's ceiling. Sharper still, from `PLAN.md`'s own reading of the same table: it said
    "laya's re-encode plateaus at ~510 ms (**its 8192 cap truncation region**) and can't serve long
    states at all. At 16k myna answers ~17x faster than laya answers at its 8k maximum" — the
    truncation was *known* in the sentence that used it as a denominator, two windows too low, and
    the same section's third reading ("even myna-full beats laya above 2k") is contradicted by the
    row printed directly above it (965 ms vs 512 ms). Recording a fact in a log is not the same as
    applying it; the gate has to be a script that refuses to print the claim.
    The honest version of the same architectural point survives 4a untouched:
    myna's answer path really is length-independent (§9.22), and laya really has no cached-state API
    — which is a capability difference and does not need a ratio to be interesting.
21. **The "≥ 5× at 50 questions" target compared a laptop to a datacenter GPU and lost.** §2.1 carried
    `771.3 ms (their measured)` in the baseline column against `≥ 5× faster, same box` in ours — the
    cell contradicted itself, since nothing about it was on one box until 4a. Measured: myna's
    50-question p50 on the M5 CPU is **870 ms** (run 1: 812), which is **1.13× (1.05×)** laya's T4
    wall time — myna takes slightly *longer* than laya did on datacenter silicon, at 29× fewer
    parameters — and **3.60×** against laya measured on this box. The per-extra-question marginals
    (myna 17.1 [15.8] ms, laya 71.2 [57.3]) bound the marginal ratio at 3.6–4.2×, so no reading of
    the same data reaches 5×. The mechanism of the miss is the mixing itself: laya's per-question
    cost on this CPU is 3.8–4.8× its T4 cost, while myna's whole call is within 13% of that T4 call,
    so projecting a laptop-scaled myna against a T4-scaled laya manufactured a multiplier. The target
    is retired as measured, not re-cut: the public claim is **3–4× on one box** plus the
    length-independence result (§9.22).
22. **"The cost is flat in context length" is true of the answer and false of the scan.** Both
    committed runs fit the `ask` path's state slope to **−10.3 [−4.0] µs/token** — indistinguishable
    from zero — and answering one question from a cached state costs 14.7–17.4 ms p50 whether that
    state is 66 or 1,060 tokens. That is the fixed-size-state claim with a witness rather than an
    architectural assertion, and it is the result that survives every retraction in this section. The
    *scan* is the other half, and §2.1's old "flat cost" cell was simply wrong: within a 256-token
    chunk `gla_chunked` evaluates the quadratic form, so 66→133 tokens costs **2.85× (3.40×)** for
    twice the input, and only stops compounding once the state spans several chunks (1.93× [1.72×],
    then 1.97× [2.20×]). What that does to the headline: myna's sub-40 ms decision is an 87-token
    result, and at 1,060 tokens with one question the same-box lead narrows to **1.45× [1.25×]** —
    while at 10 questions on the same state it is **11.6× [11.2×]**, stable across runs. So the
    advantage lives in the question axis, robustly; in the state axis it lives only as far as the
    scan's shape lets it. Two follow-ups this exposes — a larger `INFER_CHUNK` for no-grad scans, and
    a linear-in-chunk scan form — are **unmeasured, so nothing is claimed about either**.
23. **An absolute p50 from this laptop is not reproducible, and one run was contaminated by me.** Three
    runs of the same command, laya then myna, at 1/5/10/50 questions: run 1 **117.7/305.5/574.9/2,923.4**
    and **38.3/95.3/139.6/811.8**; run 2 (its artifact was overwritten by run 3)
    **101.1/268.3/482.8/2,426.0** and **35.4/73.3/113.2/658.1**; run 3 **100.7/272.0/506.3/3,589.5**
    and **33.1/74.9/120.8/870.0**. The 50-question row alone spans 22% for laya and 7% for myna, and
    the e2e ratio for one shape has been reported as 2.86×, 3.04× and 3.08×. Two causes, one of them
    mine: run 2's ladder was timed while I ran `--help` checks in the same repo, and all three ran
    while other agent sessions ran pytest on these same 10 cores. What changed as a result: the
    harness now takes a `flock` on its output path and refuses a second copy (this exists because a
    duplicate harness PID pair *was* running for ~40 s in this session, interleaving one log), and
    every ratio cell in §2.1, §2.2 and §4.3 is the **minimum across the committed runs**, so shared
    cores can hurt our number but never improve it. Not fixed, and stated plainly: this is a
    development laptop, not a benchmark box, and stable absolute latencies need dedicated hardware —
    G2 is gated on the ratio, so it does not depend on getting one.
24. **A risk/coverage curve can be a correct measurement of a model that cannot do the task, and
    the README must not let that read as a G5 pass.** 5b ran v0 over `calibration.jsonl` — 448 rows,
    568 questions, 168 question-sets (`runs/risk_coverage.md`) — and the shape is real: accuracy
    climbs **0.349 → 0.413 → 0.535 → 0.596** as coverage falls 1.00 → 0.60 → 0.20 → 0.10, so the
    committed-side probability ranks the answers. What is *not* real is any claim of usable
    confidence: the ceiling is 0.596 against a 0.95 target, the three sources G5 names measure
    0.000 (banking77), 0.025 (dbpedia14) and 0.150 (trec) at full coverage — each at its own
    guessing floor, because v0 never trained on this corpus (§4.2) — and raising a threshold on a
    model that cannot answer is not the product G5 describes. So the artifact ships labelled
    *harness witness, not a G5 pass*, `g5.pass` is `false` in `risk_coverage.json`, and the gate
    stays open pending the `KAGGLE` checkpoint (5d). Two further things this run settled, both
    against my own first draft of the sentence above:
    * **The curve is not monotone.** Rung 0.90 reads 0.376 and rung 0.80 reads 0.374. That is 57
      rows changing hands between two adjacent cuts of a 568-row sample — not a bug and not
      smoothed: the published table prints every rung, so a later reading cannot quietly
      interpolate the dip away.
    * **"It agrees with the engine" has to be an executed check, not a property of the plot.**
      The floor the 0.60 cut selects (0.693) is re-applied to `Myna(abstain_below=0.693)` in the
      same run, and the engine abstains on **227** questions where the curve puts **227** below
      that floor. The reason this is a test and not a footnote is the boundary: a floor is read
      off one real answer's confidence, so that answer sits exactly on the line, and `<` versus
      `<=` on either side of the comparison moves it — which is why both halves now go through
      one predicate (`at_or_above`) and `test_the_floor_belongs_to_the_kept_set_and_to_nobody_else`
      pins the same float against the engine's own `abstain_check`. The same class of boundary sits
      one line further down in the readout: noul at `p(Yes) = 0.5` commits to *yes*
      (`pl[1] >= 0.5`), so the harness scores the tie the same way and
      `test_a_level_tie_is_broken_the_way_the_engine_breaks_it` holds it there — because a
      coin-flip row is precisely the row the two sides of this pipeline must not disagree about.
    * **The battery earned its keep once, on the test side.** 49 mutations were written against
      49 claims; the first pass scored 48/49, and the survivor was `laya_spec`'s noul branch being
      deleted — which changed nothing because the fixture's noul question carried no `criteria`,
      the one field the branch exists to drop. The mutation was not wrong, the *sample* was: the
      suites ship `yelp`/`imdb` noul rows with a criteria dict, so the fixture now uses that shape
      and the deletion is caught. A battery's survivors are read as holes in the tests, and a hole
      in a fixture is as load-bearing as one in an assertion.
25. **"16,384 tokens" was measured about the state, and never about reading it — and the first
    needle curve says the second claim is false for every checkpoint on this machine.** §2.1's window
    row and `README.md` both carried 16k as if one number covered both questions. It does not. The
    mechanism is genuinely measured: the state is 576 KiB at any length, the scan scales linearly,
    and the answer path is flat (§9.22). Comprehension is a different measurement, and
    `bench/eval_needle.py` now runs it — a decisive sentence buried at a random position in filler,
    16 needles per rung, seed 0, `runs/myna-v0`, 6-way so the uniform floor is 0.167:

    | state tokens | 128 | 1,024 | 4,096 | 8,192 | 16,384 |
    |---|---|---|---|---|---|
    | needle accuracy | 0.188 | 0.312 | 0.312 | 0.125 | 0.062 |

    v0 is at the floor *at the shortest rung* (0.188 against 0.167, where one standard error at
    n=16 is 0.093), which means the rows after it cannot be read as decay: there is no recall here
    to lose. The 16k figure is worse than chance rather than merely low, and neither of those is a
    long-context result — it is an out-of-distribution task wearing a long-context costume, because
    no checkpoint on this box was ever trained to answer a buried-desk question, let alone at 16k
    (v0's training states are ~90 tokens). What §2.1 now says, in its own row, is
    *measured (state)* / **not met (accuracy)**; the 16k comprehension claim is G4's, it is gated on
    the `KAGGLE` long-context checkpoint (7b), and until that checkpoint exists the honest sentence
    is "myna carries a 16k-token state at fixed size" and not one word about understanding it.
    The harness change this forced is the part worth keeping: `eval_lengths` now refuses its own
    curve (`baseline_verdict`) when the shortest rung does not clear the uniform floor by the
    margin, and prints `G4: NOT MEASURED here` instead of a table a reader could quote as decay.
    A low curve and a meaningless curve print identically otherwise, and only one of them is
    interesting in the direction people want.



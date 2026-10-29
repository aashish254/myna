# Myna: a small, auditable typed-decision engine — what a 16.9M-parameter model does and does not do on a frozen decision suite

**Working draft for owner review. Not for submission. Local only.**
Every number in this file is **measured** in this repository — either read out of a committed artifact or
recomputed from a checkpoint on disk — and carries its witness beside it. Nothing is projected or
borrowed. SPEC §9.64 (commit `bffefba`) is the source of truth for the parameter counts, the stage ladder
and the dose denominators used here; where this draft cites a figure not in §9.64 it cites the artifact
that prints it. External literature is not cited anywhere in this draft: every place that will need a
reference is marked `[REF NEEDED — to be verified]` and listed at the end.

---

## 1. Abstract

We report Myna, a 16,926,848-parameter typed-decision engine (total parameters, embedding table included,
counted as `sum(p.numel() for p in model.parameters())`) that reads an observation, emits a typed decision
over a fixed option set, and ships as a file small enough to run inside a browser tab. Our contribution is
diagnostic, not a state-of-the-art claim: we pre-registered seven release gates against a frozen
decision suite and ran them, and **the accuracy, long-context and abstention gates all fail on the
measured artifacts — G1 is not met (macro 0.4893 against a 0.70 target, +0.0563 over a 0.4331 majority
floor where +0.15 is required), G4 is not met (0.1833 at 4,096 state tokens against a required 0.90,
0.150 at 16,384 against a required 0.85), G5 is not met (best accuracy 0.596 only at 10.04% coverage
against a required 0.95 at ≥60% coverage) — while G6 remains open** (v0's own test macro is 0.9523, above
the 0.94 level, but no v1 checkpoint has been run against it). What does hold is measured and reproducible:
a 59.32 MiB artifact that loads in 253.9–394.4 ms and answers in a fixed 589,824-byte (576 KiB) decision
state; a matched-condition throughput advantage over a 421.29M-parameter competitor on the same inputs
(33.083 ms against 100.691 ms at one question); and a release-gate registry in which 7/7 verdicts and
34/34 rows re-derive from committed witnesses. We report the failures as the primary result, because the
instrument that detected them — a needle-recall harness that refuses to let a flat-at-chance curve be read
as decay — is the part we can defend.

## 2. Claim and introduction

The claim this paper makes is narrow: **a typed-decision engine of this size can be audited end to end, and
auditing it shows where it stops working.** The engine answers questions by reading a state and returning a
decision over six fixed options; everything a reader needs to check — weights, calibration curves, browser
measurements, latency runs — is either in the repository or named by a registry row that says how to fetch
it.

Five things are on the table, and only four are strengths: size, latency, deployment and reproducibility.
The fifth is the accuracy of the decisions themselves, and it is the one that fails, at every gate that
governs it. No section of this draft states or implies that G1, G4 or G5 is met; §2 of the spec, the
gate table and this paper agree that three gates are not met, one is open, and the release is therefore
not clear. A reader who wants a benchmark-topping model should stop here; a reader who wants to know what
16.9M parameters do on a frozen suite, and how a project catches itself, should continue.

We state the shape of the negative result up front, because it is the interesting part. The long-context
failure in §6 is not an under-trained model: two checkpoints 400 updates apart agree *exactly* on every
rung they share, and the training loss sits on `ln(6)`, the value a uniform 6-way answer must converge to.
That is a missing mechanism, not an insufficient dose, and a harness that prints `readable: false` rather
than a table is what distinguishes the two.

## 3. Related work and positioning

The only competitor scored in this repository is **laya 0.3.20** (`convaiinnovations/laya`, checkpoint
snapshot `55cf4c4e…`, `Agent.system_one`), reported by its authors at **421.29M parameters** (encoder
394.78M + head 26.51M) — *their published figure, not measured here* — and evaluated by us on the same 16
(source, question) cells as our own model, at **0.6668** macro accuracy (§5). That number is measured on
this box with this harness, and it is the only external comparison this draft makes.

We deliberately do not use the mixed-basis laya figure that appears elsewhere in the project's history. It
is computed over a different denominator, and §9.37 and §9.41 of the spec record the two measurement bugs
that mixing bases produced: a headline moved by 0.240 depending on roll-up order, and a source looked like
a collapse when it was an aggregation artifact. Every comparison in this paper is same-cell or says it is
not comparable. `[REF NEEDED — to be verified: prior work on small typed-decision / constrained-output
classifiers, and on the decision-suite this corpus derives from; the repository contains no bibliography.]`

## 4. Method

**Parameter counting.** One expression defines every count in this paper:
`sum(p.numel() for p in model.parameters())` (`src/myna/engine.py:158-159`) — total parameters, embedding
table included, not trainable-only and not non-embedding. On the long-context checkpoint: 90 named tensors,
every one `requires_grad=True`, and 0 elements in buffers. The three counts below differ *only* by the
embedding table `trunk.tok.weight` = vocab × `d_model` 384, with `d_model` 384 and 6 layers identical
across all three configs (SPEC §9.64 i):

| count | vocab | checkpoint | used in |
|---|---|---|---|
| 14,449,280 | 1740 | `runs/myna-v0` — shipped artifact; also the model the browser build exports from (`runs/onnx/meta.json`) | §7, §9, §10 |
| 15,353,984 | 4096 (default `MynaConfig`) | `lc1-4k-s0` — the long-context arm | §6 |
| 16,926,848 | 8192 (the suite's `tokenizer-8192.json`) | `runs/v1b-kaggle-3600b` — the G1 accuracy arm | §5 |

**Architecture and task.** An observe/ask split: `observe()` folds a state of any length into a
fixed-size retained cache of per-layer `[1, H, dk, dv]` matrices (~0.6 MB at production width,
`src/myna/engine.py:66-73`), and `ask()` scores each question's six options against it
(`src/myna/engine.py:162, 95, 172`; `predict()` is the two composed). Accuracy is the per-cell unweighted
macro from `myna.report.macro`, which every harness in §5 **imports rather than restates**, over the frozen
decision-v2 split.

**Truncated backpropagation.** The long-context arm trains at `--long-context 4096 --grad-tokens 1024`,
so `split = long_context − grad_tokens = 3072`: the first 3,072 tokens of each row are scanned under
`no_grad` and only the final 1,024-token window carries gradient. The memory cost is therefore priced by
the **gradient window**, `min(grad_tokens, p95)`, not by the state length — the run's own record is
`detached_tokens_p95` 3082, `grad_window_tokens` 1024, `batch` 3, **2.4 GiB retained per step**
(`runs/lc1_4k_s0.metrics.json`). `[REF NEEDED — to be verified: truncated / periodic backpropagation
through time.]`

**Evaluation harness.** Needle recall at state lengths 128–16,384 tokens, decisive sentence at a random
position, `n = 60` per rung, `--seed 0`, `--device cpu` (`bench/eval_needle.py`). The harness computes a
baseline verdict before it will call a curve a curve: if the shortest rung is not above chance by a
margin — chance 1/6, margin 0.10, so 0.267 at `n = 60` — it prints `readable: false` and writes
`G4: **not measured by this run**` into the artifact instead of a quotable table. `temperature` is
recorded into the witness (`bench/eval_needle.py:116`) and never applied to the score, so accuracy cannot
move on that axis.

**Staged abort discipline.** Dose gates at 400 / 800 / 1200 updates, with a pre-committed accuracy rule at
the first boundary: *abort and report if the 128-rung is below 0.267*. Independently, four machine-abort
rules run continuously: step time above 3.0× the median of the last 20, free memory below 1.25× the
projection, swap above 4.0 GiB, or a Metal OOM. Both were live during §6's run, and the accuracy rule
fired (Stage A) while no machine rule ever did.

## 5. Results — accuracy against the pre-registered gates (G1)

All rows below are per-cell unweighted macros over the 16 (source, question) cells both engines score,
read from the committed report `runs/v1b_kaggle_3600b.report.json` (`macro.acc` 0.4893271976084137,
`macro.majority` 0.4330647953941198, `g1.margin_observed` 0.056262402214293905, `g1.pass` false). The
accuracy arm is **16,926,848 parameters at vocab 8192** (`runs/v1b-kaggle-3600b`, 3,600 updates on Kaggle,
last logged step 3599).

| arm | checkpoint (params / vocab) | scope | myna macro | majority floor | margin | verdict |
|---|---|---|---|---|---|---|
| **G1 primary** | v1b-kaggle-3600b, 16,926,848 / 8192 | 16 in-scope cells | **0.4893** | **0.4331** | **+0.0563** | **not met** — +0.15 required, and 0.211 short of the 0.70 level |
| sensitivity | same checkpoint | 14 cells (out-of-scope sources cut) | 0.5334 | 0.4666 | +0.0668 | not met — sensitivity only, never the headline |
| excluded | same checkpoint | 11 cells (every cell below its own floor dropped) | 0.5526 | — | +0.0949 | **not a result**: 0.147 short of 0.70, and dropping failing cells after seeing the scores is cherry-picking |
| competitor | laya 0.3.20, 421.29M (*their published count*) | the same 16 cells | 0.6668 | — | myna is 0.177 below | no parity claim |
| control | untrained scratch, 16,926,848 / 8192, 2 seeds | same 16 cells | 0.346 / 0.351 (mean 0.348) | 0.433 | below floor | the trained arm clears the floor; the control does not |
| contrast | void pre-`385e06c` checkpoint | — | 0.338 | — | — | disclosure only, never a result |

Three readings, all ours. First, **the margin and the level are separate requirements and the run fails
both**; the gate demands ≥ 0.70 *and* ≥ +0.15 over the floor the surviving cells imply. Second, **no scope
choice makes G1 pass** — the cells the model wins are also the cells a majority classifier wins, so
cutting the dead ones lifts the level and the floor together, buying eleven thousandths of a 0.094 deficit
(0.4893 → 0.5334 against 0.4331 → 0.4666). Third, the untrained control matters more than it looks: it
sits *below* the majority floor, so the trained checkpoint's +0.0563 is real movement against a classifier
that reads nothing, while remaining far from the gate.

## 6. Results — long-context correctness (G4)

G4 asks for needle accuracy **≥ 0.90 at 4,096 state tokens and ≥ 0.85 at 16,384** on a checkpoint trained
for long context. The arm is `lc1-4k-s0` — **15,353,984 parameters at vocab 4096**, cold-started (no
`--init`), trained at 4k states with a 1,024-token gradient window on this laptop's MPS device. The table
is SPEC §9.64 (iv), cell by cell against `runs/needle_lc1_stage{A,B_resume,C_final}.{md,json}` and the
three stage logs; a rung absent from a witness's `lengths` list is marked not evaluated and never
inferred.

| leg | checkpoint evaluated | boundary status | n / seed / device | 128 | 1024 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|
| Stage A (leg 1, cold) | step-399 snapshot `model_last.pt` `f389bde5…`, reconstructed into a ladder-loadable `model.pt`; the leg had executed 425 updates, last save step 424 | **gated abort point** — the 0.267 rule fired at 0.1667, which *is* the floor | 60 / 0 / `cpu`, **throttled** (`OMP_NUM_THREADS=2 nice -n 15`) | **0.1667** (10/60) | **0.2167** (13/60) | not evaluated at this stage | not evaluated | not evaluated |
| Stage B (leg 2, resumed) | step-799 snapshot `a2bbc2eb…`, resumed from step-424 `d3548a88…` | **observation point** — an accuracy rule cannot govern a run ordered past a floor it is sitting on; abort authority was the machine rules only | 60 / 0 / `cpu`, **throttled** | **0.150** (9/60) | **0.2167** (13/60) | **0.1833** (11/60) | not evaluated at this stage | not evaluated |
| Stage C (leg 2, final) | step-1199 `model_last.pt` `25b868e9…`, `model.pt` `fb37a28b…`; `steps_requested` 1200, `last_step` 1199, 425 + 775 = 1200 executed | **observation point**; no machine abort (STOP lines 0; swap 0.25M of 1024.00M at every reading) | 60 / 0 / `cpu`, **unthrottled** — the trainer had exited | **0.150** (9/60) | **0.2167** (13/60) | **0.1833** (11/60) | **0.150** (9/60) | **0.150** (9/60) |

**Verdict: G4 is NOT met, measured.** 4096 → 0.1833 against ≥ 0.90 is short by 0.7167; 16384 → 0.150
against ≥ 0.85 is short by 0.700. At `n = 60` one standard error is 0.0481, so those shortfalls are 14.9
and 14.55 standard errors — not noise — while no rung on the ladder is separable from the 0.167 uniform
floor (the largest departure is the 1024 rung at 1.04 s.e. *above* it). The harness still prints
`readable: false`, which is the honest description of the curve: a flat line at chance with a slight
negative bias, not a decay.

**Dose or mechanism? Mechanism.** Stage B and Stage C share three rungs and agree exactly on all three —
9, 13 and 11 correct of 60 — after 400 more updates. Two checkpoints do not agree on three 60-sample
counts by accident while their weights are still moving; they agree because the readout has collapsed, and
the training loss says so independently: `ema` **1.791** at step 1199 against `ln(6) = 1.791759469228055`.
Another dose at this regime is therefore not the next experiment, and we do not run one.

Two caveats are priced rather than waved away. (1) `--resume` restores the cosine schedule (T_max 1200,
`last_epoch` 425) and the Adam state, but re-seeds the data RNG from `--seed 0`, so the sampler's draw
order **replayed**: leg 2 is a schedule-correct continuation, not a bit-identical replay. (2) The dose
depends on which token axis you divide, and both denominators are printed here (SPEC §9.64 v):
1200 × 3 = **3,600 states** × `state_tokens_p95` **4106** = **14,781,600** state tokens ÷ **15,353,984**
params = **0.96 tok/param**; on the gradient-carrying axis, 1200 × 3 × 1024 = **3,686,400** tokens ÷
**15,353,984** = **0.24 tok/param**. The pretraining yardstick this is compared against — 0.28
tok/param — divides the pilot corpus's 4,246,106 state tokens by the *same* 15,353,984 (0.2765), which is
the only one of the three counts in §4 that reproduces it; `[REF NEEDED — to be verified: the
compute-optimal ≈20 tokens/parameter pretraining law that this yardstick leans on.]`

**What this section is not.** A failed G4 is not a latency or a size win in disguise. The 4k state is cheap
— 589,824 bytes fixed — and the ladder is slow per rung, and neither fact bears on a needle the model does
not find at 128 tokens either. The rungs are the accuracy axis and G4 is stated on accuracy.

## 7. Size and deployment

All figures in this section are the **shipped artifact**: v0, 14,449,280 parameters at vocab 1740
(`runs/train-v0.log:5`, `runs/bench_mlx_int8.json`). The browser build is exported from that same
checkpoint — `runs/onnx/meta.json` records `"ckpt": "runs/myna-v0"`, `"vocab": 1740`, `"params_m": 14.45`.

Measured in Chrome 153.0.8010.53 with ONNX Runtime Web 1.30.0 on a 10-core, 32 GB laptop
(`runs/browser_g3.json`, 6 checks passed / 0 failed in all four runs):

| quantity | measured |
|---|---|
| artifact | 59.32 MiB (`runs/onnx`), fetched in 1 weights request |
| bytes over the wire | 73.08–73.09 MiB |
| wasm runtime | 13.58 MiB |
| cold load | 253.9–394.4 ms |
| p50 answer | 426–435.8 ms at 10 threads; 741.8–742.9 ms at 1 thread |
| peak tile | 3 MiB |
| tab memory | 176.0–176.1 MiB (recorded on the two cross-origin-isolated runs; the plain runs record none) |
| decision state | `state_bytes_total` **589,824 B = 576 KiB, fixed** |

Quantization changes bytes, not the parameter count. `runs/bench_mlx_int8.json` reports int8
`file_bytes` **18,248,977** (tensors 18,229,760) against MLX fp32 **57,805,835** (tensors 57,797,120, which
is exactly 14,449,280 × 4); the int8 run quantizes 50 of the 90 tensors at 8 bits with group size 64 and
keeps `trunk.tok.weight` in full precision. **14,449,280 is therefore a pre-quantization count, and no
count in this paper is a quantized count** (SPEC §9.64 ii).

## 8. Latency

Matched conditions only: same box, same process, same inputs, device `cpu`, both engines in fp32, warmup 3
and 10 reps, laya at `max_len` 1024 / `head_max_len` 256 (`runs/latency_matched.md`; the harness pins laya
0.3.20's snapshot path in its own header).

| questions | laya `system_one` p50 ms | myna predict p50 ms | speedup (e2e) | myna ask-only p50 ms | speedup (stream) |
|---|---|---|---|---|---|
| 1 | 100.691 | 33.083 | 3.04× | 14.856 | 6.78× |
| 5 | 271.988 | 74.866 | 3.63× | 55.453 | 4.90× |
| 10 | 506.277 | 120.753 | 4.19× | 98.217 | 5.15× |
| 50 | 3589.496 | 869.952 | 4.13× | 787.12 | 4.56× |

These are throughput measurements of two engines over identical inputs, and nothing else. They are not an
accuracy argument: the same harness scores laya above Myna on the same cells (§5), and this paper makes no
attempt to trade one axis against the other. The two project rules that discipline this table are that
absolute millisecond figures are quoted only from this matched harness, and that ratios are derived by
dividing measured columns rather than measured by any harness — which is why §6's needle `ms` column,
recorded under a deliberate CPU throttle in two of its three stages, appears nowhere here.

## 9. Reproducibility

The mechanical claim is that every published figure re-derives from a committed artifact, and the audit
passes: `bench/gates.py --check` → **7/7 verdicts hold** (3 met, 3 not met, 1 open; the release gate is
NOT clear); `bench/reproduce.py --check` → **34/34 rows hold, 101 figures tied to a committed witness, 1
gated with no witness**; the two suites that read these documents (`tests/test_gates.py`,
`tests/test_reproduce.py`) → **67 passed**. A mutation battery of 21 string-anchored mutants backs the
gate logic, and the anchor-presence check (21/21) is what tells us a prose edit has not silently turned one
into a no-op.

The gap is as important and is stated in full. **The 4k ladder of §6 is witnessed by committed JSON but is
not reproducible from a clone.** `runs/needle_lc1_stageC_final.{md,json}` is in the tree, and it is
byte-identical to the staged original, but the checkpoint that produced it — `model_last.pt` 184,355,257 B
and the ladder-loadable `model.pt` reconstructed from it, 61,442,325 B, at
`~/myna-staging/lc1-4k-s0/stageC_final` — is not:
`.gitignore` keeps `runs/*` directories out, and unlike v0 and v1b (`src/myna/weights.py` fetches a
Release by tag, README:40-42 naming `v1b-checkpoint` as "the 0.4893 Kaggle run"), the 4k weights have no
published arm. Until a Release ships them, the ladder cannot become a registry row, and the honest form of
§6's result is *witnessed, not reproducible*. We do not paper over it by citing a fetch command that does
not resolve.

## 10. Diagnostic failures

**G5 — the machinery works, the level does not.** On `calibration.jsonl` (448 rows, 568 questions, the v0
checkpoint at vocab 1740), abstention ranks questions by confidence and the risk/coverage curve rises
monotonically as coverage falls: 0.3486 at full coverage, 0.4135 at 60.04%, 0.5351 at 20.07%, and a
**0.5965 at 10.04% coverage** (`runs/risk_coverage.json`, `g5.pass: false`). G5 requires ≥ 0.95 at ≥ 60%
coverage, so at its own operating point the curve is **0.5365 short** (0.4135 at 60.04%), and even the best
point the curve ever reaches is 0.3535 short, at a coverage nearly six times narrower than the gate asks
for. The
machinery is what works — thresholds fitted on `calibration.jsonl`, coverage falling monotonically,
accuracy rising with it — and the level is what does not. The three sources G5 names are the reason, and
their denominator must be stated: at **full coverage (rung 1.0, n = 40 each)** they are banking77 **0.000**,
dbpedia14 **0.025**, trec **0.150**; dbpedia14's best value over *any* rung is 0.0278 (rung 0.9), so even
reading each source at its maximum the gate fails. A constant emitter cannot be rescued by a threshold.

**G4 — a missing mechanism, not a missing dose** (§6): 9/13/11 correct of 60 reproduced exactly 400
updates apart, with training loss on `ln(6)`.

**G6 — open.** v0's own measured test macro is **0.9523** (the macro of the nine `=== test ===` rows in
`runs/train-v0.log`), above the 0.94 level, so the level clears today. The gate nevertheless stays open
until a v1 checkpoint is run against it, and we do not close a regression gate with the pre-regression
number.

**The data-side experiment that did not help.** The anti-prior pair changes how batches are composed, so
the label prior stops being a shortcut the model can answer with, and both arms were run at the same dose
on the same box: the `on` arm scored **0.4785** against the `off` control's **0.5339**, i.e. the lever we
expected to help cost **−0.0554** (§9.56, §9.58). An earlier reading of the same series attributed part of
that gap to a paraphrase-coverage "residue"; the claim was withdrawn as a denominator artifact once the
arms were matched on coverage (§9.58), and it is not repeated here.

## 11. Limitations and recorded defects

Beyond the §9 gap, we disclose three defects in our own documentation rather than smooth them, because two
of them would otherwise enter the literature through this paper.

**(a) A v0 weight digest that no reader can check.** A `v0 model.pt` digest quoted during the hand-off
(`88c14ef2…`) appears **nowhere in this checkout** — including README, whose weight disclosures name the
`v0-checkpoint` Release and give digests for the *other* arms (`model.pt` 67,733,781 B at
`30f0fa937e57e8ce…` for the `off` control, `model 0e73a8f07edc` / `tokenizer 0252c24627ee` for v1b). The
consequence for this paper: **no digest is printed for v0's weights here**, and the tie between
`runs/myna-v0` and its published asset remains a Release-page claim rather than a checkout-verifiable one.

**(b) An exact-float claim that is false as a 17-digit claim.** §9.59 reports four runs of the README
install snippet answering with one identical float, and the literal the repo prints is
`0.7899749875068665` (`repr`-shortest; README:31, eight times in SPEC — 1519, 1545, 1554, 1567, 3710,
3717, 3813, 3844 — and three times in TODO). Under the repo's pinned
torch 2.14.0 the same quantity expands to `0.78997498750686646`, while a run under torch 2.15.0 produced
`0.78997457027435303` — a difference of **4.17e-07**, i.e. agreement to about 5e-7, not bit identity. Two
disclosures follow: this paper quotes the confidence to ~5e-7 and pins torch, and **the torch 2.15.0
figure has no committed witness in this repository** — it comes from a session run, so it appears here as
a limitation, not as a measurement, and must be re-run into an artifact before it can be cited as one.

**(c) The generated gate block is drift-checked, not truth-checked.** README's release-gate block is
derived from `bench/gates.py`'s `readme_block()` and the test asserts only that the generated text appears
verbatim in README, so a stale *sentence* can survive while every *verdict* is re-read from its artifact —
which is exactly what happened once already: the generator claimed G1 and G4 "cannot be closed on this
box" for a day after both had been closed on it (§9.62 ix, §9.63 xii). The 21 mutants cover verdict logic
and none reaches that prose, so we present the block as a drift guard with a named hole, not as a proof.

**Scope limits we accept.** One seed for the needle ladder and two for the scratch control; a
single-box, single-browser measurement set; a frozen 16-cell suite and a needle ladder at `n = 60` per rung,
where 60 samples give a 0.0481 standard error; and a corpus whose pilot is 4.25M state tokens, under the
pretraining yardstick the model is far from data-saturated rather than capacity-limited.

## 12. Conclusion

We described Myna, a 16,926,848-parameter typed-decision engine whose shipped artifact is 59.32 MiB, loads
in a browser tab in 253.9–394.4 ms, holds state in a fixed 576 KiB, and answers 10 questions in
120.753 ms against a 421.29M-parameter competitor's 506.277 ms on the same inputs and the same box. We also
report that it does not pass its own release gates: **G1 is not met** at 0.4893 macro against a 0.70 level
and a +0.15 margin over a 0.4331 floor, with no scope choice rescuing it; **G4 is not met** at 0.1833 at
4,096 and 0.150 at 16,384 state tokens against 0.90 and 0.85, on a checkpoint that completed its approved
1,200-update dose and whose loss sits on `ln(6)`; **G5 is not met**, peaking at 0.5965 only at 10.04%
coverage against a required 0.95 at ≥ 60%; and **G6 is open**, its level already cleared at 0.9523 but no
v1 checkpoint run against it. The size, latency and reproducibility numbers are the measured strengths;
the three failures and the one open gate are the measured contribution, and the needle harness that
refused to print a decay curve for a flat one is the instrument we would hand to the next person.

---

## `[REF NEEDED — to be verified]` list (no reference in this draft is unverified)

1. **§3** — prior work on small typed-decision / constrained-output classifiers, and on the provenance of
   the decision-v2 corpus. The repository contains no bibliography.
2. **§3** — laya's own paper or model card for the 421.29M parameter count (encoder 394.78M + head 26.51M),
   which this draft attributes to its authors as *their published figure*.
3. **§4** — truncated / periodic backpropagation through time, the method lineage for the 3,072-token
   `no_grad` prefix.
4. **§6** — the compute-optimal ≈20 tokens/parameter pretraining scaling law that the 0.28 tok/param
   yardstick is measured against.
5. **§6** — needle-in-a-haystack long-context evaluation methodology.
6. **§10** — selective classification / reject-option literature for the risk-coverage curve, and the
   calibration work behind confidence thresholds.
7. **§7** — a citable reference for ONNX Runtime Web and the WebGPU/wasm execution environment (the version
   1.30.0 used here is witnessed in `runs/browser_g3.json`; the upstream citation is not in the repo).

## Proposed SPEC edits — recorded, **not applied** in this step

1. **`SPEC.md:1669`** (risk table) — "15.35M cannot learn instruction-following well enough to clear G1"
   names the default `MynaConfig` count, but the arm that actually scored G1 is the 16,926,848-parameter
   v1b checkpoint at vocab 8192. Proposed: restate the risk row against 16,926,848 / vocab 8192, or mark it
   as a planning-time sentence about the default config. Recorded in SPEC §9.64 (vi).
2. **`SPEC.md:1611-1612`** (§6, "Compute and budget") — "⇒ **0.28 tokens/param**" prints no denominator.
   Proposed: state it as 4,246,106 ÷ 15,353,984 = 0.2765, since that divisor is the only one of the three
   counts in §9.64 (i) that reproduces 0.28 (14,449,280 gives 0.2939, 16,926,848 gives 0.2509). Recorded in
   SPEC §9.64 (v) and (vi).

# Myna: a small, auditable typed-decision engine — what a 16.9M-parameter model does and does not do on a frozen decision suite

**Working draft for owner review. Not for submission. Local only.**
Every number in this file is **measured** in this repository — either read out of a committed artifact or
recomputed from a checkpoint on disk — and carries its witness beside it. Nothing is projected or
borrowed. SPEC §9.64 (commit `bffefba`) is the source of truth for the parameter counts, the stage ladder
and the dose denominators used here; where this draft cites a figure not in §9.64 it cites the artifact
that prints it. Every literature citation in this draft was opened and quoted from its live source on
2026-10-29, and any source that could not be opened for a quotable sentence is **not** cited; the list is at
the end under "References", beside the reason each entry is there. Two of the citations are deliberately
**software and documentation rather than papers**: laya in §3 (repository, package index and pinned model
artifact — no laya paper exists) and the browser runtime in §7 (npm registry and two W3C specifications). No
`[REF NEEDED]` marker remains anywhere in the text.

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
state; a matched-condition throughput advantage over a competitor **we measure at 421,293,827
parameters** (our count at their pinned snapshot, not a figure they publish) on the same inputs
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
snapshot `55cf4c4e…`, `Agent.system_one`). Its authors publish **≈421M parameters** — the card at the very
revision we measured says "*Backbone:* ModernBERT-large (395M, bidirectional, fully fine-tuned) + a
decision head trained from scratch: 2 transformer layers, an option-marker scorer, and an act/escalate
head. 421M total" — and **we** measure **421,293,827** in that artifact (`runs/latency_matched.json`
`laya_params`, produced by the same `sum(p.numel() for p in model.parameters())` expression Myna's own
count uses, `bench/bench_latency_matched.py:350`). The two reconcile exactly: that checkpoint's tensor
header sums to **421,293,830** across 206 tensors, and the 3-element `temperature` entry never reaches
`parameters()` because their code registers it as a buffer (`laya/common.py:600`). The encoder-prefixed
tensors are
394,781,696 (394.78M) and everything but `temperature` is 26,512,131 (26.51M) — our arithmetic over their
public tensor header, not a figure they publish, and no witness in this repository prints that split.
Evaluated by us on the same 16 (source, question) cells as our own model, at **0.6668** macro accuracy (§5);
that number is measured on this box with this harness, and it is the only external comparison this draft
makes. Cited **as software, explicitly not as a paper and not as external literature**: Nandhakishor M. /
Convai Innovations, *laya*, version 0.3.20 — GitHub https://github.com/NandhaKishorM/laya (Apache-2.0),
PyPI https://pypi.org/project/laya/0.3.20/ (uploaded 2026-09-24T05:41:06Z), model artifact
https://huggingface.co/convaiinnovations/laya at pinned revision
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, `typed-decisions/` subfolder; accessed 2026-10-29. **No laya
paper exists**: the card at that revision carries no BibTeX and no DOI, and the repository README's only
arXiv link is to somebody else's work. Their rounded "421M" is quoted above purely as attribution to the
source; no measurement in this paper uses it, and every laya figure in §5 and §8 is ours.

We deliberately do not use the mixed-basis laya figure that appears elsewhere in the project's history. It
is computed over a different denominator, and §9.37 and §9.41 of the spec record the two measurement bugs
that mixing bases produced: a headline moved by 0.240 depending on roll-up order, and a source looked like
a collapse when it was an aggregation artifact. Every comparison in this paper is same-cell or says it is
not comparable.

**Prior work.** The family this engine belongs to is named in the recent literature. Li & Wagle,
*LLM-as-Jev: LLMs Are Already Jev-Style Decision Models — When and How to Fine-Tune Them*, arXiv:2610.02076
(v1 2026-10-01, v3 current), describes decision models by exactly the property this draft scores: they return
"categorical probability distributions over predefined options without generating free-form text". Duan et
al., *General Decision Models: Benchmarking and Insights Beyond Jev*, arXiv:2610.03935 (v1 2026-10-02, v2
current), benchmarks that family across domains. Instruction-shaped option scoring has an older lineage in
cloze-based classification: Schick & Schütze, *Exploiting Cloze-Questions for Few-Shot Text Classification
and Natural Language Inference*, EACL 2021, pp. 255–269, DOI 10.18653/v1/2021.eacl-main.20 — Pattern-Exploiting
Training, which "reformulates input examples as cloze-style phrases to help language models understand a
given task". Constrained output is Willard & Louf, *Efficient Guided Generation for Large Language Models*,
arXiv:2307.09702 (v1 2023-07-19, v4 current), which reformulates generation "in terms of transitions between
the states of a finite-state machine"; that is constrained *generation*, adjacent to this engine's per-option
scoring rather than the same mechanism, and no more than the adjacency is claimed here. **The corpus needs no
external reference:** its provenance is witnessed inside this repository by
`data/decision-v2-pilot/upstream_manifest.json` (`derived_from`, `converters`, ten pinned
`dataset_revisions`, `base_revisions`, `records`, `state_tokens_under_first_base.sum`), whose upstream is the
public `jaredpalmer/kev` suite.

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
(`runs/lc1_4k_s0.metrics.json`). This is truncated backpropagation through time, and the cost that motivates
truncating is stated in the origin paper: Williams & Zipser, *A Learning Algorithm for Continually Running
Fully Recurrent Neural Networks*, Neural Computation 1(2):270–280, 1989, DOI 10.1162/neco.1989.1.2.270, whose
publisher-deposited abstract lists as the disadvantage of the untruncated recurrent rule that such algorithms
"require nonlocal communication in the network being trained and are computationally expensive". That the
truncation horizon bounds what can be learned is the published position, not our inference: Ke et al., *Sparse
Attentive Backtracking*, arXiv:1711.02326, record that "full backpropagation through time is rarely used on
long sequences, and truncated backpropagation through time is used as a heuristic", and Benzing et al.,
*Optimal Kronecker-Sum Approximation of Real Time Recurrent Learning*, arXiv:1902.03993, that "the most
popular training method, Truncated Backpropagation through Time (TBPTT), categorically forbids learning
dependencies beyond the truncation horizon". Truncation is not an RNN-only device — Al-Rfou et al.,
*Character-Level Language Modeling with Deeper Self-Attention*, arXiv:1808.04444, state that such attention
models "are typically trained using truncated backpropagation through time". The memory motive for detaching
rather than storing activations is Chen et al., *Training Deep Nets with Sublinear Memory Cost*,
arXiv:1604.06174, which gives "an algorithm that costs O(sqrt(n)) memory to train a n layer network, with only
the computational cost of an extra forward pass per mini-batch". These are lineage and motive only: none of
them analyses detaching a fixed prefix of a transformer state scan, which is what `split = long_context −
grad_tokens` does here.

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
| competitor | laya 0.3.20, **421,293,827 parameters measured by us** at the pinned snapshot `55cf4c4e…` (their card's rounded ≈421M is attribution to the source, never the measurement) | the same 16 cells | 0.6668 | — | myna is 0.177 below | no parity claim |
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
inferred. The ladder itself — a state length increased rung by rung, a needle planted at a random position,
accuracy reported per rung — is a long-context retrieval evaluation of a published class. Liu et al., *Lost in
the Middle: How Language Models Use Long Contexts*, TACL 12:157–173, 2024, DOI 10.1162/tacl_a_00638
(arXiv:2307.03172v3) is the positional result this harness type was built to probe: "performance can degrade
significantly when changing the position of relevant information", and "performance is often highest when
relevant information occurs at the beginning or end of the input context, and significantly degrades when
models must access relevant information in the middle of long contexts". The grid shape — one accuracy per
context-length and needle-depth cell — follows the open-source Needle In A Haystack of Kamradt,
`gkamradt/needle-in-a-haystack` at HEAD `021385d68d3202e37893e9d3cd29011c569abe30` (2026-06-08), cited **as
software**: its README describes a sweep of `(context length × needle depth)` cells that "scores each
response, and writes one result row per cell to a JSONL file" (the repository's former name,
`gkamradt/LLMTest_NeedleInAHaystack`, now answers HTTP 301). A peer-reviewed benchmark that names the
paradigm is Yu et al., *Sequential-NIAH: A Needle-In-A-Haystack Benchmark for Extracting Sequential Needles
from Long Contexts*, arXiv:2504.04713 (v1 2025-04-07, v3 current). All three are lineage, never
corroboration: that literature reports positional decay in models that make use of long contexts, and the
curve below is flat at the floor rather than a decay.

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
**15,353,984** = **0.24 tok/param**. Beside them sits the pilot corpus's own ratio: 4,246,106 state tokens
(`data/decision-v2-pilot/upstream_manifest.json`, `state_tokens_under_first_base.sum`) ÷ the *same*
**15,353,984** = **0.2765 tok/param** (SPEC §9.64 v), which is the only one of the three counts in §4 that
reproduces the 0.28 quoted elsewhere in this project. That comparison is this repository's own arithmetic
against its own pilot corpus: no external tokens-per-parameter constant is claimed here, and none appears
in this section.

**What this section is not.** A failed G4 is not a latency or a size win in disguise. The 4k state is cheap
— 589,824 bytes fixed — and the ladder is slow per rung, and neither fact bears on a needle the model does
not find at 128 tokens either. The rungs are the accuracy axis and G4 is stated on accuracy.

## 7. Size and deployment

All figures in this section are the **shipped artifact**: v0, 14,449,280 parameters at vocab 1740
(`runs/train-v0.log:5`, `runs/bench_mlx_int8.json`). The browser build is exported from that same
checkpoint — `runs/onnx/meta.json` records `"ckpt": "runs/myna-v0"`, `"vocab": 1740`, `"params_m": 14.45`.

Measured in Chrome 153.0.8010.53 with ONNX Runtime Web 1.30.0 on a 10-core, 32 GB laptop
(`runs/browser_g3.json`, 6 checks passed / 0 failed in all four runs). The version measured here is witnessed
by that artifact, and the environment itself is cited **as software and documentation, not as a paper**:
`onnxruntime-web` 1.30.0 on the npm registry (MIT; published 2026-09-14T20:20:13.234Z; described there as "A
Javascript library for running ONNX models on browsers"; the registry's `latest` is 1.31.1, published
2026-10-26T20:01:38Z, so the build measured here is one minor behind), the project's own web tutorial
(https://onnxruntime.ai/docs/tutorials/web/), which states the backends plainly — "With onnxruntime-web, you
have the option to use `webgl`, `webgpu` or `webnn` (with `deviceType` set to `gpu`) for GPU processing, and
WebAssembly (`wasm`, alias to `cpu`) or `webnn` (with `deviceType` set to `cpu`) for CPU processing" — the
*WebGPU* API specification, a W3C Candidate Recommendation Draft dated 2026-09-15
(https://www.w3.org/TR/webgpu/): "WebGPU exposes an API for performing operations, such as rendering and
computation, on a Graphics Processing Unit"; and the *WebAssembly Core Specification*, a W3C Candidate
Recommendation Draft dated 2026-10-28 (https://www.w3.org/TR/wasm-core-2/), "a safe, portable, low-level code
format designed for efficient execution and compact representation". All four were opened on 2026-10-29. No
peer-reviewed venue exists for the runtime and none is claimed for it.

The measured quantities:

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
checkpoint at vocab 1740), abstention ranks questions by confidence and **accuracy** rises monotonically as
coverage falls — 0.3486 at full coverage, 0.4135 at 60.04%, 0.5351 at 20.07%, and a
**0.5965 at 10.04% coverage** (`runs/risk_coverage.json`, whose rungs print `accuracy` and its complement
`risk` side by side; `g5.pass: false`). "Risk-coverage" is used here as the plain description of that
trade, not as a named published method. G5 requires ≥ 0.95 at ≥ 60%
coverage, so at its own operating point the curve is **0.5365 short** (0.4135 at 60.04%), and even the best
point the curve ever reaches is 0.3535 short, at a coverage nearly six times narrower than the gate asks
for. The confidence that does the ranking is the engine's own temperature-scaled softmax: one scalar per
checkpoint, fitted by an 8-point NLL grid (`src/myna/train.py:532-537`) on the split that
`temperature_source` selects — the suite's own `calibration` group, with `dev` as the fallback
(`src/myna/train.py:564-572`); `runs/train-v0.log:69` prints the resulting **3.0** without naming its
split, and this section's run records the same value (`runs/risk_coverage.json`, `temperature`). That
scalar divides the logits at inference (`src/myna/engine.py:208`) and is applied the same way in the
browser export (`src/myna/onnx_export.py:328`). The machinery is what works —
thresholds fitted on `calibration.jsonl`, coverage falling monotonically,
accuracy rising with it — and the level is what does not. The framing has a published name: Geifman &
El-Yaniv, *Selective Classification for Deep Neural Networks*, Advances in Neural Information Processing
Systems 30 (2017) (arXiv:1705.08500, v1 2017-05-23, v2 current), the line this curve sits on — "Selective
classification techniques (also known as reject option)", in which "the classifier rejects instances as
needed, to grant the desired risk (with high probability)". The scalar that ranks the questions is temperature
scaling as published in Guo, Pleiss, Sun & Weinberger, *On Calibration of Modern Neural Networks*, Proceedings
of the 34th International Conference on Machine Learning, PMLR 70:1321–1330 (arXiv:1706.04599, v1 2017-06-14,
v2 current): "on most datasets, temperature scaling — a single-parameter variant of Platt Scaling — is
surprisingly effective at calibrating predictions". That is the same single parameter our 8-point NLL grid
fits. Neither citation upgrades this section: no rejection-set construction is claimed, no coverage guarantee
is reported, and "risk-coverage" above stays the plain description of the trade rather than a named method.
The three sources G5 names are the reason, and
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
where 60 samples give a 0.0481 standard error; and a corpus whose pilot is 4.25M state tokens —
4,246,106 ÷ 15,353,984 = **0.2765 tokens per parameter**, this repository's own pilot ratio, and against
that ratio alone the model is far from data-saturated rather than capacity-limited. No external
tokens-per-parameter law is claimed for this or any other section of the draft (§6).

## 12. Conclusion

We described Myna, a 16,926,848-parameter typed-decision engine whose shipped artifact is 59.32 MiB, loads
in a browser tab in 253.9–394.4 ms, holds state in a fixed 576 KiB, and answers 10 questions in
120.753 ms against 506.277 ms from a competitor **we measure at 421,293,827 parameters** — our count at
their pinned snapshot, not a figure they publish — on the same inputs and the same box. We also
report that it does not pass its own release gates: **G1 is not met** at 0.4893 macro against a 0.70 level
and a +0.15 margin over a 0.4331 floor, with no scope choice rescuing it; **G4 is not met** at 0.1833 at
4,096 and 0.150 at 16,384 state tokens against 0.90 and 0.85, on a checkpoint that completed its approved
1,200-update dose and whose loss sits on `ln(6)`; **G5 is not met**, peaking at 0.5965 only at 10.04%
coverage against a required 0.95 at ≥ 60%; and **G6 is open**, its level already cleared at 0.9523 but no
v1 checkpoint run against it. The size, latency and reproducibility numbers are the measured strengths;
the three failures and the one open gate are the measured contribution, and the needle harness that
refused to print a decay curve for a flat one is the instrument we would hand to the next person.

---

## References — every citation the body uses, each opened on 2026-10-29

No `[REF NEEDED]` marker remains in the text: each of the five places that carried one now cites a source
that could be opened and quoted, or is answered by an internal witness. Every entry below appears in the body
above, and nothing is listed here that the body does not use. Three sources that could **not** be opened for a
quotable sentence are named as deliberately uncited rather than carried on their reputations.

1. **§3** — Yinheng Li and Justin Wagle, *LLM-as-Jev: LLMs Are Already Jev-Style Decision Models — When and How
   to Fine-Tune Them*, arXiv:2610.02076 (v1 2026-10-01, v3 current). Feiyu Duan, Jiayu Lin, Jia Wang, Jun
   Xiang, Jialiang Wu, Xinnong Zhang, Hanqi Yan, Siyuan Wang and Zhongyu Wei, *General Decision Models:
   Benchmarking and Insights Beyond Jev*, arXiv:2610.03935 (v1 2026-10-02, v2 current). Timo Schick and
   Hinrich Schütze, *Exploiting Cloze-Questions for Few-Shot Text Classification and Natural Language
   Inference*, EACL 2021 main volume, pp. 255–269, DOI 10.18653/v1/2021.eacl-main.20 (abstract read from the
   ACL Anthology record). Brandon T. Willard and Rémi Louf, *Efficient Guided Generation for Large Language
   Models*, arXiv:2307.09702 (v1 2023-07-19, v4 current). Corpus provenance is **not** a literature item: it
   is the internal witness `data/decision-v2-pilot/upstream_manifest.json`, whose upstream is the public
   `jaredpalmer/kev` suite.
2. **§4** — Ronald J. Williams and David Zipser, *A Learning Algorithm for Continually Running Fully Recurrent
   Neural Networks*, Neural Computation 1(2):270–280, 1989, DOI 10.1162/neco.1989.1.2.270 (abstract deposited
   with the DOI record). Nan Rosemary Ke, Anirudh Goyal, Olexa Bilaniuk, Jonathan Binas, Laurent Charlin,
   Chris Pal and Yoshua Bengio, *Sparse Attentive Backtracking: Long-Range Credit Assignment in Recurrent
   Networks*, arXiv:1711.02326. Frederik Benzing, Marcelo Matheus Gauy, Asier Mujika, Anders Martinsson and
   Angelika Steger, *Optimal Kronecker-Sum Approximation of Real Time Recurrent Learning*, arXiv:1902.03993.
   Rami Al-Rfou, Dokook Choe, Noah Constant, Mandy Guo and Llion Jones, *Character-Level Language Modeling
   with Deeper Self-Attention*, arXiv:1808.04444. Tianqi Chen, Bing Xu, Chiyuan Zhang and Carlos Guestrin,
   *Training Deep Nets with Sublinear Memory Cost*, arXiv:1604.06174. **Not cited in the body:** P.J. Werbos,
   *Backpropagation Through Time: What It Does and How to Do It*, Proceedings of the IEEE 78(10):1550–1560,
   1990, DOI 10.1109/5.58337 — the canonical origin, whose bibliographic record was confirmed twice here
   (Crossref and Semantic Scholar, both HTTP 200) but neither deposits an abstract and the IEEE full text was
   not opened, so no sentence of it is quoted and none is relied on.
3. **§6** — Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni and
   Percy Liang, *Lost in the Middle: How Language Models Use Long Contexts*, TACL 12:157–173, 2024,
   DOI 10.1162/tacl_a_00638 (abstract read from the DOI record; arXiv:2307.03172v3). Graham Kamradt,
   *needle-in-a-haystack*, https://github.com/gkamradt/needle-in-a-haystack, HEAD
   `021385d68d3202e37893e9d3cd29011c569abe30` dated 2026-06-08T18:01:11Z — cited **as software**; the
   repository's license field reads `NOASSERTION`, so no licence is claimed for it, and its former name
   `gkamradt/LLMTest_NeedleInAHaystack` now answers HTTP 301. Yifei Yu, Qian-Wen Zhang, Lingfeng Qiao, Di
   Yin, Fang Li, Jie Wang, Zengxi Chen, Suncong Zheng, Xiaolong Liang and Xing Sun, *Sequential-NIAH: A
   Needle-In-A-Haystack Benchmark for Extracting Sequential Needles from Long Contexts*, arXiv:2504.04713
   (v1 2025-04-07, v3 current).
4. **§7** — `onnxruntime-web` 1.30.0, npm registry record (MIT; published 2026-09-14T20:20:13.234Z; `latest`
   is 1.31.1, published 2026-10-26T20:01:38Z). ONNX Runtime web tutorial,
   https://onnxruntime.ai/docs/tutorials/web/ (page title "Web | onnxruntime"). *WebGPU*, W3C Candidate
   Recommendation Draft dated 2026-09-15, https://www.w3.org/TR/webgpu/. *WebAssembly Core Specification*, W3C
   Candidate Recommendation Draft dated 2026-10-28, https://www.w3.org/TR/wasm-core-2/. **Two entries
   previously listed here are removed rather than carried:** a Microsoft Open Source Blog post on the WebGPU
   path, and the deeper docs page `https://onnxruntime.ai/docs/tutorials/web/eps.html` — both returned HTTP
   404 when opened on 2026-10-29, so neither is cited. Also **not cited in the body:** Andreas Haas, Andreas
   Rossberg, Derek L. Schuff, Ben L. Titzer, Michael Holman, Dan Gohman, Luke Wagner, Alon Zakai and JF
   Bastien, *Bringing the web up to speed with WebAssembly*, PLDI 2017, pp. 185–200,
   DOI 10.1145/3062341.3062363 — the DOI record is confirmed (Crossref, HTTP 200), but no abstract is
   deposited and the ACM full text was not opened, so the WebAssembly substrate is cited to the W3C
   specification above instead.
5. **§10** — Yonatan Geifman and Ran El-Yaniv, *Selective Classification for Deep Neural Networks*, Advances
   in Neural Information Processing Systems 30 (2017), entry confirmed in the venue's own 2017 index at
   `papers.nips.cc` (hash `4a8423d5e91fda00bb7e46540e2b0cf1`), arXiv:1705.08500 (v1 2017-05-23, v2 current).
   Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger, *On Calibration of Modern Neural Networks*,
   Proceedings of the 34th International Conference on Machine Learning, PMLR 70:1321–1330,
   https://proceedings.mlr.press/v70/guo17a.html, arXiv:1706.04599 (v1 2017-06-14, v2 current).
   **Identifier correction recorded here:** the calibration paper is arXiv:1706.04599; the id 1503.03832,
   which an earlier note in this project carried for it, resolves to *FaceNet* (Schroff, Kalenichenko &
   Philbin) — caught by opening the arXiv record instead of repeating the note. **Dropped rather than
   carried:** Angelopoulos & Bates, arXiv:2107.07511, listed previously as optional background, was not opened
   in this pass and §10 reports no coverage guarantee, so it appears nowhere in the body.

**Resolved — no marker remains.** The laya parameter count is cited in §3 as **software** with its version
and pinned revision (PyPI `laya` 0.3.20; `convaiinnovations/laya` at
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`), not as a paper: **no laya paper exists**, the card at that
revision carries no BibTeX and no DOI, and the authors publish only a rounded "421M total". The 421,293,827
count and the 394.78M / 26.51M split are ours, derived in §3 from the artifact's own tensor header and
`runs/latency_matched.json`.

**Withdrawn — no reference will be attached.** The compute-optimal "≈20 tokens/parameter" pretraining law
is gone from §6 rather than cited: re-read in full, neither Hoffmann et al. (arXiv:2203.15556) nor Kaplan et
al. (arXiv:2001.08361) states a tokens-per-parameter constant — the former gives equal scaling ("a = 0.50
and b = 0.50", §3.1), the latter fits `D ∝ N^0.74`, which is not a constant ratio — so §6 now compares only
against the pilot corpus's own ratio. No "≈20" appears anywhere in §6.

**Absent by design.** No paper titled *CALM: Consistency-Aware Language Modelling for Text Classification*
was found, so it is not cited here, is not cited in the SPEC, and must not be added to this draft.

## Proposed SPEC edits — both **applied**: these two sentences in `fd5f138`, the reference sweep and §9.64 (vi) in `800da60`

1. **`SPEC.md:1669`** (risk table) — "15.35M cannot learn instruction-following well enough to clear G1"
   names the default `MynaConfig` count, but the arm that actually scored G1 is the 16,926,848-parameter
   v1b checkpoint at vocab 8192. Proposed: restate the risk row against 16,926,848 / vocab 8192, or mark it
   as a planning-time sentence about the default config. Recorded in SPEC §9.64 (vi).
2. **`SPEC.md:1611-1612`** (§6, "Compute and budget") — "⇒ **0.28 tokens/param**" prints no denominator.
   Proposed: state it as 4,246,106 ÷ 15,353,984 = 0.2765, since that divisor is the only one of the three
   counts in §9.64 (i) that reproduces 0.28 (14,449,280 gives 0.2939, 16,926,848 gives 0.2509). Recorded in
   SPEC §9.64 (v) and (vi).

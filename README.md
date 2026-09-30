# Myna

**The decision engine that never re-reads.** Typed `choice` / `score` / `noul`
decisions over growing observations — tickets, threads, page states,
documents — where the observation is scanned **once** and every later request
only pays for what changed.

Born from a year inside [laya](https://github.com/NandhaKishorM/laya) and
[kev](https://github.com/aashish254/kev): both are non-autoregressive System-1
decision models, and both re-encode the entire state on every request. Agent
loops don't work that way — the state grows at the edge and the questions are
short. Myna is an architecture built for that shape from the ground up.

<!-- gates:release:begin -->
## Where this stands: the release gate, all seven rows

**3 of 7 met, 3 not met, 1 open — the release gate is NOT clear.**

Every verdict below is checked against a committed artifact — the registry
row beside it and the field read out of that artifact by
`uv run python bench/gates.py --check` — and the same words head the
status column of [SPEC.md](SPEC.md) §2.2. The gate that decides whether
this is a product (G1) and the gate that decides what it can read (G4)
are the two that cannot be closed on this box.

| gate | verdict | what the committed artifact prints | what it does not claim |
|---|---|---|---|
| **G1** — Accuracy beats the witness | **not met** | `v1b-kaggle-3600b`'s committed report prints `"pass": false` for this gate: over the 16 cells both harnesses score, per-cell unweighted, **myna 0.4893 · laya 0.6668 · majority floor 0.4331** — **+0.056** over the floor against the **+0.15** required and 0.211 short of 0.70. Its `train.log` is committed beside it: 3,600 updates in 20,225 s on a T4, and a dev curve whose last 1,000 updates still gain **+0.0072** (0.4283 at step 2,500 → 0.4355 at 3,500). For contrast, the untrained control here measures **0.346 / 0.351** (`runs/scratch_decision_v2_test.md`). | not met, measured — and the measurement is an artifact rather than a transcript. §9.30 refuses a figure whose witness is not committed, and until this tick the Kaggle run's `metrics.json` lived in a notebook output directory, so the gate had to sit at `open`; `bench/reproduce.py`'s `v1b-kaggle-macro`, `kaggle-dev-tail` and `kaggle-wall-clock` rows re-derive every figure above from files in `runs/`. What the verdict does *not* settle is dose: that tail slope is why a longer run is not a free +0.21, and §9.38 writes the correction of the remembered "flat from step 2,250" claim. The void pre-`385e06c` checkpoint's 0.338 is deliberately not a row here — a `not met` may not rest on a void artifact (§9.23), so it stays in SPEC's prose as contrast and out of the evidence. |
| **G2** — Latency claim survives equal-footing re-measurement | **met** | one M5 process, both engines fp32, `Agent.system_one` called directly, ladder capped inside laya's own 1024 window: **3.04×/3.21×/4.12×/3.60×** end-to-end and **6.78×/4.67×/4.75×/3.65×** ask-only at 1/5/10/50 questions, each cell the minimum of two committed runs. | met as a ratio, which is all G2 asks. It carries no accuracy implication: the model that is faster is the one that still fails G1, and the millisecond columns behind these ratios are load-dependent on a box other sessions share (§9.23) — quote the ratio, never the milliseconds. |
| **G3** — Deployment works for real | **met** | the exported artifact answers three typed decisions in real Chrome 153 with the same probabilities the torch engine prints — 6/6 parity checks, both isolation modes, desktop and mobile widths — and its bytes, cold-load and p50 are measured there with the box's load average printed beside them. | met as a mechanics gate, and the caveat is part of the verdict rather than a footnote: what ships is v0, which fails G1 and G5; the ≤ 20 MB int8 route is open on bytes and closed on agreement (§9.28); the Apple-silicon int8 route closes for the opposite reason — 17.40 MiB at 9.24e-03 and no latency win at all, so fp32 is the artifact on both paths (§9.29). |
| **G4** — Long-context is *correct*, not just cheap | **not met** | v0's needle curve prints **0.188 at 128 tokens** against a 0.167 uniform floor — at chance on the *shortest* rung — so the longer rows are the decay of nothing, and `runs/needle_myna-v0.md` prints `G4: not measured by this run` in place of a table a reader could quote. | not met, and not met by a run that cannot answer the question. The 16k *state* is measured, fixed at 576 KiB and cheap (§9.22); the 16k *decision* needs the 4k truncated-backprop checkpoint, which is 7b / `KAGGLE`. Those are two different claims and only the first one is settled. |
| **G5** — Useful confidence, with abstention | **not met** | the curve's own `g5.pass` is `false`: accuracy climbs **0.349 → 0.535** as coverage falls 1.00 → 0.20 over 448 rows / 568 questions, **no rung reaches 0.95** (the max is 0.596, at 10% coverage), and G5's three named sources measure 0.000 / 0.025 / 0.150. | not met on the level, met on the machinery, and the gate is the level. `Myna(abstain_below=t)` withholds the commitment with a measured reason and the fallback seam labels which engine committed — a confidence that ranks the answers of a model that cannot answer is routing, not the product G5 describes (§9.24). |
| **G6** — No regression on what already worked | **open** | v0's measured test macro is **0.9523** — the macro of the nine `=== test ===` rows in `runs/train-v0.log`, recomputed from its rows because the log prints no overall line — so the level clears today. | open rather than met, because the gate is written against the *next* checkpoint: it says "no regression", and there is no trained v1 artifact in this repo to regress. A draft of this row cited 0.951, a figure no artifact prints (§9.30), which is why the value here is computed from the log's own rows. |
| **G7** — Reproducibility | **met** | `make repro` is green at this tick: 30 registry rows bind every published table cell to one command and one committed witness, and the 74 quoted figures are re-read out of those files rather than out of the prose. The python commands among those rows are checked against the tool behind them — its `--help` must still print every flag the row publishes (§9.43). The seeds live inside the printed commands (`--seed 0`, `--seeds 0 1`), not in sentences about them. | met as a binding, not as a rebuild, and the gap is disclosed rather than absorbed: no target *executes* the registry end to end — 20 rows re-run on this box, 4 retrain a checkpoint, 2 need a laya checkout on `PYTHONPATH`, 3 need Google Chrome and 1 is `KAGGLE`. That is TODO 8b, which stays unticked for exactly this reason. |

<!-- gates:release:end -->

## The architecture

1. **Gated linear attention trunk.** Each layer carries a fixed-size matrix
   state `S ∈ R^{H×d_k×d_v}` (96 KiB/layer at production width), not a KV cache
   that grows with
   context. Three execution forms — recurrent (streaming), chunked (training,
   linear memory), quadratic (reference) — are proven numerically identical,
   in float64, in CI.
2. **Persistent state = the cache.** Appending to the observation continues
   the scan from the cached `S`. Nothing below the append point is ever
   re-read, and the result is bit-exact versus a full pass — not approximate,
   not distilled, *identical*. The only thing retained per observation is the
   stack of `S` matrices: 576 KiB at production width, for a 128-token state
   or a 16,384-token one.
3. **Questions are branches, not prompt tokens.** Each question resumes every
   layer from the cached state-end, so a question reads the whole observation
   and — by architecture, not masking — nothing of any other question.
   Adding, removing, or reordering questions changes no other answer.
4. **No vocabulary readout anywhere.** All three answer types share one
   pointer-probe head over option spans. No generation, nothing to parse,
   nothing to hallucinate.

<div align="center">

| | params | weights on disk | state memory | context |
|---|---|---|---|---|
| **myna-v0** | 14.45M · *m* | 55.13 MiB fp32 · *m* | 576 KiB, 128 or 16,384 tokens alike · *m* | 16,384 scanned and answered from · *m* |
| laya | 421.29M, read from their checkpoint · *m* | ~1.7 GB fp32 · *p* | holds none — re-encodes per request | 1,024 + 256 head, from their `config.json` · *m* |
| kev-4B | ~4B, published | ~8 GB fp32 · *p* | KV grows with the context · *p* | — |

Every cell above carries its own tag. ***m*** measured here, from a file in this
repo: myna's parameter count and artifact size are `n_params` and
`sizes.mlx_fp32.file_bytes` in `runs/bench_mlx_int8.json`, the 576 KiB is the
`state_bytes_total: 589824` that `runs/browser_g3.json` prints on every one of its
four Chrome rows, and laya's two cells are the line `runs/latency_matched.md`
prints when it opens their snapshot. ***p*** projected — arithmetic from a
parameter count against a model we did not open, which is why it is not a
comparison cell. The decision *accuracy* at long context is a third thing again
and it is **gated**: the needle test puts v0 at 0.188 on a 128-token state against
a 0.167 floor, i.e. at chance before length even enters, and
`runs/needle_myna-v0.md` prints `G4: NOT MEASURED here` rather than a decay curve.
The context column is a **cost** statement, and the only 16k number this repo has
measured. Long-context *correctness* waits on the checkpoint trained for it
([SPEC.md](SPEC.md) §2.2 G4, §9.25).

</div>

Reproduce the params and weights columns — `n_params`, and both `file_bytes` — with the
MLX harness that owns the checkpoint. It is also the run whose `--drift-rows 12` prints
the 576 KiB row at every state length from 128 to 16,384:

```bash
uv run python -m bench.bench_mlx --ckpt runs/myna-v0 --reps 20 --drift-rows 12 --out runs/bench_mlx_int8.md
```

and the context column with the needle run that the 0.188-against-0.167 pair came from
(`runs/needle_myna-v0.md`, and its `G4: NOT MEASURED` line):

```bash
uv run python bench/eval_needle.py --ckpt runs/myna-v0 --device cpu --n 16 --seed 0 --lengths 128 1024 4096 8192 16384
```

## On the upstream corpus: three rows, including the one nobody likes

G1 is an accuracy gate on `kev`'s frozen `decision-v2` **test** split, and the
checkpoint that answers it trains on Kaggle. Three numbers on that split are
printable from this box today without training a model, and [SPEC.md](SPEC.md) §8
requires all three rather than the flattering pair:

| row | test MACRO over the 16 (source, question) cells both sides score | footing |
|---|---|---|
| laya 421M — their engine, their suite, 40 rows/source | **0.667** · *m* | `runs/laya_decision_v2_test.json`, joined cell by cell |
| **myna, untrained control** — random init, the suite's own tokenizer, current code | seed 0 **0.346** · seed 1 **0.351** · mean **0.348** · *m* | `runs/scratch_decision_v2_test.md` |
| myna trained on this corpus (G1) | **gated here** — the Kaggle checkpoint has run and its verdict is TODO 3i's to publish; until its witness is a file in this repo, a figure for this row is a number with nothing behind it (§9.30) | `kaggle/run.py`, TODO 3i |
| majority-label floor, same 16 cells | 0.433 · *m* | `runs/report_scratch_vs_laya.log` |
| uniform-chance floor, same 16 cells | 0.332 · *m* | same |
| majority-label **clip** of that control — `max(model, floor)` per cell, an oracle not a result | 0.436 · *m* | `runs/report_scratch_vs_laya.json` |

The last row is the one that must not be read as a score. `myna.report` now prints
`clip = max(model, majority)` beside each cell's own accuracy, and the control's
16-cell clip macro is **0.436** against its model macro of **0.346** — the +0.090
difference is "answering nothing, per cell, with this split's majority label", which
is an *oracle* statistic (the floor is measured on the rows being scored), it is
available to no model, and the report says so on the line that prints it
(`runs/report_scratch_vs_laya.log`: 13 of 16 cells answer below their own floor).
Printing it is reporting; *acting* on it would be a per-cell fallback rule — a
leaderboard decision, not a learning improvement, and SPEC §5 P9: 9c builds the
report and stays out of that decision.

The same reasoning is why the *scope* question is priced rather than argued. Two cells
(`banking77/intent`, `mnli/relation`) answer below their own floor and do not beat their own
permutation nulls, so a reader may reasonably ask whether G1 should be scored without them.
`uv run python -m bench.scope_pricing` answers it from the committed run: dropping them moves
the level 0.4893 → 0.5334 **and the majority floor 0.4331 → 0.4666 with it**, because the cells
myna wins are the ones a majority classifier wins too, so the margin G1 gates on moves
+0.0563 → +0.0668 — eleven-thousandths. Dropping their whole sources is the same 14 cells, and
even removing all five below-floor cells reaches 0.5526, still 0.147 short of 0.70. No scope
choice here passes G1 (`runs/scope_pricing.json`); whether those two cells belong in the claim
at all stays P9: 9d's call, and it is a decision about what the published number means. The
table is mutation-checked like everything else this repo publishes
(`uv run python bench/mutation_scope_pricing.py`, 24 mutants): its first pass caught 10 of them,
and the 13 survivors were all in the harness's `main()` — the guard's individual comparisons, how
the witness is assembled, and what the printed table puts under which column heading — not in the
arithmetic, where 8 of 8 died (SPEC §9.46).

The disclosure in the middle. The one checkpoint this repo ever trained on the
upstream corpus scores **0.338** on those same 16 cells
(`runs/report_void_vs_laya.log`, from
`uv run python -m myna.report --suite data/decision-v2-pilot --split test --metrics runs/myna-v1-rich/metrics.json --laya runs/laya_decision_v2_test.json --out runs/report_void_vs_laya.json`)
— *below* its own control, by 0.008, which is the
same order as the control's 0.005 seed spread, and void as evidence regardless
because it predates the `385e06c` batching fix (§5 P1). So the honest sentence
about this repo as it stands is: **no committed measurement of myna's
architecture on this corpus exists in either direction** — the control that could
not have learned anything sits on the chance floor, and the checkpoint that could
is not here. What the control does prove is the one falsification that mattered —
a probe head or an answer head that leaked the label would sit above the 0.332
chance floor, and it sits on it. The third row waits on its artifact rather than
on a run: `v1b-kaggle-3600b` has trained and reported its own metrics, and TODO 3i
publishes that number the moment its witness is a file in this repo, because a
figure for a checkpoint this repo cannot open is a number with nothing behind it —
which is the mistake §9.30 exists to prevent.

Reproduce the control (1,440 rows scored on CPU; no gradient step, on any corpus):

```bash
uv run python bench/eval_scratch.py --split test --seeds 0 1 --out runs/scratch_decision_v2_test.md --metrics-out runs/scratch_metrics_test.json
```

and the join that puts it beside laya's per-cell numbers:

```bash
uv run python -m myna.report --suite data/decision-v2-pilot --split test --metrics runs/scratch_metrics_test.json --laya runs/laya_decision_v2_test.json --out runs/report_scratch_vs_laya.json
```

Two statistics that must never be subtracted from each other. The table is the
*unweighted mean over the 16 cells both sides scored*; laya's own harness prints a
*row-weighted* overall of 0.6319 over its 546 sampled answers, which is a different
statistic over a different sample — a cell counted by rows and a cell counted once
are not comparable, and arithmetic across the two is how §9.30's eight dead cells
were born. The join also merges the competitor's two `contrastive/decision` rows (one
qid asked as a 4-way choice over 24 answers and as a noul over 16) into that cell at
**0.500** by rows; until §9.32 the report kept whichever row came last — 0.625 — which
had inflated every per-cell laya figure here by 0.008–0.009.

## Measured on an Apple M5 (10-core, 32 GB)

One process, both engines resident, `device=cpu`, both fp32 — the same Python
`str` state and the same question `dict` handed to each engine. laya is called as
`Agent.system_one` on its published `55cf4c4` checkpoint, on laya's own
Stripe-payout ticket (87 myna / 57 laya tokens), alternating 3-option `choice`
and `noul` questions exactly like laya's own latency bench. p50 over 10 reps
after 3 warm-ups:

| questions | laya `system_one` | myna end-to-end | ratio | myna, state already scanned | ratio |
|---|---|---|---|---|---|
| 1 | 100.7 ms | 33.1 ms | **3.04×** | 14.9 ms | **6.78×** |
| 5 | 272.0 ms | 74.9 ms | 3.63× | 55.5 ms | 4.90× |
| 10 | 506.3 ms | 120.8 ms | 4.19× | 98.2 ms | 5.15× |
| 50 | 3,589.5 ms | 870.0 ms | 4.13× | 787.1 ms | 4.56× |

Reproduce: `PYTHONPATH="<laya checkout>" .venv/bin/python -m
bench.bench_latency_matched` → `runs/latency_matched.md`. Every cell in this table
is *m* — measured, on this box, by the command above, and the numbers after it
("14.5–17.4 ms", "−10 µs/token", "R² 0.74") are least-squares fits over the same
run's ladder, which is why they are quoted as fitted rather than as read. That
table is one run;
**the claim shipped in `SPEC.md` §2.1 is the per-row minimum across the two
committed runs — 3.04× / 3.21× / 4.12× / 3.60× end-to-end and 6.78× / 4.67× /
4.75× / 3.65× streaming** — because two runs of this command on this laptop differ
by up to 22% in absolute ms (other sessions share these cores), and the
conservative number is the only one that belongs in a README (SPEC §9.23).
Accuracy is not measured by this table and nothing here implies it.

What the *shape* of the cost says, fitted over a state ladder capped inside laya's
1024-token window: answering from a cached state costs **14.5–17.4 ms whether that
state is 66 or 1,060 tokens** (fitted state slope −10 µs/token, i.e. zero) — the
fixed-size state, measured rather than drawn. The scan is **not** flat: within a
256-token chunk it is quadratic, so the sub-40 ms decision is a short-state result
(SPEC §9.22). laya's cost is state × questions — one sequence carrying the whole
state, once per question — which is why its additive fit lands at R² 0.7356 while
myna's end-to-end fit lands at 0.9915 (the ask-only fit is 0.9619, and the three
numbers are the harness's own `fit` block, not restatements of each other).

*An earlier version of this section published 33×–239× speedups against a "~500 ms
laya floor". The floor was `Router` truncating the state at 1,024 tokens while
myna read all of it, and the speedup column was myna against itself. Withdrawn —
SPEC.md §9.20.*

## In a browser tab, measured

The exported artifact runs with no server and no build step. `browser/index.html`
re-implements the tokenizer in JavaScript from `tokenizer.json` alone, chains
`state_step.onnx` one call per 256-token chunk, runs the pointer head and the
abstention gate over the option spans from `head.bin`, and answers the same three
typed decisions the Python engine answers — checked against the Python engine by
`browser/parity.mjs`, the same module `node browser/selftest.mjs` fails a build on
(7 checks in node, 6 in the tab, which has no filesystem to compare bytes against).

Google Chrome 153, headless, cold cache, throwaway profile, 15 decisions per row
(`bench/browser_g3.mjs` → `runs/browser_g3.json`, and `runs/browser_g3_int8.json`
for the right-hand column). Every cell is *m*:

| | fp32 — what ships | int8 — attempted |
|---|---|---|
| artifact the page fetches | 59.32 MiB | **18.56 MiB** |
| bytes on the wire, incl. the 13.58 MiB wasm runtime | 73.09 MiB | 32.33 MiB |
| `weights.bin` requests | **1** (shared by both graphs) | 1 |
| cold load | 254–394 ms | 214–336 ms |
| p50 per decision, 10 threads / 1 thread | 436 / 742 ms | 413 / 757 ms |
| option probabilities vs the torch engine | **1.03e-06** | 6.48e-02 |
| Chrome's own parity checks | **6/6** | 4/6 |

Two things that table says and one it does not. It says the trunk really is
transferred once — 93.7 MiB of duplicated fp32 weights in the first export (at its own
8 × 256-token shape, per SPEC §5 P6 6a), 59.32 MiB now, with Chrome's network log as
the witness rather than a sum in a README.
And it says the ≤ 20 MB int8 route is a **byte win and an agreement loss**: dynamic
quint8 lands at 18.56 MiB and moves option probabilities 648× past the bound the
fp32 artifact meets, for 5% latency on a box that was busier during the int8 run —
so int8 ships nowhere (SPEC §9.28; §9.29 is the same verdict on the Apple path, for
the opposite reason). What the table does *not* say is anything about
accuracy: the checkpoint in that artifact is v0, which still fails the real-corpus
gate below.

```bash
uv run python -m myna.onnx_export --ckpt runs/myna-v0 --out runs/onnx \
    --scan-chunk 16 --n-chunks 4 --suite data/decision-v2-pilot   # → runs/onnx_parity.json
uv run python -m myna.onnx_export --ckpt runs/myna-v0 --out runs/onnx-wide --questions 2 --q-len 1152 --scan-chunk 64 --report runs/onnx_parity_widest.json
uv run python bench/browser_expected.py                            # → browser/expected.json
npm ci && npm run selftest                                         # 7 checks, node
npm run g3                                                         # Chrome: bytes, cold, p50, shots
npm run g3 -- --artifact runs/onnx_int8 --out runs/browser_g3_int8.json   # the int8 column
uv run python bench/quantize_int8.py                               # the int8 row above
```

The second export is the widest real request in the suite — 2 questions × 1,152 tokens
over a 16k state, 1,067 tokens after the state is folded in — and it is the parity cell
that the `≤ 1e-3` bound is actually measured on; the first line's `--suite` export is
what backs the probabilities row. The two `npm run g3` lines are the same harness
pointed at the two artifacts, left and right column of the table above.

Absolute milliseconds from this laptop are a range, not a point, so the witness
files carry `uptime`'s load average next to every number (SPEC §9.23), and the tab's
own memory reading (176 MiB) counts JS-managed heap only — it does not attribute the
54 MiB mounted into the wasm heap, which is why no total-tab memory figure is claimed.

## On an Apple silicon Mac, measured

The same checkpoint serves natively through MLX: `src/myna/mlx_model.py` is a port of
the torch forward over identical weights, so there is no conversion step and no
re-export. One M5, both engines, same tensors, median of 20 reps, per-row **minimum**
of two committed runs (`bench/bench_mlx.py` → `runs/bench_mlx_int8.md` and
`runs/bench_mlx_int8_run2.md`, load average 4.55 and 3.94 at print). Every cell is
*m*, including the ratio ranges, which are per-row minima rather than best cases:

| engine | vs PyTorch on MPS, 128 → 16,384-token state | artifact on disk |
|---|---|---|
| MLX fp32 | **1.76–2.22×** | 55.13 MiB |
| MLX int8, group 64 | **1.79–2.41×** | **17.40 MiB** |

int8 is 3.17× smaller and **not faster** — the sign of the int8 − fp32 difference flips
between the two runs on 4 of the 7 state lengths, and fp32's own run-to-run spread
reaches 15.2% on the widest row. `bench/diag_mlx_int8_gem.py` prices that at the kernel:
5 of the 6 linear shapes this model actually uses are *slower* quantised (0.43–1.06×
fp32/int8) even though int8 reads 3.6× fewer weight bytes, because the quantised pass
gets 10–37 GB/s of weight read against fp32's 55–257 on the same box. And it is not free
in agreement — over 20 real `calibration.jsonl` questions the port moves option
probabilities 1.20e-03 while quantisation moves them 9.24e-03 and changes one argmax, on
a question whose top two options sat 0.0008 apart. So fp32 is what ships on this path
too, and the int8 artifact stays in the tree for whoever needs the bytes for a reason
other than speed (SPEC §9.29).

```bash
uv run python -m bench.bench_mlx --ckpt runs/myna-v0 --reps 20 --drift-rows 12 --out runs/bench_mlx_int8.md
uv run python -m bench.bench_mlx --ckpt runs/myna-v0 --reps 3 --lengths 128 --drift-rows 12 --keep trunk.tok.weight .gate.weight --fp32-out runs/myna-v0-mlx-fp32-keepgate --int8-out runs/myna-v0-mlx-int8-keepgate --out runs/bench_mlx_int8_keepgate.md
uv run python bench/diag_mlx_int8_gem.py --iters 200 --trials 7 --out runs/mlx_int8_gem_probe.md
```

The first line is both ratio tables (`runs/bench_mlx_int8.md` and its `_run2` twin are
two runs of it, and `runs/bench_mlx_int8.json` is the `n_params` / `file_bytes` witness
the architecture table quotes). The second keeps `trunk.tok.weight` and `.gate.weight`
in fp32 — the one row of the six that flips sign when they are, which is why it is run
separately rather than folded into the first. The third prices the claim at the kernel.

## Trained v0 — accuracy on held-out data

9,000 steps on the synthetic typed-decisions corpus, batch 32, MPS, one M5 — the
log's own elapsed counter reads **15,693 s (4.4 h) at step 8999** — and dev and
test use noun pools the trainer never showed the model. Every cell below is *m*,
copied from `runs/train-v0.log`'s final `=== test ===` block; the overall figures
are the **macro over those nine rows**, which is also the only way to average them
honestly from what the log prints:

**test 0.952 · dev 0.960** (the `dev-mid` probe line prints 0.9682 at steps 7500, 8000 and 8500 —
on a prefix subset of that same dev split, not on all of it — and an earlier draft of this section
quoted *that* as the dev overall next to a test figure of 0.951 that no committed artifact prints —
[SPEC.md](SPEC.md) §9.30).

| workflow / question | type | test acc | Brier | ECE |
|---|---|---|---|---|
| incident/page_oncall | noul | 0.946 | 0.046 | 0.026 |
| incident/severity | score | 0.958 | — | 0.019 |
| incident/team | choice | 0.960 | — | 0.020 |
| moderation/action | score | 0.974 | — | 0.012 |
| moderation/category | choice | 0.948 | — | 0.027 |
| moderation/repeat_appeal | noul | 0.946 | 0.043 | 0.037 |
| support/churn_risk | noul | 0.963 | 0.028 | 0.016 |
| support/department | choice | 0.947 | — | 0.029 |
| support/urgency | score | 0.928 | — | 0.042 |

Generalization is real, not memorization: the dev→test gap is 0.76 of a point
(0.9599 − 0.9523) on fully disjoint nouns. Test ECE runs 0.0115–0.0415 with the
single refit temperature (3.0) that `runs/train-v0.log` prints. A short RLCD
pass — 800 steps maximizing the Brier proper score under a KL leash to the
supervised model — then **halved it: mean ECE 0.0191 → 0.0076, mean Brier 0.0258
→ 0.0232, dev accuracy 0.9626 → 0.9657** (`runs/rlcd_v0.log`, every one of the 9
questions improved or held). Note what that dev accuracy is: `rlcd.py`
regenerates the dev split at 600 rows per workflow, so 0.9626 is a *different
dev draw* than the 0.9599 above — the published claim is the before/after pair
inside one draw, not either level. The policy here *is* the
reported distribution, so the score is optimized differentiably, no REINFORCE
(PLAN.md decision 10 has the post-mortem of trying it the generative way).

These two are the only tables here whose reproduction is a training run, so
`bench/reproduce.py --run` refuses them and `runs/train-v0.log` / `runs/rlcd_v0.log` are
the witnesses instead:

```bash
uv run python -m myna.train --steps 9000 --batch 32 --device mps --out runs/myna-v0
uv run python -m myna.rlcd --ckpt runs/myna-v0 --steps 800 --out runs/myna-v0-rlcd
```

## Knowing when not to answer — risk / coverage

A System-1 decider that never says "not me" is not a fast model, it is an
unauditable one. So the engine can be built with a floor under it, and the
measurement that matters is the trade the floor buys:

```python
from myna.engine import Myna

myna = Myna("runs/myna-v0", abstain_below=0.693)   # the floor the curve below picks
out = myna.observe(
    "China Invokes Deng to Send Tough Taiwan Message BEIJING (Reuters) - China invoked late "
    "leader Deng Xiaoping on Saturday in its campaign to recover Taiwan, lauding his proposal "
    "to recover the island by peaceful means, but warning against any move toward independence."
).ask({"topic": {"type": "choice", "instructions": "What is the topic of this article?",
                 "criteria": {"world": "world news", "sports": "games, athletes, teams",
                              "business": "companies, markets, economy",
                              "scitech": "research, gadgets, software, space"}}}})
a = out["answers"]["topic"]
```

The gold label for this row of `calibration.jsonl` is `world`, at 0.001. The engine was
about to answer `sports`, and the floor took the decision away:

```
a["choice"]  ->  None          out["policy"] -> {'abstain_below': 0.693, 'abstained': ['topic']}
a["abstain"] ->  True
a["reason"]  ->  'sports at p=0.602, 0.341 ahead of the runner-up, under the 0.693 floor'
a["probabilities"]  ->  {'world': 0.001, 'sports': 0.602, 'business': 0.136, 'scitech': 0.261}
```

`probabilities` rides along *because* the answer was withheld: it is what a fallback re-ranks
and what the curve below is sorted by. The withheld field is `None` on purpose — a caller
that ignores `abstain` gets an error at the seam instead of a confident wrong label.

Refusing costs the *decision*, not the numbers: `noul` keeps reporting raw
`p(Yes)` (the Brier score needs it), every answer keeps its `confidence` and its
runner-up `margin`, and `ask()` echoes the floor in force plus which questions it
withdrew. `myna.serve --abstain-below 0.693` puts the same policy behind HTTP, and
`/v1/health` names it.

The curve below is every question in `calibration.jsonl` (448 rows, 568 questions),
ranked by the probability of the side committed to — every cell *m*, from
`runs/risk_coverage.md`. `floor` is the value you would
hand the engine, and the run re-runs the engine at that floor to check it abstains
on exactly the rows the curve withheld — 227 against 227 at the 0.693 floor.

| coverage | questions answered | accuracy | risk | floor |
|---|---|---|---|---|
| 1.00 | 568 | 0.349 | 0.651 | 0.074 |
| 0.90 | 511 | 0.376 | 0.624 | 0.444 |
| 0.80 | 454 | 0.374 | 0.626 | 0.547 |
| 0.70 | 398 | 0.392 | 0.608 | 0.622 |
| 0.60 | 341 | 0.413 | 0.587 | 0.693 |
| 0.50 | 284 | 0.426 | 0.574 | 0.772 |
| 0.40 | 227 | 0.445 | 0.555 | 0.851 |
| 0.30 | 170 | 0.465 | 0.535 | 0.935 |
| 0.20 | 114 | 0.535 | 0.465 | 0.974 |
| 0.10 | 57 | 0.596 | 0.404 | 0.994 |

Read the shape, not the level. Accuracy climbs **+0.247** from full coverage to the
top decile, so this model's confidence ranks its own answers — and it does that on
`v0`, a checkpoint trained on synthetic data that scores near the guessing floor on
this corpus (the agnews row above is one of them: `world` at 0.001 against a withheld
`sports` at 0.602). That is the harness proving it measures something; it is not
the gate. G5 wants ≥ 0.95 accuracy at ≥ 60% coverage on banking77 + dbpedia14 + trec,
and on v0 **no rung reaches 0.95** (max 0.596, at 10% coverage) with those three sources
at 0.000 / 0.025 / 0.150 — so the pass is gated on the real-trained checkpoint
([SPEC.md](SPEC.md) §2.2, §9.24).

`bench/risk_coverage.py` prints it, `tests/test_risk_coverage.py` checks it against
a synthetic oracle whose answer the test knows in advance, and
`bench/mutation_p5.py` (49 mutations, all caught) checks that each published figure
moves when — and only when — the claim behind it stops holding. Reproduce:

```bash
uv run python -m bench.risk_coverage --ckpt runs/myna-v0 --out runs/risk_coverage.md
```

## Every table, and the command that made it

SPEC §9.30 buried eight headline cells that had been copied out of an artifact into
prose and then copied again. This is the repair pointed the other way: one line per
published table, the command that regenerates it, and the file its numbers came from.
`bench/reproduce.py` holds the pairs, and either of

```bash
make repro                                               # or:
uv run python bench/reproduce.py --check     # every row, or it says which one drifted
```

proves, for each row, that its witness is still committed, still contains the figure the
docs quote from it, and that the prose around the table — not this index, which is this
file's own registry echoed back and is cut out before matching — still prints the
command. The fourth assertion asks the tool rather than the file: every row whose command
names a python target gets `--help` from it, and each flag the row publishes has to be in
the answer, because a renamed flag leaves the other three untouched (§9.43).
The registry's *size* is counted out of `ROWS` by `bench/gates.py` and read back out of
SPEC §2.2 — the gate table's one hand-copied row — so `make gates` reds on a stale count
instead of printing it (§9.44).
`tests/test_reproduce.py` (33 tests) pins the block below to the registry, so a
row cannot appear here without being added there, and
`bench/mutation_reproduce.py` breaks the checker one promise
at a time — a witness that stopped existing, a figure the artifact no longer prints, an
index that counts as its own evidence, a flag check that stopped asking — and requires the
tests to go red: **29/29 all caught**, in `runs/mutation_reproduce.log`. Rows tagged
`external-*` need something this repo does not carry (a laya checkout, Google Chrome);
the `gated-kaggle` row has **no witness on purpose** — that figure does not exist yet,
and no local run may invent it. `make repro-show ROW=ID` prints what a row would do and
`make repro-run ROW=ID` does it, overwriting a committed witness the README quotes —
hence the two-step.

The release-gate table at the top of this file is deliberately **not** a row here. Its
command is

```bash
make gates                                               # or:
uv run python bench/gates.py --check    # every verdict, or it says which one lost its witness
```

and what that proves is different in kind: `bench/gates.py` imports the registry above
rather than restating it, so a verdict cannot be greener than the artifact under it, and it
re-reads one named field from a committed JSON file per gate (`g5.pass`,
`baseline.readable`, `inference_only`, `meta.myna_dtype`). But the table itself is
*generated* from the same declarations it would check, and §9.31 says what a checker
generates is not evidence — so the evidence is [SPEC.md](SPEC.md) §2.2's status column and
the files it cites, which is exactly what `--check` reads.

<!-- reproduce:registry:begin -->
```bash
# README architecture table · SPEC §2.1 (params, weights on disk)  [here]
uv run python -m bench.bench_mlx --ckpt runs/myna-v0 --reps 20 --drift-rows 12 --out runs/bench_mlx_int8.md
# README 'Measured on an Apple M5' · SPEC §2.1 G2  [external-laya]
PYTHONPATH="<laya checkout>" .venv/bin/python -m bench.bench_latency_matched
# README 'Trained v0' · SPEC §2.1 G6  [retrain]
uv run python -m myna.train --steps 9000 --batch 32 --device mps --out runs/myna-v0
# PLAN v1 · TODO 4 (RLCD scoring pass)  [retrain]
uv run python -m myna.rlcd --ckpt runs/myna-v0 --steps 800 --out runs/myna-v0-rlcd
# README abstention table · SPEC §2.2 G5  [here]
uv run python -m bench.risk_coverage --ckpt runs/myna-v0 --out runs/risk_coverage.md
# SPEC §2.2 G4 · README context column  [here]
uv run python bench/eval_needle.py --ckpt runs/myna-v0 --device cpu --n 16 --seed 0 --lengths 128 1024 4096 8192 16384
# SPEC §2.1 G1 competitor row · README real-corpus table  [external-laya]
uv run python bench/eval_laya_real.py --split test --n-per-source 40 --seed 0 --out runs/laya_decision_v2_test.json
# SPEC §2.2 G3 (torch↔onnxruntime) · README browser section  [here]
uv run python -m myna.onnx_export --ckpt runs/myna-v0 --out runs/onnx --scan-chunk 16 --n-chunks 4 --suite data/decision-v2-pilot
# SPEC §5 P6 6b (the 1,067-token request)  [here]
uv run python -m myna.onnx_export --ckpt runs/myna-v0 --out runs/onnx-wide --questions 2 --q-len 1152 --scan-chunk 64 --report runs/onnx_parity_widest.json
# SPEC §5 P6 (the node gate that backs every browser cell)  [external-chrome]
npm run selftest
# README in-tab table, left column · SPEC §2.2 G3  [external-chrome]
npm run g3
# README in-tab table, right column  [external-chrome]
npm run g3 -- --artifact runs/onnx_int8 --out runs/browser_g3_int8.json
# SPEC §5 P6 (the int8 attempt, and the node gate going red)  [here]
uv run python bench/quantize_int8.py
# README 'On an Apple silicon Mac' · SPEC §5 P6 6c  [here]
uv run python -m bench.bench_mlx --ckpt runs/myna-v0 --reps 20 --drift-rows 12 --out runs/bench_mlx_int8.md
# SPEC §5 P6 6c (the one row that flips when the gates are kept fp32)  [here]
uv run python -m bench.bench_mlx --ckpt runs/myna-v0 --reps 3 --lengths 128 --drift-rows 12 --keep trunk.tok.weight .gate.weight --fp32-out runs/myna-v0-mlx-fp32-keepgate --int8-out runs/myna-v0-mlx-int8-keepgate --out runs/bench_mlx_int8_keepgate.md
# README 'and it prices that at the kernel' · SPEC §5 P6 6c  [here]
uv run python bench/diag_mlx_int8_gem.py --iters 200 --trials 7 --out runs/mlx_int8_gem_probe.md
# SPEC §4.2 (the majority-label floors the accuracy gates are set against)  [here]
uv run pytest tests/test_report.py -q
# SPEC §5 P3 3c (the Kaggle image is python 3.12/3.11, this box is 3.13)  [here]
uv run python bench/check_python311.py
# README upstream table, control row · SPEC §8's disclosure  [here]
uv run python bench/eval_scratch.py --split test --seeds 0 1 --out runs/scratch_decision_v2_test.md --metrics-out runs/scratch_metrics_test.json
# README upstream table, the gap line · SPEC §8  [here]
uv run python -m myna.report --suite data/decision-v2-pilot --split test --metrics runs/scratch_metrics_test.json --laya runs/laya_decision_v2_test.json --out runs/report_scratch_vs_laya.json
# README upstream table's disclosure row · SPEC §5 P1, §9.23  [here]
uv run python -m myna.report --suite data/decision-v2-pilot --split test --metrics runs/myna-v1-rich/metrics.json --laya runs/laya_decision_v2_test.json --out runs/report_void_vs_laya.json
# SPEC §2.1 G1 myna row · TODO 3i  [gated-kaggle]
# no local command: the run this row named is now committed as `v1b-kaggle-macro`, so G1's verdict no longer rests here.
# SPEC §5 P9 9e (the `ce` vs `emd` verdict) · §9.35, §9.36  [retrain]
python bench/ordinal_ab.py --ab-dir /tmp/ab2
# SPEC §2.1 G1 myna row · §5 P8 · KAGGLE_LAUNCH_INSTRUCTIONS.md  [here]
uv run python -m myna.report --suite data/decision-v2-pilot --split test --metrics runs/v1b_kaggle_3600b.metrics.json --laya runs/laya_decision_v2_test.json --out runs/v1b_kaggle_3600b.report.json
# SPEC §9.41 (the dbpedia14 dev→test delta) · TODO 9h  [here]
uv run python -m myna.report --split dev --metrics runs/v1b_kaggle_3600b.metrics.json --out runs/v1b_kaggle_3600b.dev.report.json > runs/v1b_kaggle_3600b.dev.report.log
# SPEC §5 P9 9d · TODO 9d  [here]
uv run python -m bench.scope_pricing
# KAGGLE_LAUNCH_INSTRUCTIONS.md cost table · kaggle/campaign.py  [here]
grep -E "^step" runs/v1b_kaggle_3600b.train.log | tail -1
# SPEC §2.1 G1 · §9.38 · KAGGLE_LAUNCH_INSTRUCTIONS.md dose lane  [here]
grep -E "dev-mid acc" runs/v1b_kaggle_3600b.train.log | uniq | tail -5
# SPEC §5 P10 · `campaign.py`'s P10 pair · TODO 10  [here]
uv run python bench/anti_prior_audit.py --compare --out runs/anti_prior_audit.json
# SPEC §5 P10 9l (§9.48) — Tier 0 input ablation  [retrain]
uv run python bench/diag_question_ablation.py --run-dir runs/v1b-kaggle-3600b --out runs/diag_question_ablation.json
```
<!-- reproduce:registry:end -->

## Quickstart

```bash
uv sync
uv run python -m myna.train          # 4,000 steps by default; v0's 9,000 took 15,693 s on an M5
uv run pytest                        # equivalence + isolation proofs
```

```python
from myna import Myna

myna = Myna("runs/myna-v0")
obs = myna.observe("Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.")
answers = obs.ask({
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking", "drop everything"]},
    "churn_risk": {"type": "noul", "instructions": "Does the writer threaten to cancel?"},
})["answers"]

# the thread continues — only the new message is encoded
obs2 = obs.append("And the second charge is still showing on my card this morning.")

# the observation itself is portable: 576 KiB of state, resume it tomorrow
obs2.save_state("session.pt")  # ... myna.restore("session.pt")
```

### Serve API

```bash
uv run --extra serve python -m myna.serve --ckpt runs/myna-v0 --port 8080
# --abstain-below 0.693 runs the server with a floor under its answers; /v1/health
# reports the value in force, so a deployment cannot forget its own policy
```

| | |
|---|---|
| `POST /v1/predict` | `{state, questions}` → answers (laya `Router.predict` contract) |
| `POST /v1/sessions` | `{state}` → `session_id` — the state is scanned once, here |
| `POST /v1/sessions/{id}/append` | `{text}` → only the delta is scanned |
| `POST /v1/sessions/{id}/ask` | `{questions}` → answers; flat cost as the session grows |
| `DELETE /v1/sessions/{id}` | release the 576 KiB and be done |

## What v0 is and isn't

v0 trains and evaluates on a synthetic typed-decisions corpus (three
workflows: ticket triage, incident triage, content moderation) with disjoint
train/dev/test vocabulary. It proves the architecture learns and streams;
it does **not** yet claim parity with laya/kev on their frozen suites —
[SPEC.md](SPEC.md)'s gates split that into what is settled and what is not.
Speed is measured on one box against laya's own engine and clears its gate
(G2, §2.2); accuracy on the real `kev` decision-v2 corpus (G1) and the
usable-confidence gate (G5, the section above) both wait on a checkpoint
trained on that corpus, which runs on Kaggle rather than here. The full
next-generation research design — free-text questions, state algebra (merge/branch/diff
of cached states), O(window) mid-state edits, unbounded two-timescale memory,
expected-utility decisions, 15 MB on-device model — is specified with
acceptance criteria in [V2_SPEC.md](V2_SPEC.md).

The uncertainty has somewhere to go: `myna.fallback` holds a `Decider` seam and a
registry, so `Fallback(myna, laya)` re-asks the secondary **only** the questions the
fast engine refused, and every answer carries the name of the engine that committed
to it. No single model is load-bearing on that path, and the routing is auditable one
decision at a time rather than reconstructed from logs.

Honest limits: word-order-heavy tasks can be partly solved bag-of-cues style;
mid-state edits (not appends) fall back to a full re-encode — linear in the
state's length asymptotically, but with a quadratic factor inside each 256-token
chunk, so short states are cheaper than long ones (§9.22); the question
branch's fixed cost (**10.1 ms** fitted, 14.5–17.4 ms to answer one question
regardless of state length) is kernel work, not architecture work; and
abstention is a *routing* feature, not an accuracy one — it can decline to
answer, it cannot answer a question the checkpoint never learned (§9.24).

Apache-2.0 · research log in [PLAN.md](PLAN.md) · questions to
[aashish254](https://github.com/aashish254)

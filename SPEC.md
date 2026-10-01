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
| parameters | 421.29M (enc 394.78M + head 26.51M) — *their published figure* | **14.45M in the shipped v0 artifact** (`n_params` 14,449,280, `runs/bench_mlx_int8.json`), **15.35M** at the default `MynaConfig` (vocab 4096 vs v0's 1740) — the same count, two configs; ≤ 32M allowed | measured |
| accuracy, kev decision-v2 **test** | **0.6319** (our own seeded witness, n=546) | **≥ 0.70** gate · 0.75 stretch | projected |
| accuracy, their own typed-decisions bench | 0.766 (2,000 decisions) | not comparable — different suite; do not quote side by side | — |
| p50 latency, 1 question (87-token state) | **100.7–117.7 ms** measured here on the M5 CPU · 39.5 ms on their T4 (their measured) | **33.1–38.3 ms** myna on the same box and process (two runs) = **3.04×** at the per-row minimum, and inside their T4 number | measured (§5 P4 4a, `runs/latency_matched.md`) |
| p50 latency, 50 questions | **2,923–3,589 ms** here · 771.3 ms T4 (their measured) | **812–870 ms = 3.60×** same box; against their *T4* number myna takes 5–13% *longer* (812–870 ms vs 771.3), so the ≥ 5× projection did **not** hold (§9.21) | measured (4a) |
| p50 latency, 1 question, state already scanned | no such API — every call re-reads the state | **14.9–17.3 ms = 6.78×** laya's same call | measured (4a) |
| input window | **512 tokens** english / **1024** typed-decisions (`max_len` 1024 + `head_max_len` 256, read from the checkpoint config by the 4a harness) | the *state* is measured to 16,384 tokens — scanned, held at 576 KiB, answered from, at cost that grows only in the scan (§9.22). The *decision* at that length is not: v0's needle accuracy is 0.312 at 1k and 0.062 at 16k against a 0.167 floor (§9.25) | measured (state), **not met** (accuracy, G4) |
| retained state | KV-free but re-encodes; window-bounded | **576 KiB fixed, independent of length** | measured |
| calibration ECE | **0.081** after temperature fit (their measured) | **≤ 0.02**; 0.0076 on v0 **dev**, whose temperature was fit on those rows — read as optimistic (§9.11) | measured (v0), level disputed |
| refusing a decision | nothing in the response shape refuses: `system_one` returns a choice for every question, and a caller can only threshold `confidence` itself | `Myna(abstain_below=t)` withholds the commitment with a measured `reason`; risk/coverage on `calibration.jsonl` runs 0.349 → 0.596 as coverage falls 1.00 → 0.10, and **no rung reaches G5's 0.95** on v0 (§9.24) | measured (5a/5b), gate not met |
| browser artifact | ~1.7 GB fp32 / ~420 MB int8 (arithmetic) | measured in real Chrome, cold cache: **59.32 MiB fp32** artifact — the trunk shared, `weights.bin` fetched **once** — answering a live page with **6/6 parity checks green**, and **73.09 MiB** on the wire once the 13.58 MiB wasm runtime is counted. Dynamic int8 gets the artifact to **18.56 MiB** (inside the ≤ 20 MB target) and moves option probabilities by **6.48e-02** = 648× the fp32 bound, so it buys bytes and not agreement (§9.28) | measured (6b) |
| on-device serving, Apple silicon | no MLX path — laya's published numbers are a T4 and the 4a harness ran it here on the CPU | **MLX port on the M5 against torch-MPS on the same checkpoint: 1.76–2.22× fp32, 1.79–2.41× int8**, each cell the per-row minimum of two committed runs. int8 gets the artifact to **17.40 MiB** (3.17× smaller, same container) and buys **no milliseconds**: 5 of the 6 linear shapes the model actually uses are *slower* quantised, at 10–37 GB/s of weight read against fp32's 55–257 (§9.29) | measured (6c) |

The latency cells are one process on one M5 laptop, `device=cpu`, both engines fp32. An absolute p50
from that laptop is a range, not a point — two committed runs of the same harness differ by up to
22% on laya's 50-question row and 14% on myna's, because other sessions share these cores
(§9.23). So each ratio cell is the **minimum across the committed runs** (`latency_matched.md` and
`latency_matched_run1_superseded.md`), taken per row, and every cell names the run it came from.
Quote the ratio, never the millisecond figure.

### 2.2 Acceptance gates — the project is not finished until all pass

| # | gate | pass condition | status |
|---|---|---|---|
| **G1** | Accuracy beats the witness | macro decision-v2 test ≥ 0.70 on the frozen split, per-source table published, ≥ 0.4331 majority floor by ≥ +0.15 | **not met, measured** — it was open on a missing *artifact*, not a missing *run*: `v1b-kaggle-3600b` trained 3,600 updates on Kaggle and its `metrics.json`, report roll-up and `train.log` are now committed (`runs/v1b_kaggle_3600b.*`), so the verdict comes from the artifact rather than from a transcript. Over the 16 cells both harnesses score, per-cell unweighted: **myna 0.4893 · laya 0.6668 · floor 0.4331**, which is +0.056 over the floor against the +0.15 required and 0.211 short of 0.70 — the report's own line reads `target 0.7 → not met`. `bench/reproduce.py`'s `v1b-kaggle-macro` row re-derives it. Still in the row for contrast: the untrained control (**0.346 / 0.351**, `runs/scratch_decision_v2_test.md`) and the void pre-`385e06c` checkpoint (**0.338**). The tail slope of that run's own dev curve is why *dose* is not the answer (§9.38) |
| **G2** | Latency claim survives equal-footing re-measurement | same box, same window, same question count, direct `Agent` call not `Router`; published ratio recomputed from that or withdrawn | **met** (4a/4c): one M5 process, both fp32, `Agent.system_one`, ladder capped inside laya's 1024 window — **3.04×/3.21×/4.12×/3.60×** at 1/5/10/50 questions end-to-end and **6.78×/4.67×/4.75×/3.65×** on the ask-only path, each the minimum of two committed runs, `runs/latency_matched*.md`. G1 is *not* implied: v0's accuracy still fails (§4.2) |
| **G3** | Deployment works for real | ONNX browser build answers a live page's decisions; bytes + p50 + cold-load measured in Chrome | **met, as a mechanics gate only.** `browser/index.html` runs the exported artifact in real Chrome 153 and answers three typed decisions with the same probabilities the torch engine prints (6/6 parity checks, both isolation modes, desktop and mobile widths, `runs/browser_g3.json`); bytes, cold-load and p50 are measured there with the box's load average recorded beside them. What this does **not** say: the artifact it deploys is v0, which still fails G1 and G5, the ≤ 20 MB int8 route is open at the byte level and closed at the agreement level (§9.28), and the Apple-silicon int8 route closes for the opposite reason — 17.40 MiB at 9.24e-03 and **no** latency win at all, so fp32 is the artifact on both paths (§9.29) |
| **G4** | Long-context is *correct*, not just cheap | needle-style typed decision ≥ 0.90 at 4k and ≥ 0.85 at 16k state | **not met, and not yet measurable on this box.** v0's needle curve (`runs/needle_myna-v0.md`) reads 0.188 at 128 tokens against a 0.167 floor — at chance on the shortest rung — so the longer rows are decay of nothing, and the harness now says so itself rather than printing a table (§9.25). The 16k *state* is measured and cheap (§9.22); the 16k *decision* needs the long-context checkpoint, which is 7b / `KAGGLE` |
| **G5** | Useful confidence, with abstention | risk/coverage curve on `calibration.jsonl`; ≥ 0.95 accuracy at ≥ 60% coverage on banking77 + dbpedia14 + trec | **not met on the level, met on the machinery.** Abstention ships: `Myna(ckpt, abstain_below=t)` withholds the commitment and prints the measured reason, the fallback seam labels every answer with the engine that committed, and the curve is a committed artifact (`runs/risk_coverage.md`, 448 rows / 568 questions). Accuracy on v0 climbs **0.349 → 0.535** as coverage falls 1.00 → 0.20 — the confidence ranks the answers, on a checkpoint that cannot do the task — and setting the 0.693 floor on the engine abstains on **227** questions where the curve withheld **227**. But **no rung reaches 0.95** (max 0.596, at 10% coverage) and G5's own three sources sit at 0.000 / 0.025 / 0.150, so the pass needs the `KAGGLE` checkpoint (§9.24) |
| **G6** | No regression on what already worked | synthetic v0 test ≥ 0.94 | open — v0's measured test macro is **0.9523** (`runs/train-v0.log`, macro of its nine `=== test ===` rows), so the level clears today; the gate stays open until a v1 checkpoint is run against it, and a draft of this row cited "0.951", which no artifact prints (§9.30) |
| **G7** | Reproducibility | one command per result, seeds pinned, witness JSONs committed, `pytest` green | **met, as a binding rather than a rebuild.** `bench/reproduce.py` holds 32 registry rows, each binding one published table cell to one command and one committed witness, and `make repro` re-reads the 88 quoted figures out of those artifacts rather than out of the prose; the python commands among those rows are checked against the tool behind them — its `--help` must still print every flag the row publishes (§9.43); the seeds live inside the printed commands (`--seed 0`, `--seeds 0 1`), not in sentences about them. What this does **not** say: no target *executes* the registry end to end — 21 rows re-run on this box, 5 retrain a checkpoint, 2 need a laya checkout on `PYTHONPATH`, 3 need Google Chrome and 1 is `KAGGLE` — which is why TODO 8b stays unticked and why G7 is discipline rather than a one-command rebuild. Every count in this cell is now read back against `ROWS` by `bench/gates.py`, so this row — the table's one hand-copied cell — goes red rather than quietly stale (§9.44) |

**This column is not a claim.** `make gates` (`bench/gates.py --check`) re-reads each cell's
leading verdict word, the registry row that gate cites, and the named field inside a committed
artifact — and a `met` verdict is refused if the field it rests on prints `false`, while an
`open` verdict is refused unless the gate names a run this box cannot do. It also reads G7's
counts back out of this table and compares them with `ROWS`, because §2.2 is the one place a
gate row is typed by hand rather than generated — a stale number here now goes red instead of
riding along (§9.44). As of P8 8d the tally
is **3 met, 3 not met, 1 open**, so the release gate is not clear; README's copy of that table is
generated from the same file and is therefore *not* evidence (§9.31).

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
`scope_pricing.py` (the 9d scope table: the committed report's cells subset four ways, the
roll-up *imported* from `myna.report` rather than restated, and the guard that refuses to print
unless the unfiltered row reproduces the report to the digit) and its gate
`mutation_scope_pricing.py` (24 mutations; §9.46 is why it exists and what its first pass found),
`mutation_kaggle_bundle.py` (the entrypoint + packager gate, 48 mutations — and §9.40 is the reason
its catching test is capped to one update rather than a training run, and §9.48 why six of them
are P10's launcher plumbing), `check_python311.py` (the 3.11 witness
that runs the package on a real 3.11 interpreter and exits 2 rather than skipping),
`scan_secrets.py` (`make secrets`, §5 P0: every blob in the object database against nine credential
shapes, the scanned-vs-in-the-db denominator asserted, and a hit classified by whether a ref reaches
it — reachable fails, unreachable prints as local debt) with `tests/test_scan_secrets.py` as its gate,
`anti_prior_audit.py` (the P10 measurement without a model: both arms drawn through the *real*
batcher at the *published* dose, per-source weight sums against their row counts, the between-arm
mix column that is the only half belonging to the weights, and `--compare` over every `COMBINE`
rule) and its gate `mutation_anti_prior.py` (41 mutations; 28 over the trainer's drawer and
weights, 13 over what the harness *prints*, which is §9.48's point),
`bench_latency_matched.py` (the 4a/4b harness: one process, both engines, matched inputs, `flock`ed
output, and since 8a a `meta` that carries its own argv and the box's load average at the end of
the run) and its gate `mutation_latency_matched.py` (34 mutations), `risk_coverage.py` (the 5b
risk/coverage curve, its floor re-run and the G5 verdict) and its gate `mutation_p5.py`
(49 mutations across the engine's commitment, the seam's labels, the transport and the curve),
`eval_needle.py` (the 7a recall ladder — `--lengths` so the published ladder *is* the command that
ran it, and `baseline_verdict`, which refuses to let a curve that starts at the uniform floor be
read as decay) and its gate `mutation_longctx.py` (20 mutations),
`bench_mlx.py` (the 6c Apple harness: both engines over one checkpoint at real state/question
widths, artifact bytes read from disk, and the drift table with a name beside every maximum),
`diag_mlx_int8_gem.py` (the 6c mechanism probe: myna's own linear shapes, 200 matmuls queued per
sync, with the effective-GB/s column that tells the reader whether the loop measured a kernel or
Python), and their gate `mutation_mlx.py` (14 mutations over what gets quantised, how it dispatches,
and which byte figure the manifest quotes). `browser_g3.mjs` measures the artifact in real Chrome
(bytes, cold load, p50, screenshots) with `quantize_int8.py` producing its int8 row and
`mutation_browser.py` (24 mutations) as its gate.
`src/myna/onnx_export.py` is the G3 export path — two fixed-width graphs, the chained-scan
composition check, and a parity report — with its gate `bench/mutation_onnx.py` (22 mutations).
`kaggle/`: `run.py` (one-command entrypoint: corpus discovery, the measured flag set, `--resume` only
when a snapshot exists, tee to `train.log`, `run.json`, non-zero propagation), `campaign.py`
(the notebook cells: every path it emits checked against the box, every flag checked against
`run.py`'s `--help`, the GPU-hours priced off `runs/v1b_kaggle_3600b.train.log`, and an A/B pair
that differs in exactly one flag), `package_dataset.py`
(stage the pilot corpus, verify every byte against the corpus, print the upload command, never run
it), `requirements.txt`.
`tests/`: **561 passing + 1 skipped**, measured on this tick's tree — 562 collected by
`pytest --collect-only -q`, and the run's own summary line splits it 561/1: 292.95 s before the
§9.52 test edits and **289.00 s** after them, so the split survived the round-2 fixes and only the
seconds moved (the first run of this tree read 283.78 s). The re-read after the checkpoint-recovery and
GPU-denial commits (§9.53) is **258.29 s** over the same 561 passed / 1 skipped / 562 collected — that
batch added no test and removed none, so the count is confirmed rather than merely unchanged, which is
the difference 9k exists to record. The registry-row tick that follows it (§9.54) re-reads **261.60 s**
on the identical split, because it edited one test's coverage count and added none. The Tier 0
over-the-control tick (§9.55) re-reads **293.44 s** on the same 561 passed / 1 skipped / 562 collected,
its only test edit again a coverage count rather than a new test. The **+5**
over the 556 this map read earlier in the same tick are §9.51's arms, all five in
`test_reproduce.py` (the platform-limit table, the note path, the still-red-on-macOS path, the
missing-base-dependency path, and the guard that names which four rows the limit reaches) — counted
by re-collecting rather than by adding, and `--collect-only` piped through
`awk -F'::' '{print $1}' | sort | uniq -c` is now part of the re-read, because two of the three
drifts this paragraph has confessed to were invisible without a per-file baseline. The skip is
the KEV-gated parity test, green wherever `kev` is installed. **That count is a claim about this
box**, and §9.51 is what it cost to say it without the qualifier: `test_mlx_int8.py` (8) and
`test_mlx_parity.py` (3) need MLX, and off macOS MLX cannot even load — PyPI's Linux wheel ships the
bindings without `libmlx.so` — so both files skip there via `try / except ImportError` +
`pytest.skip(allow_module_level=True)`, which also catches the *unloadable* case that
`importorskip` re-raises. The four MLX registry rows say so as a printed note rather than failing
§9.43's `--help` assertion (`bench/reproduce.py`'s `mlx_unloadable_here`, pinned by 5 arms and
mutation-checked 4/4).
(A §3.4 map is present tense, so this
count is due whenever the suite moves; it read **466** until tick 10b, which added the P10 lane's
**42** — 35 in `test_anti_prior.py`, 3 in `test_kaggle_bundle.py`, 4 in `test_cli_help.py` — counted
by collecting the two trees and diffing the per-file totals rather than by subtracting remembered
numbers (§9.44's rule, and §9.46 is the tick that found the stale copy this sentence is the successor
to). It then went **48 stale** across the two ticks that followed and nobody re-read it: 3 are that
tick's `test_scan_secrets.py`; the other 45 arrived between tick 10b and it, and that sentence did not
pretend to know which files they came from because no per-file baseline had been recorded — the same
hole §9.44 is about, and the reason *that* number was a re-measurement (557 collected, 556 + 1 from
the run) rather than 508 + 48.
The 8a label pass added `test_train_logging.py` (4: `split_report`'s four
figures pinned, a lopsided split distinguishable from an even one at the same total, the empty
group set logging rather than raising) and one end-to-end `--no-laya` case in
`test_latency_matched.py` that reads the box's live `getloadavg()` against the
`meta.load_avg_after` the harness just wrote, so a stubbed zero triple fails on a working machine. The ONNX gate's 8 tests are inside that count (they skip unless the optional
`browser` extra is present), and their session fixture tears the runtime down deliberately: a live
onnxruntime pool can abort the interpreter *after* a green summary, which reads as a broken
baseline to any harness that trusts an exit code. The MLX int8 gate is `test_mlx_int8.py` (8 — the
quantised tree really holds packed uint32 and exactly the linear count, quantising one engine leaves
a second bit-identical, the artifact round-trips against the file's own byte count and *unequal* to
the tensor sum, the two refusals `mx.quantize` cannot make itself (`bits=6`, a group size that does
not divide the input), `--keep` honoured, and the drift bound sized under the smallest top-two
spread). The P5 gate is `test_abstain.py` (the flag against the distribution the same answer
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
**test 0.6319 · development 0.6392** — row-weighted over the 546 sampled answers, printed by the
harness itself. Per-source **test**, computed from that witness's own rows under the same
weighting: agnews 0.915 · trec 0.825 · dbpedia14 0.775 · mnli 0.675 · yelp 0.650 ·
boolq 0.575 · imdb 0.550 · contrastive 0.500 · sst5 0.425 · amazon 0.325 · banking77 0.250.

Three of the eleven cells that list used to carry were not per-source at all: they were
per-*question* rows read out of `runs/laya_witness_test.log` and lifted into a source list —
`agnews 0.950` is `agnews/topic`, one of that source's five questions (the source is 0.915);
`yelp 0.825` is `yelp/recommend`, one of two (0.650); and `contrastive 0.625` is the *noul*
row of a qid that the suite also asks as a choice, 16 of that source's 40 answers (0.500).
Only a source with one question was safe to quote that way, which is eight of eleven cells and
exactly the coverage that lets a bad list survive a proofread (§9.32).

Reproduce with the same seed and the same 40-per-source slice — it needs a laya checkout
on `PYTHONPATH`, which is why the row is `external-laya` in `bench/reproduce.py`:

```bash
uv run python bench/eval_laya_real.py --split test --n-per-source 40 --seed 0 --out runs/laya_decision_v2_test.json
```

The witness is that file, and the first line of its json is the command that wrote it.

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

Reproduce the two floors with the test that recomputes them out of the shipped rows and
asserts both literals — `uv run python -m myna.report` prints the same numbers for a
human, and this is the version that cannot be edited into agreement:

```bash
uv run pytest tests/test_report.py -q
```


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
- [x] Push the repo. **Closed 2026-09-30:** the tree is public at
      `github.com/aashish254/myna`, pilot corpus included. The counts *at that push* were 72 commits
      and 224 tracked files, from `git rev-list --count HEAD` and `git ls-files | wc -l`; re-read
      earlier on 2026-10-01, before the commit that adds `make secrets`, the same two commands printed
      **75** and **221** (four paid-path files removed since, one witness added). A figure that wants to
      survive has to carry its date, because a present-tense count in a prose line is the one kind of
      stale this repo has no check for (§9.44).
      Two things were checked on the way out rather than assumed. (1) The v0 weights went onto the repo
      as a **Release asset** rather than into git history, with the `risk_coverage` witness re-run
      against them to prove they are the published ones. (2) Every blob in history was scanned for
      credential shapes — and that scan is now a tool, `bench/scan_secrets.py` behind **`make
      secrets`**, because the first two passes were one-off commands and both were wrong.
      It walks `git cat-file --batch-all-objects`, not the refs, and then splits what it finds into the
      two things that are not the same fact: a hit in a blob **reachable from a ref** is in the set a
      push transfers, so it fails the gate; a hit in an **unreachable** blob is local debt on this
      machine — printed, counted, and not red, because making a clean tree permanently red is how a
      check gets muted. What is stable enough to state in prose: **no hit reachable from a ref**, for
      9 shapes (`KGAT_`, `gho_`, `ghp_`, `github_pat_`, a PEM private-key header, `AKIA`, Slack `xox*`,
      `sk-`, and a URL carrying `user:secret@`). The unreachable hits are this tool's own superseded
      drafts — the first version of its URL pattern, which matched that pattern's own source line, and
      the first version of its test fixture, which hard-coded the fake tokens instead of building them
      from pieces so the committed blob stays outside the shapes it tests. Both were staged, then
      replaced, so the old objects are in the database and in no ref. That is the value of the split: a
      refs-only scan would have called them nothing, and an object-database scan that could not tell
      them from a leak could never go green on its own tree.
      The blob and byte counts are deliberately absent from this sentence: committing the fix that made
      the scan possible added five blobs of its own, so a number written here was stale before the
      commit finished — `blobs scanned`, `blobs in the db` and `unreachable blobs` are the command's
      output, and §9.44's rule applies to a scanner the same way it applies to a table.
      The ninth shape had a second history worth keeping. Written loosely (`://` … `:` … `@` over any
      bytes) it hit **three** blobs, all of them `docs/screenshots/*.png`, because a compressed PNG
      byte stream happens to spell that sequence; each was read, confirmed to start with the `\x89PNG`
      magic, and re-checked with the same pattern restricted to printable ASCII, which returned
      nothing. So the pattern is printable-only — the real fix was the shape, not an allowlist that
      would have quietly swallowed the next hit. No credential-shaped string in this history has ever
      been a credential: the only `KGAT_` text committed is prose reading `KGAT_…`.
      What the old line said was "845 blobs, 0 matches", and 845 is not a number any command here
      reproduces. Re-running the same recipe this session printed "0 blobs scanned, 0 matches" — a
      mis-parsed `git rev-list --objects` field count that walked nothing and still came back green.
      So `bench/scan_secrets.py` prints `blobs scanned` and `blobs in the db` on separate lines,
      compares them, and exits 1 on a mismatch: **"0 matches" is only a result with the denominator
      printed beside it** (§9.46's class, aimed at a secret scanner this time). The tool is itself
      falsified — `tests/test_scan_secrets.py`'s three arms build a throwaway repo, require the clean
      arm to print its denominator, plant two fake credentials and require the hit by shape and path
      with a non-zero exit, then `--amend` them out of every ref and require the same blobs to be
      still found, named as local debt, and *not* to fail the gate.
      This line has now miscounted twice, both by remembering instead of reading: an earlier draft read
      "71 commits and 238 tracked files", copied from a recursive tree listing that counts directories
      and from a pre-push log.
      What the push did *not* fix is the opposite half of §9.47: the V1-B weights lived under
      `/private/tmp` and are gone, so no remote holds them either. *(This held for one day: §9.53
      recovered them from the notebook's Output download and they are on the `v1b-checkpoint`
      Release, hash-matched to the committed Tier 0 witness.)*
- [x] **CI has never run a test, and the reason is now a gate rather than a surprise.** Closed
      2026-10-01: `main` run **36785555798** (SHA `3ba18a3`) printed **545 passed, 8 skipped, 0
      failed in 599.72 s** and exit 0, with `make secrets` walking 222 of 222 blobs at 0 matches.
      All 7 runs the
      Actions tab records fail identically at collection — `ModuleNotFoundError: No module named
      'mlx'` in `tests/test_mlx_int8.py`, so **zero tests executed** while this box reported 556
      passing beside them. §9.51 is the full measurement, including the first fix (`--extra mlx` in
      the workflow) being falsified by the runner itself: PyPI's Linux wheel installs MLX's bindings
      without `libmlx.so`, so off macOS the package is present and unloadable at once, and
      `importorskip` re-raises that instead of skipping. What landed instead: the two MLX test files
      guard the import with `try / except ImportError` + a module-level skip, and
      `bench/reproduce.py` learned to tell a dependency this platform cannot load (a printed note on
      exactly 2 error strings, 4 rows) from a missing base dependency or a dropped flag (red, always).
      README's Quickstart names which rows each install choice leaves unread. **Round 2, same
      bullet, one tick later (§9.52):** the collection fix worked — run 36779801999 executed
      **519.31 s** of tests and printed **13 failed, 520 passed, 20 skipped**, which is the first
      witness this repo has ever had that a test ran. The 13 are four box assumptions a Linux clone
      falsified and nothing else could: onnx missing for the `int8-quantize` row's `--help` (5 tests,
      so CI installs `--extra browser` and the Quickstart now says the *test* install needs it), no
      `runs/myna-v0/model.pt` in a fresh clone because the weights travel as a Release (5 tests, so
      CI downloads `gh release download v0-checkpoint`, which also puts §5 P0's own release claim
      under test instead of under prose), two MLX bench harnesses whose usage text cannot render
      off macOS (2, marked `macos_only` after a CPU-only rehearsal that prints its own blocker),
      and one test that asserted `resolve_device("mps") == "mps"` — i.e. that this laptop has MPS
      (1, now monkeypatched so it reads the policy). **Round 3, measured the same day:** run
      36781802262 went green — `--extra browser` installed *and* loaded, `bench/quantize_int8.py
      --help` answered on Linux, `gh release download v0-checkpoint` filled `runs/myna-v0`, and the
      runner printed **545 passed, 8 skipped, 0 failed in 572.45 s** with the credential scan walking
      **222 of 222** blobs at **0 matches** — a PR run, so the bullet stayed unticked one more pass.
      **Closed on `main`:** run **36785555798**, SHA `3ba18a3b822b869eb47c63151550839fb58d782c`,
      **545 passed, 8 skipped in 599.72 s**, exit 0, scan 0 matches over 222 blobs. The pytest line
      carries `-rs`, so a future green must name each skip instead of just counting it — after 7 red
      runs the run is the witness, §9.30, and §9.52 is the entry for what the first real run found.
- [ ] Kaggle CLI auth is dead, and it gates the GPU lane rather than a row. Reproduced
      2026-10-01: `~/.kaggle/access_token` is present (38 B, `KGAT_…`, never printed here) and both
      credential paths fail identically — `kaggle quota` exits **1** on
      `Authentication required to call the Kaggle API.`, and the same call with the file's contents
      exported as `KAGGLE_API_TOKEN` exits **1** on the same line. The CLI is 2.2.4, whose `auth`
      subcommands are `login`, `print-access-token` and `revoke`; the repair is a browser OAuth
      `kaggle auth login` or a fresh token from kaggle.com → Settings → API, and **only the account
      holder can do either**. What it actually blocks: the registry's one `gated-kaggle` row
      (`g1-v1`) and every campaign cell that wants the T4 inside the 30 h/week quota. It does not
      block the 4 `retrain` rows — this box runs those at §9.50's wall price, which is why 10c is
      running here at all. The status tally behind those numbers is read from `ROWS`, not copied
      from prose: 20 `here`, 4 `retrain`, 3 `external-chrome`, 2 `external-laya`, 1 `gated-kaggle`.

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
      `bench/mutation_kaggle_bundle.py` → **25/25** (first pass 21/25; §9.15). That figure is the
      state of P3's list, not a re-runnable number: the harness could not reach the end of its own
      `the dry run runs the job` mutant until §9.40 capped what the catching test was willing to
      launch. Current: **48/48, exit 0** (42 inherited + P10's six), in
      `runs/mutation_kaggle_bundle.log`.
- [x] **3c** Python 3.11. `requires-python` is `>=3.11` (it was `>=3.13`, which made the box's own
      interpreter a *silent fallback*), every file under `src/ tests/ bench/ kaggle/` parses under
      `feature_version=(3, 11)`, and `bench/check_python311.py` runs the whole thing on a real 3.11
      interpreter: prints **`PASS: the package imports, compiles and trains on python 3.11`** on
      3.11.15 / torch 2.6.0, and exits 2 rather than skipping if no 3.11 is found.
      What 3c does **not** claim is the image's own version: the one box that ran printed 3.12 in
      session output that was never committed, so a version assertion here would be a remembered
      number, and `run.py --check` prints the interpreter on the box instead — which is what the
      cell `kaggle/campaign.py` generates now runs.
      Witness: `runs/python311_check.log`, whose first line is the command that wrote it.
      Reproduce (`MYNA_PY311` points at the interpreter; it refuses rather than reporting a
      silent skip, and the GPU half of the same check stays `KAGGLE`):

      ```bash
      uv run python bench/check_python311.py
      ```

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
      — **93 mutations over `src/myna/report.py`, all caught, exit 0**, run against a frozen repo
      (§9.18 is the process correction that bought that discipline; 54 of them are 3g's, P8 8c added
      the 13 that cover the competitor line and the §9.32 merge — which is also why the laya
      per-cell figures below moved by 0.008–0.009 — and §5 P9's 9b/9c added the 26 that cover the
      clip column, the Brier column and the table's geometry, §9.34).
- [ ] **3h** The default-config model (**15.35M** at vocab 4096; v0's artifact is **14.45M** at its
      1,740-entry tokenizer — §2.1 names both) against the ≤ 32M allowance, head-to-head on
      identical data — `KAGGLE`.
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
- [x] **6a** ONNX export of the two graphs, with torch-vs-ONNX parity measured rather than
      assumed (`src/myna/onnx_export.py`; `runs/onnx_parity.json`, `runs/onnx_export.log`).
      The split *is* the architecture: `state_step.onnx` takes a fixed 256-token chunk plus the
      fixed state stack and returns the next state — a 16k document is 64 calls to the same
      weights, never a graph that grows — and `question.onnx` resumes from that state, leaving
      the pointer head (a few matmuls over span means) to JavaScript beside the spans.
      Measured at chunk 256, 8 questions × 64 tokens, opset 18, `runs/myna-v0` (14.45M), on the
      shared-weight artifact 6b ships: chained scan of 853 tokens over 4 chunks with a padded tail
      **|Δ| 7.32e-04 = 1.63e-06 relative** on a state of scale 449; question branch **2.83e-05
      absolute = 4.58e-06 relative**; masked-row inertness **exactly 0.0**; and the number a caller
      acts on — option probabilities through the pointer head — **8.94e-08**. (These are the numbers
      `runs/onnx_parity.json` and `runs/onnx_export.log` print today; 6a's first pass recorded the
      same properties on the duplicated-trunk graphs at 3.13e-06 / 1.88e-05 / 1.86e-07, and the file
      was re-derived after sharing rather than left as a memory. The per-chunk wall-clock figure the
      run also prints is quoted nowhere, because §9.23 established that this box's absolutes move by
      3× under load and G3's cost claim is measured in Chrome.)
      That last figure is why the
      gate is split in two units: the state's entries reach ~4.5e2, so asking 1e-4 of it
      absolutely demands bitwise agreement between two BLAS implementations, and G3 would read
      as an unbuildable target instead of a checked one.
      Three findings the export forced, none of them in the original plan:
      * **`dynamo=True` is required.** The legacy TorchScript exporter emits graphs this trunk
        needs that `onnxruntime` refuses to load (`/fwd/Mul_2 … Incompatible dimensions`), so
        the "successful" export would have failed on the reader's machine instead of ours.
      * **`opset 17` was a label, not a fact.** torch warns that 17 is below its implementations,
        writes 18, and its version-converter fallback aborts; `meta.json` now records the opset
        read back out of the file (`opset_actual`) rather than the one requested.
      * **Byte budget: 93.7 MiB fp32, and it is two copies of the same trunk** (state_step 35.6
        + question 58.13). §2.1's ≤ 20 MB int8 target is not reachable by quantising both graphs
        independently — the weights have to be shared, or the browser has to carry one graph and
        call it twice. That is 6b's problem, now measured instead of assumed. On the narrower
        profile 6b actually ships the same comparison is 93.2 MiB unshared against **59.32 MiB**
        shared, and that is the pair §2.1 quotes; 93.7 belongs to 6a's own 8 × 256-token export.
      What 6a left open, and 6b closed: the widest request the suite contains is `banking77/intent`
      at **1,067 tokens** (77 options), measured over `calibration.jsonl`, which does not fit the
      64-token question width this profile exports — `runs/onnx_parity.json` records it as
      `"measured": false, "graph_width": 64`. The same request through a graph wide enough to carry
      it (width 1152, `runs/onnx_parity_widest.json`) is measured at **4.25e-07** on option
      probabilities, so real-width parity is a profile choice with a price in peak tile bytes
      (3.00 → 12.0 MiB) rather than an unproven claim.
- [x] **6b** Measured in Chrome via `onnxruntime-web`: download bytes fp32/int8, cold-load ms, p50
      per decision on a real page, with the two things 6a proved necessary both closed.
      * **The duplicated trunk is gone.** `onnx_export.share_weights` rewrites both graphs to read
        their initializers from one `weights.bin`, deduplicated by the SHA-256 of each tensor's
        bytes — sound because initializers are constants that no node writes. Sharing is then
        *counted by content region, not name*: `dynamo` numbers initializers `val_N` per graph, and
        on this artifact **36 initializer names are common to the two graphs while only 33
        byte-regions actually are** — a name intersection measures the exporter's counter, not the
        sharing. `min_bytes = 4096` keeps shape-inference constants inline, because onnxruntime
        refuses to read them from an external file. Artifact on the shipped profile:
        **93.2 → 59.32 MiB**, of which `weights.bin` is 54.32 MiB transferred once.
      * **Parity at the widest real request.** The `banking77/intent` question 6a left as
        `"measured": false` — 1,067 tokens, 77 options — is now exported at the width it needs
        (graph width 1152) and measured: decision probabilities **4.25e-07** absolute, chain
        **4.25e-06** relative on a state of scale 287, `runs/onnx_parity_widest.json` with
        `"measured": true`. `--scan-chunk` stays the tab's memory knob: 3.00 MiB peak tile at the
        shipped profile, 12.0 MiB at the wide one.
      * **The page is a second implementation, and it is checked as one.** `browser/bpe.js` rebuilds
        the ByteLevel BPE from `tokenizer.json` alone (separators deleted rather than turned into
        `Ġ`, byte→symbol identity for the printable ranges, no `max_word_length` cutoff because this
        tokenizer writes none and a 260-character word merges normally); `browser/head.js` runs the
        pointer head and the abstention gate out of `head.bin`; `browser/myna.js` chains the scan.
        `browser/parity.mjs` is the single witness both `node browser/selftest.mjs` (7 checks) and
        the tab (6 — the on-disk byte comparison needs a filesystem) import, so a green tick in
        Chrome is produced by the code the build gate fails on. Node: 1,243 tokens across 32 strings
        and 3 question layouts identical, chained state over 525 tokens in 3 calls within
        **2.86e-05** relative, decisions within **1.03e-06**, abstention reproduced at three floors,
        every fetched file byte-equal to the file on disk.
      * **Chrome, cold cache, throwaway profile** (`bench/browser_g3.mjs`, CDP over Node's own
        WebSocket, `Network.loadingFinished.encodedDataLength` as the byte witness):
        **Google Chrome 153.0.8010.53**, onnxruntime-web 1.30.0, four rows — COOP/COEP-isolated and
        plain, 1440×900 and 390×844, 15 decisions each. Artifact **59.32 MiB**, wire **73.09 MiB**
        with the **13.58 MiB** wasm runtime counted separately and never folded into the model's
        figure, `weights.bin` requested **exactly once** in all four rows, cold load 254–394 ms,
        p50 **436/426 ms** on 10 threads against **742/743 ms** single-thread. Screenshots:
        `docs/screenshots/{isolated,plain}-{1440px,390px}.png`; witness `runs/browser_g3.json`,
        which carries `uptime`'s load average (5.30 at the time of the run; the int8 sweep beside it
        ran at 7.58, and each file records its own) next to every millisecond in it, per §9.23.
      * **int8 was attempted, and it is a byte win only** — see §9.28.
      * **Gates.** `pytest` 295 passed / 1 skipped (the KEV-gated parity test),
        `node browser/selftest.mjs` 7/7 (`runs/browser_selftest.log`), `bench/mutation_onnx.py`
        **22/22**, and `bench/mutation_browser.py` **24/24** (`runs/mutation_browser.log`) across
        five clusters — tokenizer, chain, head, abstention, mount + bytes. Its first pass left
        three survivors, two of which were the instrument's blind spot rather than the code's;
        §9.27 records all three and the coverage assertion that keeps the second kind from
        recurring.
      Still open, and stated rather than smoothed: the tab's own `measureUserAgentSpecificMemory`
      reading (176 MiB) counts JS-managed heap only and does **not** attribute the 54 MiB mounted
      into the wasm heap, so no total-tab memory figure is claimed here; plain mode has no memory
      API at all, which is printed as `n/a (API absent in this browser)` rather than as zero; and
      every probability above is v0's, so a green G3 is a working artifact, not a shippable model.
- [x] **6c** MLX int8 path for Apple, re-measured after any parameter growth — the path is
      built, and the measured verdict is **bytes yes, milliseconds no**:
      * **The artifact.** `MynaMLX.quantized_()` runs `mx.quantize` over every 2-D weight the
        engine *multiplies by* — `trunk.tok.weight` stays fp32 because it is gathered, not
        matmulled — `_linear` dispatches on the presence of `weight_q` to `mx.quantized_matmul`,
        and `save()` writes a self-describing directory (`params.safetensors` + a `meta.json`
        carrying `{group_size, bits, keep, n_quantized}` and both byte figures). `from_mlx_dir`
        refuses a `bits` outside {4, 8} — the only two `mx.quantized_matmul` implements — naming
        the file and the value it read, so a hand-edited manifest fails at load instead of
        answering with the wrong dequantisation. 50 of 51 weights quantised at group 64 / bits 8.
      * **Bytes: met.** `params.safetensors` goes **55.13 → 17.40 MiB** (57,805,835 →
        18,248,977 B), **3.17×** smaller in the same container, and the reloaded artifact is
        bit-identical to the engine that wrote it (`max |d| 0.0e+00` on logits, printed by the
        bench and asserted by the test).
      * **Latency: not moved.** MLX over torch-MPS on the same checkpoint and the same box,
        per-row *minimum* across two committed runs (`runs/bench_mlx_int8.md`,
        `runs/bench_mlx_int8_run2.md`; load average 4.55 and 3.94 at print, §9.23):
        fp32 **1.76–2.22×**, int8 **1.79–2.41×**. int8 − fp32 changes sign between the two runs on
        4 of the 7 state lengths (+2.6% → −9.3% at 4096, −1.6% → +1.9% at 512) while fp32's own
        run-to-run spread reaches **15.2%** on that 4096 row, so no "int8 is faster" claim survives
        the two files and none is made.
      * **Why, measured rather than argued.** `bench/diag_mlx_int8_gem.py` times the model's own
        six linear shapes with 200 matmuls queued and one sync (holding every node: evaluating
        only the last array leaves the other 199 unreferenced and unrun, and the broken probe read
        0.4–3.1 µs/op with int8 up to 1.31× ahead where the fixed one reads 3.3–28.7 µs/op).
        **5 of 6 shapes are slower quantised** —
        ratios fp32/int8 **0.43–1.06** — while reading 3.6× fewer weight bytes. The kernel gets
        **10–37 GB/s** of weight read against fp32's **55–257 GB/s** on the same box at the same
        load, so the unpack costs more than the smaller read saves. Weight-only int8 removes bytes
        from the weight traffic and no MACs; at 16–128 rows per call these GEMMs are not waiting
        on the bytes it removes.
      * **Agreement cost.** 14 `calibration.jsonl` rows / 20 questions answered at their own state
        and question widths (no padding to a fixed shape): port error (fp32 vs torch-MPS)
        **1.20e-03**, quantisation error (int8 vs fp32, same device and same stream) **9.24e-03**,
        int8 vs torch **9.48e-03**, and **1 flip in 20 choices**. The flipped row is a 2-option
        boolq question at 283 tokens whose reference top-two margin is **0.0008** against torch and
        **0.0001** against fp32 — quantisation moved a coin, which is exactly what a
        probability-error bound alone cannot see.
      * **One null result, stated.** The gate projections were the obvious suspect (their output
        feeds a sigmoid carried multiplicatively across the whole state). Keeping `.gate.weight`
        in fp32 quantises 38/51 weights instead of 50, takes the artifact back up to 22.25 MiB —
        outside the byte target — and moves the drift from **9.24e-03** to **9.53e-03** while
        flipping **the same row** (`runs/bench_mlx_int8_keepgate.md`). The gate weights are not
        the carriers, so there is no cheap subset that buys the agreement back.
      * **Consequence for the gates: none of them ships on int8.** G3's parity bound is 1e-4;
        MLX int8 sits at 9.24e-03, ~92× over — about 7× tighter than the wasm int8 of §9.28 — and
        refused for a different reason: wasm int8 lost on agreement, Apple int8 buys nothing at all.
        fp32 is the artifact on both paths.
      * **Gate.** `tests/test_mlx_int8.py` (8 — the quantised tree really holds packed uint32 and
        exactly `_n_linear` tensors; quantising one engine leaves another bit-identical; the
        artifact round-trips bit-exact with `meta["param_bytes_file"]` equal to the file's own
        byte count and *unequal* to the tensor sum; `bits=6` refused; group 64 on a 96-wide input
        refused; `--keep` honoured; and the closeness bound `max|dp| < 2e-2` sitting under the
        smallest top-two spread on the tiny model) plus `bench/mutation_mlx.py` **14/14**
        (`runs/mutation_mlx.log`) across five clusters — filter, dispatch (`transpose`, and
        `group_size`/`bits` read from the artifact rather than hard-coded), the divisibility
        guard, the manifest and its loader, and the two byte figures swapped in both directions.
      One label was fixed on the way: `bench_mlx.py`'s markdown header said "token embedding kept
      fp32" in prose while `--keep` was a real list, so the first run that moved the flag would
      have printed the old default — the §9.26 class of defect, now printing what it ran with.
      **Re-measure trigger stands:** the ratios above are v0's 14.4M-parameter artifact, and §2.1
      carries them as such.

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
- [x] README + PLAN headline tables from committed artifacts only, each cell labelled
      measured/projected/gated. Every headline cell now carries its own ***m***/***p***/***g*** tag
      and the file it was copied from, and eight cells that had no artifact behind them died in the
      pass — §9.30 lists each with the arithmetic that buried it, and §4.2's floors, §2.1's two
      parameter counts and the fitted-vs-read distinction are stated so a derived number can never
      again stand in for a measured one.
- [ ] Reproduction script per table.
      The registry side is done — `bench/reproduce.py` binds every published row to one canonical
      command and one committed witness, `make repro` checks all of them, and the counts and the
      one thing still open (a target that *runs* them, which 9 of the 27 rows cannot do on
      this box) are in TODO 8b. §9.31 is what the mutation battery caught on the way. TODO 8e
      added the fourth assertion — `--check` asks the tool behind each python row for `--help` and
      requires every published flag to be in the answer, which is the half of "a target that runs
      them" that is decidable here — and §9.43 is its measured negative: 20 of 27 rows are python
      commands, all 20 accept their flags, and 7 rows say what kind of command they are instead.
      TODO 8f closed the seam 8e left open: the registry's counts are counted out of `ROWS` in
      `bench/gates.py`, and `make gates` reads them back out of §2.2 — the table's one hand-copied
      row — so a stale number there reds instead of riding along (§9.44).
- [x] **The upstream-corpus comparison, published as three rows** — the §8 risk line's mandate.
      README's *On the upstream corpus* section prints laya **0.667**, myna's untrained control
      **0.346 / 0.351 (mean 0.348)**, and myna-trained as **gated on its witness rather than on
      a run**, over the same 16
      (source, question) cells with the 0.433 and 0.332 floors beside them, and discloses that
      the one local checkpoint that ever saw the corpus scores **0.338** — below its own control
      and void under §5 P1 — which is the sentence that settles what has and has not been
      measured about this architecture. `bench/eval_scratch.py` is the control harness
      (11 tests, 13 mutations of it all caught by `bench/mutation_scratch.py`); `--metrics-out`
      is what lets `myna.report` read the control through the same code path a checkpoint comes
      in by, so the two witnesses have to agree to the printed digit — and §9.32 is the
      competitor-side bug that join found on the way.
- [ ] Public release gate: P0 push + G1–G7 all pass or all explicitly marked not met.
      The verdict half is done — P8 8d — and it says **no**. `bench/gates.py` binds each of
      G1–G7 to a registry row *and* to a named field read out of a committed artifact, so no
      verdict can be greener than the witness behind it: **3 met** (G2 on the matched ratios,
      G3 on the in-Chrome parity, G7 on the checker itself), **3 not met** (G1, on the trained
      run's own committed report rather than on the untrained control that used to be the only
      architecture number here; G4, whose own harness prints `readable: false`; G5, whose curve
      prints `g5.pass: false`), **1 open** (G6 — the level clears at 0.9523, but the gate is
      written against a trained v1 checkpoint and there is nothing to regress against until
      that run exists). `make gates` is the
      check and SPEC §2.2 is the prose it reads. What keeps this line unticked is the release
      itself: G1 is measured and short by 0.211, raising it is a GPU decision whose price §9.38
      writes down, G4 still needs Kaggle, and P0's push is user-gated with no remote yet
      created.

### P9 — The three defects the first valid checkpoint named *(9a shipped; 9b's fix measured dead; 9c local; 9d the user's call, now priced (§9.45); 9e measured dead too; 9f–9j are TODO's ticks: G1 is now `not met` on a committed witness, one mutation battery had never finished, the last unexplained V1-B number resolved into an aggregation, the §9.40 sweep across all 14 batteries found no second unaffordable probe, and the 15th battery — 9j's, for 9d's own scope table — found its tests could not see `main()` (§9.46)*

Opened by the diagnostic run against `v1b-kaggle-3600b`'s own logits (TODO 3i's lane), which answers
the capacity question in the direction that matters: **the mechanism is not broken, the cost
function is.** Option-order permutation tracking shows the pointer head reads option *text* — under
permutation a cell's accuracy barely moves — so the trunk can find the answer and the training
signal is what misprices it. **The permutation magnitudes belong to 3i's diagnostic and are
deliberately not printed here.** The checkpoint itself now has a committed witness — its
`metrics.json`, its roll-up and its `train.log` are in `runs/`, which is what moved G1 from open
to measured — but the option-order-permutation numbers below came from a diagnostic harness whose
output was never committed, so §9.30 still bars them from prose. What is structural, and
what this repo can act on without the artifact:

- [x] **9a** `score` questions are trained with a **nominal** cost. `typed_loss` is
      `F.log_softmax(...).gather(gold)`, so over an ordered legend "predicted 4, gold 3" is charged
      the same as "predicted 1, gold 5" — the first is one rung of a scale and the second is the
      opposite end of it. Every other part of the stack already believes the legend is ordered:
      `data.py` documents `options` as "for score this is the ordinal legend", and both the engine
      and the MLX bench read a `score` as the *index-weighted expectation* over it. The loss is the
      one component that does not, which makes this an internal inconsistency before it is a
      research idea — and it is the only one of the three defects that a training run can fix
      without new data or new parameters. Fix: a squared-Cramér / EMD² path over the option index
      (`Σ_k (F_p(k) − F_gold(k))²`, normalised by `K−1` so a two-option cell and a six-option cell
      are on one scale), selected by `--score-loss {ce,emd}` with **`ce` the default**, so no
      published number moves and the next Kaggle run is the experiment. Note the property that
      makes the normalisation right rather than cosmetic: at `K = 2` this expression *is* the Brier
      score of the "Yes" probability, which is the statistic §9.24 already insists on quoting
      beside a noul argmax. `KAGGLE` for the verdict, `here` for the code and its gate.
      **Shipped.** `typed_loss(logits, gold, has_gold, ordinal, opt_valid, score_loss)` prices the
      cells `ordinal` marks with the Cramér sum divided by the cell's own `K−1` and every other cell
      with the old cross-entropy, in **one mean over live cells** — not a mean of per-cell means,
      which a rectangular batch holding a padded question row would otherwise turn into a per-row
      weighting. `build_batch` and `build_row_batch` emit the mask from `q.type == "score"`, the
      loop's single call site passes flag and masks, and `--score-loss` defaults to `ce`, so the
      published figures are untouched. Three properties are the gate rather than the formula: a
      one-rung miss and an opposite-end miss with the *same* `p(gold)` are byte-identical under `ce`
      (that is the defect, pinned in a test) and strictly ordered under `emd`; the worst reachable
      price is 1.0 for every legend length; and at `K = 2` the number *is* the Brier score of the
      second option — for noul that is "yes", which is §9.24's statistic arriving from the loss
      rather than from a report. `tests/test_ordinal_loss.py`: 18 tests, three of which run the
      trainer for one CPU update through a `typed_loss` spy, because the trainer seeds no torch and
      two processes therefore never share an init — "the printed loss changed" would pass even if
      the flag did nothing, so the call site is watched instead of its output. The absent flag is
      one of those three: it must mean `ce`. `bench/mutation_ordinal.py` **19/19 caught**
      (`runs/mutation_ordinal.log`). Two findings from building it: the sum is multiplied by
      `opt_valid` because a masked tail is only *harmless by coincidence* — hand it the builders'
      real `opt_valid` beside unmasked logits and an unmasked sum prices a cell's padding — and the
      `[N] → [B,N]` expand that was written first is **gone**, because torch's own broadcasting makes
      the shared and per-row forms the same tensor, no input could tell them apart, and that
      mutation survived its own battery (§9.33). The level is 3i's: nothing published moves, because
      nothing published was trained with `emd`.
- [x] **9b** *Premise corrected by measurement; the readout fix is closed without being built.* This
      box used to say that `noul` cells are scored by an argmax that collapses onto one index while
      the continuous probability ranks, and that the fix is a threshold or a label-prior bias in the
      readout. The follow-up diagnostic refutes both halves on most of those cells (its artifact is
      a Kaggle-side file, so §9.30 keeps its numbers out of this prose): the argmaxes are not
      pinned to one index, and **fitting a per-cell decision threshold on the suite's own
      `calibration` split makes the binary cells worse, not better** — 40–52 calibration rows per
      cell cannot estimate a threshold, and that split's class balance is not dev/test's. One cell
      formerly cited as a readout casualty turns out to be *anti-ranked* instead, which is the
      failure mode 9d calls dead rather than mis-ruled. What survives of 9b is the reporting half,
      and it is still worth building: a Brier column beside every `noul` accuracy in `myna.report`,
      because §9.24's point is that an argmax alone over-reports a binary cell either way — it just
      is not a *fix*, and it must not be listed as one. Nothing here raises G1's ceiling.
      **Built (the surviving half only).** `myna.report` reads `evaluate()`'s
      `"<set>/<question>:brier"` sidecars through `brier_cells()`, weights them by the rows in each
      question-set for the same reason `accuracy_cells` does, and prints the result in a `brier`
      column beside every binary cell's accuracy, with the group's macro under it. The three things
      this column can lie about are the three it is tested on: a metrics file that predates the
      sidecars prints an **em dash** and the report says "the column is empty because this metrics
      file has none, which is missing data and not a model that scores 0"; a sidecar whose
      question-set is unknown to the split contributes **no weight** rather than a zero; and the
      count on the line is `priced of noul`, never `noul of noul`. The box is ticked because the
      surviving half shipped and the dead half is recorded as dead — the readout applies no
      per-cell bias and no fitted threshold anywhere in `src/myna/`, and §9.24's negative result
      stands. Where Briers exist on this box at all, they are not quotable: the diagnostic's
      artifact is Kaggle-side and outside the repo, and the only local metrics file carrying
      `:brier` keys is the void pre-`385e06c` checkpoint's (§9.23), which is itself uncommitted — so
      `runs/report_void_vs_laya.log` does print a Brier column, and the registry note for that row
      says plainly that neither its clip nor its Brier figure enters the prose (§9.30). The
      committed control is the clean case, and it is the em-dash path: `bench/eval_scratch.py`
      strips `:brier`/`:ece` on purpose — a Brier column off a random init is noise with a decimal
      point on it — so `runs/report_scratch_vs_laya.json` reports `"priced_cells": 0` of its 8 noul
      cells and the report says the column is empty because the file has none.
- [x] **9c** Some cells score **below their own majority-label floor**. The first diagnostic named
      four; the follow-up that corrected 9b leaves two of them standing as floor-losers and moves the
      rest into 9d's dead-cell question, which is the right split — a cell below its floor *and*
      below its own permutation null is not a decision-rule casualty. This is a reporting failure
      before it is a modelling one: a cell that loses
      to the majority answer is a cell where the model should not be answering, and G5's abstention
      machinery already exists to say so. 9c is the `myna.report` change that surfaces it —
      per-cell `max(acc, majority)` printed next to `acc`, and the delta labelled as the clip's
      worth rather than the model's. Local, and its arithmetic is checkable against the committed
      pilot split today. The boundary has to be drawn in the same sentence: *printing* the clip is
      reporting, and it is worth exactly the arithmetic it shows; *acting* on it — predicting the
      training-set majority whenever a cell's dev accuracy sits under its own floor — is a per-cell
      fallback rule, which is a leaderboard decision and not a learning improvement, and 9b is the
      evidence of what a readout rule fitted on this suite's calibration data measured. 9c builds
      the report and stays out of the decision.
      **Built, and it stayed out of the decision.** `roll_up` carries `clip = max(acc, majority)`
      and `clip_worth = max(0, majority − acc)` beside the model's own number, the legend defines
      the column on the line above the table where it prints ("It is an oracle — the floor is
      measured on the rows being scored"), and `main()` names every floor-loser with its worth and
      then says what the difference is: "+0.090 of arithmetic and 0 of model … a leaderboard
      decision, not a learning improvement". Measured on what this box can measure without
      training — the untrained control over the committed pilot split — **13 of its 16 cells answer
      below their own majority floor**, and the clip macro is **0.4365** where the model is
      **0.3462** and the floor **0.4331**. That last pair is the reason `clip` is its own column
      and not the floor renamed: three cells (both `yelp` rows, `mnli/relation`) score above their
      floor and keep their own number, so a per-cell maximum beats the average of the floors while
      thirteen cells are losing. Those 13 are the *control's* — the two trained floor-losers the
      diagnostics named are a Kaggle-side list and stay 9d's. `--out` carries `macro.clip`,
      `macro.clip_worth`, `macro.brier_noul` and a `below_majority_floor` array naming each losing
      cell, so the sentence has fields behind it (§9.30). Nothing in `src/myna/` reads the clip: no
      fallback consults `clip_worth`, and the per-cell abstention rule this column makes tempting
      is the user's call, listed as such under 9d.
- [ ] **9d** Two cells are dead and no decision rule fixes them (`banking77/intent`,
      `mnli/relation`): their association numbers do not beat their own permutation nulls. 9d is a
      scope decision for G1 — in or out — not a fix, and it is the user's call, not a checkbox to
      tick by shipping something. Recorded so the next run does not silently average them back in.
      **Priced, so the call is not blind** (`uv run python -m bench.scope_pricing`, witness
      `runs/scope_pricing.json`, §9.45): dropping the two named cells moves the level 0.4893 →
      0.5334 and the majority floor 0.4331 → 0.4666 with it, so the margin G1 gates on moves
      +0.0563 → +0.0668 — eleven-thousandths, not forty-four. Dropping their whole sources is the
      same 14 cells (each contributes exactly one), and dropping *every* cell that answers below
      its own floor — 5 of 16, a cherry-pick no release would ship — reaches 0.5526 at +0.0949,
      still 0.147 short of the target. **No scope choice in the table makes G1 pass**, so 9d is a
      decision about what the published claim means, not a route to clearing the bar, and it should
      not be argued for with GPU hours on the expectation that the run would then pass.
- [x] **9e** The ablation 9a opened for, settled on this box for 0 GPU-hours — and the two
      reproducibility defects that had to be fixed before it could be settled. `--score-loss` was
      one flag and one output directory away from a *pair* of runs, which is the only reason 9a
      was closable without the account: four CPU arms (`ce`/`emd` × seed 0/1, 250 updates) starting
      from one shared init. Getting there required two fixes, because the arms were not comparable
      as written. (i) `--seed` named the data order and nothing else — model init came from whatever
      torch's global RNG held, so two runs of one command differed by init noise. That is why 9a's
      own loss tests push a single update through a `typed_loss` spy *inside one process*: across
      processes the init differed for a reason nobody was testing. (ii) The seed was also resizing
      the step, because `state_token_p95` drew its sample with `random.Random(args.seed)`, so the
      experiment's seed decided the batch size (measured, on the full pilot: p95 267/batch 15 and
      p95 274/batch 14 from one command at two seeds). Fixed both ways it should be:
      `torch.manual_seed(args.seed)` before the model is built, and the p95 taken from the corpus
      rather than from the run — which moves no published figure, because every run this repo has
      published used seed 0, where the two are the same draw. `--warm-start` is a third addition and
      it is not a synonym for `--resume`: resume restores the saved `CosineAnnealingLR` state
      including `T_max` *and* AdamW's `param_groups` including `lr`, so continuing a finished cosine
      schedule trains at LR ≈ 0 while printing a perfectly plausible curve. The load-bearing test is
      in `tests/test_memory_plan.py`: two cold runs of one command must be bit-identical, *and* a
      warm start must land on different weights, because a warm start that lands on identical
      weights loaded nothing.
      **The verdict: 9a's term does not pay at this dose.** `bench/ordinal_ab.py` judges the arms
      from their own `cmd:` lines — a pair may differ only in `--score-loss`, `--seed`, `--out`, and
      it asserts the two arms of a pair share batch, p95 and `last_step`, and that neither stopped
      early — and `runs/ordinal_ab.json` carries the per-cell rollups, the pairs, and the **churn
      floor**: the mean |delta| across the 13 cells `emd` cannot price, which move anyway because
      the trunk is shared. A score-cell delta under that floor is not an effect. Per seed, dev:
      score-cell mean **+0.008333 / +0.004167** against floors of **0.013982 / 0.010086** (ratio
      **0.596 / 0.4131**). Per seed, test: **+0.020833 / 0.0** against **0.011843 / 0.011925**
      (**1.7591 / 0.0**). The headline moves *opposite ways in the two seeds* on both splits (dev
      **−0.011828** vs **+0.003155**, test **+0.007952** vs **−0.001408**), and sign agreement
      across seeds on the three priced cells is **2/3** on dev, **1/3** on test. One arm clears
      its floor and its own replicate is exactly zero, so nothing here separates the term from
      trunk noise. This also corrects how the same data was first read: the
      ratios quoted then (**0.52 dev / 0.88 test**) pooled both seeds into a single mean, and
      pooling is what made *every* ratio look below 1 — an average of a positive draw and a zero one
      is a statement about neither, and §9.36 keeps the lesson. What the dose cannot say, do not
      borrow it for: it does not price `emd` at 3,600 updates, and the init every arm warm-started
      from (`v1b-kaggle-3600b`) has its *figures* committed (`runs/v1b_kaggle_3600b.metrics.json`,
      so its level is quotable — 0.4893 test macro, §2.1) but not its *weights* (`model.pt` is a
      65 MB artifact `runs/` does not track), so no local arm can be rebuilt from the same starting
      point. These stay *relative* movements from that point rather than levels (§9.30's rule,
      applied to ourselves). `ce` stays the
      default in `kaggle/run.py`'s `DEFAULTS`, which is what keeps every published number standing. Re-judge the four arms with
      `python bench/ordinal_ab.py --ab-dir /tmp/ab2`.

### P10 — The prior the rows pay for a constant answer *(10a diagnosed from Tier 0, 10b built, audited and mutation-checked 41/41 (§9.48); 10c is the GPU pair, priced and unrun)*

P9 named three defects from accuracies, and an accuracy cannot say *which input* the answer came
from. Tier 0 (§9.47) ablated the inputs on the committed V1-B checkpoint and eight of decision-v2's
sixteen cells turned out not to read the state at all — swapping it moves their accuracy by ≤0.001 —
and three of those eight are agnews `noul` cells that emit one label on *every* test row and finish
exactly on their own majority floor (0.6531 / 0.8085 / 0.7778). The cheapest mechanistic reason the
training data offers for a constant answer is the label prior, and unlike 9d's scope call that part
is measurable **without a model**:

- [x] **10a** The shortcut is the best-scoring policy the rows support. agnews' train marginal is
      **0.7390–0.7545** inside its four `noul` cells, so a cell that answers one label always is
      worth ~0.75 — more than anything a constant predictor can be worth against a flat prior, and
      the same order as the 0.739–0.755 the cells finished on. `--anti-prior` (10b) flattens the
      marginal the mini-batch is drawn from, which makes the constant answer worth ~0.5. **What
      this is not:** a claim that the model will then read. It removes a shortcut; it does not add
      an ability, and if removing the shortcut does not move the macro, the reading is that the
      association was never the training signal's fault. The floor the report judges cells against
      is the untouched *test* split's, and this lever does not move it — see 10b's limits.
- [x] **10b** `--anti-prior on|off` in `myna.train`, default **off**. Each row is weighted by the
      product, over the questions it answers, of `1 / (share of its gold label inside its
      (source, question) cell)`, and the weights are rescaled so every source's sum equals its row
      count — the task mix cannot move, only the answer histogram inside each cell can. Only cells
      whose natural majority reaches `ANTI_PRIOR_SKEW = 0.55` are treated: six of sixteen.
      `bench/anti_prior_audit.py` prices all of it on the shipped rows, through the *real* batcher
      (`myna.train.draw_row_batch`) at the *published* dose — batch 10, 2,048 question cells per
      forward, 8 sets per update, 3,600 updates, seed 0, both arms in one process:

      ```bash
      uv run python bench/anti_prior_audit.py --compare --out runs/anti_prior_audit.json
      ```

      `runs/anti_prior_audit.log` is the witness. The six treated cells flatten as designed —
      is_business 0.7545 → **0.6184**, is_sports 0.7483 → **0.6278**, is_scitech 0.7467 → **0.6247**,
      is_world 0.7390 → **0.6172**, boolq/answer 0.6243 → **0.5023**, yelp/recommend 0.6055 →
      **0.5095** — while the control arm reproduces the natural marginals (largest move +0.0077).
      The two guards that make the rest of the table mean something: per-source weight sums equal
      their row counts to **worst relative deviation 0.00e+00**, and the arms' drawn source shares
      differ by at most **0.13 points**. That last column is the one that belongs to the weights;
      the gap between a source's *row* share (9.15 pts) and its drawn share in *both* arms —
      banking77 sits at 3.97 either way, **−5.18** — belongs to `--max-q-cells` skipping rows whose
      question branch would blow the budget, and is why reading a single arm's mix as an effect
      would have been a bug. `--compare` draws every rule in `COMBINE` and `prod` is the minimum in
      all six columns, which is the measured reason it is the shipped rule; mean/max/geo/sum stay in
      `COMBINE` only so the table stays reproducible. **The trap this tick found:** at a reduced
      dose (40 updates × 2 sets) `prod` is the minimum in **1 of 6** columns and `mean` reads 0.8947
      on a cell whose prior is 0.7390 — a ranking claimed from a short run is a claim about the
      seed, so the ranking test reads the committed witness, not a cheap run.
      Two limits the artifact states rather than hides. (i) **Nothing here is an accuracy claim**:
      flattening the train marginal does not flatten the test marginal, and the floors cells are
      judged against come from the test split. (ii) **Row-level weighting cannot hold a cell fixed
      when its rows carry several labels.** Of the ten cells the skew threshold does not target,
      nine flatten anyway — contrastive/decision 0.507 → 0.337, down to its own 1/3 uniform, −0.163
      — and **one sharpens**: yelp/rating 0.203 → 0.259, +0.055, because a yelp row answers `rating`
      and `recommend` at once and these weights are computed on the other question. And the reach
      over Tier 0's eight is three: the other five (amazon/stars, banking77/intent,
      contrastive/decision, mnli/relation, sst5/sentiment) are already flatter than 0.55 in their own
      train marginal, so an inverse-prior re-weighting has almost nothing to remove for them. If the
      shortcut there is a label prior, it is not this split's majority.
      `bench/mutation_anti_prior.py` holds the harness's printed face: **41 mutants, 41 caught, exit
      0** (`runs/mutation_anti_prior.log`), and §9.48 is what that pass taught — including that
      `kaggle/run.py` had never plumbed the flag, so the whole lane below was not executable from
      the staged bundle. That gap is now mutation-checked too:
      `bench/mutation_kaggle_bundle.py` went 42 → **48, 48 caught, exit 0**
      (`runs/mutation_kaggle_bundle.log`), and §9.48's last paragraph records the one that went green
      until its test was fixed.
- [ ] **10c** The GPU pair: `antiprior_off_s0` / `antiprior_on_s0` in `kaggle/campaign.py`
      (`--include-dead`; `--list` labels both *open, unspent*). Two arms × 3,600 updates at the
      measured 5.618 s/update (§2.1's `kaggle-wall-clock` row) is **~11.2 GPU-hours** of the 30
      h/week free quota — the cheapest mechanism-level question in the repo. Status as of
      2026-10-01 12:52 NPT: **the `off` arm is done, and the `on` arm is training on this Mac.** The
      owner's decision moved it off Kaggle: *"run the on arm on the Mac, not Kaggle. Continue it to
      3,314 updates (dose-matched to the off control), one run at a time, keep the Mac from sleeping
      for the whole run, then download + Release the weights immediately and eval macro vs the
      control's 0.5339."* §9.56 records the launch and the two mechanics that make the dose-matching
      real (the step index, and which scheduler the resumed run carries).
      The `off` arm ran on the local M5 through the canonical launcher (§9.50's lane move) and
      finished at **3,314 of 3,600** updates — ended by its own `--stop-factor 3.0` guard on a
      596.40 s step that `pmset` attributes to battery 'Maintenance Sleep', not to allocator paging
      (§9.53(iv)). Its wall price is the measured **6.719 s/update** averaged over 0→3,300
      (22,173 s of stamped log), 1.20× the T4's slope, and its weights are on the
      `antiprior_off_s0-weights` Release.
      **It reports test macro 0.5339** (floor 0.4331, uniform 0.3321) against V1-B's 0.4893 — a
      **+0.0445** (exactly 0.0445463372; the "+0.0446" this bullet printed until §9.55(vi) was the
      difference of two *rounded* macros) that belongs to the data path and not to the flag, which is
      precisely the confound the fresh control was built to expose. The registry row `antiprior-off-macro` binds that figure
      to one command over committed bytes:
      ```bash
      uv run python -m myna.report --suite data/decision-v2-pilot --split test \
          --metrics runs/antiprior_off_s0.metrics.json \
          --laya runs/laya_decision_v2_test.json \
          --out runs/antiprior_off_s0.report.json
      ```
      Re-run on 2026-10-01 to a scratch path rather than over the witness, and the diff is the artifact
      `runs/antiprior_off_s0.report_repro.log`: **372 leaves compared, 2 differ — `cmd` and
      `metrics_file`, both provenance strings naming the pre-flatten
      `runs/antiprior_off_s0/metrics.json` — and the whole `g1` block plus all 16 cells compare equal.**
      A row whose own command rewrites the file it is checked against cannot show that, so this re-run
      printed elsewhere and left the witness byte-unchanged. G1 is still not met on this arm: 0.70 target, and the
      "+0.15 over the floor" clause observes **+0.1008**; laya's 0.667 leaves the gap at −0.133.
      The dev-mid trace flattened rather than climbing: 13 points from 0.3682 to 0.4332, OLS
      **+0.0231 per 1,000** over all of them, +0.0166 over the last five, +0.0030 over the last three
      (the earlier +0.0354 and +0.0191 readings were windows of this same live series, §9.44; the
      completed log supersedes both). Coverage held across the short dose: `paraphrase_draws`
      **207,411** over 3,314 updates = 62.59 per update, against V1-B's 225,547 over 3,600 = 62.65.
      The pair has since been audited on the commands that *ran*, not on the cells the generator
      prints: `runs/antiprior_off_s0/run.json` and `runs/antiprior_on_s0/run.json` each carry 20
      flags and **18 compare equal — the only differences are `--out` and `--anti-prior`**
      (§9.55(i)). Both are `--device mps`, both `--seed 0`, both on this box, so this is the one
      comparison in the whole 10c lane with no machine and no dose-direction confound in it.
      **And the stopped `on` arm is not zero evidence.** It stamped 540 updates before the lane was
      shut down by hand (TASK 3, exact PID), and its last save is the step-500 snapshot, so the
      dose-matched read of the flag is two points: dev-mid at 250 is off **0.3682** / on **0.3592**
      (−0.0090) and at 500 off **0.3962** / on **0.3789** (−0.0173). Both are *smaller than the `off`
      arm's own swing between adjacent evals* (0.3962 → 0.3638 = −0.0324 between its 500 and 750
      readings), so the honest sentence is **the flag is not resolvable at 540 updates** — the signs
      agree with the pre-registered downside scenario and the magnitude does not exceed the noise
      floor (§9.55(ii)). What the same 540 steps *do* settle is the wall price: after its first
      interval, which costs **89.55 s/update** and is data warm-up (the `off` arm's identical first
      interval costs 6.75), the `on` arm runs **6.63–7.65 s/update** against the `off` arm's
      **6.43–7.27** over the same intervals. `--anti-prior` buys no wall, so the ~11.2 GPU-hour
      estimate at the top of this bullet needs no correction for the flag.
      The `on` arm had moved back to Kaggle because the Mac's thermal budget was thought spent; that
      lane is now closed by the owner's decision above and the measurements below stand as the record
      of why it cannot be run there at all. The kernel is
      built (`kaggle/notebooks/antiprior-on-t4`, every arm through `kaggle/run.py`, `--dry-run`
      printed in-cell) and pinned to the SHA that carries the `ask()` fix, but the new account
      `aashish124` is handed a CPU session even though the server's own record of the pushed kernel
      reads `enable_gpu: true` and `machine_shape: "NvidiaTeslaT4"` — five request variants across two
      notebooks all come back `torch 2.10.0+cpu`, `device_count 0`, `ACCELERATOR_TYPE None`, while
      `kaggle quota` prints 30.00 h of GPU sitting unused and the identical recipe trained all 3,600
      updates on `aashishkumarmahato01`'s T4 (§9.53(vi), which also retracts the phone-verification
      guess first written there). Cell 1 refuses to train on CPU rather than print identical-looking
      numbers, and
      that refusal has been fired once on purpose. `--free-gib 9.0` was *not* adapted: V1-B's own
      recovered `metrics.json` records `mem_plan_free_gib = 9.0` from the T4 run, so the pin is
      shared by both boxes and changing it would silently re-clamp the batch (§9.53(v)).
      The guard that this bullet has always promised — "judge it on the emitter count, not the
      macro" — now has its baseline measured on the box the pair actually ran on. Tier 0 over the
      `off` control's own weights:

      ```bash
      uv run python bench/diag_question_ablation.py --run-dir runs/antiprior_off_s0 \
          --metrics runs/antiprior_off_s0.metrics.json \
          --out runs/tier0_antiprior_off_s0.json
      ```

      The harness prints its blocker before its numbers — `VOID: this run STOPPED at step 3314 of
      3600 requested … No claim may be read from them` — so what follows describes truncated bytes,
      and it is still the most informative comparison in the repository, because the as-scored arm
      reproduces **all 716 keys** of the committed report (0 tie flips), which binds all six arms to
      the published 0.5339. Against `tier0-ablation` on V1-B, **the +0.0445 arrives with less prompt
      selectivity, not more**: blank-instruction rises 0.3341 → **0.4315** (+0.0974) and blank-options
      0.2669 → **0.3674** (+0.1005), both by *more* than as-scored's +0.0445, so the instruction-reliance
      gap closes **0.1553 → 0.1024** and the options gap **0.2225 → 0.1665**. The state axis is the
      exception and the one unambiguously good line: swapping a row's state now costs
      0.0720 → **0.1263**, and the cells that move at all go **8 → 10** (`contrastive/decision`,
      `mnli/relation`, `sst5/sentiment` join them). The count that 10c's verdict turns on — `grep -c
      '"collapse_share": 1.0'`, which is 3 in V1-B's witness — is **4** here: `agnews/is_business`
      became a constant emitter, answering label 0 on 43/43 rows and landing exactly on its own 0.698
      floor, which is how its accuracy rose 0.6512 → 0.6977 (§9.55(iii), row `tier0-off-control`).
      The confound this bullet used to carry — `off` on the M5 at 3,314 against `on` on a T4 at 3,600,
      so the pair priced the flag *and* the machine *and* the dose — is gone, and the reason is the
      owner's decision above: both arms are now this box at this step index. What replaces it is
      narrower and has to be named: the `on` arm is a **continuation**, not an uninterrupted run. It
      resumes the step-500 snapshot with `--resume`, which re-seeds the data RNG from `--seed 0` at
      step 501, so the batch sequence after 500 is a fresh draw rather than the one an
      uninterrupted 3,600-request `on` arm would have made, and its `--steps` request reads 3,315
      because that is the value whose last executed index is the control's 3,314 (§9.56(ii)). The
      learning-rate curve is *not* a third variable — the resumed scheduler carries the saved
      `T_max = 3600`, which is what the control annealed against. Two guards on the read: the verdict
      stays on Tier 0's **emitter count over the 16 cells**, measured against the **4** the `off`
      control prints on this same box at this same step, and any macro quoted from this pair is
      quoted *with* the resume sentence attached, never as the flag alone.
      The clause the first of those guards leaned on — that the emitter count is "a property of the
      batching rule, not of the box" — is **measured and wrong**: `off` (M5, 3,314) and V1-B (T4, 3,600)
      share `--anti-prior off` and
      differ only in box and dose, and their emitter counts are 3 and 4. The emitter count is
      therefore *not* box- or dose-invariant, and it can only be read as a verdict against a baseline
      from the same box at the same dose (§9.55(iv)). With the baseline measured, that rule is now
      falsifiable in one number: **an `on` arm that comes back with 4 or more constant emitters has
      bought the mechanism nothing**, and one that comes back with 3 or fewer has moved it in the
      direction the anti-prior rule was written to move it.
      The control is *run*, not borrowed, because 0.4893 came off
      the `myna-code` dataset version as it stood before any of the data-loader work landed on this
      branch: judged against that number a difference would price the flag *and* the code drift,
      judged against a fresh `off` arm it prices the flag. Both arms keep `--score-loss ce`, `--seed
      0`, `--stop-factor 3.0`, `--save-every 250` and the same `--free-gib` pin; `tests/test_kaggle_bundle.py`
      asserts the two generated cells are byte-identical after normalizing the one flag value, so a
      third variable cannot enter the pair later. That last clause is about the two generated Kaggle
      cells, and the local lane did get two more differences when its `on` arm was continued rather
      than restarted — `--steps` and `--resume` — which §9.56 names instead of letting this sentence
      stand as a guarantee it never covered. `off` stays the default at every layer
      (`DEFAULTS["anti_prior"]`, every existing cell, `train.py`'s own default), which is what keeps
      every published number standing.

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
  on-device) — those cannot move. **The directive itself was lifted by the account holder on
  2026-10-01**, in writing ("run the training locally … I gave you all permission"): with the T4
  unreachable (§5 P0's auth line), keeping it would have stopped every dose experiment, so the 10c
  pair runs on this box. What the lift does not change is the box's memory ceiling — this arm
  requests `--batch 10` and pins `--free-gib 9.0` precisely so the plan picks the same batch V1-B
  trained at instead of sizing to whatever bytes are free at the moment the plan runs. The blockers
  this loop could clear are cleared: (1) the remote now exists (§5 P0, 2026-09-30), though that
  changes nothing about how the bundle travels: a Kaggle kernel has no internet unless it is
  granted, so the code still ships as the kernel's uploaded source and the data as a dataset
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
| Upstream-data comparison reads as unfair | medium | **disclosed, P8 8c**: README prints myna-scratch (0.348 mean over two seeds), myna-trained (**gated on its witness being committed here** — the run has happened, and 3i publishes its number from its own artifact), and laya (0.667) as three rows over the same 16 cells, with the 0.433/0.332 floors and the void-checkpoint disclosure (0.338, below its own control) beside them |
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
    from zero — and answering one question from a cached state costs 14.5–17.4 ms p50 whether that
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
26. **Two numbers in the export path were labels, and reading the artifact out loud fixed both.**
    6a shipped `meta.json` with `"opset": 17` because that is what was passed to the exporter. It is
    not what the exporter wrote: torch warns that 17 is below its implementations, emits opset 18,
    and its fallback version-converter aborts with a `RuntimeError` that is printed and then
    survived — so the metadata asserted a property of a file that did not have it, and the fix is not
    the number but the source (`opset_actual`, read back from each graph with `onnx.load`). The same
    mistake in the other direction was the byte budget: `files_mib` summed `*.onnx` and reported
    0.43 MiB for a model with 36 MB of weights, because the dynamo exporter stores initializers in a
    sibling `*.onnx.data`. `bench/mutation_onnx.py`'s first pass caught exactly that — deleting the
    `*` from the glob survived two rounds of assertions, because a threshold like "> 0.01 MiB" cannot
    distinguish a skeleton from an artifact; what catches it is reading the graph and requiring the
    reported size to cover the bytes its own initializers declare.
    What the honest numbers now say is worse news than the draft implied: the two graphs are
    **35.6 + 58.13 MiB fp32 — the trunk twice over** — so §2.1's ≤ 20 MB int8 target cannot be met by
    quantising each graph on its own. Either the weights are shared between them or the browser
    carries one graph and calls it twice, and that is now 6b's premise rather than a surprise at the
    end of it.
27. **Three of the browser battery's 24 mutations survived on the first pass, and two of those were
    the instrument's fault, not the code's.** `bench/mutation_browser.py` edits one line of the
    JavaScript implementation at a time and requires `node browser/selftest.mjs` to go red, so a
    survivor normally means an untested property. These three did not all mean that:
    * **The scan loop had two sufficient exits.** Disabling its `if (take.length < chunk) break;`
      survived because the loop *bound* — `fed < ids.length || fed === 0` — already stops an empty
      document after its one padded call. The program was right and the mutation was wrong, so the
      mutation was replaced by the one that attacks the same property from the side that is actually
      load-bearing (dropping `|| fed === 0`, which dies on the empty-document check), and the
      redundancy is now listed in the battery's own "deliberately absent" section instead of being
      quietly kept as a passing line.
    * **The score answer's expectation was compared as `null` against `null`.** The two probe floors
      in `browser/expected.json` were 0.9 and 0.9995, and at both of them the `score` question
      abstains — so `readout` returned `null` for `score`, the engine's want was `null`, and
      `probs.reduce((a, p, j) => a + j * p)` could be rewritten as `a + p` (which is 1.0 by
      construction, i.e. always wrong) while the gate printed green. Two fixes: a third floor at 0.6,
      below every argmax this checkpoint produced (the softest is 0.7802), so all three types have a
      committed case; and `parity.mjs` now asserts that coverage rather than assuming it — a type
      with no committed case fails the gate and says *"no committed score case: its commit-branch
      arithmetic is never compared"*. An instrument that reports only what it happened to reach is
      how a gate stays green for a quarter of its life while one of its six checks tests nothing.
    * **The byte total was never added up.** Per-file equality held while the printed *sum* was
      rewritten to count only `*.onnx`, producing "4.15 MiB across 6 files" next to a 54 MiB
      `weights.bin` — the exact shape of the 6a bug §9.26 records, arriving again through a second
      door. What catches it is comparing the printed total against the directory's own total, which
      is what the node gate now checks in the same line.
28. **int8 meets ≤ 20 MB and misses the parity bound by 648×, and the tab is barely faster for
    it.** §2.1's browser row carried a single compound target — "≤ 20 MB int8" — that was really two
    claims, and measuring them separately split them. `bench/quantize_int8.py` runs onnxruntime's
    dynamic quantisation over the *exported graphs* (not a fresh torch model, so the artifact
    `bench/mutation_onnx.py` gates is the thing being shrunk) and re-shares the result through the
    same `share_weights`: the artifact is **18.56 MiB**, inside the target, and the wire is 32.33 MiB
    once the 13.58 MiB wasm runtime is counted (`runs/browser_g3_int8.json`). onnxruntime-web 1.30
    runs the result — `MatMulInteger`, `DynamicQuantizeLinear` and `DequantizeLinear` are all
    implemented in the wasm build, which was the assumption most likely to kill this route and it did
    not. What killed it is the agreement: chained state **2.67e-02** relative and option
    probabilities **6.48e-02** absolute against the fp32 gate's 1e-4, i.e. four of the tab's six
    checks green and the two that carry numbers red. The three decisions' *labels* were unchanged on
    that document — one document, 525 tokens, three questions, which is not a suite and is not
    offered as evidence that quint8 is safe at 77 options. And the speed the loss bought is small:
    p50 **413 ms** against fp32's **436 ms** at 10 threads (5%, inside the run-to-run spread §9.23
    measures on this box, and the int8 sweep ran at a *higher* load average) and 757 ms against
    742 ms single-thread, i.e. slower. So the byte target is met and the artifact that ships is
    fp32; reopening int8 needs a quantisation tolerance argued against the decision suite, not
    inherited from the fp32 export's bound.



29. **int8 on MLX: 3.17× the bytes, 0× the milliseconds, and 9.24e-03 of agreement — and the
    reasoning that predicted a win was wrong twice.** Two earlier drafts of this finding were both
    wrong in the same direction, and both were killed by arithmetic rather than by the clock.
    (a) "the linears are a small share of the model, so quantising them cannot matter":
    they are **99.4%** of the per-state-token work — 13,565,952 MACs through the matmul'd weights
    against 82,944 in the attention (the state read/update and the causal intra-chunk piece,
    counted at chunk 16 from the shapes in `runs/myna-v0/model.pt`; the split is not sensitive to
    how generously the attention terms are counted). (b) "then it must be
    bandwidth-bound and int8 will win": weight-only int8 removes bytes from the *weight* read and
    no MACs, so the whole claim rests on the kernel being starved, and `bench/diag_mlx_int8_gem.py`
    says it is not — at myna's own shapes the fp32 matmul already runs at 55–257 GB/s of weight
    read on this box and the quantised one at **10–37 GB/s**, so `mx.quantized_matmul` spends more
    unpacking group-64 int8 than `x @ w.T` spends reading fp32, and 5 of the 6 shapes are outright
    slower (ratios fp32/int8 0.43–1.06). Between (a) and (b) sits a measurement bug worth naming:
    the first version of that probe enqueued 200 matmuls and evaluated only the last array, so the
    other 199 nodes were unreferenced, never ran, and the probe printed 0.4–3.1 µs/op with int8 up
    to 1.31× ahead. Holding every node and syncing once costs 3.3–28.7 µs/op and reverses the
    sign. A microbenchmark whose nodes can be garbage collected is measuring Python.
    The engine-level result is the honest version of all this: **no latency claim on the int8
    path**, sign of the int8 − fp32 delta flipping between two committed runs on 4 of 7 state
    lengths while fp32's own spread reaches 15.2% (§5 P6 6c). So the artifact that ships for Apple
    is fp32 — 55.13 MiB, 1.76–2.22× over torch-MPS — and 17.40 MiB int8 stays in the tree as a
    measured option for whoever needs the bytes for a different reason (a memory-constrained
    unified buffer, say), not as a speed route.
    Second correction, about the target rather than the number: **"`≤ 20 MB int8`" was one cell and
    is now two.** §2.1 carried it as *the* int8 target; §9.28 killed it on wasm agreement
    (6.48e-02 = 648× the fp32 bound) and this pass kills it on Apple latency, for the opposite
    reason — MLX int8 reaches 9.24e-03, ~7× tighter than wasm's, and is refused because it is not
    faster. A single "int8 status" line would have hidden that: the two paths fail differently, so
    each now has its own row and its own §9 entry. And the reopen condition is not "find a better
    quantiser" — it is that a *latency* route needs a kernel whose dequant pass is not the
    bottleneck, at which point the agreement question has to be re-argued against the decision
    suite anyway (the 1 flip in 20 above sits on a 0.0008 reference margin, so probability-error
    bounds alone would not have caught it, and keeping `.gate.weight` in fp32 changes 9.24e-03 to
    9.53e-03 and flips the same row).

30. **The headline cells were copied from prose, so eight of them were wrong — the labelling pass
    (P8 8a) killed them all.** Not one bad number but seven, in three files, and they share a
    mechanism: a figure was *typed into* README/PLAN/SPEC from a run's summary rather than *copied
    from* the artifact, and each retyping was allowed to round, merge or misremember. The dead
    cells, each with the line that buries it:
    - **"v0 dev 0.968 / test 0.951 overall."** Neither number is an overall. `runs/train-v0.log`
      prints no overall at all — it prints nine per-question rows under `=== dev ===` and nine
      under `=== test ===`, whose macros are **0.9599** and **0.9523**. 0.9682 is a `dev-mid`
      probe line (steps 7500, 8000 *and* 8500 print it), and `dev_probe` is the first
      `--eval-n` rows of that same dev split — 192 by default, so a subset scored every 500 steps,
      not an overall. And "0.951" is printed by no artifact in this repo. The macros are now
      computed from the log's own rows, and the sentence that quotes them says "macro of the nine
      rows", which is the only honest way to average what is actually committed. (The probe's row
      count is itself unrecorded — the log carries no argv, so 192 is read off the current default
      and cited only to say the probe is a subset, not the final eval.)
    - **"dev→test gap is 1.7 points."** It is **0.76** (0.9599 − 0.9523). 1.7 was 0.968 − 0.951 —
      arithmetic on two numbers that were themselves wrong, which is the fastest way to compound
      an error: a derived cell looks more precise than its inputs and inherits both mistakes.
    - **"~3.7 h."** The log's elapsed counter at step 8999 reads **15,693 s** = 4.4 h. The earlier
      figure was hours-by-eyeball from a partial line, and it appeared as "~3.7 h" in one file and
      "~3 h" in another.
    - **"3,000 synthetic examples."** Deleted rather than corrected, because *nothing records the
      corpus size* — `train.py` logged the vocab and the parameter count and not the row counts.
      The nine printed test accuracies each round to their 4-dp value at 1200 rows per workflow
      (3,600 test rows) and at no smaller n — searched over n = 50…4000, the only solutions are
      1200, 2400, 3600, 4000 — but that is arithmetic on printed rounding, so it is tagged *p* in
      PLAN.md and not quoted as a measurement. Fixed forward: the trainer prints its own `cmd:` and
      one `split <name>: N rows over M groups, per-group lo-hi` line per split, guarded by
      `tests/test_train_logging.py`.
    - **"14.7–17.4 ms to answer one question from a cached state."** The 1-question `ask` p50s on
      the published ladder are 17.433 / 16.082 / 14.52 / 14.59 / 14.70 ms, so the band is
      **14.5–17.4** — the low end was the 1,060-token row mistaken for the minimum, which is a
      one-keystroke error that survives any amount of rereading and dies to one `min()` over the
      ladder. Same pass: "R² 0.74 vs myna's 0.99" now quotes all three fits from the witness's own
      `fit` block (laya 0.7356, myna end-to-end 0.9915, myna ask-only 0.9619), because two rounded
      R²s in one sentence read as one number restated.
    - **"ECE ≤ 0.04 everywhere."** Test ECE runs **0.0115–0.0415** and dev **0.0080–0.0438**; both
      maxima exceed 0.04, so the sentence was false in the direction that flatters. Related and
      subtler: §9.11 quotes the dead claim it corrects as "v0's dev ECE ≤ 0.042" — 0.042 is the
      *test* max, with dev's 0.0438 not reachable at all (dev's temperature fit is what makes it
      optimistic). The §9.11 correction still stands on its own point; only its number was
      labelled with the wrong split.
    - **"15.35M measured" in §2.1.** True of the default `MynaConfig` (15,353,984 params at
      vocab 4096) and false of the artifact the project ships: v0 has **14,449,280** params,
      because its tokenizer learned 1,740 entries. `runs/bench_mlx_int8.json`'s `n_params` says so
      and `runs/train-v0.log` printed both at the time. §2.1 now carries both counts with their
      configs; "measured" was never the issue, *which configuration* was.
    - **"dev accuracy up to 0.9657" from the RLCD pass.** The number is real
      (`runs/rlcd_v0.log`: 0.9626 → 0.9657, mean ECE 0.0191 → 0.0076, mean Brier 0.0258 → 0.0232)
      but it is not *this* dev: `rlcd.py` regenerates the split at 600 rows per workflow under its
      own rng stream, so 0.9626 and the headline 0.9599 are different samples of the same
      distribution and their difference is sampling, not training. The publishable claim is the
      before/after pair inside one draw, which is what README now says.
    Two instrumentation gaps came out of the same pass. `runs/latency_matched.md` and its run-1
    sibling record platform, device, thread counts, reps, both checkpoints by path and commit, and
    both dtypes — and no load average and no argv, because §9.23 was written after they ran. That
    is disclosed in PLAN.md rather than quietly re-run (a third run could only lower a minimum, and
    §2.1's ratios are minima), and `bench_latency_matched.py` now writes
    `meta.load_avg_after`/`meta.cmd` and prints them in the markdown header, so the next P4 run is
    restatable. And a "~0.6 MB" in an API docstring became 576 KiB once measured: the S stack is
    589,824 B (the `state_bytes_total` every `runs/browser_g3.json` row prints), while the saved
    file for a 51-token thread is 592,533 B — the token list is the difference, and unit drift
    ("~0.6 MB", "~100 KB") is exactly what a measured figure in the same sentence prevents.
    The structural lesson is the convention §2.1 now uses everywhere: every headline cell carries
    its own tag — ***m*** measured from a file in this repo, ***p*** projected by arithmetic from a
    model we did not open, ***g*** gated — and the artifact it came from, because a table can only
    be audited cell by cell. A whole-table "all numbers measured" label is what let nine wrong cells
    ride together.
31. **The reproduction registry's own doc check was vacuous, and the mutation battery said so
    before it ever printed a red row.** P8 8b added `bench/reproduce.py`: 19 rows, each binding one
    published table to the command that regenerates it and to the committed artifact whose bytes the
    prose quotes. One of its three assertions was "a doc still prints this command" — and README's
    new *Every table* section is generated from `ROWS`, so it prints all 19 commands by construction.
    The assertion could not fail; `--check` was green on 19/19 the day it was written and would have
    stayed green if every one of those commands had been deleted from the prose around its table.
    Caught only because `bench/mutation_reproduce.py` was about to be run and the entry
    *"the docs are read one file at a time, so only README counts"* looked like it would survive —
    which was the signal that README alone was already sufficient, i.e. that the whole `DOCS` tuple
    was decorative. Fix: `doc_text()` cuts everything between two markers around the generated block
    before matching, so only hand-written prose counts as evidence, and the assertion immediately
    went **7/19** — twelve rows' commands lived nowhere but in the index that echoed them. Those
    twelve are now printed next to the tables they belong to (README's architecture, browser, MLX
    and trained-v0 sections; §4.1, §4.2 and §5 P3 3c here), one of which —
    `bench/check_python311.py` — had to be re-run to rewrite its log, because the harness gained its
    `$ <cmd>` provenance line *after* `runs/python311_check.log` was committed, and §4.2's sentence
    about that first line was true of the file only from the rerun onward. Two sibling invariants
    moved from test assertions into `check_row()` for the same reason: a row quoting no falsifiable
    figure, and the runner's refusal to launch a `gated-kaggle` or `retrain` row. A rule that lives
    only in a test is a rule the source can drop without anything noticing. Battery:
    **21/21 caught** (`runs/mutation_reproduce.log`, 25 tests in `tests/test_reproduce.py`), and
    every `--run` mutation is paired with a test that passes `--dry-run` — the scratch repo symlinks
    `runs/`, so a battery that launched a real harness would overwrite the witnesses it is checking.
    The general form, which is the one to look for next time: *when a checker generates part of what
    it checks, the generated part is not evidence.* The count it produced was green, and green was
    the sound of it reading its own output back.
32. **A competitor cell the suite asks two ways was scored by whichever row its JSON wrote
    last, and that inflated every published laya figure by about 0.008.** P8 8c joins
    `bench/eval_laya_real.py`'s witness to myna's table inside `src/myna/report.py`
    (`laya_cells()`), and the two sides do not share a key: myna's cell is
    `(source, question)`, laya's row is `(source, qid, type)` — because the suite asks
    `contrastive/decision` *both* as a 4-way choice and as a noul. Seventeen rows, sixteen
    cells, and `laya_cells()` was a dict comprehension over the rows, so the 24-answer
    choice row (0.417) was silently overwritten by the 16-answer noul row (0.625) and the
    cell was published as **0.625 over 40 answers**. Which row survives is the one whose
    `type` sorts last (`eval_laya_real.py` writes `sorted(acc.items())`), and on this
    witness that happened to be the higher-scoring one — an accident of the alphabet, not
    a bias, and the kind that reads as a result. Merging by answers instead
    ((0.416667×24 + 0.625×16)/40 = **0.500**) moves, recomputed from the same committed
    witness: the 16-cell MACRO over the cells both sides scored **0.674628 → 0.666815**,
    the *shared-instruction* stratum's laya column (14 cells) **0.681718 → 0.672789**, and
    the control's published gap **−0.3285 → −0.3207** against myna's 0.3462. The
    *per-row-instruction* stratum moved by exactly nothing — two cells, neither of them
    contrastive — which is the signature of this bug class: one bad cell, and every
    average that contains it carries the error without being able to name it.
    What caught it was a test written for an adjacent reason: one that deletes a cell the
    competitor never scored and re-derives the 15-cell intersection from `data["rows"]` by
    hand (`test_a_cell_the_competitor_never_scored_moves_the_floor_too`) rather than reading
    the figure back out of the function under test. Two implementations of the same average
    disagreed by more than a deleted cell can explain, and the disagreement was in the one
    that was shipping. Fix: `laya_cells()` indexes a *list* per cell and merges by answer
    count, returns the `rows`/`parts` it merged, `main()` prints the merged cell in full
    (`contrastive/decision is 2 rows of that JSON over 40 answers: choice 0.417 (24) +
    noul 0.625 (16) → merged by rows to 0.500`), and `--out` carries `laya_merged` so the
    prose that quotes 0.500 has a field to quote. `bench/mutation_report.py` grew four
    entries for it — restore last-row accuracy, restore last-row `n`, empty `laya_merged`,
    drop two-row cells from the join instead of merging — and the battery went 54 → 67 with
    all 67 caught (`runs/mutation_report.log`).
    The same mistake, one file over: §4.1's per-source list had three cells that were not
    per-source. `agnews 0.950` is `agnews/topic`, one question of that source's five, whose
    row-weighted figure is **0.915**; `yelp 0.825` is `yelp/recommend`, one of two, against
    **0.650**; and `contrastive 0.625` is this entry's 16-answer noul row, against **0.500**
    for the source. Only a single-question source was safe to quote that way — eight of
    eleven cells, so the list was 73% correct and read as if it were 100%. The eleven
    figures now printed are recomputed from the witness's own rows and their weights sum to
    the 546 answers whose row-weighted mean the harness itself prints (0.6319), which is the
    check that a per-source list is a *partition* and not a selection.
    The general form: *a join is a claim about keys.* Either the right-hand side is unique
    per key, and something asserts it, or the join aggregates — and then the aggregation has
    to be printed, because a merged cell is a different quantity from either row and a
    reader cannot tell which one they are being shown. A comprehension over rows is how a
    uniqueness assumption hides: it cannot fail, it just keeps the last one.

33. **A branch no input could distinguish sat in the ordinal loss, and its mutation
   survived its own battery; the fix was to delete the branch, not to write a harder
   test.** P9 9a gave `typed_loss` an `ordinal`/`opt_valid` pair that arrives either
   per-row (`[B,N]`, `[B,N,O]`) or shared across the batch (`[N]`, `[N,O]`), the same
   two forms the question tensors have carried since P2. The first version expanded the
   shared form by hand — `if ordinal.dim() == 1: ordinal = ordinal.expand(gold.shape[0],
   -1)` — which reads like a shape contract and is not one: broadcasting already turns
   `[N]` against `[B,N]` into exactly the tensor the expand builds, and a shared-form
   mask *by definition* says the same thing about every row, so no input exists in which
   the two forms disagree. `bench/mutation_ordinal.py` proved it the only way that
   counts — `if ordinal.dim() == 1:` → `if False:` left all 18 tests green — and so the
   branch is gone, with the contract now living in a comment on the arithmetic and in
   `test_the_shared_and_per_row_forms_agree`, which is the witness worth keeping because
   it can still fail. Two rules fall out. *A battery entry that cannot fail is evidence
   about the code, not about the battery*: either the instrument is under-sized or the
   branch is dead, and the two are told apart by trying an input in which they should
   disagree. *An explicit broadcast is not documentation* — writing a conversion the
   language already performs buys a line nobody can test.
   The same run produced its exact opposite, which is why the rule is not "delete
   defensive masking". The `* opt_valid` that bounds the CDF sum to a cell's own options
   looks equally redundant, because a caller that masked its logits to -1e9 makes every
   tail term `(1 − 1)²`; and `MynaModel.forward` does mask them. But `typed_loss` cannot
   see that — it receives a tensor — and the value of a sum over `O` columns is *not* the
   value of a sum over the cell's `K` options for a caller that built its logits by
   hand, which is what three of this file's tests do. Masked away, the mutation is caught
   by exactly one of them (`test_the_builder_masks_feed_the_loss_in_the_order_the_loop_
   passes_them`); that single test is the reason the multiply stays and the reason its
   name is written down rather than left to the next reader.

34. **A committed report table put its macro row six columns right of the header it
   belongs to, and 29 tests, 125 assertions and 67 green mutations were satisfied by
   it.** `runs/scratch_decision_v2_test.md` and both `runs/report_*.log` carry the
   per-stratum macro line that `table_text` builds. Its label,
   `  macro over its 14 scored cells`, is 32 characters and the label field was 26
   wide, so the row's first figure ended at column 59 where the header's `model` ends
   at 53: every average in the most-quoted line in the artifact sat under the column to
   its left. No number was wrong — the misplacement cost nothing but meaning — which is
   precisely why it survived three rounds of this file's reporting work. Every
   assertion in `tests/test_report.py` greps a line and looks for a substring;
   `bench/mutation_report.py` mutates formulas and values; and `bench/reproduce.py
   --check` asks whether a quoted figure is present in its witness. All three were
   green on the misaligned table, and all three would still be green today if the
   widths had been left alone.
   What found it was a *new* column. 9c's `clip` and 9b's `brier` are honest only if a
   figure sits under its own header, so the test added for them
   (`test_the_macro_row_lands_in_the_same_columns_as_the_cells`) reads each line by
   character offset, with the offsets derived from the printed header rather than from
   a constant — and the first run showed the macro row one field out of place. The fix
   is structural: `LBL`/`WID` and one `_trow` that the header, every cell row and every
   macro row pass through. Adding the columns then produced a second failure pointing
   the other way: with seven numeric fields, a six-character delta butted against the
   Brier figure and the table printed `0.312-0.125`, which a reader takes as one number.
   No assertion about column *edges* can see that — right-aligned fields land a
   six-character value on their edge whether or not anything separates it from its
   neighbour — so `WID`'s last field 7→6 survives every positional check and needed an
   instrument of its own (`assert not re.search(r"\d-\d\.\d", ln)`). The battery now
   carries the label-width mutation and the field-width mutation as separate entries,
   because they are failures in opposite directions and only the second one was visible
   before anything was written about it. `LBL`'s 21 stopped being a design taste too: it
   is this split's widest real cell name, `contrastive/decision` at 20, and the test
   asserts `LBL > widest` over `cell_stats(load_split(...))`, so a future source with a
   longer name breaks the build instead of the table.
   The general form: *a number's column is part of the number.* Position is a claim
   about which quantity a reader is being shown, no value assertion can see it, so a
   table with more than one row type needs at least one test that reads by offset — and
   widths have to be checked both for overrunning and for touching. The process form
   matters more: `--check` verifies that a figure is in its witness and that a doc
   quotes its command, and it does **not** re-run the command, so an artifact can carry
   a layout its own code would no longer produce, indefinitely and green. Re-rendering
   the control to settle this produced a determinism witness for free —
   `runs/scratch_metrics_test.json` came out byte-identical (sha256 `99f54cb0…`), seeds
   0/1 printed 0.346/0.351 again, and every per-cell figure matched the copy it replaced.
   Layout moved, nothing scored did, which is what licenses saying the re-rendered
   artifact is the same measurement better drawn.
35. **"The memory plan is a property of the corpus."** The code made it a property of the *run*.
   `state_token_p95` sampled its rows with `random.Random(args.seed)`, so changing the experiment's
   seed changed the p95, which changed the batch the plan approved — measured on the full pilot:
   p95 267 with batch 15 at seed 0, p95 274 with batch 14 at seed 1, from one command over one
   corpus. No published figure moves, because every run this repo has published used seed 0, where
   the seeded draw and the fixed `random.Random(0)` default are the same sample. What nearly
   happened is an ablation whose two arms trained different batch sizes for a reason no comment
   mentions, and the assertion that caught it was one a reviewer would call over-strict: a
   comparison script demanded all four arms share a batch, and it was *right* to fail. The fix went
   into the call site (`p95 = state_token_p95(tok, sample)` — the function's own default), not into
   relaxing the assertion. The general form: **a control that differs from its treatment in more
   than the variable under test has compared two experiments.** Read it off the runs rather than
   off the launcher — `bench/ordinal_ab.py` reconstructs each arm's flags from that arm's own
   `cmd:` line and asserts a pair differs only in `{--score-loss, --seed, --out}`, then checks the
   two arms of a pair on batch, p95, `last_step` and "did not stop early".
36. **A delta averaged across seeds whose headline moved in opposite directions.** The first read
   of the `ce` vs `emd` ablation quoted the resolution ratio *pooled over both seeds* (0.5194 dev,
   0.8765 test) and concluded "below 1, not resolvable". The conclusion stands; the evidence offered
   for it did not. Pooling three priced cells across two seeds hid the fact that on test, seed 0
   clears its churn floor (ratio 1.7591) while seed 1's mean is 0.0 — the two draws
   disagree, which is a different finding from "both small" — and the number that actually carries
   the verdict is the sign agreement across seeds (2/3 dev, 1/3 test) beside macro deltas of
   −0.011828/+0.003155 and +0.007952/−0.001408. The pooled arithmetic was not wrong; it was the wrong unit,
   and it erred *toward* the negative result, which is the direction nobody audits.
   `runs/ordinal_ab.json` stores the per-pair figures first and the pooled ones under
   `pooled_over_seeds`, and the judge prints the pooled line labelled as the one to distrust. The
   general form: **average across replicates only when they agree** — a mean over contradictory
   draws is a number no arm produced, and it will look like whichever conclusion the averaging
   accidentally supports.
37. **One run, two "test macro" numbers, and the launch doc quoted the flattering-to-doubt one.**
   For a session every sentence about `v1b-kaggle-3600b` said *test macro 0.3983*. That figure is
   real — `/tmp/kgwork/diag_dev_cells.json` prints `0.3982739096904652` — and it is not the number
   G1 is written against. It came out of the permutation-diagnostic harness, whose roll-up is over
   the cells *that* harness chose to score, with *its* filters; the release roll-up over the 16
   cells both myna and laya score, per-cell unweighted, prints **0.4893271976084137** from the same
   weights and the same split (`runs/v1b_kaggle_3600b.report.json`, computed by the command
   `bench/reproduce.py`'s `v1b-kaggle-macro` row re-runs). The gap between them is not rounding and
   it is not symmetric: 0.3983 says myna is 0.302 under the 0.70 target, 0.4893 says 0.211 under it,
   and the smaller number made "the gap is a dose problem" look likelier than the artifact supports.
   What fixed it is not "pick the right one": it is that **a roll-up is a named statistic with a
   cell set and a weighting, and a gate may quote only the one whose command is committed.** The
   diagnostic's figure is now quoted in the launch docs solely as the number *not* to quote; the
   release report prints `macro/acc` beside `macro/cells_scored` (16) and `macro/cells_kept` (16),
   so the cell set travels with the value instead of living in the prose; and §9.30's rule gets its
   sharpened form — *the witness has to bind the number and the unit*, because a committed value
   under an uncommitted roll-up is still a projection about which statistic was taken.
38. **"Dev was flat from step 2,250" — a slope carried in prose for a session, and the committed
   log says otherwise.** That claim was in the launch docs and in the reasoning behind them: it is
   why the 3,600-update run looked converged, why dose was framed as the wrong question, and part
   of why nobody re-bought the lane. The log does not say flat. `grep -E "dev-mid acc"
   runs/v1b_kaggle_3600b.train.log | uniq | tail -5` prints the run's own tail — 0.4122 at 2,250,
   0.4283 at 2,500, 0.4319, 0.4336, 0.4321, and **0.4355 at 3,500, which is the run's high** — so
   the last 1,000 updates gained **+0.0072** and the curve was still rising when the cosine
   schedule spent out, and the jump from 2,250 to 2,500 alone is +0.0161, which is not a flat
   region by any reading. (The `uniq` is load-bearing: the box wrote every progress line twice,
   and `bench/reproduce.py`'s `kaggle-dev-tail` row commits both the tail and the doubled-log fact.)
   The correction does not rescue dose, and it is worth being precise about which way it cuts:
   extrapolating the *measured* tail slope over the 6,400 extra updates per arm in the 10k lane
   projects ≈ **+0.046**, against the +0.211 G1 needs — so the conclusion "more steps is not the
   answer" survives, but it survives on a number rather than on a remembered adjective, and the
   price of testing it is now computable: 5.62 s/update measured from the same log means the pair
   is ≈31.2 GPU h against a 30 h/week quota.
   The general form: **a slope is a figure, not a mood.** Before a remembered curve shape prices
   GPU hours, re-read it from the artifact with the command that prints it committed — because
   "flat" and "+0.007 per 1,000" imply different decisions, and the second one is a number a
   reader can check.
39. **Two mutation batteries had been refusing to run for a tick, and the repo kept citing their
   logs.** `bench/mutation_gates.py` and `bench/mutation_reproduce.py` both print
   `baseline (unmutated) copy is RED -- the battery proves nothing` and exit 1 — the guard
   working correctly — and this went unnoticed because the last thing anyone reads from a battery
   is its `N/N caught` line, not its exit code. The cause was the scratch, never the checker: each
   battery builds a temp repo out of `src/ bench/ tests/` plus a **hand-maintained list** of
   top-level files, and two things changed underneath that list — `tests/test_gates.py` grew a test
   that reads `Makefile`, which neither list ever contained, and `reproduce.DOCS` grew two launch
   docs while `mutation_reproduce.py` kept its own copy of the same tuple. So
   `runs/mutation_gates.log`'s committed `baseline copy: green` line described a tree that no
   longer existed, which is §9.34's finding wearing a harness: an artifact witnesses one run, not
   the current one.
   Both scratches now derive the file set from the filesystem (`*.md` + `pyproject.toml` +
   `Makefile`) instead of a second list, and re-running them writes logs that are **byte-identical
   to the committed ones** — the same 17 mutants, the same 21, all caught, exit 0 — which is the
   evidence that the checker under test never changed and only the harness had been quietly
   disabled: **gates 17/17, reproduce 21/21**. That sentence is where this entry's own rule bites:
   it ends `alongside mutation_kaggle_bundle.py's 39/39`, and that third figure was inherited rather
   than re-run — which entry 40 turns out to be a story about a battery that could not have printed
   it. The general form: **a guard that aborts is only as good as the habit of checking that it
   fired** — a battery that refuses is honest, but an unfixed refusal turns every later green
   claim about it into a report about the past.
40. **The mutant that entry 39's own last line quoted had never been survived — because the test
   that catches it was ready to run the job.** `bench/mutation_kaggle_bundle.py` carries
   `the dry run runs the job` (`if args.dry_run: return 0` → `if args.dry_run and False:`), and its
   docstring explained that the probe is cheap: *"a test that asks for one update so the mutant
   costs one update"*. That was true of one of the seven `--dry-run` call sites in
   `tests/test_kaggle_bundle.py` and false of the first one. `pytest -x` reaches
   `test_dry_run_prints_the_exact_command_and_nothing_else` before the cheap one, and that test
   invoked the entrypoint at the **default dose** — so the mutant does not cost one update, it
   costs ten thousand on the CPU of whoever is checking. Measured 2026-09-28: the battery stopped
   after `[13/40]`, and `ps` inside it showed `myna.train … --steps 10000 --batch 32 --vocab 8192`
   alive at 11 minutes, which I killed by PID. Worse, `pytest_in` had carried `timeout=1800` with no
   handler since the file's first commit (`e032bd3`, where the mutant and the uncapped test already
   coexisted): past half an hour that battery's end state is a `TimeoutExpired` traceback, and
   `subprocess.run`'s timeout kills only the child it spawned, so the trainer is left reparented
   and the MacBook hot. Which of the two the run would have become, nobody found out, because the
   wait ended by hand at 11 minutes — and that is the whole epistemics of this entry: the figures
   quoted for the battery (`25/25` at §5 P3 3b, `39/39` in entry 39) were neither observed nor
   refuted, they were inherited. Two fixes. `_run_py` now appends `MINI_DOSE` (`--steps 1 --batch 1
   --vocab 128`) to any `--dry-run` that did not name a dose — steps alone is not a cap, the batch
   and the vocabulary are what the seconds go to — and `pytest_in` runs pytest in its own process
   group (`start_new_session`) and `killpg`s it, so a hang costs 600 s and leaves nothing running.
   Capping the print has a price the repo should not have paid twice: the printed command stopped
   carrying the defaults, so `"--steps 10000"`, `"--batch 32"` and `"--vocab 8192"` left the
   assertion loop. A test that reads `run.DEFAULTS` without launching anything took their place,
   and two new mutants (`"steps": 10_000`→`500`, `"batch": 32`→`512`) exist to prove it bites, and
   both are caught by that test and by nothing else. The battery is now **42/42 caught, exit 0**,
   logged to `runs/mutation_kaggle_bundle.log` and committed, because §9.30 is the rule this entry
   exists to learn from; `the dry run runs the job` is caught at `[16/42]` by the test it was always
   meant to fail.
   The general form: **a probe has to be affordable by the test that answers it** — a mutation
   whose honest cost is a training run is a mutation nobody re-runs, so the figure decays into
   inheritance. And entry 39's rule applies to entry 39: it named the habit of checking that a
   battery finished, then quoted this battery's `39/39` from the record rather than from a run.
41. **dbpedia14's "unexplained dev→test collapse" was two roll-ups of one run, not two
   measurements of one quantity.** The item carried out of `v1b-kaggle-3600b` said dev 0.5324 falls
   to test 0.2047 across 37 question-sets per side, called that "too large to call overfitting
   without looking", and told whoever read it next to investigate before trusting the test macro.
   Both literals reproduce exactly today as the **unweighted mean over `evaluate()`'s per-set keys**
   of the committed `runs/v1b_kaggle_3600b.metrics.json` — and the cell's own shape says why that
   roll-up cannot carry a conclusion: 116 rows sit in 37 sets, **36 of which hold exactly one row**
   (`runs/v1b_kaggle_3600b.report.json`: `n=116 sets=37`; the dev render's strata block: *"37
   exact-signature sets, 0.69 of rows in a reused set"*). So 36 of the 37 terms are 0-or-1 coin
   flips, and because dev and test share **one** signature between them the pair never was the same
   quantity twice. On the basis the shipped report actually uses — accuracy **row-weighted within each
   cell** across the question-sets that hold it, then an unweighted mean over the 16 cells (entry 37's
   "per-cell unweighted" is that outer step, and it is a different thing from this inner one), because
   *"an unweighted mean lets a 1-row group outvote a 116-row one"* — the cell is dev **0.647** →
   test **0.457**: a −0.190 drop against a **0.129** majority floor, which makes it one of the cells
   furthest *above* guessing rather than the suite's weak point. And −0.190 does not need a
   mechanism: over the 16 cells both splits score, mean |Δ| is 0.079 with sd 0.089, five other cells
   move by ≥ 0.10 (agnews/topic −0.164, sst5/sentiment −0.163, amazon/stars −0.138, agnews
   is_scitech −0.113, is_business −0.104), and imdb/positive swings +0.125 the other way; overall
   macro is dev 0.5356 → test 0.4893. The dev half is now an artifact of its own —
   `uv run python -m myna.report --split dev --metrics runs/v1b_kaggle_3600b.metrics.json \
   --out runs/v1b_kaggle_3600b.dev.report.json > runs/v1b_kaggle_3600b.dev.report.log` — which is the
   committed metrics re-rendered, not the run repeated: no GPU, no re-run of a gated lane, and
   `bench/reproduce.py` row `v1b-kaggle-dev` re-reads 0.5356, `n=116 / sets=37` and the 0.647 line out
   of it.
   The distortion is not local to dbpedia14, and the suite-wide count is the number to carry: of the
   per-set roll-up's own terms, **414 of 690 dev (60.0%) and 417 of 716 test (58.2%) hold exactly one
   row**. Delete those terms and the same unweighted mean reads **0.6756 dev / 0.6092 test** instead of
   0.4355 / 0.3983 — so the basis was never a rounding choice; it moved the headline by 0.240 on dev
   and 0.211 on test, and a roll-up where the majority of terms are single-row cannot support a claim
   about a *model* at all. One consequence for entry 38, which this clause does not undo: the training
   log's `dev-mid acc` ladder is `train.macro_acc`, i.e. this same per-set basis — its last rung prints
   0.4355, and the committed dev block's per-set mean is 0.43547, the same statistic to the four places
   both print. The **slope** §9.38 measured
   (+0.0072 per 1,000 updates) stands, because it pairs one basis against itself across steps; but
   0.4355 must never be read as the level of a dev run that the report puts at 0.5356, and a reader who
   does will "discover" a 0.100 gap that is pure bookkeeping. **The item is closed with "nothing
   to find", and G1 is untouched** — 0.4893 against 0.70 stays *not met* on entry 37's basis.
   The general form: **read a dev→test delta on the basis the artifact publishes**; a gap between two
   aggregations of one run is a fact about the aggregation. One trap sits on the way to that rule, and
   it looks like waste: the
   37 sets come from 11 (dev) / 10 (test) distinct legend *key-sets*, because `_signature` hashes
   the criteria dict in the row's own order. That fragmentation is load-bearing, not waste —
   `parse_questions` resolves each gold through `list(crit).index(lab)`, so grouping rows whose
   legends agree only up to order would keep the first row's option list and silently mis-index
   every later row's label. Canonicalising the key without canonicalising the options would turn a
   noisy cell into a wrong one.
   The same re-check was then run against the repo's other negative result, because its verdict
   sentence quotes a macro: `runs/ordinal_ab.json`'s `meta.macro_dev/macro_test` are `macro_acc`, i.e.
   the per-set roll-up, and on the report's per-cell basis its "macro moves opposite ways in the two
   seeds" reads dev −0.0013 / +0.0053 (still opposite) and test +0.0071 / +0.0005 (same sign, the
   second one nil). **The verdict does not move**, because it never rested on that clause — the
   resolution ratios are computed from per-cell deltas of the priced cells against the churn floor of
   the unpriced ones (dev 0.5194, test 0.8765), which is the same basis `myna.report` uses. What the
   check buys is a rule: **a quoted macro has to name its basis, and a null has to be re-derived on
   the other basis before anyone calls it an artifact of this one.**
42. **§9.40's defect lives in the driver too, and sweeping it across all 14 batteries found no
   second unaffordable probe.** The entry above capped the test that answers a mutant; it did not cap
   the loop that runs the batteries. Measured 2026-09-28: the first sweep script iterated
   `paraphrase report p5 longctx onnx mlx ordinal scratch` with no per-battery deadline, so one hung
   battery would have sat inside its own uncaught `timeout=1800` and then serialised everything behind
   it — an hour of fan for a summary nobody can read. It was stopped mid-`onnx` at PID 32557 plus its
   `onnxruntime` children (exact PIDs; a broad `pkill` is not available to this repo while anything may
   be training), and parts 2 and 3 ran every battery under a `SECONDS`/`kill -0` watchdog at
   `CAP=2400`/`1800` — a killed battery prints `TIMEOUT`, keeps its partial log, and that partial log is
   the witness that it did not finish. The watchdog itself was checked against a dummy `sleep 600`
   first (rc=143 after 8 s, process confirmed gone) rather than trusted. All 14 then finished inside this
   session: paraphrase **29/29** · report **93/93** · p5 **49/49** · longctx **20/20** · onnx **22/22**
   (1,200 s wall, two graph re-exports per mutant) · mlx **14/14** · ordinal **19/19** · scratch
   **13/13** · gates **17/17** · reproduce **21/21** · browser **24/24** · latency-matched **34/34** ·
   memory plan **44/44** · kaggle bundle **42/42** — **441 mutants, 441 caught, zero `MISSED`, zero
   `BAD-PATTERN`, zero `TIMEOUT`**, summed from the 14 files in `runs/mutation_*.log` (11 of them
   rewritten from these runs; each closes with an `EXIT=0` line naming the wrapper that produced it).
   What this entry is *for* is the negative, because a green sweep reads like nothing happened. The
   static half of the audit predicted it: `grep` for `myna.train`, `run.py` and `--steps` across
   `bench/mutation_*.py` returns only the kaggle bundle, so `MINI_DOSE` is the single place a probe can
   reach training; two batteries handle a timeout (`mutation_browser.py` gives its `node` selftest 180 s
   and turns `TimeoutExpired` into a *caught* mutant with the reason printed — the right shape), and the
   remaining twelve carry `timeout=1800` uncaught, which costs a slow exit rather than a training run.
   So the process-group refactor stayed in the one file with evidence against it instead of being
   spread over thirteen "for consistency", which is the same speculative reach that produced the mutant
   in entry 40. The general form: **a driver that runs a probe owes the probe the same affordability
   rule** — an uncapped wrapper inherits the cost of every uncapped probe inside it. And a sweep that
   finds nothing gets written down anyway, because the alternative is a later session re-running it or
   "fixing" batteries that were never broken.
43. **"Every published table row has a command that regenerates it" was checked by reading
   files, so it never asked whether the command runs.** `bench/reproduce.py --check` makes three
   assertions per row — the witness is committed, the witness still prints the quoted figure, the
   docs print the command verbatim — and all three are answered from disk. A row that publishes
   `--score-loss emd` against a build that renamed the flag passes every one of them: the artifact
   was written while the flag existed, the docs still quote the line, and `git ls-files` still
   returns it. The registry's own history is full of the rename: `--free-gib`, `--warm-start`,
   `--seed` resizing the step (§9.34), and 9g's `MINI_DOSE` cap, which *removed* three flags from a
   printed command line and had to be paid for with a new test and two new mutants. None of that
   would a file-reading checker have seen.
   `--check` now makes a fourth assertion, and it asks the tool: every row whose command names a
   python target (`-m module` or `script.py`, after skipping a `uv run` launcher and a `VAR=value`
   prefix) is run as `<this interpreter> <target> --help`, and every `--flag` the row publishes has
   to appear in the answer. Nothing but `--help` is ever executed, so no row can launch a run from
   inside its own gate — §9.40's rule applied to the registry — and a `--help` that hangs costs
   `HELP_TIMEOUT` and becomes a report, not a traceback (mutant 20, caught by the test that sets
   `HELP_TIMEOUT = 0`). The measured result on 2026-09-28 is a **negative**: 20 of the 27 rows are
   python commands, all 20 answer, and every flag they publish is accepted. The gate costs
   **0.72 s → 5.85 s** in one process with the same cache cold each time (measured by stubbing
   `command_problem` to return `None` and re-timing `--check`), and stayed at **27/27**. Seven rows go unchecked and say so: three `npm` scripts
   (gated for real by `bench/mutation_browser.py`, which runs the node selftest), two `grep` lines
   over a committed log, `uv run pytest tests/test_report.py`, and the `gated-kaggle` row, which is
   prose — feeding it to `shlex` raises on the apostrophe in `eval_laya_real.py's`, which is exactly
   the mutant (`a gated Kaggle row is asked to parse as a shell line`) that the committed-tree test
   catches. Battery: **29/29 caught** (8 new, `runs/mutation_reproduce.log`, 270 s), tests
   `tests/test_reproduce.py` **33**.
   One stale literal surfaced on the way, in the gate's own docstring: `mutation_reproduce.py`
   described a weakened checker as one where "`--check` still says **19/19**", a registry size that
   had been overtaken three ticks ago. It is now corrected to 27/27 in place, because a count inside
   a battery's explanation is read as a claim about what the battery guarantees — and entry 39 is the
   rule that a number like that decays into inheritance.
   G7's prose in `bench/gates.py` names what the registry guarantees, so it gained the fourth
   assertion; README's release table re-renders from that string under a test and SPEC §2.2's row is
   copied by hand, which is the one place this repo has prose that can fall behind generated text.
   Because the file the gate battery mutates changed, `bench/mutation_gates.py` was re-run rather
   than inherited: **17/17 in 240 s** (`runs/mutation_gates.log`).
   The general form: **a check that reads files proves what was written, not what still runs — where
   a command is part of the claim, ask the command.**
44. **Entry 43 named the seam and left it open: §2.2 is the one gate cell typed by hand, and
   nothing read it back.** `bench/gates.py` counted its own registry — `27 registry rows`,
   `58 quoted figures`, `18 rows re-run on this box`, and the rest — in two sentences that a test
   compared with `ROWS`, while SPEC §2.2's G7 row repeated the same numbers in its own words and
   no test and no gate ever looked at them. The words had already drifted apart once — P9 9f
   found `22 rows` written in G7's prose beside a registry of 23 — and the only reason the two
   agreed on 2026-09-28 is that 8e re-synced this row by hand to write §9.43 into it. A number
   that agrees because somebody remembered it is not a check.
   `registry_counts()` now derives all seven phrases from `ROWS`, `count_phrase()` says the
   five-status half once and G7's note interpolates it, and `check_gate` — for any gate whose
   proof *is* the registry, which today is G7 alone — reads §2.2's cell off disk and reds on each
   phrase it no longer prints. So the gate that certifies reproducibility is the first one whose
   own spec row can contradict it. Falsified both ways, in the committed tree and against a live
   count: deleting `58 quoted figures` from the row goes red naming that token, and growing `ROWS`
   by one row goes red on `28 registry rows` — which is what
   `test_specs_g7_row_prints_the_registry_it_actually_has` does rather than what it says, because
   a literal count baked into `registry_counts()` would pass the first and fail the second.
   Battery: `bench/mutation_gates.py` **17 → 21** mutants, **21/21 caught, exit 0** in **334 s**
   (`runs/mutation_gates.log`, re-run rather than inherited because this tick changed the file the
   battery mutates). Two of the four new entries are the theatre shapes §9.43 warns about — the
   comparison that runs and never reports, and the table read from the wrong column so a count it
   cannot find is a count that always matches — and each is caught by that same test.
   `tests/test_gates.py` **28 → 29**, and the older prose test lost its hand-built token list to
   `registry_counts()` so the phrasing has one home instead of three. The asymmetry is disclosed
   rather than tidied: `gates.py` reads its G7 sentences off `ROWS` at import, README's release table
   renders from them and is therefore not evidence (§9.31), and §2.2 stays typed — it is the prose
   a human reads, so it is the prose that has to be asked. What still has nothing reading it back
   is a battery log: `17/17` appears in entries 42 and 43 as what those ticks measured, which is
   why both are dated, and the rule that keeps such a number honest is the existing one — re-run
   the battery whose subject changed, and say which one you re-ran.
   The general form: **a count in prose is either generated from the artifact or read back against
   it — the third option is a number that decays into an inheritance.**
45. **9d was the one open decision in the repo, and it had no numbers beside it.** Two cells
   (`banking77/intent`, `mnli/relation`) answer below their own majority floor and their
   association numbers do not beat their own permutation nulls, which makes them dead rather than
   mis-measured; whether they belong inside G1's scope is a call only the user can make. What could
   be done here was to price it: `bench/scope_pricing.py` subsets the committed report's own `cells`
   and calls `myna.report.macro` and `g1_verdict` on each scope, so the question "does dropping them
   help?" answers as arithmetic instead of as hope.
   It does not, and the reason is the trap in the question. G1 is a level *and* a margin over the
   floor the surviving cells imply, and the cells myna wins are also the ones a majority classifier
   wins, so dropping the dead ones lifts both sides: 0.4893 → 0.5334 on the level, 0.4331 → 0.4666
   on the floor, margin +0.0563 → +0.0668 — **eleven-thousandths of the 0.094 still missing**. "Drop
   their whole sources" is not a third option, it is the same 14 cells (each of those sources
   contributes exactly one cell to the published scope), and even the indefensible cherry-pick —
   every one of the 5 cells below its own floor out, 11 left — lands at 0.5526 / +0.0949, still
   0.147 short of 0.70. **No scope choice in the table makes G1 pass**, so 9d is a decision about
   what the published claim means, and nobody should spend GPU hours on the expectation that a
   scope edit rescues the run.
   The harness rule that makes the table trustworthy is the one §9.37 and §9.41 taught: the
   arithmetic is *imported*, never restated, and the unfiltered row has to reproduce the report's
   own published 0.4893 / 0.4331 / `g1.pass` to the digit or the script refuses to print anything.
   A second implementation of a roll-up is how one run gets two "test macro" numbers, and a scope
   table computed by hand would have been the third. `tests/test_scope_pricing.py` (**18**; **11**
   when this entry was written, and §9.46 is what made the difference visible) holds
   the line in both directions — the row-weighted basis over the same 16 cells is asserted *not* to
   reproduce 0.4893, because a control that any aggregation would pass proves nothing — and one test
   edits the report's published macro and requires the guard to raise. Registry: `scope-pricing`,
   **28 rows / 61 figures**, and this is the first tick where §9.44's read-back bites for real: the
   added row moved G7's own counts (18 → 19 rows re-run here), and §2.2 went red until it was
   re-synced.
   The general form: **an open decision the user owns can still be prepared — price every option out
   of committed artifacts, refuse to tick the box, and hand over the arithmetic with the trap named.**
46. **The scope harness had 11 tests and they could not see two thirds of it.** §9.45 shipped
   `bench/scope_pricing.py` with a test file beside it, and the repo's §7.1 rule ("a test that cannot
   fail is not a gate") had nothing to say about the harness until `bench/mutation_scope_pricing.py`
   ran for the first time: **10 of 24 mutations caught**. The survivors' *shape* is the finding, not
   the tally — every one of the 8 mutations inside the pure `price()` arithmetic died, and **none** of
   the 13 elsewhere did: 0/5 against the guard's individual comparisons (drop the `majority`
   comparison, point the `myna` comparison at the report's `uniform` field, delete the `g1.pass`
   check, widen `TOLERANCE` from 1e-12 to 0.05 — fifty times the smallest difference the table
   exists to measure — the script still refused a
   *doctored macro*, which is the one case the shipped test tried, so a guard that had stopped
   checking was indistinguishable from a guard that worked), 0/5 in how the witness is assembled
   (the `cmd` line, the `published` headline row, a `scopes` dict missing its control row, the two
   cell lists), 0/4 in the printed table — the artifact README and the 9d note actually quote —
   and 0/1 on the missing-`--report` boundary. `main()` had never been run successfully by anything:
   no test asserted it exits 0, no test read what it prints, no test compared its output to the
   witness in `runs/`. A harness that printed three scopes instead of four, or lost the line naming
   its own command, kept a green suite.
   Closed the same tick, in the order §7.1 prescribes: the battery reported, seven tests were written
   against the reported holes (guard per compared field, verdict, the not-on-disk boundary, a clean
   `main()` run whose regenerated JSON must equal the committed witness byte for byte minus its own
   `--out`, and a `capsys` test that each row line carries that row's own cells, floor, margin and
   verdict under the headers that name them), and the second pass is **24/24 caught, exit 0** in
   **17 s** (`runs/mutation_scope_pricing.log`, baseline copy green with 18 tests). Suite:
   **466 passed / 1 skipped in 193.95 s**; `test_cli_help.py` gained the harness and its battery, so
   `--help` cannot become a run.
   The pass also corrected one hand-copied count it did not cause: §3.4's map said **312 passing**
   where the repo now collects 467 — 154 tests behind, in the one section written in present tense
   (§9.44's decay, in a doc that had no read-back over it; the registry's counts do, the map's do
   not, and this tick updated it by hand for the same reason it updated the others). The 51-vs-54
   mutation count this pass first "found" was itself wrong — `grep` against `HEAD` showed the line
   never existed — which is §9.44's rule turned on the instruments reading them: a count is quoted
   from the artifact, and the artifact here is `git diff`.
   The general form: **tests written beside a harness in one sitting inherit the harness's blind
   spots — they pin the numbers that were interesting and skip the plumbing that decides what gets
   printed. Read a battery's survivors by which half of the file they sit in: 8/8 in the pure
   function and 0/13 everywhere else is a sentence about the tests, not about the mutants.**
47. **Tier 0 read the trained model's inputs, and half of decision-v2 never reads the state.**
   P9's three defects were named from accuracies, and an accuracy cannot say *which input* the
   answer came from — while three pillars of the architecture say it should be recoverable: the
   probe reads the question, the option spans carry the labels, the state carries the evidence.
   `bench/diag_question_ablation.py` scores the committed V1-B test split six ways, one forward
   pass per arm per row, through `myna.train.build_batch` and `evaluate()`'s own chunking, with the
   roll-up imported rather than restated (§9.41) — so `as-scored` *is* the statistic G1 was judged
   on, and the guard is that it must reproduce the report's 716 per-question-set accuracies key for
   key before anything prints. It did: **0.4893271976, 716 keys, 0 single-row tie flips**, in
   **186 s of CPU for all six arms** — no GPU, no retrain, no new data. Macros, with `Δ` against
   `as-scored` (floor 0.4331, uniform-floor macro 0.3321): `blank-instruction` **0.3341 / −0.1553**,
   `permute-instruction` **0.4927 / +0.0034**, `cross-source-instruction` **0.4364 / −0.0529**,
   `blank-options` **0.2669 / −0.2225**, `swap-state` **0.4173 / −0.0720**.
   Three readings, each forced by a named arm. (i) **Option descriptions are the answer channel.**
   Blanking them — same `K` spans, text replaced by `option 1..K`, so the pointer head still gets a
   span per label — costs more than any other arm and lands *below* the uniform floor, 0.2669
   against 0.3321: the head is not reading position. That puts a committed artifact behind P9's
   "the mechanism is not broken, the cost function is", and it does not retire the sentence's
   caveat — the option-*order* permutation magnitudes that sentence declines to quote are still
   uncommitted, because Tier 0 asks a different question of the same spans (§9.30 is per figure, not
   per conclusion). (ii) **The cue is read as a bag, per source.** Blanking it collapses the model
   (−0.1553), permuting it within a source changes nothing (+0.0034) because nine sources print one
   cue text on every row — which is why every arm also prints how many strings it actually rewrote,
   260/716 for that one — and substituting a *cross-source* cue of the same option count costs
   −0.0529. (iii) **Half the cells do not read the state at all.** Eight of sixteen print
   `state-Δ` of exactly **+0.000**, and all eight also commit their most-predicted label to a larger
   share of rows than the gold majority holds: an association, not evidence. Three are `noul` cells
   emitting one label on *every* row and finishing on their own floor to four decimals
   (agnews/is_scitech 0.6531, agnews/is_sports 0.8085, agnews/is_world 0.7778); the stateless eight
   are those three plus amazon/stars, banking77/intent, contrastive/decision, mnli/relation and
   sst5/sentiment. The arm is not a no-op — 1,164 of 1,176 states were rewritten, and it moves the
   other eight cells, six of them by more than 0.06 (dbpedia14/category −0.388, imdb/positive
   −0.275, yelp/recommend −0.250, yelp/rating −0.137, trec/answer_type −0.078, agnews/topic
   −0.060).
   One count this entry published first and then re-derived is corrected here rather than deleted:
   an earlier draft said **ten** of the sixteen cells answer from an association. No rule over the
   committed distributions yields 10 — **8** at or below their own majority floor, **3** constant
   predictors, **13** over-emitting their top label, **8** exactly state-invariant, and that last
   pair's intersection is **8**, not 10 (§9.44: a count travels with the rule that produces it).
   Why this is a diagnosis and not a fix. The lever it points at is the input distribution, so it is
   P10's question, priced there — and priced *small*: three of the stateless eight sit inside the
   six cells `--anti-prior` reaches, and the other five already carry train majorities below the
   skew threshold, so an inverse-prior re-weighting has almost nothing to remove for them.
   The gate stays visibly open, and the row that was added to close half of it names the other
   half: this harness scores `model.pt` from the Kaggle-side checkpoint directory, whose *figures*
   are committed and whose *weights* are not.

   ```bash
   uv run python bench/diag_question_ablation.py --run-dir runs/v1b-kaggle-3600b \
       --out runs/diag_question_ablation.json
   ```

   That line re-runs nowhere today, and saying so is the correction this tick exists for. The
   67.7 MB of weights it was measured against lived under `/private/tmp`, which is ephemeral, and
   the directory is gone as of 2026-09-30; the committed `runs/diag_question_ablation.{json,log}`
   still carry the absolute paths the measurement actually used, because a witness is written once
   and read forever and is not edited to look tidy. So the published line now names the directory a
   re-run of the 3,600-update dose writes — `runs/v1b-kaggle-3600b`, the same convention every other
   row uses — instead of a temp path that no longer resolves. `make repro` keeps it a `retrain` row:
   it ties the six arm macros and the 716-key guard to those committed artifacts and asks the tool
   that every flag in the line still exists. What it cannot do is re-derive the table for a reader
   holding only this repository, and after the temp clear it could not do that for this box either.
   That asymmetry is the honest shape of the artifact, not an omission.
   The general form: **an accuracy is a claim about an input. Ablate the inputs before buying
   parameters, and read a per-cell delta's exact zero as a finding rather than as a rounding — a
   cell whose score does not move when its evidence is replaced is not under-trained, it is not
   looking.**
48. **The P10 lever was built, gated 41/41, and nothing on the launch path could pull it.**
   `--anti-prior` went into `myna.train`, `bench/anti_prior_audit.py` measured what it reaches, and
   `bench/mutation_anti_prior.py` closed every mutant it had — across three passes: 25/28 caught on
   the first (the 3 survivors all inside `draw_row_batch`), 28/28 on the second, and after 13 more
   mutants aimed at what the harness *prints*, **41/41 caught** with a baseline green on 35 tests in
   53.90 s (`runs/mutation_anti_prior.log`; 36 min for the 41). None of that made the open GPU
   decision runnable: `kaggle/run.py` — the file the launch docs tell a human to run — had no
   `--anti-prior`, and `campaign.py` generated cells for the `--score-loss` pair but not for this
   one, so 10c could only have been executed by editing the staged bundle on the box. The registry
   could not see the hole either: `--check` asks a tool's `--help` about the flags a row *publishes*
   (§9.43), and until this tick no row published the audit's command.
   **What the 13 printed-face mutants taught is §9.46 run backwards.** That entry's finding was
   that tests written beside a harness inherit its blind spots — 0/13 of `scope_pricing`'s
   non-arithmetic mutants died. Here all 13 died on the first try, and the difference is the seam:
   every one of those tests asserts against output `main()` produces *inside the test*, and the
   scratch repo symlinks `runs/`, so a witness-reading test could not have seen a mutated harness
   at all. Labeling both arms under the control arm's name, printing the natural majority in the
   realized column, reading the mix delta off the wrong arm, counting rows per question set so every
   denominator becomes a set count, and pricing the shared-set batcher instead of the row batcher —
   each is now a red test.
   Three claims that had already shipped were corrected the same tick, all by re-deriving them from
   the artifacts rather than remembering them: **"ten of sixteen cells"** (Tier 0's count is eight —
   the invariant set and the over-emitting set intersect at 8 and no rule over the committed
   distributions yields 10, §9.44); **"~31 GPU hours"** for the P10 pair (that is the *10k* paired
   lane; two arms at the published 3,600 updates cost 2 × 5.618 s/update × 3,600 = **~11.2 h**, and
   the constant came back to `runs/v1b_kaggle_3600b.train.log`'s `step 3599 … 20225s`); and the
   flag's own help text, which promised that "on the 10 cells whose prior is already flat it changes
   nothing" — the audit measures nine of those ten moving anyway (down to 0.163, contrastive/decision
   0.507 → 0.337) and **one sharpening** (yelp/rating 0.203 → 0.259), because a yelp row answers two
   questions and the weight is computed on the other one. A sentence about what a mechanism does
   *not* reach is as much a claim as a number, and was as unverified.
   Two more, about the record rather than the measurement. **`EXIT=0` in that log is inferred, not
   watched** — the battery ran under `nohup`, the shell that would have echoed `$?` was gone, and
   the pass body was written from the absence of a `SURVIVED:` line. Correct as reasoning, wrong as
   an instrument: a battery is a measurement, so it gets launched `; echo EXIT=$? >> log`. And the
   dedupe: `tests/test_anti_prior.py` defined **four test functions twice** — 109 duplicated lines,
   two byte-identical blocks — because a rewrite this session inserted the corrected block instead
   of replacing the old one. Python kept the last definition, so the file collected 35 tests while
   defining 39, and the two dead copies still *looked* like gates: one of them carried the
   reduced-dose ranking assertion this tick had deliberately removed, i.e. the file's visible face
   contradicted its shipped reasoning. `tests/test_kaggle_bundle.py` now fails on a duplicate `def`
   anywhere in `src/`, `tests/`, `bench/` or `kaggle/`, which is the class, not this instance.
   The general form: **a lever the trainer has and the launcher lacks is not a feature, it is prose
   — plumb it through the entrypoint, generate the paired cell, and give the harness a registry row
   in the same tick, because the gate that asks the tool about published flags cannot see a flag
   nobody published. And a green suite is not a claim that the suite is the one you think it is:
   count definitions as well as collected tests.**
   Then §7.1 was paid against the launcher (§5 P10's 10b): `bench/mutation_kaggle_bundle.py` 42 →
   **48 mutants, 48 caught, exit 0**, six over the P10 plumbing. One of those six is a finding of the
   same family as §9.46 and it was measured before being written: the test that proves the two GPU
   cells differ in *exactly one flag* proves it by normalizing `--anti-prior on` and `--anti-prior off`
   to one token — so the mutant that hardcodes `off` in the generator, i.e. the bug that would spend
   ~5.6 GPU-hours to re-run the control arm twice, went green. Running that mutant against the test
   with the added `assert f"--anti-prior {arm}" in cell(name)` line deleted printed **32 passed**: the
   normalization that makes two things comparable is what deletes the difference the check is about, so
   a comparison test must assert the raw values on both sides *and* the sameness of everything else.
   The battery's own wall time is also now stated as bounded-by-polls rather than stamped, because this
   driver was launched without `ts` per line — §9.40's lesson one step further out, into the wrapper.
49. **A commit subject can be a published figure too, and this one was wrong the moment it was
   typed.** `e9c73c1` opens `10c: §7.1 paid against the launcher`. 10c is the GPU pair, and it is
   still unspent and still `- [ ]` in both §5 and TODO — what that commit did was finish 10b's gate
   obligation for the launch path. Nothing in the tree was wrong: the plan's checkboxes, the
   registry row and the launch doc all say the pair has not run. The record that drifted was the
   history, and history is the one artifact no `--check` reads. Not amended (a local rewrite is
   still a rewrite of something already cited); corrected here, so anyone reading `git log` and
   then §5 finds this entry between the two. The rule: **a tick ID in a subject is a claim about
   the plan's state, so it gets read back against the checkbox before the commit is made** — the
   same §9.44 discipline, applied one file earlier.
50. **The one artifact this project cannot rebuild was kept in a temp directory, so shipping
   weights stopped being an intention and became a process.**
   *Superseded on its central claim, same day: the weights were recovered from `~/Downloads` and the
   recovery was hash-verified — §9.53(i)–(ii). What that entry got wrong is not the risk model (a
   temp directory is still the wrong home) but the word "filesystem-wide", which described a search
   of `/tmp` and `/private/tmp`. Everything below is kept as written because it is what was true when
   it was written.*
   V1-B's `model.pt` and `tokenizer.json` lived under `/private/tmp/kgwork`. That directory is gone
   — `ls` on the path fails and a filesystem-wide `find` for `v1b-kaggle-3600b` returns nothing —
   and what survives of the run is only what git holds: `runs/v1b_kaggle_3600b.{metrics,report}.json`
   and `.train.log`. So the *level* is quotable (0.4893 test macro) while the *weights* are not, the
   asymmetry §9.47's tail had already been forced to write down for the Tier 0 row. The price of the
   loss is bigger than one number: every arm that wanted to warm-start from that checkpoint (§9.41
   names the same gap) and every re-derivation §5 P0 still lists as open is now unspendable. §9.49's
   "still unspent" survives as a claim about the GPU quota, but on the local lane this entry
   supersedes it.
   The regeneration 10c needed anyway got the durability rule attached in the same pass. Both arms
   run through the canonical launcher on the M5 — `runs/pair_driver.sh` calls `kaggle/run.py`, which
   prints the expanded command into `runs/antiprior_off_s0/train.log`, and the two arms differ by
   `--anti-prior` alone. `runs/ckpt_backup.sh` then ships each arm's weights to a GitHub Release with
   nobody awake for it: the first stable `model_last.pt` as a mid-run insurance copy, then the final
   `model.pt` plus `tokenizer.json` once the driver writes that arm's exit line. Both are box glue
   under `runs/`, which `.gitignore` keeps out of the tree — what is published here is the rule and
   the launcher command, not a wrapper with this laptop's absolute path in it. The uploader was
   proven before it had anything real to upload — it created a throwaway release, read the asset back
   at its exact **40,000 B**, and that release was deleted — because an unattended process that
   publishes is not allowed to fail for the first time at 07:00. And the run's own weights sit in the
   repository working tree rather than under `/private/tmp`, which is already a different failure
   class: a reboot does not clear it. Only a disk fault or a `git clean -xdf` does, which is what the
   insurance copy is for — the first one is `antiprior_off_s0`'s step-250 `model_last.pt` at
   **203,229,625 B**, roughly three times the final `model.pt` because it carries the optimizer and
   scheduler as well as the weights, and it shipped: the watcher logged the file stable at 00:53:16
   and the release went live 19 minutes later, which GitHub stamps `publishedAt
   2026-09-30T19:27:29Z` — the two clocks agree once the box's UTC+05:45 is applied, and saying so
   here because an hour reading a mismatched timestamp would conclude the artifact was invented.
   That log is committed as `runs/ckpt_backup.log`, since the entry is making a timed claim and §9.30
   does not exempt a process just because it is a shell script. GitHub's own asset digest is the
   checksum a reviewer should verify against — `model_last.pt` on that release carries
   `sha256:db124b8fdb6a04afa5a626e20acdac9470c66c73d520f368eb6ff0b1b4281cdf` — so "these are the run's
   weights" is a download-and-hash claim, not a promise. It is *not* checkable against this laptop:
   `model_last.pt` is a rolling snapshot, the step-500 write landed at 01:20 and re-hashed the local
   path to `770bf49afe30b96264b34f5cc3ae29ee3ba1f937e512942a0d1fd81d83a928e5`. Same 203,229,625 B,
   different weights — which is the reason the watcher ships on a size-and-stability trigger and not a
   filename, and the reason a step-250 checkpoint has to leave the machine before step 500 arrives.
   That is also why a staged run dir reports its step as *not
   recorded*: `train.py` writes `model.pt` as `{state_dict, cfg, temperature}` and puts the step only
   in `model_last.pt`.
   Three things this run must not borrow, said here so the next reader does not try. (i) It is not a
   Kaggle measurement, and the wall price belongs to this lane: the ~11.2 h in §5 and §9.48 is the
   T4's price and 0 hours of the 30 h/week quota are spent. This box reads **6.52–6.82 s/update** in
   the windows between snapshots, 7.27 in the one that carries the step-250 eval and checkpoint
   write, and 6.75 averaged over the first 480 updates — the log stamps one line per 60, so this is
   a rate read off a live log rather than a stopwatch, and it is a *mid-run* rate from 480 of 3,600
   updates that the completed log will supersede (§9.48's bounded-by-polls). Against the T4's
   measured 5.618 that is 1.20×, i.e. ~6.75 h per arm and ~13.5 h for the pair. (ii) The pair is a
   control only if it reads the corpus V1-B read, and
   neither end prints a corpus hash — the mount carried a
   `SOURCE_SHA256.json` that `runs/v1b_kaggle_3600b.train.log` names without quoting. What closes the
   chain is statistics derived *from the data*, identical on both sides: `train question-sets:
   17112`, `params: 16926848`, and the memory plan's `p95 267 tokens` row width — and
   `runs/v1b_kaggle_3600b.metrics.json` records the same two derived values as its own keys,
   `state_tokens_p95: 267` and `batch: 10`. On this side `data/decision-v2-pilot/train.jsonl` re-hashes
   on 2026-10-01 to the `sha256_train` pinned in `upstream_manifest.json` —
   **`5f5f93ff7ca03e4522dbe5bdff50e0a92cd5467d613d32d6a5988b8a00870a64`**, `MATCH`. One wording
   difference that is not a difference: V1-B's plan line reads `--batch 32 exceeds it, using 10` and
   this arm's reads `batch 10 fits`, because the requested flag differs while the *effective* batch is
   10 in both. The honest remainder is that the Kaggle end of the corpus chain is a filename.
   (iii) The arm is not a bitwise replay of V1-B and must not be sold as one. At step 250 it prints
   `dev-mid acc 0.3682` where V1-B's log prints **0.3401** — same seed, same corpus, same effective
   batch, different device, so the float trajectory differs update by update. The pair therefore
   prices `--anti-prior` against a *fresh* control on this box; it says nothing about whether that
   control reproduces 0.4893, and a reader who wants that comparison owes it to the T4 lane. The
   check worth reading when each arm's `metrics.json` lands is `paraphrase_draws`: V1-B records
   **225547** over 17,112 sets, and an arm that matches it has pushed the same data through the same
   re-wording plan — a tighter fingerprint than any single line in a log.
   The general form: **a checkpoint is an artifact only where git or a release can be read back.
   Whatever took GPU hours to make gets a remote copy in the same pass that produces it, the
   publishing path gets proven against a throwaway asset before the real one exists, and a rate
   lifted from another machine's log is labelled as the price of that machine rather than of the
   result.**
51. **"The suite is green" was a sentence about this laptop, and CI has never run a single
    test.** Every count published since the push — 556 passed, 1 skipped, and the §3.4 line that
    carries it — was measured here, where `mlx` is installed. On `ubuntu-latest` it is not: `mlx`
    is an optional extra (`[project.optional-dependencies] mlx = ["mlx>=0.20"]`), and CI's install
    step was bare `uv sync`. So `tests/test_mlx_int8.py` line 22 did a plain
    `import mlx.core as mx` at module scope, and `gh run view --log-failed` on all **7** runs this
    repo has ever recorded — 36771444494 back to 36749769385, every one `failure`, each finishing in
    29–48 s — prints the same four lines:
    `ERROR collecting tests/test_mlx_int8.py` → `ModuleNotFoundError: No module named 'mlx'` →
    `Interrupted: 1 error during collection` → `Process completed with exit code 2`.
    **Zero tests executed**, on every run, since the repo went public. The wall time is the
    tell you can read without the log: those 557 items take **294 s** on this box, so a job that
    finishes in 29 s *including* checkout and dependency install ran none of them — and the 556-green
    sentence next to a red Actions tab is what a reviewer reads first.
    The bug is one line wide and the distinction is not cosmetic. A module-level `import` of an
    absent package raises during *collection*, which aborts the whole session;
    `pytest.importorskip` raises `Skipped` and costs one file. The file that already got it right —
    `tests/test_mlx_parity.py` — imports `mlx` *inside* each test body, which is why it never took
    the suite down with it.
    The probe was falsified before it was believed, twice. The first blocker used
    `find_spec`'s predecessor `find_module`, removed in Python 3.12, so it blocked nothing and
    reported **11 passed**: a green probe run is not evidence that the probe ran. Its replacement
    raised a bare `ImportError`, which `importorskip` re-raises, so the suite still died at
    collection — a *different* red for a *different* reason, and easy to misread as "the fix did not
    work". Only `ModuleNotFoundError(msg, name="mlx")`, which is what CPython itself raises,
    reproduces the runner. Lesson: the test of a test harness gets a negative control too.
    With a blocker that works, the first measured rehearsal was **8 failed, 540 passed, 2 skipped**,
    and the 8 split 5 + 3 in a way that turned out to be the whole decision: the 3 are
    `tests/test_mlx_parity.py`'s own tests, which genuinely need the engine, and the **5 — 4 in
    `tests/test_gates.py` and 1 in `tests/test_reproduce.py` — are all** `bench/reproduce.py --check`,
    which prints
    **26/30 rows hold** there with `! the module behind the command does not answer --help (rc=1):
    ModuleNotFoundError` on four rows — `arch-params`, `mlx-int8`, `mlx-int8-keepgate`, `mlx-kernel`.
    That is §9.43's runnability assertion doing its job on a machine that lacks the module — and the
    inference drawn from it, "**so CI installs the extra**", was the claim CI then falsified.
    Run 36776477857 got past the install step and printed, from the runner:
    `ImportError: libmlx.so: cannot open shared object file: No such file or directory`.
    PyPI's Linux `mlx` wheel ships the bindings and not the library (`mlx-metal` is marked
    `sys_platform == 'darwin'`), so on Linux MLX is **installable and unloadable at once**, and
    `pytest.importorskip` re-raises that bare `ImportError` instead of skipping — the same
    second-failure shape the probe had already taught, now delivered by the real runner instead of a
    rehearsal. Wheel filenames on a lockfile are not a measurement of `import`: the lock's
    `cp313-cp313-manylinux_2_35_x86_64` entry was real, and what it proved about `import mlx.core`
    on that runner was nothing. That is the whole argument for rehearsing a CI change on a branch —
    the PR's run cost 28 s and returned a fact no local probe could have produced.
    What the runner can therefore never have is MLX, so the gate had to learn to tell three things
    apart instead of conflating them: **absent** (skip the file), **present but unloadable here**
    (also skip — `tests/test_mlx_int8.py` and `test_mlx_parity.py` now wrap the import in
    `try / except ImportError` + `pytest.skip(allow_module_level=True)`, measured both ways with the
    two blockers, each printing `MLX unavailable on this box: …`), and **a module that exists here
    and is broken** (red, always). The registry got the same distinction in
    `bench/reproduce.py`: `mlx_unloadable_here(tail, platform)` forgives exactly two error strings
    (`No module named 'mlx'`, `libmlx.so`) and only off Darwin, and the row then prints
    `… cannot be imported on linux (MLX is Darwin-only at runtime), so its --help was not asked here`
    as a note rather than a pass — `python -m bench.bench_mlx` failing to load on a Mac is still red,
    and so is `No module named 'numpy'` anywhere. Five arms in `tests/test_reproduce.py` pin that
    table, and §7.1 was paid against them: **4 mutants, 4 caught** (drop the Darwin guard → 2 fail;
    broaden to any `ImportError` → 2 fail; unwire the note → 1; unwire the limit → 1), file restored
    and byte-compared afterwards.
    Three details worth keeping, because each was written wrong once inside this tick. (i) The CI
    comment first said "two registry rows (`arch-params`, `mlx-kernel`)". It is four, and the way to
    know is `grep -v '^ok'` on the check's own output, not a remembered number — §9.44's rule applies
    to a comment in a workflow file exactly as it applies to a table. (ii) README's architecture line
    "proven numerically identical, in float64, in CI" names
    `tests/test_trunk_numerics.py`, which is torch-only at `atol=1e-8` and so never needed `mlx` —
    the claim was never about the file that broke, yet it was still false, because on that runner
    nothing ran at all. It becomes true for the first time with this commit, which is the reason the
    fix went on a branch and through a PR before it touched `main`: the claim is only worth anything
    once the Actions tab shows it, and §9.30 does not exempt a badge. (iii) README's Quickstart said
    bare `uv sync` and `uv run pytest`, and the first draft of this tick "fixed" it by adding
    `--extra mlx` — which off macOS installs a package that cannot be imported, i.e. it would have
    handed the next contributor the same collection error through the documented recipe. The
    corrected line is `uv sync` everywhere, `--extra mlx` on Apple silicon, and README says which
    rows each choice leaves unread.
    Counts, each measured on the tree it describes. The MLX-hidden rehearsal reads
    **5 failed / 540 passed / 3 skipped**, and note what that rehearsal *cannot* show: on this box
    `sys.platform` is `darwin` inside `bench/reproduce.py`, so the platform-note path never fires
    however thoroughly `mlx` is hidden — the 5 failures (4 in `test_gates.py`, 1 in
    `test_reproduce.py`, all `bench/reproduce.py --check` printing **26/30 rows hold**) are the
    correct result here, and the Linux behaviour is reached only by the monkeypatched arms and, for
    real, by the runner. That asymmetry is why the gate landed with tests that fake the platform
    rather than with a rehearsal that could not have caught it. Against the final tree on this box:
    **561 passed, 1 skipped**, 562 collected, in 283.78 s — the +5 being `test_reproduce.py`'s new
    arms — with MLX loading and both MLX files running for real. (§3.4 carries the re-read: the same
    split at 292.95 s on 2026-10-01, so the counts held across the 10c run and only the seconds moved.)
    The general form: **a green local suite is a claim about one machine, and "the wheel exists" is a
    claim about a filename. Where a check re-runs a published command it inherits that command's
    optional dependencies, so it must be able to say *why* it could not ask: a dependency this
    platform cannot load becomes a printed note, a dependency that went missing or a flag that went
    away stays red, and no optional import sits at module scope unguarded in a test file.**

52. **CI ran the suite for the first time and printed 13 red, and not one of them was findable
    from this laptop.** Run 36779801999 (commit `1282457`) executed 519.31 s of tests on
    `ubuntu-latest` and reported **13 failed, 520 passed, 20 skipped**. The collection defect is
    dead: the 9 MLX tests skipped on their own, the four MLX registry rows printed the platform
    note instead of failing §9.43, and `make secrets` was reached at all for the first time. That
    is §9.51 working on the machine no rehearsal could reach — and it is also the moment the
    sentence "the suite is green" got its second half, because the 13 are thirteen claims about
    *this* box that a Linux clone falsified. They split four ways:
    - **5 from one missing package.** `bench/quantize_int8.py` imports onnx at module scope, so its
      `--help` could not be asked, which made the `int8-quantize` row red, which made gate G7's
      proof ("every published row binds to a command that still answers") red, which failed
      `test_reproduce.py::test_the_committed_tree_passes` and four `test_gates.py` arms. Fixed by
      installing `--extra browser` in CI — and by README, because the documented contributor recipe
      was `uv sync` + `uv run pytest`, i.e. the exact two lines that produce this red. That is
      §9.51's third recurrence of the same shape: a check that re-runs a published command
      inherits that command's dependencies, and the install text has to carry them.
    - **5 from a fresh clone having no weights.** `runs/myna-v0/model.pt` is gitignored and travels
      as a Release asset, so `bench/risk_coverage.py`, `bench/eval_needle.py` and the two
      `test_latency_matched.py` runs died on `FileNotFoundError`. CI now downloads
      `gh release download v0-checkpoint` before pytest, deliberately instead of skipping those
      five: §5 P0's claim that the weights ship as a Release had never been executed by anything,
      and a renamed or missing asset now fails one loud step rather than hiding behind five
      legitimate-looking skips.
    - **2 from the CLI-surface test meeting an unloadable MLX.** `test_cli_help.py` asserts every
      bench script renders `--help`; `bench/bench_mlx.py` and `bench/diag_mlx_int8_gem.py` import
      the engine at module scope, so off macOS there is no usage text to render. Same
      absent/unloadable-vs-broken rule as §9.51, new file: a `macos_only` marker on exactly those
      two params, with the other 22 rows of the table still required to answer anywhere.
    - **1 from a test that read the laptop.** `test_explicit_device_never_rewritten` asserted
      `resolve_device("mps") == "mps"`, which is "this machine has MPS" wearing a policy's clothes.
      Availability is now monkeypatched for both accelerators, so the test reads the policy — an
      explicit name comes back unchanged — on any box. Its neighbour already made MPS unavailable by
      monkeypatch rather than "assumed either way", which is the argument for the shape.
    The rehearsal that justified the last two was run *before* pushing, and it carried its own
    negative control: a CPU-only simulation (`sys.platform = "linux"`, both `is_available()` probes
    forced false) that **prints** the state it simulated and aborts if the patch did not take, then
    reports **52 passed, 2 skipped** for those two files with the skip reason named. §9.51's probe
    failed twice because a green probe is not evidence that the probe ran; this one prints its own
    blocker. What the rehearsal still cannot reach is the Linux *install* — see below.
    Labelled honestly, and then measured: onnx and onnxruntime being loadable on Linux was a
    **projection**, made from the fact that Linux is onnxruntime's primary target rather than from a
    wheel filename, and run 36781802262 is the measurement that closed it. **CI is green for the
    first time in this repository's history**: that run printed **545 passed, 8 skipped, 1 warning
    in 572.45 s**, exit 0, with `make secrets` reached and reporting **222 blobs scanned of 222 in
    the db, 0 unreachable, 0 matches for 9 shapes** — a fresh clone's object database is a fifth the
    size of this laptop's (222 vs 574), which is the denominator assertion doing its job rather than
    a smaller scan. The `int8-quantize` row's `--help` was asked and answered on Linux, so the
    projection's failure branch — `int8-quantize` joining the four MLX rows under a platform note —
    did not fire, and the extra stays where a real dependency belongs.
    The two boxes' reported totals differ by exactly 9 items (562 collected here, 553 reported there)
    and the reason is the §9.51 design rather than a lost test: a module-level `pytest.skip` collapses
    a whole file to **one** reported skip, so `test_mlx_int8.py` (8 tests) and `test_mlx_parity.py` (3)
    become 2 skip lines on a box that cannot load MLX — eleven items in, two lines out, a net −9. That
    collapse is the same shape the hidden-`mlx` rehearsal printed locally as `2 skipped`, so the gap
    is arithmetic of the skip design and not a collection failure. The 8 skips the runner reported
    are now named by the run itself, because CI moved to `pytest -q -rs` — a green that cannot say
    *why* it skipped is how §9.51's sentence survived seven red runs:
    `SKIPPED [1] test_mlx_int8.py:31` and `[1] test_mlx_parity.py:39` (both "MLX unavailable on this
    box: No module named 'mlx'"), `SKIPPED [2] test_cli_help.py:60` (the two `macos_only` harnesses),
    `SKIPPED [1]` at each of `test_report.py:330 / :480 / :526` ("the v1-rich … artifact is not on
    disk"), and `SKIPPED [1] test_upstream.py:131` (needs `KEV_ROOT` plus `datasets`). Four of the
    eight are this tick's design, three are external artifacts a clone cannot have, one is the KEV
    gate that has always skipped here too — and that is the whole list, read off the log rather than
    inferred, which is what makes 545 green mean 545 and not 545-plus-something-quiet.
    The general form: **a clone is a different machine, and green is a claim about whichever one ran.
    Fixing a defect that hid every test does not make the suite green; it lets the suite report the
    thirteen things that were never tested anywhere but here.**
53. **The weights §9.50 recorded as destroyed were sitting in a Downloads folder, and the fresh
    control built to price one flag priced a +0.0446 improvement instead.** Two separate findings
    landed in one morning on 2026-10-01, and the first one reverses §9.50 outright.

    **(i) "Gone" was a statement about one directory, not about the artifact.** §9.50's whole entry
    is that V1-B's `model.pt` lived under `/private/tmp/kgwork`, that the directory is ephemeral, and
    that this was "the one irreversible risk" — and the check that closed it looked at
    `/private/tmp` and reported the truth: nothing. What nobody checked was that the same bytes had
    *also* been downloaded through the notebook's Output tab, because that is the ordinary way to get
    a file off Kaggle and it lands in `~/Downloads`. `results (1).zip` (285 MB, dated 04:36) holds
    `runs/v1b-kaggle-3600b/{model.pt,tokenizer.json,metrics.json,model_last.pt}`. So the risk was
    already closed by an action taken for another reason, and the entry that warned about it stayed
    true-as-written and false-as-current for hours. The general form: **before publishing that an
    artifact is unrecoverable, enumerate every place a copy could be, not only the place it was made.**
    A `find` over the directory that produced it is not a search for the artifact.

    **(ii) Recovery was proved by three hashes, not by a filename.** Extracted to
    `runs/v1b-kaggle-3600b/` (durable, inside the working tree, `*.pt` still gitignored), the files
    are: `model.pt` 67,734,997 B at sha256 `0e73a8f07edc87d8…`, `tokenizer.json` 526,662 B at
    `0252c24627eeebc2…`, `metrics.json` 295,060 B at `eb70f25bd775092b…`. The first two prefixes are
    *already in the tree*: line 2 of the committed `runs/diag_question_ablation.log` reads
    `model 0e73a8f07edc · tokenizer 0252c24627ee`, and that log is the witness for the 0.4893 macro
    and the Tier 0 arm table. The third file `cmp`s byte-identical to the committed
    `runs/v1b_kaggle_3600b.metrics.json`. Then `bench/diag_question_ablation.py` was re-run on the
    recovered bytes (new witness `runs/v1b_recovered_tier0.json`): **guard green, 716 committed keys,
    0 tie flips, macro 0.4893271976, and 0 field-level differences across all 16 cells** against the
    committed JSON. A folder called `v1b-kaggle-3600b` is a claim; 716 reproduced keys are the fact.
    Both assets are on the `v1b-checkpoint` Release, and the release was verified by downloading it
    back and re-hashing (`gh release download` → same two digests), because §9.18's rule is that an
    upload claim needs the round trip.

    **(iii) The `off` control finished, and its number is not what the pair was built to look for.**
    `antiprior_off_s0` reports **test macro 0.5339** (floor 0.4331, uniform 0.3321) over the 16 of 16
    cells both harnesses score, against V1-B's 0.4893 — **+0.0446 with the flag off**. That is exactly
    the confound §9.50 said the fresh control existed to remove: 0.4893 came off the code as it stood
    before the data-loader work, so the difference belongs to the data path, not to `--anti-prior`.
    G1 is still not met on this arm either — 0.70 target, and the "+0.15 over the floor" clause
    observes **+0.1008** — and laya's 0.667 leaves the gap at −0.133. Three secondary readings, all
    from the committed artifacts: the dose held its coverage (`paraphrase_draws` 207,411 over 3,314
    updates = 62.59 per update, against V1-B's 225,547 over 3,600 = 62.65, a 0.1 % difference, so the
    short arm is not a short sample of phrasings); the dev-mid series flattened rather than rising
    (13 points from 0.3682 to 0.4332, OLS **+0.0231 per 1,000** over all of them, +0.0166 over the
    last five, +0.0030 over the last three — V1-B's tail was +0.0072 and *still climbing*, §9.38, so
    this arm is nearer its plateau than V1-B was to hers); and the wall price is
    **6.719 s/update** over 0→3,300 (22,173 s stamped), 1.20× the T4's 5.618.

    **(iv) It stopped at 3,314 of 3,600 because the laptop went to sleep, and the guard cannot tell
    that from a memory spiral.** `pair_exit_codes.txt` records `train exit=1`; `metrics.json` records
    `stopped = "step time 596.40s > 3.0x median 6.58s of the last 20"`. The guard is working as
    designed — and what it detected was `pmset -g log` showing back-to-back 'Maintenance Sleep'
    entries on battery at 08:08:17 (244 s), 08:12:40 (222 s), 08:16:41 (194 s) and 08:20:03 (133 s),
    with the run ending at 08:22:17. `src/myna/train.py` stamps updates with `time.time()`, which
    advances while the process is frozen, so a nap is indistinguishable from allocator paging. The
    compute itself was healthy to the last step: across all 56 stamped intervals the median is
    6.633 s/update and the maximum 7.267. **Open defect, deliberately not fixed in this tick:** the
    monotonic clock is the right instrument (`mach_absolute_time` stops across sleep on this box), but
    changing the trainer mid-pair would mean the two arms ran different code, which is the one thing
    the pair exists to avoid. Also note the guard is inert for its first 20 updates
    (`len(step_times) >= window + 1`), which is the only reason the `on` arm survived an 86-minute
    nap inside its first 60 updates.

    **(v) Two premises in the handover were checked against the artifacts and one of them is wrong.**
    The instruction to "adapt `--free-gib 9.0` for the T4's 16 GB" would have changed the dose: the
    recovered V1-B `metrics.json` records `mem_plan_free_gib = 9.0` **from the Tesla T4 run itself**
    (`device: Tesla T4`, `free 14806 MiB of 14911 MiB`), so 9.0 is not a Mac value at all — it is the
    pin both boxes already share, chosen so the memory plan clamps to batch 10 everywhere. It stays.
    The other premise, that the Kaggle launcher scripts had been deleted, is half wrong:
    `kaggle/run.py`, `kaggle/campaign.py` and `kaggle/package_dataset.py` are all present and
    `campaign.py` still writes a pushable kernel; what is gone is the notebook directory, so the new
    `kaggle/notebooks/antiprior-on-t4/` is hand-built on the same rule — every arm through
    `kaggle/run.py`, and its `--dry-run` printed into cell 7 so the command that ran is in the output.
    Verified against the Mac arm's own log: the two commands differ in `--device` and the output path
    and in nothing else.

    **(vi) The Mac lane is stopped; the T4 lane is blocked on the account, and the first cause named
    here for that block was a guess that measurement then killed.** The `on` arm was killed at step
    540 by exact PID (dev-mid 0.3789 at 500; its log is committed as
    `runs/antiprior_on_s0.train.log` so the abandoned leg is a record rather than a silence), then the
    driver and the backup watcher exited with it. The fresh token authenticates — `kaggle kernels list
    --mine` answers, account `aashish124` — and the kernel pushes and runs, but every session comes
    back CPU-only: `torch 2.10.0+cpu`, `cuda build None`, `device_count 0`, no `nvidia-smi` binary,
    no `/dev/nvidia*`, `ACCELERATOR_TYPE None`. What is *not* the cause, each measured rather than
    assumed:
    - **Not the metadata.** `kaggle kernels pull -m` on the pushed kernel returns the *server's* record
      with `"enable_gpu": true` and `"machine_shape": "NvidiaTeslaT4"` in it. The field arrives and is
      stored; nothing is silently dropped between the laptop and Kaggle.
    - **Not the CLI's request shape.** Five variants, each its own completed run, all CPU: metadata
      `NvidiaTeslaT4`; metadata `NvidiaL4`; `--accelerator NvidiaTeslaT4`; `--accelerator
      NVIDIA_TESLA_T4_X_2` (the value the September run used); and `docker_image_pinning_type:
      "latest"` with `docker_image` removed, which is Kaggle's documented fix for a pinned image
      overriding an accelerator request (kaggle-cli #1197, maintainer reply 2026-09-17). Per #1196
      `NvidiaTeslaT4` *is* the identifier for the editor's "GPU T4 ×2", so the value asked for was
      never wrong.
    - **Not the quota.** `kaggle quota` prints `GPU 0.00h used 30.00h remaining, refreshAt
      2026-10-03` — thirty hours unused. So the sentence I first wrote here, that this "means the
      account has no GPU entitlement until phone verification", is unsupported: a quota row is a
      budget, not a provisioning decision, and nothing I can run from this box reaches the gate.
      **The gate itself stays OPEN** — something between the stored `enable_gpu: true` and session
      scheduling denies `aashish124` hardware, and the CLI has no verb that reads it.

    The comparison that does locate it is the same recipe on the other account.
    `runs/v1b_kaggle_3600b.train.log:120–121`, pushed by `kaggle kernels push` for
    `aashishkumarmahato01`, prints `device: Tesla T4` and `free 14806 MiB of 14911 MiB` — a real T4
    with 14.5 GiB free — and trained 3,600 updates on it. Same CLI 2.2.4, same
    `kernel-metadata.json` shape that `kaggle/campaign.py:249–256` writes, same `--accelerator` route,
    same repo. The failure is specific to `aashish124`, and both closes are the user's, not mine:
    **look at the Accelerator dropdown in the notebook editor for `aashish124`** (a greyed-out T4 is
    the witness, and it is only visible in a browser), or **re-mint a token for
    `aashishkumarmahato01`**, whose GPU path is already committed as a log line.

    Cell 1 of the real notebook still refuses to train on CPU rather than run ~10× longer and print
    numbers that look identical, and that guard was proved by firing it: kernel version 2 ends in
    ERROR with exactly its message. **Its message now states the measured denial and the two closes
    instead of the phone-verification guess**, and firing it again is how version 3 of
    `https://www.kaggle.com/code/aashish124/antiprior-on-t4` ends — that URL is the launch handle for
    Task 4's arm, and the version-3 log prints `torch 2.10.0+cpu | cuda build None | device_count 0`
    above the refusal, so the CPU denial is recorded in the same artifact as the guard that catches it.
    Training resumes the moment a session holds a T4. Kernel **version 4** is what a browser
    "Save & Run All" would then execute, and it hardens the one cell that stands between the clone and
    5.6 hours: cell 3 used to `grep -n -B1 "def ask"` and *show* the decorator, which is how a silent
    miss survives to the end of a run — it now raises `THE CLONED TREE LACKS THE ask() FIX` when no
    `@torch.no_grad()` sits above `def ask(`. That guard was mutation-checked before it was pushed: on
    the real `src/myna/engine.py` the regex resolves with the group set, and on a copy with the
    decorator line deleted it resolves empty, so the assert has one passing and one failing arm on
    this box rather than only the happy path. (Version 4 itself still ends in ERROR at cell 1 — the
    CPU guard — so the new assert has not yet run on Kaggle, and cannot until the gate in this
    paragraph opens.)

    **(vii) The pair's judging bar has to move, and the move is stated before the result.** The plan
    was `off` vs `on` on one machine at one dose. What exists is `off` on an M5 at 3,314 updates and,
    when it runs, `on` on a T4 at 3,600 — so that pair prices the flag *and* the machine *and* the
    dose, which is the exact shape §9.50 rejected. Two consequences, both cheap to state now: the
    verdict stays on Tier 0's **emitter count over the 16 cells** (how many cells stop answering a
    constant label), not on the macro, because the emitter count is a property of the batching rule
    rather than of the box; and if the macro is quoted at all it is quoted as provisional until the
    `off` arm is re-run on the T4 — 5.6 h of the 30 h/week quota, which is the user's call to spend,
    not a step to take quietly.

54. **A Release title claimed a dose the run never did, and closing it took a hash round trip plus a
    registry row that refuses to overwrite its own witness.**

    **(i) The off control's weights are on a Release, proved the only way that means anything.**
    `runs/antiprior_off_s0/model.pt` (67,733,781 B) hashes to
    `30f0fa937e57e8ce8b21c58315dc7e800151a0c95e003dc0930e0f75d588f1e5`, which is the digest
    `antiprior_off_s0-weights`'s own body already printed, and `gh release download` +
    `shasum -a 256` returns that digest at that byte count, plus `tokenizer.json` at 526,662 B /
    `0252c24627ee…` — the same tokenizer digest as V1-B, which is what a shared vocab should do. So §5
    P10 10c's "its weights are on the release" is now a witnessed sentence rather than the shape of
    sentence §9.50 wrote before it had to be retracted.

    **(ii) What was wrong was the label, and a label is where a stranger reads.** Both the release
    name and its first line said `3,600 updates`. `runs/antiprior_off_s0.metrics.json` says
    `last_step: 3314` beside `steps_requested: 3600`, and `myna.report` prints the VOID line over those
    bytes — so the title wore the request as if it were the achieved dose, the exact category §9.53(iv)
    is about, in the one place with no surrounding prose to correct it. The tag now reads *"antiprior_off_s0
    weights (3,314 of 3,600 updates — the off control, stopped by its own sleep guard)"*, and the body
    carries the 596.40 s-against-6.58 s guard reading, the three digests, and the curl-and-re-hash
    commands a fresh clone needs. No bytes were re-uploaded: the assets are untouched, which is why the
    hashes above still match after the edit.

    **(iii) `antiprior-off-macro` is the 31st row, and it was verified without letting it eat its own
    receipt.** The row's command ends `--out runs/antiprior_off_s0.report.json`, so running it as
    written would rewrite the file `--check` compares against — a registry whose proof overwrites its
    witness can only ever report itself green. The re-run therefore printed to a scratch path and the
    *comparison* became the artifact: `runs/antiprior_off_s0.report_repro.log` prints
    **372 leaves compared, 2 differing — `/cmd` and `/metrics_file`** — with `g1` equal, all 16 cells
    equal, and `macro_acc 0.5338735348381732` on both sides. Those two leaves are provenance, not
    measurement: the committed report was computed while the arm's directory still existed, and the row
    names the flattened `runs/antiprior_off_s0.metrics.json`. A row that reprints the number and moves
    the provenance is a row that reproduced it.

    **(iv) The counts moved because the registry said so, not because prose did.** `make repro` went
    **30/31 → FAIL** on the new row (`no doc quotes the command`), which is §9.30's rule biting on the
    author rather than on a stranger: the command had to be printed in the prose next to the published
    figure, inside SPEC §5 P10 10c, before the row could hold — README's generated block was not
    accepted as the quote, and `doc_text()` cuts that block out precisely so the registry cannot check
    itself. Then `make gates` named the three live numbers the G7 cell owed it (**31 registry rows, 80
    quoted figures, 21 rows re-run on this box**), and `tests/test_reproduce.py`'s coverage assertion
    went red at 24 checked commands. All four are now consistent: `make repro` prints
    **31/31 rows hold; 80 figures tied to a committed witness**, `make gates` prints **6/7 verdicts
    hold — 3 of 7 met, 3 not met, 1 open**, so the release gate is still not clear for the reasons
    §9.53 records rather than for a stale count. The suite is **561 passed / 1 skipped in 261.60 s**
    (562 collected), and CI was green on `38ad176` at run 36821510754 before this tick's edits.

    **(v) The new coverage number was mutation-checked, and the mutation that matters is a shape, not a
    typo.** Rewriting the row's command as `grep -o macro_acc …` drops the checked-command count
    24 → 23 and the assertion fires; stripping the `uv run` prefix leaves it at 24, because
    `python_target` resolves `python -m myna.report` just as happily. So the guard bites when a python
    row quietly degrades into something the `--help` question cannot be asked of — the failure §9.43
    exists to catch — and does not bite on formatting.

55. **The stopped `on` arm turned out to be evidence, the `off` control turned out to be a weaker
    reader than V1-B, and one of this section's own guards turned out to be false.**

    **(i) The pair was audited where it actually ran.** `tests/test_kaggle_bundle.py` has long
    asserted that the two *generated* Kaggle cells are identical but for one flag value, which says
    nothing about the two commands this box executed. Reading the launcher records —
    `runs/antiprior_off_s0/run.json` and `runs/antiprior_on_s0/run.json`, each a 20-token flag list —
    gives the executed audit: **18 flags compare equal, and the only differences are `--out` and
    `--anti-prior`**. Both carry `--suite data/decision-v2-pilot --device mps --config v0 --steps
    3600 --seed 0 --batch 10 --accum-groups 8 --max-q-cells 2048 --group-sample pool
    --eval-every 250 --save-every 250 --mem-safety 0.6
    --stop-factor 3.0 --vocab 8192 --score-loss ce --paraphrase on --row-batch --free-gib 9.0`. That
    makes this the only comparison in the whole 10c lane with no machine, no seed and no
    code-drift confound in it, and it is the reason (ii) and (iii) are worth reading as pair results.

    **(ii) The `on` arm reached 540 stamped updates before the lane was shut down by hand, and that
    is not nothing.** Its two evals sit at matched dose against the control: dev-mid 250 = off
    **0.3682** / on **0.3592**, dev-mid 500 = off **0.3962** / on **0.3789**. Both differences are
    negative — the direction the anti-prior scenario predicted, since the six cells the rule flattens
    are the six myna scores best — and both are *smaller than the control's own swing between
    adjacent evals* (its next reading is 0.3638, i.e. −0.0324 from the 500 point). **So the flag is
    not resolvable at 540 updates**, and the honest output of the partial is a bound, not a verdict:
    nothing here says `on` helps, and nothing here says it hurts by more than the noise floor.
    What the same 540 steps do settle is the wall price, because two arms on one box can be timed
    interval by interval. The `on` arm's first interval (0→60) costs **89.55 s/update**, which is
    data warm-up and not the flag — the control's identical first interval costs 6.75 — and its
    remaining eight intervals run **6.63–7.65 s/update** against the control's **6.43–7.27** over the
    same windows. `--anti-prior` buys no wall, so 10c's ~11.2 GPU-hour price needed no correction.
    One labelling trap worth writing down, because it is the class that ate V1-B: the
    `antiprior_on_s0-wip` Release asset and this box's `runs/antiprior_on_s0/model_last.pt` are both
    **203,229,625 bytes** and are **not the same weights** — the asset hashes to
    `8ed38b240bd843bf…` (downloaded back and re-hashed to prove it) and is the ≤step-250 save its body
    dates at 10:21:55, while the local file hashes to `3b45dd6a4ae65e77…` and is the step-500 save.
    `--save-every 250` overwrites a fixed-size file, so a byte count identifies a format, never a
    step. The step-500 weights exist only on this laptop and only under a gitignored path: that is an
    open durability gap, deliberately left open rather than closed by a public upload nobody asked
    for, because those weights cannot answer the 3,600-update question anyway.

    **(iii) Tier 0 over the control's own weights says the +0.0445 came with less selectivity.**
    `bench/diag_question_ablation.py --run-dir runs/antiprior_off_s0` prints its blocker before its
    numbers — `VOID: this run STOPPED at step 3314 of 3600 requested … No claim may be read from
    them` — so the six arms describe truncated bytes; what makes them useful is that the as-scored arm
    reproduces **716/716 keys** of `runs/antiprior_off_s0.report.json` with 0 tie flips, which binds
    every arm to the published 0.5339 instead to a fresh number. Against the committed V1-B witness:

    | arm | V1-B (T4, 3,600, off) | control (M5, 3,314, off) | Δ |
    |---|---|---|---|
    | as-scored | 0.4893 | **0.5339** | +0.0445 |
    | blank-instruction | 0.3341 | **0.4315** | +0.0974 |
    | blank-options | 0.2669 | **0.3674** | +0.1005 |
    | permute-instruction | 0.4927 | 0.5367 | +0.0440 |
    | cross-source-instruction | 0.4364 | 0.4902 | +0.0537 |
    | swap-state | 0.4173 | **0.4076** | −0.0097 |

    Every arm rises but one, and the *worst*-prompt arms rise furthest, so the differences between
    arms — the selectivity — shrink: instruction reliance 0.155255 → **0.102365**, option reliance
    0.222473 → **0.166500**. The one selectivity that grows is the state: swapping a row's state for
    another row's from the same source costs 0.072038 → **0.126311**, and the cells that move at all
    go **8 → 10** (`contrastive/decision`, `mnli/relation`, `sst5/sentiment` join; `agnews/is_business`
    leaves). The mechanism reading is therefore not "more data made it read the cue better" — it is
    *better-remembered priors plus a firmer state lookup*, and the three noul cells that had already
    collapsed are joined by a fourth: `grep -c '"collapse_share": 1.0'` is **3** in V1-B's witness and
    **4** here, because `agnews/is_business` now answers label 0 on **43/43** rows and lands exactly on
    its own 0.698 floor — that is the whole of its apparent gain, 0.6512 → 0.6977, purchased by
    emitting a constant. Row `tier0-off-control` binds all of it to the two committed files.

    **(iv) The guard 10c had been leaning on is false, and saying so is cheaper than being caught
    with it.** The bullet reads: "the verdict stays on Tier 0's emitter count over the 16 cells — *a
    property of the batching rule, not of the box*." The second half is contradicted by (iii): V1-B
    and the control share `--anti-prior off` and differ only in box and dose, and their emitter counts
    are 3 and 4. **The emitter count is not box- or dose-invariant**, so it was never a confound-free
    yardstick; it is only a verdict against a baseline taken on the same box at the same dose. That
    is now exactly what (iii) supplies, which is the useful consequence: the `on` arm will be judged
    against **4 emitters on the M5**, not against 3 on the T4. SPEC §5 P10 10c carries the corrected
    sentence.

    **(v) What the machinery printed after the row was added.** `make repro` went **30/31 → FAIL**
    (`witness not committed` for both new files) and, after staging, **32/32 rows hold; 88 figures
    tied to a committed witness**; `make gates` demanded the three live G7 numbers (**32 registry
    rows, 88 quoted figures, 5 retrain a checkpoint**) and now prints **7/7 verdicts hold — 3 of 7
    met, 3 not met, 1 open**, so the release gate is still not clear. The coverage assertion in
    `tests/test_reproduce.py` moved 24 → 25 checked commands and was mutation-checked the same way as
    §9.54(v): rewriting the new row's command as `grep -o collapse_share …` drops it to 24 and the
    assertion fires, and the row is a python command whether or not a fresh clone can run it. The
    suite is **561 passed / 1 skipped in 293.44 s** (562 collected) — the identical split §9.54
    printed at 261.60 s, since this tick edited one test's counts and added no test, so the seconds
    are the only thing that moved. Before
    these edits, CI answered for both pushed SHAs of the last tick: run **36822216666** at `8c346a7`
    and run **36823676017** at `0e77d72` each printed **545 passed, 8 skipped** (12m38s–13m17s) with
    the credential scan at **0 matches for 9 shapes across 230 blobs**. This tick's own tree is
    covered by run **36826569290** at `d2e0dc9`, the tip: **545 passed, 8 skipped in 582.78 s**, scan
    **0 matches for 9 shapes across 232 blobs** — the same 545/8 split again, which is the expected
    signature of a tick that adds one registry row and edits one test's counts without adding tests.
    The intervening `4eca685` run was still `in_progress` when the tip's result was read, so the tip's
    green is the witness for the tree §9.55 ships in, not for that SHA. §9.52's arithmetic still
    closes on the pair of boxes (562 collected here − 11 items in the two MLX modules + the 2
    module-level skip lines they collapse to = 553 there), and adding a registry row added no
    collected test, which is why both runs print the same 545.

    **(vi) "+0.0446" was the difference of two rounded macros.** §5 P10 10c and §9.53 both printed
    the control's gain over V1-B as +0.0446, which is 0.5339 − 0.4893 — the subtraction of the two
    figures *after* each had been rounded to four places. From the committed witnesses the values are
    0.5338735348381732 and 0.4893271976084137, so the gap is **0.0445463372297595**, i.e.
    **+0.0445**. The live prose now prints +0.0445 with the exact value beside it; the dated entries
    keep what they said, which is what makes this one findable.

    **(vii) The checkpoint inventory, because "the last irreversible risk" was never one risk.** An
    audit of which rows read weights, against which Releases carry them, leaves exactly one gap and
    one soft one. Covered: `myna-v0` → `v0-checkpoint` (the 7 `here` rows that read it),
    `v1b-kaggle-3600b` → `v1b-checkpoint`, `antiprior_off_s0` → `antiprior_off_s0-weights`
    (§9.54(i)), and the `on` arm's ≤step-250 snapshot → `antiprior_on_s0-wip`. **Not covered:**
    `runs/myna-v1-rich/` — `model.pt` 67,733,781 B and `metrics.json`, both matched by
    `.gitignore:24`'s `runs/*` rule — is the only input set behind a *quoted* published figure that no
    Release carries, and row `void-vs-laya` is status `here`, i.e. counted in the G7 cell's "21 rows
    re-run on this box". It is honestly labelled (the row's own note says the artifact is committed
    and the metrics file behind it is not, and §5 P1 calls the figure void as evidence because the
    checkpoint predates the `385e06c` batching fix), so nothing over-claims; the exposure is only that
    a disclosure number — 0.338 against a 0.346 control — becomes unreproducible with the laptop.
    Softly uncovered: `runs/antiprior_on_s0/model_last.pt` (the step-500 save, §9.55(ii)) and
    `runs/myna-v1/model_last.pt`, `runs/myna-v0-rlcd/`, `runs/v1-smoke/` — none of which backs a
    quoted figure except `rlcd`, whose row is `retrain` and therefore claims no local re-run.
    **Priced, not decided:** shipping `myna-v1-rich` is one 65 MB Release upload, roughly the size of
    `antiprior_off_s0-weights`, and it would make every quoted figure in the repository
    download-reproducible; it is left as the user's call because the figure it protects is explicitly
    void-as-evidence, and because publishing an artifact is shared state (§9.53's rule that a Release
    is the user's to open, not the agent's).

56. **The `on` arm came home to this Mac, and dose-matching it turned out to be a question about
    which index the loop stops on and which schedule the snapshot carries.** The owner's decision at
    2026-10-01 12:52 NPT: *"run the on arm on the Mac, not Kaggle. Continue it to 3,314 updates
    (dose-matched to the off control), one run at a time, keep the Mac from sleeping for the whole
    run, then download + Release the weights immediately and eval macro vs the control's 0.5339."*
    That closes the Kaggle lane for 10c (the kernel and its five CPU-only measurements stay in §9.53(vi)
    as the record of why the account cannot run it) and it makes this entry a launch note, so every
    number below is either stamped by the run or labelled projected.

    **(i) The canonical launcher already knew how to do this, which is why nothing was hand-typed.**
    `kaggle/run.py`'s `continuation()` returns `["--resume"]` when `out/model_last.pt` exists, and
    refuses — naming `--warm-start` and a longer `--steps` as the two things that are not a resume —
    when `metrics.json` says the schedule is already spent. §5 P9 9e measured that refusal's failure:
    resuming a finished run restores a spent scheduler and trains at lr ~0 while printing numbers that
    read as drift. Here the snapshot is at step 500 and the target is 3,314, so the resume is *before*
    the decay ends and the branch is the right one. The expanded command, printed by `--dry-run` before
    anything was launched and then by the run itself:

    ```bash
    uv run python kaggle/run.py --corpus data/decision-v2-pilot --out-root runs --device mps \
        --config v0 --steps 3315 --batch 10 --seed 0 --score-loss ce --stop-factor 3.0 \
        --save-every 250 --free-gib 9.0 --name antiprior_on_s0c --anti-prior on
    ```

    which is `myna.train` with 21 flags, i.e. the `off` control's 20 plus `--resume`. A fresh `--name`
    was required twice over: `build_command` raises on `--warm-start` with a snapshot present, and
    `runs/antiprior_on_s0/` is the directory §9.55(i) and §9.55(ii) make claims about. Writing into it
    would have replaced the `run.json` whose 20 flags that audit counts, and would have overwritten the
    step-500 `model_last.pt` whose digest `3b45dd6a4ae65e77…` that entry quotes. So `runs/antiprior_on_s0c/`
    holds a *copy* of the seed, and both directories survive: `shasum -a 256` returns the same
    `3b45dd6a4ae65e77c7f9cbfa800ccb185637da4d57c3863c84644a800e700917` for `203,229,625 B` on both
    sides of the copy, which is the round trip that says the resumed weights are the weights the
    partial trained.

    **(ii) Dose-matching is `--steps 3315`, not `3314`, because the loop's last executed index is
    `args.steps − 1`.** `train.py:1000` reads `for step in range(start_step, args.steps)` and
    `train.py:1032` stamps `last_step = step` inside it, while the control's committed
    `runs/antiprior_off_s0.metrics.json` records `last_step: 3314` under `steps_requested: 3600` — so
    the control *executed* index 3314. Requesting 3,314 here would have stopped the arm one update
    early, at index 3,313, and broken the comparison with the literal the registry already binds
    (`antiprior-off-macro` quotes `"last_step": 3314`). One consequence of changing the request is not
    cosmetic: `log_every = max(1, min(200, args.steps // 60))` (`train.py:924`) makes the *print
    cadence* a function of the requested dose, so the control stamped every 60 updates and this run
    stamps every 55 — visible as the first line arriving at `step 550` where the partial's stopped at
    `step 540`. The *eval* grid did not move, and that is the one that matters for the comparison:
    `--eval-every 250` still fires at 750, 1,000 … 3,250 (11 points), which with the partial's 250 and
    500 is the control's 13 dev-mid readings on the same steps.

    **(iii) The scheduler is the part that could have quietly ruined the arm, and it was read out of
    the file rather than inferred.** `--warm-start` would have loaded those weights onto a fresh cosine
    fitted to 3,315 updates — a third learning-rate curve beside the control's. `--resume` restores the
    saved one, and `torch.load` on the seed prints `step 500`, `T_max 3600`, `last_epoch 501`,
    `lr 5.717815913939212e-4`, `eta_min 0.0`. So the continuation walks the control's own 3,600-step
    annealing curve and stops at index 3,314 on it, which is precisely where the control stopped
    (about 1.5 % of the 6e-4 base lr, low but not the ~0 that §9e's failure mode is).

    **(iv) The flag audit, run on the three commands rather than asserted.** As parsed flag-by-flag:
    `off` 20, `on` partial 20, continuation 21. Against the partial, the continuation differs in
    `--out`, `--steps` and `--resume`. Against the `off` control it differs in `--anti-prior` (the
    variable), `--out` (a directory cannot be shared), `--steps 3600 → 3315` and the presence of
    `--resume`. Every other flag compares equal, including the four that move the numbers: `--batch 10`,
    `--seed 0`, `--score-loss ce`, `--stop-factor 3.0`, `--save-every/--eval-every 250`, `--accum-groups 8`,
    `--max-q-cells 2048`, `--mem-safety 0.6`, `--vocab 8192`, `--free-gib 9.0`, `--row-batch`,
    `--paraphrase on`, `--group-sample pool`. Two lines of the new run's own output are the witness that
    the regime really is shared, not merely the argv: `memory plan: batch 10 fits 9.0 GiB free x 0.6,
    less 3.2 GiB of question branch at p95 267 tokens (needs 5.3 GiB)` — byte-identical to the
    control's, which is what the `--free-gib` pin is for (§9.53(v)) — and `anti-prior: mini-batches
    drawn by inverse label prior; 6 of 16 cells carry a majority label at or above 0.55`, identical to
    the partial's. The resume line the trainer stamped is `resume: runs/antiprior_on_s0c/model_last.pt
    at step 500 -> continuing at 501 of 3315; data RNG re-seeded from --seed 0, so this is a
    continuation, not a bit-identical replay` — that re-seed is the honest residue in the pair and
    §5 P10 10c now names it instead of claiming a clean two-flag pair.

    **(v) Sleep, thermals and the one-run-at-a-time rule were checked before the launch, not after.**
    `pmset -g batt`: 'Now drawing from AC Power', 74 % charging on a 70 W Apple adapter — which is what
    makes `caffeinate -s` mean anything, since that assertion does not hold on battery. `pgrep -fl
    myna.train` before launch: no matches, so nothing was already training. `caffeinate -i -s -w 80297`
    is PID 80330 and `pmset -g assertions` confirms both halves with "caffeinate asserting on behalf of
    Process ID 80297": `PreventUserIdleSystemSleep` and `PreventSystemSleep`. The driver is 80297, the
    trainer 80299, and the pre-existing unrelated `caffeinate` and Chrome were left exactly as found.
    This is not belt-and-braces: the `off` control's own death was `step time 596.40s > 3.0x median
    6.58s of the last 20`, which §9.53(iv) attributes to battery 'Maintenance Sleep' rather than to
    anything the model does. The corollary is a rule for the next five hours on this box: **no test
    suite, no Tier 0 re-run, no `make repro` while the arm runs**, because `--stop-factor 3.0` is armed
    here and a heavy local job is exactly the shape that trips it — a self-inflicted version of the
    event that truncated the control.
    Wall clock, measured rather than hoped: the first 50 resumed updates stamped **355 s = 7.10 s/update**,
    inside the control's 6.43–7.27 band, and there is no repeat of the 89.55 s/update first interval
    because AdamW's state came back with the snapshot instead of warming up. At that slope the remaining
    2,764 updates are **5.45 h (projected)**, plus eleven dev-mid evals at the ~37 s the control's
    intervals imply, so the projected finish is ≈ 18:35 NPT.

    **(vi) The weights ship without anyone awake, and the tag says the step the run reached.**
    `runs/ckpt_backup_on_s0c.sh` (PID 80600) does two uploads and neither is what
    `runs/ckpt_backup.sh` would have done with these paths: it ships the first snapshot *this* run
    writes — identified by a digest different from the `3b45dd6a…` seed, because uploading the seed
    would insure nothing (it is byte-identical to a file already in the working tree) — and then the
    final `model.pt` + `tokenizer.json` once `stable()` sees the size stop moving. Its titles take the
    step from `metrics.json`'s `last_step`, never from the requested dose, which is the §9.54 defect
    this script exists to not repeat. Its log and state file are gitignored alongside
    `runs/pair_driver.log`, because line 2 of the log is the expanded command and its first token is
    this laptop's absolute interpreter path.

    **(vii) What this run is judged against, fixed before it finishes.** Test macro against the
    control's committed **0.5338735348381732** (not V1-B's 0.4893, which is a different code path and a
    different box); Tier 0's constant-emitter count against the **4** the same harness printed for this
    control on this box at this step, so 5 or more means the batching rule bought the mechanism nothing
    and 3 or fewer means it moved as designed; `cells that move at all` against 10; and the dev-mid
    trace on the control's 13 points. The pre-registered direction from the scenario work is **flat to
    down** (0.42–0.49), because `--anti-prior` re-weights the six cells myna already scores best in — so
    a rise here is a finding about a continued 500-update partial not being the same object as a fresh
    3,600-request run, and it will be written that way. *(OPEN: every figure in this bullet is gated
    until the run stamps its last step; the only measured numbers in this entry are the ones above.)*

# Myna — research log

A persistent-state, delta-encoded System-1 decision engine. Typed choice /
score / noul decisions over growing observations, in less than the time an
encoder takes to re-read one sentence.

## The thesis

laya (encoder) and kev (repurposed decoder LM) both re-read the entire
observation on every request. Agent loops and document threads don't work
that way — the state grows, and each request only needs the new part plus a
short question. Gated linear attention keeps a fixed-size matrix state per
layer, so the observation scan can be *continued* instead of repeated,
exactly, with zero drift.

## Design decisions (with evidence)

1. **Causal global state + full-bidi question branches.** kev's own research
   log shows causal-only matched bidirectional on clean probes. A causal
   trunk is what makes prefix caching exact; questions are short, so they
   afford a full backward scan inside the branch without threatening the
   cache. (trunk.py, `scan_state` / `scan_question`)
2. **Questions are branches, not prompt tokens.** Each question resumes every
   layer from the cached state-end `S`. Cross-question isolation is
   architectural, not a mask — verified by
   `test_question_branches_are_isolated` (adding/removing/reordering other
   questions changes nothing, atol 1e-9).
3. **No vocabulary readout.** One pointer mechanism over option spans serves
   all three types. kev removed the vocab head too; ours goes further —
   there is no LM trunk at all, so no 150k-token embedding table sitting in
   the hot path.
4. **Three execution forms of the same scan, proven equal.** Recurrent (the
   streaming path), chunked (the training path, linear memory), quadratic
   (the reference). `test_trunk_numerics.py` pins all three to each other in
   float64, and `test_engine_parity.py` pins engine == batch end-to-end.
5. **Padded tokens must not decay the memory.** First real bug found via the
   parity test: masked tokens had k=v=0 but a random forget gate, which
   silently decayed the carried state in backward scans. Gates are now forced
   to 1 at masked positions. Lesson: equivalence tests catch what loss
   curves hide.
6. **The retained state is S, and only S.** The engine initially cached
   per-layer hidden states over the whole observation (`h_cache`) for no
   reader — nothing downstream touches them. At 16k tokens that wasted 151 MB
   per observation and contradicted the fixed-size pitch. Removed: retained
   state is now 576 KiB at production width, *independent of length*, and
   `test_state_is_fixed_size_independent_of_length` + the benchmark's memory
   line make the claim checkable. Lesson: audit what the cache actually holds
   vs what the algorithm reads.
7. **Inference scans take a larger chunk.** Chunk equivalence (recurrent ==
   chunked, any chunk size) is proven in the numerics tests, so the no-grad
   paths use chunk=256 while training keeps 16 (autograd memory). A 16k
   observation goes from 1024 Python-loop slices per layer to 64.
8. **Sessions as an API primitive.** `myna.serve` exposes `/v1/sessions`:
   open once, append per new message, ask anytime — the HTTP shape of the
   architecture. laya/kev can only offer stateless `/predict`. Gotcha:
   `from __future__ import annotations` makes FastAPI treat body models as
   query params (422); serve.py deliberately omits it.
9. **Long memory is the prior; forgetting is learned.** The GLA forget gate
   initializes at weight 0, bias sigmoid⁻¹(0.99). With the default bias 0 the
   gate sits at 0.5 and erases a token's trace to 1e-6 within 20 steps —
   training from that regime never escaped chance on the full corpus.
10. **The policy IS the answer — no REINFORCE in a non-sampling head.** The
    first RLCD run copied laya's recipe (REINFORCE + batch-mean baseline) and
    degraded the model: train reward fell 1.0 → −0.98 in 200 steps, because
    above-baseline examples had p(gold) pushed *down*. That machinery assumes
    sampled trajectories; our pointer head reports a distribution directly,
    and the strictly proper score of a reported distribution is
    differentiable in it. Correct objective: `max E[Brier(p, gold)] − β·KL(p‖ref)`
    → acc 0.9626 → 0.9657, mean ECE 0.0191 → 0.0076, mean Brier 0.0258 →
    0.0232, every one of 9 questions improved or held. Lesson: transplant the
    objective, not the optimizer folklore. (`runs/rlcd_v0.log`, measured. The
    sample is the dev split `rlcd.py` regenerates at 600 rows per workflow, so
    0.9626 is a different dev draw than the train log's 0.9599 — the claim is
    the before/after pair, not either level.)

## Benchmarks

### WITHDRAWN (2026-09-26, SPEC §9.20) — kept here as the record, not as a result

Final v0-checkpoint run on M5, idle machine, CPU, n=8 workflows x 6 appends,
3 questions/request (`runs/bench_stream.md`, log in `runs/bench_stream_v0.log`):

| state tokens | myna-stream | myna re-encode | laya re-encode | speedup |
|---|---|---|---|---|
| 128 | 43.3 ms | 91.3 ms | 179.4 ms | 2.1x |
| 512 | 31.9 ms | 271.2 ms | 507.4 ms | 8.5x |
| 1,024 | 34.0 ms | 418.8 ms | 504.5 ms | 12.3x |
| 2,048 | 28.9 ms | 965.3 ms | 512.5 ms | 33.4x |
| 4,096 | 38.4 ms | 1,748.1 ms | 506.1 ms | 45.6x |
| 8,192 | 28.4 ms | 3,132.9 ms | 510.7 ms | 110.5x |
| 16,384 | 28.9 ms | 6,904.0 ms | n/a (8192 cap) | 239.3x |

Short-state avg: myna-stream 32.5 ms, myna-full 48.1 ms, laya 127.8 ms.
Retained state: 576 KiB fixed at every length.

Why it is withdrawn, against its own numbers:
1. The `speedup` column is `myna re-encode / myna-stream` — myna against itself. It never measured
   laya, and reading 1 ("the delta-engine claim holds") is the only thing it supports.
2. Reading 2 called laya's ~510 ms plateau "its 8192 cap truncation region" in the same breath as
   "~17x faster than laya answers at its 8k maximum". The laya checkpoint's typed-decisions window
   is **1024**, so every row above 1,024 fed it the same input: the plateau *is* the truncation, and
   comparing a complete 16k read against a 1k read is not a speedup.
3. Reading 3 is contradicted by the row above it: myna-full 965.3 ms vs laya 512.5 ms at 2,048
   tokens. myna's non-streaming path does not "beat laya above 2k" — it loses until the comparison
   becomes meaningless.
4. laya was driven through `Router` with `noul` criteria stripped by a local shim: neither the call
   path nor the options-per-question matched what laya publishes.

### What replaced it: matched-condition, one process, both engines (SPEC §5 P4)

`bench/bench_latency_matched.py` → `runs/latency_matched.md`. Same `str` state, same question
`dict`, both fp32, laya as `Agent.system_one` on `55cf4c4`, ladder capped inside laya's window,
`flock`ed so no second copy can share the box mid-run. p50, 10 reps, 3 warm-ups:

| questions | laya `system_one` | myna end-to-end | ratio | myna, state cached | ratio |
|---|---|---|---|---|---|
| 1 | 100.7 ms | 33.1 ms | 3.04x | 14.9 ms | 6.78x |
| 5 | 272.0 ms | 74.9 ms | 3.63x | 55.5 ms | 4.90x |
| 10 | 506.3 ms | 120.8 ms | 4.19x | 98.2 ms | 5.15x |
| 50 | 3,589.5 ms | 870.0 ms | 4.13x | 787.1 ms | 4.56x |

Every cell is *m*: this table is run 2 transcribed from `runs/latency_matched.md`, where each
millisecond is a p50 the harness printed, and its `meta` records the platform, device, thread
counts, 3 warm-ups / 10 reps, both checkpoints by path and commit, both dtypes and laya's
1024/256 window. Run 1 is `runs/latency_matched_run1_superseded.md`. What neither records is
the box's load average — §9.23's convention postdates them — so these two runs are reproducible
but not restatable under it (§9.30).

Shipped claim = per-row minimum over the two committed runs: **3.04 / 3.21 / 4.12 / 3.60×** and
**6.78 / 4.67 / 4.75 / 3.65×**. Cost fits on the ladder: ask is 10.1 ms + 9.6 ms/question with a
state slope of −10.3 µs/token (zero — the fixed-size state, measured), observe is 360 µs/token with
a negative intercept (quadratic inside a 256-token chunk), and laya's additive fit reaches only
R² 0.74 because its cost is state × questions.


## The plateau post-mortem (2026-09-25/26)

The first 6000-step run sat at chance (dev acc 0.335, EMA loss 1.15 = the
label-prior entropy) through 1000 steps and was stopped. Two findings:

1. **Gate init.** With `sigmoid(0)=0.5` forget gates, a token's memory decayed
   to ~1e-6 within 20 steps — long observations were unreadable at init.
   Fixed by zeroing the gate weight and biasing to a≈0.99 (decision 9 below).
2. **Epoch budget, not capacity.** A fit-ladder diagnostic (bench/diag_*.py)
   showed: 192 examples fit to 0.97 in 10 epochs; 768 examples are still
   memorizing at 8 epochs (fit 0.45). The 6000-step plan over 18k examples
   is only ~30 epochs — the old run was stopped at 5 epochs, mid-noise, and
   the "plateau" was slow learning, not a dead end. Lesson: judge a run by
   fit-accuracy per EPOCH, not loss per step.

The open question from the post-mortem — *does generalization stay far behind
memorization at scale?* — is answered for v0: it does not, once the data has
enough surface variety. The gate fix plus 50 disjoint nouns per split (verified
by counting `SPLIT_NOUNS`, and 15% generic-noun dropout inside `generate`)
produced, from `runs/train-v0.log`:

**v0 final: dev 0.960 / test 0.952 overall** — macros of the nine per-question
rows the log prints under `=== dev ===` and `=== test ===` (0.9599 / 0.9523) —
after 9,000 steps at batch 32 on MPS, last step line `15693s`, i.e. 4.4 h.
Per-question test accuracy 0.928–0.974, test ECE 0.0115–0.0415 with the single
refit temperature the log prints as 3.0, test noul Brier 0.0278–0.0461. The
dev→test gap is **0.76 of a point** on fully disjoint nouns, so the model reads
templates, not nouns. dev-mid trajectory: 0.4682 @500 → 0.9207 @1500 (the
grokking knee) → 0.9682 @7500/8000/8500 — a prefix subset of dev scored every
500 steps, so it is a trajectory, not an overall.

Two cells that used to sit in this paragraph are dead and are named in
SPEC §9.30: "dev 0.968" was the @8000 mid-run probe, not the final dev macro,
and "test 0.951" is printed by no artifact in this repo. The split sizes are
also not in the log: each printed test accuracy rounds to its 4-dp value at
1200 rows per workflow (3,600 test rows) and at no smaller n up to 4000, but
that is arithmetic on printed rounding, so it is *p* and not a measurement;
`train.py` now logs the split sizes so the next run does not need the
inference.

## Open questions / next

- Gate init fix: keep a≈0.99 or try a floor parameterization
  `a = exp(-softplus(x))` like Mamba's dt? Measure both.

- Chunk size 16 is arbitrary; sweep 8–64 for MPS throughput.
- `torch.compile` the chunked scan; or a custom Metal kernel.
- ~~The 16k-context memory audit~~ done: S-only cache is 576 KiB fixed
  (decision 6 above). Fine.
- Train a "long" checkpoint with truncated-backprop on 4k states.
- Real corpora: laya's typed-decisions benchmark and kev's frozen suites for
  apples-to-apples accuracy (speed is already apples-to-apples: same box).
- ~~RLCD-style training (strictly proper scoring) as in laya~~ done: Brier
  maximization with a KL leash (decision 10) took dev acc 0.9626 → 0.9657 and
  mean ECE 0.0191 → 0.0076 on `runs/myna-v0-rlcd` — calibration roughly halved
  with no accuracy sacrifice. The v0 line closes here; v1 (real corpora, MLX
  kernel) and the v2 research program are specced in [V2_SPEC.md](V2_SPEC.md).

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

## Benchmarks

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

Readings:
1. myna-stream is flat 28-43 ms from 128→16k — the delta-engine claim holds
   on the trained model, not just the probe.
2. laya's re-encode plateaus at ~510 ms (its 8192 cap truncation region) and
   can't serve long states at all. At 16k myna answers ~17x faster than laya
   answers at its 8k maximum.
3. Even myna-full (a plain encoder of the same size) beats laya above 2k
   tokens — that's the architecture of the trunk, before streaming.


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
enough surface variety. The gate fix plus expanding the noun pools to ~50 per
split (with 15% generic-noun dropout to force template keying) produced:

**v0 final: dev 0.968 / test 0.951 overall** (9000 steps, batch 32, MPS,
~3.7 h on M5). Per-question test accuracy 0.928–0.974, ECE 0.012–0.042 with a
single refit temperature of 3.0, noul Brier 0.028–0.046. The dev→test gap is
1.7 points on fully disjoint nouns, so the model reads templates, not nouns.
dev-mid trajectory: 0.468 @500 → 0.921 @1500 (the grokking knee) → 0.968 @8000.

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
- RLCD-style training (strictly proper scoring) as in laya — v0 used plain CE
  with a fitted temperature; RLCD is the v1 experiment.

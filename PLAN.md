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

## Benchmarks

(filled in by runs/ — streaming vs full re-encode vs laya, and long-context
tables. Probe-checkpoint sweep on M5, idle machine: myna-stream flat
~27-28 ms from 128→16384 tokens vs myna-full 49 ms→1664 ms → 1.5x at 128,
16x at 4k, 31.5x at 8k, 58x at 16k. To be re-measured on the v0 checkpoint
and against laya.)

## Open questions / next

- Chunk size 16 is arbitrary; sweep 8–64 for MPS throughput.
- `torch.compile` the chunked scan; or a custom Metal kernel.
- ~~The 16k-context memory audit~~ done: S-only cache is 576 KiB fixed
  (decision 6 above). Fine.
- Train a "long" checkpoint with truncated-backprop on 4k states.
- Real corpora: laya's typed-decisions benchmark and kev's frozen suites for
  apples-to-apples accuracy (speed is already apples-to-apples: same box).
- RLCD-style training (strictly proper scoring) as in laya — v0 used plain CE
  with a fitted temperature; RLCD is the v1 experiment.

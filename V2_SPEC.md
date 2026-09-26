# Myna v2 — Spec: the state is the database

> **Superseded on numbers by [SPEC.md](SPEC.md)**, which is the authoritative master spec: goals,
> acceptance gates, plan phases, baselines and the corrections log. This file remains the
> long-range *research* program (Pillars 1–6: state algebra, O(window) edits, two-timescale
> memory, utility heads, on-device learning). Where a figure here disagrees with SPEC.md, SPEC.md
> wins — notably 14.4M → **15.35M measured**, and the 239× figure, which is an internal ablation
> and not a competitor comparison (SPEC §9.20 withdraws the table it came from).

v0 proved the engine: a 14.4M-param persistent-state decision model that
answers typed questions in ~30 ms flat from 128→16k tokens, 239x faster than
re-reading, calibrated to ECE ≤ 0.04, on a MacBook.

*Flat is right; the range is not measured.* The matched-condition harness fits the answer path's
state slope at −10 µs/token and lands one question at 14.7–17.4 ms for states from 66 to 1,060
tokens (SPEC §9.22); beyond that it is the *scan*, not the answer, that has to be re-timed, and the
16k row's witness is the withdrawn table. Quote 128→1k until P7 re-measures it.

v1 takes it to production reality: MLX/Metal kernel (5 ms), real laya/kev
corpora, long-context checkpoint, multilingual routing.

v2 changes what the product **is**. The thesis:

> If the retained state is a fixed-size, exactly-continuable, linear object,
> then it can have an **algebra** — merge, branch, diff, edit — and decisions
> stop being "answers about a document" and become "queries over a live world
> model." No architecture online (laya, kev, any LM) can do this, because
> their "state" is either nothing (re-encode) or a KV cache that grows and
> can't be combined.

Everything below is designed so that each pillar is *provable* the way v0's
scan-equivalence tests are — not vibes.

---

## Pillar 1 — Question freedom (free-text questions, zero-shot)

v0 keys on per-workflow question templates. v2 must answer **any** typed
question about a state it has never seen phrased that way.

- The pointer head gains a question encoder: `q = W_q · h(question tokens)`
  where question tokens are scanned *inside the branch* (v0 mechanism, now
  with learned content instead of fixed templates).
- Options are spans **inside the state** (point to entities in the
  observation: "which team owns the fix" → the team entity in thread 4) or
  free criterion strings encoded on the fly.
- Training: question-paraphrase augmentation (≥8 phrasings per template) +
  held-out-phrasing dev split; metric = new-phrasing accuracy ≥ 0.90 while
  template accuracy holds.

**Acceptance:** a question never seen in training, paraphrased arbitrarily,
scored in one pass; answer distribution unchanged by rewording that preserves
semantics (paraphrase-consistency test, KL < 0.02).

## Pillar 2 — State algebra (merge, branch, diff)

GLA state update is additive across the linear scan, so cached states of
disjoint observations can be combined *without re-reading either*:

- **Merge** `S(A ⊎ B) ≈ S(A) ⊕ S(B)` — exact for order-agnostic decision
  queries; tested against a full re-encode of A+B the way v0 tests append.
  Use: a fleet of agent sessions rolls up into one org-level decision state
  in O(#sessions) matrix additions — thousands of tickets, one 576 KiB state.
- **Branch** — copy-on-write state snapshots (already trivially true; make it
  an API: `obs.branch()` for counterfactual "what if this message arrives").
- **Diff** — `S(B) − S(A)` as a *what changed* detector: flag appended spans
  whose contribution to the answer exceeds a threshold. Use: audit trails and
  change-triggered re-decisions.
- Every op ships with a float64 equivalence or bound test in CI, same
  discipline as `test_trunk_numerics.py`.

**Acceptance:** merged-state answers over 100 disjoint tickets within 2 points
of full-re-encode accuracy; merge of 1000 states < 50 ms; diff localizes the
gold-edited span > 95% of the time on synthetic edits.

## Pillar 3 — O(window) mid-state edits

v0's honest limit: an edit in the middle costs a full re-encode. Bidirectional
branches make this unnecessary:

- Cache prefix state `S_→i` and suffix state `S_←j` at block boundaries
  (fixed budget, e.g. one boundary per 256 tokens).
- Editing `[i, j)` = re-scan the window only, then recompose
  `S = S_→i ⊕ scan(window) ⊕ decay-transport(S_←j)`.
- This turns document editing — the workload every encoder still re-bills in
  full — into a local O(window) update with exactness bounds.

**Acceptance:** a 1-token edit in a 16k state answered in < 5 ms (vs 30 ms
full); accuracy within 0.5 points of re-encode; parity tests at every block
size.

## Pillar 4 — Two-timescale memory → effectively unbounded context

576 KiB fixed is the point, but one matrix state can't hold both verbatim
recent detail and distant gist. v2:

- Fast state (current, a≈0.99) + **slow state** (a≈0.9999, gist-level) per
  layer; the question branch reads both — 2x state, still fixed size.
- Learned eviction writes episodic facts into a small key-addressable slot
  array (fixed 1-2k slots, same budget logic as any KV cache but bounded) so
  "what did the customer say in message 3" survives 100k tokens.
- Streaming eval to 1M tokens: accuracy on needle-style typed decisions vs
  context length must be *flat*, the way latency already is.

**Acceptance:** 1M-token stream, fixed 1.5 MB/observation total; needle
decision accuracy ≥ 0.95 at 1M with flat latency.

## Pillar 5 — Decisions, not answers (expected-utility heads)

noul/choice/score output calibrated probabilities; real agent loops need an
action. v2 adds a decision layer **on top of** the probabilities:

- Per-question cost matrix (wrong-refund cost vs right-refund cost, paging a
  human at 3am cost) → argmax expected utility + abstain when the utility gap
  is inside the uncertainty band.
- RLCD v1 already optimizes proper scores; v2 optimizes the *utility* with
  the same strictly-proper machinery on the distribution, keeping the KL
  leash to the calibrated base — accuracy without losing calibration.
- Everything stays one forward pass: the utility head is a [O×A] table plus
  a matmul.

**Acceptance:** on cost-asymmetric dev tasks, expected cost ≥ 30% below
argmax-of-probability and ≥ 10% below laya's confidence-gated routing, at
ECE ≤ 0.02 after utility training.

## Pillar 6 — The model is small enough to live on the user's machine

v0 ships at fp32/58 MB. v2 target:

- MLX port of append + question branch (the recurrent form is already
  kernel-friendly: one `S ← a·S + kᵀv` per token — trivially Metal/MLX).
- int8 weights, fp16 state: **~15 MB checkpoint**, cold-load < 1 s.
- On-device continual learning: nightly local RLCD fine-tune on the user's
  own accepted decisions (LoRA, 5 min on M5) — the model gets personal without
  any data leaving the laptop. laya/kev cannot do this (1.7 GB / 8 GB).

**Acceptance:** 5 ms median per request (3 questions, 4k state) on M5;
on-device fine-tune improves personal-corpus score without > 1-point global
regression (KL-leashed, measured on frozen suites).

---

## Sizing v2

d_model 768, n_layers 12, ~85M params — still under a laptop's breath, ~7x
v0's capacity, required by Pillars 1/4/5 (free-text questions and slot memory
need real width). Trains in < 24 h on one M5 via the v0 recipe (gate-init
prior, epoch-budget judging, chunked scans).

## Non-goals (v2 stays a System-1 model)

- No generation, no vocabulary head, no beam search. If you need text out,
  that's a different model; our contract is decisions in constant time.
- No KV-cache "context window" — if a mechanism needs the token list at
  answer time, it failed the architecture.
- No cloud training requirement: if a pillar can't be trained and served on
  consumer hardware, it's cut.

## Milestones

| # | deliverable | proves |
|---|---|---|
| M1 | slow-state + slots, long-stream eval | Pillar 4 (unbounded context) |
| M2 | question-paraphrase training + held-out-phrasing eval | Pillar 1 (free-text Qs) |
| M3 | merge/branch/diff ops + CI equivalence tests | Pillar 2 (state algebra) |
| M4 | block-boundary caching + O(window) edit path | Pillar 3 (edits) |
| M5 | utility heads + RLCD-on-utility | Pillar 5 (actions) |
| M6 | MLX int8 port + on-device LoRA loop | Pillar 6 (on-device) |

Order rationale: M1/M2 are pure data/model changes on the existing codebase
(fast wins, no new kernels); M3/M4 are the genuinely novel research bets and
benefit from M1's two-timescale design; M5/M6 are product layers that ride on
whatever the earlier milestones prove.

## Risks, stated honestly

1. **Merge exactness is approximate** for order-sensitive queries — the bound
   must be measured per task, not assumed. If the bound is bad on decision
   tasks, Pillar 2 narrows to branch+diff (both exact).
2. **Free-text questions** may re-expose the template-keying failure mode v0
   only partly solved — the paraphrase eval in M2 is the kill-criterion.
3. **Slot memory** is the only pillar without a precedent in v0's passing
   tests; budget it as research, not engineering.
4. On-device fine-tuning safety: local RLCD must never degrade a shipped
   head — the frozen-suite regression gate in M6 is mandatory, not optional.

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

## The architecture

1. **Gated linear attention trunk.** Each layer carries a fixed-size matrix
   state `S ∈ R^{H×d_k×d_v}` (~100KB/layer), not a KV cache that grows with
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

| | params | checkpoint | state memory | context |
|---|---|---|---|---|
| **myna-v0** | 14.4M | ~58 MB fp32 | ~0.6 MB | linear — 16k+ |
| laya | 421M | ~1.7 GB | n/a (re-encode) | 512–8192 |
| kev-4B | 4B | ~8 GB | ~GB-scale KV | — |

</div>

## Measured on an Apple M5 (10-core, 32 GB)

Streaming workload on the trained v0 checkpoint: grow the observation in
6 appends, re-ask 3 typed questions after each — the agent loop. Average per
request, idle machine, CPU:

| state tokens | myna-stream | myna re-encode | laya re-encode | myna speedup |
|---|---|---|---|---|
| 128 | 43 ms | 91 ms | 179 ms | 2.1x |
| 512 | 32 ms | 271 ms | 507 ms | 8.5x |
| 1,024 | 34 ms | 419 ms | 505 ms | 12.3x |
| 2,048 | 29 ms | 965 ms | 512 ms | 33.4x |
| 4,096 | 38 ms | 1,748 ms | 506 ms | 45.6x |
| 8,192 | 28 ms | 3,133 ms | 511 ms | 110.5x |
| 16,384 | 29 ms | 6,904 ms | n/a (8192 cap) | 239.3x |

The streaming column is **flat by design**: cost is the question branches
plus the delta, never the prefix. laya pays a ~500 ms floor above 512 tokens
and cannot serve long states at all; myna answers a 16k-token observation
~17x faster than laya answers an 8k one.

## Trained v0 — accuracy on held-out data

9,000 steps (~3.7 h on an M5, batch 32, MPS), 3,000 synthetic examples, dev
and test splits using nouns never seen in training. Overall accuracy:
**0.968 dev / 0.951 test** — generalization is real, not memorization.

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

ECE ≤ 0.04 everywhere with a single refit temperature (3.0). A short RLCD
pass — 800 steps maximizing the Brier proper score under a KL leash to the
supervised model — then **halved it: mean ECE 0.0191 → 0.0076, dev accuracy
up to 0.9657**, every question improved or held. The policy here *is* the
reported distribution, so the score is optimized differentiably, no REINFORCE
(PLAN.md decision 10 has the post-mortem of trying it the generative way).

## Quickstart

```bash
uv sync
uv run python -m myna.train          # ~3 h on an M5, no GPU required
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

# the observation itself is portable: ~0.6 MB, resume it tomorrow
obs2.save_state("session.pt")  # ... myna.restore("session.pt")
```

### Serve API

```bash
uv run --extra serve python -m myna.serve --ckpt runs/myna-v0 --port 8080
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
it does **not** yet claim parity with laya/kev on their frozen suites — that
comparison is the next milestone in [PLAN.md](PLAN.md), along with RLCD-style
training, Metal kernels for the scan, and multilingual coverage. The full
next-generation design — free-text questions, state algebra (merge/branch/diff
of cached states), O(window) mid-state edits, unbounded two-timescale memory,
expected-utility decisions, 15 MB on-device model — is specified with
acceptance criteria in [V2_SPEC.md](V2_SPEC.md).

Honest limits: word-order-heavy tasks can be partly solved bag-of-cues style;
mid-state edits (not appends) fall back to a full — still linear-time —
re-encode; the question-branch constant (~28 ms CPU today) is kernel work,
not architecture work.

Apache-2.0 · research log in [PLAN.md](PLAN.md) · questions to
[aashish254](https://github.com/aashish254)

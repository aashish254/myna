# Myna ⚡

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
   not distilled, *identical*.
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

Streaming workload: observe a long state, append a small delta, ask 3 typed
questions — the jev-style agent loop. Average per request, CPU:

| state tokens | myna-stream | myna re-encode | speedup |
|---|---|---|---|
| 512 | 28 ms | 85 ms | 3.0x |
| 1,024 | 33 ms | 137 ms | 4.2x |
| 2,048 | 28 ms | 247 ms | 8.7x |
| 4,096 | 28 ms | 443 ms | 15.8x |

The streaming column is **flat by design**: cost is the question branches
plus the delta, never the prefix. (Numbers above from the v0 probe
checkpoint; the full checkpoint's table lands below.)

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
```

## What v0 is and isn't

v0 trains and evaluates on a synthetic typed-decisions corpus (three
workflows: ticket triage, incident triage, content moderation) with disjoint
train/dev/test vocabulary. It proves the architecture learns and streams;
it does **not** yet claim parity with laya/kev on their frozen suites — that
comparison is the next milestone in [PLAN.md](PLAN.md), along with RLCD-style
training, Metal kernels for the scan, and multilingual coverage.

Honest limits: word-order-heavy tasks can be partly solved bag-of-cues style;
mid-state edits (not appends) fall back to a full — still linear-time —
re-encode; the question-branch constant (~28 ms CPU today) is kernel work,
not architecture work.

Apache-2.0 · research log in [PLAN.md](PLAN.md) · questions to
[aashish254](https://github.com/aashish254)

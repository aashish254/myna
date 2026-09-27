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
uv run python bench/browser_expected.py                            # → browser/expected.json
npm ci && npm run selftest                                         # 7 checks, node
npm run g3                                                         # Chrome: bytes, cold, p50, shots
uv run python bench/quantize_int8.py                               # the int8 row above
```

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
uv run python -m bench.bench_mlx --ckpt runs/myna-v0 --reps 20 --drift-rows 12  # both tables
uv run python bench/diag_mlx_int8_gem.py --iters 200 --trials 7                 # the kernel price
```

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

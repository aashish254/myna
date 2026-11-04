# Draft 2 — the work the reviewers asked for that draft 1 cannot carry

Draft 1 is a complete submission. Draft 2 is what turns "major revision" into "accept". Ordered by
impact per hour of compute.

## 1. Converged in-domain baseline, 3 seeds  (~20 GPU-hours, 1 day)

All three reviewers called this the blocking issue. The 2-epoch run answers whether it matters; if
it does, run the full schedule:

```bash
kaggle kernels push -p release/kaggle   # edit run2ep.py: --epochs 3, --seeds 42 7 13
```

Report mean ± std and the paired interval per seed. If the gap closes below ~5 points, rewrite §V
and the abstract around that; if it holds near 14, the claim survives with a converged control.

## 2. A second Myna-v6 seed  (~6 GPU-hours, 1 day)

`release/train_v6.py` trains the packed readout through the engine's own renderer. Train one
additional seed, score the same test split with `bench/v6_audit.py`, and report both seeds. This
turns the single-checkpoint objection into a variance sentence. Note the routed arm needs the
set-aware head, so either retrain both heads or report the packed arm's two seeds and say so.

## 3. Out-of-domain cells  (~2 GPU-hours, half a day)

Hold out two sources entirely (e.g. `dbpedia14`, `trec`) and fine-tune on the other nine. Report
in-domain and held-out macro side by side. This is the cheapest way to answer "does the readout
generalize", and it does not need new data.

## 4. TF-IDF per-question records  (~1 hour, CPU)

Re-run the TF-IDF arm writing row outcomes, then `bench/paired_bootstrap.py` will produce its
paired interval. Right now the paper says the artifact is missing; this removes the sentence.

## 5. One external benchmark  (~3–5 days)

A JEVal or RAFT subset, schema-mapped into the typed contract. Expensive; do it only if a reviewer
insists that the self-curated suite is disqualifying.

## 6. Smaller items  (an afternoon each)

- Cohen's d and ECE sensitivity: **done in draft 1**.
- Docker reproduction: `Dockerfile` added; wire it into CI if you want a badge.
- Fairness audit: subgroup error rates need demographic labels the public corpora do not carry;
  scope it as future work rather than pretending.
- Glossary for internal names: the body no longer uses them, so this is optional.

## Order I would run it

2 seeds baseline (1) → second v6 seed (2) → OOD cells (3) → TF-IDF records (4). That is about two
GPU-days and it addresses four of the five blocking items from the TMLR review.

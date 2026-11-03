# Reproducibility package: Myna v6 measurement study

This directory holds the artifacts that make the paper's claims checkable without the
authors' machines. Everything below is a command plus the output it should print.

## What is released

| Artifact | Path | What it establishes |
|---|---|---|
| Frozen suite | `data/decision-v2-pilot/` | 16 cells over 11 public sources; train/dev/calibration/test |
| Checkpoint | `myna-v5-prototype 2/` | the shipped 446,631,182-parameter v6 engine |
| Reference trainer | `release/train_v6.py` | the packed-readout recipe, the same renderer the engine serves with, calibration fit |
| Consistency predicate | `src/myna/consistency.py` | the definition behind 3/172, 31/172, 32/172 |
| Evaluation harness | `bench/v6_audit.py` | per-question records; `--max-packed` for the routing ablation |
| Aggregation | `bench/paper_numbers.py` | every table from the released records, with a correctness guard |
| Per-question records | `runs/*.rowoutcomes.json` | Myna, Laya, VegaML on the same 1,176 bundles |
| External baselines | `runs/external_{majority,tfidf,nli}.json`, `paper/shared/external_baselines.py` | the three classical references |
| Correction record | `paper/MEASUREMENT-CORRECTION.md` | the ordinal off-by-one, its signature, before/after |
| Campaign notebooks | `release/notebooks/` | the v5 training loop and the v6 campaign: CAL split, temperature/conformal fit, v6 checkpoint assembly |
| CAL split builder | `release/build_cal_split.py` | the 1,999-bundle split the v6 temperatures were fitted on |

## Verify in this order

```bash
# 1. Re-score the routed arm (≈950 s CPU) and the packed-forced arm (≈540 s).
.venv/bin/python -m bench.v6_audit --ckpt "myna-v5-prototype 2" --split test --every 1 \
  --out runs/paper_v6_audit.json --md runs/paper_v6_audit.md \
  --metrics runs/paper_v6_test.metrics.json \
  --rowout runs/paper_myna_v6_test.rowoutcomes.json
.venv/bin/python -m bench.v6_audit --ckpt "myna-v5-prototype 2" --split test --every 1 \
  --max-packed 1000 --out runs/v6_packed_forced.json \
  --rowout runs/v6_packed_forced.rowoutcomes.json

# 2. Every aggregate in the paper, from those records.
.venv/bin/python -m bench.paper_numbers --out runs/paper_numbers.json
#    expects macro 0.7777 / overall 0.7882 / bundle 0.7509 for the routed arm,
#    and 0.7378 / 0.7368 / 0.6879 for the packed-forced arm.

# 3. Bundle consistency under the released predicate.
.venv/bin/python -m bench.consistency_audit
#    expects Myna 3/172, Laya 31/172, VegaML 32/172.

# 4. The intervals and the routing cost the paper quotes.
.venv/bin/python -m bench.paper_bootstrap --out runs/paper_bootstrap.json
#    expects macro 95% CI [0.7601, 0.7946] and bundle CI [0.7318, 0.7696].
.venv/bin/python -m bench.routing_cost --n 6 --out runs/routing_cost.json
#    expects about 4.9 s routed vs 0.43 s packed on banking77 (ratio ~11).

# 5. The CAL split and the calibration refit.
.venv/bin/python release/build_cal_split.py --check
#    expects 1,999 bundles / 2,548 questions, ids digest 692648492dfd3846c04e12fa84c7cbeabaa6822c686e975f10ccf8eec574dc67.
.venv/bin/python -m bench.cal_refit --reuse-rows
#    expects the ordinal temperature at 0.9 under the notebook's (mis-indexed) gold and at the
#    3.0 grid ceiling under the corrected gold; the full inference run takes ~45 min and writes
#    runs/v6_cal_raw.jsonl, which is committed so the refit itself is instant.

# 6. The reference trainer, end to end on a CPU-sized smoke run.
.venv/bin/python release/train_v6.py --steps 2 --limit 4 --out /tmp/v6_smoke --no-encoder
#    expects "packed_scorer reloads with strict=True".

# 7. The renderer-parity test: training and serving tokenize identically.
.venv/bin/python -m pytest -q tests/test_reference_trainer.py
```

## What is not released, and why

- **The original 1,999-row CAL file.** The split is *drawn*, not curated, and the draw is
  deterministic: `release/build_cal_split.py` groups the train pool by source, shuffles each
  group with `random.Random(42)`, and takes `max(20, round(2000 * n_source / n_train))` from
  each. That returns 1,999 bundles (the model card's "2,000" is the rounding target), 2,548
  questions. The file itself was never committed, so the builder is a definition-level
  reproduction: same ids, same composition, same digest, not a hash of the lost file.
- **Nothing else.** The v6 assembly cell is `04-v6-campaign.ipynb` (it writes
  `myna-v6-unified.pt` with the config, heads and audits), and the temperature/conformal fit
  is cell 5 of the same notebook, fitted on the CAL split above.

## What the notebooks settle

| Question the paper used to answer by assertion | What the released cells show |
|---|---|
| Was a consistency loss trained? | No. The v5 config defines `CONSISTENCY_WEIGHT = 1.0`, but no cell reads it; the loss is per-question cross-entropy (v5 notebook, cell 9). |
| Is the graph head doing anything? | No. `graph_gate` is 0.0 and the ablation prints identical predictions on/off (v5 notebook, cell 25); the v6 notebook runs the same 4-arm ablation. |
| Where were the v6 temperatures fitted? | On the CAL split drawn from train, by NLL over a 51-point grid 0.5--3.0, per primitive (v6 notebook, cell 5). The ordinal temperature reproduces at 0.9 under the notebook's gold and 3.0 under the corrected gold. |
| Where does 0/172 come from? | From a rule that only inspects choice--boolean pairs whose option text appears in the question; it cannot see yelp, whose rating is a `score` (v5 notebook, cell 25). |
| Why is the CORAL ordinal head disabled? | It was trained and gated on development accuracy against the packed head's 0.6542; it did not clear the gate (v6 notebook, cell 4). Both numbers were measured under the pre-correction score convention. |

## The one thing to read first

`paper/MEASUREMENT-CORRECTION.md`. It documents an off-by-one in the ordinal cell of the
evaluation harness that inflated the macro from 0.7777 to 0.8620, how the error was found, why
the competitor columns were unaffected, and which published quantities changed. The
pre-correction records are kept under `runs/*.predates-fix.*` so the arithmetic can be replayed
rather than trusted.

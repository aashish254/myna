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

# 5. The reference trainer, end to end on a CPU-sized smoke run.
.venv/bin/python release/train_v6.py --steps 2 --limit 4 --out /tmp/v6_smoke --no-encoder
#    expects "packed_scorer reloads with strict=True".

# 6. The renderer-parity test: training and serving tokenize identically.
.venv/bin/python -m pytest -q tests/test_reference_trainer.py
```

## What is not released, and why

- **The Colab notebook that produced the shipped weights.** It is not in the archive. The
  reference trainer implements the same recipe and is verified to render and serve through the
  same code path, but it does not reproduce the shipped trajectory, and the paper does not
  claim it does.
- **The 2,000-bundle calibration split** named in the model card. Only the 448-row split in
  `data/decision-v2-pilot/calibration.jsonl` is committed, and every number in the paper is
  computed from that one.
- **The original contradiction predicate.** It was never shipped; the released predicate in
  `src/myna/consistency.py` replaces it, and the paper reports the two numbers side by side
  (0/172 as recorded, 3/172 as measured) rather than adopting the record.

## The one thing to read first

`paper/MEASUREMENT-CORRECTION.md`. It documents an off-by-one in the ordinal cell of the
evaluation harness that inflated the macro from 0.7777 to 0.8620, how the error was found, why
the competitor columns were unaffected, and which published quantities changed. The
pre-correction records are kept under `runs/*.predates-fix.*` so the arithmetic can be replayed
rather than trusted.

# Draft 1 — status when you wake

## The paper, in one paragraph

A measurement study of a typed decision engine on a frozen 16-cell suite. The accuracy arm,
Myna-v6, reaches 0.8620 macro. Its fair comparison is the same backbone fine-tuned on the same
split with a conventional per-cell classifier: 0.7814 after two epochs over three seeds (0.7126 after
one), so Myna leads by 8.1 points, concentrated on multi-class choice cells. Cardinality routing is a budget repair worth
0.0399 macro on one cell; conformal sets are reported as a bounded result with three calibrations;
the streaming engine is a compute result that does not learn the suite, and the paper says so.

## What changed in this revision (all committed)

**Required by the reviewers**
- Table II arithmetic corrected: the other-15-cells figure is 0.8747, which reproduces both macros.
- Abstract now leads with the same-backbone comparison, not the external engines.
- Formalisation section: nine numbered equations (GLA recurrence, readout, packed scorer,
  set-aware scorer, objective, conformal quantile) plus Algorithm 1 for the inference path.
- Paired row-clustered tests: −DeBERTa +0.145 [0.122, 0.168], −Laya +0.196, −VegaML +0.246,
  McNemar p < 1e-38; Cohen's d 0.37–0.51.
- Conformal section now reports released (83.68), refit per primitive (86.94), disjoint
  temperature/quantile halves (87.29), pooled refit (89.17) and Mondrian (83.96), and states that
  the shortfall is calibration size plus within-cell shift, not a single global violation.
- CAL split committed with SHA-256 (`runs/cal_split.jsonl`, 1,999 bundles, 2,548 questions).
- Streaming arm demoted to a compute result; the VOID anti-prior ablation is gone.
- `noul` defined at first use; K=78 explained; bootstrap counts reconciled; ECE bin sensitivity
  reported (0.0184–0.0223 across 5–20 bins, 0.0162 adaptive); pre-training exposure and fairness
  discussed; CORAL, poly-encoders, ColBERT and Mondrian cited; repository paths moved out of the body.
- TMLR template: `myna_tmlr.pdf`, 23 pp, anonymous, official `tmlr.sty`.

## Files to upload

Everything is in `upload/` — see `SUBMISSION-GUIDE.md` for the form-by-form instructions.

## Numbers to know at a glance

| quantity | value |
|---|---|
| Myna-v6 macro / overall / bundle | 0.8620 / 0.8632 / 0.8393 |
| Same-backbone baseline, 1 epoch (2 seeds) | 0.7126 |
| Same-backbone baseline, 2 epochs | 0.7814 over three seeds (0.7746, 0.7813, 0.7884; sd 0.7) |
| Readout gap | 8.1 points (paired CIs [0.067,0.108], [0.059,0.102], [0.054,0.094]; McNemar p<1e-19) |
| Routing ablation | 0.0345 -> 0.6724 on banking77, +0.0399 macro |
| Consistency | 3/172 vs 10/172 (Laya), 29/172 (VegaML) |
| Conformal, released / refit / Mondrian | 83.68 / 89.17 / 83.96 at 90% |
| ECE | 0.0190 (0.0184–0.0223 across 5–20 bins) |

## What I could not fix overnight (and the paper says so)

1. The in-domain baseline is trained for 2 epochs, not the 3–5 a reviewer asked for; 3 seeds are
   still outstanding.
2. Myna-v6 has a single training seed.
3. No external benchmark (JEVal/RAFT); the suite is self-curated and the paper says so.
4. TF-IDF has no per-question release, so its paired interval is not computed.
5. No fairness audit across demographic subgroups.

These are draft-2 items: 3 seeds for the baseline, a second Myna-v6 seed, an out-of-domain cell,
TF-IDF per-question records and one external benchmark.

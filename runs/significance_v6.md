# Paired significance on the `test` split of `data/decision-v2-pilot`

Measured by `bench/significance.py` over 1176 rows / 1440 paired question instances, 20,000 row-clustered replicates, seed 20261031 (first replicate digest `d45e5c018ee550da`). myna is `myna-v6` at model `207b41fea5df`, temperature {'choice': 0.9, 'score': 0.9, 'noul': 0.85}; laya is `0.4.19` at commit `55a19f7b7b56`. Both engines are read row for row, and the unit resampled is `_meta.id`, not the question.

Point estimates reproduce the committed report exactly: macro 0.8620375366903146 against `0.8620375366903147` in `runs/v6_test.report.json`. Nothing here re-measures inference; it resamples outcomes that `bench/row_outcomes.py` and `bench/eval_laya_row_outcomes.py` dumped and guarded against that report.

## The headline numbers with their intervals

| statistic | point | 95% CI | what it is |
|---|---|---|---|
| `myna_macro` | 0.8620 | [0.8430, 0.8800] | cell-macro of myna over the published 16 cells — G1's number |
| `majority_floor` | 0.4331 | [0.4270, 0.4647] | the same macro for a per-cell constant predictor, recomputed in every replicate |
| `uniform_floor` | 0.3321 | [0.3313, 0.3330] | the same macro at 1/options, row-weighted |
| `margin_over_floor` | 0.4290 | [0.3900, 0.4424] | the difference G1's second half is judged on |
| `laya_macro` | 0.6664 | [0.6439, 0.6888] | laya over the same cells of the same rows |
| `gap_myna_minus_laya` | 0.1956 | [0.1701, 0.2210] | the paired difference, not two intervals subtracted |

## Claims, adjudicated

The rule was fixed before the intervals existed: an interval that covers a threshold forces the sentence to be rewritten in the same change, and a verdict that moves a gate is escalated rather than edited here.

- **stands** — myna's test-split macro is above its own majority floor. `margin_over_floor` = 0.4290 [0.3900, 0.4424], rule: CI excludes 0. the sentence as written stands.
- **stands** — the margin over the floor reaches G1's +0.15. `margin_over_floor` = 0.4290 [0.3900, 0.4424], rule: CI lower bound ≥ 0.15. the sentence as written stands.
- **stands** — the macro reaches G1's target of 0.7. `myna_macro` = 0.8620 [0.8430, 0.8800], rule: CI lower bound ≥ 0.7. the sentence as written stands.
- **does not stand** — laya's macro is above myna's on the same rows and cells. `gap_myna_minus_laya` = 0.1956 [0.1701, 0.2210], rule: CI excludes 0 on the negative side. the sentence must be rewritten in the same PR.

## McNemar cross-check

The bootstrap says how far apart the engines are; McNemar says whether the direction is decided by the rows at all. Both cells of a pair are read on the same `(row, question)`, so the only information is discordance.

- myna vs laya, pooled over 458 pairs: myna right and laya wrong 391, the reverse 67, exact two-sided p = 1.05e-56.
- myna vs each cell's majority label, pooled over 795 pairs: 746 in myna's favour, 49 against, p = 4.89e-161. This is the floor claim tested without a resample.

| cell | n pairs | myna | laya | Δ | myna-only | laya-only | McNemar p | 95% CI on the cell Δ |
|---|---|---|---|---|---|---|---|---|
| agnews/is_business | 43 | 0.9535 | 0.8837 | +0.0698 | 4 | 1 | 0.375 | [-0.0263, 0.1739] |
| agnews/is_scitech | 49 | 0.9184 | 0.9388 | -0.0204 | 1 | 2 | 1 | [-0.0943, 0.0465] |
| agnews/is_sports | 47 | 0.9787 | 0.9787 | +0.0000 | 0 | 0 | n/a (no discordant pairs) | [0.0000, 0.0000] |
| agnews/is_world | 45 | 1.0000 | 0.8000 | +0.2000 | 9 | 0 | 0.00391 | [0.0870, 0.3256] |
| agnews/topic | 116 | 0.9052 | 0.9138 | -0.0086 | 6 | 7 | 1 | [-0.0703, 0.0526] |
| amazon/stars | 80 | 0.6375 | 0.3000 | +0.3375 | 32 | 5 | 7.43e-06 | [0.2029, 0.4659] |
| banking77/intent | 116 | 0.6724 | 0.2759 | +0.3966 | 51 | 5 | 1.17e-10 | [0.2917, 0.5000] |
| boolq/answer | 80 | 0.8500 | 0.7000 | +0.1500 | 17 | 5 | 0.0169 | [0.0395, 0.2623] |
| contrastive/decision | 196 | 0.9847 | 0.5969 | +0.3878 | 77 | 1 | 5.23e-22 | [0.3184, 0.4573] |
| dbpedia14/category | 116 | 0.9569 | 0.7672 | +0.1897 | 27 | 5 | 0.000113 | [0.1000, 0.2793] |
| imdb/positive | 80 | 0.9625 | 0.4625 | +0.5000 | 43 | 3 | 4.62e-10 | [0.3718, 0.6222] |
| mnli/relation | 116 | 0.8621 | 0.5948 | +0.2672 | 38 | 7 | 3.12e-06 | [0.1626, 0.3725] |
| sst5/sentiment | 80 | 0.6125 | 0.3875 | +0.2250 | 30 | 12 | 0.00792 | [0.0694, 0.3750] |
| trec/answer_type | 116 | 0.9483 | 0.7500 | +0.1983 | 27 | 4 | 3.4e-05 | [0.1121, 0.2871] |
| yelp/rating | 80 | 0.6250 | 0.5250 | +0.1000 | 18 | 10 | 0.185 | [-0.0286, 0.2278] |
| yelp/recommend | 80 | 0.9250 | 0.7875 | +0.1375 | 11 | 0 | 0.000977 | [0.0652, 0.2208] |

## What is not in this table

- `1176` rows are the whole `test` split, so the interval is the sampling error of *this* split only. The rows are a fixed draw from eleven public datasets; nothing here generalises to their upstreams.
- Cells under `min_rows = 30` were dropped before resampling, as in the published report, so the macro is over 16 cells and not over every question the split asks.
- laya's confidence values are not usable here: its own loader warns that this checkpoint ships temperatures outside its [0.5, 5] clamp, which is recorded in `laya_warnings` in the JSON. Only correctness is compared.
- The interval is a percentile interval from a multinomial row resample. It is not a statement about a population of checkpoints: a second training run of the same 3,600-update dose is a different sampling layer and would need its own checkpoint, which this box has not produced.

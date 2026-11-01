# Paired significance on the `test` split of `data/decision-v2-pilot`

Measured by `bench/significance.py` over 1176 rows / 1440 paired question instances, 20,000 row-clustered replicates, seed 20261031 (first replicate digest `d45e5c018ee550da`). myna is `myna` at model `0e73a8f07edc`, temperature 1.2; laya is `0.4.19` at commit `55a19f7b7b56`. Both engines are read row for row, and the unit resampled is `_meta.id`, not the question.

Point estimates reproduce the committed report exactly: macro 0.4893271976084137 against `0.4893271976084137` in `runs/v1b_kaggle_3600b.report.json`. Nothing here re-measures inference; it resamples outcomes that `bench/row_outcomes.py` and `bench/eval_laya_row_outcomes.py` dumped and guarded against that report.

## The headline numbers with their intervals

| statistic | point | 95% CI | what it is |
|---|---|---|---|
| `myna_macro` | 0.4893 | [0.4647, 0.5139] | cell-macro of myna over the published 16 cells — G1's number |
| `majority_floor` | 0.4331 | [0.4270, 0.4647] | the same macro for a per-cell constant predictor, recomputed in every replicate |
| `uniform_floor` | 0.3321 | [0.3313, 0.3330] | the same macro at 1/options, row-weighted |
| `margin_over_floor` | 0.0563 | [0.0192, 0.0676] | the difference G1's second half is judged on |
| `laya_macro` | 0.6664 | [0.6439, 0.6888] | laya over the same cells of the same rows |
| `gap_myna_minus_laya` | -0.1771 | [-0.2080, -0.1463] | the paired difference, not two intervals subtracted |

## Claims, adjudicated

The rule was fixed before the intervals existed: an interval that covers a threshold forces the sentence to be rewritten in the same change, and a verdict that moves a gate is escalated rather than edited here.

- **stands** — myna's test-split macro is above its own majority floor. `margin_over_floor` = 0.0563 [0.0192, 0.0676], rule: CI excludes 0. the sentence as written stands.
- **does not stand** — the margin over the floor reaches G1's +0.15. `margin_over_floor` = 0.0563 [0.0192, 0.0676], rule: CI lower bound ≥ 0.15. the sentence must be rewritten in the same PR.
- **does not stand** — the macro reaches G1's target of 0.7. `myna_macro` = 0.4893 [0.4647, 0.5139], rule: CI lower bound ≥ 0.7. the sentence must be rewritten in the same PR.
- **stands** — laya's macro is above myna's on the same rows and cells. `gap_myna_minus_laya` = -0.1771 [-0.2080, -0.1463], rule: CI excludes 0 on the negative side. the sentence as written stands.

## McNemar cross-check

The bootstrap says how far apart the engines are; McNemar says whether the direction is decided by the rows at all. Both cells of a pair are read on the same `(row, question)`, so the only information is discordance.

- myna vs laya, pooled over 594 pairs: myna right and laya wrong 158, the reverse 436, exact two-sided p = 4.75e-31.
- myna vs each cell's majority label, pooled over 487 pairs: 291 in myna's favour, 196 against, p = 1.94e-05. This is the floor claim tested without a resample.

| cell | n pairs | myna | laya | Δ | myna-only | laya-only | McNemar p | 95% CI on the cell Δ |
|---|---|---|---|---|---|---|---|---|
| agnews/is_business | 43 | 0.6512 | 0.8837 | -0.2326 | 2 | 12 | 0.0129 | [-0.3913, -0.0750] |
| agnews/is_scitech | 49 | 0.6531 | 0.9388 | -0.2857 | 2 | 16 | 0.00131 | [-0.4400, -0.1356] |
| agnews/is_sports | 47 | 0.8085 | 0.9787 | -0.1702 | 1 | 9 | 0.0215 | [-0.3000, -0.0492] |
| agnews/is_world | 45 | 0.7778 | 0.8000 | -0.0222 | 7 | 8 | 1 | [-0.1944, 0.1500] |
| agnews/topic | 116 | 0.3707 | 0.9138 | -0.5431 | 2 | 65 | 3.09e-17 | [-0.6387, -0.4444] |
| amazon/stars | 80 | 0.2375 | 0.3000 | -0.0625 | 11 | 16 | 0.442 | [-0.1910, 0.0633] |
| banking77/intent | 116 | 0.0345 | 0.2759 | -0.2414 | 2 | 30 | 2.46e-07 | [-0.3274, -0.1587] |
| boolq/answer | 80 | 0.5000 | 0.7000 | -0.2000 | 10 | 26 | 0.0113 | [-0.3418, -0.0571] |
| contrastive/decision | 196 | 0.5051 | 0.5969 | -0.0918 | 30 | 48 | 0.0535 | [-0.1791, -0.0049] |
| dbpedia14/category | 116 | 0.4569 | 0.7672 | -0.3103 | 11 | 47 | 2.03e-06 | [-0.4262, -0.1939] |
| imdb/positive | 80 | 0.8000 | 0.4625 | +0.3375 | 29 | 2 | 4.63e-07 | [0.2237, 0.4545] |
| mnli/relation | 116 | 0.3276 | 0.5948 | -0.2672 | 11 | 42 | 2.25e-05 | [-0.3805, -0.1530] |
| sst5/sentiment | 80 | 0.2875 | 0.3875 | -0.1000 | 13 | 21 | 0.229 | [-0.2439, 0.0441] |
| trec/answer_type | 116 | 0.3190 | 0.7500 | -0.4310 | 4 | 54 | 3.17e-12 | [-0.5327, -0.3273] |
| yelp/rating | 80 | 0.3625 | 0.5250 | -0.1625 | 12 | 25 | 0.047 | [-0.3077, -0.0141] |
| yelp/recommend | 80 | 0.7375 | 0.7875 | -0.0500 | 11 | 15 | 0.557 | [-0.1750, 0.0753] |

## What is not in this table

- `1176` rows are the whole `test` split, so the interval is the sampling error of *this* split only. The rows are a fixed draw from eleven public datasets; nothing here generalises to their upstreams.
- Cells under `min_rows = 30` were dropped before resampling, as in the published report, so the macro is over 16 cells and not over every question the split asks.
- laya's confidence values are not usable here: its own loader warns that this checkpoint ships temperatures outside its [0.5, 5] clamp, which is recorded in `laya_warnings` in the JSON. Only correctness is compared.
- The interval is a percentile interval from a multinomial row resample. It is not a statement about a population of checkpoints: a second training run of the same 3,600-update dose is a different sampling layer and would need its own checkpoint, which this box has not produced.


# Vegaml as a control on the `test` split of `data/decision-v2-pilot`

Measured by `bench/vegaml_control.py`, which imports `pack`, `statize`, `boot`, `ci`, `mcnemar_exact` and `_p` from `bench/significance.py` rather than restating them. 1176 rows / 1440 paired question instances, 20,000 row-clustered replicates, seed 20261031, first replicate digest `d45e5c018ee550da` — the resample depends only on the row count and the seed, so this is the same draw that produced `runs/significance.json`'s intervals.

The control is `vegaml` 0.15.0 reading `Qwen3.5-0.8B` on `cpu`, mode `engine`, views 1, torch 2.15.0 / transformers 5.22.1. myna is `myna` at model `0e73a8f07edc`.

## The second-engine slot

`bench/significance.py` names its competitor columns `laya_macro` and `gap_myna_minus_laya` and its published intervals use those names; this harness puts a different dump in that slot and writes the results as `vegaml_macro` and `gap_myna_minus_vegaml`. The mapping is `{"laya_macro": "vegaml_macro", "gap_myna_minus_laya": "gap_myna_minus_vegaml"}`, recorded in the JSON so the relabel is auditable.

| statistic | point | 95% CI |
|---|---|---|
| `myna_macro` | 0.4893 | [0.4647, 0.5139] |
| `majority_floor` | 0.4331 | [0.4270, 0.4647] |
| `uniform_floor` | 0.3321 | [0.3313, 0.3330] |
| `margin_over_floor` | 0.0563 | [0.0192, 0.0676] |
| `vegaml_macro` | 0.6160 | [0.5898, 0.6415] |
| `gap_myna_minus_vegaml` | -0.1266 | [-0.1578, -0.0955] |

myna's own four statistics are re-derived from the rows and agree with `v1b_kaggle_3600b.report.json` to 1e-9 before anything is resampled (`myna_macro, majority_floor, uniform_floor, margin_over_floor`).

## Claims

- **stands** — myna and vegaml differ on the shared rows (rule: CI excludes 0).
- **stands** — vegaml's macro is above myna's on the same rows (rule: CI excludes 0 on the negative side).

## McNemar

- myna vs vegaml, pooled over 609 discordant pairs: myna right and the control wrong 205, the reverse 404, exact two-sided p = 5.69e-16.

## The control's own health

- abstained `{"n": 631, "rate": 0.43819444444444444, "accuracy_of_abstained": 0.46275752773375595, "mean_set_size": 1.3722222222222222, "rule": "scored forced-choice; the flag is recorded, not applied"}`
- `unbound` — the engine's other decline flag, a distance-from-manifold test rather than a confidence one: 261 of 1440 outcomes, 178 of them also flagged `abstain`
- argmax disagreements against the engine's own probability table: 0
- rows/questions that raised or went unanswered: 0
- reader tokens read across the pass: 300193 in 928.9 s
- warnings emitted: []

Scores are forced-choice: both of the control's decline flags are recorded, never applied, because neither myna nor laya can decline. Read the abstain and unbound rates with the accuracy.

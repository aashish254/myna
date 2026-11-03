# Measurement correction: the `score` primitive was scored one level off

**Status:** corrected 2026-11-04. Every number in the manuscript dated before this file is
re-derived; the pre-correction witnesses are kept under `*.predates-fix.*` and are not cited.

## What was wrong

The suite stores an ordinal answer as a **1-based star level** (`"label": 4` means four
stars) while the option vector the model scores is **0-based** (`ranked[3]` is the four-star
option). `bench/v6_audit.py` compared the two directly:

```python
hit = top == ranked[golds[i]]          # pre-fix: golds[i] is 1..5, ranked is 0..4
```

so a prediction was recorded as correct when it named the level **above** gold. The
prediction itself was fine — the bookkeeping was not.

## Evidence

For the 240 `score` questions in the frozen test split:

| | pre-fix | corrected |
|---|---:|---:|
| amazon/stars | 0.6375 | 0.1750 |
| sst5/sentiment | 0.6125 | 0.1500 |
| yelp/rating | 0.6250 | 0.2000 |
| score primitive | 0.6250 | 0.1750 |

Two facts fix the interpretation:

1. On a directly inspected yelp row (`yelp/test/2720`, gold 3 stars) the engine's argmax is
   the *"3 stars: average"* option at p=0.574, and the pre-fix record says `ok=0`.
2. The corrected rate (0.175) is at the uniform floor for five options (0.20). The pre-fix
   rate (0.625) is the rate at which `pred_index == gold_level`, which is the signature of
   the off-by-one, not of a working ordinal head.

## Effect on the headline

| metric | pre-fix | corrected |
|---|---:|---:|
| 16-cell macro | 0.8620 | 0.7777 |
| overall accuracy | 0.8632 | 0.7882 |
| joint bundle exact | 0.8393 | 0.7509 |
| ECE (10 bins) | 0.0190 | 0.0637 |
| Brier (top-1) | 0.0434 | 0.1187 |
| selective accuracy @ 60% | 0.9873 | 0.9653 |
| conformal 90% coverage (shipped quantiles) | 0.8549 | 0.7715 |
| conformal 90% coverage (refitted on corrected gold) | — | 0.8778 |

The relative ordering of the systems does not change (Myna v6 0.7777 still leads the
supervised TF-IDF baseline at 0.7111 and the best Laya configuration at 0.7050), but the
margin shrinks and the absolute claims change. The `score` primitive is now reported as
**at chance** — a finding in its own right, since a TF-IDF classifier reaches 0.525 on
amazon/stars where the fine-tuned model reaches 0.175.

The same target reached the calibration stage. The v6 notebook fitted the temperatures and the
conformal quantiles on a CAL split drawn from the training pool, and its score loader carries the
same 1-based/0-based mismatch. Refitting on that split (`bench/cal_refit.py`) reproduces the stored
ordinal temperature exactly under the notebook's gold (0.9) and moves it to the grid ceiling (3.0)
under the corrected gold; refitting the quantiles lifts ordinal coverage from 38.3% to 90.4% and
overall 90% coverage from 77.15% to 87.78%, with ECE 0.0637 to 0.0326 (`bench/refit_coverage.py`).

## Why the competitor columns are unaffected

Laya and VegaML store `pred` and `gold` in the same space, so `ok == (pred == gold)` holds
for all 240 of their score questions. Their numbers are unchanged by this correction; the
Myna arms were the only ones scored by `bench/v6_audit.py`.

## The fix

- `bench/v6_audit.py` subtracts one for `score`: `hit = top == ranked[int(golds[i]) - 1]`.
- Row outcomes now store `pred` as the 1-based level, so `pred == gold` is correct without a
  second convention.
- `bench/paper_numbers.py` recomputes every aggregate from row outcomes and accepts legacy
  files with `--legacy-score`, so the two conventions cannot be silently mixed.
- `runs/paper_myna_v6_test.predates-fix.rowoutcomes.json` keeps the old file; the correction
  reproduces `0.7777` from it, which is the regression test that the fix is arithmetic and
  not a re-run artifact.

## The general lesson

A benchmark harness that stores labels and predictions in one space can still compare them
in another. The failure was invisible for two months because the corrupted cells were
plausible (0.62 on a 5-way task looks like a decent ordinal model) and because nothing in
the pipeline compared `pred` to `gold` in the raw records. `bench/paper_numbers.py` now
does that comparison on every load; a mismatch between `ok` and `pred == gold` raises.

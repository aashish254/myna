---
license: apache-2.0
library_name: transformers
base_model: microsoft/deberta-v3-large
tags:
- text-classification
- zero-shot
- decision-making
- agent
- conformal-prediction
- calibrated
model_name: myna-v6-unified
---

# Myna v6 Unified

A fine-tuned `microsoft/deberta-v3-large` encoder with a **cardinality-routed readout**:
questions with up to 16 options are scored from one packed sequence, and any question with
more is routed to a set-aware head that scores each option against the competition in its
bundle. Temperatures and split-conformal quantiles are fitted on a CAL split drawn from the
training pool.

> **Measurement correction (2026-11-04).** An off-by-one in the evaluation harness scored the
> ordinal (`score`) cells one level off, and the same shifted target was used when the stored
> temperatures and quantiles were fitted. Corrected, this checkpoint scores **0.7777** macro
> (not 0.8620) and the `score` primitive is at chance. The correction record is
> [paper/MEASUREMENT-CORRECTION.md](https://github.com/aashish254/myna/blob/main/paper/MEASUREMENT-CORRECTION.md).
> Figures below are the corrected ones; the pre-correction numbers are kept only in that record.

## 16-cell results on `decision-v2-pilot/test.jsonl`

1,176 bundles / 1,440 questions, one frozen split, identical rows for every column.

| arm | macro | overall | bundle exact | footing |
| :-- | --: | --: | --: | :-- |
| Myna v6 (routed) | 0.7777 | 0.7882 | 0.7509 | in-domain on all 16 cells |
| Myna v6 (packed everywhere) | 0.7378 | 0.7368 | 0.6879 | routing off, same weights |
| TF-IDF + logistic regression | 0.7111 | 0.6840 | 0.6293 | in-domain, text only |
| Laya 0.4.19 (typed-decisions best) | 0.7050 | 0.6792 | 0.6182 | 6/16 cells in-domain |
| VegaML 0.15.0 | 0.6160 | 0.5833 | 0.5119 | zero-shot |
| Train-prior majority | 0.3726 | 0.2965 | 0.2041 | in-domain |
| Zero-shot NLI | 0.3278 | 0.3465 | 0.3027 | zero-shot |

**Footing.** Myna v6 is fine-tuned on the suite's train split and is in-domain on all 16
cells; Laya and VegaML are out-of-domain or zero-shot here. The comparison is not a
like-for-like training-budget comparison, and the numbers above are not a leaderboard claim.

## Per cell (Myna v6, corrected)

| cell | n | accuracy |
| :-- | --: | --: |
| `agnews/is_business` | 43 | 0.9535 |
| `agnews/is_scitech` | 49 | 0.9184 |
| `agnews/is_sports` | 47 | 0.9787 |
| `agnews/is_world` | 45 | 1.0000 |
| `agnews/topic` | 116 | 0.9052 |
| `amazon/stars` | 80 | 0.1750 |
| `banking77/intent` | 116 | 0.6724 |
| `boolq/answer` | 80 | 0.8500 |
| `contrastive/decision` | 196 | 0.9847 |
| `dbpedia14/category` | 116 | 0.9569 |
| `imdb/positive` | 80 | 0.9625 |
| `mnli/relation` | 116 | 0.8621 |
| `sst5/sentiment` | 80 | 0.1500 |
| `trec/answer_type` | 116 | 0.9483 |
| `yelp/rating` | 80 | 0.2000 |
| `yelp/recommend` | 80 | 0.9250 |

## Calibration and selective prediction

- test ECE (10 bins) **0.0637**, top-1 Brier **0.1187**
- selective accuracy at 60% coverage **0.9653**
- selective accuracy at 90% coverage **0.8318**
- split-conformal sets at 90%: see the paper. The shipped quantiles were fitted against the
  shifted ordinal target and under-cover on test, so the sets are a ranking signal here, not a
  coverage guarantee.

Refitted on the rebuilt CAL split under the corrected target, the same engine reaches:
90% sets cover **0.8778**, 95% sets cover **0.9451**, ECE **0.0326** (against 0.7715 and 0.8639 shipped).

## Quickstart

```python
from myna import Myna

engine = Myna("Aashish254/myna-v6")
out = engine.predict(
    state="I was charged twice for the same order.",
    questions={"intent": {"type": "choice",
                          "instructions": "What is the customer asking about?",
                          "criteria": {"billing": "a charge or invoice problem",
                                       "shipping": "a delivery problem",
                                       "account": "a login or profile problem"}}},
)
print(out)
```

## What this card does not claim

- Not a streaming engine: v6 re-encodes the whole state on every question.
- Not state of the art: the comparison footing above is part of the result.
- The `score` primitive is at chance under the corrected gold; do not use it for ordinal
  decisions without re-training that head.

Full paper, code, and per-question records: https://github.com/aashish254/myna

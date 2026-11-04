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

## 16-cell results on `decision-v2-pilot/test.jsonl`

1,176 bundles / 1,440 questions, one frozen split, identical rows for every column.

| arm | macro | overall | bundle exact | footing |
| :-- | --: | --: | --: | :-- |
| Myna v6 (routed) | 0.8620 | 0.8632 | 0.8393 | in-domain on all 16 cells |
| Myna v6 (packed everywhere) | 0.8222 | 0.8118 | 0.7764 | routing off, same weights |
| TF-IDF + logistic regression | 0.7111 | 0.6840 | 0.6293 | in-domain, text only |
| Laya 0.4.19 (typed-decisions best) | 0.7050 | 0.6792 | 0.6182 | 6/16 cells in-domain |
| VegaML 0.15.0 | 0.6160 | 0.5833 | 0.5119 | zero-shot |
| Train-prior majority | 0.3726 | 0.2965 | 0.2041 | in-domain |
| Zero-shot NLI | 0.3278 | 0.3465 | 0.3027 | zero-shot |

**Footing.** Myna v6 is fine-tuned on the suite's train split and is in-domain on all 16
cells; Laya and VegaML are out-of-domain or zero-shot here. The comparison is not a
like-for-like training-budget comparison, and the numbers above are not a leaderboard claim.

## Per cell (Myna v6)

| cell | n | accuracy |
| :-- | --: | --: |
| `agnews/is_business` | 43 | 0.9535 |
| `agnews/is_scitech` | 49 | 0.9184 |
| `agnews/is_sports` | 47 | 0.9787 |
| `agnews/is_world` | 45 | 1.0000 |
| `agnews/topic` | 116 | 0.9052 |
| `amazon/stars` | 80 | 0.6375 |
| `banking77/intent` | 116 | 0.6724 |
| `boolq/answer` | 80 | 0.8500 |
| `contrastive/decision` | 196 | 0.9847 |
| `dbpedia14/category` | 116 | 0.9569 |
| `imdb/positive` | 80 | 0.9625 |
| `mnli/relation` | 116 | 0.8621 |
| `sst5/sentiment` | 80 | 0.6125 |
| `trec/answer_type` | 116 | 0.9483 |
| `yelp/rating` | 80 | 0.6250 |
| `yelp/recommend` | 80 | 0.9250 |

## Calibration and selective prediction

- test ECE (10 bins) **0.0190**, multiclass Brier **0.0449**
- selective accuracy at 60% coverage **0.9815**
- selective accuracy at 90% coverage **0.9035**
- split-conformal sets: 90% sets cover **83.68%** and 95% sets cover **93.33%** on test, so
  they are a ranking signal here, not a coverage guarantee.

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
- The ordinal (`score`) cells score about 0.63, well above the 0.20 floor but below the
  boolean cells; do not read them as a calibrated five-point scale.

Full paper, code, and per-question records: https://github.com/aashish254/myna

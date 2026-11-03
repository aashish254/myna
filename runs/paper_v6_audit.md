# v6 re-score on test — uv run python -m bench.v6_audit --ckpt "myna-v5-prototype 2" --split test --every 1 --out runs/paper_v6_audit.json --md runs/paper_v6_audit.md --metrics runs/paper_v6_test.metrics.json --rowout runs/paper_myna_v6_test.rowoutcomes.json

1176 bundles / 1440 questions / 16 cells | macro 0.7777 | overall 0.7882 | abstained 119 | 972.5s wall

| cell | correct | n | acc | abstained |
| :-- | --: | --: | --: | --: |
| `agnews/is_business` | 41 | 43 | 0.9535 | 11 |
| `agnews/is_scitech` | 45 | 49 | 0.9184 | 4 |
| `agnews/is_sports` | 46 | 47 | 0.9787 | 1 |
| `agnews/is_world` | 45 | 45 | 1.0000 | 6 |
| `agnews/topic` | 105 | 116 | 0.9052 | 1 |
| `amazon/stars` | 14 | 80 | 0.1750 | 0 |
| `banking77/intent` | 78 | 116 | 0.6724 | 26 |
| `boolq/answer` | 68 | 80 | 0.8500 | 30 |
| `contrastive/decision` | 193 | 196 | 0.9847 | 11 |
| `dbpedia14/category` | 111 | 116 | 0.9569 | 0 |
| `imdb/positive` | 77 | 80 | 0.9625 | 8 |
| `mnli/relation` | 100 | 116 | 0.8621 | 1 |
| `sst5/sentiment` | 12 | 80 | 0.1500 | 0 |
| `trec/answer_type` | 110 | 116 | 0.9483 | 0 |
| `yelp/rating` | 16 | 80 | 0.2000 | 0 |
| `yelp/recommend` | 74 | 80 | 0.9250 | 20 |

| banking77 variant | correct/n |
| :-- | --: |
| `clean` | 60/80 |
| `none_absent` | 0/12 |
| `none_present` | 9/12 |
| `permuted` | 9/12 |

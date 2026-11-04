# Recomputed paper numbers

| arm | macro | overall | bundle exact | n |
|---|---:|---:|---:|---:|
| myna_v6 | 0.8620 | 0.8632 | 0.8393 | 1440 |
| myna_v6_prefix | 0.8620 | 0.8632 | 0.8393 | 1440 |
| myna_v6_packed_forced | 0.8222 | 0.8118 | 0.7764 | 1440 |
| laya_0419 | 0.6664 | 0.6382 | 0.5663 | 1440 |
| vegaml_0150 | 0.6160 | 0.5833 | 0.5119 | 1440 |
| majority | 0.3726 | 0.2965 | 0.2041 | 1440 |
| tfidf | 0.7111 | 0.6840 | 0.6293 | 1440 |
| nli | 0.3278 | 0.3465 | 0.3027 | 1440 |

## Myna v6 per cell

| cell | n | acc |
|---|---:|---:|
| agnews/is_business | 43 | 0.9535 |
| agnews/is_scitech | 49 | 0.9184 |
| agnews/is_sports | 47 | 0.9787 |
| agnews/is_world | 45 | 1.0000 |
| agnews/topic | 116 | 0.9052 |
| amazon/stars | 80 | 0.6375 |
| banking77/intent | 116 | 0.6724 |
| boolq/answer | 80 | 0.8500 |
| contrastive/decision | 196 | 0.9847 |
| dbpedia14/category | 116 | 0.9569 |
| imdb/positive | 80 | 0.9625 |
| mnli/relation | 116 | 0.8621 |
| sst5/sentiment | 80 | 0.6125 |
| trec/answer_type | 116 | 0.9483 |
| yelp/rating | 80 | 0.6250 |
| yelp/recommend | 80 | 0.9250 |

ECE 0.0190; Brier 0.0879; selective@60% 0.9815 (k=864)

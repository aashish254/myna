`python bench/eval_scratch.py --split test --seeds 0 1 --out runs/scratch_decision_v2_test.md --metrics-out runs/scratch_metrics_test.json`

load average 15.99/10.66/7.22, device `cpu`, tokenizer `data/decision-v2-pilot/tokenizer-8192.json` (vocab 8192), weights: `torch.manual_seed(seed)` then the default initialiser — no gradient step, on any corpus.

```
stratified test — one row per (source, question) cell, n = rows in it

shared-instruction — 9 source(s): agnews, amazon, banking77, contrastive, dbpedia14, imdb, sst5, trec, yelp
source/question              type     n  sets   model    laya     maj    unif    -maj
agnews/is_business           noul    43    41   0.465     —     0.698   0.500  -0.233
agnews/is_scitech            noul    49    48   0.571     —     0.653   0.500  -0.082
agnews/is_sports             noul    47    46   0.532     —     0.809   0.500  -0.277
agnews/is_world              noul    45    43   0.622     —     0.778   0.500  -0.156
agnews/topic               choice   116   113   0.276     —     0.328   0.245  -0.052
amazon/stars                score    80     1   0.175     —     0.263   0.200  -0.088
banking77/intent           choice   116   116   0.009     —     0.043   0.013  -0.034
contrastive/decision         noul   196    29   0.372     —     0.454   0.396  -0.082
dbpedia14/category         choice   116    37   0.095     —     0.129   0.071  -0.034
imdb/positive                noul    80     1   0.438     —     0.537   0.500  -0.100
sst5/sentiment              score    80     4   0.125     —     0.250   0.200  -0.125
trec/answer_type           choice   116    37   0.138     —     0.284   0.164  -0.147
yelp/rating                 score    80     4   0.287     —     0.275   0.200   0.012
yelp/recommend               noul    80     4   0.562     —     0.537   0.500   0.025
  macro over its 14 scored cells                      0.333     —     0.431   0.321  -0.098

per-row-instruction — 2 source(s): boolq, mnli
source/question              type     n  sets   model    laya     maj    unif    -maj
boolq/answer                 noul    80    80   0.500     —     0.537   0.500  -0.037
mnli/relation              choice   116   112   0.371     —     0.353   0.325   0.017
  macro over its 2 scored cells                      0.435     —     0.445   0.412  -0.010
```

MACRO, 2 seeds: seed 0 = 0.346, seed 1 = 0.351; mean **0.348**, spread 0.005. Floors over the same 16 cells: majority 0.433, uniform 0.332.

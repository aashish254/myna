# Bundle consistency: released predicate and the recovered campaign rule

| engine | bundles | released | campaign | agnews | yelp |
|---|---:|---:|---:|---:|---:|
| Myna v6 | 172 | 3 | 0 | 2 | 1 |
| Laya 0.4.19 | 172 | 31 | 1 | 5 | 26 |
| VegaML 0.15.0 | 172 | 32 | 2 | 26 | 6 |
| Myna v6 (campaign row outcomes) | 172 | 10 | 0 | 2 | 8 |

Predicate: `src/myna/consistency.py`; recompute with `python -m bench.consistency_audit`.

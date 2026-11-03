# Bundle consistency under the released predicate

| engine | bundles | contradictions | rate | agnews | yelp |
|---|---:|---:|---:|---:|---:|
| Myna v6 | 172 | 3 | 0.0174 | 2 | 1 |
| Laya 0.4.19 | 172 | 31 | 0.1802 | 5 | 26 |
| VegaML 0.15.0 | 172 | 32 | 0.1860 | 26 | 6 |
| Myna v6 (campaign row outcomes) | 172 | 10 | 0.0581 | 2 | 8 |

Predicate: `src/myna/consistency.py`; recompute with `python -m bench.consistency_audit`.

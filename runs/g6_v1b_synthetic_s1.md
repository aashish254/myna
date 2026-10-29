`python bench/eval_synthetic.py --ckpt runs/v1b-kaggle-3600b --device cpu --seed 1 --n-train 4000 --n-eval 1200 --out runs/g6_v1b_synthetic_s1.md`

load average 3.76/4.31/5.00, device `cpu`, checkpoint `runs/v1b-kaggle-3600b`, vocab 8192, params 16,926,848.

Split recipe: seed `1`, `--n-train 4000`, `--n-eval 1200` per workflow, `myna.train`'s own train→dev→test draw order over `WORKFLOWS` = ['support', 'incident', 'moderation']. test split sha256 `fc9cbe3b0aec5c94fbe3fa5e6c3126c4fb67d823f138b5cd67b3a8b484182235`.

```
=== test === (maj = modal-label floor, uni = chance, both on these same rows)
incident/page_oncall               0.5050   maj 0.5008  uni 0.5000
incident/severity                  0.2825   maj 0.2575  uni 0.2500
incident/team                      0.2917   maj 0.2642  uni 0.2500
moderation/action                  0.3725   maj 0.3358  uni 0.3333
moderation/category                0.2475   maj 0.2642  uni 0.2500
moderation/repeat_appeal           0.5558   maj 0.5058  uni 0.5000
support/churn_risk                 0.4675   maj 0.5142  uni 0.5000
support/department                 0.1625   maj 0.2117  uni 0.2000
support/urgency                    0.2550   maj 0.2625  uni 0.2500
```

MACRO over the 9 synthetic test cells: **0.3489** (dev on the same recipe: 0.3501).

MACRO floors over the same 9 cells, read off these rows: majority **0.3463**, uniform (chance) **0.3370**.

G6 asks for **≥ 0.94** on this split, so on these weights the gate is **NOT met** at **0.3489**.

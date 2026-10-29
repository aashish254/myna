`python bench/eval_synthetic.py --ckpt runs/myna-v0 --device cpu --seed 1 --n-train 4000 --n-eval 1200 --out runs/g6_v0_synthetic_s1.md`

load average 3.91/4.96/5.40, device `cpu`, checkpoint `runs/myna-v0`, vocab 1740, params 14,449,280.

Split recipe: seed `1`, `--n-train 4000`, `--n-eval 1200` per workflow, `myna.train`'s own train→dev→test draw order over `WORKFLOWS` = ['support', 'incident', 'moderation']. test split sha256 `fc9cbe3b0aec5c94fbe3fa5e6c3126c4fb67d823f138b5cd67b3a8b484182235`.

```
=== test === (maj = modal-label floor, uni = chance, both on these same rows)
incident/page_oncall               0.9367   maj 0.5008  uni 0.5000
incident/severity                  0.9542   maj 0.2575  uni 0.2500
incident/team                      0.9617   maj 0.2642  uni 0.2500
moderation/action                  0.9825   maj 0.3358  uni 0.3333
moderation/category                0.9492   maj 0.2642  uni 0.2500
moderation/repeat_appeal           0.9375   maj 0.5058  uni 0.5000
support/churn_risk                 0.9542   maj 0.5142  uni 0.5000
support/department                 0.9467   maj 0.2117  uni 0.2000
support/urgency                    0.9242   maj 0.2625  uni 0.2500
```

MACRO over the 9 synthetic test cells: **0.9496** (dev on the same recipe: 0.9588).

MACRO floors over the same 9 cells, read off these rows: majority **0.3463**, uniform (chance) **0.3370**.

G6 asks for **≥ 0.94** on this split, so on these weights the gate is **met** at **0.9496**.

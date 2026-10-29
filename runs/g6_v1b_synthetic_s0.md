`python bench/eval_synthetic.py --ckpt runs/v1b-kaggle-3600b --device cpu --seed 0 --n-train 4000 --n-eval 1200 --out runs/g6_v1b_synthetic_s0.md`

load average 5.86/6.30/5.89, device `cpu`, checkpoint `runs/v1b-kaggle-3600b`, vocab 8192, params 16,926,848.

Split recipe: seed `0`, `--n-train 4000`, `--n-eval 1200` per workflow, `myna.train`'s own train→dev→test draw order over `WORKFLOWS` = ['support', 'incident', 'moderation']. test split sha256 `998d73757872d426564b4cb25356e621e5d2a6eca2e45bf140d066bd1de5abee`.

```
=== test === (maj = modal-label floor, uni = chance, both on these same rows)
incident/page_oncall               0.4883   maj 0.5092  uni 0.5000
incident/severity                  0.2925   maj 0.2667  uni 0.2500
incident/team                      0.2850   maj 0.2708  uni 0.2500
moderation/action                  0.3358   maj 0.3658  uni 0.3333
moderation/category                0.2425   maj 0.2633  uni 0.2500
moderation/repeat_appeal           0.5500   maj 0.5167  uni 0.5000
support/churn_risk                 0.4492   maj 0.5158  uni 0.5000
support/department                 0.1842   maj 0.2208  uni 0.2000
support/urgency                    0.2533   maj 0.2625  uni 0.2500
```

MACRO over the 9 synthetic test cells: **0.3423** (dev on the same recipe: 0.3446).

MACRO floors over the same 9 cells, read off these rows: majority **0.3546**, uniform (chance) **0.3370**.

G6 asks for **≥ 0.94** on this split, so on these weights the gate is **NOT met** at **0.3423**.

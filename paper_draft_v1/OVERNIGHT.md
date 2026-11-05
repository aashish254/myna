# What ran while you slept

| job | what | ETA | output |
|---|---|---|---|
| `myna-baseline-2ep-s13` | third seed, 2 epochs | ~10:30 | `runs/baseline_seed13_2ep.json` |
| `myna-baseline-3ep-s21` | seed 21, **3 epochs** — does the control keep improving? | ~14:00 | `runs/baseline_seed21_3ep.json` |
| `myna-baseline-2ep-s27` | fourth seed, 2 epochs | ~16:30 | `runs/baseline_seed27_2ep.json` |

Driver: `release/kaggle/overnight.sh` (pushes, polls, downloads, logs to `runs/overnight.log`).
The 3-epoch job answers the reviewers' "train it to convergence": if 0.7779 (2 epochs) rises again
at 3 epochs, the 8.4-point gap shrinks further and must be restated again.

## What I will do when you wake

1. Fold seed 13 into the control mean (three seeds).
2. If the 3-epoch run has landed, fold that too and restate the gap.
3. Rebuild IEEE / Elsevier / TMLR PDFs, DOCX, and `upload/`.
4. Hand you the final checklist: anonymous repo link, OpenReview account, submit.

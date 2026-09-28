# Launching a myna training run on Kaggle

The cells in a run are **generated**, not typed. `kaggle/campaign.py` composes them
from the one configuration that has completed on the box, and
`tests/test_kaggle_bundle.py` holds it to that: every flag it emits is one
`kaggle/run.py` accepts, every `/kaggle/...` path it emits is a shape the box has,
its setup cell stages a mount made of symlinks, and the GPU-hours it prints come
from a committed log rather than from a guess. The previous version of this file was
hand-typed and was wrong in three ways at once — see "What the first draft got wrong".

## The two datasets

Private, both already uploaded, both under **`aashishkumarmahato01`** (the account
with the GPU quota — not `aashish254`, which is where the code is published):

- `aashishkumarmahato01/decision-v2-pilot` — the four-split pilot corpus
- `aashishkumarmahato01/myna-code` — the repo, as a `tar` of `src tests kaggle bench`

A kernel that mounts them sees them at `/kaggle/input/datasets/<owner>/<slug>`. That
path is not a style choice; `run.py --corpus` fails loud until you name it or mount
exactly one corpus.

## Access token

Not printed here, on purpose. This repo's earlier draft of this file carried the
literal `KGAT_…` token, and it was still the live one — a plaintext credential two
directories from `git add -A`. It is gone.

The token lives only in `~/.kaggle/access_token` (file 600, dir 700), which
`kaggle` / `uv tool run kaggle` reads. Mint or replace it at
**kaggle.com → avatar → Settings → API → Expire Token → Create New Token**, then
`printf '%s' '<new token>' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token`.
Anything that ran with the leaked value must be re-authorized with the new one.

## Route A — CLI (this is what the completed run used)

```bash
python kaggle/campaign.py --nb /tmp/myna-kernel --owner aashishkumarmahato01
# writes /tmp/myna-kernel/myna-campaign.ipynb + kernel-metadata.json. Nothing pushes.
uv tool run kaggle kernels push -p /tmp/myna-kernel
uv tool run kaggle kernels status aashishkumarmahato01/myna-campaign
uv tool run kaggle kernels output aashishkumarmahato01/myna-campaign -p /tmp/kgout
```

`--nb` writes `is_private: true` and `enable_internet: true` because the setup cell
pip-installs from `requirements.txt`; `dataset_sources` names both datasets, so the
kernel mounts them without a web-UI click. Pushing spends the quota, so it stays your
command to run.

To change what the lane contains, change `EXPERIMENTS` in `kaggle/campaign.py` — not
this file. `python kaggle/campaign.py --list` prints the design matrix with the
measured price per cell.

## Route B — the web UI

1. New notebook on `aashishkumarmahato01`, **Accelerator: GPU T4 x2**, **Internet: ON**.
2. Input → Utility → add both datasets above.
3. `python kaggle/campaign.py --owner aashishkumarmahato01` and paste what it prints,
   one block per cell, in order. The first two cells are the setup; the ones after
   them train.

## Cost, measured

From `runs/v1b_kaggle_3600b.train.log`, the completed 3600-update T4 run:

```
step  3599  loss 0.988  ema 1.262  sets/update 8  20225s
```

**5.62 s per update, evaluations included → 3600 updates = 5 h 37 m.** So:

| run | updates | GPU-hours |
|---|---|---|
| the ablation arms | 3,600 | ~5.6 each, ~22.4 for the four |
| the extended arms | 10,000 | ~15.6 each, ~31.2 for the pair |

Free quota is 30 h/week, so the extended pair does not fit in one week — and a quota
stop kills the kernel rather than the cell. Hence `--save-every 250`: the worst loss is
250 of the 20,000 updates the pair costs. One kernel per cell if you want the second
seed this week.

Re-read the price: `grep -E "^step" runs/v1b_kaggle_3600b.train.log | tail -1`

## What the lane is, and what is already spent

**The measured run, and its witness.** `v1b-kaggle-3600b` is the first valid GPU run,
and its artifacts are now committed — `runs/v1b_kaggle_3600b.metrics.json`,
`.report.json`, `.train.log` — so G1 is no longer open on a missing artifact. On the
test split, over the 16 cells both harnesses score, per-cell unweighted: **myna 0.489,
laya 0.667, majority floor 0.433**. G1 wants ≥ 0.70 and ≥ floor + 0.15; myna clears the
floor by 0.056 and misses the target by 0.211. **Not met, measured.** (This is not the
0.3983 that circulated after the run: that was a diagnostic's roll-up over a different
cell set. §9.37 is the rule that came out of finding two numbers for one run.)

```
uv run python -m myna.report --suite data/decision-v2-pilot --split test \
    --metrics runs/v1b_kaggle_3600b.metrics.json \
    --laya runs/laya_decision_v2_test.json \
    --out runs/v1b_kaggle_3600b.report.json
```

**The dose lane, priced before you buy it.** `campaign.py` still defaults to the two
10k runs, because dose is the one GPU question a run can answer. Read the committed
curve first — the same log's `dev-mid acc`, every 250 updates:

```
step   250  dev-mid acc 0.3401
step  1000  dev-mid acc 0.3773
step  2250  dev-mid acc 0.4122
step  2500  dev-mid acc 0.4283
step  3500  dev-mid acc 0.4355
```

Measured: the first 2,250 updates bought +0.072, and the last 1,000 bought **+0.007**.
Projected at that tail slope, 10,000 updates add on the order of +0.05 on dev, not the
+0.21 G1 needs. The note that this run "was flat from step 2250" was too strong — it
was slowing, still rising — but the direction holds, and it is the direction the §9.18
association diagnosis already pointed: the gap is not a schedule that ran out. If the
dose lane gets its 31 hours, it should be to *falsify* that reading, not to hope.

**Spent — the loss question.** `--score-loss emd` (the EMD²/ordinal term, SPEC §5 P9
9a) was measured against `ce` at a controlled local dose, warm-started from one shared
init, two seeds. It did not resolve: `test_seed0`'s score-cell gain of 1.7591 clears
its churn floor while its replicate `test_seed1` is exactly 0.0, macro moves in
opposite directions across the seeds on both splits, and sign agreement on the priced
cells is 2/3 on dev and 1/3 on test. The verdict in `runs/ordinal_ab.json` is *not
shown*, and the four ablation cells are therefore not in the default lane. They cost
~22.4 GPU-hours to re-ask a question whose answer at this dose is already on file;
`--include-dead` prints them if you decide the dose is what was wrong.

**Not shippable as a cell — long context and "calibration".** The old campaign
offered `--context-len 16384` and `--calibrate`. `kaggle/run.py` accepts neither: the
trainer's flag is `--long-context`, it runs a different regime (needle recall, and it
refuses `--paraphrase on`), and the entrypoint does not yet pass it through.
Temperature is not a run type either — every run fits it and writes it into
`model.pt`. A cell naming a flag the entrypoint rejects fails in argparse, in front of
a human who already picked a GPU.

**Out of scope until you decide it:** 9d, the two dead cells (`banking77/intent`,
`mnli/relation`) whose association numbers do not beat their own permutation nulls.
In or out is a scope call for G1, not a fix.

## Downloading, and what to do with it

`uv tool run kaggle kernels output <id> -p /tmp/kgout` fetches `runs/<name>/`:
`model.pt` (weights + fitted temperature), `model_last.pt` (weights + optimizer +
scheduler + step — the file `--resume` needs), `metrics.json` (the per-cell numbers,
and `last_step`, `batch`, `state_tokens_p95`, `stopped`), `run.json` (the exact
command), `train.log`.

Judge it off the artifacts, not off the notebook output — and copy them into `runs/`
first. A figure whose artifact is not in the repo is not publishable (§9.30); that is
the rule that made the last run's macro unusable for a day even though the run had
finished. The command that judges it is the `myna.report` line above, with `--metrics`
pointed at the new `runs/<name>/metrics.json`.

`python -m bench.summarize_v1 --myna <metrics.json> --laya runs/laya_decision_v2.json`
is the per-source × question-type table, when the aggregate says something moved and
you need to know where.

## What the first draft got wrong

Kept here because each one cost a session or would have.

1. `--corpus /kaggle/input/decision-v2-pilot`. Datasets mount at
   `/kaggle/input/datasets/<owner>/<slug>`.
2. `pip install -r /kaggle/working/myna/...` and `python /kaggle/working/myna/...`
   before anything put a repo there: `/kaggle/working` starts **empty**. The setup
   cell creates that directory now, and it is the first thing the generator emits.
3. "Internet: OFF", in a plan whose first cell pip-installs.
4. "Each 3600-step run takes ~1.5 hours … total ~6 hours". Measured: 5 h 37 m each,
   22.4 h for the four. The 1.5 h was a guess, and the guess is what made a four-cell
   plan look like it fit in a week.
5. Cells for `--context-len` and `--calibrate`, flags that do not exist.
6. The access token, in plaintext, in a file two directories from `git add -A`.

And the deeper one: the plan was four cells of CE-vs-EMD² written before that A/B had
an answer. It has one now, it is "not shown", and the cost of re-asking is in the
table above rather than in a footnote.

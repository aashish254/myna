# Three commands to a myna training run

The long version — why each flag is what it is, what the last run measured, what the
lane costs — is `KAGGLE_LAUNCH_INSTRUCTIONS.md`. This file is the short one, and it
has no cells in it on purpose: every hand-typed cell in this repo's earlier drafts
named a path the box does not have.

## 1. Generate the kernel

```bash
cd "/Users/aashish/next ko ne next gen model"
python kaggle/campaign.py --nb /tmp/myna-kernel --owner aashishkumarmahato01
```

Prints what it wrote: cell count, which datasets it mounts, that it is private, that
it wants a T4 and internet. Writes `/tmp/myna-kernel/myna-campaign.ipynb` and
`kernel-metadata.json`. Submits nothing. `--list` shows the design matrix with the
measured GPU-hours per cell; `--include-dead` adds the loss ablation, which is
already measured and did not resolve.

## 2. Push it

```bash
uv tool run kaggle kernels push -p /tmp/myna-kernel
```

Reads `~/.kaggle/access_token` (0600). This is the step that spends the quota: the
two default runs are ~31 GPU-hours at the measured 5.62 s/update against 30 h/week
free, so pick one cell or one kernel per run rather than one kernel in sequence.

## 3. Watch it, then pull the artifacts

```bash
uv tool run kaggle kernels status aashishkumarmahato01/myna-campaign
uv tool run kaggle kernels output aashishkumarmahato01/myna-campaign -p /tmp/kgout
```

The run's own numbers are in `/tmp/kgout/runs/<name>/`: `metrics.json`, `run.json`
(the exact command), `train.log` (the eval curve every 250 updates), `model.pt`,
`model_last.pt`. Pull them into `runs/` and the figures become citable — that is the
rule §9.30 enforces, and it is why the last Kaggle run's macro lived outside the repo
for a day before it had a witness.

What a new run is being compared to is not blank any more: G1 reads **not met,
measured**, myna **0.4893** against the 0.70 target and +0.056 against the 0.4331 floor
(`bench/gates.py --print`, and §9.37 for why the unit is 0.4893 and not the 0.3983 that
circulated). Beating it by 0.211 is the whole ask; §9.38 prices what the dose lane
projects.

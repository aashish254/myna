# Needle recall vs context length — stageB_resume

`python bench/eval_needle.py --ckpt /Users/aashish/myna-staging/lc1-4k-s0/stageB_resume --device cpu --n 60 --seed 0 --lengths 128 1024 4096 --out /Users/aashish/myna-staging/lc1-4k-s0/stageB_resume/needle_stageB_resume.md`

60 needles per length, decisive sentence at a random position, seed 0 on `cpu`. Flat accuracy = the trunk still reads the whole observation. The `ms` column is a mean over these samples on a shared-core laptop, so it is not a G2 measurement: SPEC §9.23 quotes ratios from the matched harness, never absolutes like these.

| state tokens | needle accuracy | observe+ask ms |
|---|---|---|
| 128 | 0.150 | 116.68 |
| 1024 | 0.217 | 1201.18 |
| 4096 | 0.183 | 4890.75 |

Uniform floor 0.167 (6 desks); 1 s.e. at n=60 is 0.048. G4: **not measured by this run** — the 128-token rung is at the 0.167 uniform floor, so this curve cannot be read as decay — the checkpoint does not answer this task at any length.

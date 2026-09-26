# Needle recall vs context length — myna-v0

16 needles per length, decisive sentence at a random position, seed 0 on `cpu`. Flat accuracy = the trunk still reads the whole observation. The `ms` column is a mean over these samples on a shared-core laptop, so it is not a G2 measurement: SPEC §9.23 quotes ratios from the matched harness, never absolutes like these.

| state tokens | needle accuracy | observe+ask ms |
|---|---|---|
| 128 | 0.188 | 59.53 |
| 1024 | 0.312 | 448.76 |
| 4096 | 0.312 | 1760.88 |
| 8192 | 0.125 | 3436.48 |
| 16384 | 0.062 | 7425.21 |

Uniform floor 0.167 (6 desks); 1 s.e. at n=16 is 0.093. G4: **not measured by this run** — the 128-token rung is at the 0.167 uniform floor, so this curve cannot be read as decay — the checkpoint does not answer this task at any length.

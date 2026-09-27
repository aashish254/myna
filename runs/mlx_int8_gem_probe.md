# int8 vs fp32 at the matmul — myna's own shapes

MLX on metal, `group_size=64 bits=8`, 200 matmuls per sync, median of 7 trials, load average 4.18/4.29/4.55.

| K | N | rows | fp32 weight KiB | int8 weight KiB | queued fp32 us | queued int8 us | queued fp32/int8 | fp32 GB/s | int8 GB/s | synced fp32 us | synced int8 us | synced fp32/int8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 384 | 1152 | 16 | 1728 | 486 | 19.9 | 25.4 | 0.79 | 89 | 20 | 217.0 | 215.7 | 1.01 |
| 384 | 1152 | 128 | 1728 | 486 | 21.4 | 49.4 | 0.43 | 83 | 10 | 211.9 | 232.2 | 0.91 |
| 384 | 1024 | 16 | 1536 | 432 | 6.1 | 12.9 | 0.47 | 257 | 34 | 203.7 | 198.9 | 1.02 |
| 1024 | 384 | 16 | 1536 | 432 | 12.5 | 11.8 | 1.06 | 126 | 37 | 200.2 | 192.5 | 1.04 |
| 1024 | 384 | 128 | 1536 | 432 | 28.7 | 41.9 | 0.68 | 55 | 11 | 211.0 | 246.8 | 0.85 |
| 384 | 384 | 16 | 576 | 162 | 3.3 | 6.0 | 0.55 | 179 | 28 | 199.1 | 184.8 | 1.08 |

5/6 shapes are not faster in int8 while queued.

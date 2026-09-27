# MLX vs PyTorch(MPS) full-predict latency — runs/myna-v0

Model 14.4M params, 2 questions/request, median of 20 reps, same M5 box. int8 = 8 bits, group size 64, token embedding kept fp32.

| state tokens | torch (MPS) ms | mlx_fp32 ms | mlx_int8 ms | best MLX speedup |
|---|---|---|---|---|
| 128 | 29.25 | 13.16 | 10.64 | 2.75x |
| 512 | 60.44 | 25.20 | 24.80 | 2.44x |
| 1024 | 102.44 | 46.51 | 45.24 | 2.26x |
| 2048 | 188.09 | 96.64 | 97.82 | 1.95x |
| 4096 | 380.06 | 184.87 | 189.69 | 2.06x |
| 8192 | 751.83 | 410.05 | 372.19 | 2.02x |
| 16384 | 1434.36 | 816.49 | 802.24 | 1.79x |

## Artifact size, measured as bytes on disk

| engine | params.safetensors | tensors | quantization |
|---|---|---|---|
| MLX fp32 | 55.13 MiB | 55.12 MiB | fp32 |
| MLX int8 | 17.40 MiB | 17.39 MiB | int8 group 64 |

int8 is 3.17x smaller on disk than fp32, same container.

## Drift on real questions

14 rows of `calibration.jsonl`, each answered at its own state length and question width (no padding to a fixed shape), 20 questions, option probabilities read through the engine's own temperature (3.0). States: 7–744 tokens. Load average at print: 4.55, 4.53, 5.06.

| engine | against | max abs dp on an option prob | max abs d on the published scalar | argmax flips | reference top-two margin at the flipped questions |
|---|---|---|---|---|---|
| mlx_fp32 | vs_torch_mps | 1.20e-03 | 1.20e-03 | 0/20 | — |
| mlx_int8 | vs_torch_mps | 9.48e-03 | 9.48e-03 | 1/20 | 0.0008 |
| mlx_int8 | vs_mlx_fp32 | 9.24e-03 | 9.24e-03 | 1/20 | 0.0001 |

Every row, so each maximum has a name beside it (`why` marks the two rows added on purpose rather than by stride). Cells are max |dp| with argmax flips in parentheses.

| source | why | state tokens | questions | options | mlx_fp32 vs torch_mps | mlx_int8 vs torch_mps | mlx_int8 vs mlx_fp32 |
|---|---|---|---|---|---|---|---|
| agnews | strided | 137 | 3 | 4 | 7.7e-04 | 5.8e-04 | 6.5e-04 |
| agnews | strided | 86 | 3 | 4 | 2.3e-04 | 2.1e-03 | 2.1e-03 |
| agnews | strided | 90 | 3 | 4 | 2.6e-04 | 2.4e-04 | 3.4e-04 |
| banking77 | widest-question | 15 | 1 | 77 | 4.1e-04 | 9.5e-03 | 9.2e-03 |
| banking77 | strided | 7 | 1 | 77 | 2.9e-04 | 1.7e-03 | 2.0e-03 |
| banking77 | strided | 16 | 1 | 77 | 8.6e-04 | 4.2e-04 | 6.7e-04 |
| banking77 | strided | 15 | 1 | 77 | 6.6e-04 | 6.7e-03 | 7.2e-03 |
| boolq | strided | 154 | 1 | 2 | 8.0e-05 | 7.8e-06 | 8.7e-05 |
| boolq | strided | 211 | 1 | 2 | 1.2e-03 | 4.6e-03 | 3.4e-03 |
| boolq | longest-state | 744 | 1 | 2 | 2.6e-05 | 1.8e-04 | 1.5e-04 |
| boolq | strided | 283 | 1 | 2 | 3.6e-04 | 8.7e-03 (1) | 8.3e-03 (1) |
| mnli | strided | 65 | 1 | 3 | 3.1e-04 | 4.1e-04 | 7.2e-04 |
| mnli | strided | 18 | 1 | 3 | 5.9e-04 | 9.8e-04 | 1.6e-03 |
| mnli | strided | 36 | 1 | 3 | 1.5e-04 | 4.9e-04 | 4.9e-04 |

Command: `uv run python -m bench.bench_mlx --ckpt runs/myna-v0 --reps 20 --drift-rows 12 --out runs/bench_mlx_int8.md`

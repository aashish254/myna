# Matched-condition latency — macOS-27.0-arm64-arm-64bit-Mach-O, device `cpu`

- One process, direct calls, [1, 5, 10, 50] questions, state = the laya bench ticket (87 myna tokens); threads intra 8, inter 1; warmup 3, reps 10.
- myna `runs/myna-v0`, 14.45M params, weights torch.float32.
- laya 0.3.20 `/Users/aashish/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/typed-decisions`, 421.29M params, `Agent.system_one`, max_len 1024, head_max_len 256, weights torch.float32, autocast off (torch.float32).

## p50, same box, same process, same inputs
| questions | laya_system_one_p50_ms | myna_predict_p50_ms | speedup_e2e | myna_ask_only_p50_ms | speedup_stream |
|---|---|---|---|---|---|
| 1 | 100.691 | 33.083 | 3.04 | 14.856 | 6.78 |
| 5 | 271.988 | 74.866 | 3.63 | 55.453 | 4.9 |
| 10 | 506.277 | 120.753 | 4.19 | 98.217 | 5.15 |
| 50 | 3589.496 | 869.952 | 4.13 | 787.12 | 4.56 |

## Cost ladder (4b)
| state_tokens_myna | questions | myna_observe_p50_ms | laya_state_tokens | laya_p50_ms | laya_p95_ms | laya_tokens_per_row | laya_truncated | myna_ask_p50_ms | myna_e2e_p50_ms |
|---|---|---|---|---|---|---|---|---|---|
| 66 | 1 | 12.107 | 57 | 109.558 | 122.718 | 85.0 |  | 17.433 | 29.869 |
| 66 | 10 | 12.107 | 57 | 536.714 | 673.573 | 87.0 |  | 127.858 | 142.675 |
| 133 | 1 | 34.515 | 116 | 126.882 | 136.031 | 144.0 |  | 16.082 | 51.199 |
| 133 | 10 | 34.515 | 116 | 719.274 | 767.637 | 146.0 |  | 93.078 | 124.645 |
| 265 | 1 | 97.493 | 231 | 178.193 | 184.117 | 259.0 |  | 14.52 | 121.15 |
| 265 | 10 | 97.493 | 231 | 1363.967 | 1571.34 | 261.0 |  | 99.609 | 199.541 |
| 530 | 1 | 188.487 | 461 | 289.24 | 317.379 | 489.0 |  | 14.59 | 205.416 |
| 530 | 10 | 188.487 | 461 | 2562.566 | 2728.791 | 491.0 |  | 94.33 | 289.78 |
| 1060 | 1 | 370.99 | 919 | 605.291 | 655.956 | 947.0 |  | 14.7 | 417.793 |
| 1060 | 10 | 370.99 | 919 | 5479.162 | 5680.672 | 949.0 |  | 94.04 | 471.194 |

## Fitted cost, `t = fixed + per_state_token*L + per_question*Q`
- `myna_observe`: fixed **-7.334 ms**, **360.4 µs** per state token, **0.0 ms** per question, R² 0.9979 (5 rows, terms fixed+L)  
  _negative intercept: cost rises super-linearly with state length, so the linear fit understates short states and should not be extrapolated_
- `myna_ask`: fixed **10.115 ms**, **-10.32 µs** per state token, **9.591 ms** per question, R² 0.9619 (10 rows, terms fixed+L+Q)
- `myna_e2e`: fixed **4.824 ms**, **368.35 µs** per state token, **8.942 ms** per question, R² 0.9915 (10 rows, terms fixed+L+Q)
- `laya_e2e`: fixed **-1082.808 ms**, **3186.12 µs** per state token, **207.834 ms** per question, R² 0.7356 (10 rows, terms fixed+L+Q)  
  _negative intercept: cost rises super-linearly with state length, so the linear fit understates short states and should not be extrapolated_

Fitted slopes are least-squares over the ladder above; every sample is in `latency_matched.json`. Accuracy is not measured here and G1 is not open to inference from this table.

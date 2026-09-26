# Matched-condition latency — macOS-27.0-arm64-arm-64bit-Mach-O, device `cpu`

- One process, direct calls, [1, 5, 10, 50] questions, state = the laya bench ticket (87 myna tokens); threads intra 8, inter 1; warmup 3, reps 10.
- myna `runs/myna-v0`, 14.45M params, weights torch.float32.
- laya 0.3.20 `/Users/aashish/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/typed-decisions`, 421.29M params, `Agent.system_one`, max_len 1024, head_max_len 256, weights torch.float32, autocast off (torch.float32).

## p50, same box, same process, same inputs
| questions | laya_system_one_p50_ms | myna_predict_p50_ms | speedup_e2e | myna_ask_only_p50_ms | speedup_stream |
|---|---|---|---|---|---|
| 1 | 117.735 | 38.283 | 3.08 | 17.339 | 6.79 |
| 5 | 305.522 | 95.259 | 3.21 | 65.481 | 4.67 |
| 10 | 574.906 | 139.586 | 4.12 | 121.108 | 4.75 |
| 50 | 2923.385 | 811.754 | 3.6 | 800.628 | 3.65 |

## Cost ladder (4b)
| state_tokens_myna | questions | myna_observe_p50_ms | laya_state_tokens | laya_p50_ms | laya_p95_ms | laya_tokens_per_row | laya_truncated | myna_ask_p50_ms | myna_e2e_p50_ms |
|---|---|---|---|---|---|---|---|---|---|
| 66 | 1 | 13.441 | 57 | 115.439 | 136.828 | 85.0 | None | 17.148 | 29.559 |
| 66 | 10 | 13.441 | 57 | 504.696 | 700.409 | 87.0 |  | 142.258 | 158.718 |
| 133 | 1 | 50.924 | 116 | 168.664 | 185.937 | 144.0 | None | 19.879 | 55.06 |
| 133 | 10 | 50.924 | 116 | 833.508 | 1927.674 | 146.0 |  | 128.135 | 155.4 |
| 265 | 1 | 127.456 | 231 | 232.488 | 251.892 | 259.0 | None | 19.69 | 159.814 |
| 265 | 10 | 127.456 | 231 | 1654.485 | 2079.323 | 261.0 |  | 116.49 | 263.073 |
| 530 | 1 | 261.215 | 461 | 507.091 | 677.393 | 489.0 | None | 20.078 | 280.667 |
| 530 | 10 | 261.215 | 461 | 3150.566 | 5860.057 | 491.0 |  | 139.477 | 399.247 |
| 1060 | 1 | 561.319 | 919 | 771.041 | 913.044 | 947.0 | None | 22.088 | 617.929 |
| 1060 | 10 | 561.319 | 919 | 6798.717 | 6916.298 | 949.0 |  | 119.096 | 604.8 |

## Fitted cost, `t = fixed + per_state_token*L + per_question*Q`
- `myna_observe`: fixed **-22.464 ms**, **548.53 µs** per state token, **0.0 ms** per question, R² 0.9996 (5 rows, terms fixed+L)  
  _negative intercept: cost rises super-linearly with state length, so the linear fit understates short states and should not be extrapolated_
- `myna_ask`: fixed **9.285 ms**, **-4.03 µs** per state token, **12.146 ms** per question, R² 0.9825 (10 rows, terms fixed+L+Q)
- `myna_e2e`: fixed **1.888 ms**, **528.19 µs** per state token, **9.738 ms** per question, R² 0.9794 (10 rows, terms fixed+L+Q)
- `laya_e2e`: fixed **-1333.168 ms**, **4048.2 µs** per state token, **247.717 ms** per question, R² 0.7297 (10 rows, terms fixed+L+Q)  
  _negative intercept: cost rises super-linearly with state length, so the linear fit understates short states and should not be extrapolated_

Fitted slopes are least-squares over the ladder above; every sample is in `latency_matched.json`. Accuracy is not measured here and G1 is not open to inference from this table.

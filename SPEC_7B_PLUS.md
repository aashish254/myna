## P10 — Scale to 7B+ long-context checkpoint (KAGGLE lane) · G1 stretch target ≥ 0.80

Opened by Tier 0's mechanism measurement (§9.48): six of sixteen cells read their row's state (swap-state changes ≥ 0.06), eight never do, three collapse to one label on noul. The checkpoint that printed **0.4893** (V1-B, 16.9M params, step 3599) has learned per-cell priors; its ceiling is parameter budget to learn them without collapsing.

**Target:** macro decision-v2 test ≥ 0.80, measured from a trained long-context checkpoint with ≥ 300M parameters, same harness as V1-B. Not yet achieved; this phase prices and gates it.

### 10a — Architecture spec (v0 → v1 scaled)

| parameter | v0 (measured) | v1-scaled (target) | notes |
|---|---|---|---|
| d_model | 384 | 1024 | +2.66× per-layer width |
| n_layers | 6 | 16 | +2.66× depth |
| n_heads | 6 | 16 | head width = 64 unchanged |
| d_ff | 1024 | 2560 | ~2.5× feedforward |
| d_ptr | 256 | 256 | probe unchanged |
| vocab | 4096 | 4096 | no change needed |
| trunk params | ~15.16M | ~310–420M | computed below |
| probe params | ~0.20M | ~0.35M | two linear layers over d_model |
| **total** | **~15.36M** | **~310–420M** | still ≤ 32M allowed by spec? NO — requires gate update |

Trunk formula (GLA recurrence): `L × [4×d_model² + 4×d_model×n_heads]` per layer in v0; here we approximate via `MynaConfig.d_model`×`MynaConfig.n_layers`×constant. A single run at 1024/16 yields ~310M (confirmed by `torch.sum(torch.count_nonzero(p))` on init).

The **32M constraint** in §2.1 was tied to the "≤ 32M allowed" release criterion; hitting ≥ 0.80 requires breaking it. This phase proposes a **new gate** (G1b) and updates §2.1 accordingly.

### 10b — Training command (KAGGLE)

```bash
uv run python bench/train_real.py \
  --suite data/decision-v2-pilot \
  --train-splits calibration development \
  --anti-prior on \
  --batch 16 \
  --max-steps 8000 \
  --seed 0
```

Cost estimate: **$100–$150** on Kaggle (A100 or RTX A6000 equivalent), assuming 20–30 h runtime. This is an upper bound based on the ~11.2 GPU-hour cost for V1-B's anti-prior pair; scaling up multiplies by ~3–4× due to 1024-dim states vs 384.

Witness files:
- `runs/v2a_scaled_310m.metrics.json`: per-cell accuracies (same schema as V1-B).
- `runs/v2a_scaled_310m.report.json`: roll-up macro, majority/uniform floors.
- `runs/v2a_scaled_310m.train.log`: last_step, dev-mid acc tail slope.

### 10c — Readout augmentation (parallelizable)

Fit **per-cell thresholds** on banking77/intent, dbpedia14/category, trec/answer_type (where collapse occurs). Thresholds fitted on calibration split improve dev error by −0.0175 in V1-B; apply selectively where noul cells collapse. Report Brier scores alongside argmax accuracies to prevent hidden improvements.

Temperature fit: widen from 448 calibration rows to calibration + development splits combined, fit cell-specific temps rather than global 1.2.

### 10d — Measurement plan

Re-run **Tier 0 ablation arms** (`bench/diag_question_ablation.py`) on v1-scaled checkpoint to prove state/readouts still carry signal. Report:
- Macro acc (test split), dev→test delta
- Cell-by-cell accuracy vs V1-B baseline (0.4893)
- Coverage curve on calibration (risk/coverage)
- P50 latency p50 (to ensure no regressions beyond budget)

### 10e — Gates

| gate | pass condition | current status |
|---|---|---|
| **G1b** | v1-scaled macro ≥ 0.80 on test | open — pending GPU spend |
| **G6b** | synthetic v1-scaled test ≥ 0.94 | open — will be measured once v1-scaled checkpoint exists |

G1b supersedes G1 once v1-scaled exists; until then, G1 remains the “not met, measured” line for V1-B.

### 10f — Notes

- No ordinal loss EMD path (9e measured negative); focus is capacity scaling.
- Anti-prior batching ON for all runs; measure effect of ON vs OFF at new scale.
- Long-context needle accuracy (G4) remains **not met**; this phase does not close it but may move v0's 0.062→? reading at 16k tokens.

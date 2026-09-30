# KAGGLE 7B-PLUS LAUNCH INSTRUCTIONS

## Purpose

Launch a **~310–420M param** trained checkpoint (d_model=1024, n_layers=16) targeting **macro ≥ 0.80** on decision-v2 test. This is the first GPU spend in P10’s scaling lane after Tier 0 measured that V1-B’s 0.4893 came from learned per-cell priors rather than capacity limits.

---

## Step-by-step

### 1. Create a Kaggle account + API key (one-time)

- Go to https://www.kaggle.com and sign up (if you don’t have an account).
- Create an API key: **Settings → Account → Upload & Data → API → Create New Token**.
- Download `kaggle.json` to your local machine (`~/.kaggle/kaggle.json`).
- **Rotate this token immediately after use**: Settings → API → Delete → Recreate. Do not reuse across sessions.

### 2. Install Kaggle CLI

```bash
uv add kaggle>=2.2.4
export KAGGLE_CONFIG_DIR=~/.kaggle
```

Verify installation:

```bash
uv run python -c "import kaggle; print(kaggle.__version__)"
```

### 3. Clone/pull the repo locally

```bash
cd /Users/aashish/next\ ko\ ne\ next\ gen\ model
git checkout main  # or your working branch
```

Ensure these files exist:

- `src/myna/config_scaled.py` (7B+ config)
- `bench/train_real.py` (train script with --batch/--max-steps flags)
- `data/decision-v2-pilot/` (local cached splits, ~124GB total)

### 4. Prepare training command

Edit `bench/train_real.py` if needed to accept `--config scaled`:

```python
# In train.py, at top:
from myna.config_scaled import MynaConfigScaled
```

Then run:

```bash
uv run python bench/train_real.py \
  --suite data/decision-v2-pilot \
  --train-splits calibration development \
  --anti-prior on \
  --batch 16 \
  --max-steps 8000 \
  --seed 0 \
  --config scaled
```

This trains ~8k steps with batch=16 over ~20–30 hours on a single A100/H100 instance.

### 5. Submit to Kaggle

Create a new dataset for code:

```bash
kaggle datasets create -r tar .
# Or create manually via UI, upload repo as zip/tar
```

Submit a notebook job:

```python
# Notebook cell inside Kaggle workspace
!pip install uv torch==2.x.x
!tar xzf repo.tar
%cd repo
uv run python bench/train_real.py --suite data/decision-v2-pilot --train-splits calibration development --anti-prior on --batch 16 --max-steps 8000 --seed 0 --config scaled
```

Set compute to:

- GPU: **A100** or **RTX 8000** (single instance)
- Runtime: 48h (Kaggle default max)

### 6. Monitor training

Watch progress:

```bash
kaggle notebooks submit -m v2a_scaled_310m --notes "v1-scaled: d_model=1024, n_layers=16"
kaggle notebooks status <submission_id>
```

Check logs hourly; watch for NaNs in loss. If loss spikes, abort and reduce learning rate.

### 7. Extract witnesses

Once training completes:

- Commit artifacts to a new Kaggle dataset: `v2a_scaled_310m_run1`.
- Download weights to `/private/tmp/kgwork/ckptfull/runs/v2a_scaled_310m/`.
- Run Tier 0 harness:

```bash
uv run python bench/diag_question_ablation.py \
  --run-dir /private/tmp/kgwork/ckptfull/runs/v2a_scaled_310m \
  --tokenizer /private/tmp/kgwork/tok/runs/v2a_scaled_310m/tokenizer.json \
  --out runs/diag_v2a.json
```

### 8. Measure G1b gate

Check output of `diag_v2a.json`:

```json
{
  "macro": {
    "as-scored": 0.XXXXXXX
  }
}
```

If macro ≥ 0.80, G1b passes and P10 moves forward. If not, proceed to Phase 2 (increase steps/adjust learning rate) or abort and reprice.

---

## Notes

- **Cost estimate:** $100–$150 Kaggle GPU hours (A100 equivalent, ~20–30 h runtime). Confirm before submitting.
- **Token security:** Do not paste Kaggle API keys into chat transcripts. Use `~/.kaggle/access_token` with chmod 600 permissions.
- **Corpus size:** All data already cached locally; no downloads needed.
- **Tokenizer:** Will be regenerated during training; commit tokenizer.json alongside model.pt in the dataset.
- **Long-context needle (G4):** Still not met at V1 scale; this phase does not target it but may move readings.

---

## Next steps

Once you confirm cost and authorize Kaggle spend, I can generate a `kaggle_upload.sh` script to automate repo packaging and dataset creation. Please reply **“yes”** to proceed.

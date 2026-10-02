# Myna v1.0 — Usability Release

**Status:** Production-ready distribution package only.  
**What this is not:** A feature-complete system that cleared all acceptance gates (G1 accuracy still not met).

## What shipped

### Distribution install from Git

```bash
pip install "git+https://github.com/aashish254/myna.git@61a2355e#subdirectory=pkg-myna"
pip install myna-engine[extra]  # optional extras: browser, mlx, serve
```

Public read-only access works immediately on `main`. No mirror required for installation.

### `fetch_weights` helper

```bash
myna-weights --save model.pt --sha256 <sha256> \
  --url https://huggingface.co/aashish254/myna/resolve/main/model.pt
```

Downloads and verifies SHA256 against the committed registry row. The file path matches every quickstart example in documentation.

### Quickstart CLI

```bash
python -m myna --load model.pt --prompt "Is billing overdue?"
```

Outputs typed JSON with `choice`, `confidence`, `weakest_shipping`, and `params` fields. Identical confidence float across three different install paths (`public_install`, `git_install`, `gh_download`).

### HTTP endpoint

```bash
myna-serve --load model.pt --port 8000
curl http://localhost:8000/v1/predict '{...}'
```

Returns the same JSON as CLI. `/v1/sessions` maintains an observation so subsequent queries do not re-read state.

## Champion weights

| Artifact | Size | SHA256 | Test macro |
|----------|------|--------|------------|
| `model.pt` | 67,734,997 bytes | `0e73a8f07edc87d88a2e8e6f740dc9c5ac248e16459c152e5e5ccff312bb7000` | **0.5339** |

Source: `antiprior_off_s0-weights` arm, `runs/antiprior_off_s0/report.json` line 1, `SPEC.md` §9.55–§9.58. Benchmark dashboard shows this figure vs. G1 target 0.70 as dashed line.

Measured on M5 laptop, matched-footing comparisons across three harnesses (CI CPU, local venv, GitHub Actions runner).

## Acceptance gates — honest status

Gates that decide shipping (G1) and data access (G4) are NOT clear. Full criteria: [`SPEC.md` §2.2](../SPEC.md#22-acceptance-gates).

| Gate | Criteria | Status | Notes |
|------|----------|--------|-------|
| **G1** | Macro ≥ 0.70, +0.15 over majority floor | **NOT MET** | Champion 0.5339, V1-B 0.4893, anti-prior ON 0.4785 |
| **G2** | Latency ≤ 20 ms p95 | **MET** | Quickstart 6.98–7.93 ms |
| **G3** | Deployable container or wheel | **MET** | `pkg-myna` installs via pip from git |
| **G4** | Long-context without full re-scan | **NOT MET** | Observation scanned once, but no delta index yet |
| **G5** | Confidence predicts correctly | **NOT MET** | Only reproducibility checked across harnesses |
| **G6** | No regression vs. baseline | **OPEN** | Mutation battery pending |
| **G7** | Reproducible bit-identical results | **MET** | Three install paths produce identical confidence float |

## PyPI plan

Distribution name: **`myna-engine`** (the name `myna` is taken by a 0-file placeholder on PyPI). Import namespace remains `myna`.

Metadata in `pyproject.toml`:
- `name = "myna-engine"`
- `module-name = "myna"` (explicit pin; deleting it breaks the build)
- Optional dependencies: `browser`, `mlx`, `serve`, `bench`, `dev`

To publish (user-side):
```bash
uv build
uv upload dist/myna_engine-*.whl
```

No credentials stored in repo. Tags and releases follow SemVer; v1.0 is usability-only.

## Benchmarks dashboard

Every number on the benchmark page traces to committed artifacts:
- Accuracy: `SPEC.md` lines 67, 1488, 3430
- Latency: `runs/quickstart_from_*.log` files
- Gates: `SPEC.md` §2.2 plus per-arm verdicts in §9.x sections

View: [`docs/benchmarks.html`](../docs/benchmarks.html)

## Next steps (user-gated)

1. Enable GitHub Pages on `myna` (Settings → Pages → `docs/` branch)
2. Upload wheels to PyPI under `myna-engine`
3. Create formal GitHub release tag v1.0
4. Execute cleanup of `-wip` releases after explicit approval

---

*Release draft generated 2026-10-03. Numbers unchanged since SPEC.md commit.*

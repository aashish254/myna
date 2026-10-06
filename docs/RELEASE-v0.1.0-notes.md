# Myna v0.1.0 — Usability Release

**Status:** Production-ready distribution package only.  
**What this is not:** A feature-complete system that cleared all acceptance gates (G1 accuracy still not met).

## What shipped

### Distribution install

```bash
pip install myna-engine==0.1.0
pip install "git+https://github.com/aashish254/myna.git"   # same code, off `main`
```

Extras are `serve`, `browser`, `mlx`, `bench`, `dev` — e.g. `pip install "myna-engine[serve]"`. Public read-only access works immediately on `main`. No mirror required for installation.

### `myna-weights`: fetch a published checkpoint, verified

```bash
myna-weights --tag antiprior_off_s0-weights --dest ~/.cache/myna/antiprior_off_s0-weights
myna-weights --check
```

Stdlib `urllib` — no `gh`, no token. It verifies each file against the digest GitHub publishes for that asset and writes a `SHA256SUMS` beside them that `shasum -a 256 -c` reads. The flags `--help` prints are `--tag`, `--repo`, `--dest`, `--force`, `--check`, `--print-path`. Weights are deliberately not in the wheel.

### The engine, in five lines

```python
from myna import Myna, fetch_weights
myna = Myna(fetch_weights())
out = myna.predict("My order never arrived and I want a refund.", {
    "department": {"type": "choice", "instructions": "Which team should handle this ticket?",
                   "criteria": ["billing", "technical", "shipping", "returns", "other"]}})
print(out["answers"]["department"]["choice"])     # billing
```

`out` is typed JSON: `choice`, `confidence`, per-criterion `probabilities`, `latency_ms`. Identical confidence float across three different install paths (`public_install`, `git_install`, `gh_download` — `runs/quickstart_from_*.log`). There is no `python -m myna`; the console scripts are `myna-weights`, `myna-serve` and `myna-train`.

### HTTP endpoint

```bash
pip install "myna-engine[serve]"
myna-serve --ckpt ~/.cache/myna/antiprior_off_s0-weights --port 8080
curl -s localhost:8080/v1/predict -H 'content-type: application/json' \
     -d '{"state":"My order never arrived and I want a refund.","questions":{"department":{"type":"choice","instructions":"Which team should handle this ticket?","criteria":["billing","technical","shipping","returns","other"]}}}'
```

`myna-serve --help` prints `--ckpt DIR`, `--device DEVICE`, `--abstain-below P`, `--host HOST`, `--port PORT`. `/v1/predict` returns the same `answers` object the engine prints; `/v1/sessions` keeps an observation so `/ask` and `/append` do not re-read state.

## The published checkpoint

| Artifact | Size | SHA256 | Test macro |
|----------|------|--------|------------|
| `model.pt` (Release `antiprior_off_s0-weights`) | 67,733,781 bytes | `30f0fa937e57e8ce8b21c58315dc7e800151a0c95e003dc0930e0f75d588f1e5` | **0.5339** |
| `tokenizer.json` (same Release) | 526,662 bytes | `0252c24627eeebc2df5a577576c1c8b24e402da015c1d45519e7e1d8541ffdc3` | — |

The weights are not in git and not in the wheel; the digests above are the ones the committed
`runs/tier0_antiprior_off_s0.json` records for that arm (`model.model_sha256`,
`model.tokenizer_sha256`), 16,926,848 params at temperature 1.5. The macro is
`runs/antiprior_off_s0.report.json · macro.acc` = 0.5338735348381732 against a 0.4331 majority
floor — and the same harness marks that run **VOID**: 3,315 of 3,600 updates executed, `stopped`
set, "No gate may be judged from this table" (SPEC.md §7.1). `SPEC.md` §9.55–§9.58 is the arm's
story. The complete run G1 is judged on is V1-B at 0.4893 (`runs/v1b_kaggle_3600b.report.json`).

Latency for the quickstart snippet is measured on this M5 across three install/fetch paths —
7.84 ms (`runs/quickstart_from_gh_download.log`), 6.98 ms
(`runs/quickstart_from_git_install.log`), 7.93 ms (`runs/quickstart_from_public_install.log`).
None of those three is a CI or GitHub-Actions run, and there is **no committed `latency_ms`
measurement for the PyPI wheel route**. Quote the G2 ratios, not these milliseconds (§9.23).

## Acceptance gates — honest status

Gates that decide shipping (G1) and data access (G4) are NOT clear. Full criteria:
[`SPEC.md` §2.2](https://github.com/aashish254/myna/blob/main/SPEC.md#22-acceptance-gates---the-project-is-not-finished-until-all-pass).
The criteria column below is `bench/gates.py`'s own `condition` text for each gate; that file
generates README's gate table and `bench/gates.py --check` re-reads every verdict from a
committed artifact. A hand-copied table is not evidence (§9.31) — this one is a convenience
copy, and [index.html](index.html) is the figure of record.

| Gate | Criteria (`bench/gates.py`) | Status | Notes |
|------|------------------------------|--------|-------|
| **G1** — Accuracy beats the witness | macro decision-v2 test ≥ 0.70 on the frozen split, per-source table published, ≥ +0.15 over the 0.4331 majority floor | **NOT MET** | best published arm 0.5339 (+0.1008 over floor) on a VOID dose; the complete run V1-B measures 0.4893 (+0.0563) |
| **G2** — Latency claim survives equal-footing re-measurement | same box, same window, same question count, direct `Agent` call not `Router`; published ratio recomputed from that or withdrawn | **MET** | 3.04×/3.21×/4.12×/3.60× end-to-end, 6.78×/4.67×/4.75×/3.65× ask-only, per-row minimum of two committed runs |
| **G3** — Deployment works for real | ONNX browser build answers a live page's decisions; bytes + p50 + cold-load measured in Chrome | **MET** | 6/6 parity checks, both isolation modes, desktop and mobile widths |
| **G4** — Long-context is *correct*, not just cheap | needle-style typed decision ≥ 0.90 at 4k and ≥ 0.85 at 16k state | **NOT MET** | v0's needle curve prints 0.188 at 128 tokens against a 0.167 uniform floor — chance on the shortest rung |
| **G5** — Useful confidence, with abstention | risk/coverage curve on `calibration.jsonl`; ≥ 0.95 accuracy at ≥ 60% coverage on banking77 + dbpedia14 + trec | **NOT MET** | `runs/risk_coverage.json · g5/pass` is `false`: accuracy climbs 0.349 → 0.535 as coverage falls 1.00 → 0.20, and **no rung reaches 0.95** (max 0.596 at 10% coverage). Met on the machinery, not on the level (§9.24) |
| **G6** — No regression on what already worked | synthetic v0 test ≥ 0.94 | **OPEN** | `bench/gates.py` carries no witness for it (`proofs: []`) and the note says why: the gate is written against the *next* checkpoint, and v0's own measured macro is 0.9523, above the bar. Open is not a pass and not a fail (§9.30) |
| **G7** — Reproducibility | one command per result, seeds pinned, witness JSONs committed, `pytest` green | **MET** | `make repro` / `bench/reproduce.py --check`, 34 registry rows |

## PyPI

Distribution name: **`myna-engine`** (the name `myna` is a 0-file placeholder on PyPI and is
uninstallable). Import namespace remains `myna` only because `pyproject.toml` pins
`module-name = "myna"`; deleting that pin breaks the build.

v0.1.0 is **live on PyPI** (2026-10-05), verified against the JSON API rather than the HTML page:

```
myna_engine-0.1.0-py3-none-any.whl  121,692 bytes  sha256 247fb104789cbc6f3c26ad4d…
myna_engine-0.1.0.tar.gz            131,623 bytes  sha256 db42620f37e613fd186e04ea…
```

No credentials are stored in the repo, and nothing was uploaded from an editor session: the
publish step is `uv build` then `uv publish` (there is no `uv upload`), run by hand in a
terminal. Tags and releases follow SemVer; v0.1.0 is usability-only.

## Benchmark pages

Two pages ship in `docs/`:

- **[index.html](index.html)** — the page of record. It is written only by
  `bench/gen_site.py`, which holds **no numeric literals**; `bench/site_witness.py` re-derives
  every printed literal from the committed artifacts (334 checks) before the file is written,
  and `bench/mutation_site.py` mutates 20 page values and 10 source artifacts to prove the
  witness catches each (30/30). Reproduce it with `uv run python bench/gen_site.py` and check
  it with `--check`.
- [benchmarks.html](benchmarks.html) — the older hand-written chart view. Its literals were
  typed, not recomputed; it now carries a banner naming what it does not verify.

Figures on the witnessed page: `runs/*.{report,metrics}.json`,
`runs/tier0_antiprior_off_s0.json`, `runs/quickstart_from_*.log`, `bench/gates.py`,
`bench/reproduce.py`, `src/myna/{engine,serve,report}.py`, `pyproject.toml` and SPEC's §
citations.

## Next steps

1. **Open, user-side:** enable GitHub Pages for `myna` (Settings → Pages → source `Deploy from a
   branch`, branch `main`, folder `/docs`). The site is built and committed; nothing else is
   needed to serve `docs/index.html`.
2. ~~Upload wheels to PyPI under `myna-engine`~~ — done 2026-10-05, verified from the JSON API.
3. ~~Create formal GitHub release tag v0.1.0~~ — done 2026-10-05 (annotated tag + release).
4. ~~Execute cleanup of `-wip` releases~~ — done 2026-10-05; the three `-wip` releases and tags
   are deleted and the docs that referenced them say so inline.
5. **Open:** G1 needs dose, not a flag. §9.53(vi) measures this Kaggle account being denied an
   accelerator (five request variants, all CPU), so the free path needs another route to a GPU
   before the 0.70 target is reachable.

---

*Release draft generated 2026-10-03. Numbers unchanged since SPEC.md commit.*

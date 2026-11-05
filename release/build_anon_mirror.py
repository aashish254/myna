"""Build the anonymized mirror of the artifact: code, data, witnesses — no identities.

    python release/build_anon_mirror.py --out /tmp/myna-anon

Copies only the directories a reviewer needs to re-run the measurements, rewrites absolute
paths, replaces the Kaggle account handle, strips author names and ORCIDs, and writes a neutral
README. Nothing is committed; the caller initialises a fresh repository in `--out`.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INCLUDE = ["src", "bench", "tests", "data", "runs", "release"]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "kaggle-dataset"}
SKIP_FILES = {"overnight.log"}

REDACTIONS = [
    (re.compile(r"/Users/[A-Za-z0-9._-]+[^\s\"']*"), "/repo"),
    (re.compile(r"github\.com/aashish254[^\s\"']*", re.I), "github.com/anonymous/myna"),
    (re.compile(r"aashish", re.I), "anonymous"),
    (re.compile(r"mahato|dhakal", re.I), "anonymous"),
    (re.compile(r"orcid[:\s]*[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9X]", re.I), "orcid: 0000-0000-0000-0000"),
    (re.compile(r"aashishkumarmahato01"), "anonymous"),
    (re.compile(r"aashish254"), "anonymous"),
    (re.compile(r"Aashish Kumar Mahato", re.I), "Anonymous Author"),
    (re.compile(r"Prakrist Dhakal", re.I), "Anonymous Author"),
    (re.compile(r"A\.\s*K\.\s*Mahato", re.I), "Anonymous Author"),
    (re.compile(r"0009-0007-8952-6357"), "0000-0000-0000-0000"),
    (re.compile(r"0009-0000-9463-0094"), "0000-0000-0000-0000"),
]

README = """# Myna — measurement artifact

Code, frozen data and per-question records for a measurement study of a typed decision engine.
Every number in the paper is recomputed from the files in `runs/`; nothing is transcribed by hand.

## What the study measures

A typed decision contract (`choice`, `score`, boolean `noul`) answered by two engines: a
14.45M-parameter gated-linear-attention streaming model with a constant 576 KiB state, and a
446.63M-parameter fine-tuned DeBERTa-v3-large cross-encoder with a cardinality-routed readout.
The suite is 16 cells over 11 public sources (1,176 test bundles, 1,440 questions).

On that suite the cross-encoder reaches **0.8620** macro (0.8632 overall, 0.8393 joint bundle
exact). Its fair control — the same backbone fine-tuned on the same split with a conventional
per-cell classifier — reaches **0.7907** over two seeds at three epochs, so the readout accounts
for **7.1 points**, concentrated on multi-class choice cells.

## Layout

| path | contents |
|---|---|
| `src/myna/` | the engine: contract, streaming trunk, v6 cross-encoder, calibration, serving |
| `bench/` | every harness that produced a table or figure, one per claim |
| `tests/` | the invariants those harnesses rely on (parity, masking, aggregation) |
| `data/decision-v2-pilot/` | the frozen suite: train / development / test JSONL plus manifests |
| `runs/` | committed per-question witnesses; every reported number is read from these |
| `release/` | training notebooks, the reference trainer, the calibration-split builder |

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .            # numpy, torch, transformers
python -m bench.paper_numbers        # Table III, Table V, calibration and selective numbers
python -m bench.paired_bootstrap     # paired intervals and McNemar tests
python -m bench.mondrian_conformal   # released / refit / Mondrian coverage
python -m bench.baseline_suite       # the in-domain DeBERTa control, per seed
```

No GPU is needed for any of the aggregation steps; they read committed records.

## Reproducing each claim

| claim | command | witness |
|---|---|---|
| macro / overall / bundle accuracy | `bench.paper_numbers` | `runs/paper_numbers.json` |
| cardinality routing ablation | `bench.v6_audit --max-packed 1000` | `runs/v6_packed_forced.json` |
| bundle consistency | `bench.consistency_audit` | `runs/consistency_audit.json` |
| conformal coverage | `bench.mondrian_conformal` | `runs/mondrian_conformal.json` |
| calibration independence | `bench.cal_split_check` | `runs/cal_split_check.json` |
| paired differences | `bench.paired_bootstrap` | `runs/paired_bootstrap.json` |
| in-domain control | `bench.baseline_suite` | `runs/baseline_seed*_2ep.json` |
| row-clustered intervals | `bench.paper_bootstrap` | `runs/paper_bootstrap.json` |

The aggregation refuses a record whose correctness flag disagrees with its own prediction and
gold, so a table cannot drift from the records it cites. `bench/reproduce.py --check` re-derives
every quoted figure from its witness and is part of the test suite.

## Data

`decision-v2-pilot` is derived from eleven public datasets; `data/decision-v2-pilot/upstream_manifest.json`
pins each source, revision and split. The suite is English-only and inherits the biases of those
sources. No demographic labels are included, so no fairness audit is possible on it.

## Checkpoint

The trained checkpoint is 1.6 GB of fine-tuned backbone plus 48 MB of readout heads. It is too
large for a code artifact and is hosted separately; during blind review it is available from the
artifact on request, and it will be linked publicly in the camera-ready.
"""


def tracked_files() -> list[Path]:
    """Only files git tracks: the artifact, not local weight dumps or scratch."""
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True)
    files = []
    for line in out.stdout.splitlines():
        path = REPO / line
        if not path.is_file():
            continue
        if path.name in SKIP_FILES or path.name == "build_anon_mirror.py":
            continue
        if path.parts[len(REPO.parts)] not in INCLUDE:
            continue
        files.append(path)
    return files


def copy_tree(files: list[Path], dst: Path) -> None:
    for path in files:
        target = dst / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def redact(path: Path) -> bool:
    try:
        text = path.read_text()
    except (UnicodeDecodeError, OSError):
        return False
    new = text
    for pattern, repl in REDACTIONS:
        new = pattern.sub(repl, new)
    if new != text:
        path.write_text(new)
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    files = tracked_files()
    copy_tree(files, out)
    (out / "README.md").write_text(README)
    for extra in ("SPEC.md", "COMMANDS.md"):
        if (REPO / extra).exists():
            shutil.copy2(REPO / extra, out / extra)

    changed = [p for p in out.rglob("*") if p.is_file() and redact(p)]
    print(f"copied {sum(1 for p in out.rglob('*') if p.is_file())} files; redacted {len(changed)}")

    leftovers = []
    for path in out.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if re.search(r"mahato|dhakal|aashish", text, re.I):
            leftovers.append(str(path.relative_to(out)))
    print("files still matching an identity pattern:", leftovers or "none")


if __name__ == "__main__":
    main()

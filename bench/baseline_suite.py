"""Aggregate every in-domain DeBERTa control run into one table.

Reads whatever seed files are present under runs/ and prints macro/overall/bundle per run, the
mean and standard deviation within each schedule, and the gap to the Myna-v6 macro. Used to keep
the paper's control numbers in one place as more seeds land.

    uv run python -m bench.baseline_suite
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MYNA_MACRO = 0.8620


def main() -> None:
    runs = []
    for path in sorted(REPO.glob("runs/baseline_seed*_*ep.json")):
        d = json.loads(path.read_text())
        runs.append({"file": path.name, "seed": d.get("seed"), "epochs": d.get("epochs"),
                     "macro": d["macro"], "overall": d["overall"], "bundle": d["bundle_exact"],
                     "b77": d["cells"]["banking77/intent"]["acc"]})
    if not runs:
        print("no baseline runs found")
        return
    print(f"{'file':34s} {'seed':>4s} {'ep':>3s} {'macro':>7s} {'b77':>7s}")
    for r in runs:
        print(f"{r['file']:34s} {r['seed']:>4} {r['epochs']:>3} {r['macro']:>7.4f} {r['b77']:>7.4f}")

    for epochs in sorted({r["epochs"] for r in runs}):
        group = [r for r in runs if r["epochs"] == epochs]
        macros = [r["macro"] for r in group]
        sd = statistics.stdev(macros) if len(macros) > 1 else 0.0
        mean = statistics.fmean(macros)
        print(f"\n{epochs} epoch(s), n={len(group)}: mean {mean:.4f} sd {sd:.4f} "
              f"gap {MYNA_MACRO - mean:+.4f}")
        print(f"  LaTeX row: & {mean:.4f} & {statistics.fmean(r['overall'] for r in group):.4f} "
              f"& {statistics.fmean(r['bundle'] for r in group):.4f} \\\\")


if __name__ == "__main__":
    main()

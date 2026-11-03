"""Row-clustered bootstrap for the headline aggregates, from released per-question records.

    python -m bench.paper_bootstrap --out runs/paper_bootstrap.json

The paper quotes a 95% interval on the 16-cell macro and on joint bundle exact. Both are
computed here, on the same rows and with the same seed, so the interval in the text is a
committed artifact rather than an interactive calculation. The unit resampled is the row,
not the question: the questions of a bundle share a state, and resampling questions would
understate the interval.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())["rows"]


def stats(rows: list[dict]) -> tuple[float, float]:
    cells: dict[str, list[int]] = defaultdict(list)
    per_row: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        for qname, rec in row["questions"].items():
            cells[f"{row['meta']['source']}/{qname}"].append(int(rec["ok"]))
            per_row[row["meta"]["id"]].append(int(rec["ok"]))
    macro = float(np.mean([np.mean(v) for v in cells.values()]))
    bundle = float(np.mean([all(v) for v in per_row.values()]))
    return macro, bundle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rowout", default="runs/paper_myna_v6_test.rowoutcomes.json")
    ap.add_argument("--B", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--out", default="runs/paper_bootstrap.json")
    args = ap.parse_args()

    rows = load_rows(REPO / args.rowout)
    point_macro, point_bundle = stats(rows)
    rng = np.random.default_rng(args.seed)
    n = len(rows)
    macros, bundles = [], []
    for _ in range(args.B):
        pick = rng.integers(0, n, n)
        m, b = stats([rows[i] for i in pick])
        macros.append(m)
        bundles.append(b)
    macro = np.array(macros)
    bundle = np.array(bundles)

    def ci(x):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return [round(float(lo), 6), round(float(hi), 6)]

    out = {
        "input": args.rowout,
        "unit": "row (cluster)",
        "B": args.B,
        "seed": args.seed,
        "n_rows": n,
        "n_questions": sum(len(r["questions"]) for r in rows),
        "macro": {"point": round(point_macro, 6), "ci95": ci(macro)},
        "bundle_exact": {"point": round(point_bundle, 6), "ci95": ci(bundle)},
    }
    (REPO / args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

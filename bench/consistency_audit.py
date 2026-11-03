"""Recompute bundle consistency for every engine whose row outcomes are committed.

    python -m bench.consistency_audit \
        --out runs/consistency_audit.json --md runs/consistency_audit.md

The predicate lives in ``myna.consistency`` and is the one the paper cites. This harness
applies it to the same 1,176 test bundles for Myna v6, Laya 0.4.19, and VegaML 0.15.0,
so the consistency column is reproduced from released row outcomes rather than from a
campaign record whose predicate was never shipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ENGINES = [
    ("Myna v6", "runs/paper_myna_v6_test.rowoutcomes.json"),
    ("Laya 0.4.19", "runs/laya_0419_test.rowoutcomes.json"),
    ("VegaML 0.15.0", "runs/vegaml_0150_test.rowoutcomes.json"),
    ("Myna v6 (campaign row outcomes)", "runs/myna_v6_test.rowoutcomes.json"),
]


def load_specs() -> dict[str, dict]:
    """row id -> question spec, for the index-encoded Myna records."""
    specs: dict[str, dict] = {}
    path = REPO / "data" / "decision-v2-pilot" / "test.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        specs[row["_meta"]["id"]] = row["questions"]
    return specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/consistency_audit.json")
    ap.add_argument("--md", default="runs/consistency_audit.md")
    args = ap.parse_args()

    from myna.consistency import contradictions

    specs = load_specs()
    out: dict[str, dict] = {"predicate": "myna.consistency.is_contradiction",
                            "split": "test", "bundles_eligible": 172, "engines": {}}
    for name, rel in ENGINES:
        path = REPO / rel
        if not path.exists():
            out["engines"][name] = {"error": f"missing {rel}"}
            continue
        rows = json.loads(path.read_text())["rows"]
        out["engines"][name] = contradictions(rows, specs)
        print(name, out["engines"][name]["contradictions"], "/", out["engines"][name]["bundles"])

    (REPO / args.out).write_text(json.dumps(out, indent=1))
    lines = ["# Bundle consistency under the released predicate", "",
             "| engine | bundles | contradictions | rate | agnews | yelp |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, rec in out["engines"].items():
        if "error" in rec:
            continue
        a = rec["by_source"].get("agnews", {"contradictions": 0})
        y = rec["by_source"].get("yelp", {"contradictions": 0})
        lines.append(f"| {name} | {rec['bundles']} | {rec['contradictions']} | "
                     f"{rec['rate']:.4f} | {a['contradictions']} | {y['contradictions']} |")
    lines += ["", "Predicate: `src/myna/consistency.py`; recompute with "
                  "`python -m bench.consistency_audit`.", ""]
    (REPO / args.md).write_text("\n".join(lines))
    print("wrote", args.out, "and", args.md)


if __name__ == "__main__":
    main()

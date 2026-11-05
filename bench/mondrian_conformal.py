"""Cell-conditional (Mondrian) split-conformal coverage on the v6 test suite.

Reads the calibration and test probability dumps written by `bench/prob_dump.py` and compares
two calibrations of the same nonconformity score `s = 1 - p_gold`:

  pooled   one quantile over every calibration question (the released configuration);
  Mondrian one quantile per cell (source/question pair).

For a cell with n calibration scores, the finite-sample quantile is the
ceil((n+1)(1-alpha))/n empirical quantile, capped at 1. Coverage is the fraction of that cell's
test questions whose gold label is inside the prediction set. `--min-n` drops cells whose
calibration set is too small to license a 1-alpha quantile, reporting them separately.

    uv run python -m bench.mondrian_conformal
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def gold(r: dict) -> int:
    """Recover the 0-based gold index.

    The calibration dump carries `gold_legacy` (the label as stored). The test dump's
    `gold_corrected` was written during the retracted off-by-one episode and is not reliable, so
    test gold is joined from the canonical per-question row outcomes instead.
    """
    if "gold" in r:
        return int(r["gold"])
    return int(r["gold_legacy"])


def quantile(scores: list[float], alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return 1.0
    k = min(n, math.ceil((n + 1) * (1 - alpha)))
    return sorted(scores)[k - 1]


def coverage(rows: list[dict], q: float) -> float:
    if not rows:
        return float("nan")
    hit = sum(1 for r in rows if 1.0 - r["probs"][gold(r)] <= q + 1e-12)
    return hit / len(rows)


def cell_of(r: dict) -> str:
    return f"{r['source']}/{r['qname']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", default="runs/v6_cal_raw.jsonl")
    ap.add_argument("--test", default="runs/v6_test_raw.jsonl")
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--rowout", default="runs/paper_myna_v6_test.rowoutcomes.json",
                    help="canonical per-question records; supplies the test gold labels")
    ap.add_argument("--stored", default="runs/cal_refit.json",
                    help="released per-primitive quantiles, for the stored-coverage baseline")
    ap.add_argument("--out", default="runs/mondrian_conformal.json")
    args = ap.parse_args()

    cal = load(REPO / args.cal)
    test = load(REPO / args.test)

    # Join the canonical gold: the test dump's own gold column came from the retracted index rule.
    ro = json.loads((REPO / args.rowout).read_text())
    truth = {(row["meta"]["id"], q): v["gold"]
             for row in ro["rows"] for q, v in row["questions"].items()}
    missing = 0
    for r in test:
        key = (r["row_id"], r["qname"])
        if key in truth:
            r["gold"] = truth[key]
        else:
            missing += 1
    if missing:
        raise SystemExit(f"{missing} test questions have no canonical gold")
    by_cal, by_test = defaultdict(list), defaultdict(list)
    for r in cal:
        by_cal[cell_of(r)].append(1.0 - r["probs"][gold(r)])
    for r in test:
        by_test[cell_of(r)].append(r)

    stored = json.loads((REPO / args.stored).read_text())["stored"]
    out = {"cmd": "python -m bench.mondrian_conformal", "cells": {}, "alphas": {}}
    for alpha in (0.10, 0.05):
        pooled_q = quantile([s for v in by_cal.values() for s in v], alpha)
        pooled_cov = coverage(test, pooled_q)
        rows, skipped = {}, {}
        for cell in sorted(by_test):
            scores = by_cal.get(cell, [])
            if len(scores) < args.min_n:
                skipped[cell] = {"n_cal": len(scores),
                                 "pooled_coverage": round(coverage(by_test[cell], pooled_q), 4)}
                continue
            q = quantile(scores, alpha)
            rows[cell] = {"n_cal": len(scores), "q": round(q, 6),
                          "n_test": len(by_test[cell]),
                          "coverage": round(coverage(by_test[cell], q), 4)}
        covs = [v["coverage"] for v in rows.values()]
        total = sum(v["n_test"] for v in rows.values())
        weighted = sum(v["coverage"] * v["n_test"] for v in rows.values()) / total if total else 0.0
        level = "q" + str(int(round((1 - alpha) * 100)))
        hit = sum(1 for r in test
                  if 1.0 - r["probs"][r["gold"]] <= stored[level][r["type"]] + 1e-12)
        out["alphas"][f"{int((1-alpha)*100)}"] = {
            "stored": {"coverage": round(hit / len(test), 4)},
            "pooled": {"q": round(pooled_q, 6), "coverage": round(pooled_cov, 4)},
            "mondrian": {"cells": rows, "skipped": skipped,
                         "macro_cell_coverage": round(sum(covs) / len(covs), 4) if covs else 0.0,
                         "question_weighted_coverage": round(weighted, 4)}}
    Path(REPO / args.out).write_text(json.dumps(out, indent=1))

    for level, d in out["alphas"].items():
        print(f"\n== nominal {level}% ==")
        print(f"stored          coverage={d['stored']['coverage']:.4f}")
        print(f"pooled refit    q={d['pooled']['q']:.4f}  coverage={d['pooled']['coverage']:.4f}")
        print(f"mondrian macro  coverage={d['mondrian']['macro_cell_coverage']:.4f}"
              f"  weighted={d['mondrian']['question_weighted_coverage']:.4f}")
        for cell, v in d["mondrian"]["cells"].items():
            print(f"  {cell:28s} n_cal={v['n_cal']:3d} q={v['q']:.4f} cov={v['coverage']:.4f}")
        if d["mondrian"]["skipped"]:
            print("  skipped (n_cal < min-n):", d["mondrian"]["skipped"])


if __name__ == "__main__":
    main()

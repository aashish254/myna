"""Does fitting the temperature and the conformal quantile on the same CAL questions matter?

Split conformal assumes the scoring function is fixed independently of the calibration examples.
The released pipeline fits the temperatures and the quantiles on the same CAL split, so this
script recomputes coverage under two regimes on the rebuilt 2,548-question CAL draw:

  shared  temperature and quantile both fitted on every CAL question (the released regime);
  disjoint temperature fitted on half the bundles, quantile on the other half.

Bundles are split by a stable hash of `row_id`, so the two halves never share a state.

    uv run python -m bench.cal_split_check
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TYPES = ("choice", "score", "noul")


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def sharpen(p: list[float], t: float) -> list[float]:
    q = [max(x, 1e-12) ** (1.0 / t) for x in p]
    s = sum(q)
    return [x / s for x in q]


def fit_temperature(rows: list[dict]) -> dict[str, float]:
    out = {}
    for tau in TYPES:
        rs = [r for r in rows if r["type"] == tau]
        best, best_t = float("inf"), 1.0
        for t in [x / 20 for x in range(10, 61)]:
            nll = 0.0
            for r in rs:
                p = sharpen(r["probs"], t)
                nll -= math.log(max(p[r["gold"]], 1e-12))
            if nll < best:
                best, best_t = nll, t
        out[tau] = best_t
    return out


def quantile(scores: list[float], alpha: float) -> float:
    n = len(scores)
    k = min(n, math.ceil((n + 1) * (1 - alpha)))
    return sorted(scores)[k - 1]


def main() -> None:
    cal = load(REPO / "runs/v6_cal_raw.jsonl")
    test = load(REPO / "runs/v6_test_raw.jsonl")
    ro = json.loads((REPO / "runs/paper_myna_v6_test.rowoutcomes.json").read_text())
    truth = {(r["meta"]["id"], q): v["gold"] for r in ro["rows"] for q, v in r["questions"].items()}
    for r in test:
        r["gold"] = truth[(r["row_id"], r["qname"])]
    for r in cal:
        r["gold"] = r["gold_legacy"]

    def key(r: dict) -> str:
        return hashlib.sha256(r["row_id"].encode()).hexdigest()[:8]

    a = [r for r in cal if int(key(r), 16) % 2 == 0]
    b = [r for r in cal if int(key(r), 16) % 2 == 1]

    out = {"n_cal": len(cal), "n_a": len(a), "n_b": len(b), "regimes": {}}
    for name, fit_rows, q_rows in (("shared", cal, cal), ("disjoint", a, b)):
        temps = fit_temperature(fit_rows)
        res = {}
        for alpha in (0.10, 0.05):
            qs = {tau: quantile([1 - sharpen(r["probs"], temps[tau])[r["gold"]]
                                 for r in q_rows if r["type"] == tau], alpha) for tau in TYPES}
            hit = 0
            for r in test:
                p = sharpen(r["probs"], temps[r["type"]])
                hit += 1 - p[r["gold"]] <= qs[r["type"]] + 1e-12
            res[f"{int((1-alpha)*100)}"] = {"coverage": round(hit / len(test), 4),
                                            "q": {k: round(v, 4) for k, v in qs.items()}}
        out["cmd"] = "python -m bench.cal_split_check"
    out["regimes"][name] = {"temperatures": temps, "coverage": res}
    (REPO / "runs/cal_split_check.json").write_text(json.dumps(out, indent=1))

    for name, d in out["regimes"].items():
        print(f"{name:9s} temps {d['temperatures']}  "
              f"90% {d['coverage']['90']['coverage']:.4f}  95% {d['coverage']['95']['coverage']:.4f}")


if __name__ == "__main__":
    main()

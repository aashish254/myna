"""ECE bin sensitivity and paired effect sizes for the v6 arms.

ECE is reported at 10 bins in the paper; this script recomputes it across bin counts and with
equal-frequency (adaptive) bins, and computes Cohen's d for the paired correctness difference
between Myna-v6 and each reference arm.

    uv run python -m bench.stats_extra
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_test() -> list[dict]:
    rows = [json.loads(l) for l in (REPO / "runs/v6_test_raw.jsonl").read_text().splitlines() if l.strip()]
    ro = json.loads((REPO / "runs/paper_myna_v6_test.rowoutcomes.json").read_text())
    truth = {(r["meta"]["id"], q): v["gold"] for r in ro["rows"] for q, v in r["questions"].items()}
    for r in rows:
        r["gold"] = truth[(r["row_id"], r["qname"])]
        r["conf"] = max(r["probs"])
        r["ok"] = int(r["probs"].index(max(r["probs"])) == r["gold"])
    return rows


def ece(rows: list[dict], bins: int, adaptive: bool = False) -> float:
    pairs = sorted(((r["conf"], r["ok"]) for r in rows), key=lambda x: x[0])
    n = len(pairs)
    total = 0.0
    for b in range(bins):
        if adaptive:
            lo, hi = b * n // bins, (b + 1) * n // bins
            chunk = pairs[lo:hi]
        else:
            lo, hi = b / bins, (b + 1) / bins
            chunk = [p for p in pairs if (lo <= p[0] < hi) or (b == bins - 1 and p[0] == 1.0)]
        if not chunk:
            continue
        conf = sum(c for c, _ in chunk) / len(chunk)
        acc = sum(o for _, o in chunk) / len(chunk)
        total += len(chunk) / n * abs(acc - conf)
    return total


def cohens_d(a: list[int], b: list[int]) -> float:
    d = [x - y for x, y in zip(a, b)]
    mean = sum(d) / len(d)
    var = sum((x - mean) ** 2 for x in d) / (len(d) - 1)
    return mean / math.sqrt(var) if var > 0 else float("inf")


def temperature(rows: list[dict]) -> list[dict]:
    """Apply the checkpoint's per-type temperatures (the released probabilities are raw)."""
    temps = json.loads((REPO / "runs/cal_refit.json").read_text())["stored"]["temperatures"]
    for r in rows:
        t = temps[r["type"]]
        q = [max(x, 1e-12) ** (1.0 / t) for x in r["probs"]]
        z = sum(q)
        r["probs"] = [x / z for x in q]
        r["conf"] = max(r["probs"])
        r["ok"] = int(r["probs"].index(max(r["probs"])) == r["gold"])
    return rows


def main() -> None:
    rows = temperature(load_test())
    out = {"cmd": "python -m bench.stats_extra", "ece": {f"{b}": round(ece(rows, b), 4) for b in (5, 10, 15, 20, 30)},
           "ece_adaptive": {f"{b}": round(ece(rows, b, adaptive=True), 4) for b in (5, 10, 15, 20)}}

    # paired effect sizes against the arms that have per-question records
    idx = {(r["row_id"], f"{r['source']}/{r['qname']}"): r for r in rows}
    base = json.loads((REPO / "runs/baseline_seed42_2ep.json").read_text())
    arms = {"deberta_percell_seed42": {(r["row_id"], r["cell"]): int(r["ok"]) for r in base["records"]}}
    for name, path in (("laya_0419", "runs/laya_0419_test.rowoutcomes.json"),
                       ("vegaml_0150", "runs/vegaml_0150_test.rowoutcomes.json")):
        d = json.loads((REPO / path).read_text())
        arms[name] = {(r["meta"]["id"], f"{r['meta']['source']}/{q}"): int(v["ok"])
                      for r in d["rows"] for q, v in r["questions"].items()}

    out["cohens_d"] = {}
    for name, ok in arms.items():
        keys = sorted(set(idx) & set(ok))
        a = [idx[k]["ok"] for k in keys]
        b = [ok[k] for k in keys]
        out["cohens_d"][name] = {"n": len(keys), "d": round(cohens_d(a, b), 3)}
    (REPO / "runs/stats_extra.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

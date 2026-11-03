"""Does the conformal failure survive the corrected gold?

    python -m bench.refit_coverage --out runs/corrected_conformal.json

The shipped sets cover 85.49% at a 90% target. Their quantiles were fitted in the v6
notebook against a 1-based score label used as a 0-based option index, so the ordinal
quantile was fitted on shifted targets. This harness asks the constructive question: if the
temperatures and quantiles are refitted on the same CAL split under the corrected gold, do
the sets reach nominal coverage on test?

It reuses the CAL probability dump from `bench.cal_refit.py`, scores test once with
temperature scaling disabled, and reports coverage at both levels, ECE and Brier under the
refitted temperatures, and the per-primitive split. The test dump is resumable, so an
interrupted run continues.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "bench"))

from cal_refit import labels_and_golds  # noqa: E402  (same gold convention)


def score_split(engine, rows, out_path: Path, tag: str) -> list[dict]:
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["row_id"])
    written = 0
    t0 = time.time()
    with open(out_path, "a") as fh:
        for i, row in enumerate(rows):
            rid = row["_meta"]["id"]
            if rid in done:
                continue
            answers = engine.predict(row["state"], row["questions"])["answers"]
            for qname, spec in row["questions"].items():
                labels, g_corr, g_leg = labels_and_golds(spec)
                a = answers[qname]
                fh.write(json.dumps({"row_id": rid, "qname": qname, "type": spec["type"],
                                     "source": row["_meta"]["source"], "labels": labels,
                                     "probs": [a["probabilities"][lab] for lab in labels],
                                     "gold_corrected": g_corr}) + "\n")
                written += 1
            if (i + 1) % 50 == 0:
                fh.flush()
                rate = (time.time() - t0) / max(1, i + 1)
                print(f"{tag}: {i+1}/{len(rows)} bundles, {written} q, "
                      f"{rate:.2f} s/bundle, eta {(len(rows)-i-1)*rate/60:.1f} min", flush=True)
    return [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]


def fit(recs: list[dict]) -> dict:
    temps, q90, q95 = {}, {}, {}
    grid = np.linspace(0.5, 3.0, 51)
    for prim in ("choice", "score", "noul"):
        sub = [r for r in recs if r["type"] == prim]
        logits = [np.log(np.clip(np.asarray(r["probs"], float), 1e-12, 1.0)) for r in sub]
        gold = [r["gold_corrected"] for r in sub]
        best_t, best_nll = 1.0, float("inf")
        for t in grid:
            nll = 0.0
            for z_row, g in zip(logits, gold):
                z = z_row / t
                z -= z.max()
                p = np.exp(z) / np.exp(z).sum()
                nll -= math.log(max(1e-12, float(p[g])))
            if nll < best_nll:
                best_t, best_nll = float(t), nll
        temps[prim] = round(best_t, 3)
        nc = []
        for z_row, g in zip(logits, gold):
            z = z_row / best_t
            z -= z.max()
            p = np.exp(z) / np.exp(z).sum()
            nc.append(1.0 - float(p[g]))
        n = len(nc)
        q90[prim] = float(np.quantile(nc, min(1.0, math.ceil((n + 1) * 0.90) / n)))
        q95[prim] = float(np.quantile(nc, min(1.0, math.ceil((n + 1) * 0.95) / n)))
    return {"temperatures": temps, "q90": q90, "q95": q95}


def evaluate(recs: list[dict], fitted: dict) -> dict:
    temps, q90, q95 = fitted["temperatures"], fitted["q90"], fitted["q95"]
    cov90 = {"total": [0, 0], "by_type": {}}
    cov95 = {"total": [0, 0], "by_type": {}}
    confs, oks = [], []
    for r in recs:
        t = temps[r["type"]]
        z = np.log(np.clip(np.asarray(r["probs"], float), 1e-12, 1.0)) / t
        z -= z.max()
        p = np.exp(z) / np.exp(z).sum()
        g = r["gold_corrected"]
        for cov, q in ((cov90, q90), (cov95, q95)):
            hit = int((1.0 - float(p[g])) <= q[r["type"]])
            cov["total"][0] += hit
            cov["total"][1] += 1
            slot = cov["by_type"].setdefault(r["type"], [0, 0])
            slot[0] += hit
            slot[1] += 1
        confs.append(float(p.max()))
        oks.append(int(int(p.argmax()) == g))
    confs, oks = np.array(confs), np.array(oks, float)
    ece = 0.0
    for b in range(10):
        m = (confs >= b / 10) & ((confs <= (b + 1) / 10) if b == 9 else (confs < (b + 1) / 10))
        if m.any():
            ece += float(m.sum()) / len(confs) * abs(confs[m].mean() - oks[m].mean())
    out = {"n": len(recs),
           "ece": round(ece, 4), "brier": round(float(np.mean((confs - oks) ** 2)), 4),
           "acc": round(float(oks.mean()), 4), "flips_vs_argmax": None}
    for name, cov in (("coverage_90", cov90), ("coverage_95", cov95)):
        out[name] = {"all": round(cov["total"][0] / cov["total"][1], 4),
                     "by_type": {k: round(v[0] / v[1], 4) for k, v in cov["by_type"].items()}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="myna-v5-prototype 2")
    ap.add_argument("--cal-rows", default="runs/v6_cal_raw.jsonl")
    ap.add_argument("--test-rows", default="runs/v6_test_raw.jsonl")
    ap.add_argument("--out", default="runs/corrected_conformal.json")
    args = ap.parse_args()

    import torch

    torch.set_num_threads(8)
    from myna.v6_engine import MynaV6

    cal = [json.loads(line) for line in (REPO / args.cal_rows).read_text().splitlines() if line.strip()]
    # `bench.cal_refit.py` already writes the corrected-gold fit for the same CAL dump; reuse it
    # rather than paying the 51-point grid again (it is the slowest part of this harness).
    cached = REPO / "runs/cal_refit.json"
    if cached.exists():
        fitted = json.loads(cached.read_text())["corrected"]
        print("using cached corrected fit from runs/cal_refit.json", flush=True)
    else:
        fitted = fit(cal)

    engine = MynaV6(args.ckpt, device="cpu")
    shipped = {"temperatures": dict(engine.temperatures),
               "q90": dict(engine.conformal_q90), "q95": dict(engine.conformal_q95)}
    engine.temperatures = {"choice": 1.0, "score": 1.0, "noul": 1.0}

    test_path = REPO / args.test_rows
    rows = [json.loads(line) for line in
            (REPO / "data" / "decision-v2-pilot" / "test.jsonl").read_text().splitlines()
            if line.strip()]
    # `score_split` is append-only and skips rows already on disk, so a partial dump resumes.
    test = score_split(engine, rows, test_path, "test")

    result = {"cal_questions": len(cal), "n_test": len(test),
              "shipped": shipped, "fitted_corrected": fitted,
              "test_under_fitted": evaluate(test, fitted),
              "test_under_shipped": evaluate(test, {
                  "temperatures": shipped["temperatures"],
                  "q90": shipped["q90"], "q95": shipped["q95"]})}
    (REPO / args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

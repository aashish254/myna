"""Recompute the v6 temperatures and conformal quantiles on the rebuilt CAL split.

    python -m bench.cal_refit --out runs/cal_refit.json

The v6 notebook fits three temperatures and two quantile sets on a CAL split it draws from
the train pool, then stores them in the checkpoint. This harness rebuilds that split
(`release/build_cal_split.py`), scores it through the released engine with temperature
scaling disabled, and refits both under two gold conventions:

* `gold_legacy` — the label as stored, which *is* the 0-based option index; this is the correct
  target. An earlier note called it "legacy" while treating a shifted label as "corrected";
  that mapping is retracted, and the refit uses `gold_legacy`.

The comparison answers a question the paper can otherwise only assert: whether the stored
values reproduce, and whether the mis-indexed score gold moved them.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "release"))


def load_cal() -> list[dict]:
    from build_cal_split import build

    cal, _ = build(REPO / "data" / "decision-v2-pilot" / "train.jsonl")
    return cal


def labels_and_golds(spec: dict) -> tuple[list[str], int, int]:
    """(option labels, corrected gold index, legacy gold index)."""
    if spec["type"] == "noul":
        return ["no", "yes"], (1 if spec["label"] else 0), (1 if spec["label"] else 0)
    crit = spec["criteria"]
    if spec["type"] == "choice":
        labels = list(crit) if isinstance(crit, dict) else [str(c) for c in crit]
        gold = labels.index(spec["label"])
        return labels, gold, gold
    if spec["type"] == "score":
        labels = [str(c) for c in crit]
        level = int(spec["label"])
        return labels, level - 1, level  # legacy keeps the notebook's 1-based index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="myna-v5-prototype 2")
    ap.add_argument("--limit", type=int, default=0, help="first N bundles (smoke runs)")
    ap.add_argument("--reuse-rows", action="store_true", help="refit from an existing rows dump")
    ap.add_argument("--rows-out", default="runs/v6_cal_raw.jsonl")
    ap.add_argument("--out", default="runs/cal_refit.json")
    args = ap.parse_args()

    out_path = REPO / args.rows_out

    def fit(recs: list[dict], gold_key: str) -> dict:
        """Per-primitive temperature (51-point grid) and conformal quantiles.

        Option counts differ within a primitive (choice runs from 2 to 77), so the fit is a
        loop over questions rather than a padded matrix.
        """
        temps, q90, q95 = {}, {}, {}
        grid = np.linspace(0.5, 3.0, 51)
        for prim in ("choice", "score", "noul"):
            sub = [r for r in recs if r["type"] == prim]
            if not sub:
                temps[prim] = q90[prim] = q95[prim] = None
                continue
            logits = [np.log(np.clip(np.asarray(r["probs"], dtype=float), 1e-12, 1.0)) for r in sub]
            gold = [r[gold_key] for r in sub]
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
            q90[prim] = round(float(np.quantile(nc, min(1.0, math.ceil((n + 1) * 0.90) / n))), 4)
            q95[prim] = round(float(np.quantile(nc, min(1.0, math.ceil((n + 1) * 0.95) / n))), 4)
        return {"temperatures": temps, "q90": q90, "q95": q95}

    def finish(recs: list[dict], stored: dict) -> None:
        # The score label is the 0-based class index, so `gold_legacy` (the label as stored)
        # is the correct target; `gold_corrected` was built on the retracted off-by-one.
        # §9.30: flags only, so the line is pasteable however a checkout invoked it.
        cmd = shlex.join(["python", "-m", "bench.cal_refit", *sys.argv[1:]])
        result = {"cmd": cmd,
                  "split": {"n_bundles": len({r["row_id"] for r in recs}),
                            "n_questions": len(recs)},
                  "stored": stored,
                  "refit": fit(recs, "gold_legacy")}
        (REPO / args.out).write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))

    if args.reuse_rows:
        recs = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
        print(f"refitting from {len(recs)} cached questions", flush=True)
        import torch

        from myna.v6_engine import MynaV6

        engine = MynaV6(args.ckpt, device="cpu")
        stored = {"temperatures": dict(engine.temperatures),
                  "q90": dict(engine.conformal_q90), "q95": dict(engine.conformal_q95)}
        finish(recs, stored)
        return

    import torch

    torch.set_num_threads(8)
    from myna.v6_engine import MynaV6

    cal = load_cal()
    if args.limit:
        cal = cal[: args.limit]
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["row_id"])

    engine = MynaV6(args.ckpt, device="cpu")
    stored = {"temperatures": dict(engine.temperatures),
              "q90": dict(engine.conformal_q90), "q95": dict(engine.conformal_q95)}
    engine.temperatures = {"choice": 1.0, "score": 1.0, "noul": 1.0}

    t0 = time.time()
    written = 0
    with open(out_path, "a") as fh:
        for i, row in enumerate(cal):
            rid = row["_meta"]["id"]
            if rid in done:
                continue
            answers = engine.predict(row["state"], row["questions"])["answers"]
            for qname, spec in row["questions"].items():
                labels, g_corr, g_legacy = labels_and_golds(spec)
                a = answers[qname]
                probs = [a["probabilities"][lab] for lab in labels]
                fh.write(json.dumps({"row_id": rid, "qname": qname, "type": spec["type"],
                                     "source": row["_meta"]["source"], "labels": labels,
                                     "probs": probs, "gold_corrected": g_corr,
                                     "gold_legacy": g_legacy}) + "\n")
                written += 1
            if (i + 1) % 50 == 0:
                fh.flush()
                rate = (time.time() - t0) / max(1, i + 1)
                print(f"{i+1}/{len(cal)} bundles, {written} questions, "
                      f"{rate:.2f} s/bundle, eta {(len(cal)-i-1)*rate/60:.1f} min", flush=True)

    recs = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    print(f"scored {len(recs)} questions", flush=True)
    finish(recs, stored)


if __name__ == "__main__":
    main()

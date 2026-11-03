"""Cost of the routing switch on the high-cardinality cell, measured rather than inferred.

    python -m bench.routing_cost --n 6 --out runs/routing_cost.json

The paper states that a 77-option question costs about eleven times more under the decoupled
path than under the packed one, because decoupled scoring re-encodes the state once per
option. This harness measures that on the same checkpoint, the same rows, and the same
process: one `predict` call per row, timer around the call only, model already warm.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="myna-v5-prototype 2")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", default="runs/routing_cost.json")
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(REPO / "src"))
    from myna.v6_engine import MynaV6

    rows = [json.loads(line) for line in
            (REPO / "data" / "decision-v2-pilot" / "test.jsonl").read_text().splitlines()
            if line.strip()]
    bank = [r for r in rows if r["_meta"]["source"] == "banking77"][: args.n]
    engine = MynaV6(args.ckpt, device="cpu")
    engine.max_packed_options = 1000          # packed path on every cell
    packed_t, routed_t = [], []
    for i, row in enumerate(bank):
        for _ in range(args.warmup):
            engine.predict(row["state"], row["questions"])
        t0 = time.perf_counter()
        engine.predict(row["state"], row["questions"])
        packed_t.append(time.perf_counter() - t0)
        engine.max_packed_options = 16        # decoupled path for K > 16
        t0 = time.perf_counter()
        engine.predict(row["state"], row["questions"])
        routed_t.append(time.perf_counter() - t0)
        engine.max_packed_options = 1000
        print(f"{i+1}/{len(bank)} packed {packed_t[-1]:.2f}s routed {routed_t[-1]:.2f}s",
              flush=True)

    def summary(xs):
        return {"n": len(xs), "median_s": round(statistics.median(xs), 3),
                "mean_s": round(statistics.fmean(xs), 3),
                "min_s": round(min(xs), 3), "max_s": round(max(xs), 3)}

    out = {
        "cell": "banking77/intent",
        "k": 77,
        "rows": [r["_meta"]["id"] for r in bank],
        "packed": summary(packed_t),
        "routed": summary(routed_t),
        "ratio_median": round(statistics.median(routed_t) / statistics.median(packed_t), 2),
        "note": "one predict() call per row, warm model, same process; the routed arm is the "
                "77-option decoupled path",
    }
    (REPO / args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

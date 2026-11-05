"""End-to-end latency of Myna-v6 on this box: packed (K<=16) and routed (K>16) paths.

Times `predict` for N questions against one state, mirroring the streaming arm's protocol in
Section VIII: one process, warm-up, then median of repeated runs. Reports p50 per question count.

    python -m bench.v6_latency --device mps --out runs/v6_latency.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def load_bundle(source: str, split: str = "test") -> dict:
    for line in (REPO / f"data/decision-v2-pilot/{split}.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["_meta"]["source"] == source:
            return row
    raise SystemExit(f"no {source} row in {split}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="myna-v5-prototype 2")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="runs/v6_latency.json")
    args = ap.parse_args()

    from myna import Myna  # noqa: E402

    engine = Myna(str(REPO / args.ckpt), device=args.device)
    out = {"cmd": "python -m bench.v6_latency", "device": args.device, "cells": {}}
    for source, label in (("agnews", "packed"), ("banking77", "routed")):
        row = load_bundle(source)
        state = row["state"]
        base = dict(row["questions"])
        name, spec = next(iter(base.items()))
        timings = {}
        for n in (1, 5, 10, 25):
            qs = {f"{name}_{i}": spec for i in range(n)}
            engine.predict(state, qs)  # warm-up
            runs = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                engine.predict(state, qs)
                runs.append((time.perf_counter() - t0) * 1000)
            timings[str(n)] = round(statistics.median(runs), 2)
        out["cells"][label] = {"source": source, "k": len(spec.get("options", [])),
                               "p50_ms": timings}
    (REPO / args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

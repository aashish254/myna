"""Dump the v6 engine's full option-probability vector for every question of a split.

    python -m bench.prob_dump --ckpt "myna-v5-prototype 2" --split test \
        --out runs/v6_prob_test.jsonl

The row-outcome witnesses store only `pred`, `gold`, `conf` and `ok`, which is enough for
accuracy and top-1 calibration but not for conformal analysis: a split-conformal set
depends on the whole probability vector and on the quantile it is compared against.
This harness writes the vector, the labels, and the gold answer for every question, so
`bench/conformal_analysis.py` can recompute coverage, set size, and per-cell (Mondrian)
coverage without re-running the 434M-parameter encoder.

The dump is append-only and resumable: a row id that is already on disk is skipped, so an
interrupted run continues instead of restarting. Rows are written in file order.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def gold_of(spec: dict) -> str:
    """The gold answer in the same key space as `MynaV6._labels`."""
    lab = spec.get("label")
    if spec["type"] == "noul":
        return "yes" if lab else "no"
    if spec["type"] == "score":
        return str(lab)
    return str(lab)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="myna-v5-prototype 2")
    ap.add_argument("--split", default="test", choices=["test", "development", "calibration"])
    ap.add_argument("--out", default="runs/v6_prob_test.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "src"))
    import os

    import torch  # noqa: E402

    from myna.v6_engine import MynaV6  # noqa: E402  (optional dependency, imported late)

    # The encoder is 434M parameters and this harness is CPU-only; the default thread
    # count leaves most of an Apple M-series box idle, which turns a 5 s row into 100 s.
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["row_id"])

    rows = [json.loads(line) for line in
            (REPO / "data" / "decision-v2-pilot" / f"{args.split}.jsonl").read_text().splitlines()
            if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    engine = MynaV6(args.ckpt, device="cpu")
    t0 = time.time()
    n_written = 0
    with open(out, "a") as fh:
        for i, row in enumerate(rows):
            rid = row["_meta"]["id"]
            if rid in done:
                continue
            answers = engine.predict(row["state"], row["questions"])["answers"]
            for qname, spec in row["questions"].items():
                ans = answers[qname]
                labels = list(ans["probabilities"])
                rec = {
                    "row_id": rid,
                    "group": row["_meta"].get("group_id", rid),
                    "source": row["_meta"]["source"],
                    "cell": f"{row['_meta']['source']}/{qname}",
                    "qname": qname,
                    "type": spec["type"],
                    "k": len(labels),
                    "labels": labels,
                    "probs": [ans["probabilities"][lab] for lab in labels],
                    "gold": gold_of(spec),
                    "pred": labels[int(max(range(len(labels)),
                                           key=lambda k: ans["probabilities"][labels[k]]))],
                    "conf": ans["confidence"],
                    "conformal_set_90": ans["conformal_set_90"],
                    "abstain": ans["abstain"],
                }
                fh.write(json.dumps(rec) + "\n")
                n_written += 1
            if (i + 1) % 10 == 0:
                fh.flush()
                rate = (time.time() - t0) / max(1, i + 1)
                print(f"{args.split}: {i+1}/{len(rows)} rows, {n_written} questions, "
                      f"{rate:.2f} s/row, eta {(len(rows)-i-1)*rate/60:.1f} min", flush=True)
    print(f"wrote {n_written} questions to {out}", flush=True)


if __name__ == "__main__":
    main()

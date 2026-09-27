"""Laya Router accuracy on the real kev decision-v2 corpus — the witness baseline.

myna-v1 trains and evaluates on decision-v2; this measures the same split with
laya's public Router so the accuracy table is apples-to-apples (speed already is:
same box). Reports per-question accuracy (choice/noul argmax, score argmax over
the level distribution), noul Brier, and grouped by source and by type.

Usage:
  PYTHONPATH="/Users/aashish/github contribution /GIT/laya" \
    uv run python -m bench.eval_laya_real --split development --n-per-source 30
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import random
from collections import defaultdict
from pathlib import Path


def laya_questions(qs):
    """laya's noul schema wants criteria omitted (or a false/true dict), not the raw list."""
    return {
        qid: ({"type": "noul", "instructions": q["instructions"]}
              if q["type"] == "noul" else
              {k: v for k, v in q.items() if k != "label"})
        for qid, q in qs.items()
    }


def score_pred(ans):
    """Discrete level for a score answer: argmax of the level distribution."""
    probs = ans["probabilities"]
    return max(probs, key=probs.get)  # keys are str(level)


def run(records, router, rng):
    # (source, qid, type) -> [correct_count, n, brier_sum]
    acc = defaultdict(lambda: [0, 0, 0.0])
    for rec in records:
        qs = rec["questions"]
        src = rec.get("_meta", {}).get("source", "?")
        try:
            out = router.predict(rec["state"], laya_questions(qs))
        except Exception as e:  # a truncated state or unsupported schema shouldn't kill the sweep
            print(f"  skip {src}: {type(e).__name__}: {e}")
            continue
        ans = out["answers"]
        for qid, q in qs.items():
            a = ans.get(qid)
            if a is None:
                continue
            t = q["type"]
            if t == "choice":
                ok = int(a["choice"] == q["label"])
                key = (src, qid, "choice")
            elif t == "score":
                ok = int(score_pred(a) == str(q["label"]))
                key = (src, qid, "score")
            else:  # noul
                p = a["noul"]
                ok = int((p >= 0.5) == bool(q["label"]))
                key = (src, qid, "noul")
                acc[key][2] += (p - float(bool(q["label"]))) ** 2
            acc[key][0] += ok
            acc[key][1] += 1
    return acc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="/Users/aashish/github contribution /GIT/kev/evals/decision-v2")
    ap.add_argument("--split", default="development")
    ap.add_argument("--n-per-source", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None,
                    help="cpu/cuda/mps; default auto. Pin to cpu to run alongside MPS training")
    ap.add_argument("--out", default="runs/laya_decision_v2.json")
    args = ap.parse_args(argv)
    # The command the witness answers to, without --suite: that default is a local
    # checkout of someone else's repo, and a published command must not depend on it
    # (§9.30 — SPEC §4.2's laya figures are quoted with the suite named in prose).
    flags = list(argv if argv is not None else sys.argv[1:])
    cmd = shlex.join(["python", "bench/eval_laya_real.py", *flags])

    from laya import Router
    router = Router(device=args.device) if args.device else Router()

    path = Path(args.suite) / f"{args.split}.jsonl"
    by_src = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by_src[r.get("_meta", {}).get("source", "?")].append(r)

    rng = random.Random(args.seed)
    records = []
    for src, rows in by_src.items():
        rng.shuffle(rows)
        records.extend(rows[: args.n_per_source])
    rng.shuffle(records)
    print(f"{len(records)} records across {len(by_src)} sources from {args.split}")

    acc = run(records, router, rng)

    rows = []
    print(f"\n{'source/qid [type]':<40} {'acc':>6} {'n':>4} {'brier':>7}")
    for (src, qid, t), (c, n, bs) in sorted(acc.items()):
        a = c / max(n, 1)
        line = f"{src}/{qid} [{t}]".ljust(40) + f" {a:6.3f} {n:4d}"
        if t == "noul":
            line += f" {bs / max(n, 1):7.4f}"
        print(line)
        rows.append({"source": src, "qid": qid, "type": t, "acc": a, "n": n,
                     "brier": (bs / n) if t == "noul" else None})

    # per-type and overall aggregates (question-weighted, matching myna's report shape)
    def agg(pred):
        sel = [r for r in rows if pred(r)]
        tot = sum(r["n"] for r in sel)
        return (sum(r["acc"] * r["n"] for r in sel) / tot if tot else float("nan")), tot
    overall, N = agg(lambda r: True)
    print(f"\nlaya {args.split}: overall acc {overall:.4f} over {N} question instances")
    for t in ("choice", "score", "noul"):
        a, n = agg(lambda r, t=t: r["type"] == t)
        print(f"  {t:6s} acc {a:.4f}  (n={n})")
    nb = [r["brier"] for r in rows if r["type"] == "noul" and r["brier"] is not None]
    print(f"  noul   Brier {sum(nb)/len(nb):.4f}" if nb else "  noul   Brier n/a")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"cmd": cmd, "split": args.split, "n_per_source": args.n_per_source,
                   "seed": args.seed, "overall_acc": overall, "n": N,
                   "by_type": {t: agg(lambda r, t=t: r["type"] == t)[0] for t in ("choice", "score", "noul")},
                   "rows": rows}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

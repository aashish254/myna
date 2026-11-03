#!/usr/bin/env python
"""Rebuild the v6 campaign's 2,000-bundle CAL split from the committed train pool.

    python release/build_cal_split.py --check
    python release/build_cal_split.py --write data/decision-v2-pilot/cal-2000.jsonl

The v6 notebook draws the split it fits temperatures and conformal quantiles on from
`train.jsonl`: group the train bundles by source, shuffle each group with
`random.Random(42)`, and take `max(20, round(2000 * n_source / n_train))` from each. This
script implements exactly that. The original 2,000-row file was never committed, so this is
a definition-level reproduction: the ids, the count, and the per-source composition are
reproducible, and a hash comparison against the campaign file is not possible.

`--check` prints the composition and a digest of the id list, so two machines can compare
the split without shipping the data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEED = 42
CAL_SIZE = 2000
MIN_PER_SOURCE = 20


def bundle_id(row: dict) -> str:
    meta = row.get("_meta", {})
    return meta.get("id") or f"{meta.get('source', '?')}/{meta.get('row', '?')}"


def build(train_path: Path) -> tuple[list[dict], list[dict]]:
    rows = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["_meta"]["source"]].append(row)
    cal, rest = [], []
    rng = random.Random(SEED)
    for src in sorted(by_source):
        items = list(by_source[src])
        rng.shuffle(items)
        n_cal = max(MIN_PER_SOURCE, int(round(CAL_SIZE * len(items) / len(rows))))
        cal.extend(items[:n_cal])
        rest.extend(items[n_cal:])
    return cal, rest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/decision-v2-pilot/train.jsonl")
    ap.add_argument("--write", default=None, help="write the split to this JSONL path")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    cal, rest = build(REPO / args.train)
    ids = [bundle_id(r) for r in cal]
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    comp = Counter(r["_meta"]["source"] for r in cal)
    questions = sum(len(r["questions"]) for r in cal)
    out = {"seed": SEED, "target": CAL_SIZE, "n_cal": len(cal), "n_train_prime": len(rest),
           "n_questions": questions, "per_source": dict(sorted(comp.items())),
           "ids_sha256": digest}
    if args.check or not args.write:
        print(json.dumps(out, indent=1))
    if args.write:
        dest = REPO / args.write
        dest.write_text("\n".join(json.dumps(r) for r in cal) + "\n")
        print("wrote", dest, len(cal), "bundles")


if __name__ == "__main__":
    main()

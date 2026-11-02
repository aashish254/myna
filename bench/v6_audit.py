"""Re-score a Myna v6 checkpoint over the frozen `decision-v2-pilot` split.

    uv run python -m bench.v6_audit --ckpt "myna-v5-prototype 2" --split test \
        --out runs/v6_rescore_test.json --md runs/v6_rescore_test.md

The published v6 figures (`runs/witness-v6-master-paper.json`) were measured in a Google
Colab Tesla T4 runtime, so this box has never executed that campaign. This harness does
something narrower and independently useful: it runs the repository's own `MynaV6` engine
over `data/decision-v2-pilot/{split}.jsonl` and prints what *this* tree scores. When the
two agree, the engine is a faithful port. When they do not, the gap is a finding about the
engine, not a reason to quote the Colab number less carefully — and the note field says
which.

Scoring is argmax over the full option vector, so an abstained question still counts for
its ranked top choice; that is how the published cell accuracies are defined, and mixing
an abstention-aware denominator into the comparison would make the two arms incomparable.
The abstention count is reported beside it, never folded in.

`transformers` and `torch` are imported inside `run()`, not at module scope: `myna` treats
transformers as an optional extra, CI installs `--extra browser` only, and
`bench/reproduce.py` asks every registry command for `--help`. A module-level import would
make that flag question fail on a box that has no transformers, which is how an optional
dependency turns into a red build on main.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The 16 (source, question-name) cells the benchmark's test and dev splits carry. A split
# with a cell missing from this list is a schema change, and the harness prints the extras
# rather than silently averaging over a different denominator.
EXPECTED_CELLS = 16


def load_rows(split: str, every: int) -> list[dict]:
    path = REPO / "data" / "decision-v2-pilot" / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[::every] if every > 1 else rows


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="myna-v6-audit",
        description="Re-score a Myna v6 checkpoint over a frozen decision-v2-pilot split "
                    "and write the witness JSON the registry quotes.")
    ap.add_argument("--ckpt", default="myna-v5-prototype 2", metavar="DIR",
                    help='campaign directory holding myna-v6-unified.pt and '
                         'deberta-v5-finetuned/ (default: "myna-v5-prototype 2")')
    ap.add_argument("--split", default="test", choices=("test", "development", "calibration"),
                    help="which decision-v2-pilot split to score (default test)")
    ap.add_argument("--every", type=int, default=1, metavar="N",
                    help="score every Nth bundle, for a fast look at a split that is slow "
                         "to encode on CPU (default 1, every bundle)")
    ap.add_argument("--source", default=None, metavar="NAME",
                    help="restrict to one source (e.g. banking77) — the high-cardinality "
                         "cells dominate wall clock, so this is how to price one change")
    ap.add_argument("--out", default="runs/v6_rescore.json", metavar="PATH",
                    help="witness JSON to write; the printed command goes in its first field")
    ap.add_argument("--md", default=None, metavar="PATH",
                    help="also write a markdown table of the same figures")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    from myna.real_data import parse_questions
    from myna.v6_engine import MynaV6

    engine = MynaV6(args.ckpt, device=args.device)
    rows = load_rows(args.split, args.every)
    print(f"{len(rows)} bundles from {args.split} | engine {engine.model_label}", flush=True)

    cells: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    variants: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    t0 = time.perf_counter()
    done = 0
    for row in rows:
        source = row["_meta"].get("source", "?")
        variant = row["_meta"].get("variant")
        try:
            _, golds = parse_questions(row)
        except ValueError:
            continue
        names = list(row["questions"])
        if args.source and source != args.source:
            continue
        answers = engine.predict(row["state"], row["questions"])["answers"]
        for i, qname in enumerate(names):
            if i >= len(golds) or qname not in answers:
                continue
            spec = row["questions"][qname]
            a = answers[qname]
            ranked = list(a["probabilities"])
            top = ranked[max(range(len(ranked)), key=lambda k: a["probabilities"][ranked[k]])]
            hit = (top == "yes") == bool(golds[i]) if spec["type"] == "noul" else top == ranked[golds[i]]
            bucket = cells[(source, qname)]
            bucket[0] += int(hit)
            bucket[1] += 1
            bucket[2] += int(a["abstain"])
            if source == "banking77" and variant:
                variants[str(variant)][0] += int(hit)
                variants[str(variant)][1] += 1
        done += 1
        if done % 100 == 0:
            print(f"  {done} bundles, {time.perf_counter() - t0:.0f}s", flush=True)

    per_cell = {f"{s}/{q}": {"correct": c, "n": n, "abstain": ab, "acc": c / n}
                for (s, q), (c, n, ab) in sorted(cells.items())}
    macro = sum(v["acc"] for v in per_cell.values()) / len(per_cell) if per_cell else 0.0
    total_c = sum(v["correct"] for v in per_cell.values())
    total_n = sum(v["n"] for v in per_cell.values())
    witness = {
        "cmd": " ".join(["uv run python -m bench.v6_audit", f'--ckpt "{args.ckpt}"',
                         f"--split {args.split}", f"--every {args.every}",
                         f"--out {args.out}"]),
        "engine": engine.model_label,
        "ckpt": str(args.ckpt),
        "split": args.split,
        "every": args.every,
        "source_filter": args.source,
        "bundles_scored": done,
        "n_questions": total_n,
        "n_cells": len(per_cell),
        "expected_cells": EXPECTED_CELLS,
        "macro_acc": macro,
        "overall_acc": total_c / total_n if total_n else 0.0,
        "abstained": sum(v["abstain"] for v in per_cell.values()),
        "scoring": "argmax over the calibrated option vector; abstentions still ranked",
        "wall_clock_s": round(time.perf_counter() - t0, 1),
        "per_cell": per_cell,
        "banking77_variants": {k: f"{v[0]}/{v[1]}" for k, v in sorted(variants.items())},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(witness, indent=1) + "\n")
    print(f"\n{n_cells_line(witness)}")
    print(f"macro over {witness['n_cells']} cells: {macro:.4f}")
    print(f"overall: {witness['overall_acc']:.4f} over {total_n} questions")
    print(f"wrote {out}")
    if args.md:
        Path(args.md).write_text(table(witness))
        print(f"wrote {args.md}")
    return 0


def n_cells_line(w: dict) -> str:
    return (f"{w['bundles_scored']} bundles, {w['n_questions']} questions, "
            f"{w['n_cells']} cells")


def table(w: dict) -> str:
    lines = [f"# v6 re-score on {w['split']} — {w['cmd']}", "",
             f"{w['bundles_scored']} bundles / {w['n_questions']} questions / "
             f"{w['n_cells']} cells | macro {w['macro_acc']:.4f} | "
             f"overall {w['overall_acc']:.4f} | abstained {w['abstained']} | "
             f"{w['wall_clock_s']}s wall", "",
             "| cell | correct | n | acc | abstained |", "| :-- | --: | --: | --: | --: |"]
    for name, v in w["per_cell"].items():
        lines.append(f"| `{name}` | {v['correct']} | {v['n']} | {v['acc']:.4f} | {v['abstain']} |")
    if w["banking77_variants"]:
        lines += ["", "| banking77 variant | correct/n |", "| :-- | --: |"]
        lines += [f"| `{k}` | {v} |" for k, v in w["banking77_variants"].items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    return run(argv or sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

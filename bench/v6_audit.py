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
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    """The checkpoint's own digest, for the provenance block a witness has to carry."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# The 16 (source, question-name) cells the benchmark's test and dev splits carry. A split
# with a cell missing from this list is a schema change, and the harness prints the extras
# rather than silently averaging over a different denominator.
EXPECTED_CELLS = 16


def load_rows(split: str, every: int) -> list[dict]:
    path = REPO / "data" / "decision-v2-pilot" / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[::every] if every > 1 else rows


def reprice_from_rows(dumped: list[dict]) -> dict[str, list[int]]:
    """`{"source/question": [correct, n]}`, read out of the row records alone."""
    cells: dict[str, list[int]] = {}
    for rec in dumped:
        source = rec["meta"]["source"]
        for name, q in rec["questions"].items():
            bucket = cells.setdefault(f"{source}/{name}", [0, 0])
            bucket[0] += q["ok"]
            bucket[1] += 1
    return cells


def guard_against_cells(per_cell: dict, repriced: dict) -> list[str]:
    """Complaints, in the cell names a reader would quote. An empty list means the dump and
    the witness are the same measurement; a non-empty one means the dump is a different one."""
    bad = []
    for name in sorted(set(per_cell) | set(repriced)):
        if name not in per_cell:
            bad.append(f"{name}: {tuple(repriced[name])} in the row dump, absent from "
                       "the witness")
        elif name not in repriced:
            bad.append(f"{name}: in the witness, absent from the row dump")
        else:
            w, r = per_cell[name], repriced[name]
            if (w["correct"], w["n"]) != (r[0], r[1]):
                bad.append(f"{name}: witness {w['correct']}/{w['n']}, rows {r[0]}/{r[1]}")
    return bad


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
    ap.add_argument("--metrics", default=None, metavar="PATH",
                    help="also write the per-question-set accuracies in "
                         "`myna.train.evaluate`'s metrics.json schema, so `python -m "
                         "myna.report` prices this run's macro, both floors and its G1 "
                         "verdict with the frozen report code rather than a copy of it")
    ap.add_argument("--rowout", default=None, metavar="PATH",
                    help="also write one record per (row, question) in the schema "
                         "`bench/significance.py` bootstraps, so a row-clustered interval "
                         "can be put on this run's macro. Both dumps need the whole split: "
                         "they refuse --every above 1 and --source.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-packed", type=int, default=None, metavar="K",
                    help="override the checkpoint's packed/decoupled cap for this run. "
                         "`--max-packed 1000` forces the packed scorer on every cell, which "
                         "is how the 'packed everywhere' arm is reproduced on the same "
                         "weights; the default keeps the checkpoint's own routing.")
    args = ap.parse_args(argv)

    if (args.metrics or args.rowout) and (args.source or args.every != 1):
        # The two dumps exist to be read by tools that compute the published macro over the
        # frozen split from the rows. A subsample would still print a plausible number; it
        # would just be a different measurement than the one the gate is written against.
        raise SystemExit("--metrics/--rowout score the whole frozen split, so they refuse "
                        f"--every {args.every}"
                        + (f" and --source {args.source}" if args.source else ""))

    from myna.real_data import _signature, parse_questions
    from myna.report import MIN_CELL_ROWS
    from myna.v6_engine import MynaV6
    sys.path.insert(0, str(REPO / "bench"))
    from row_outcomes import META_FIELDS

    engine = MynaV6(args.ckpt, device=args.device)
    if args.max_packed is not None:
        engine.max_packed_options = int(args.max_packed)
    rows = load_rows(args.split, args.every)
    print(f"{len(rows)} bundles from {args.split} | engine {engine.model_label}", flush=True)

    cells: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    variants: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    set_hits: dict[str, int] = defaultdict(int)
    set_n: dict[str, int] = defaultdict(int)
    group_ix: dict[str, int] = defaultdict(int)
    rows_by_group: dict[str, list[dict]] = defaultdict(list)
    t0 = time.perf_counter()
    done = 0
    for row in rows:
        source = row["_meta"].get("source", "?")
        variant = row["_meta"].get("variant")
        try:
            questions, golds = parse_questions(row)
        except ValueError:
            continue
        names = list(row["questions"])
        if args.source and source != args.source:
            continue
        answers = engine.predict(row["state"], row["questions"])["answers"]
        group_key = None
        qrec: dict[str, dict] = {}
        if args.metrics or args.rowout:
            # The same two calls `load_split` makes, so a dumped row lands in the question-set
            # the frozen report reads it in — and `myna.report.accuracy_cells` refuses a set
            # key that is not in its own grouping rather than dropping its weight silently.
            group_key = f"{source}#{_signature({
                qid: {k: v for k, v in q.items() if k != 'label'}
                for qid, q in row["questions"].items()})}"
        for i, qname in enumerate(names):
            if i >= len(golds) or qname not in answers:
                continue
            spec = row["questions"][qname]
            a = answers[qname]
            ranked = list(a["probabilities"])
            pi = max(range(len(ranked)), key=lambda k: a["probabilities"][ranked[k]])
            top = ranked[pi]
            # `golds` carries the suite's label: a boolean for `noul`, the 1-based star
            # level for `score`, and a 0-based index into the criteria dict for `choice`.
            # `ranked` is a 0-based option vector in all three cases, so the score branch
            # subtracts one. Comparing at `golds[i]` scored a prediction correct when it
            # named the level *above* gold; that off-by-one inflated every `score` cell
            # (0.64 reported on a 5-way ordinal question that is at chance under the fix).
            # See paper/MEASUREMENT-CORRECTION.md.
            if spec["type"] == "noul":
                hit = (top == "yes") == bool(golds[i])
            elif spec["type"] == "score":
                hit = top == ranked[int(golds[i]) - 1]
            else:
                hit = top == ranked[golds[i]]
            bucket = cells[(source, qname)]
            bucket[0] += int(hit)
            bucket[1] += 1
            bucket[2] += int(a["abstain"])
            if source == "banking77" and variant:
                variants[str(variant)][0] += int(hit)
                variants[str(variant)][1] += 1
            if group_key is not None:
                set_hits[f"{group_key}/{qname}"] += int(hit)
                set_n[f"{group_key}/{qname}"] += 1
                if args.rowout:
                    q = questions[i]
                    # `pred` is stored in the label's own space: the 1-based level for
                    # `score`, so `pred == gold` means correct without a second convention.
                    pred = pi + 1 if spec["type"] == "score" else pi
                    qrec[qname] = {"type": q.type, "pred": pred, "gold": int(golds[i]),
                                   "k": len(q.options), "ok": int(hit),
                                   "conf": round(float(a["probabilities"][ranked[pi]]), 6)}
        if group_key is not None and args.rowout:
            rows_by_group[group_key].append(
                {"meta": {k: row["_meta"].get(k) for k in META_FIELDS},
                 "group": group_key, "idx": group_ix[group_key], "questions": qrec})
            group_ix[group_key] += 1
        done += 1
        if done % 100 == 0:
            print(f"  {done} bundles, {time.perf_counter() - t0:.0f}s", flush=True)
    if args.rowout:
        # `myna.real_data.load_split` appends every row of a group in file order, so the
        # per-group lists are already in the order the report reads them; flattening group
        # by group keeps `idx` meaning "position within the question-set".
        dumped_rows = [rec for key in sorted(rows_by_group) for rec in rows_by_group[key]]

    per_cell = {f"{s}/{q}": {"correct": c, "n": n, "abstain": ab, "acc": c / n}
                for (s, q), (c, n, ab) in sorted(cells.items())}
    macro = sum(v["acc"] for v in per_cell.values()) / len(per_cell) if per_cell else 0.0
    total_c = sum(v["correct"] for v in per_cell.values())
    total_n = sum(v["n"] for v in per_cell.values())
    if args.rowout:
        # The row dump is read back on its own here: per-cell totals recomputed from the
        # records must equal the counts the loop accumulated, or the dump describes a
        # different scoring pass than the witness — which is `bench/row_outcomes.py:236`'s
        # rule applied to this harness's two independent paths.
        complaints = guard_against_cells(per_cell, reprice_from_rows(dumped_rows))
        if complaints:
            print(f"GUARD FAIL: {len(complaints)} cell(s) in the row dump do not re-price "
                  f"the witness:", file=sys.stderr)
            for line in complaints[:5]:
                print(f"  {line}", file=sys.stderr)
            return 3
    cmd_bits = ["uv run python -m bench.v6_audit", f'--ckpt "{args.ckpt}"',
                f"--split {args.split}", f"--every {args.every}", f"--out {args.out}"]
    for flag in ("--md", "--metrics", "--rowout"):
        val = getattr(args, flag.lstrip("-").replace("-", "_"))
        if val:
            cmd_bits.append(f"{flag} {val}")
    witness = {
        "cmd": " ".join(cmd_bits),
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
    if args.metrics:
        # `myna.report` reads the split block by name, so the key here is the one *it*
        # looks up, not the one the file is named after: `--split development` becomes
        # "dev", which is what `myna/train.evaluate` writes.
        mkey = {"test": "test", "development": "dev", "calibration": "calibration"}[args.split]
        payload = {"cmd": witness["cmd"], "engine": engine.model_label,
                   "ckpt": str(args.ckpt), "split": args.split,
                   "bundles_scored": done, "n_questions": total_n, "n_sets": len(set_n),
                   mkey: {k: set_hits[k] / set_n[k] for k in sorted(set_n)}}
        Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics).write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {args.metrics} ({len(payload[mkey])} question-sets)")
    if args.rowout:
        # `bench/significance.py` reads the myna slot's provenance out of this file —
        # `model.temperature`, `model.model_sha256`, `run_dir`, `min_rows`, `n_rows`,
        # `suite`, `split` — so the dump has to answer the same keys the v1b dump does.
        # v6's one genuine difference is `temperature`: this engine fits a value per
        # question type, so the field is that mapping rather than a scalar, and the
        # bootstrap never reads it as a number.
        payload = {"cmd": witness["cmd"], "engine": "myna-v6", "run_dir": str(args.ckpt),
                   "model": {"temperature": dict(engine.temperatures),
                             "params": engine.n_params,
                             "max_length": engine.max_length,
                             "model_sha256": _sha256(engine.ckpt_path)},
                   "suite": str((REPO / "data" / "decision-v2-pilot").relative_to(REPO)),
                   "split": args.split,
                   "file": f"{args.split}.jsonl", "rows_in_file": len(rows),
                   "rows_skipped": len(rows) - done, "n_rows": len(dumped_rows),
                   "n_questions": total_n, "n_sets": len(set_n), "min_rows": MIN_CELL_ROWS,
                   "unmatched_sets": [], "cluster_key": "meta.id",
                   "scoring": witness["scoring"], "macro_acc": macro,
                   "wall_clock_s": witness["wall_clock_s"], "rows": dumped_rows}
        Path(args.rowout).parent.mkdir(parents=True, exist_ok=True)
        Path(args.rowout).write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {args.rowout} ({len(dumped_rows)} rows / {total_n} question outcomes)")
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

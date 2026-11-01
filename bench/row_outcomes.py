#!/usr/bin/env python
"""Per-question outcomes for myna on a frozen split — the artifact a paired test needs.

    uv run python bench/row_outcomes.py --out runs/myna_v1b_test.rowoutcomes.json

Every committed witness in this repository stops at one accuracy per `(source, question)`
cell: `runs/v1b_kaggle_3600b.metrics.json` keys are `group#sig/question` and its values are
means, and `runs/v1b_kaggle_3600b.report.json` collapses those means to 16 numbers. A mean
cannot be bootstrapped. To put an interval on 0.4893 — or to ask whether myna and a
competitor differ *on the same rows* rather than on two different samples of the same
cells — the unit has to be the row, and 1,440 question outcomes have to exist somewhere
outside the head that computed them. Nothing here decides a gate; it manufactures the
input that `bench/significance.py` reads.

Three things make this more than a dump:

* **The loop is imported, not rewritten.** `diag_question_ablation.score()` is
  `myna.train.evaluate()`'s chunk-64 loop with the per-row predictions kept, and it is the
  code that already reproduced all 716 committed per-set accuracies with 0 tie flips
  (SPEC §9.48). Re-deriving it here — a different chunk width, a different summation order,
  `--device mps` instead of cpu — would silently reprice the checkpoint being measured.
* **The cells must reproduce the report.** `--report`'s 16 cell accuracies, sizes, majority
  and uniform floors are recomputed through `myna.report.cell_stats` / `accuracy_cells` /
  `roll_up` from *these rows* and compared key for key, under the ablation harness's own
  float tolerance and row-slack. A dump whose guard fails is not written: an outcome table
  that is a different measurement than the published one is worse than none, because it
  looks like the published one.
* **Rows keep their identity.** `myna.real_data.load_split` discards `_meta` — an `Example`
  is `(state, workflow, gold)` — so predictions have nothing to be joined back to. This
  recovers the join by calling `load_split`'s own parts (`parse_questions` for the skip
  rule, `_signature` for the group key) and then refuses to continue unless every group's
  row count matches the split it loaded. The cluster key a bootstrap needs is
  `_meta.id`, and it is load-bearing: `_meta.row` collides on 316 of these 1,176 rows and
  `_meta.row_sha256` on 219, so clustering on either would merge distinct rows and shrink
  the resample.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "bench"))

from myna.real_data import _signature, load_suite, parse_questions  # noqa: E402
from myna.report import (MIN_CELL_ROWS, accuracy_cells, cell_stats, macro,  # noqa: E402
                         roll_up, strata)

from diag_question_ablation import (FLOAT_TOL, ROW_SLACK_KEYS, load_run,  # noqa: E402
                                    score)

#: `score()`'s chunk width, named so the reconstruction below is checked against the
#: contract rather than against a copy of the number.
CHUNK = 64

#: The `_meta` fields a reader needs to cluster and to cite the corpus. Not a filter on the
#: rows — every row carries all of them (`_meta.id` is unique over 1,176/1,176).
META_FIELDS = ("id", "source", "row", "row_sha256", "text_sha256", "group_id", "variant",
               "repo", "revision", "split")


def row_identities(path):
    """{group_key: [row metadata, ...]} in the order `load_split` appends the rows.

    Mirrors `load_split` by calling the same two functions it calls, so the mirror cannot
    drift from the loader without the counts disagreeing — which `check_identities` makes
    fatal. `_meta` is kept whole for the fields listed above and nothing else: the state
    text is already the `Example`, and copying it would double the witness for free.
    """
    ids: dict[str, list] = defaultdict(list)
    n_skipped = 0
    n_rows = 0
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            n_rows += 1
            try:
                parse_questions(row)
            except ValueError:
                n_skipped += 1
                continue
            sig = _signature({qid: {k: v for k, v in q.items() if k != "label"}
                              for qid, q in row["questions"].items()})
            meta = row.get("_meta") or {}
            source = meta.get("source", "suite")
            ids[f"{source}#{sig}"].append({k: meta.get(k) for k in META_FIELDS})
    return dict(ids), n_rows, n_skipped


def check_identities(groups, ids):
    """The join is only good if the mirror and the loader saw the same rows, in the same order."""
    bad = []
    if set(ids) != set(groups):
        miss, extra = sorted(set(groups) - set(ids)), sorted(set(ids) - set(groups))
        bad.append(f"the two readings of the split disagree about which groups exist: "
                   f"{len(miss)} only in the loader (first {miss[:1]}), "
                   f"{len(extra)} only in the mirror (first {extra[:1]})")
    for key in sorted(set(ids) & set(groups)):
        if len(ids[key]) != len(groups[key][1]):
            bad.append(f"{key}: the loader made {len(groups[key][1])} examples and the mirror "
                       f"found {len(ids[key])} rows")
    return bad


def outcomes(groups, preds, ids):
    """(rows, per-set hits) — one record per row, per question, plus the hits that re-price acc.

    `score()` appends one tuple of parallel lists per chunk, in the order the chunks were
    taken, so flattening per group recovers the row order `data[i:i+64]` walked. The chunk
    sizes are re-derived from `range(0, len(data), CHUNK)` — the same expression the loop
    uses — and compared with what came back, so a changed chunk width in `score()` fails
    here instead of mis-aligning 1,440 predictions.
    """
    out_rows, set_hits, set_n, q_total = [], defaultdict(int), defaultdict(int), 0
    for wf, (questions, data) in groups.items():
        sizes = [min(CHUNK, j) for j in (len(data) - i for i in range(0, len(data), CHUNK))]
        flat = {}
        for q in questions:
            key = f"{wf}/{q.name}"
            chunks = preds[key]
            if [len(c[0]) for c in chunks] != sizes:
                raise SystemExit(
                    f"{key}: score() returned chunks {[len(c[0]) for c in chunks]} but this "
                    f"group of {len(data)} rows chunks as {sizes} at width {CHUNK}")
            flat[q.name] = [row for ch in chunks for row in zip(*ch)]
            if len(flat[q.name]) != len(data):
                raise SystemExit(f"{key}: flattened {len(flat[q.name])} outcomes for a "
                                 f"{len(data)}-row group")
        for j, example in enumerate(data):
            qrec = {}
            for q in questions:
                pred, gold, conf = flat[q.name][j]
                qrec[q.name] = {"type": q.type, "pred": pred, "gold": gold,
                                "k": len(q.options), "ok": int(pred == gold),
                                "conf": round(conf, 6)}
            out_rows.append({"meta": ids[wf][j], "group": wf, "idx": j,
                             "questions": qrec})
            q_total += len(qrec)
        for q in questions:
            key = f"{wf}/{q.name}"
            set_hits[key] += sum(1 for pr, gd, _c in flat[q.name] if pr == gd)
            set_n[key] += len(flat[q.name])
    return out_rows, dict(set_hits), dict(set_n), q_total


def guard_against_report(cell_rows, report_rows):
    """Compare the recomputed cells with the published report, under the ablation's tolerance.

    Both sides are `roll_up` rows, so the guard is on the whole table a reader would print —
    accuracy, size, both floors, the type and the set count — not just the accuracy.

    Returns `(bad, tie_flips, compared)`. A one-row difference is the same slack
    `diag_question_ablation.guard_against_metrics` grants a tie that flips between the T4
    that trained and the CPU that replays; more than `ROW_SLACK_KEYS` of them is a second
    measurement, not a rounding.
    """
    bad, flips = [], 0
    published = {(r["source"], r["question"]): r for r in report_rows}
    recomputed = {(r["source"], r["question"]): r for r in cell_rows}
    if set(published) != set(recomputed):
        miss, extra = sorted(set(published) - set(recomputed)), sorted(set(recomputed) - set(published))
        bad.append(f"cell sets differ: {len(miss)} only in the report "
                   f"{[f'{s}/{q}' for s, q in miss][:3]}, {len(extra)} only here "
                   f"{[f'{s}/{q}' for s, q in extra][:3]}")
    for key in sorted(set(published) & set(recomputed)):
        p, c = published[key], recomputed[key]
        delta = abs(c["acc"] - p["acc"])
        if delta > FLOAT_TOL:
            one_row = 1.0 / max(p["n"], 1)
            if delta <= 1.5 * one_row:
                flips += 1
            else:
                bad.append(f"{key[0]}/{key[1]}: these rows give {c['acc']:.6f}, the committed "
                           f"report gives {p['acc']:.6f} (delta {delta:.6f} > "
                           f"{one_row:.6f} = one row)")
        for field in ("n", "majority", "uniform", "type", "sets"):
            if c[field] != p[field]:
                bad.append(f"{key[0]}/{key[1]}: {field} is {c[field]!r} here and "
                           f"{p[field]!r} in the report")
    return bad, flips, len(set(published) & set(recomputed))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", default="runs/v1b-kaggle-3600b",
                    help="directory holding model.pt and tokenizer.json")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--suite", default="data/decision-v2-pilot")
    ap.add_argument("--split", default="test", choices=("test", "dev", "calibration"))
    ap.add_argument("--report", default="runs/v1b_kaggle_3600b.report.json",
                    help="the committed report whose 16 cells this dump must reproduce")
    ap.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    ap.add_argument("--min-rows", type=int, default=MIN_CELL_ROWS)
    ap.add_argument("--out", default="runs/myna_v1b_test.rowoutcomes.json")
    args = ap.parse_args(argv)
    given = list(sys.argv[1:]) if argv is None else list(argv)
    cmd = " ".join(["uv run python bench/row_outcomes.py"] + [json.dumps(a) if " " in a
                                                               else a for a in given])
    print("$ " + cmd, flush=True)

    path = Path(args.suite) / {"test": "test.jsonl", "dev": "development.jsonl",
                               "calibration": "calibration.jsonl"}[args.split]
    if not path.exists():
        print(f"--suite {args.suite} has no {path.name}: the rows are the unit here, so the "
              f"split that was scored has to be the split on disk", file=sys.stderr)
        return 2
    report_path = Path(args.report)
    if not report_path.exists():
        print(f"--report {report_path} is not on disk: this dump exists to reproduce a "
              f"committed table, so it needs one", file=sys.stderr)
        return 2

    model, tok, meta = load_run(args.run_dir, args.device, args.tokenizer)
    groups = load_suite(args.suite)[args.split]
    if not groups:
        print(f"{args.split} split of {args.suite} is empty on disk", file=sys.stderr)
        return 2
    ids, n_rows_file, n_skipped = row_identities(path)
    bad = check_identities(groups, ids)
    if bad:
        for line in bad:
            print(f"GUARD FAIL: {line}", file=sys.stderr)
        return 3

    t0 = time.time()
    accs, preds, totals = score(model, tok, groups, args.device, meta["temperature"])
    rows, set_hits, set_n, n_q = outcomes(groups, preds, ids)
    print(f"{len(rows)} rows / {n_q} question outcomes over {len(groups)} question-sets "
          f"in {time.time() - t0:.1f}s on {args.device} · temperature "
          f"{meta['temperature']} · model {meta['model_sha256'][:12]}", flush=True)

    # The dumped rows must price the same accuracies `score()` summed, or the row list is a
    # picture of the run rather than its record.
    diff = [k for k in accs if abs(set_hits[k] / max(set_n[k], 1) - accs[k]) > 1e-9]
    if diff:
        print(f"GUARD FAIL: {len(diff)} per-set accuracies differ between the summed loop and "
              f"the dumped rows (first {diff[0]})", file=sys.stderr)
        return 3

    stats = cell_stats(groups)
    cells, unmatched = accuracy_cells(accs, groups)
    cls = strata(groups)
    kept = roll_up(stats, cells, args.min_rows)
    scored = [r for r in kept if r["acc"] is not None]
    # `accuracy_cells` weights each per-set accuracy by the rows in that set; feeding it the
    # per-set means rebuilt from the dumped hits is the same call `myna.report` makes on
    # metrics.json, so a difference here is the dump, not the aggregation.
    recalc, _ = accuracy_cells({k: set_hits[k] / set_n[k] for k in set_hits}, groups)
    report = json.loads(report_path.read_text())
    bad, flips, compared = guard_against_report(roll_up(stats, recalc, args.min_rows),
                                                report["cells"])
    macro_acc = macro(scored, "acc")
    if abs(macro_acc - report["macro"]["acc"]) > 1e-9:
        bad.append(f"macro over these rows is {macro_acc!r} and the committed report prints "
                   f"{report['macro']['acc']!r}")
    if bad:
        for line in bad:
            print(f"GUARD FAIL: {line}", file=sys.stderr)
        print("nothing was written", file=sys.stderr)
        return 3
    print(f"guard: {compared} cells reproduced {report_path.name} exactly "
          f"(tie flips {flips}, tol {FLOAT_TOL}) · macro {macro_acc!r}", flush=True)

    # Hits per cell, counted from the dumped rows: `acc` above is a mean, and a reader who
    # wants to re-derive it — or a test that wants to mutate one outcome and watch the
    # interval move — needs the numerator beside it.
    hits = defaultdict(int)
    for row in rows:
        source = row["meta"]["source"]
        for name, q in row["questions"].items():
            hits[(source, name)] += q["ok"]
    payload = {"cmd": cmd, "engine": "myna", "run_dir": str(args.run_dir), "model": meta,
               "suite": str(args.suite), "split": args.split, "file": path.name,
               "rows_in_file": n_rows_file, "rows_skipped": n_skipped,
               "n_rows": len(rows), "n_questions": n_q, "n_sets": len(groups),
               "min_rows": args.min_rows, "unmatched_sets": unmatched,
               "cluster_key": "meta.id",
               "cluster_key_check": {
                   "distinct_ids": len({r["meta"]["id"] for r in rows}),
                   "distinct_row": len({r["meta"]["row"] for r in rows}),
                   "distinct_row_sha256": len({r["meta"]["row_sha256"] for r in rows})},
               "strata": cls,
               "cells": [dict(r, hits=hits[(r["source"], r["question"])]) for r in kept],
               "macro": {"acc": macro_acc, "majority": macro(scored, "majority"),
                         "uniform": macro(scored, "uniform"),
                         "cells_scored": len(scored), "cells_kept": len(kept)},
               "guard": {"report": str(report_path), "report_macro": report["macro"]["acc"],
                         "cells_compared": compared,
                         "tie_flips": flips, "float_tol": FLOAT_TOL,
                         "row_slack_keys": ROW_SLACK_KEYS, "ok": True},
               "rows": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"row identity: {payload['n_rows']} rows → "
          f"{payload['cluster_key_check']['distinct_ids']} distinct _meta.id "
          f"(vs {payload['cluster_key_check']['distinct_row']} distinct _meta.row and "
          f"{payload['cluster_key_check']['distinct_row_sha256']} distinct _meta.row_sha256 — "
          f"clustering on either of those two would merge rows that are not the same row)")
    for r in payload["cells"]:
        acc = "   —  " if r["acc"] is None else f"{r['acc']:.4f}"
        print(f"  {r['source']}/{r['question']:<12s} {r['type']:6s} n={r['n']:>4d} "
              f"acc={acc} maj={r['majority']:.4f} unif={r['uniform']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

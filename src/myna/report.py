"""Stratified reporting: every accuracy number next to the floors it must beat (SPEC §5 P3 3g).

The aggregate is marketing and the strata are evidence, so this module's unit is
the **(source, question) cell** — the same granularity `evaluate()` already
reports, and the same one §4.2's floors are computed on. For each cell it carries
three numbers that mean something only together:

* the model's accuracy, **weighted by rows** across the question-sets that hold
  that cell (an unweighted mean lets a 1-row group outvote a 116-row one);
* the cell's **majority floor** — what a constant predictor scores on those exact
  labels — and its **uniform floor**, `1/options`;
* `max(model, majority)` beside the model's own number, and the gap labelled for
  what it is: the clip's worth, not the model's (SPEC §5 P9: 9c);
* which **stratum** the source is in: *shared-instruction* (a stable schema reused
  by many rows) or *per-row-instruction* (boolq, mnli, where the instruction text
  *is* the row). Nine vs two on this suite is why one aggregate number is not a
  result — and `strata()` derives that split from the instruction strings rather
  than naming sources, because the group keys would say something else entirely.

For `noul` cells the table also carries **Brier** beside the accuracy, because an
argmax alone over-reports a binary cell in both directions (§9.24): 0.55 accuracy and
0.55 accuracy are different results if one ranks the gold label and the other does
not. It reads `evaluate()`'s `:brier` sidecars, so a metrics file that predates them
leaves the column empty — which prints as an em dash and is said out loud, never as a
0.

Nothing here runs inference. It reads what `myna.train` wrote (`metrics.json`) and
the frozen split it scored, so the floors and the weights come from the rows on
disk rather than from constants in this file — and a Kaggle checkpoint is reportable
from the laptop with the split that shipped beside it.

Usage:
    python -m myna.report --suite data/decision-v2-pilot --split test \\
        --metrics runs/myna-v1-rich/metrics.json
    python -m myna.report ... --laya runs/laya_decision_v2_test.json --out runs/report.json
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections import defaultdict
from pathlib import Path

from .real_data import load_split

#: §4.2's floor macro is pooled over cells with at least this many rows, so a cell
#: of 2 rows cannot join the average that decides G1.
MIN_CELL_ROWS = 30


def _source_of(group_key: str) -> str:
    return group_key.split("#", 1)[0]


def cell_stats(groups: dict) -> dict:
    """{group_key: (questions, examples)} -> {(source, qname): cell}.

    Two floors, both read off the rows rather than from a constant:

    * `majority` — one label index answered every time, scored on this cell's golds.
    * `uniform` — chance: pick uniformly among the options *of the set being asked*,
      so a cell whose sets disagree about option count is row-weighted, not priced at
      whichever set happened to be read first. They do disagree: the suite
      randomizes distractors per row, so agnews/topic appears with 4 options (104
      rows) and with 5 (12), and banking77/intent with 77 and 78. `options` keeps the
      whole distribution for that reason, and `main()` prints the mixed cells.

    noul is scored by Brier elsewhere, but its gold is still a label index, so the
    same histogram gives its floor."""
    out: dict[tuple[str, str], dict] = {}
    for key, (questions, examples) in groups.items():
        source = _source_of(key)
        for i, q in enumerate(questions):
            c = out.setdefault((source, q.name), {"n": 0, "type": q.type, "sets": 0,
                                                  "options": defaultdict(int),
                                                  "chance": 0.0, "labels": defaultdict(int)})
            c["n"] += len(examples)
            c["sets"] += 1
            c["options"][len(q.options)] += len(examples)
            c["chance"] += len(examples) / max(len(q.options), 1)
            for e in examples:
                c["labels"][e.gold[i]] += 1
    for c in out.values():
        labels = c.pop("labels")
        c["options"] = dict(c["options"])
        c["majority"] = (max(labels.values()) / c["n"]) if c["n"] and labels else 0.0
        c["uniform"] = (c["chance"] / c["n"]) if c["n"] else 0.0
    return out


def accuracy_cells(metrics: dict, groups: dict) -> tuple[dict, list[str]]:
    """`evaluate()`'s {"source#sig8/qname": acc} -> ({(source, qname): {acc, n, sets}},
    unmatched keys).

    Weighted by the rows in each question-set, because the map is one entry *per
    set*: without the weighting a singleton set and a 300-row set both count once.
    Returns `(cells, unmatched)` — a set key that is not in `groups` contributes no
    weight, and that has to be printed rather than absorbed.
    """
    accs: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    unmatched = []
    for key, val in metrics.items():
        if key.endswith(":brier") or key.endswith(":ece"):
            continue
        wf, qname = key.split("/", 1)
        n = len(groups[wf][1]) if wf in groups else 0
        if wf not in groups:
            unmatched.append(key)
        accs[(_source_of(wf), qname)].append((val, n))
    out = {}
    for k, pairs in accs.items():
        tot = sum(n for _v, n in pairs)
        out[k] = {"acc": (sum(v * n for v, n in pairs) / tot) if tot
                  else sum(v for v, _n in pairs) / len(pairs),
                  "n": tot, "sets": len(pairs)}
    return out, unmatched


def strata(groups: dict) -> dict:
    """Per source: is its instruction a stable schema, or is the row's own text
    the question?

    Measured on the instruction strings, not on the adapter's group keys, and the
    difference matters. `_signature` covers the full question JSON *including* the
    per-option `criteria` descriptions, which kev randomizes row to row, so on the
    eval splits almost every row is its own question-set (agnews: 116 rows over 113
    sets) even though all 116 ask the identical question. Keying the strata on sets
    would call nine sources "per-row" and hide the real point: boolq and mnli are
    per-row because their *instruction text* is the row's question or hypothesis, so
    no amount of grouping makes them share a template.

    So: the share of a source's *labelled question slots* whose instruction repeats
    somewhere in that source (slots, not rows, because one row can ask five
    questions). Also reported, because they are the numbers a reader will want
    next: `distinct_instructions`, `sets` (the exact-signature groups), and
    `multi_row_set_share` — how often a whole question-set is reused.
    """
    per: dict[str, dict] = defaultdict(lambda: {"rows": 0, "sets": 0, "multi_rows": 0,
                                                 "instr_rows": defaultdict(int)})
    for key, (questions, examples) in groups.items():
        p = per[_source_of(key)]
        p["rows"] += len(examples)
        p["sets"] += 1
        if len(examples) > 1:
            p["multi_rows"] += len(examples)
        for q in questions:
            p["instr_rows"][q.instruction] += len(examples)
    out = {}
    for source, p in per.items():
        slots = sum(p["instr_rows"].values())
        repeated = sum(n for n in p["instr_rows"].values() if n > 1)
        share = repeated / slots if slots else 0.0
        out[source] = {"rows": p["rows"], "sets": p["sets"], "slots": slots,
                       "distinct_instructions": len(p["instr_rows"]),
                       "multi_row_set_share": p["multi_rows"] / p["rows"] if p["rows"] else 0.0,
                       "repeated_instruction_share": share,
                       "class": "shared-instruction" if share >= 0.5 else "per-row-instruction"}
    return out


def brier_cells(metrics: dict, groups: dict) -> dict:
    """`evaluate()`'s `{"<key>/<qname>:brier": mean}` sidecars -> {(source, qname): brier}.

    Row-weighted across the question-sets, for the same reason `accuracy_cells` is:
    the map is one entry per set. Only `noul` cells have a sidecar to find, and a
    cell with none is *not* a cell with a Brier of zero — `main()` prints the count
    of noul cells left unpriced rather than averaging an empty column into the
    macro. Keys the split does not carry contribute no weight, exactly as in
    `accuracy_cells`.
    """
    pairs: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    for key, val in metrics.items():
        if not key.endswith(":brier"):
            continue
        wf, name = key[:-len(":brier")].split("/", 1)
        pairs[(_source_of(wf), name)].append((val, len(groups[wf][1]) if wf in groups else 0))
    out = {}
    for k, pp in pairs.items():
        tot = sum(n for _v, n in pp)
        # Zero rows is zero evidence, exactly as in `roll_up`: a sidecar whose sets are
        # all unknown to this split leaves the cell unpriced rather than averaging it in.
        if tot:
            out[k] = sum(v * n for v, n in pp) / tot
    return out


def roll_up(stats: dict, acc: dict, keep_min: int = MIN_CELL_ROWS,
            briers: dict | None = None) -> list[dict]:
    """One row per (source, question) cell, floors and model side by side.

    Only cells with `keep_min` rows are kept, and the kept set is *derived from
    the data*, so a table built on a subsample says which cells it dropped.

    `clip` is `max(acc, majority)` and `clip_worth` the part of it the model did not
    earn, `max(0, majority - acc)`. Both are an *oracle* statistic — the floor is
    measured on the split being scored — so they bound what abstaining to a constant
    per-cell label could recover and are not attainable by any model. That is the
    whole reason 9c prints them beside `acc` instead of replacing it (SPEC §5 P9).
    """
    rows = []
    for (source, qname), s in sorted(stats.items()):
        if s["n"] < keep_min:
            continue
        a = acc.get((source, qname))
        acc_v = a["acc"] if a and a["n"] else None
        # A set that is not in the split carries no rows, so a cell whose only scored
        # sets are unknown has no evidence in it. Printing 0.5 there would be a model
        # number derived from zero rows — the note in `main()` is the story instead.
        rows.append({"source": source, "question": qname, "type": s["type"],
                     "n": s["n"], "options": max(s["options"]), "options_by_size": s["options"],
                     "sets": s["sets"], "majority": s["majority"], "uniform": s["uniform"],
                     "acc": acc_v, "n_scored": a["n"] if a else 0,
                     "clip": None if acc_v is None else max(acc_v, s["majority"]),
                     "clip_worth": None if acc_v is None else max(0.0, s["majority"] - acc_v),
                     "brier": (briers or {}).get((source, qname))})
    return rows


def macro(rows: list[dict], field: str) -> float | None:
    vals = [r[field] for r in rows if r[field] is not None]
    return sum(vals) / len(vals) if vals else None


def laya_cells(path) -> tuple[dict, dict]:
    """`bench/eval_laya_real.py`'s JSON -> {(source, qname): {"acc", "n", "rows", "parts"}}.

    Its rows are keyed on `(source, qid, type)`, and one qid can be asked two ways inside
    one sample: `contrastive/decision` is a four-way choice in some records and a noul in
    others — 24 and 16 rows of that source's 40. myna's cell key is `(source, question
    name)`, so those two phrasings are *one* cell here, and the competitor's two rows have
    to be merged the same way `accuracy_cells` merges its per-set rows: by rows. Indexing
    the row list by key instead keeps whichever row came last and publishes a 16-row
    figure as if it were the cell's (§9.32).
    """
    data = json.loads(Path(path).read_text())
    cells: dict[tuple[str, str], dict] = {}
    for r in data["rows"]:
        c = cells.setdefault((r["source"], r["qid"]), {"acc": 0.0, "n": 0, "rows": 0,
                                                       "parts": []})
        c["rows"] += 1
        c["n"] += r["n"]
        c["acc"] += r["acc"] * (r["n"] or 0)
        c["parts"].append({"type": r["type"], "acc": r["acc"], "n": r["n"]})
    for c in cells.values():
        if c["n"]:
            c["acc"] /= c["n"]
        else:
            c["acc"] = sum(p["acc"] for p in c["parts"]) / len(c["parts"])
    return cells, data


def g1_verdict(macro_acc: float | None, macro_majority: float | None, target=0.70,
               margin=0.15) -> dict:
    """G1: test macro ≥ `target` *and* ≥ `margin` over the majority floor (§2.2).

    Both halves are reported separately, because clearing 0.70 on a floor of 0.65
    is not the same result as clearing it on 0.43 — the gate exists to stop the
    first from being sold as the second.
    """
    hit = macro_acc is not None and macro_majority is not None
    return {"target": target, "margin": margin,
            "meets_target": hit and macro_acc >= target,
            "meets_margin": hit and (macro_acc - macro_majority) >= margin,
            "pass": hit and macro_acc >= target and (macro_acc - macro_majority) >= margin,
            "macro_acc": macro_acc, "macro_majority": macro_majority,
            "margin_observed": (macro_acc - macro_majority) if hit else None}


def _fmt(x, nd=3):
    return "  —  " if x is None else f"{x:.{nd}f}"


def _macro_with(rows, values):
    vals = [v for v in (values.get((r["source"], r["question"])) for r in rows) if v is not None]
    return sum(vals) / len(vals) if vals else None


#: The table is one geometry, not three f-strings that happen to agree: the header,
#: every cell row and every macro row come through `_trow`, because a hand-padded
#: macro label overran its field by six characters in the committed artifact and the
#: columns under it were not lined up. Every value is five characters except the
#: delta, which can be `-1.000`, so it alone gets the extra column of air. 21 for the
#: label because the longest real cell name is `contrastive/decision` at 20.
LBL, WID = 21, (6, 6, 6, 6, 6, 6, 7)


def _trow(label, type_="", n="", sets="", vals=()):
    """One table line: `label`, then type/n/sets, then the right-aligned numeric cells."""
    assert len(vals) == len(WID), f"{len(vals)} values for {len(WID)} columns"
    return (f"{label:<{LBL}}{type_:>7}{n:>6}{sets:>5}"
            + "".join(f"{v:>{w}}" for v, w in zip(vals, WID, strict=True))).rstrip()


def table_text(rows, strata_map, laya=None, split="test"):
    """The table as text, grouped by stratum, with the macro lines under it."""
    laya_acc = {k: v["acc"] for k, v in (laya or {}).items()}
    lines = [f"stratified {split} — one row per (source, question) cell, n = rows in it",
             "clip = max(model, maj): what the cell scores answering nothing with this split's "
             "majority label. It is an oracle (the floor is measured on the rows being scored), "
             "so a clip above the model is the clip's worth, not the model's, and 9c prints it "
             "beside the number rather than instead of it. brier = mean (p-g)^2 over a noul "
             "cell's second option, 0.25 being the coin flip; an empty cell means the metrics "
             "file carried no :brier sidecar, which is not a 0."]
    hdr = _trow("source/question", "type", "n", "sets",
                ("model", "clip", "laya", "maj", "unif", "brier", "-maj"))
    for cls in ("shared-instruction", "per-row-instruction"):
        group = [r for r in rows if strata_map.get(r["source"], {}).get("class") == cls]
        if not group:
            continue
        srcs = sorted({r["source"] for r in group})
        lines.append("")
        lines.append(f"{cls} — {len(srcs)} source(s): {', '.join(srcs)}")
        lines.append(hdr)
        for r in group:
            la = laya_acc.get((r["source"], r["question"]))
            d = None if r["acc"] is None else r["acc"] - r["majority"]
            lines.append(_trow(r["source"] + "/" + r["question"], r["type"], str(r["n"]),
                               str(r["sets"]),
                               (_fmt(r["acc"]), _fmt(r["clip"]), _fmt(la), _fmt(r["majority"]),
                                _fmt(r["uniform"]), _fmt(r["brier"]), _fmt(d))))
        scored = [r for r in group if r["acc"] is not None]
        m = macro(scored, "acc")
        margin = None if m is None else m - macro(scored, "majority")
        nb = [r["brier"] for r in scored if r["brier"] is not None]
        lines.append(_trow(f"  macro: {len(scored)} cell(s)", vals=(
            _fmt(m), _fmt(macro(scored, "clip")), _fmt(_macro_with(scored, laya_acc)),
            _fmt(macro(scored, "majority")), _fmt(macro(scored, "uniform")),
            _fmt(sum(nb) / len(nb) if nb else None), _fmt(margin))))
        if nb and len(nb) < len([r for r in scored if r["type"] == "noul"]):
            lines.append(f"  brier macro over {len(nb)} of the group's noul cells; "
                         f"the rest have no :brier sidecar in this metrics file")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="stratified source x question report with floors")
    ap.add_argument("--suite", default="data/decision-v2-pilot",
                    help="directory holding the frozen splits (the one that was scored)")
    ap.add_argument("--split", default="test", choices=["dev", "test", "development", "calibration"])
    ap.add_argument("--metrics", required=True,
                    help="metrics.json from a run, or the run directory holding it")
    ap.add_argument("--laya", default=None, help="laya witness JSON for the same split")
    ap.add_argument("--min-rows", type=int, default=MIN_CELL_ROWS)
    ap.add_argument("--out", default=None, help="write the json report here")
    args = ap.parse_args(argv)
    # §9.30: this report is the witness for every "model vs floor" cell published from
    # it, and which metrics.json and which laya split it joined are part of the claim.
    cmd = shlex.join(["python", "-m", "myna.report", *(argv if argv is not None
                                                      else sys.argv[1:])])
    print("$ " + cmd)
    if not (0 < args.min_rows):
        raise SystemExit(f"--min-rows must be positive, got {args.min_rows}")

    fname = {"dev": "development.jsonl", "test": "test.jsonl", "development": "development.jsonl",
             "calibration": "calibration.jsonl"}[args.split]
    path = Path(args.suite) / fname
    if not path.exists():
        raise SystemExit(f"--suite {args.suite} has no {fname}: the floors and the row weights "
                         f"come from the split that was scored, so it must be the same one")
    metrics_path = Path(args.metrics)
    if metrics_path.is_dir():
        metrics_path = metrics_path / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"--metrics: no {metrics_path}")

    groups = load_split(path)
    stats = cell_stats(groups)
    cls = strata(groups)
    metrics = json.loads(metrics_path.read_text())
    key = "test" if args.split in ("test",) else ("dev" if args.split in ("dev", "development")
                                                  else "calibration")
    if key not in metrics:
        raise SystemExit(f"{metrics_path} has no {key!r} block "
                         f"(it has: {', '.join(k for k in metrics if isinstance(metrics[k], dict))})")
    acc, unmatched = accuracy_cells(metrics[key], groups)
    briers = brier_cells(metrics[key], groups)
    rows = roll_up(stats, acc, args.min_rows, briers)
    laya = None
    laya_merged: dict = {}
    if args.laya:
        laya, ldata = laya_cells(args.laya)
        print(f"laya: {args.laya} ({ldata['split']} split, n={ldata['n']}, "
              f"<= {ldata['n_per_source']}/source) — read its own per-cell accuracies")
        laya_merged = {f"{s}/{q}": c for (s, q), c in laya.items() if c["rows"] > 1}
        for key, c in sorted(laya_merged.items()):
            print(f"  {key} is {c['rows']} rows of that JSON over "
                  f"{c['n']} answers: " + " + ".join(
                      f"{p['type']} {p['acc']:.3f} ({p['n']})" for p in c["parts"])
                  + f" → merged by rows to {c['acc']:.3f} (§9.32)")
    dropped = [f"{s}/{q}" for (s, q) in sorted(stats)
               if stats[(s, q)]["n"] < args.min_rows]
    scored = [r for r in rows if r["acc"] is not None]
    print(f"suite {args.suite} / {fname}: {sum(g['n'] for g in stats.values())} rows over "
          f"{len(stats)} cells, {len(rows)} kept at n>={args.min_rows}"
          + (f", dropped {len(dropped)}: {', '.join(dropped)}" if dropped else ""))
    mixed = [r for r in rows if len(r["options_by_size"]) > 1]
    if mixed:
        print(f"{len(mixed)} cell(s) mix option counts across their sets "
              + "; ".join(f"{r['source']}/{r['question']} {r['options_by_size']}" for r in mixed)
              + " — the uniform floor is row-weighted over them")
    if unmatched:
        print(f"note: {len(unmatched)} scored question-sets are not in {fname} "
              f"(e.g. {unmatched[0]}) — they carry no row weight here")
    if not scored:
        print(f"no cell in {metrics_path.name}[{key!r}] matches a cell in {fname}: this report "
              f"would be floors with no model beside them")
    print(table_text(rows, cls, laya, args.split))
    print("\nstrata, measured on this split (the class is derived, never listed by name):")
    for source in sorted(cls):
        s = cls[source]
        print(f"  {source:<12}{s['class']:>22}  {s['repeated_instruction_share']:.2f} of slots "
              f"repeat an instruction ({s['distinct_instructions']} distinct over "
              f"{s['slots']} slots / {s['rows']} rows, {s['sets']} exact-signature sets, "
              f"{s['multi_row_set_share']:.2f} of rows in a reused set)")
    # The floors are averaged over the *same* cells as the model, so "model vs
    # floor" is one comparison over one set of rows; a checkpoint that never scored
    # a cell cannot shift the floor it is being measured against. With nothing
    # scored at all the floors are still a property of the split, so they print on
    # the kept cells rather than as an em dash.
    basis = scored if scored else rows
    ma, mm, mu = macro(scored, "acc"), macro(basis, "majority"), macro(basis, "uniform")
    v = g1_verdict(ma, mm if scored else None)
    extra = "" if len(scored) == len(rows) else (
        f"  (floors over all {len(rows)} kept cells: majority {_fmt(macro(rows, 'majority'))}"
        f" · uniform {_fmt(macro(rows, 'uniform'))})")
    print(f"\nMACRO over the {len(scored)} scored cells: model {_fmt(ma)} · "
          f"majority floor {_fmt(mm)} · uniform floor {_fmt(mu)}{extra}")
    # 9c's honest sentence: which cells the model loses to a constant predictor, and
    # what clipping them would buy. The clip is an oracle — its floor is measured on
    # the rows being scored — so the delta is arithmetic, and saying so here is the
    # only thing that keeps the column from being read as a result (SPEC §5 P9).
    mc = macro(scored, "clip")
    losers = [r for r in scored if r["clip_worth"] > 0]
    clip_worth = None if (mc is None or ma is None) else mc - ma
    if losers:
        print(f"{len(losers)} of {len(scored)} scored cell(s) answer below their own "
              f"majority floor: "
              + ", ".join(f"{r['source']}/{r['question']} {_fmt(r['acc'])} < "
                          f"{_fmt(r['majority'])} (worth +{_fmt(r['clip_worth'])})"
                          for r in losers))
        print(f"  clipping those to the split's majority label would print {_fmt(mc)} where "
              f"the model prints {_fmt(ma)}: +{_fmt(clip_worth)} of arithmetic and 0 of "
              f"model. The floor is read off the scored rows, so a real fallback would use "
              f"a training-set prior and land under this number — that is a leaderboard "
              f"decision, not a learning improvement, and not what this column claims.")
    elif scored:
        clip_worth = 0.0
        print(f"no scored cell answers below its own majority floor, so the clip column is "
              f"the model column: {_fmt(mc)} = {_fmt(ma)}")
    # 9b's reporting half: an argmax alone over-reports a binary cell either way.
    noul = [r for r in scored if r["type"] == "noul"]
    priced = [r for r in noul if r["brier"] is not None]
    brier_macro = sum(r["brier"] for r in priced) / len(priced) if priced else None
    if noul:
        print(f"noul: {len(priced)} of {len(noul)} scored binary cells carry a :brier sidecar"
              + (f"; their mean Brier is {_fmt(brier_macro)} against 0.250 for a coin flip, "
                 f"which is the number §9.24 asks for beside an accuracy because an argmax "
                 f"alone over-reports a binary cell either way (SPEC §5 P9: 9b)" if priced else
                 "; the column is empty because this metrics file has none, which is missing "
                 "data and not a model that scores 0"))
    # The competitor line is computed over the intersection, never over each side's
    # own best set of cells: two macros over two different cell sets are not a gap,
    # and "laya 0.63 vs myna 0.34" from the two harnesses' own summaries is exactly
    # how that mistake gets published. Row-weighted n and cell-macro are different
    # statistics too, so this line names which one it is.
    both = None
    if laya:
        shared = [r for r in scored if (r["source"], r["question"]) in laya]
        la = {k: v["acc"] for k, v in laya.items()}
        mb, lb = macro(shared, "acc"), _macro_with(shared, la)
        fb = macro(shared, "majority")
        both = {"cells": len(shared), "cells_scored": len(scored),
                "myna": mb, "laya": lb, "majority": fb,
                "gap": None if (mb is None or lb is None) else mb - lb}
        print(f"\nMACRO over the {len(shared)} of {len(scored)} cells BOTH scored "
              f"(per-cell, unweighted — not the row-weighted overall each harness prints):")
        print(f"  myna {_fmt(mb)} · laya {_fmt(lb)} · gap {_fmt(both['gap'])} · "
              f"majority floor {_fmt(fb)} · laya over the floor "
              f"{_fmt(None if lb is None else lb - fb)}")
        if len(shared) != len(scored):
            print(f"  note: {len(scored) - len(shared)} scored cell(s) are absent from the "
                  f"laya JSON, so this line is not the {len(scored)}-cell macro either side")
    print(f"G1: target {v['target']} → {'PASS' if v['meets_target'] else 'not met'}; "
          f"+{v['margin']} over the floor "
          f"({'PASS' if v['meets_margin'] else 'not met'}, observed "
          f"{_fmt(v['margin_observed'])})")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"cmd": cmd, "metrics_file": str(metrics_path),
                                   "laya_file": args.laya, "laya_merged": laya_merged,
                                   "suite": str(args.suite), "split": args.split,
                                   "file": fname, "min_rows": args.min_rows,
                                   "dropped_cells": dropped, "unmatched_sets": unmatched,
                                   "strata": cls, "cells": rows,
                                   "macro": {"acc": ma, "majority": mm, "uniform": mu,
                                             "clip": mc, "clip_worth": clip_worth,
                                             "brier_noul": brier_macro,
                                             "cells_scored": len(scored), "cells_kept": len(rows),
                                             "both": both},
                                   "below_majority_floor": [
                                       {"cell": f"{r['source']}/{r['question']}", "acc": r["acc"],
                                        "majority": r["majority"], "clip": r["clip"],
                                        "clip_worth": r["clip_worth"]} for r in losers],
                                   "brier": {"noul_cells": len(noul), "priced_cells": len(priced),
                                             "macro": brier_macro,
                                             "cells": {f"{r['source']}/{r['question']}":
                                                       r["brier"] for r in priced}},
                                   "g1": v}, indent=2) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Row-clustered paired bootstrap and McNemar cross-check for every headline accuracy.

    uv run python bench/significance.py --B 20000 --seed 20261031

The gates in this repository are decided by single numbers: G1 reads 0.4893 against a
majority floor of 0.4331 and a competitor's 0.6668, and SPEC §2.1 prints all three to four
decimals as if they were measurements of the architecture. They are point estimates from
1,176 rows, and a paper cannot say "above the floor" without an interval around the
difference. `runs/v1b_kaggle_3600b.report.json` could not supply one: its unit is a mean,
and a mean has no resample left in it. `bench/row_outcomes.py` and
`bench/eval_laya_row_outcomes.py` produce the per-question outcomes this needs, and both
sides of every number here are the same rows.

The two design calls are the substance, so they are stated where a reader will trip over
them:

1. **Rows are resampled, questions are not.** The 1,440 scored question instances come from
   1,176 rows — 92 rows ask three questions and 80 ask two, and every question in a row
   shares one state, one source and one `_meta.id`. Drawing question instances independently
   would treat five `agnews/*` cells as five samples of agnews when they are one 116-row
   sample read five ways, and it would understate the variance of anything that correlates
   inside a row. So the cluster is `_meta.id` and each replicate resamples rows, carrying
   their questions with them. (`_meta.row` collides on 316 rows and `_meta.row_sha256` on
   219 of these 1,176, which is why the cluster key is `id` and the dump prints all three
   distinct-counts beside each other.)
2. **The floors are recomputed inside every replicate.** The majority floor is the accuracy
   of a constant predictor *on the labels of the scored rows*, so it is a statistic of the
   same sample as the accuracy and it moves when the sample moves. Freezing it at the
   published 0.4331 while resampling the accuracy would put an interval on the wrong
   quantity — the gap would inherit none of the floor's own sampling error, which matters
   most on `banking77/intent`, where the floor is 0.043 from 116 rows over 77 labels.

Also deliberate, and also visible below:

* **The cell set is fixed to the published 16**, at the report's own `min_rows`, and a cell
  that a replicate leaves with zero rows is dropped from that replicate's macro and counted.
  With the smallest cell at 43 question instances over 1,176 rows the probability of that is
  on the order of e-19, so the count prints as a diagnostic, never as a caveat.
* **Paired means both engines answered the same `(row, question)`**, so a row laya could not
  answer is removed from *both* sides, and the point estimate on the intersection is then
  compared against the committed macro. If they differ, that difference is the price of the
  pairing and it is printed, not absorbed.
* **The gap is a gap between two macros over the same cells of the same rows**, computed per
  replicate, so its interval is the interval of the difference — not the two intervals
  subtracted, which is the mistake that produces "the CIs overlap, so there is no difference".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

#: The headline statistics. Each is a function of the per-replicate cell sums, so the point
#: estimate and its interval come from one implementation — a CI around a statistic the
#: bootstrap does not actually compute is the failure mode this whole file exists to avoid.
STATS = ("myna_macro", "laya_macro", "majority_floor", "uniform_floor", "margin_over_floor",
         "gap_myna_minus_laya")


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p on the discordant counts.

    The binomial tail is built from its largest term downward, because P(X=0) = 2^-n
    underflows float64 at n > 1074 and this test runs on up to 1,440 pairs.
    """
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    log_pk = (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
              - n * math.log(2.0))
    total, term = 1.0, 1.0
    for i in range(k, 0, -1):
        term *= i / (n - i + 1)
        total += term
    return min(1.0, 2.0 * math.exp(log_pk) * total)


def pack(myna, laya, cells, min_rows):
    """The per-row sums a replicate re-weights, and the paired outcomes McNemar reads.

    Shapes: counts and hits are (rows, cells); `u` carries Σ 1/K per cell so the uniform
    floor is row-weighted the way `myna.report.cell_stats` weights it; `lab` is
    (rows, cells, labels) over the *myna* gold option index, which is the same histogram
    `cell_stats` builds — including inside `contrastive/decision`, where one cell mixes a
    noul's 0/1 with a four-way choice's indices and the published floor of 0.4541 is
    computed exactly that way.
    """
    keys = [(c["source"], c["question"]) for c in cells if c["n"] >= min_rows]
    cix = {k: i for i, k in enumerate(keys)}
    by_myna = {r["meta"]["id"]: r for r in myna["rows"]}
    by_laya = {r["meta"]["id"]: r for r in laya["rows"]}
    ids = sorted(set(by_myna) & set(by_laya))
    dropped = sorted(set(by_myna) - set(by_laya))

    labels = defaultdict(set)
    for r in by_myna.values():
        for name, q in r["questions"].items():
            if (r["meta"]["source"], name) in cix:
                labels[(r["meta"]["source"], name)].add(q["gold"])
    lab_ix = {k: {v: i for i, v in enumerate(sorted(s))} for k, s in labels.items()}
    n_labels = max((len(s) for s in labels.values()), default=1)

    N, C = len(ids), len(keys)
    # float64 for the sums, so the point estimate below is the published number and not a
    # float32 shadow of it: 0.4893271976084137 has to come out of this file, not
    # 0.4893271923, or the interval is around a lookalike. The replicates drop to float32 in
    # `boot()`, where seven digits is more than a four-decimal CI asks for.
    nm = np.zeros((N, C), np.float64)
    hm = np.zeros((N, C), np.float64)
    nl = np.zeros((N, C), np.float64)
    hl = np.zeros((N, C), np.float64)
    um = np.zeros((N, C), np.float64)
    lab = np.zeros((N, C, n_labels), np.float64)
    pairs = []  # (cell_ix, myna_ok, laya_ok, gold) per shared question instance
    laya_only = 0
    for i, rid in enumerate(ids):
        mr = by_myna[rid]["questions"]
        lr = by_laya[rid]["questions"]
        source = by_myna[rid]["meta"]["source"]
        for name, q in mr.items():
            key = (source, name)
            j = cix.get(key)
            if j is None:
                continue
            nm[i, j] += 1
            hm[i, j] += q["ok"]
            um[i, j] += 1.0 / q["k"]
            lab[i, j, lab_ix[key][q["gold"]]] += 1
            a = lr.get(name)
            if a is None:
                laya_only += 1
                continue
            nl[i, j] += 1
            hl[i, j] += a["ok"]
            pairs.append((j, q["ok"], a["ok"], q["gold"]))
    return {"ids": ids, "keys": keys, "cell_ix": cix, "n_labels": n_labels,
            "nm": nm, "hm": hm, "nl": nl, "hl": hl, "um": um, "lab": lab,
            "pairs": pairs, "dropped_rows": dropped, "dropped_questions": laya_only}


def statize(nm, hm, nl, hl, um, lab):
    """Every headline statistic from one set of per-cell sums. Broadcasting handles (1, C)
    point estimates and (B, C) replicates with the same code."""
    with np.errstate(invalid="ignore", divide="ignore"):
        acc_m = np.where(nm > 0, hm / np.maximum(nm, 1), np.nan)
        acc_l = np.where(nl > 0, hl / np.maximum(nl, 1), np.nan)
        floor = np.where(nm > 0, lab.max(axis=-1) / np.maximum(nm, 1), np.nan)
        unif = np.where(nm > 0, um / np.maximum(nm, 1), np.nan)
        keep = nm > 0
        both = keep & (nl > 0)
        out = {
            "myna_macro": _nanmean(np.where(keep, acc_m, np.nan)),
            "laya_macro": _nanmean(np.where(both, acc_l, np.nan)),
            "majority_floor": _nanmean(np.where(keep, floor, np.nan)),
            "uniform_floor": _nanmean(np.where(keep, unif, np.nan)),
            "gap_myna_minus_laya": _nanmean(np.where(both, acc_m - acc_l, np.nan)),
        }
        out["margin_over_floor"] = out["myna_macro"] - out["majority_floor"]
        out["_cells_kept"] = keep.astype(np.int32).sum(axis=-1) if keep.ndim > 1 else int(keep.sum())
        out["_acc_m"], out["_acc_l"], out["_floor"] = acc_m, acc_l, floor
        return out


def _nanmean(a):
    return np.nanmean(a, axis=-1)


def boot(packed, B, seed, block=1000):
    """Resample rows with replacement; carry each row's whole question set with it."""
    N, C = len(packed["ids"]), len(packed["keys"])
    rng = np.random.default_rng(seed)
    m = {k: packed[k].astype(np.float32) for k in ("nm", "hm", "nl", "hl", "um")}
    lab2d = packed["lab"].reshape(N, C * packed["n_labels"]).astype(np.float32)
    acc_m = np.empty((B, C), np.float32)
    acc_l = np.empty((B, C), np.float32)
    scal = {k: np.empty(B, np.float32) for k in STATS}
    kept = np.empty(B, np.int32)
    drawn = []
    for lo in range(0, B, block):
        hi = min(lo + block, B)
        w = rng.multinomial(N, [1.0 / N] * N, size=hi - lo).astype(np.float32)
        if lo == 0:
            drawn.append(hashlib.sha256(w.tobytes()).hexdigest()[:16])
        s = statize(w @ m["nm"], w @ m["hm"], w @ m["nl"], w @ m["hl"],
                    w @ m["um"], (w @ lab2d).reshape(hi - lo, C, packed["n_labels"]))
        acc_m[lo:hi] = s["_acc_m"]
        acc_l[lo:hi] = s["_acc_l"]
        kept[lo:hi] = s["_cells_kept"]
        for k in STATS:
            scal[k][lo:hi] = s[k]
    return scal, acc_m, acc_l, kept, drawn[0] if drawn else "no replicates"


def ci(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return [float(lo), float(hi)]


def claims(point, ci_map, report):
    """The pre-registered rule: an interval that covers the threshold rewrites the sentence."""
    g1 = report["g1"]
    out = []

    def add(name, claim, stat, rule, holds):
        out.append({"id": name, "claim": claim, "statistic": stat, "point": point[stat],
                    "ci95": ci_map[stat], "threshold": rule,
                    "survives": bool(holds),
                    "consequence": "the sentence as written stands" if holds
                                   else "the sentence must be rewritten in the same PR"})

    m = ci_map["margin_over_floor"]
    add("floor", f"myna's {report['split']}-split macro is above its own majority floor",
        "margin_over_floor", "CI excludes 0", m[0] > 0)
    add("g1-margin", f"the margin over the floor reaches G1's +{g1['margin']}",
        "margin_over_floor", f"CI lower bound ≥ {g1['margin']}", m[0] >= g1["margin"])
    a = ci_map["myna_macro"]
    add("g1-target", f"the macro reaches G1's target of {g1['target']}",
        "myna_macro", f"CI lower bound ≥ {g1['target']}", a[0] >= g1["target"])
    gap = ci_map["gap_myna_minus_laya"]
    add("competitor", "laya's macro is above myna's on the same rows and cells",
        "gap_myna_minus_laya", "CI excludes 0 on the negative side", gap[1] < 0)
    return out


def _p(x):
    """A p-value, or the reason there is not one."""
    return "n/a (no discordant pairs)" if x is None else f"{x:.3g}"


def md_text(res):
    L = [f"# Paired significance on the `{res['split']}` split of `{res['suite']}`",
         "",
         f"Measured by `bench/significance.py` over {res['bootstrap']['n_rows']} rows / "
         f"{res['paired_question_instances']} paired question instances, "
         f"{res['bootstrap']['B']:,} row-clustered replicates, seed {res['bootstrap']['seed']} "
         f"(first replicate digest `{res['bootstrap']['first_replicate_digest']}`). "
         f"myna is `{res['myna']['engine']}` at model "
         f"`{res['myna']['model_sha256'][:12]}`, temperature {res['myna']['temperature']}; "
         f"laya is `{res['laya']['version']}` at commit "
         f"`{res['laya']['commit'][:12]}`. Both engines are read row for row, and the unit "
         f"resampled is `_meta.id`, not the question.",
         "",
         f"Point estimates reproduce the committed report exactly: macro "
         f"{res['point']['myna_macro']!r} against "
         f"`{res['report_macro']}` in `{res['report']}`. Nothing here re-measures inference; "
         f"it resamples outcomes that `bench/row_outcomes.py` and "
         f"`bench/eval_laya_row_outcomes.py` dumped and guarded against that report.",
         "",
         "## The headline numbers with their intervals",
         "",
         "| statistic | point | 95% CI | what it is |",
         "|---|---|---|---|"]
    what = {"myna_macro": "cell-macro of myna over the published 16 cells — G1's number",
            "majority_floor": "the same macro for a per-cell constant predictor, recomputed "
                              "in every replicate",
            "uniform_floor": "the same macro at 1/options, row-weighted",
            "margin_over_floor": "the difference G1's second half is judged on",
            "laya_macro": "laya over the same cells of the same rows",
            "gap_myna_minus_laya": "the paired difference, not two intervals subtracted"}
    for k in ("myna_macro", "majority_floor", "uniform_floor", "margin_over_floor",
              "laya_macro", "gap_myna_minus_laya"):
        p = res["point"][k]
        lo, hi = res["ci95"][k]
        L.append(f"| `{k}` | {p:.4f} | [{lo:.4f}, {hi:.4f}] | {what[k]} |")
    L += ["", "## Claims, adjudicated", "",
          "The rule was fixed before the intervals existed: an interval that covers a "
          "threshold forces the sentence to be rewritten in the same change, and a verdict "
          "that moves a gate is escalated rather than edited here.", ""]
    for c in res["claims"]:
        lo, hi = c["ci95"]
        L.append(f"- **{'stands' if c['survives'] else 'does not stand'}** — {c['claim']}. "
                 f"`{c['statistic']}` = {c['point']:.4f} [{lo:.4f}, {hi:.4f}], rule: "
                 f"{c['threshold']}. {c['consequence']}.")
    L += ["", "## McNemar cross-check", "",
          "The bootstrap says how far apart the engines are; McNemar says whether the "
          "direction is decided by the rows at all. Both cells of a pair are read on the "
          "same `(row, question)`, so the only information is discordance.", ""]
    mc = res["mcnemar"]
    L.append(f"- myna vs laya, pooled over {mc['vs_laya']['n']} pairs: myna right and laya "
             f"wrong {mc['vs_laya']['b']}, the reverse {mc['vs_laya']['c']}, exact "
             f"two-sided p = {_p(mc['vs_laya']['p'])}.")
    L.append(f"- myna vs each cell's majority label, pooled over {mc['vs_floor']['n']} pairs: "
             f"{mc['vs_floor']['b']} in myna's favour, {mc['vs_floor']['c']} against, p = "
             f"{_p(mc['vs_floor']['p'])}. This is the floor claim tested without a resample.")
    L += ["", "| cell | n pairs | myna | laya | Δ | myna-only | laya-only | McNemar p | "
          "95% CI on the cell Δ |", "|---|---|---|---|---|---|---|---|---|"]
    for c in res["per_cell"]:
        lo, hi = c["gap_ci95"]
        L.append(f"| {c['cell']} | {c['n']} | {c['myna']:.4f} | {c['laya']:.4f} | "
                 f"{c['gap']:+.4f} | {c['mcnemar']['b']} | {c['mcnemar']['c']} | "
                 f"{_p(c['mcnemar']['p'])} | [{lo:.4f}, {hi:.4f}] |")
    L += ["", "## What is not in this table", "",
          f"- `{res['bootstrap']['n_rows']}` rows are the whole `{res['split']}` split, so "
          f"the interval is the sampling error of *this* split only. The rows are a fixed "
          f"draw from eleven public datasets; nothing here generalises to their upstreams.",
          f"- Cells under `min_rows = {res['min_rows']}` were dropped before resampling, as "
          f"in the published report, so the macro is over {res['n_cells']} cells and not "
          f"over every question the split asks.",
          "- laya's confidence values are not usable here: its own loader warns that this "
          "checkpoint ships temperatures outside its [0.5, 5] clamp, which is recorded in "
          "`laya_warnings` in the JSON. Only correctness is compared.",
          "- The interval is a percentile interval from a multinomial row resample. It is "
          "not a statement about a population of checkpoints: a second training run of the "
          "same 3,600-update dose is a different sampling layer and would need its own "
          "checkpoint, which this box has not produced.",
          ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--myna", default="runs/myna_v1b_test.rowoutcomes.json")
    ap.add_argument("--laya", default="runs/laya_0419_test.rowoutcomes.json")
    ap.add_argument("--report", default="runs/v1b_kaggle_3600b.report.json")
    ap.add_argument("--B", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--min-rows", type=int, default=None,
                    help="default: the myna dump's own, which is the report's")
    ap.add_argument("--allow-subset", action="store_true",
                    help="accept a paired row set that is smaller than the published split")
    ap.add_argument("--out", default="runs/significance.json")
    ap.add_argument("--md", default="runs/significance.md")
    args = ap.parse_args(argv)
    given = list(sys.argv[1:]) if argv is None else list(argv)
    cmd = " ".join(["uv run python bench/significance.py"]
                   + [json.dumps(a) if " " in a else a for a in given])
    print("$ " + cmd, flush=True)

    for p in (args.myna, args.laya, args.report):
        if not Path(p).exists():
            print(f"{p} is not on disk — this harness resamples committed outcomes and writes "
                  f"nothing of its own", file=sys.stderr)
            return 2
    myna = json.loads(Path(args.myna).read_text())
    laya = json.loads(Path(args.laya).read_text())
    report = json.loads(Path(args.report).read_text())
    min_rows = args.min_rows or myna["min_rows"]

    packed = pack(myna, laya, report["cells"], min_rows)
    if not packed["ids"]:
        print("the two dumps share no row id: nothing is paired and no interval means anything",
              file=sys.stderr)
        return 3
    C = len(packed["keys"])
    N = len(packed["ids"])
    if packed["dropped_rows"]:
        print(f"{len(packed['dropped_rows'])} rows are in the myna dump but not laya's "
              f"(first {packed['dropped_rows'][:3]}); they leave the paired set")
    if not args.allow_subset and N != myna["n_rows"]:
        print(f"GUARD FAIL: the paired set is {N} rows and the published macro is over "
              f"{myna['n_rows']}. The point estimate below is not G1's number. Pass "
              f"--allow-subset to say so out loud and continue.", file=sys.stderr)
        return 3

    def one(a):
        return a.sum(0)
    point = statize(one(packed["nm"]), one(packed["hm"]), one(packed["nl"]),
                    one(packed["hl"]), one(packed["um"]), one(packed["lab"]))
    point = {k: (v if k.startswith("_") else float(v)) for k, v in point.items()}
    agrees = [("myna_macro", report["macro"]["acc"]),
              ("majority_floor", report["macro"]["majority"]),
              ("uniform_floor", report["macro"]["uniform"]),
              ("margin_over_floor", report["g1"]["margin_observed"])]
    off = [(k, point[k], want) for k, want in agrees
           if want is not None and abs(point[k] - want) > 1e-9]
    if off:
        for k, got, want in off:
            print(f"GUARD FAIL: {k} from these rows is {got!r} and the committed report "
                  f"prints {want!r}. The interval below would be around a different "
                  f"statistic than the one published.", file=sys.stderr)
        return 3
    print(f"point: myna {point['myna_macro']:.4f} · floor {point['majority_floor']:.4f} · "
          f"laya {point['laya_macro']:.4f} · gap {point['gap_myna_minus_laya']:+.4f} · "
          f"macro, both floors and G1's margin match {Path(args.report).name} to 1e-9",
          flush=True)

    t0 = time.time()
    scal, acc_m, acc_l, kept_cells, digest = boot(packed, args.B, args.seed)
    ci_map = {k: ci(scal[k]) for k in STATS}
    print(f"{args.B:,} replicates over {N} rows × {C} cells in {time.time() - t0:.1f}s · "
          f"cells kept per replicate min {int(kept_cells.min())} max {int(kept_cells.max())}",
          flush=True)

    # McNemar on the paired instances: myna vs laya, and myna vs the cell's own majority
    # label — the floor claim without a resample.
    maj = {}
    for j, key in enumerate(packed["keys"]):
        col = packed["lab"][:, j, :]
        tot = col.sum(0)
        maj[j] = int(np.argmax(tot))
        # tie-break note: np.argmax takes the lowest index, cell_stats takes max() of the
        # histogram values, which is the same number here; the label index is not compared.
    per = defaultdict(lambda: [0, 0, 0, 0, 0, 0])  # b, c, nmyna, nlaya, mh, lh
    pooled_myna = [0, 0]
    pooled_floor = [0, 0]
    for j, ok_m, ok_l, gold in packed["pairs"]:
        d = per[j]
        d[0] += int(ok_m and not ok_l)
        d[1] += int(ok_l and not ok_m)
        d[2] += 1
        d[4] += ok_m
        d[5] += ok_l
        pooled_myna[0] += int(ok_m and not ok_l)
        pooled_myna[1] += int(ok_l and not ok_m)
        floor_ok = int(gold == maj[j])
        pooled_floor[0] += int(ok_m and not floor_ok)
        pooled_floor[1] += int(floor_ok and not ok_m)

    per_cell = []
    for j, key in enumerate(packed["keys"]):
        b, c, n, _, mh, lh = per[j]
        gap = acc_m[:, j] - acc_l[:, j]
        per_cell.append({"cell": f"{key[0]}/{key[1]}", "n": n, "myna": mh / n, "laya": lh / n,
                         "gap": (mh - lh) / n,
                         "myna_ci95": [float(x) for x in ci(acc_m[:, j])],
                         "gap_ci95": [float(x) for x in ci(gap[np.isfinite(gap)])],
                         "mcnemar": {"b": b, "c": c, "n": b + c, "p": mcnemar_exact(b, c)}})
    res = {"cmd": cmd, "suite": myna["suite"], "split": myna["split"],
            "report": str(args.report), "report_macro": report["macro"]["acc"],
            "min_rows": min_rows, "n_cells": C,
            "myna": {"engine": myna["engine"], "dump": str(args.myna),
                     "model_sha256": myna["model"]["model_sha256"],
                     "temperature": myna["model"]["temperature"],
                     "run_dir": myna["run_dir"]},
            "laya": {"engine": laya["engine"], "dump": str(args.laya),
                     "version": laya["laya"]["version"], "commit": laya["laya"]["commit"],
                     "path": laya["laya"]["path"]},
            "laya_warnings": laya["laya_warnings"],
            "bootstrap": {"B": args.B, "seed": args.seed, "n_rows": N,
                          "cluster_key": "meta.id",
                          "resampled": "rows, each carrying all of its questions",
                          "floor": "recomputed from the resampled labels each replicate",
                          "first_replicate_digest": digest,
                          "cells_kept_min": int(kept_cells.min()),
                          "cells_kept_max": int(kept_cells.max())},
            "paired_question_instances": len(packed["pairs"]),
            "unpaired_myna_questions": packed["dropped_questions"],
            "dropped_rows": packed["dropped_rows"],
            "point": {k: float(v) for k, v in point.items() if not k.startswith("_")},
            "ci95": ci_map,
            "claims": claims({k: float(v) for k, v in point.items() if not k.startswith("_")},
                             ci_map, report),
            "mcnemar": {"vs_laya": {"b": pooled_myna[0], "c": pooled_myna[1],
                                    "n": sum(pooled_myna), "p": mcnemar_exact(*pooled_myna)},
                        "vs_floor": {"b": pooled_floor[0], "c": pooled_floor[1],
                                     "n": sum(pooled_floor),
                                     "p": mcnemar_exact(*pooled_floor)}},
            "per_cell": per_cell}
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    Path(args.md).write_text(md_text(res) + "\n")
    print(f"wrote {args.out} and {args.md}")
    for c in res["claims"]:
        lo, hi = c["ci95"]
        print(f"  {'STANDS   ' if c['survives'] else 'REWRITTEN'} {c['claim']} · "
              f"{c['statistic']} {c['point']:.4f} [{lo:.4f}, {hi:.4f}] vs {c['threshold']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

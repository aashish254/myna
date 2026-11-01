#!/usr/bin/env python
"""Run the paired instrument against a second external engine — vegaml as a control.

    uv run python bench/vegaml_control.py --control runs/vegaml_0150_test.rowoutcomes.json

`bench/significance.py` is the frozen instrument: it produced the intervals G1's prose now
carries, its four myna-side statistics are refused unless they match the committed report to
1e-9, and it is named that way throughout because the second engine it was written against is
laya. Rather than fork it — two bootstraps that can drift apart is exactly the defect this
repository keeps finding in its own prose — this file **imports its arithmetic** and feeds it
a different dump: `pack`, `statize`, `boot`, `ci`, `mcnemar_exact` and `_p` are the same six
functions that decided `[0.0192, 0.0676]`.

The consequence is a naming fact worth stating plainly. `statize` returns the second engine's
columns as `laya_macro` and `gap_myna_minus_laya` (`bench/significance.py:67 STATS`), and those
two keys are the published competitor row. Here they are read as *the second engine's slot* and
written out as `vegaml_macro` and `gap_myna_minus_vegaml`. Nothing in `significance.py` is
edited, so `runs/significance.json`, `tests/test_significance.py`'s 37 tests and
`bench/mutation_significance.py`'s 39 mutants keep their meaning, and this file's own mapping
is what `bench/mutation_vegaml_control.py` attacks.

What is not re-derived: the myna-side guard. The four statistics the report owns — macro, both
floors and G1's margin — are checked to 1e-9 here for the same reason they are checked there,
and a control pass that changes one of them has broken the myna dump, not the control. The
first replicate's digest is printed because the resample depends only on the row count and the
seed: with the same 1,176 rows and seed 20261031 it is the *same draw* the laya run used, which
is a cross-check that the two arms are intervals over one resample rather than two different
Monte Carlo problems.

The claim this harness can actually adjudicate about the control is one sentence — myna and the
control differ, or they do not — and the abstention rate is reported beside it because the
control can decline to answer and neither myna nor laya can.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "bench"))

from significance import (_p, boot, ci, mcnemar_exact, pack, statize)  # noqa: E402

#: `statize`'s second-engine columns to the names this control is reported under.
RELABEL = {"laya_macro": "vegaml_macro", "gap_myna_minus_laya": "gap_myna_minus_vegaml"}
#: The statistics the committed report owns, checked to 1e-9 exactly as significance.py checks
#: them. A control run cannot move these and is not allowed to try.
MINE = ("myna_macro", "majority_floor", "uniform_floor", "margin_over_floor")


def relabel(d):
    out = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        out[RELABEL.get(k, k)] = float(v)
    return out


def unbound_tally(control):
    """Count the control's second decline flag out of its own rows.

    `abstain` is a confidence test and `unbound` is a distance-from-the-calibration-manifold test
    (`_api.py`: `bool(float(out["nearest"][r]) > cal["unbound"])`). Both are recorded by the dump
    and neither is applied, so the forced-choice macro needs both rates printed beside it rather
    than a reader recounting 1,440 rows.
    """
    rows = control.get("rows") or []
    qs = [q for r in rows for q in r.get("questions", {}).values()]
    un = [q for q in qs if q.get("unbound")]
    return {"n": len(un), "of": len(qs),
            "also_abstained": sum(1 for q in un if q.get("abstain")),
            "rule": "recorded, not applied; `unbound` is a distance test, `abstain` a "
                    "confidence test"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--myna", default="runs/myna_v1b_test.rowoutcomes.json")
    ap.add_argument("--control", default="runs/vegaml_0150_test.rowoutcomes.json")
    ap.add_argument("--report", default="runs/v1b_kaggle_3600b.report.json")
    ap.add_argument("--B", type=int, default=20000,
                    help="the published run's replicate count, so the two arms are the same "
                         "Monte Carlo problem and not a sharper one")
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--min-rows", type=int, default=None)
    ap.add_argument("--allow-subset", action="store_true")
    ap.add_argument("--out", default="runs/vegaml_control.json")
    ap.add_argument("--md", default="runs/vegaml_control.md")
    args = ap.parse_args(argv)
    given = list(sys.argv[1:]) if argv is None else list(argv)
    cmd = " ".join(["uv run python bench/vegaml_control.py"]
                   + [json.dumps(a) if " " in a else a for a in given])
    print("$ " + cmd, flush=True)

    for p in (args.myna, args.control, args.report):
        if not Path(p).exists():
            print(f"{p} is not on disk — this harness resamples committed outcomes and writes "
                  f"nothing of its own", file=sys.stderr)
            return 2
    myna = json.loads(Path(args.myna).read_text())
    control = json.loads(Path(args.control).read_text())
    report = json.loads(Path(args.report).read_text())
    if control.get("engine") not in ("vegaml",):
        print(f"GUARD FAIL: {args.control} reports engine "
              f"{control.get('engine')!r}; this harness relabels the second-engine slot for "
              f"vegaml specifically, and a silent relabel of a different engine would put "
              f"laya's name on someone else's numbers", file=sys.stderr)
        return 3
    min_rows = args.min_rows or myna["min_rows"]

    #: `significance.pack`'s second argument is the competitor slot; the control goes there.
    packed = pack(myna, control, report["cells"], min_rows)
    if not packed["ids"]:
        print("the two dumps share no row id: nothing is paired and no interval means anything",
              file=sys.stderr)
        return 3
    N, C = len(packed["ids"]), len(packed["keys"])
    if packed["dropped_rows"]:
        print(f"{len(packed['dropped_rows'])} rows are in the myna dump but not the control's "
              f"(first {packed['dropped_rows'][:3]}); they leave the paired set", flush=True)
    if not args.allow_subset and N != myna["n_rows"]:
        print(f"GUARD FAIL: the paired set is {N} rows and the published macro is over "
              f"{myna['n_rows']}. The point estimate below is not G1's number. Pass "
              f"--allow-subset to say so out loud and continue.", file=sys.stderr)
        return 3

    point = statize(packed["nm"].sum(0), packed["hm"].sum(0), packed["nl"].sum(0),
                    packed["hl"].sum(0), packed["um"].sum(0), packed["lab"].sum(0))
    off = [(k, float(point[k]), report["macro"][w]) for k, w in
           (("myna_macro", "acc"), ("majority_floor", "majority"), ("uniform_floor", "uniform"))
           if abs(float(point[k]) - report["macro"][w]) > 1e-9]
    if report["g1"]["margin_observed"] is not None and abs(
            float(point["margin_over_floor"]) - report["g1"]["margin_observed"]) > 1e-9:
        off.append(("margin_over_floor", float(point["margin_over_floor"]),
                    report["g1"]["margin_observed"]))
    if off:
        for k, got, want in off:
            print(f"GUARD FAIL: {k} from these rows is {got!r} and the committed report prints "
                  f"{want!r}. A control pass cannot move myna's own numbers; if it did, the "
                  f"myna dump changed under the instrument.", file=sys.stderr)
        return 3

    t0 = time.time()
    scal, acc_m, acc_l, kept_cells, digest = boot(packed, args.B, args.seed)
    boot_s = time.time() - t0
    ci_map = {RELABEL.get(k, k): ci(scal[k]) for k in scal}
    point_r = relabel(point)
    print(f"point: myna {point_r['myna_macro']:.4f} · floor "
          f"{point_r['majority_floor']:.4f} · control {point_r['vegaml_macro']:.4f} · gap "
          f"{point_r['gap_myna_minus_vegaml']:+.4f} · {args.B:,} replicates in "
          f"{boot_s:.1f}s", flush=True)

    per = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    pooled = [0, 0]
    for j, ok_m, ok_c, gold in packed["pairs"]:
        d = per[j]
        d[0] += int(ok_m and not ok_c)
        d[1] += int(ok_c and not ok_m)
        d[2] += 1
        d[4] += ok_m
        d[5] += ok_c
        pooled[0] += int(ok_m and not ok_c)
        pooled[1] += int(ok_c and not ok_m)

    per_cell = []
    for j, key in enumerate(packed["keys"]):
        b, c, n, _, mh, lh = per[j]
        gap = acc_m[:, j] - acc_l[:, j]
        per_cell.append({"cell": f"{key[0]}/{key[1]}", "n": n, "myna": mh / n,
                         "vegaml": lh / n, "gap": (mh - lh) / n,
                         "myna_ci95": [float(x) for x in ci(acc_m[:, j])],
                         "gap_ci95": [float(x) for x in ci(gap[np.isfinite(gap)])],
                         "mcnemar": {"b": b, "c": c, "n": b + c, "p": mcnemar_exact(b, c)}})

    gap = ci_map["gap_myna_minus_vegaml"]
    claims = [
        {"id": "myna-vs-control", "claim": "myna and vegaml differ on the shared rows",
         "statistic": "gap_myna_minus_vegaml", "point": point_r["gap_myna_minus_vegaml"],
         "ci95": gap, "threshold": "CI excludes 0", "survives": gap[0] > 0 or gap[1] < 0},
        {"id": "control-above-myna", "claim": "vegaml's macro is above myna's on the same rows",
         "statistic": "gap_myna_minus_vegaml", "point": point_r["gap_myna_minus_vegaml"],
         "ci95": gap, "threshold": "CI excludes 0 on the negative side", "survives": gap[1] < 0},
    ]
    res = {"cmd": cmd, "suite": myna["suite"], "split": myna["split"],
           "report": str(args.report), "report_macro": report["macro"]["acc"],
           "min_rows": min_rows, "n_cells": C,
           "myna": {"engine": myna["engine"], "dump": str(args.myna),
                    "model_sha256": myna["model"]["model_sha256"],
                    "temperature": myna["model"]["temperature"]},
           "control": {"engine": control["engine"], "dump": str(args.control),
                       "version": control["vegaml"]["version"],
                       "package_sha256": control["vegaml"]["init_sha256"],
                       "reader": control["vegaml"]["reader"],
                       "checkpoint_dir": control["vegaml"]["checkpoint_dir"],
                       "device": control["vegaml"]["device"],
                       "mode": control["vegaml"]["mode"],
                       "views": control["vegaml"]["views"],
                       "torch": control["vegaml"]["torch"],
                       "transformers": control["vegaml"]["transformers"]},
           "relabel": RELABEL,
           "guard": {"stats_checked_to_1e-9": list(MINE), "passed": True},
           "bootstrap": {"B": args.B, "seed": args.seed, "n_rows": N,
                         "first_replicate_digest": digest,
                         "cells_kept_min": int(kept_cells.min()),
                         "cells_kept_max": int(kept_cells.max())},
           "paired_question_instances": len(packed["pairs"]),
           "dropped": {"rows": packed["dropped_rows"],
                       "questions": packed["dropped_questions"]},
           "point": point_r, "ci95": ci_map, "claims": claims, "per_cell": per_cell,
           "mcnemar": {"vs_vegaml": {"b": pooled[0], "c": pooled[1], "n": sum(pooled),
                                     "p": mcnemar_exact(*pooled)}},
           "control_health": {"abstain": control.get("abstain"),
                              "unbound": unbound_tally(control),
                              "argmax_disagreements": control.get("argmax_disagreements"),
                              "errors": len(control.get("errors") or []),
                              "reader_tokens": control.get("reader_tokens"),
                              "inference_seconds": control.get("seconds"),
                              "warnings": control.get("warnings")}}
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    Path(args.md).write_text(md_text(res) + "\n")
    for cl in claims:
        print(f"  {'SURVIVES' if cl['survives'] else 'REWRITTEN'}  {cl['id']}: {cl['claim']} "
              f"[{cl['ci95'][0]:.4f}, {cl['ci95'][1]:.4f}]", flush=True)
    print(f"wrote {args.out} and {args.md}", flush=True)
    return 0


def md_text(res):
    p, c = res["point"], res["ci95"]
    L = [f"# Vegaml as a control on the `{res['split']}` split of `{res['suite']}`",
         "",
         f"Measured by `bench/vegaml_control.py`, which imports `pack`, `statize`, `boot`, `ci`, "
         f"`mcnemar_exact` and `_p` from `bench/significance.py` rather than restating them. "
         f"{res['bootstrap']['n_rows']} rows / "
         f"{res['paired_question_instances']} paired question instances, "
         f"{res['bootstrap']['B']:,} row-clustered replicates, seed "
         f"{res['bootstrap']['seed']}, first replicate digest "
         f"`{res['bootstrap']['first_replicate_digest']}` — the resample depends only on the "
         f"row count and the seed, so this is the same draw that produced "
         f"`runs/significance.json`'s intervals.",
         "",
         f"The control is `vegaml` {res['control']['version']} reading "
         f"`{res['control']['reader']}` on `{res['control']['device']}`, "
         f"mode `{res['control']['mode']}`, views {res['control']['views']}, torch "
         f"{res['control']['torch']} / transformers {res['control']['transformers']}. "
         f"myna is `{res['myna']['engine']}` at model "
         f"`{res['myna']['model_sha256'][:12]}`.",
         "",
         "## The second-engine slot",
         "",
         f"`bench/significance.py` names its competitor columns `laya_macro` and "
         f"`gap_myna_minus_laya` and its published intervals use those names; this harness puts "
         f"a different dump in that slot and writes the results as `vegaml_macro` and "
         f"`gap_myna_minus_vegaml`. The mapping is "
         f"`{json.dumps(res['relabel'])}`, recorded in the JSON so the relabel is auditable.",
         "",
         "| statistic | point | 95% CI |", "|---|---|---|"]
    for k in ("myna_macro", "majority_floor", "uniform_floor", "margin_over_floor",
              "vegaml_macro", "gap_myna_minus_vegaml"):
        lo, hi = c[k]
        L.append(f"| `{k}` | {p[k]:.4f} | [{lo:.4f}, {hi:.4f}] |")
    L += ["", f"myna's own four statistics are re-derived from the rows and agree with "
              f"`{Path(res['report']).name}` to 1e-9 before anything is resampled "
              f"(`{', '.join(res['guard']['stats_checked_to_1e-9'])}`).",
          "", "## Claims", ""]
    for cl in res["claims"]:
        L.append(f"- **{'stands' if cl['survives'] else 'does not stand'}** — {cl['claim']} "
                 f"(rule: {cl['threshold']}).")
    mc = res["mcnemar"]["vs_vegaml"]
    ub = res["control_health"]["unbound"]
    L += ["", "## McNemar",
          "",
          f"- myna vs vegaml, pooled over {mc['n']} discordant pairs: myna right and the "
          f"control wrong {mc['b']}, the reverse {mc['c']}, exact two-sided "
          f"p = {_p(mc['p'])}.",
          "",
          "## The control's own health",
          "",
          f"- abstained `{json.dumps(res['control_health']['abstain'])}`",
          f"- `unbound` — the engine's other decline flag, a distance-from-manifold test rather "
          f"than a confidence one: {ub['n']} of {ub['of']} outcomes, {ub['also_abstained']} of "
          f"them also flagged `abstain`",
          f"- argmax disagreements against the engine's own probability table: "
          f"{res['control_health']['argmax_disagreements']}",
          f"- rows/questions that raised or went unanswered: "
          f"{res['control_health']['errors']}",
          f"- reader tokens read across the pass: {res['control_health']['reader_tokens']} "
          f"in {res['control_health']['inference_seconds']} s",
          f"- warnings emitted: {json.dumps(res['control_health']['warnings'])}",
          "",
          "Scores are forced-choice: both of the control's decline flags are recorded, never "
          "applied, because neither myna nor laya can decline. Read the abstain and unbound "
          "rates with the accuracy."]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())

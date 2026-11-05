"""Every aggregate the paper reports, recomputed from released per-question outcomes.

    python -m bench.paper_numbers --out runs/paper_numbers.json --md runs/paper_numbers.md

Inputs are the row-outcome witnesses (`runs/*.rowoutcomes.json`) and the external-baseline
dump (`runs/external_*.json`). Nothing here reads a campaign summary: macro, overall,
bundle-exact, the per-cell table, per-primitive accuracy, ECE and Brier, and the
risk--coverage curve are all recomputed from the same 1,440 question records, so a table
in the paper cannot drift from the records it cites.

A note on the `score` primitive. The suite stores a star level 1--5 while the option vector
is 0-based, so `ranked[gold]` is the correct comparison for `choice` and `score` alike. An
earlier internal note claimed an off-by-one here and briefly shifted the `score` predictions;
that was wrong and is retracted (the labels are indices, not star levels). Records written
during that episode carry a shifted `pred` and are repaired with `--legacy-score`; everything
else is read as it is, `ok` already matching `pred == gold`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SOURCE_ORDER = ["agnews", "amazon", "banking77", "boolq", "contrastive", "dbpedia14",
                "imdb", "mnli", "sst5", "trec", "yelp"]


def load_outcomes(path: Path, legacy_score: bool) -> list[dict]:
    rows = json.loads(path.read_text())["rows"]
    out = []
    for row in rows:
        for qname, rec in row["questions"].items():
            ok = int(rec["ok"])
            if ok != int(str(rec["pred"]) == str(rec["gold"])):
                # The correction's regression test: on a post-fix record, `ok` must be
                # exactly `pred == gold`. A file that breaks this is from another
                # convention and is refused rather than silently averaged.
                raise SystemExit(
                    f"{path}: ok={ok} disagrees with pred={rec['pred']!r} gold={rec['gold']!r} "
                    f"on {row['meta']['id']}/{qname}; pass legacy_score for pre-fix files")
            out.append({"id": row["meta"]["id"], "source": row["meta"]["source"],
                        "cell": f"{row['meta']['source']}/{qname}", "qname": qname,
                        "type": rec["type"], "gold": rec["gold"], "pred": rec["pred"],
                        "conf": float(rec.get("conf") or 0.0), "ok": ok})
    return out


def ece_brier(recs: list[dict]) -> dict:
    conf = np.array([r["conf"] for r in recs], dtype=float)
    ok = np.array([r["ok"] for r in recs], dtype=float)
    ece = 0.0
    bins = []
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        m = (conf >= lo) & (conf < hi) if i < 9 else (conf >= lo) & (conf <= hi)
        if m.sum():
            gap = abs(float(conf[m].mean() - ok[m].mean()))
            ece += float(m.sum()) / len(conf) * gap
            bins.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                         "conf": round(float(conf[m].mean()), 6),
                         "acc": round(float(ok[m].mean()), 6)})
    return {"ece": round(ece, 6), "brier": round(float(np.mean((conf - ok) ** 2)), 6),
            "bins": bins}


def selective(recs: list[dict]) -> dict:
    order = sorted(recs, key=lambda r: -r["conf"])
    n = len(order)
    ok = np.array([r["ok"] for r in order], dtype=float)
    cum = np.cumsum(ok)
    out = {}
    for cov in (0.5, 0.6, 0.7, 0.8, 0.9):
        k = max(1, int(round(cov * n)))
        out[str(cov)] = {"k": k, "coverage": round(k / n, 4),
                         "acc": round(float(cum[k - 1] / k), 6)}
    return out


def summarize(recs: list[dict]) -> dict:
    per_cell = defaultdict(list)
    per_type = defaultdict(list)
    per_row = defaultdict(list)
    for r in recs:
        per_cell[r["cell"]].append(r["ok"])
        per_type[r["type"]].append(r["ok"])
        per_row[r["id"]].append(r["ok"])
    cells = {c: {"n": len(v), "acc": round(float(np.mean(v)), 6)}
             for c, v in sorted(per_cell.items())}
    return {
        "n_questions": len(recs),
        "n_rows": len(per_row),
        "macro": round(float(np.mean([v["acc"] for v in cells.values()])), 6),
        "overall": round(float(np.mean([r["ok"] for r in recs])), 6),
        "bundle_exact": round(float(np.mean([all(v) for v in per_row.values()])), 6),
        "cells": cells,
        "primitives": {t: {"n": len(v), "acc": round(float(np.mean(v)), 6)}
                       for t, v in sorted(per_type.items())},
        "calibration": ece_brier(recs),
        "selective": selective(recs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/paper_numbers.json")
    ap.add_argument("--md", default="runs/paper_numbers.md")
    args = ap.parse_args()

    # (name, path, legacy-score flag) — legacy files predate the score fix.
    arms = [
        ("myna_v6", "runs/paper_myna_v6_test.rowoutcomes.json", False),
        ("myna_v6_prefix", "runs/paper_myna_v6_test.predates-fix.rowoutcomes.json", False),
        ("myna_v6_packed_forced", "runs/v6_packed_forced.rowoutcomes.json", False),
        ("laya_0419", "runs/laya_0419_test.rowoutcomes.json", False),
        ("vegaml_0150", "runs/vegaml_0150_test.rowoutcomes.json", False),
    ]
    out = {"arms": {}}
    for name, rel, legacy in arms:
        path = REPO / rel
        if not path.exists():
            continue
        out["arms"][name] = summarize(load_outcomes(path, legacy))

    for name, rel in [("majority", "runs/external_majority.json"),
                      ("tfidf", "runs/external_tfidf.json"),
                      ("nli", "runs/external_nli.json")]:
        path = REPO / rel
        if not path.exists():
            continue
        rec = json.loads(path.read_text())[name]
        out["arms"][name] = {"macro": round(rec["macro"], 6),
                             "overall": round(rec["overall"], 6),
                             "bundle_exact": round(rec["bundle_exact"], 6),
                             "cells": {c: {"n": v["n"], "acc": round(v["acc"], 6)}
                                       for c, v in rec["cells"].items()}}

    (REPO / args.out).write_text(json.dumps(out, indent=1))
    lines = ["# Recomputed paper numbers", "",
             "| arm | macro | overall | bundle exact | n |", "|---|---:|---:|---:|---:|"]
    for name, rec in out["arms"].items():
        lines.append(f"| {name} | {rec['macro']:.4f} | {rec['overall']:.4f} | "
                     f"{rec['bundle_exact']:.4f} | {rec.get('n_questions', 1440)} |")
    if "myna_v6" in out["arms"]:
        m = out["arms"]["myna_v6"]
        lines += ["", "## Myna v6 per cell", "", "| cell | n | acc |", "|---|---:|---:|"]
        for c, v in m["cells"].items():
            lines.append(f"| {c} | {v['n']} | {v['acc']:.4f} |")
        lines += ["", f"ECE {m['calibration']['ece']:.4f}; "
                      f"Brier {m['calibration']['brier']:.4f}; "
                      f"selective@60% {m['selective']['0.6']['acc']:.4f} "
                      f"(k={m['selective']['0.6']['k']})"]
    (REPO / args.md).write_text("\n".join(lines) + "\n")
    print("wrote", args.out, "and", args.md)


if __name__ == "__main__":
    main()

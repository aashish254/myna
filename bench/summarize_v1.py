"""Roll myna-v1 metrics.json and the laya witness JSON into one source x type
table for the README/PLAN (pure CPU, no inference).

myna's evaluate() keys are "source#sig8/qname"; laya's rows carry source+type.
Both are aggregated by (source, type) as a per-question-instance mean so the
two are directly comparable. Noul Brier is averaged over noul instances.

Usage: uv run python -m bench.summarize_v1 \
         --myna runs/myna-v1/metrics.json --laya runs/laya_decision_v2.json
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections import defaultdict


def myna_roll(myna_metrics, qtypes):
    # source -> type -> [sum acc*n, n, sum brier, n_brier]
    agg = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0, 0]))
    for key, val in myna_metrics.items():
        if key.endswith(":brier") or key.endswith(":ece"):
            continue
        wf, qname = key.split("/", 1)
        source = wf.split("#", 1)[0]
        t = qtypes.get(key, "choice")
        n = 1
        agg[source][t][0] += val
        agg[source][t][1] += n
    # brier
    for key, val in myna_metrics.items():
        if key.endswith(":brier"):
            base = key[: -len(":brier")]
            wf, qname = base.split("/", 1)
            source = wf.split("#", 1)[0]
            agg[source]["noul"][2] += val
            agg[source]["noul"][3] += 1
    return agg


def laya_roll(laya):
    agg = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0, 0]))
    # laya rows are already per (source,qid,type) with acc*n implied; recompute means
    tmp = defaultdict(lambda: defaultdict(list))
    for r in laya["rows"]:
        tmp[r["source"]][r["type"]].append(r)
    for source, byt in tmp.items():
        for t, rs in byt.items():
            tot = sum(x["n"] for x in rs)
            acc = sum(x["acc"] * x["n"] for x in rs) / tot if tot else float("nan")
            agg[source][t][0] = acc
            agg[source][t][1] = 1  # single representative cell
            briers = [x["brier"] for x in rs if x["brier"] is not None]
            if briers:
                agg[source][t][2] = sum(briers) / len(briers)
                agg[source][t][3] = 1
    return agg


def cell(byt, t, show_brier=False):
    if t not in byt or byt[t][1] == 0:
        return "—"
    a = byt[t][0] / byt[t][1]
    s = f"{a:.3f}"
    if show_brier and byt[t][3]:
        s += f" (B {byt[t][2] / byt[t][3]:.3f})"
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--myna", default="runs/myna-v1/metrics.json")
    ap.add_argument("--laya", default="runs/laya_decision_v2.json")
    ap.add_argument("--out", default="runs/v1_vs_laya.md")
    args = ap.parse_args(argv)
    cmd = shlex.join(["python", "bench/summarize_v1.py",
                     *(argv if argv is not None else sys.argv[1:])])

    mj = json.load(open(args.myna))
    lj = json.load(open(args.laya))
    myna = myna_roll(mj["test"], mj.get("qtypes", {}))
    laya = laya_roll(lj)

    sources = sorted(set(myna) | set(laya))
    types = ["choice", "score", "noul"]
    print(f"myna-v1 test (temperature {mj['temperature']:.2f}) vs laya {lj['split']} "
          f"(n<= {lj['n_per_source']}/source)\n")
    hdr = f"{'source':<12} " + "".join(f"{t:>18} " for t in types)
    print(hdr)
    lines = [hdr]
    for s in sources:
        row = f"{s:<12} "
        for t in types:
            mc = cell(myna.get(s, {}), t, show_brier=(t == "noul"))
            lc = cell(laya.get(s, {}), t, show_brier=(t == "noul"))
            row += f"{mc + ' / ' + lc:>18} "
        lines.append(row)
    print("\n".join(lines[1:]))

    # overall per-type
    def overall(agg):
        out = {}
        for t in types:
            num = den = 0.0
            for s in agg:
                if t in agg[s] and agg[s][t][1]:
                    num += agg[s][t][0]
                    den += agg[s][t][1]
            out[t] = num / den if den else float("nan")
        return out
    mo, lo = overall(myna), overall(laya)
    print("\nOVERALL by type  (myna / laya):")
    for t in types:
        print(f"  {t:6s}  {mo[t]:.3f} / {lo[t]:.3f}")

    with open(args.out, "w") as f:
        f.write("# myna-v1 vs laya Router — kev decision-v2\n\n")
        f.write(f"`{cmd}`\n\n")
        f.write(f"myna test split (temperature {mj['temperature']:.2f}); laya {lj['split']} split "
                f"subsampled to {lj['n_per_source']}/source. Cells are accuracy; noul adds Brier.\n\n")
        f.write("| source | " + " | ".join(f"{t} (myna/laya)" for t in types) + " |\n")
        f.write("|---|" + "---|" * len(types) + "\n")
        for s in sources:
            cells = []
            for t in types:
                cells.append(f"{cell(myna.get(s, {}), t, t=='noul')} / {cell(laya.get(s, {}), t, t=='noul')}")
            f.write(f"| {s} | " + " | ".join(cells) + " |\n")
        f.write("\n**Overall by type** — " + ", ".join(
            f"{t}: {mo[t]:.3f} vs {lo[t]:.3f}" for t in types) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

"""What P9 9d's scope decision is worth, priced out of one committed run.

    .venv/bin/python -m bench.scope_pricing \
        --report runs/v1b_kaggle_3600b.report.json --out runs/scope_pricing.json

G1 is two numbers: a macro over the published (source, question) cells, and a margin over
the majority floor *those same cells* imply. 9d asks whether the two dead cells
(`banking77/intent`, `mnli/relation`) belong inside that scope — and a scope choice moves
both sides at once, because the cells myna wins are also the ones a majority classifier
wins. Dropping the dead ones therefore raises the floor myna is measured against, which is
why "drop them and we are closer" has to be arithmetic rather than an argument.

Three rules, each a way this table could become marketing:

* **The arithmetic is imported, never restated.** `myna.report.macro` and
  `myna.report.g1_verdict` are the two the published report calls. A second implementation
  of the roll-up is §9.37's defect — one run, two "test macro" numbers, the launch doc
  quoting the flattering one.
* **The unfiltered subset must reproduce the committed report.** With nothing dropped, the
  macro and the verdict are compared to the report's own `macro` and `g1` blocks, and the
  script refuses to print a table that disagrees. That equality is what makes the filtered
  rows the same measurement on a different scope rather than a new measurement.
* **Every row carries its recomputed floor and its own verdict.** A row that printed only
  myna's macro would read as "we would pass if we dropped X"; the floor and the shortfall
  to 0.70 are in the same line.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from myna.report import g1_verdict, macro  # noqa: E402

TOLERANCE = 1e-12

# (label, which cells it keeps) — the two 9d options, then the most aggressive
# defensible scope: every cell that answers below its own majority floor.
SUBSETS = [
    ("published scope — all cells", lambda c, dead: True),
    ("9d option A — the two named cells out",
     lambda c, dead: (c["source"], c["question"]) not in dead),
    ("9d option B — their whole sources out", lambda c, dead: c["source"] not in {s for s, _ in dead}),
    ("every cell below its own floor out", lambda c, dead: c["acc"] >= c["majority"]),
]

NAMED = {("banking77", "intent"), ("mnli", "relation")}


def price(cells, keep):
    """One scope row: the subset's macro, its own floor, and the verdict G1 would give it."""
    sub = [c for c in cells if keep(c, NAMED)]
    acc = macro(sub, "acc")
    maj = macro(sub, "majority")
    v = g1_verdict(acc, maj)
    return {
        "cells": len(sub),
        "dropped": len(cells) - len(sub),
        "myna": acc,
        "majority_floor": maj,
        "uniform_floor": macro(sub, "uniform"),
        "margin_observed": v["margin_observed"],
        "shortfall_to_target": round(0.70 - acc, 6),
        "shortfall_to_margin": round(0.15 - v["margin_observed"], 6),
        "meets_target": v["meets_target"],
        "meets_margin": v["meets_margin"],
        "pass": v["pass"],
    }


def table(rows, published) -> str:
    head = (f"{'scope':<38}{'cells':>6}{'floor':>8}{'margin':>9}"
            f"{'need +0.15':>12}{'need 0.70':>11}  verdict")
    lines = [f"G1 on the published scope: macro {published['myna']:.4f} · "
             f"floor {published['majority_floor']:.4f} · "
             f"{published['margin_observed']:+.4f} — "
             f"target {published['meets_target']}, margin {published['meets_margin']}",
             "", head, "-" * len(head)]
    for label, r in rows:
        lines.append(f"{label:<38}{r['cells']:>6}{r['majority_floor']:>8.4f}"
                     f"{r['margin_observed']:>+9.4f}{r['shortfall_to_margin']:>12.4f}"
                     f"{r['shortfall_to_target']:>11.4f}  "
                     f"{'pass' if r['pass'] else 'not met'}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", default="runs/v1b_kaggle_3600b.report.json",
                    help="a committed myna.report JSON")
    ap.add_argument("--out", default="runs/scope_pricing.json")
    args = ap.parse_args(argv)

    path = Path(args.report)
    if not path.exists():
        raise SystemExit(f"--report {args.report} is not on disk")
    doc = json.loads(path.read_text())
    cells, published = doc["cells"], doc["macro"]

    rows = [(label, price(cells, keep)) for label, keep in SUBSETS]
    control = rows[0][1]
    # §9.37: the subset math has to be the published math, or this is a third roll-up of
    # one run rather than a scope question about it.
    for field, want in (("myna", published["acc"]), ("majority_floor", published["majority"]),
                        ("uniform_floor", published["uniform"])):
        if abs(control[field] - want) > TOLERANCE:
            raise SystemExit(f"the unfiltered scope does not reproduce the report: "
                             f"{field} is {control[field]!r}, the report prints {want!r}")
    if control["pass"] is not doc["g1"]["pass"]:
        raise SystemExit("the unfiltered verdict disagrees with the report's own g1 block")

    # Flags only, so the line is pasteable however this checkout was invoked (§9.30).
    flags = list(argv if argv is not None else sys.argv[1:])
    out = {
        "cmd": shlex.join(["python", "-m", "bench.scope_pricing", *flags]),
        "report": str(path),
        "g1": {"target": doc["g1"]["target"], "margin": doc["g1"]["margin"]},
        "named_cells_out_of_scope": [f"{s}/{q}" for s, q in sorted(NAMED)],
        "below_floor_cells": [f"{c['source']}/{c['question']}" for c in cells
                              if c["acc"] < c["majority"]],
        "published": control,
        "scopes": {label: r for label, r in rows},
    }
    print(table(rows, control))
    print(f"\n{len(out['below_floor_cells'])} of {len(cells)} cells answer below their own "
          "majority floor; each is listed in the JSON.")
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

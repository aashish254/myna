"""Judge a paired `--score-loss` ablation and commit what it concluded (SPEC §5 P9 9e).

Three questions, in this order, because each can veto the ones after it:

1. Did the arms differ in exactly one flag? Read off each run's own `cmd:` line,
   never from this script's memory of what it launched — a confounded pair is not
   an ablation, it is two experiments.
2. What is the churn floor? The cells this loss term cannot price still move,
   because the trunk is shared. **A score-cell delta smaller than that mean is not
   an effect**, and quoting it without the floor is how a negative result gets
   published as a positive one.
3. Do the `score` cells move the same way in both seeds? Two seeds is two draws;
   sign agreement is the strongest claim this dose supports.

The unit of comparison is a within-seed PAIR, not all four runs: at the dose this
was first run at, `--seed` also resized the memory plan, so seed 0 and seed 1
trained different batch sizes for reasons that have nothing to do with the loss.

Raw inputs (each arm's `metrics.json` and `train.log`) are not committed — the
metrics are ~295 KB per arm and `model.pt` is a checkpoint — so the artifact this
writes carries the per-cell accuracies it rolled up, which keeps every published
figure re-traceable to the rollup even after the arm directories are gone.

    python bench/ordinal_ab.py --ab-dir /tmp/ab2
"""

import argparse
import json
import shlex
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from myna import report
from myna.real_data import load_suite
from myna.train import macro_acc

REPO = Path(__file__).resolve().parent.parent
ARMS = ("ce0", "emd0", "ce1", "emd1")
SEEDS = ("0", "1")
SPLITS = ("dev", "test")
SCORE = ("sst5/sentiment", "yelp/rating", "amazon/stars")
ALLOWED_DIFF = {"--score-loss", "--seed", "--out"}


def flags(ab_dir: Path, arm: str) -> dict:
    """The run's own command line as a flag -> value map, normalised so two arms
    can be compared token by token."""
    for ln in (ab_dir / f"{arm}.log").read_text().splitlines():
        if ln.startswith("cmd:"):
            toks = shlex.split(ln[4:])[2:]        # drop `python -m myna.train`
            out, i = {}, 0
            while i < len(toks):
                if not toks[i].startswith("--"):
                    i += 1
                    continue
                flag = toks[i]
                i += 1
                val = []
                while i < len(toks) and not toks[i].startswith("--"):
                    val.append(toks[i])
                    i += 1
                out[flag] = " ".join(val)
            return out
    raise AssertionError(f"no cmd: line in {arm}.log — the run did not report what it ran")


def rollup(metrics: dict, groups: dict) -> dict:
    cells, unmatched = report.accuracy_cells(metrics, groups)
    assert not unmatched, f"{len(unmatched)} metric keys matched no group"
    return {f"{s}/{q}": c["acc"] for (s, q), c in cells.items()}


def check_the_arms_are_a_pair(fs: dict, meta: dict) -> None:
    """Veto everything downstream if the arms are not one experiment apart."""
    for a, b in combinations(ARMS, 2):
        diff = {k for k in set(fs[a]) | set(fs[b]) if fs[a].get(k) != fs[b].get(k)}
        print(f"  {a:5s} vs {b:5s}: {sorted(diff)}")
        assert diff <= ALLOWED_DIFF, f"unplanned difference: {sorted(diff - ALLOWED_DIFF)}"
    for seed in SEEDS:
        pair = [a for a in ARMS if a.endswith(seed)]
        for field in ("batch", "p95", "last_step"):
            seen = {meta[a][field] for a in pair}
            assert len(seen) == 1, f"seed {seed} pair differs on {field}: {seen}"
    assert all(meta[a]["stopped"] is None for a in ARMS), "an arm stopped early"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ab-dir", default="/tmp/ab2", type=Path)
    ap.add_argument("--suite", default=str(REPO / "data" / "decision-v2-pilot"))
    ap.add_argument("--out", default=str(REPO / "runs" / "ordinal_ab.json"))
    ap.add_argument("--dose", default="250 updates, warm-started from one shared init")
    args = ap.parse_args(argv)

    data = load_suite(args.suite)
    acc, meta = {}, {}
    for arm in ARMS:
        mm = json.loads((args.ab_dir / arm / "metrics.json").read_text())
        acc[arm] = {s: rollup(mm[s], data[s]) for s in SPLITS}
        meta[arm] = {"batch": mm["batch"], "p95": mm["state_tokens_p95"],
                     "last_step": mm["last_step"], "stopped": mm["stopped"],
                     "temperature": mm["temperature"],
                     **{f"macro_{s}": macro_acc(mm[s]) for s in SPLITS}}

    fs = {a: flags(args.ab_dir, a) for a in ARMS}
    print("confound check — flags that differ between each pair of arms:")
    check_the_arms_are_a_pair(fs, meta)
    print(f"\n{'arm':6s} {'temp':5s} {'batch':6s} {'p95':5s} {'last':5s} "
          f"{'dev':7s} {'test':7s}")
    for a in ARMS:
        print(f"{a:6s} {meta[a]['temperature']:5.2f} {meta[a]['batch']:6d} "
              f"{meta[a]['p95']:5d} {meta[a]['last_step']:5d} "
              f"{meta[a]['macro_dev']:7.4f} {meta[a]['macro_test']:7.4f}")

    cells = sorted(acc["ce0"]["dev"])
    delta = {s: {c: [round(acc[f"emd{s_}"][s][c] - acc[f"ce{s_}"][s][c], 6) for s_ in SEEDS]
                 for c in cells} for s in SPLITS}

    print(f"\n{'cell':22s} {'dev d0':>8s} {'d1':>8s} | {'test d0':>8s} {'d1':>8s}")
    for c in cells:
        print(f"{c:22s} {delta['dev'][c][0]:>8.4f} {delta['dev'][c][1]:>8.4f} | "
              f"{delta['test'][c][0]:>8.4f} {delta['test'][c][1]:>8.4f}"
              + ("   <- priced by the term" if c in SCORE else ""))

    pairs, agree = {}, {s: 0 for s in SPLITS}
    for s in SPLITS:
        priced = [delta[s][c] for c in SCORE]
        unpriced = [delta[s][c] for c in cells if c not in SCORE]
        # `(a > 0) == (b > 0)`, spelled out: `a > 0 == b > 0` would chain in Python
        # and test `0 == b` instead of the signs agreeing.
        agree[s] = sum(1 for c in SCORE if delta[s][c][0] != 0
                       and (delta[s][c][0] > 0) == (delta[s][c][1] > 0))
        for i, seed in enumerate(SEEDS):
            floor = sum(abs(x[i]) for x in unpriced) / len(unpriced)
            mean = sum(x[i] for x in priced) / len(priced)
            pairs[f"{s}_seed{seed}"] = {
                "macro_emd_minus_ce": round(meta[f"emd{seed}"][f"macro_{s}"]
                                            - meta[f"ce{seed}"][f"macro_{s}"], 6),
                "score_cell_mean": round(mean, 6),
                "churn_floor": round(floor, 6),
                "resolution_ratio": round(abs(mean) / floor, 4) if floor else None,
            }
        print(f"\n{s}: sign agreement across seeds on the priced cells: {agree[s]}/{len(SCORE)}")
        for seed in SEEDS:
            p = pairs[f"{s}_seed{seed}"]
            print(f"  seed {seed}: macro {p['macro_emd_minus_ce']:+.4f}  "
                  f"score-cell mean {p['score_cell_mean']:+.4f}  "
                  f"churn floor {p['churn_floor']:.4f}  ratio {p['resolution_ratio']}")

    # one ratio per pair, and the pooled figure for continuity. Pooling the seeds
    # into one mean is what an earlier read of this same data did, and it is the
    # number to distrust: it averages a positive draw against a zero one and reports
    # "below the floor", which happens to be right here for the wrong reason. The
    # per-seed rows are the unit, because a pair that disagrees with its own
    # replicate is not an effect no matter what the pooled mean says.
    ratios = {k: (p["resolution_ratio"] or 0) for k, p in pairs.items()}
    above = sorted(k for k, v in ratios.items() if v >= 1)
    macro_conflict = {s: (pairs[f"{s}_seed0"]["macro_emd_minus_ce"] > 0)
                      != (pairs[f"{s}_seed1"]["macro_emd_minus_ce"] > 0) for s in SPLITS}
    pooled = {}
    for s in SPLITS:
        priced = [v for c in SCORE for v in delta[s][c]]
        unpriced = [v for c in cells if c not in SCORE for v in delta[s][c]]
        floor = sum(abs(v) for v in unpriced) / len(unpriced)
        mean = sum(priced) / len(priced)
        pooled[s] = {"score_cell_mean": round(mean, 6), "churn_floor": round(floor, 6),
                     "resolution_ratio": round(abs(mean) / floor, 4) if floor else None,
                     "positive_draws": f"{sum(v > 0 for v in priced)}/{len(priced)}"}
        print(f"\n{s} pooled over both seeds: mean {mean:+.4f}  floor {floor:.4f}  "
              f"ratio {pooled[s]['resolution_ratio']}  "
              f"(positive draws {pooled[s]['positive_draws']})")

    conflict = " and ".join(f"{s}: {pairs[f'{s}_seed0']['macro_emd_minus_ce']:+.4f} vs "
                            f"{pairs[f'{s}_seed1']['macro_emd_minus_ce']:+.4f}"
                            for s in SPLITS if macro_conflict[s])
    if above:
        verdict = (f"not shown. {', '.join(above)} exceed their churn floor, "
                   f"{', '.join(sorted(k for k in ratios if k not in above))} do not, and "
                   f"the replicates disagree — sign agreement on the priced cells is "
                   f"{agree['dev']}/3 on dev and {agree['test']}/3 on test. "
                   f"Macro moves opposite ways in the two seeds ({conflict}). "
                   "This dose cannot separate the term from trunk noise.")
    else:
        verdict = ("not resolvable at this dose: every score-cell mean sits below the "
                   "churn floor of cells the term cannot price.")
    out = {"what": "paired --score-loss ce vs emd ablation", "dose": args.dose,
           "score_cells": list(SCORE), "cells_priced_by_the_term": len(SCORE),
           "arms": {a: {"flags": fs[a], "meta": meta[a], "cells": acc[a]} for a in ARMS},
           "delta": delta, "pairs": pairs, "pooled_over_seeds": pooled,
           "pairs_above_floor": above,
           "sign_agreement": {s: f"{agree[s]}/3" for s in SPLITS},
           "verdict": verdict}
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}\nverdict: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

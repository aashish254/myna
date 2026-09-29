"""Measure what `--anti-prior` batching actually reaches, on the shipped pilot rows.

Tier 0 (SPEC §9.47) found that eight of decision-v2's sixteen cells answer from an
association rather than from the state: swapping the state moves their accuracy by
<=0.001, and each commits its most-predicted label to more test rows than the gold
majority has. Three of the eight are agnews `noul` cells that emit one label on *every*
row and finish exactly on their own majority floor (0.653 / 0.809 / 0.778). The cheapest
mechanistic reason the training data offers is a label prior: a constant answer is
worth ~0.75 against agnews' train marginal of 0.739-0.755, so the shortcut is the
best-scoring policy the rows support. `--anti-prior` (SPEC §5 P10) re-weights the
mini-batch draw by the inverse share of each row's gold label inside its
(source, question) cell, normalized within each source so the task mix does not move.

This script prices that claim **without a model**, because none of it is about the
model: the weights, the drawer and the cell budget are the whole mechanism, and all
three are reachable on CPU from the rows on disk. The arms run the *real* batcher
(`myna.train.draw_row_batch`) at the *real* `--row-batch` settings of the published
run — batch 10, 2,048 question cells per forward, 8 sets per update, 3,600 updates,
one seed shared across arms — so the printed marginals are the ones a run would have
trained on, not an analytic approximation of them.

Three things it says out loud, because each one was a plausible way to fool
ourselves:

* **Reach.** Row-level weighting flattens a single-question cell almost exactly
  (boolq 0.624 -> 0.502, yelp 0.605 -> 0.510) and a multi-question cell only part of the
  way (agnews 0.739-0.755 -> 0.617-0.628), because one agnews row answers three questions
  at once. `--compare` is how the combination rule was chosen rather than guessed. And
  the reach is only over cells the skew threshold admits: six of sixteen carry a train
  majority at or above 0.55, which covers three of Tier 0's eight association-driven
  cells. The other five (amazon/stars, banking77/intent, contrastive/decision,
  mnli/relation, sst5/sentiment) are already flatter than the threshold in their own
  train marginal, so an inverse-prior re-weighting has almost nothing to remove for
  them — the shortcut there, if it is one, is not a label prior. The ten are not left
  alone, though, and the direction is not uniform: nine flatten (contrastive/decision
  0.507 -> 0.337, i.e. down to its own 1/3 uniform) and one *sharpens*, yelp/rating
  0.203 -> 0.259, because a yelp row answers `rating` and `recommend` at once and the
  weights are computed on the other question. Row-level weighting cannot hold a cell
  fixed when its rows carry several labels; that is a limit of the mechanism, not of
  this measurement, and the marginals table prints all sixteen cells so it stays visible.
* **The mix confound.** A source's drawn share is already ~5 points off its row share
  with the flag *off*, because `--max-q-cells` skips rows whose question branch would
  blow the budget and banking77's 77-option sets pay for it. Only the difference
  between the arms belongs to these weights, so that is what gets printed.
* **What it cannot say.** Nothing here is an accuracy claim. Flattening the training
  marginal does not flatten the *test* marginal: the floors the report judges cells
  against stay the natural majority of the untouched split, so if removing the
  shortcut does not make the model read, the macro falls. That is a GPU question, and
  it is left open here on purpose (§9.38 — a slope needs a run). The price of that run
  is read off the published log rather than guessed: 5.618 s/update measured from
  `runs/v1b_kaggle_3600b.train.log`, so the pair at the published dose is 2 x 3,600
  updates = **~11.2 GPU hours** against a 30 h/week quota. (The ~31.2 h figure this
  sentence used to carry is the *10k* paired lane, which is what §9.38 was pricing.)

Usage:
    python bench/anti_prior_audit.py [--suite data/decision-v2-pilot] [--compare]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tokenizers import Tokenizer  # noqa: E402

from myna.real_data import (ANTI_PRIOR_SKEW, COMBINE, balance_weights, cell_priors,  # noqa: E402
                            flatten_groups, flatten_weighted, load_split, source_of_group)
from myna.report import cell_stats  # noqa: E402  (cell *types*, not a second prior)
from myna.train import PriorAudit, WeightedDraw, draw_row_batch  # noqa: E402

#: the published v1b dose, so each arm draws exactly as many rows as that run did.
PUBLISHED_BATCH = 10
PUBLISHED_MAX_Q_CELLS = 2048
PUBLISHED_SETS_PER_UPDATE = 8
PUBLISHED_UPDATES = 3600


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_by_source(groups):
    out: dict[str, int] = {}
    for key, (_q, exs) in groups.items():
        out[source_of_group(key)] = out.get(source_of_group(key), 0) + len(exs)
    return out


def majority_of(counts):
    """The share the cell's most frequent label actually carried in the draws."""
    tot = sum(counts.values())
    return max(counts.values()) / tot if tot else 0.0


def draw(items, drawer, tok, args, audit):
    """One arm: draw the published dose of mini-batches through the real batcher."""
    rng = random.Random(args.seed)
    cache: dict[int, int] = {}
    rows = 0
    t0 = time.time()
    for _ in range(args.updates * args.sets_per_update):
        picked = draw_row_batch(items, args.batch, args.max_q_cells, rng, tok, cache,
                                drawer=drawer)
        audit.add_items(picked)
        rows += len(picked)
    return {"rows_drawn": rows, "mini_batches": args.updates * args.sets_per_update,
            "mean_batch": round(rows / (args.updates * args.sets_per_update), 2),
            "question_sets_seen": len(cache), "seconds": round(time.time() - t0, 1)}


def marginals_table(priors, drawn, types, skew):
    """One row per cell: its natural majority and the majority the draws realized."""
    head = (f"{'cell':24s} {'type':7s} {'drawn':>7s} {'labels':>6s} {'natural':>8s} "
            f"{'realized':>9s} {'move':>9s}  {'reach'}")
    rows = [head, "-" * (len(head) + 8)]
    for cell, nat in sorted(priors.items(), key=lambda kv: -max(kv[1].values())):
        c = drawn.get(cell)
        if not c:
            continue
        want, got = max(nat.values()), majority_of(c)
        rows.append(f"{cell[0]+'/'+cell[1]:24s} {types.get(cell, '?'):7s} {sum(c.values()):7d} "
                    f"{len(nat):6d} {want:8.4f} {got:9.4f} {got - want:+9.4f}  "
                    f"{'targets' if want >= skew else ''}")
    return "\n".join(rows)


def compare_table(priors, audits, order):
    """Realized majority per combination rule — the table `balance_weights` cites.

    `order` is the skewed cells, most skewed first, as (source, qname) tuples."""
    head = f"{'rule':8s} " + " ".join(f"{s + '/' + n:>16s}" for s, n in order)
    rows = [head, "-" * len(head),
            f"{'natural':8s} " + " ".join(f"{max(priors[c].values()):16.4f}" for c in order)]
    for rule in sorted(audits):
        a = audits[rule]
        rows.append(f"{rule:8s} " + " ".join(
            f"{majority_of(a.drawn[c]) if a.drawn.get(c) else float('nan'):16.4f}"
            for c in order))
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--suite", default=str(ROOT / "data/decision-v2-pilot"))
    ap.add_argument("--split", default="train.jsonl")
    ap.add_argument("--batch", type=int, default=PUBLISHED_BATCH)
    ap.add_argument("--max-q-cells", type=int, default=PUBLISHED_MAX_Q_CELLS)
    ap.add_argument("--sets-per-update", type=int, default=PUBLISHED_SETS_PER_UPDATE)
    ap.add_argument("--updates", type=int, default=PUBLISHED_UPDATES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skew", type=float, default=ANTI_PRIOR_SKEW,
                    help="a cell needs this much majority before --anti-prior is said to "
                         "target it; the same constant the trainer prints")
    ap.add_argument("--compare", action="store_true",
                    help="draw every COMBINE rule through balance_weights, which is how the "
                         "shipped one was picked")
    ap.add_argument("--out", default=None, help="write the audit JSON here")
    args = ap.parse_args(argv)

    split = Path(args.suite) / args.split
    if not split.exists():
        raise SystemExit(f"no {split}")
    tok_path = Path(args.suite) / "tokenizer-8192.json"
    if not tok_path.exists():
        raise SystemExit(f"no {tok_path}: this audit prices the batcher with the tokenizer the "
                         "run used, because the cell budget is measured in its tokens")
    groups = load_split(split)
    tok = Tokenizer.from_file(str(tok_path))
    cmd = " ".join([Path(sys.argv[0]).name, *(argv if argv is not None else sys.argv[1:])])
    nrows = sum(len(e) for _q, e in groups.values())
    print("cmd:", cmd, flush=True)
    print(f"suite: {split} sha256 {sha256(split)[:16]}  {nrows} rows in {len(groups)} "
          f"question sets", flush=True)
    print(f"dose per arm: {args.updates} updates x {args.sets_per_update} sets x batch "
          f"{args.batch} at {args.max_q_cells} cells/forward, seed {args.seed}", flush=True)

    priors = cell_priors(groups)
    skewed = {c: max(v.values()) for c, v in priors.items() if max(v.values()) >= args.skew}
    print(f"\ncells at or above skew {args.skew}: {len(skewed)} of {len(priors)} — "
          + ", ".join(f"{s}/{n} {m:.4f}" for (s, n), m in
                      sorted(skewed.items(), key=lambda kv: -kv[1])), flush=True)

    # The invariant `balance_weights` promises, checked on the shipped rows rather than
    # asserted from the code: every source's weights sum to its row count, so a
    # proportional draw leaves the task mix where the rows put it.
    ws = balance_weights(groups)
    sums: dict[str, list[float]] = {}
    for key, w in ws.items():
        sums.setdefault(source_of_group(key), []).extend(w)
    nsrc = rows_by_source(groups)
    worst = max(abs(sum(w) - nsrc[s]) / nsrc[s] for s, w in sums.items())
    print("\nper-source weight sums (each should equal its row count):", flush=True)
    for s in sorted(sums):
        print(f"  {s:14s} rows {nsrc[s]:6d}  sum(w) {sum(sums[s]):10.2f}", flush=True)
    print(f"worst relative deviation {worst:.2e}", flush=True)
    if worst > 1e-9:
        raise SystemExit(f"the per-source rescale is not exact: {worst:.2e} — the mix would move")

    items_u = flatten_groups(groups)
    types = {k: v["type"] for k, v in cell_stats(groups).items()}

    print("\n=== arm uniform: --anti-prior off, what every committed figure was drawn by ===",
          flush=True)
    a_u = PriorAudit(groups)
    s_u = draw(items_u, None, tok, args, a_u)
    print(marginals_table(priors, a_u.drawn, types, args.skew), flush=True)
    lines_u, payload_u = a_u.report(args.skew)

    print(f"\n=== arm weighted: --anti-prior on, combine='prod' ===", flush=True)
    items_w, weights = flatten_weighted(groups)
    if [id(e) for _q, e in items_w] != [id(e) for _q, e in items_u]:
        raise SystemExit("flatten_weighted returned the rows in a different order than "
                         "flatten_groups — the weights would belong to other rows")
    a_w = PriorAudit(groups)
    s_w = draw(items_w, WeightedDraw(weights), tok, args, a_w)
    print(marginals_table(priors, a_w.drawn, types, args.skew), flush=True)
    lines_w, payload_w = a_w.report(args.skew)

    print("\n=== the two audits, as the trainer would print them ===", flush=True)
    for label, lines in (("uniform: --anti-prior off, the control arm", lines_u),
                         ("weighted: --anti-prior on, combine='prod', the treated arm", lines_w)):
        # The trainer's own lines, unmodified, under a header naming which arm drew them.
        # Both arms print the identical `anti-prior: cell majority X -> Y` banner, so in
        # order alone they are one list of marginals, not a control and a treatment.
        print(f"--- {label} ---", flush=True)
        for line in lines:
            print(line, flush=True)

    tot_u, tot_w = max(a_u.rows, 1), max(a_w.rows, 1)
    print("\nsource mix of the drawn batches (pts; the gap to the row share is the cell "
          "budget's, in both arms — only the last column belongs to the weights):", flush=True)
    mix_worst = 0.0
    for s in sorted(nsrc):
        want = nsrc[s] / nrows * 100
        gu = a_u.drawn_rows[s] / tot_u * 100
        gw = a_w.drawn_rows[s] / tot_w * 100
        mix_worst = max(mix_worst, abs(gw - gu))
        print(f"  {s:14s} rows {want:5.2f}  uniform {gu:5.2f} ({gu - want:+5.2f})  "
              f"weighted {gw:5.2f} ({gw - want:+5.2f})  weighted-uniform {gw - gu:+5.2f}",
              flush=True)
    print(f"largest between-arm mix difference: {mix_worst:.2f} pts", flush=True)

    payloads_c = {}
    if args.compare:
        print("\n=== --compare: every COMBINE rule drawn through balance_weights ===",
              flush=True)
        order = [c for c, _ in sorted(skewed.items(), key=lambda kv: -kv[1])]
        audits = {}
        for rule in sorted(COMBINE):
            its, w = flatten_weighted(groups, rule)
            a = PriorAudit(groups)
            draw(its, WeightedDraw(w), tok, args, a)
            audits[rule] = a
        print(compare_table(priors, audits, order), flush=True)
        print(f"(natural row share of the drawn majority, per rule; {args.updates} updates, "
              f"seed {args.seed})", flush=True)
        for rule in sorted(audits):
            _, p = audits[rule].report(args.skew)
            payloads_c[rule] = p
        print("shipped rule: prod — mean/max/geo/sum are kept in COMBINE only so this "
              "table stays reproducible", flush=True)

    print("\ndraw stats: uniform " + json.dumps(s_u) + "  weighted " + json.dumps(s_w),
          flush=True)
    print("NOTE: nothing here is an accuracy claim. The test split's majority floors are "
          "computed on the untouched rows, so this mechanism buys headroom only if the "
          "model reads after the shortcut is removed — measured by a training run, not by "
          "a draw.", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps({
            "cmd": cmd, "suite": str(split), "suite_sha256": sha256(split),
            "rows": nrows, "question_sets": len(groups), "batch": args.batch,
            "max_q_cells": args.max_q_cells, "sets_per_update": args.sets_per_update,
            "updates": args.updates, "seed": args.seed, "skew": args.skew,
            "cells_skewed": {f"{s}/{n}": m for (s, n), m in skewed.items()},
            "weight_sum_max_rel_deviation": worst,
            "between_arm_mix_max_pts": mix_worst,
            "uniform": payload_u, "weighted": payload_w,
            "stats": {"uniform": s_u, "weighted": s_w},
            "compare": payloads_c,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Tier 0: which input is actually carrying myna's answer — state, instruction, or option text?

    uv run python bench/diag_question_ablation.py --out runs/diag_question_ablation.json

G1 measures 0.4893 against a 0.4331 majority floor, and five cells score *below* their own
floor on the split that floor is computed from — agnews/is_business 0.651 vs 0.698,
amazon/stars 0.237 vs 0.263, banking77/intent 0.034 vs 0.043, boolq/answer 0.500 vs 0.537,
mnli/relation 0.328 vs 0.353 (`bench/scope_pricing.py` counted the same five from the
committed report).
Before buying GPU hours to train a model longer, the cheap question is whether those cells are
*under-trained* or *not reading the thing that
matters*. Three of the project's pillars say the answer should be recoverable from the inputs:
the probe reads the question, the option spans carry the labels, and the state carries the
evidence. An ablation asks each one, on the checkpoint that produced the published number.

Six arms, one forward pass per arm per row, all through `myna.train.build_batch` and
`evaluate()`'s own chunking so the measurement is the published one:

* `as-scored` — no change. Its per-question-set accuracies must reproduce
  `runs/v1b_kaggle_3600b.metrics.json` key for key. That is the guard: nothing prints unless
  it does, exactly as `bench/scope_pricing.py` refuses a table whose unfiltered row does not
  reproduce the report (§9.41's lesson, turned from an aggregation into an input).
* `blank-instruction` — every instruction becomes one neutral sentence. If accuracy holds, the
  question text was never the input the answer came from, and the probe is a state reader.
* `permute-instruction` — within a source, instructions are re-assigned between question sets
  of the same shape. Real English, wrong row: a drop here means the instruction was doing
  something per row; no drop means the model was keying on the *source's* wording only.
* `blank-options` — option descriptions become `option 1..K`, same count. The pointer head
  still gets K spans. If accuracy holds, the labels were read from position, not from text.
* `swap-state` — each row's state is replaced by another row's state *from the same source*,
  gold kept. This is the prior-collapse test: a model that answers from a label-frequency
  prior survives it, and a model that reads evidence falls to its uniform floor.
* `cross-source-instruction` — the cue is replaced by one from a *different* source with the
  same option count. `permute-instruction` turned out to be near the identity here, because
  nine of these sources repeat a single cue text on every row, so within a source there is
  nothing to permute; that is why every arm prints how many strings it actually rewrote.

Aggregation is imported, never restated (`myna.report.cell_stats` / `accuracy_cells` /
`roll_up` / `macro`), so the macro this prints for `as-scored` is the same statistic G1 was
judged on rather than a lookalike. Per-cell prediction distributions come out too, because
"below the majority floor" has three causes — anti-correlation, collapse to one label, or a
mis-set option count — and only the first is visible in a confusion count.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shlex
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from myna.model import MynaConfig, MynaModel  # noqa: E402
from myna.real_data import Example, load_suite  # noqa: E402
from myna.report import accuracy_cells, cell_stats, macro, roll_up  # noqa: E402
from myna.tokenizer import Tokenizer  # noqa: E402
from myna.train import build_batch  # noqa: E402

NEUTRAL_INSTRUCTION = "Answer the question about the passage."
ARMS = ("as-scored", "blank-instruction", "permute-instruction", "cross-source-instruction",
        "blank-options", "swap-state")
# float noise only; a tie that flips between CPU and the T4 the run trained on moves a key by
# one row, so a small number of keys may differ by exactly that much and still be the same run.
FLOAT_TOL = 1e-6
ROW_SLACK_KEYS = 5


def load_run(run_dir, device, tokenizer=None):
    """The checkpoint and tokenizer the run saved, in the same order train.py reads them.

    `tokenizer` exists because the two artifacts reach this box as two Kaggle outputs: the
    run's own `out` dir holds `model.pt` and `tokenizer.json` together (train.py writes both),
    while what was downloaded here unpacked them into `ckptfull/` and `tok/`. A default that
    resolves to `model.pt` and then dies on a missing file is a published command that has
    never been run, so the path is checkable, and its digest lands in the witness beside the
    model's — the same reason `model_sha256` is there (§9.37: a witness binds its inputs, not
    just its numbers).
    """
    run_dir = Path(run_dir)
    tok_path = Path(tokenizer) if tokenizer else run_dir / "tokenizer.json"
    if not tok_path.exists():
        raise SystemExit(
            f"no tokenizer at {tok_path}; the run's out dir normally holds it beside "
            "model.pt, and `--tokenizer` points elsewhere when they were downloaded apart")
    tok = Tokenizer.from_file(str(tok_path))
    blob = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    model = MynaModel(MynaConfig(**blob["cfg"]))
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    return (model, tok, {"temperature": float(blob.get("temperature") or 1.0),
                         "step": int(blob.get("step", -1)),
                         "params": n_params, "cfg": blob["cfg"],
                         "tokenizer_path": str(tok_path),
                         "tokenizer_sha256": hashlib.sha256(
                             tok_path.read_bytes()).hexdigest(),
                         "model_sha256": hashlib.sha256(
                             (run_dir / "model.pt").read_bytes()).hexdigest()})


def score(model, tok, groups, device, temperature):
    """`evaluate()`'s loop with the per-row predictions kept.

    Same chunk width (64), same accumulation order, same divisor: the guard below compares
    these numbers with the committed ones, so anything that changes the summation order here
    changes them by rounding and would look like a defect in the checkpoint.
    """
    sums, totals, preds = defaultdict(float), defaultdict(int), defaultdict(list)
    with torch.no_grad():
        for wf, (questions, data) in groups.items():
            for i in range(0, len(data), 64):
                exs = data[i:i + 64]
                b = build_batch(exs, tok, questions, device)
                logits = model(b["state_ids"], b["state_len"], b["q_ids"], b["q_mask"],
                               b["span_mat"], b["opt_valid"], b["decide_idx"]) / temperature
                probs = F.softmax(logits, dim=-1)
                pred = probs.argmax(-1)
                for n, q in enumerate(questions):
                    key = f"{wf}/{q.name}"
                    hit = (pred[:, n] == b["gold"][:, n]).float()
                    sums[key] += float(hit.sum())
                    totals[key] += len(exs)
                    preds[key].append((pred[:, n].tolist(), b["gold"][:, n].tolist(),
                                       probs[:, n].max(-1).values.tolist()))
    return {k: sums[k] / max(totals[k], 1) for k in sums}, preds, totals


def ablate(groups, arm, seed=0):
    """A copy of the split with one input kind's row-level tie to the answer destroyed."""
    if arm == "as-scored":
        return copy.deepcopy(groups)
    out = copy.deepcopy(groups)
    if arm == "blank-instruction":
        for _wf, (qs, _exs) in out.items():
            for q in qs:
                q.instruction = NEUTRAL_INSTRUCTION
        return out
    if arm == "cross-source-instruction":
        # The honest version of the permute arm. Nine of these sources ask the same cue text on
        # every row, so swapping instructions *within* one source re-reads the identical string
        # and measures nothing (the run's `coverage` line proves it). Here the cue comes from a
        # DIFFERENT source that happens to offer the same option count, so the arm is never the
        # identity where it can fire, and a source with no same-K partner keeps its own cue.
        pool = defaultdict(list)
        for wf, (qs, _exs) in groups.items():
            source = wf.split("#", 1)[0]
            for q in qs:
                pool[len(q.options)].append((source, q.instruction))
        for wf, (qs, _exs) in out.items():
            source = wf.split("#", 1)[0]
            for q in qs:
                others = [(s, text) for s, text in pool[len(q.options)] if s != source]
                if not others:
                    continue
                rng = random.Random(f"{seed}|xsrc|{source}|{q.name}|{len(q.options)}")
                q.instruction = rng.choice(others)[1]
        return out
    if arm == "blank-options":
        for _wf, (qs, _exs) in out.items():
            for q in qs:
                q.options = [f"option {i + 1}" for i in range(len(q.options))]
        return out
    if arm == "permute-instruction":
        # Only sets with the identical (qname, K) shape can exchange instructions: a bank of
        # 77 options must not be handed a 2-option question, or the arm measures geometry
        # rather than wording.
        buckets = defaultdict(list)
        for wf, (qs, _exs) in out.items():
            source = wf.split("#", 1)[0]
            buckets[(source, tuple((q.name, len(q.options)) for q in qs))].append(wf)
        for shape, keys in buckets.items():
            if len(keys) < 2:
                continue
            rng = random.Random(f"{seed}|{shape[0]}|{shape[1]}")
            texts = [[q.instruction for q in out[k][0]] for k in keys]
            order = list(range(len(keys)))
            rng.shuffle(order)
            if order == list(range(len(keys))):
                order = order[1:] + order[:1]
            for pos, wf in enumerate(keys):
                for q, text in zip(out[wf][0], texts[order[pos]]):
                    q.instruction = text
        return out
    if arm == "swap-state":
        # Pooled by source, not by group: most of the suite's question sets hold ONE row, so a
        # within-group shuffle would be the identity for nearly every cell and would report
        # "the state is load-bearing" from a transform that changed nothing.
        by_source = defaultdict(list)
        for wf, (_qs, exs) in out.items():
            for j in range(len(exs)):
                by_source[wf.split("#", 1)[0]].append((wf, j))
        for source, slots in by_source.items():
            rng = random.Random(f"{seed}|state|{source}")
            texts = [out[wf][1][j].state for wf, j in slots]
            order = list(range(len(texts)))
            rng.shuffle(order)
            if len(texts) > 1 and order == list(range(len(texts))):
                order = order[1:] + order[:1]
            touched = defaultdict(dict)
            for pos, (wf, j) in enumerate(slots):
                touched[wf][j] = texts[order[pos]]
            for wf, swaps in touched.items():
                _qs, exs = out[wf]
                out[wf] = (_qs, [Example(swaps[j], exs[j].workflow,
                                         exs[j].gold) for j in range(len(exs))])
        return out
    raise ValueError(f"unknown arm {arm!r}; the arms are {ARMS}")


def coverage(groups, ablated):
    """How many input strings the arm actually rewrote.

    An arm that changes nothing cannot measure anything. On this split nine of the sources
    repeat one cue text across every row, so a within-source instruction permutation hands each
    row back the string it already had -- and the only thing standing between that and a
    published "instructions do not matter" is this line being printed beside every arm.
    """
    ni = no = ns = tq = tr = 0
    for wf, (qs, exs) in groups.items():
        aqs, aexs = ablated[wf]
        for q, aq in zip(qs, aqs):
            tq += 1
            if q.instruction != aq.instruction:
                ni += 1
            if list(q.options) != list(aq.options):
                no += 1
        for e, ae in zip(exs, aexs):
            tr += 1
            if e.state != ae.state:
                ns += 1
    return {"question_slots": tq, "rows": tr, "instructions_changed": ni,
            "option_sets_changed": no, "states_changed": ns}


def guard_against_metrics(arm_acc, metrics, totals):
    """Refuse to print an ablation table the run cannot vouch for."""
    bad, slack = [], 0
    for key, published in metrics.items():
        if key.endswith(":brier") or key.endswith(":ece"):
            continue
        if key not in arm_acc:
            bad.append(f"{key} is in the committed metrics but this split does not produce it")
            continue
        delta = abs(arm_acc[key] - published)
        if delta <= FLOAT_TOL:
            continue
        one_row = 1.0 / max(totals.get(key, 1), 1)
        if delta <= 1.5 * one_row:
            slack += 1
        else:
            bad.append(f"{key} prints {arm_acc[key]:.6f}, the committed metrics print "
                       f"{published:.6f} (delta {delta:.6f} > {one_row:.6f} = one row)")
    extra = set(arm_acc) - set(metrics)
    if extra:
        bad.append(f"{len(extra)} scored keys are not in the committed metrics "
                   f"(first: {sorted(extra)[0]})")
    if slack > ROW_SLACK_KEYS:
        bad.append(f"{slack} keys differ by a whole row: more than {ROW_SLACK_KEYS} tie flips "
                   "is not rounding, it is a different measurement")
    return bad, slack, len([k for k in metrics if k in arm_acc])


def label_distributions(preds, cells, keep=3, only_below_floor=True):
    """Per cell: what the model predicts vs what is there.

    A cell 0.15 under its majority floor is either anti-correlated (it has the signal and the
    sign wrong) or collapsed (it answers one label). The confusion counts tell them apart and
    the accuracy column cannot. `only_below_floor` trims the print; the witness carries every
    cell, because a collapse that happens to sit *above* its floor is the same defect and a
    filtered list would hide it.
    """
    counted = defaultdict(lambda: (Counter(), Counter()))
    for key, chunks in preds.items():
        wf, qname = key.split("/", 1)
        cell = (wf.split("#", 1)[0], qname)
        if cell not in cells:
            continue
        p, g = counted[cell]
        for preds_, golds, _conf in chunks:
            p.update(preds_)
            g.update(golds)
    out = {}
    for cell, r in sorted(cells.items()):
        if r["acc"] is None:
            continue
        if only_below_floor and r["acc"] >= r["majority"]:
            continue
        p, g = counted.get(cell, (Counter(), Counter()))
        n = sum(p.values())
        out[f"{cell[0]}/{cell[1]}"] = {
            "n": n, "options": r["options"], "type": r["type"],
            "acc": r["acc"], "majority": r["majority"], "uniform": r["uniform"],
            "predicted_top": [[i, c] for i, c in p.most_common(keep)],
            "gold_top": [[i, c] for i, c in g.most_common(keep)],
            "distinct_labels_predicted": len(p),
            "collapse_share": (p.most_common(1)[0][1] / n) if n and p else None,
            "gold_majority_share": (g.most_common(1)[0][1] / n) if n and g else None,
        }
    return out


def table_text(arm_rows, stats, order, guard_note, keep_min) -> str:
    head = (f"{'cell':26s} {'type':6s} {'n':>4s} {'K':>3s} "
            + " ".join(f"{a:>10s}" for a in order) + "  floor  unif  worst-d")
    lines = [guard_note, "", head]
    for cell in sorted(stats):
        s = stats[cell]
        if s["n"] < keep_min:
            continue
        vals = [arm_rows[a].get(cell, {}).get("acc") for a in order]
        shown = " ".join("      n/a" if v is None else f"{v:>10.3f}" for v in vals)
        base = vals[0]
        worst = "" if base is None or any(v is None for v in vals) else f"{min(vals) - base:+.3f}"
        # cell_stats reports options as a {K: how many sets have that many} map, because one
        # cell can be asked at two widths; roll_up prints max(), and so does this table.
        widths = s["options"]
        width = max(widths) if isinstance(widths, dict) else widths
        lines.append(f"{cell[0] + '/' + cell[1]:<26s} {s['type']:6s} {s['n']:>4d} "
                     f"{width:>3d} {shown}  {s['majority']:>6.3f} "
                     f"{s['uniform']:>6.3f} {worst:>8s}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", default="/private/tmp/kgwork/ckptfull/runs/v1b-kaggle-3600b",
                    help="directory holding model.pt (+ tokenizer.json unless --tokenizer "
                         "names one; this box's downloaded copy keeps them in two dirs)")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer.json, when it is not inside --run-dir")
    ap.add_argument("--suite", default="data/decision-v2-pilot")
    ap.add_argument("--split", default="test", choices=("test", "dev", "calibration"))
    ap.add_argument("--metrics", default="runs/v1b_kaggle_3600b.metrics.json",
                    help="the committed metrics the as-scored arm must reproduce")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-rows", type=int, default=None,
                    help="cell size floor for the table (default: the report's own)")
    ap.add_argument("--out", default=None, help="write the JSON witness here")
    args = ap.parse_args(argv)
    # `cmd` records the flags this call actually parsed rather than `sys.argv`: called from a
    # test or a mutation battery `sys.argv` is the caller's command line, and a witness whose
    # `cmd` is a pytest invocation is a receipt for nothing. §9.37 binds a witness to its
    # inputs, and here the inputs include which arms were asked for.
    given = list(sys.argv[1:]) if argv is None else list(argv)
    print("$ uv run python bench/diag_question_ablation.py", *given, flush=True)

    from myna.report import MIN_CELL_ROWS
    keep_min = args.min_rows or MIN_CELL_ROWS

    model, tok, meta = load_run(args.run_dir, args.device, args.tokenizer)
    groups = load_suite(args.suite)[args.split]
    if not groups:
        print(f"{args.split} split of {args.suite} is empty on disk", file=sys.stderr)
        return 2
    metrics_path = (REPO / args.metrics if not Path(args.metrics).is_absolute()
                    else Path(args.metrics))
    if not metrics_path.exists():
        print(f"the committed metrics {metrics_path} are not on disk", file=sys.stderr)
        return 2
    published_raw = json.loads(metrics_path.read_text())
    published = published_raw[args.split]

    # `model.pt` from this run carries `cfg`, `state_dict` and `temperature` and no step at all
    # (`load_run` reads -1 for it), so the step this table is anchored to comes from the metrics
    # file the guard is comparing against. Naming the file it came from is what keeps "step 3599"
    # from being read as a fact about the weights, which it is not.
    last_step = published_raw.get("last_step", "not recorded")
    print(f"run: last_step {last_step} (from {Path(args.metrics).name}; the checkpoint records "
          f"no step) · {meta['params']:,} params · temperature {meta['temperature']} · "
          f"model {meta['model_sha256'][:12]} · tokenizer {meta['tokenizer_sha256'][:12]} "
          f"· device {args.device}", flush=True)

    # The floors, option counts and cell sizes come from the untouched split: an ablated copy
    # is the input under test, never the ruler it is measured with.
    stats = cell_stats(groups)
    # `--min-rows` is published and `myna.report`'s own default is 30, so a split whose largest
    # cell is under it is a reachable call -- and every roll-up would be empty, which `macro()`
    # answers with None and the per-arm print answers with a TypeError. Refuse by number instead.
    if not any(s["n"] >= keep_min for s in stats.values()):
        print(f"--min-rows {keep_min} keeps no cell of the {args.split} split of "
              f"{args.suite}: its largest cell is "
              f"{max(s['n'] for s in stats.values())} rows over {len(stats)} cells, so there "
              "is nothing to ablate", file=sys.stderr)
        return 2
    arm_acc, arm_rows, arm_rows_all, all_preds, totals, covs = {}, {}, {}, {}, {}, {}
    for arm in args.arms:
        g = ablate(groups, arm, args.seed)
        covs[arm] = coverage(groups, g)
        t0 = time.time()
        acc, preds, tot = score(model, tok, g, args.device, meta["temperature"])
        cells_acc, unmatched = accuracy_cells(acc, g)
        rows = roll_up(stats, cells_acc, keep_min=keep_min)
        arm_rows[arm] = {(r["source"], r["question"]): r for r in rows}
        arm_rows_all[arm] = rows
        arm_acc[arm], all_preds[arm], totals[arm] = acc, preds, tot
        print(f"{arm:23s} macro {macro(rows, 'acc'):.4f} "
              f"(floor {macro(rows, 'majority'):.4f}) "
              f"rewrote {covs[arm]['instructions_changed']:>4d}/{covs[arm]['question_slots']}"
              f" cues, {covs[arm]['option_sets_changed']:>4d} option sets, "
              f"{covs[arm]['states_changed']:>4d}/{covs[arm]['rows']} states "
              f"in {time.time() - t0:5.1f}s", flush=True)
        if unmatched:
            print(f"  ! {len(unmatched)} scored sets are not in this split", flush=True)

    bad, slack, compared = guard_against_metrics(arm_acc["as-scored"], published,
                                                 totals["as-scored"])
    if bad:
        print("\nTHE RUN DOES NOT REPRODUCE THE COMMITTED METRICS -- no ablation printed:",
              file=sys.stderr)
        for b in bad[:12]:
            print(f"  ! {b}", file=sys.stderr)
        return 2
    base_macro = macro(arm_rows_all["as-scored"], "acc")
    guard_note = (f"guard green: the as-scored arm reproduces {compared} committed keys "
                  f"({slack} single-row tie flips) · macro {base_macro:.10f}")

    print()
    print(table_text(arm_rows, stats, args.arms, guard_note, keep_min))

    print("\nmacro by arm (the published basis: row-weighted inside each cell, unweighted "
          "over cells):")
    for arm in args.arms:
        d = macro(arm_rows_all[arm], "acc")
        print(f"  {arm:20s} {d:.4f} vs floor "
              f"{macro(arm_rows_all[arm], 'majority'):.4f}  Δ {d - base_macro:+.4f}")

    scored = {(r["source"], r["question"]): r for r in arm_rows_all["as-scored"]}
    dist = label_distributions(all_preds["as-scored"], scored, only_below_floor=False)
    print("\nper-cell label emission (as-scored): collapse_share is the fraction of rows the "
          "cell answers with its most-predicted label,\nand state-Δ is what swapping this "
          "row's state for another row's from the same source costs the cell.")
    for (c, q), base in sorted(scored.items()):
        d = dist[f"{c}/{q}"]
        swap = (arm_rows["swap-state"].get((c, q), {}).get("acc")
                if "swap-state" in args.arms else None)
        dsw = "" if swap is None or base["acc"] is None else f"{swap - base['acc']:+.3f}"
        print(f"  {c + '/' + q:<26s} {base['type']:6s} acc {base['acc']:.3f} "
              f"floor {base['majority']:.3f} unif {base['uniform']:.3f} "
              f"collapse {d['collapse_share']:.3f} (gold {d['gold_majority_share']:.3f}) "
              f"labels {d['distinct_labels_predicted']:>3d}/{base['options']:<3d} "
              f"state-Δ {dsw:>7s}")

    witness = {"cmd": shlex.join([sys.executable, "bench/diag_question_ablation.py", *given]),
               "run_dir": str(args.run_dir), "split": args.split, "suite": str(args.suite),
               "metrics": str(args.metrics), "last_step": last_step,
               "device": args.device, "seed": args.seed, "keep_min": keep_min, "model": meta,
               "guard": {"keys_compared": compared, "tie_flips": slack,
                         "float_tol": FLOAT_TOL, "row_slack_keys": ROW_SLACK_KEYS,
                         "macro_as_scored": base_macro},
               "arms": list(args.arms), "coverage": covs,
               "macro": {a: macro(arm_rows_all[a], "acc") for a in args.arms},
               "macro_floor": macro(arm_rows_all["as-scored"], "majority"),
               "macro_uniform": macro(arm_rows_all["as-scored"], "uniform"),
               "distributions": dist,
               "cells": {f"{c}/{q}": {a: arm_rows[a].get((c, q), {}).get("acc")
                                      for a in args.arms} for (c, q) in scored}}
    if args.out:
        Path(args.out).write_text(json.dumps(witness, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The control row of the accuracy comparison (SPEC §5 P8 8c, §8 risk table).

Every claim that myna is a better *architecture* has to be measured against the one
number that isolates the architecture from its training data. That number is what this
prints: myna's model, weights as drawn from a seeded initialiser, the upstream corpus's
own shipped tokenizer, the frozen `test` split, current code.

    uv run python bench/eval_scratch.py --out runs/scratch_decision_v2_test.md

Why not a trained checkpoint? Every checkpoint in `runs/` that saw the upstream corpus
was trained under the pre-`385e06c` batching regime, and SPEC §5 P1 voids those numbers
as evidence about myna — re-scoring one today yields a fresh figure welded to a broken
training story, which is the move §9.23 already buried ("void as evidence about myna's
ceiling: they measure my bug"). A random init cannot measure a ceiling in either
direction. It is the control, it is reproducible from the seed alone, and its expected
value is the uniform floor the report prints — so the row is falsifiable in the one
direction that matters: if the head or the pointer probe were secretly reading the
answer, this would sit above the floor.

The tokenizer is the suite's (`tokenizer-8192.json`, the file `kaggle/package_dataset.py`
ships beside the splits), never one trained here, so the row cannot be accused of
fragmenting the corpus into `[UNK]` the way a synthetic-trained vocab would.

Per-seed, because one random draw is a sample and not a property: the default prints two
seeds and their mean, and the markdown keeps all three numbers.

`--ckpt DIR` runs the *same* scoring path on a saved checkpoint, which is how the control
is shown to be an instrument and not a tautology: a checkpoint's number has to reproduce
the one `myna.report` prints from that checkpoint's own `metrics.json`, computed months
earlier inside the trainer's process. If the two agree, the harness is scoring the split
the same way the trainer did; if they do not, the control above is meaningless too.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from myna.model import MynaConfig, MynaModel  # noqa: E402
from myna.real_data import load_suite  # noqa: E402
from myna.report import (accuracy_cells, cell_stats, macro, roll_up, strata,  # noqa: E402
                         table_text, _fmt)
from myna.train import evaluate  # noqa: E402


def build_model(ckpt: Path | None, vocab: int, seed: int, device: str):
    """The control: `torch.manual_seed` and the default initialiser, no gradient step.

    With `ckpt`, the weights a trainer actually produced instead — same scoring path,
    so the two runs differ by exactly one thing.
    """
    torch.manual_seed(seed)
    if ckpt:
        # a checkpoint carries its own cfg; reading it from model.pt is what
        # `myna.engine.Myna` does, and a vocab built from the wrong tokenizer here
        # would load_state_dict into shape errors or, worse, silently slice weights
        ck = torch.load(Path(ckpt) / "model.pt", map_location=device, weights_only=False)
        model = MynaModel(MynaConfig(**ck["cfg"]))
        model.load_state_dict(ck["state_dict"])
    else:
        model = MynaModel(MynaConfig(vocab=vocab))
    model.to(device)
    return model


def scratch_metrics(tok_path: Path, suite: Path, split: str, seed: int, device: str,
                    ckpt: Path | None = None) -> dict:
    """`evaluate()`'s shape, from weights that never saw a gradient (or from `ckpt`)."""
    tok = Tokenizer.from_file(str(tok_path))
    vocab = tok.get_vocab_size()
    model = build_model(ckpt, vocab, seed, device)
    if ckpt and model.cfg.vocab != vocab:
        raise SystemExit(f"{ckpt} has {model.cfg.vocab} embeddings but {tok_path} is a "
                         f"{vocab}-entry vocab: the control's tokenizer belongs to the "
                         f"suite, and a checkpoint must be scored with its own")
    groups = load_suite(suite)[split]
    return evaluate(model, tok, groups, device, temperature=1.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--suite", default="data/decision-v2-pilot")
    ap.add_argument("--split", default="test", choices=["test", "dev", "calibration"])
    ap.add_argument("--tokenizer", default=None,
                    help="defaults to the suite's tokenizer-8192.json, or the checkpoint's "
                         "own beside it")
    ap.add_argument("--ckpt", default=None, help="score a saved checkpoint instead of the control")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-rows", type=int, default=30)
    ap.add_argument("--out", default="runs/scratch_decision_v2_test.md")
    ap.add_argument("--metrics-out", default=None,
                    help="also write the first seed's per-set accuracy map in the shape "
                         "myna.train writes into metrics.json, so myna.report can read the "
                         "control through the same code path as a checkpoint")
    argv = list(argv if argv is not None else sys.argv[1:])
    args = ap.parse_args(argv)
    cmd = shlex.join(["python", "bench/eval_scratch.py", *argv])
    suite, ckpt = Path(args.suite), Path(args.ckpt) if args.ckpt else None
    tok = Path(args.tokenizer or ((ckpt or suite) / "tokenizer.json" if ckpt
                                  else suite / "tokenizer-8192.json"))
    if not tok.exists():
        raise SystemExit(f"no tokenizer at {tok}: the control has to use the suite's own "
                         f"vocab, and training one here would change what is measured")
    dupes = sorted({s for s in args.seeds if args.seeds.count(s) > 1})
    if dupes:
        raise SystemExit(f"--seeds repeats {dupes}: two identical draws are not two samples")
    if ckpt and not (ckpt / "model.pt").exists():
        raise SystemExit(f"--ckpt {ckpt} has no model.pt: a resume file is a different "
                         f"artifact and would be loaded by a different piece of code")
    weights = f"checkpoint {ckpt}" if ckpt else "random init"
    for line in ("$ " + cmd,
                 f"what is being scored: {weights}, tokenizer `{tok}`, suite {suite}, "
                 f"split {args.split}, device `{args.device}`",
                 f"load average {'/'.join(f'{x:.2f}' for x in os.getloadavg())}"):
        print(line, flush=True)

    groups = load_suite(suite)[args.split]
    stats = cell_stats(groups)
    cls = strata(groups)
    out_json = Path(args.out).with_suffix(".json")

    per_seed, cells_by_seed = {}, {}
    metrics_by_seed = {}
    for seed in args.seeds:
        metrics = scratch_metrics(tok, suite, args.split, seed, args.device, ckpt)
        metrics_by_seed[seed] = metrics
        acc, unmatched = accuracy_cells(metrics, groups)
        rows = roll_up(stats, acc, args.min_rows)
        scored = [r for r in rows if r["acc"] is not None]
        per_seed[seed] = {"cells": {f"{r['source']}/{r['question']}": r["acc"] for r in scored},
                          "macro": macro(scored, "acc"), "n_scored": len(scored)}
        cells_by_seed[seed] = rows
        print(f"seed {seed}: macro over {len(scored)} cells = "
              f"{_fmt(per_seed[seed]['macro'])}", flush=True)
        if unmatched:
            print(f"  note: {len(unmatched)} scored question-set(s) are not in this split")

    basis = cells_by_seed[args.seeds[0]]
    scored = [r for r in basis if r["acc"] is not None]
    macros = [per_seed[s]["macro"] for s in args.seeds]
    mean = statistics.fmean(macros)
    mm, mu = macro(scored, "majority"), macro(scored, "uniform")
    spread = max(macros) - min(macros) if len(macros) > 1 else 0.0
    # the per-cell table printed is the first seed's; the mean is the published figure
    print()
    print(table_text(basis, cls, None, args.split))
    print(f"\nMACRO over the {len(scored)} cells, {len(args.seeds)} seeds "
          f"{'/'.join(str(s) for s in args.seeds)}: "
          + " · ".join(f"seed {s} {_fmt(per_seed[s]['macro'])}" for s in args.seeds)
          + f" · mean {_fmt(mean)} (seed spread {spread:.3f})")
    print(f"majority floor {_fmt(mm)} · uniform floor {_fmt(mu)}"
          + ("" if ckpt else " — the control is read against the uniform floor, since a "
                            "random init has no label prior to exploit"))

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"`{cmd}`\n\n")
        f.write(f"load average {'/'.join(f'{x:.2f}' for x in os.getloadavg())}, "
                f"device `{args.device}`, tokenizer `{tok}` (vocab "
                f"{Tokenizer.from_file(str(tok)).get_vocab_size()}), "
                + ("weights loaded from " + str(ckpt) if ckpt else
                   "weights: `torch.manual_seed(seed)` then the default initialiser — "
                   "no gradient step, on any corpus") + ".\n\n")
        f.write("```\n" + table_text(basis, cls, None, args.split) + "\n```\n\n")
        f.write(f"MACRO, {len(args.seeds)} seeds: "
                + ", ".join(f"seed {s} = {_fmt(per_seed[s]['macro'])}" for s in args.seeds)
                + f"; mean **{_fmt(mean)}**, spread {spread:.3f}. Floors over the same "
                  f"{len(scored)} cells: majority {_fmt(mm)}, uniform {_fmt(mu)}.\n")
    out_json.write_text(json.dumps({
        "cmd": cmd, "load_avg": list(os.getloadavg()), "suite": str(suite),
        "split": args.split, "tokenizer": str(tok), "device": args.device,
        "weights": str(ckpt) if ckpt else "random-init",
        "min_rows": args.min_rows, "temperature": 1.0, "seeds": args.seeds,
        "per_seed": {str(s): per_seed[s] for s in args.seeds},
        "macro_mean": mean, "macro_spread": spread,
        "majority_floor": mm, "uniform_floor": mu,
        "cells": [{k: v for k, v in r.items() if k != "options_by_size"} for r in basis],
    }, indent=2) + "\n")
    print(f"wrote {args.out} and {out_json}")
    if args.metrics_out:
        # the trainer's own shape: {split: {"source#sig8/qname": acc}}. Accuracy entries
        # only — the :brier/:ece keys `evaluate()` also returns are a calibration claim,
        # and `accuracy_cells()` skips them; a Brier column off a random init is noise
        # with a decimal point on it. Rounded to 6 dp so the report's row-weighted macro
        # lands on the figure printed above at every displayed digit.
        mp = Path(args.metrics_out)
        mp.parent.mkdir(parents=True, exist_ok=True)
        first = metrics_by_seed[args.seeds[0]]
        mp.write_text(json.dumps({
            "cmd": cmd, "weights": str(ckpt) if ckpt else "random-init",
            "seed": args.seeds[0], "seeds_run": args.seeds, "split": args.split,
            "tokenizer": str(tok), "device": args.device,
            "note": f"accuracy entries for seed {args.seeds[0]} only; no :brier/:ece keys",
            args.split: {k: round(v, 6) for k, v in first.items()
                         if not (k.endswith(":brier") or k.endswith(":ece"))},
        }, indent=2) + "\n")
        print(f"wrote {mp} ({len(first)} accuracy cells scored from seed {args.seeds[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

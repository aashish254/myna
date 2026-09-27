"""V1-D acceptance eval: needle-in-a-haystack recall vs observation length.

Loads a checkpoint through the streaming engine and measures choice accuracy
when the single decisive sentence is buried in filler of increasing length. A
model that reads its state keeps accuracy flat out to 4k+; one that has lost
the needle decays toward chance. Also reports per-length observe+ask latency.

The ladder is `--lengths`, defaulting to G4's four rungs (1k/4k/8k/16k): 16,384
is the length §2.1 claims the window reaches, so the claim is only worth
publishing if this table has a row for it.

Usage: uv run python -m bench.eval_needle --ckpt runs/myna-v1 [--device mps]
       uv run python -m bench.eval_needle --ckpt runs/myna-v0 --lengths 128 512 1024 4096 8192 16384
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from myna.engine import Myna
from myna.longctx import DESKS, eval_lengths

#: G4's pass condition is stated over these four rungs (§2.2).
LENGTHS = [1024, 4096, 8192, 16384]


def baseline_verdict(rows: list[dict], n_per_length: int, chance: float,
                     margin: float = 0.10) -> dict:
    """Is the shortest rung readable at all?

    A recall-vs-length curve is a statement about *decay*, and decay is only
    measurable if the model can answer the question when the state is short. If the
    128-token rung sits at the uniform floor, every longer rung inherits that
    result, and "0.06 at 16k" is not a long-context finding — it is an
    out-of-distribution task wearing one. This is the difference between a flat
    curve and a low one, and the table cannot tell them apart by itself."""
    first = rows[0]
    se = (chance * (1 - chance) / max(n_per_length, 1)) ** 0.5
    readable = first["acc"] >= chance + margin
    return {"shortest_rung": first["tokens"], "acc": first["acc"], "chance": chance,
            "n": n_per_length, "one_se": round(se, 4), "margin": margin,
            "readable": bool(readable),
            "note": (None if readable else
                     f"the {first['tokens']}-token rung is at the {chance:.3f} uniform floor, "
                     f"so this curve cannot be read as decay — the checkpoint does not "
                     f"answer this task at any length")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/myna-v1")
    ap.add_argument("--device", default="cpu", help="cpu is fine and avoids MPS contention")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lengths", type=int, nargs="+", default=LENGTHS, metavar="TOKENS",
                    help="state lengths to sweep; each one is a row of the curve")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    # §9.30: --lengths 128 is a rung the defaults never sweep, and --n 16 is not
    # --n 40, so a curve quoted without its command is a curve nobody can re-draw.
    flags = list(argv if argv is not None else sys.argv[1:])
    cmd = shlex.join(["python", f"bench/{Path(__file__).name}", *flags])
    if not args.lengths or min(args.lengths) < 1:
        raise SystemExit(f"--lengths must be positive token counts, got {args.lengths}")
    if args.n < 1:
        raise SystemExit(f"--n must be at least 1 needle per rung, got {args.n}")

    myna = Myna(args.ckpt, device=args.device)
    rows = eval_lengths(myna, args.lengths, n_per_length=args.n, seed=args.seed)
    v = baseline_verdict(rows, args.n, 1.0 / len(DESKS))

    name = Path(args.ckpt).name
    print(f"needle recall vs context length — {name} ({args.device})\n")
    print(f"{'tokens':>8} | {'acc':>6} | {'ms':>8}")
    for r in rows:
        print(f"{r['tokens']:>8} | {r['acc']:>6.3f} | {r['ms']:>8.2f}")
    print(f"\nuniform floor {v['chance']:.3f} ({len(DESKS)} desks), 1 s.e. at n={args.n} "
          f"is {v['one_se']:.3f}")
    if v["readable"]:
        print(f"G4: this curve CAN be read as decay — the {v['shortest_rung']}-token rung "
              f"clears the floor by {v['acc'] - v['chance']:+.3f}")
    else:
        print(f"G4: NOT MEASURED here — {v['note']}")

    out = args.out or f"runs/needle_{name}.md"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# Needle recall vs context length — {name}\n\n")
        f.write(f"`{cmd}`\n\n")
        f.write(f"{args.n} needles per length, decisive sentence at a random position, "
                f"seed {args.seed} on `{args.device}`. Flat accuracy = the trunk still reads "
                f"the whole observation. The `ms` column is a mean over these samples on a "
                f"shared-core laptop, so it is not a G2 measurement: SPEC §9.23 quotes ratios "
                f"from the matched harness, never absolutes like these.\n\n")
        f.write("| state tokens | needle accuracy | observe+ask ms |\n|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['tokens']} | {r['acc']:.3f} | {r['ms']:.2f} |\n")
        f.write(f"\nUniform floor {v['chance']:.3f} ({len(DESKS)} desks); 1 s.e. at n={args.n} "
                f"is {v['one_se']:.3f}. ")
        f.write(f"G4: this curve **can** be read as decay — the {v['shortest_rung']}-token "
                f"rung clears the floor by {v['acc'] - v['chance']:+.3f}."
                if v["readable"] else
                f"G4: **not measured by this run** — {v['note']}.")
        f.write("\n")
    # G7 commits the witness, not just the rendering: the numbers a reader audits
    # are the ones the run produced, in the file the run wrote.
    js = Path(str(out).rsplit(".", 1)[0] + ".json")
    js.write_text(json.dumps({"cmd": cmd,
                             "ckpt": str(args.ckpt), "device": args.device, "seed": args.seed,
                             "n_per_length": args.n, "lengths": args.lengths,
                             "params_m": round(myna.n_params / 1e6, 2),
                             "temperature": myna.temperature,
                             "n_options": len(DESKS), "baseline": v,
                             "rows": rows}, indent=2) + "\n")
    print(f"\nwrote {out} and {js}")


if __name__ == "__main__":
    main()

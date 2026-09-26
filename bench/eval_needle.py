"""V1-D acceptance eval: needle-in-a-haystack recall vs observation length.

Loads a checkpoint through the streaming engine and measures choice accuracy
when the single decisive sentence is buried in filler of increasing length. A
model that reads its state keeps accuracy flat out to 4k+; one that has lost
the needle decays toward chance. Also reports per-length observe+ask latency.

Usage: uv run python -m bench.eval_needle --ckpt runs/myna-v1 [--device mps]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from myna.engine import Myna
from myna.longctx import eval_lengths

LENGTHS = [128, 512, 1024, 2048, 4096, 8192]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/myna-v1")
    ap.add_argument("--device", default="cpu", help="cpu is fine and avoids MPS contention")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    myna = Myna(args.ckpt, device=args.device)
    rows = eval_lengths(myna, LENGTHS, n_per_length=args.n, seed=args.seed)

    name = Path(args.ckpt).name
    print(f"needle recall vs context length — {name} ({args.device})\n")
    print(f"{'tokens':>8} | {'acc':>6} | {'ms':>8}")
    for r in rows:
        print(f"{r['tokens']:>8} | {r['acc']:>6.3f} | {r['ms']:>8.2f}")

    out = args.out or f"runs/needle_{name}.md"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# Needle recall vs context length — {name}\n\n")
        f.write(f"{args.n} needles per length, decisive sentence at a random position. "
                f"Flat accuracy = the trunk still reads the whole observation.\n\n")
        f.write("| state tokens | needle accuracy | observe+ask ms |\n|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['tokens']} | {r['acc']:.3f} | {r['ms']:.2f} |\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

"""Peak-memory profile of a training step (SPEC P2's third box).

The plan assumed the pointer head's option pooling — [B, N, O, d] — was what
filled the M5. Measured at the shipped size, it is not. Retained activations per
unit of work, from the eight-config grid below: 1.6 MiB per question-token
position (rows x questions x tokens, the scan's [H, chunk, chunk, d_k] terms),
0.8 MiB per state-token position (one scan instead of two), and 3 KiB per
option cell — a factor of ~500. `draw_row_batch` budgets the first quantity:
questions and question tokens both pad to the batch maximum, so one long
instruction charges the whole batch for itself.

Each config runs in its own subprocess because `ru_maxrss` is a high-water mark
that never falls.

    PYTHONPATH=src .venv/bin/python -m bench.mem_profile [--device cpu|mps] [--batch 8]
"""

from __future__ import annotations

import argparse
import resource
import subprocess
import sys
import time

import torch

from myna.model import MynaConfig, MynaModel, typed_loss

# state lengths: pilot median 49, p95 263, max 402 (measured, upstream_manifest).
# option counts: 2 (noul), 14 (dbpedia14), 77 (banking77 - the widest set in the suite).
# The axes vary one at a time so the table says *which* term prices the step.
GRID = [
    dict(Ls=48, N=1, O=2, Lq=32),
    dict(Ls=48, N=3, O=2, Lq=32),
    dict(Ls=48, N=3, O=14, Lq=32),
    dict(Ls=48, N=3, O=77, Lq=32),
    dict(Ls=48, N=3, O=14, Lq=96),
    dict(Ls=48, N=12, O=14, Lq=32),
    dict(Ls=402, N=3, O=14, Lq=32),
    dict(Ls=402, N=3, O=77, Lq=96),
]


def peak_bytes() -> int:
    """Resident set high-water mark. `ru_maxrss` never goes down, so each config
    is measured in its own subprocess and the number reported is that
    process's true peak: model + optimizer state + one step's autograd graph.
    macOS reports bytes, Linux kilobytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def make_batch(cfg: MynaConfig, dev, rng, **s):
    """Built on CPU then moved: the same generator seeds both devices, and the
    random content is irrelevant — only the shapes cost memory."""
    b = {
        "state_ids": torch.randint(1, cfg.vocab, (s["B"], s["Ls"]), generator=rng),
        "state_len": torch.full((s["B"],), s["Ls"], dtype=torch.int64),
        "q_ids": torch.randint(1, cfg.vocab, (s["B"], s["N"], s["Lq"]), generator=rng),
        "q_mask": torch.ones(s["B"], s["N"], s["Lq"]).index_fill_(2, torch.arange(s["Lq"] - 3, s["Lq"]), 0.0),
        "span_mat": torch.zeros(s["B"], s["N"], s["O"], s["Lq"]).index_fill_(
            3, torch.arange(s["Lq"] - 6), 1.0 / (s["Lq"] - 6)),
        "opt_valid": torch.ones(s["B"], s["N"], s["O"], dtype=torch.bool),
        "decide_idx": torch.full((s["B"], s["N"]), s["Lq"] - 4, dtype=torch.int64),
        "gold": torch.randint(0, s["O"], (s["B"], s["N"]), generator=rng),
        "has_gold": torch.ones(s["B"], s["N"], dtype=torch.bool),
    }
    return {k: v.to(dev) for k, v in b.items()}


def run_one(cfg: MynaConfig, dev: str, s: dict) -> None:
    model = MynaModel(cfg).to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    rng = torch.Generator().manual_seed(0)
    b = make_batch(cfg, dev, rng, **s)
    base = peak_bytes()
    t0 = time.time()
    logits = model(b["state_ids"], b["state_len"], b["q_ids"], b["q_mask"],
                   b["span_mat"], b["opt_valid"], b["decide_idx"])
    loss = typed_loss(logits, b["gold"], b["has_gold"])
    loss.backward()
    opt.step()
    dt = time.time() - t0
    step = (peak_bytes() - base) / 2**20  # the batch's own cost, over weights + Adam
    pooled = s["B"] * s["N"] * s["O"] * cfg.d_model * 4 / 2**20
    print(f"{s['B']:>4} {s['Ls']:>5} {s['N']:>3} {s['O']:>4} {s['Lq']:>4} "
          f"{step:>9.0f} {pooled:>8.1f} {s['B'] * s['N'] * s['Lq']:>9,} {dt:>7.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--only", type=int, default=None, help="internal: run one grid index and exit")
    args = ap.parse_args()
    dev = args.device
    if dev == "mps" and not torch.backends.mps.is_available():
        sys.exit("mps unavailable on this machine")

    cfg = MynaConfig(vocab=8192, d_model=args.d_model)
    n_params = sum(p.numel() for p in MynaModel(cfg).parameters())
    if args.only is not None:
        run_one(cfg, dev, dict(GRID[args.only], B=args.batch))
        return

    print(f"device={dev}  vocab={cfg.vocab} d_model={cfg.d_model}  "
          f"params={n_params:,} ({n_params * 12 / 2**20:.0f} MiB for fp32 weights + grads + Adam)")
    print(f"{'B':>4} {'Ls':>5} {'N':>3} {'O':>4} {'Lq':>4} {'step MiB':>9} "
          f"{'pool MiB':>9} {'B*N*Lq':>9} {'sec':>6}")
    for i, s in enumerate(GRID):
        subprocess.run([sys.executable, "-m", "bench.mem_profile", "--only", str(i),
                        "--device", dev, "--d-model", str(cfg.d_model),
                        "--batch", str(args.batch)], check=True)


if __name__ == "__main__":
    main()

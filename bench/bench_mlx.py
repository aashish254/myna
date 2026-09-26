"""V1-C latency benchmark: MLX (Metal) vs PyTorch (MPS) full-predict path.

The delta-streaming claim is already carried by bench_stream (torch). This one
measures the *serving* payoff of the MLX port: the same checkpoint, the same
questions, a state of matched token length, timed end-to-end on the two
inference stacks. Lower is better; the point is MLX answers faster and with no
torch dependency on consumer Apple silicon.

Usage: uv run python -m bench.bench_mlx --ckpt runs/myna-v0 [--reps 20]
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from myna.model import MynaConfig, MynaModel
from myna.mlx_model import MynaMLX
from myna.tokenizer import Tokenizer, encode_text, question_tensors

QUESTIONS = [
    ("Which team should this route to?", ["billing", "technical support", "sales", "abuse"]),
    ("Is this urgent?", ["No", "Yes"]),
]
LENS = [128, 512, 1024, 2048, 4096, 8192, 16384]


def build_state(tok, target):
    seed = encode_text(tok, "customer reports a recurring payment failure on the pro plan since monday")
    ids = []
    while len(ids) < target:
        ids += seed
    return ids[:target]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/myna-v0")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="runs/bench_mlx.md")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt)
    tok = Tokenizer.from_file(str(ckpt_dir / "tokenizer.json"))
    ck = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
    cfg = MynaConfig(**ck["cfg"])

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tmodel = MynaModel(cfg)
    tmodel.load_state_dict(ck["state_dict"])
    tmodel.to(dev).eval()
    mlx = MynaMLX.from_checkpoint(ckpt_dir / "model.pt")
    n_params = sum(p.numel() for p in tmodel.parameters())

    qt = question_tensors(tok, QUESTIONS)
    q_np = {k: v.numpy() for k, v in qt.items()}

    def t_forward(state_ids):
        Ls = len(state_ids)
        s = torch.tensor([state_ids], dtype=torch.int64, device=dev)
        ln = torch.tensor([Ls], dtype=torch.int64, device=dev)
        qi = qt["q_ids"].to(dev); qm = qt["q_mask"].to(dev)
        sm = qt["span_mat"].to(dev); ov = qt["opt_valid"].to(dev); di = qt["decide_idx"].to(dev)
        with torch.no_grad():
            tmodel(s, ln, qi, qm, sm, ov, di)
        if dev == "mps":
            torch.mps.synchronize()

    def m_forward(state_ids):
        Ls = len(state_ids)
        mlx.forward(np.array([state_ids], dtype=np.int64), np.array([Ls], dtype=np.int64),
                    q_np["q_ids"], q_np["q_mask"], q_np["span_mat"], q_np["opt_valid"], q_np["decide_idx"])

    print(f"ckpt {ckpt_dir}  params {n_params/1e6:.1f}M  device torch={dev} mlx=metal\n")
    print(f"{'state tokens':>13} | {'torch ms':>9} | {'MLX ms':>9} | {'speedup':>7}")
    rows = []
    for L in LENS:
        ids = build_state(tok, L)
        # warmup
        t_forward(ids); m_forward(ids)
        tt = []
        for _ in range(args.reps):
            t0 = time.perf_counter(); t_forward(ids); tt.append((time.perf_counter() - t0) * 1000)
        tm = statistics.median(tt)
        mt = []
        for _ in range(args.reps):
            t0 = time.perf_counter(); m_forward(ids); mt.append((time.perf_counter() - t0) * 1000)
        mm = statistics.median(mt)
        print(f"{L:>13} | {tm:>9.2f} | {mm:>9.2f} | {tm/mm:>6.2f}x")
        rows.append((L, tm, mm))

    with open(args.out, "w") as f:
        f.write(f"# MLX vs PyTorch(MPS) full-predict latency — {ckpt_dir}\n\n")
        f.write(f"Model {n_params/1e6:.1f}M params, {QUESTIONS.__len__()} questions/request, "
                f"median of {args.reps} reps, same M5 box.\n\n")
        f.write("| state tokens | torch (MPS) ms | MLX (Metal) ms | speedup |\n|---|---|---|---|\n")
        for L, tm, mm in rows:
            f.write(f"| {L} | {tm:.2f} | {mm:.2f} | {tm/mm:.2f}x |\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Reference training implementation for Myna v6 (packed-readout arm).

The shipped checkpoint was trained in a Colab runtime whose notebook is not in the archive.
This script is a *reference* implementation of the recipe the paper describes, not a replay
of that run: it builds the same packed sequences with the engine's own renderer, trains the
same packed-scorer head on the same suite, and writes a checkpoint directory that
`myna.v6_engine.MynaV6` loads with `strict=True`. Its purpose is to make the method
reproducible and auditable; it does not claim to reproduce the shipped weights.

    python release/train_v6.py --steps 400 --out runs/v6_reference
    python release/train_v6.py --steps 2 --limit 8 --out /tmp/v6_smoke --no-encoder

Design notes
------------
* The renderer is the engine's: `Renderer` rebinds the tokenizer-only methods of `MynaV6`
  (`_prefix`, `_options`, `_labels`, `_special`, `_packed_ids`) instead of re-implementing
  them. A reference trainer that renders its inputs differently from the engine trains a
  different task, and this is the one place where a silent divergence is most likely.
* Only the packed arm is trained here (`K <= 16`). The decoupled high-cardinality head is
  a separate stage that needs per-option batches; the reference run reports how many
  questions it skipped for that reason rather than folding them into the packed loss.
* Calibration (`--fit-calibration`) fits one temperature per primitive on the calibration
  split by minimizing negative log-likelihood over a fixed grid, then stores the 90% and
  95% conformal quantiles of the nonconformity score `1 - p(gold)`. That is the recipe the
  engine reads from `config`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


class Renderer:
    """Tokenizer-only view of `MynaV6`, so training renders exactly what serving renders."""

    _prefix = None
    _options = None
    _labels = None
    _special = None
    _packed_ids = None

    def __init__(self, tokenizer, max_length: int = 512):
        from myna.v6_engine import MynaV6

        self.tok = tokenizer
        self.max_length = max_length
        self._prefix = MynaV6._prefix.__get__(self)
        self._options = MynaV6._options.__get__(self)
        self._labels = MynaV6._labels   # a staticmethod; binding it would pass self
        self._special = MynaV6._special.__get__(self)
        self._packed_ids = MynaV6._packed_ids.__get__(self)


def gold_index(qtype: str, label, labels: list[str]) -> int:
    """The gold option index, in the engine's own label space."""
    if qtype == "noul":
        return 1 if label else 0
    if qtype == "score":
        return int(label) - 1
    return labels.index(label)


class PackedSuite(Dataset):
    def __init__(self, path: Path, renderer: Renderer, limit: int = 0, max_k: int = 16):
        from myna.real_data import _state_text

        self.items: list[dict] = []
        self.skipped_high_k = 0
        self.rows = 0
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        if limit:
            lines = lines[:limit]
        for line in lines:
            row = json.loads(line)
            self.rows += 1
            source = row["_meta"]["source"]
            state = _state_text(row["state"])
            for qname, spec in row["questions"].items():
                options = renderer._options(spec)
                if len(options) > max_k:
                    self.skipped_high_k += 1
                    continue
                labels = renderer._labels(spec)
                prefix = renderer._prefix(f"{source}/{qname}", spec)
                seq, positions = renderer._packed_ids(prefix, state, options)
                if not positions:
                    continue
                self.items.append({
                    "seq": seq, "positions": positions,
                    "gold": gold_index(spec["type"], spec.get("label"), labels),
                    "primitive": spec["type"],
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch: list[dict], pad_id: int = 0):
    width = max(len(b["seq"]) for b in batch)
    k = max(len(b["positions"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), width), dtype=torch.long)
    pos = torch.zeros((len(batch), k), dtype=torch.long)
    valid = torch.zeros((len(batch), k), dtype=torch.bool)
    gold = torch.zeros(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, :len(b["seq"])] = torch.tensor(b["seq"])
        attn[i, :len(b["seq"])] = 1
        pos[i, :len(b["positions"])] = torch.tensor(b["positions"])
        valid[i, :len(b["positions"])] = True
        gold[i] = b["gold"]
    return ids, attn, pos, valid, gold


def packed_logits(encoder, head, ids, attn, pos, valid):
    hs = encoder(input_ids=ids, attention_mask=attn).last_hidden_state
    b, k = pos.shape
    gathered = hs[torch.arange(b).unsqueeze(1), pos]          # [B, K, d]
    logits = head(gathered).squeeze(-1)                        # [B, K]
    return logits.masked_fill(~valid, -1e4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/decision-v2-pilot/train.jsonl")
    ap.add_argument("--calibration", default="data/decision-v2-pilot/calibration.jsonl")
    ap.add_argument("--out", default="runs/v6_reference")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr-encoder", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N rows (smoke runs)")
    ap.add_argument("--max-k", type=int, default=16)
    ap.add_argument("--no-encoder", action="store_true", help="do not write encoder weights")
    ap.add_argument("--fit-calibration", action="store_true", default=False)
    ap.add_argument("--cal-limit", type=int, default=0, help="cap calibration rows (smoke runs)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    from myna.v6_engine import (CoralOrdinalScoreHead, GraphModule, MynaV6,
                                SetAwareBankingHead, _readout)

    torch.manual_seed(args.seed)
    backbone = REPO / "myna-v5-prototype 2" / "deberta-v5-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(str(backbone))
    renderer = Renderer(tokenizer)
    train = PackedSuite(REPO / args.train, renderer, limit=args.limit, max_k=args.max_k)
    print(f"train: {len(train)} packed questions from {train.rows} rows; "
          f"skipped {train.skipped_high_k} with K > {args.max_k}", flush=True)

    encoder = AutoModel.from_pretrained(str(backbone))
    head = _readout(1024, 512, 0.1)
    optim = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": args.lr_encoder},
        {"params": head.parameters(), "lr": args.lr_head},
    ], weight_decay=0.01)
    warmup = max(1, int(0.05 * args.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, 1 - s / max(1, args.steps)))

    encoder.train()
    order = torch.randperm(len(train)).tolist()
    cursor, t0, running = 0, time.time(), 0.0
    for step in range(args.steps):
        batch = [train[i] for i in order[cursor:cursor + args.batch]]
        cursor += args.batch
        if cursor + args.batch > len(order):
            cursor = 0
        ids, attn, pos, valid, gold = collate(batch, pad_id=tokenizer.pad_token_id or 0)
        logits = packed_logits(encoder, head, ids, attn, pos, valid)
        loss = F.cross_entropy(logits, gold)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 1.0)
        optim.step()
        sched.step()
        running += loss.item()
        if (step + 1) % 10 == 0:
            print(f"step {step+1}/{args.steps} loss {running/10:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            running = 0.0

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    temperatures = {"choice": 1.0, "score": 1.0, "noul": 1.0}
    q90 = {"choice": 0.5, "score": 0.5, "noul": 0.5}
    q95 = {"choice": 0.5, "score": 0.5, "noul": 0.5}
    if args.fit_calibration:
        temperatures, q90, q95 = fit_calibration_split(
            encoder, head, renderer, REPO / args.calibration, args.max_k, args.batch, args.cal_limit)
        print("calibration:", temperatures, q90, q95)

    config = {
        "temperatures": temperatures,
        "conformal_q90": q90,
        "conformal_q95": q95,
        "max_packed_options": 16,
        "use_coral_score_head": False,
        "b77_none_mode": "T_fallback",
        "reference_trainer": "release/train_v6.py",
        "train_rows": train.rows,
        "train_questions": len(train),
        "steps": args.steps,
        "batch": args.batch,
        "lr_encoder": args.lr_encoder,
        "lr_head": args.lr_head,
        "seed": args.seed,
    }
    ckpt = {
        "config": config,
        "packed_scorer": head.state_dict(),
        "b77_set_head": SetAwareBankingHead().state_dict(),
        "ordinal_score_head": CoralOrdinalScoreHead().state_dict(),
        "graph_module": GraphModule().state_dict(),
        "dev_audit": None, "test_audit": None, "calibration_test": None,
        "model_label": "myna-v6-reference",
    }
    torch.save(ckpt, out / "myna-v6-unified.pt")
    if not args.no_encoder:
        encoder.save_pretrained(out / "deberta-v5-finetuned")
        tokenizer.save_pretrained(out / "deberta-v5-finetuned")
    (out / "reference_config.json").write_text(json.dumps(config, indent=1))
    print("wrote", out, flush=True)

    # A checkpoint that the engine cannot load is not a deliverable; verify the heads here.
    probe = torch.load(out / "myna-v6-unified.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(probe["packed_scorer"], strict=True)
    print("packed_scorer reloads with strict=True", flush=True)


def fit_calibration_split(encoder, head, renderer, path, max_k, batch_size, cal_limit=0):
    """Per-primitive temperature and conformal quantiles on the calibration split."""
    ds = PackedSuite(path, renderer, limit=cal_limit, max_k=max_k)
    encoder.eval()
    raw = []
    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            chunk = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
            ids, attn, pos, valid, gold = collate(chunk, pad_id=0)
            logits = packed_logits(encoder, head, ids, attn, pos, valid)
            for j, item in enumerate(chunk):
                raw.append((item["primitive"], logits[j, :len(item["positions"])].tolist(),
                            item["gold"]))
    temperatures, q90, q95 = {}, {}, {}
    grid = [round(0.5 + 0.05 * i, 2) for i in range(31)]      # 0.50 .. 2.00
    for prim in ("choice", "score", "noul"):
        rows = [(lg, g) for p, lg, g in raw if p == prim]
        if not rows:
            temperatures[prim] = 1.0
            q90[prim] = q95[prim] = 0.5
            continue
        best, best_nll = 1.0, math.inf
        for t in grid:
            nll = 0.0
            for lg, g in rows:
                probs = F.softmax(torch.tensor(lg) / t, dim=-1)
                nll -= math.log(max(1e-12, float(probs[g])))
            if nll < best_nll:
                best, best_nll = t, nll
        temperatures[prim] = best
        scores = []
        for lg, g in rows:
            probs = F.softmax(torch.tensor(lg) / best, dim=-1)
            scores.append(1.0 - float(probs[g]))
        scores.sort()

        def quant(scores, alpha):
            n = len(scores)
            rank = min(n - 1, max(0, math.ceil((n + 1) * alpha) - 1))
            return scores[rank]

        q90[prim] = quant(scores, 0.90)
        q95[prim] = quant(scores, 0.95)
    return temperatures, q90, q95


if __name__ == "__main__":
    main()

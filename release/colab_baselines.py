"""In-domain fine-tuned DeBERTa baseline, second seed, and a leave-one-source-out check.

Run on a GPU (Colab T4). On CPU it is only useful with `--limit` as a smoke test.

    python release/colab_baselines.py --out /content/results
    python release/colab_baselines.py --limit 40 --max-len 64 --batch 2 --out /tmp/smoke

What it trains
--------------
The same backbone as Myna v6 (`microsoft/deberta-v3-large`), on the same train split, with a
conventional readout: one linear classifier per (source, question) cell over the `[CLS]`
state. This is the like-for-like in-domain baseline the paper is missing. It has no
cardinality routing, no set-aware head, no consistency term and no conformal gate; if Myna's
readout buys anything, this is where it shows.

Experiments
-----------
1. `baseline_seed42.json`  — in-domain baseline, seed 42
2. `baseline_seed7.json`   — the same run with a second seed (variance, not a new method)
3. `ood_<source>.json`     — leave-one-source-out: train without the source, test on it.
   This is a genuine out-of-domain number *for the baseline*. It is not an out-of-domain
   number for the shipped Myna v6, which saw every source in training.

Outputs are JSON with per-cell accuracy, macro, overall, joint bundle exact, and per-question
outcomes, so the paper's aggregation script can price them the same way it prices Myna.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]

MODEL_NAME = "microsoft/deberta-v3-large"


def state_text(state) -> str:
    """The suite's own renderer, inlined so the script does not import the package.

    `myna.real_data._state_text` flattens str / dict / list states in insertion order;
    keeping the copy here means the notebook needs only the JSONL files from the clone,
    not a working install of the package at whatever commit the clone happens to carry.
    """
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return " ".join(state_text(v) for v in state.values())
    if isinstance(state, (list, tuple)):
        return " ".join(state_text(v) for v in state)
    raise ValueError(f"unsupported state type {type(state).__name__}")


def gold_index(spec: dict) -> int:
    if spec["type"] == "noul":
        return 1 if spec["label"] else 0
    if spec["type"] == "score":
        return int(spec["label"]) - 1
    crit = spec["criteria"]
    labels = list(crit) if isinstance(crit, dict) else [str(c) for c in crit]
    return labels.index(spec["label"])


def n_options(spec: dict) -> int:
    if spec["type"] == "noul":
        return 2
    return len(spec["criteria"])


class CellDataset(Dataset):
    """One example per question, grouped by cell id."""

    def __init__(self, rows: list[dict]):
        self.by_cell: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            rid = row["_meta"]["id"]
            text = state_text(row["state"])
            for qname, spec in row["questions"].items():
                cell = f"{row['_meta']['source']}/{qname}"
                self.by_cell[cell].append({"row_id": rid, "cell": cell, "qname": qname,
                                           "text": text, "gold": gold_index(spec),
                                           "n": n_options(spec)})
        self.cells = sorted(self.by_cell)

    def __len__(self):
        return sum(len(v) for v in self.by_cell.values())


class MultiCellBaseline(nn.Module):
    def __init__(self, n_labels: dict[str, int], dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel

        # The DeBERTa-v3-large config declares float16; load float32 so the heads and the
        # encoder share a dtype, and let autocast handle the GPU path. `dtype=` is the 5.x
        # spelling, `torch_dtype=` the 4.x one.
        try:
            self.encoder = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
        except TypeError:
            self.encoder = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({c: nn.Linear(hidden, n) for c, n in n_labels.items()})

    def forward(self, cell: str, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.heads[cell](self.dropout(cls))


def collate(batch: list[dict], tok, max_len: int):
    enc = tok([b["text"] for b in batch], padding=True, truncation=True, max_length=max_len,
              return_tensors="pt")
    gold = torch.tensor([b["gold"] for b in batch], dtype=torch.long)
    return enc["input_ids"], enc["attention_mask"], gold


@torch.no_grad()
def evaluate(model, ds: CellDataset, tok, device, max_len, batch_size=8):
    model.eval()
    per_cell: dict[str, list[int]] = defaultdict(list)
    per_row: dict[str, list[int]] = defaultdict(list)
    records = []
    for cell in ds.cells:
        if cell not in model.heads:
            continue
        items = ds.by_cell[cell]
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            ids, attn, gold = collate(chunk, tok, max_len)
            logits = model(cell, ids.to(device), attn.to(device))
            pred = logits.argmax(dim=-1).cpu()
            for b, p, g in zip(chunk, pred.tolist(), gold.tolist()):
                ok = int(p == g)
                per_cell[cell].append(ok)
                per_row[b["row_id"]].append(ok)
                records.append({"row_id": b["row_id"], "cell": cell, "pred": p, "gold": g,
                                "ok": ok})
    cells = {c: {"n": len(v), "acc": round(float(np.mean(v)), 6)} for c, v in sorted(per_cell.items())}
    return {
        "cells": cells,
        "macro": round(float(np.mean([v["acc"] for v in cells.values()])), 6),
        "overall": round(float(np.mean([r["ok"] for r in records])), 6),
        "bundle_exact": round(float(np.mean([all(v) for v in per_row.values()])), 6),
        "n_questions": len(records),
        "n_rows": len(per_row),
        "records": records,
    }


def train(rows: list[dict], seed: int, epochs: int, batch: int, accum: int, lr_enc: float,
          lr_head: float, max_len: int, device: str, limit: int = 0,
          grad_checkpoint: bool = True,
          size_rows: list[dict] | None = None) -> tuple[MultiCellBaseline, object]:
    from transformers import DebertaV2Tokenizer

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if limit:
        rows = rows[:limit]
    # The DeBERTa-v3 tokenizer ships as a SentencePiece model; `AutoTokenizer` picks a
    # fast class that tries to convert it as tiktoken and fails, so name the slow class.
    tok = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    ds = CellDataset(rows)
    # Heads are sized over train *and* test cardinalities: the option count is schema, not a
    # label, and a head capped at the train maximum can never emit a test option that the
    # training split happened not to contain (banking77's 78th option is the case in point).
    sizing = CellDataset(size_rows) if size_rows is not None else ds
    n_labels = {c: max(item["n"] for item in sizing.by_cell[c]) for c in sizing.cells}
    for c in ds.cells:
        worst = max(item["gold"] for item in ds.by_cell[c])
        assert worst < n_labels[c], f"{c}: gold {worst} >= head {n_labels[c]}"
    model = MultiCellBaseline(n_labels).to(device)
    if grad_checkpoint and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    optim = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": lr_enc},
        {"params": model.heads.parameters(), "lr": lr_head},
    ], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(ds) / (batch * accum))
    total = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, (s + 1) / max(1, int(0.05 * total))) *
        max(0.0, 1 - s / max(1, total)))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    model.train()
    cursor = {c: 0 for c in ds.cells}
    t0 = time.time()
    step = 0
    for epoch in range(epochs):
        order = list(ds.cells)
        random.shuffle(order)
        for cell in order:
            items = ds.by_cell[cell]
            idx = list(range(len(items)))
            random.shuffle(idx)
            for start in range(0, len(idx) - batch + 1, batch):
                chunk = [items[i] for i in idx[start:start + batch]]
                ids, attn, gold = collate(chunk, tok, max_len)
                with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    logits = model(cell, ids.to(device), attn.to(device))
                    loss = F.cross_entropy(logits, gold.to(device)) / accum
                scaler.scale(loss).backward()
                if (start // batch + 1) % accum == 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim)
                    scaler.update()
                    sched.step()
                    optim.zero_grad(set_to_none=True)
                    step += 1
                    if step % 200 == 0:
                        print(f"  epoch {epoch+1} step {step}/{total} "
                              f"({time.time()-t0:.0f}s)", flush=True)
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr-enc", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ood-source", default="yelp")
    ap.add_argument("--skip-ood", action="store_true")
    ap.add_argument("--only-ood", action="store_true", help="skip the seed runs")
    ap.add_argument("--data-dir", default="data/decision-v2-pilot")
    ap.add_argument("--force", action="store_true", help="re-run seeds already on disk")
    ap.add_argument("--out", default="runs/colab_baselines")
    ap.add_argument("--grad-checkpoint", action="store_true", default=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data_dir)
    if not data.is_absolute():
        data = REPO / data
    load = lambda name: [json.loads(l) for l in (data / name).read_text().splitlines() if l.strip()]
    train_rows, test_rows = load("train.jsonl"), load("test.jsonl")
    print(f"train {len(train_rows)} rows | test {len(test_rows)} rows | device {args.device}",
          flush=True)

    test_ds = CellDataset(test_rows)
    for seed in ([] if args.only_ood else args.seeds):
        done_path = out / f"baseline_seed{seed}.json"
        if done_path.exists() and not args.force:
            print(f"seed {seed}: already written, skipping ({done_path})", flush=True)
            continue
        print(f"=== in-domain baseline, seed {seed} ===", flush=True)
        model, tok = train(train_rows, seed, args.epochs, args.batch, args.accum,
                           args.lr_enc, args.lr_head, args.max_len, args.device, args.limit,
                           args.grad_checkpoint, size_rows=train_rows + test_rows)
        res = evaluate(model, test_ds, tok, args.device, args.max_len)
        res.update({"seed": seed, "epochs": args.epochs, "batch": args.batch, "accum": args.accum,
                    "lr_enc": args.lr_enc, "lr_head": args.lr_head, "max_len": args.max_len,
                    "model": MODEL_NAME, "train_rows": len(train_rows), "kind": "in_domain"})
        (out / f"baseline_seed{seed}.json").write_text(json.dumps(res, indent=1))
        print(f"seed {seed}: macro {res['macro']:.4f} | overall {res['overall']:.4f} | "
              f"bundle {res['bundle_exact']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    if not args.skip_ood and args.ood_source:
        src = args.ood_source
        kept = [r for r in train_rows if r["_meta"]["source"] != src]
        held = [r for r in test_rows if r["_meta"]["source"] == src]
        print(f"=== leave-one-source-out: train without {src} ({len(kept)} rows), "
              f"test on {len(held)} ===", flush=True)
        model, tok = train(kept, args.seeds[0], args.epochs, args.batch, args.accum,
                           args.lr_enc, args.lr_head, args.max_len, args.device, args.limit,
                           args.grad_checkpoint, size_rows=train_rows + test_rows)
        res = evaluate(model, CellDataset(held), tok, args.device, args.max_len)
        res.update({"seed": args.seeds[0], "held_out_source": src, "kind": "leave_one_source_out",
                    "train_rows": len(kept)})
        (out / f"ood_{src}.json").write_text(json.dumps(res, indent=1))
        print(f"OOD {src}: macro {res['macro']:.4f} | overall {res['overall']:.4f}", flush=True)


if __name__ == "__main__":
    main()

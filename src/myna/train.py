"""Train a Myna checkpoint on the synthetic typed-decisions corpus.

Usage:  uv run python -m myna.train [--steps 4000] [--device mps]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import WORKFLOWS, generate
from .model import MynaConfig, MynaModel, typed_loss
from .tokenizer import question_tensors, train_tokenizer, encode_text


def build_batch(exs, tok, wf, device):
    questions = WORKFLOWS[wf][0]
    qt = question_tensors(tok, [(q.instruction, q.options) for q in questions])
    texts = [e.state for e in exs]
    ids = [encode_text(tok, t) for t in texts]
    ls = max(len(x) for x in ids)
    state = torch.zeros(len(ids), ls, dtype=torch.int64)
    lens = torch.tensor([len(x) for x in ids], dtype=torch.int64)
    for i, x in enumerate(ids):
        state[i, : len(x)] = torch.tensor(x)
    gold = torch.tensor([e.gold for e in exs], dtype=torch.int64)
    return {
        "state_ids": state.to(device),
        "state_len": lens.to(device),
        "gold": gold.to(device),
        **{k: v.to(device) for k, v in qt.items()},
    }


def evaluate(model, tok, splits, device, temperature=1.0):
    model.eval()
    rows = {}
    ece_bin = {}  # key -> (sum_conf, sum_correct, count) aggregated in 10 bins
    with torch.no_grad():
        for wf, data in splits.items():
            questions = WORKFLOWS[wf][0]
            for i in range(0, len(data), 64):
                exs = data[i : i + 64]
                b = build_batch(exs, tok, wf, device)
                logits = model(b["state_ids"], b["state_len"], b["q_ids"], b["q_mask"],
                               b["span_mat"], b["opt_valid"], b["decide_idx"]) / temperature
                probs = F.softmax(logits, dim=-1)
                pred = probs.argmax(-1)
                for n, q in enumerate(questions):
                    hit = (pred[:, n] == b["gold"][:, n]).float()
                    key = f"{wf}/{q.name}"
                    acc, tot = rows.get(key, (0.0, 0))
                    rows[key] = (acc + float(hit.sum()), tot + len(exs))
                    conf = probs[:, n].max(-1).values
                    bins = (conf * 10).clamp(max=9).long()
                    for bi in range(10):
                        sel = bins == bi
                        if not bool(sel.any()):
                            continue
                        c, s, m = ece_bin.get(key, (0.0, 0.0, 0))
                        ece_bin[key] = (
                            c + float(conf[sel].sum()),
                            s + float(hit[sel].sum()),
                            m + int(sel.sum()),
                        )
                    if q.type == "noul":
                        p = probs[:, n, 1]
                        g = b["gold"][:, n].float()
                        bk, bt = rows.get(key + ":brier", (0.0, 0))
                        rows[key + ":brier"] = (bk + float(((p - g) ** 2).sum()), bt + len(exs))
    out = {k: v[0] / max(v[1], 1) for k, v in rows.items()}
    for key, (c, s, m) in ece_bin.items():
        out[f"{key}:ece"] = abs(c - s) / max(m, 1)  # single-bin-per-example ECE (M-norm, 10 bins)
    return out


def fit_temperature(model, tok, splits, device):
    """1-D grid on dev NLL — calibration is a single scalar per checkpoint."""
    best, best_t = 1e18, 1.0
    model.eval()
    with torch.no_grad():
        for t in [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0]:
            tot, cnt = 0.0, 0
            for wf, data in splits.items():
                for i in range(0, min(len(data), 512), 64):
                    b = build_batch(data[i : i + 64], tok, wf, device)
                    logits = model(b["state_ids"], b["state_len"], b["q_ids"], b["q_mask"],
                                   b["span_mat"], b["opt_valid"], b["decide_idx"]) / t
                    lp = F.log_softmax(logits, dim=-1)
                    sel = lp.gather(-1, b["gold"][..., None]).squeeze(-1)
                    tot, cnt = tot - float(sel.sum()), cnt + sel.numel()
            if tot / cnt < best:
                best, best_t = tot / cnt, t
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=700)
    ap.add_argument("--out", default="runs/myna-v0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-n", type=int, default=192)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    data = {
        "train": {wf: generate(args.n_train, wf, rng, "train") for wf in WORKFLOWS},
        "dev": {wf: generate(args.n_eval, wf, rng, "dev") for wf in WORKFLOWS},
        "test": {wf: generate(args.n_eval, wf, rng, "test") for wf in WORKFLOWS},
    }
    tok = train_tokenizer([e.state for wf in data["train"].values() for e in wf], vocab_size=4096)
    print("tokenizer trained:", tok.get_vocab_size())

    cfg = MynaConfig(vocab=tok.get_vocab_size())
    model = MynaModel(cfg).to(device)
    print("params:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    wfs = list(data["train"])
    dev_probe = {wf: data["dev"][wf][: args.eval_n] for wf in wfs}
    ema = None
    step_t0 = time.time()
    for step in range(args.steps):
        model.train()
        wf = wfs[step % len(wfs)]
        pool = data["train"][wf]
        exs = rng.sample(pool, args.batch)
        b = build_batch(exs, tok, wf, device)
        logits = model(b["state_ids"], b["state_len"], b["q_ids"], b["q_mask"],
                       b["span_mat"], b["opt_valid"], b["decide_idx"])
        loss = typed_loss(logits, b["gold"], torch.ones_like(b["gold"], dtype=torch.bool))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        if step % 200 == 0 or step == args.steps - 1:
            el = time.time() - step_t0
            print(f"step {step:5d}  loss {loss.item():.3f}  ema {ema:.3f}  {el:.0f}s", flush=True)
        if args.eval_every and step and step % args.eval_every == 0 and step != args.steps - 1:
            m = evaluate(model, tok, dev_probe, device)
            names = [k for k in m if not k.endswith((":brier", ":ece"))]
            acc = sum(m[k] for k in names) / len(names)
            print(f"step {step:5d}  dev-mid acc {acc:.4f}", flush=True)

    temperature = fit_temperature(model, tok, data["dev"], device)
    print(f"temperature: {temperature}")
    print("=== dev ===")
    for k, v in sorted(evaluate(model, tok, data["dev"], device, temperature).items()):
        print(f"{k:32s} {v:.4f}")
    print("=== test ===")
    for k, v in sorted(evaluate(model, tok, data["test"], device, temperature).items()):
        print(f"{k:32s} {v:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": vars(cfg), "temperature": temperature}, out / "model.pt")
    tok.save(str(out / "tokenizer.json"))
    print(f"saved checkpoint to {out}")


if __name__ == "__main__":
    main()


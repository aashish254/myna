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
from tokenizers import Tokenizer

from .data import WORKFLOWS, generate
from .model import MynaConfig, MynaModel, typed_loss
from .tokenizer import question_tensors, train_tokenizer, encode_text


def build_batch(exs, tok, questions, device):
    """questions: list[Question] fixed for the batch (the v0 contract: one
    question set shared across rows; real-corpus batches group by question
    signature, see myna.real_data)."""
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
    """splits: {key: (questions, examples)} — synthetic workflows and real
    suite groups both fit this shape."""
    model.eval()
    rows = {}
    ece_bin = {}  # key -> (sum_conf, sum_correct, count) aggregated in 10 bins
    with torch.no_grad():
        for wf, (questions, data) in splits.items():
            for i in range(0, len(data), 64):
                exs = data[i : i + 64]
                b = build_batch(exs, tok, questions, device)
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
            for wf, (questions, data) in splits.items():
                for i in range(0, min(len(data), 512), 64):
                    b = build_batch(data[i : i + 64], tok, questions, device)
                    logits = model(b["state_ids"], b["state_len"], b["q_ids"], b["q_mask"],
                                   b["span_mat"], b["opt_valid"], b["decide_idx"]) / t
                    lp = F.log_softmax(logits, dim=-1)
                    sel = lp.gather(-1, b["gold"][..., None]).squeeze(-1)
                    tot, cnt = tot - float(sel.sum()), cnt + sel.numel()
            if tot / cnt < best:
                best, best_t = tot / cnt, t
    return best_t


def build_needle_batch(exs, tok, device, entity):
    """One shared question (fixed entity) across the batch; the needle varies."""
    from .longctx import DESKS
    instr = f"Which desk owns {entity}?"
    qt = question_tensors(tok, [(instr, list(DESKS))])
    ids = [encode_text(tok, e.state) for e in exs]
    lens = torch.tensor([len(x) for x in ids], dtype=torch.int64)
    Lmax = max(len(x) for x in ids)
    state = torch.zeros(len(ids), Lmax, dtype=torch.int64)
    for i, x in enumerate(ids):
        state[i, : len(x)] = torch.tensor(x)
    gold = torch.tensor([[e.gold] for e in exs], dtype=torch.int64)
    return {
        "state_ids": state.to(device), "state_len": lens.to(device), "gold": gold.to(device),
        "Lmax": Lmax,
        **{k: v.to(device) for k, v in qt.items()},
    }


def train_long_context(args, rng, device):
    """V1-D: fine-tune (or train) on needle-in-long-context recall, truncated
    backprop so only the trailing grad window carries gradient."""
    from .longctx import DESKS, ENTITIES, FILLER, Needle, make_needle

    if args.init:
        ip = Path(args.init)
        tok = Tokenizer.from_file(str(ip / "tokenizer.json"))
        ck = torch.load(ip / "model.pt", map_location=device, weights_only=False)
        cfg = MynaConfig(**ck["cfg"])
        model = MynaModel(cfg)
        model.load_state_dict(ck["state_dict"])
    else:
        cfg = MynaConfig(vocab=args.vocab)
        model = MynaModel(cfg)
        tok = train_tokenizer([" ".join(FILLER), " ".join(DESKS), " ".join(ENTITIES)],
                              vocab_size=args.vocab)
    model.to(device)
    print(f"long-context: state={args.long_context} grad={args.grad_tokens} "
          f"params={sum(p.numel() for p in model.parameters())}", flush=True)

    split = max(0, args.long_context - args.grad_tokens)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    ema, t0 = None, time.time()
    for step in range(args.steps):
        model.train()
        entity = ENTITIES[rng.randrange(len(ENTITIES))]
        exs = [make_needle(tok, args.long_context, rng, tail_cap=args.grad_tokens,
                           entity=entity) for _ in range(args.batch)]
        b = build_needle_batch(exs, tok, device, entity)
        logits = model.forward_truncated(b["state_ids"], split, b["state_len"], b["q_ids"],
                                         b["q_mask"], b["span_mat"], b["opt_valid"], b["decide_idx"])
        loss = typed_loss(logits, b["gold"], torch.ones_like(b["gold"], dtype=torch.bool))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        acc = (logits.argmax(-1) == b["gold"]).float().mean().item()
        ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        if step % 100 == 0 or step == args.steps - 1:
            print(f"step {step:5d}  loss {loss.item():.3f}  ema {ema:.3f}  "
                  f"batch-acc {acc:.3f}  {time.time()-t0:.0f}s", flush=True)
        if args.eval_every and step and step % args.eval_every == 0 and step != args.steps - 1:
            out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
            tok.save(str(out / "tokenizer.json"))
            torch.save({"state_dict": model.state_dict(), "cfg": vars(cfg),
                        "temperature": 1.0, "step": step}, out / "model_last.pt")


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
    ap.add_argument("--suite", default=None,
                    help="frozen-suite dir (train/development/test.jsonl, laya/kev request shape); "
                         "overrides the synthetic corpus")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--long-context", type=int, default=0,
                    help="train needle-in-long-context recall at this state token length (V1-D); "
                         "uses truncated backprop and ignores --suite/--synthetic")
    ap.add_argument("--grad-tokens", type=int, default=768,
                    help="with --long-context: keep the needle inside this trailing window so it "
                         "receives gradient; everything before is scanned detached")
    ap.add_argument("--init", default=None,
                    help="with --long-context: checkpoint dir to warm-start weights + tokenizer from")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    if args.long_context:
        train_long_context(args, rng, device)
        return

    if args.suite:
        from .real_data import load_suite, suite_texts

        data = load_suite(args.suite)
        tok = train_tokenizer(suite_texts(data["train"]), vocab_size=args.vocab)
    else:
        data = {
            "train": {wf: generate(args.n_train, wf, rng, "train") for wf in WORKFLOWS},
            "dev": {wf: generate(args.n_eval, wf, rng, "dev") for wf in WORKFLOWS},
            "test": {wf: generate(args.n_eval, wf, rng, "test") for wf in WORKFLOWS},
        }
        tok = train_tokenizer([e.state for wf in data["train"].values() for e in wf], vocab_size=4096)
    # uniform split shape: {split: {group_key: (questions, examples)}}
    if not args.suite:
        data = {
            split: {wf: (WORKFLOWS[wf][0], exs) for wf, exs in groups.items()}
            for split, groups in data.items()
        }
    print("tokenizer trained:", tok.get_vocab_size())

    cfg = MynaConfig(vocab=tok.get_vocab_size())
    model = MynaModel(cfg).to(device)
    print("params:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    wfs = list(data["train"])
    group_weights = [len(data["train"][wf][1]) for wf in wfs]
    dev_probe = {
        wf: (q, exs[: args.eval_n]) for wf, (q, exs) in data["dev"].items()
    } if data["dev"] else {
        wf: (q, exs[: args.eval_n]) for wf, (q, exs) in data["train"].items()
    }
    ema = None
    step_t0 = time.time()
    for step in range(args.steps):
        model.train()
        wf = rng.choices(wfs, weights=group_weights, k=1)[0]
        questions, pool = data["train"][wf]
        exs = rng.choices(pool, k=args.batch)
        b = build_batch(exs, tok, questions, device)
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
            # rolling snapshot so a long run can be evaluated or stopped early
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            tok.save(str(out / "tokenizer.json"))
            torch.save(
                {"state_dict": model.state_dict(), "cfg": vars(cfg), "temperature": 1.0, "step": step},
                out / "model_last.pt",
            )

    temperature = fit_temperature(model, tok, data["dev"], device)
    print(f"temperature: {temperature}")
    dev_m = evaluate(model, tok, data["dev"], device, temperature)
    test_m = evaluate(model, tok, data["test"], device, temperature)
    print("=== dev ===")
    for k, v in sorted(dev_m.items()):
        print(f"{k:32s} {v:.4f}")
    print("=== test ===")
    for k, v in sorted(test_m.items()):
        print(f"{k:32s} {v:.4f}")

    # machine-readable metrics + a key->type map so downstream tables can roll
    # up by source x question-type (apples-to-apples with the laya witness).
    qtypes = {}
    for split in ("dev", "test"):
        for wf, (questions, _exs) in data[split].items():
            for q in questions:
                qtypes[f"{wf}/{q.name}"] = q.type

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump({"temperature": temperature, "dev": dev_m, "test": test_m, "qtypes": qtypes}, f, indent=2)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": vars(cfg), "temperature": temperature}, out / "model.pt")
    tok.save(str(out / "tokenizer.json"))
    print(f"saved checkpoint to {out}")


if __name__ == "__main__":
    main()


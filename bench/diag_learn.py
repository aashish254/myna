"""Diagnostic: can the full model FIT train at all, and does dev follow?

Distinguishes optimization failure (train stuck at chance) from a
generalization gap (train climbs, dev flat). Trains the production-size
model on a 2000/workflow pool and probes both splits every 200 steps.
"""

import argparse
import random
import time

import torch

from myna.data import WORKFLOWS, generate
from myna.model import MynaConfig, MynaModel, typed_loss
from myna.tokenizer import train_tokenizer
from myna.train import build_batch, evaluate

ap = argparse.ArgumentParser(description="fit(train) vs gen(dev) on the synthetic corpus")
ap.add_argument("--steps", type=int, default=1200)
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--pool", type=int, default=2000, help="train examples per workflow")
args = ap.parse_args()

STEPS, LR = args.steps, args.lr

rng = random.Random(42)
device = "mps" if torch.backends.mps.is_available() else "cpu"
train_pool = {wf: (WORKFLOWS[wf][0], generate(args.pool, wf, rng, "train")) for wf in WORKFLOWS}
data = {
    "train": train_pool,
    "probe_train": {wf: (q, exs[:200]) for wf, (q, exs) in train_pool.items()},
    "dev": {wf: (WORKFLOWS[wf][0], generate(200, wf, random.Random(99), "dev")) for wf in WORKFLOWS},
}
tok = train_tokenizer([e.state for _, exs in data["train"].values() for e in exs], vocab_size=4096)
cfg = MynaConfig(vocab=tok.get_vocab_size())
model = MynaModel(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
wfs = list(WORKFLOWS)
print(f"device={device} lr={LR} steps={STEPS} pool={args.pool}/workflow "
      f"params={sum(p.numel() for p in model.parameters())}", flush=True)

t0 = time.time()
ema = None
for step in range(STEPS):
    model.train()
    wf = wfs[step % len(wfs)]
    questions, exs = data["train"][wf]
    b = build_batch(rng.sample(exs, 32), tok, questions, device)
    logits = model(b["state_ids"], b["state_len"], b["q_ids"], b["q_mask"],
                   b["span_mat"], b["opt_valid"], b["decide_idx"])
    loss = typed_loss(logits, b["gold"], torch.ones_like(b["gold"], dtype=torch.bool))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
    if step % 200 == 0 or step == STEPS - 1:
        tr = evaluate(model, tok, data["probe_train"], device)
        dv = evaluate(model, tok, data["dev"], device)
        names = [k for k in tr if not k.endswith((":brier", ":ece"))]
        a_tr = sum(tr[k] for k in names) / len(names)
        a_dv = sum(dv[k] for k in names) / len(names)
        print(f"step {step:5d}  ema {ema:.3f}  fit(train) {a_tr:.3f}  gen(dev) {a_dv:.3f}  {time.time()-t0:.0f}s", flush=True)

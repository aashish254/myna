"""Diagnostic: can the full model FIT train at all, and does dev follow?

Distinguishes optimization failure (train stuck at chance) from a
generalization gap (train climbs, dev flat). Trains the production-size
model on a 2000/workflow pool and probes both splits every 200 steps.
"""

import random
import sys
import time

import torch

from myna.data import WORKFLOWS, generate
from myna.model import MynaConfig, MynaModel, typed_loss
from myna.tokenizer import train_tokenizer
from myna.train import build_batch, evaluate

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 6e-4

rng = random.Random(42)
device = "mps" if torch.backends.mps.is_available() else "cpu"
train_pool = {wf: generate(2000, wf, rng, "train") for wf in WORKFLOWS}
data = {
    "train": train_pool,
    "probe_train": {wf: exs[:200] for wf, exs in train_pool.items()},
    "dev": {wf: generate(200, wf, random.Random(99), "dev") for wf in WORKFLOWS},
}
tok = train_tokenizer([e.state for wf in data["train"].values() for e in wf], vocab_size=4096)
cfg = MynaConfig(vocab=tok.get_vocab_size())
model = MynaModel(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
wfs = list(WORKFLOWS)
print(f"device={device} lr={LR} steps={STEPS} params={sum(p.numel() for p in model.parameters())}", flush=True)

t0 = time.time()
ema = None
for step in range(STEPS):
    model.train()
    wf = wfs[step % len(wfs)]
    b = build_batch(rng.sample(data["train"][wf], 32), tok, wf, device)
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

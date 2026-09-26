"""Stage 1 of a fit ladder: can the production-size model memorize 64
examples? If not, the training path — not the data or capacity — is broken.
Raise N (256, 1024, 2000) in later stages to find where fitting breaks."""

import random
import sys
import time

import torch

from myna.data import WORKFLOWS, generate
from myna.model import MynaConfig, MynaModel, typed_loss
from myna.tokenizer import train_tokenizer
from myna.train import build_batch, evaluate

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 6e-4

rng = random.Random(42)
device = "mps" if torch.backends.mps.is_available() else "cpu"
pool = {wf: (WORKFLOWS[wf][0], generate(N, wf, rng, "train")) for wf in WORKFLOWS}
dev = {wf: (WORKFLOWS[wf][0], generate(200, wf, random.Random(99), "dev")) for wf in WORKFLOWS}
tok = train_tokenizer([e.state for _, exs in pool.values() for e in exs], vocab_size=2048)
cfg = MynaConfig(vocab=tok.get_vocab_size())
model = MynaModel(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
wfs = list(WORKFLOWS)
print(f"N={N} steps={STEPS} lr={LR} device={device}", flush=True)

t0 = time.time()
ema = None
for step in range(STEPS):
    model.train()
    wf = wfs[step % len(wfs)]
    questions, exs = pool[wf]
    b = build_batch(rng.sample(exs, min(32, N)), tok, questions, device)
    logits = model(b["state_ids"], b["state_len"], b["q_ids"], b["q_mask"],
                   b["span_mat"], b["opt_valid"], b["decide_idx"])
    loss = typed_loss(logits, b["gold"], torch.ones_like(b["gold"], dtype=torch.bool))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
    if step % (STEPS // 10) == 0 or step == STEPS - 1:
        fit = evaluate(model, tok, pool, device)
        dv = evaluate(model, tok, dev, device)
        names = [k for k in fit if not k.endswith((":brier", ":ece"))]
        a_fit = sum(fit[k] for k in names) / len(names)
        a_gen = sum(dv[k] for k in names) / len(names)
        print(f"step {step:5d}  ema {ema:.4f}  fit {a_fit:.3f}  gen(dev) {a_gen:.3f}  {time.time()-t0:.0f}s", flush=True)

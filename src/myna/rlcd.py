"""RLCD-style fine-tuning: optimize the DECISION score, not the token.

The supervised checkpoint maximizes CE, which is a proper score for the
reported distribution — but CE gradients shrink once the argmax is right,
so confidence drifts away from calibration. Here we fine-tune directly on
a strictly proper reward with a KL leash to the reference:

    reward(p, gold) = log p[gold]            (log)  or
                      1 - ||p - onehot||^2   (brier)
    loss = -reward.mean() + beta * KL(p || ref)

Note why there is no REINFORCE here, unlike laya's RLCD: our policy IS the
reported distribution — there is no sampled action to take a gradient
through. A first attempt with REINFORCE + batch-mean baseline actively
degraded the model (train reward 1.0 -> -0.98 in 200 steps): it pushes
p(gold) DOWN on examples scoring above the batch mean, which is meaningless
when p is the answer, not a step of a trajectory. The proper score of a
reported distribution is differentiable in the distribution; we optimize it
directly.

Usage:
    uv run python -m myna.rlcd --ckpt runs/myna-v0 --steps 800 --score brier

Eval (per-workflow accuracy, brier, ECE, temperature) is printed before and
after so the calibration delta is checkable, not asserted.
"""

from __future__ import annotations

import argparse
import copy
import random

import torch
import torch.nn.functional as F

from .data import WORKFLOWS, generate
from .engine import Myna
from .tokenizer import encode_text
from .train import build_batch, evaluate


def proper_score(probs: torch.Tensor, gold: torch.Tensor, kind: str) -> torch.Tensor:
    """per-example [B,N] strictly proper score of the reported distribution."""
    onehot = F.one_hot(gold, probs.shape[-1]).float()
    if kind == "log":
        return torch.log(probs.gather(-1, gold[..., None]).squeeze(-1).clamp_min(1e-8))
    if kind == "brier":
        return 1.0 - ((probs - onehot) ** 2).sum(-1)
    # spherical: 2p - ||p||^2 - 1 shifted to (0,1] properness equivalent
    return 2 * probs.gather(-1, gold[..., None]).squeeze(-1) - (probs.pow(2).sum(-1))


def rlcd_step(model, ref, b, args):
    logits = model(b["state_ids"], b["state_len"], b["q_ids"], b["q_mask"],
                   b["span_mat"], b["opt_valid"], b["decide_idx"]) / args.policy_temp
    probs = F.softmax(logits, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    # the reported distribution IS the policy: the proper score is
    # differentiable in it, so maximize it directly (no REINFORCE).
    r = proper_score(probs, b["gold"], args.score)
    with torch.no_grad():
        ref_logp = F.log_softmax(
            ref(b["state_ids"], b["state_len"], b["q_ids"], b["q_mask"],
                b["span_mat"], b["opt_valid"], b["decide_idx"]) / args.policy_temp, dim=-1)
    kl = (probs * (logp - ref_logp)).sum(-1).mean()
    loss = -r.mean() + args.beta * kl
    return loss, float(r.mean().detach())


def report(model, tok, splits, device, temperature, tag):
    m = evaluate(model, tok, splits, device, temperature)
    names = [k for k in m if not k.endswith((":brier", ":ece"))]
    acc = sum(m[k] for k in names) / len(names)
    eces = [m[k + ":ece"] for k in names if k + ":ece" in m]
    briers = [m[k + ":brier"] for k in names if k + ":brier" in m]
    print(f"{tag:8s} acc {acc:.4f}  mean-ece {(sum(eces)/max(len(eces),1)):.4f}  "
          f"mean-brier {(sum(briers)/max(len(briers),1)):.4f}", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/myna-v0")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--beta", type=float, default=0.1, help="KL leash to reference")
    ap.add_argument("--policy-temp", type=float, default=1.0)
    ap.add_argument("--score", default="brier", choices=["log", "brier", "spherical"])
    ap.add_argument("--n-eval", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/myna-v0-rlcd")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    myna = Myna(args.ckpt, device=args.device)
    model, tok = myna.model, myna.tok
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    temperature = myna.temperature

    train = {wf: (WORKFLOWS[wf][0], generate(1500, wf, rng, "train")) for wf in WORKFLOWS}
    dev = {wf: (WORKFLOWS[wf][0], generate(args.n_eval, wf, rng, "dev")) for wf in WORKFLOWS}
    print(f"reference temperature {temperature}")
    m_before = report(model, tok, dev, args.device, temperature, "before")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    wfs = list(WORKFLOWS)
    for step in range(args.steps):
        model.train()
        wf = wfs[step % len(wfs)]
        questions, pool = train[wf]
        b = build_batch(rng.sample(pool, args.batch), tok, questions, args.device)
        loss, r = rlcd_step(model, ref, b, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            print(f"step {step:5d}  loss {float(loss.detach()):.4f}  mean-reward {r:.4f}", flush=True)

    # refit the single temperature scalar after RLCD — the leash changes the scale
    from .train import fit_temperature

    temperature = fit_temperature(model, tok, dev, args.device)
    m_after = report(model, tok, dev, args.device, temperature, "after")

    import json
    from pathlib import Path

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": vars(myna.cfg), "temperature": temperature},
               out / "model.pt")
    tok.save(str(out / "tokenizer.json"))
    with open(out / "rlcd_report.json", "w") as f:
        json.dump({"before": m_before, "after": m_after, "temperature": temperature}, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

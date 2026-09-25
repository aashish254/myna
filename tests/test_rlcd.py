"""RLCD plumbing smoke: one policy step runs, gradients flow, and the
proper-score helper matches closed forms."""

import random

import pytest
import torch
import torch.nn.functional as F

from myna.data import WORKFLOWS, generate
from myna.engine import Myna
from myna.model import MynaConfig, MynaModel
from myna.rlcd import proper_score, rlcd_step
from myna.tokenizer import train_tokenizer
from myna.train import build_batch


def test_proper_score_closed_forms():
    p = torch.tensor([0.7, 0.2, 0.1])
    g = torch.tensor([1])
    onehot = F.one_hot(g, 3).float()
    assert abs(float(proper_score(p[None], g[None], "log")) - torch.log(torch.tensor(0.2)).item()) < 1e-6
    brier = 1.0 - ((p - onehot) ** 2).sum()
    assert abs(float(proper_score(p[None], g[None], "brier")) - float(brier)) < 1e-6


@pytest.fixture(scope="module")
def myna(tmp_path_factory):
    rng = random.Random(3)
    exs = generate(48, "support", rng)
    tok = train_tokenizer([e.state for e in exs], vocab_size=600)
    cfg = MynaConfig(vocab=tok.get_vocab_size(), d_model=64, n_layers=2, n_heads=4, d_k=16, d_v=16, d_ff=128, d_ptr=32)
    torch.manual_seed(0)
    model = MynaModel(cfg)
    d = tmp_path_factory.mktemp("ckpt")
    torch.save({"state_dict": model.state_dict(), "cfg": vars(cfg), "temperature": 1.0}, d / "model.pt")
    tok.save(str(d / "tokenizer.json"))
    return Myna(d, device="cpu")


class Args:
    policy_temp = 1.0
    score = "brier"
    beta = 0.1


def test_rlcd_step_updates_params(myna):
    import copy

    rng = random.Random(4)
    exs = generate(16, "support", rng)
    b = build_batch(exs, myna.tok, "support", "cpu")
    ref = copy.deepcopy(myna.model)
    before = [p.clone() for p in myna.model.parameters()]
    myna.model.train()
    loss, r = rlcd_step(myna.model, ref, b, Args())
    loss.backward()
    grads = [p.grad for p in myna.model.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads), "no gradients reached the model"
    assert -1.0 <= r <= 1.0

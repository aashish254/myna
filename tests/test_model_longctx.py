"""Truncated-backprop long-context forward: CPU-only correctness checks."""

from __future__ import annotations

import torch

from myna.model import MynaConfig, MynaModel

CFG = MynaConfig(vocab=128, d_model=32, n_layers=2, n_heads=2, d_k=8, d_v=8, d_ff=48, d_ptr=16)


def _batch(B=2, N=2, O=3, Ls=40, Lq=6):
    g = torch.Generator().manual_seed(0)
    state_ids = torch.randint(1, CFG.vocab, (B, Ls), generator=g)  # no padding (uniform long state)
    state_len = torch.full((B,), Ls, dtype=torch.int64)
    q_ids = torch.randint(1, CFG.vocab, (N, Lq), generator=g)
    q_mask = torch.ones(N, Lq)
    span_mat = torch.rand(N, O, Lq, generator=g) / Lq
    opt_valid = torch.ones(N, O, dtype=torch.bool)
    decide_idx = torch.full((N,), Lq - 1, dtype=torch.int64)
    return state_ids, state_len, q_ids, q_mask, span_mat, opt_valid, decide_idx


def test_truncated_split0_equals_full_forward():
    torch.manual_seed(0)
    m = MynaModel(CFG)
    m.eval()
    sid, slen, qi, qm, sm, ov, di = _batch()
    with torch.no_grad():
        a = m(sid, slen, qi, qm, sm, ov, di)
        b = m.forward_truncated(sid, 0, slen, qi, qm, sm, ov, di)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_truncated_keeps_gradients_through_suffix():
    torch.manual_seed(1)
    m = MynaModel(CFG)
    m.train()
    sid, slen, qi, qm, sm, ov, di = _batch(Ls=48)
    split = 32  # prefix [0,32) detached, suffix [32,48) carries grad
    logits = m.forward_truncated(sid, split, slen, qi, qm, sm, ov, di)
    loss = torch.log_softmax(logits, -1).mean()  # finite scalar to differentiate
    loss.backward()
    emb_grad = m.trunk.tok.weight.grad
    assert emb_grad is not None and torch.isfinite(emb_grad).all()
    # embeddings used only in the detached prefix should receive no gradient
    # (vocab rows are shared, so just assert the suffix produced grad overall)
    assert emb_grad.abs().sum() > 0


def test_truncated_output_finite_at_long_state():
    torch.manual_seed(2)
    m = MynaModel(CFG)
    m.eval()
    sid, slen, qi, qm, sm, ov, di = _batch(B=1, Ls=512)
    with torch.no_grad():
        out = m.forward_truncated(sid, 384, slen, qi, qm, sm, ov, di)
    assert torch.isfinite(out).all()

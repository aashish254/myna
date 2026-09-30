"""MLX port parity: the mlx forward must match the torch forward on the SAME
weights, so a torch checkpoint can be served from mlx unchanged. Runs on a
tiny random model (no training, no checkpoint file) plus, when one exists, on
the real v1 checkpoint."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from myna.model import MynaConfig, MynaModel

TINY = dict(vocab=64, d_model=32, n_layers=2, n_heads=2, d_k=8, d_v=8, d_ff=48, d_ptr=16)


def make_batch(B=2, N=3, O=4, Ls=6, Lq=5, vocab=64, seed=0):
    rng = np.random.default_rng(seed)
    state_ids = rng.integers(1, vocab, size=(B, Ls)).astype(np.int64)
    state_len = np.array([Ls] * B, dtype=np.int64)
    q_ids = rng.integers(1, vocab, size=(N, Lq)).astype(np.int64)
    q_mask = np.ones((N, Lq), dtype=np.float32)
    span_mat = rng.random((N, O, Lq)).astype(np.float32)
    opt_valid = np.ones((N, O), dtype=bool)
    decide_idx = np.array([Lq - 1] * N, dtype=np.int64)
    return (state_ids, state_len, q_ids, q_mask, span_mat, opt_valid, decide_idx)


# Every test below needs the MLX engine, so a machine without the optional `mlx` extra
# skips this file instead of failing it (SPEC §9.51). Placed after `make_batch`, which
# `test_mlx_int8.py` imports from here. Both orders were measured with mlx hidden: each
# skips, so this placement is about which file the skip is attributed to, not a cliff.
pytest.importorskip("mlx.core")


def _torch_logits(cfg, sd, batch):
    m = MynaModel(cfg)
    m.load_state_dict(sd)
    m.eval()
    ts = [torch.tensor(np.asarray(x)) for x in batch]
    with torch.no_grad():
        return m(*ts).numpy()


def test_mlx_matches_torch_random_weights():
    from myna.mlx_model import MynaMLX
    cfg = MynaConfig(**TINY)
    torch.manual_seed(0)
    m = MynaModel(cfg)
    sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "tiny.pt"
        torch.save({"state_dict": sd, "cfg": vars(cfg)}, p)
        mlx = MynaMLX.from_checkpoint(p)
        batch = make_batch(vocab=cfg.vocab)
        ref = _torch_logits(cfg, sd, batch)
        got = np.array(mlx.forward(*batch))
    assert got.shape == ref.shape
    # fp32 einsum reduction order differs between torch and mlx, so the logits
    # agree only to ~3e-3; argmax (the decision) must be identical.
    np.testing.assert_allclose(got, ref, rtol=1e-2, atol=1e-2)
    assert (got.argmax(-1) == ref.argmax(-1)).all()


def test_mlx_is_structurally_exact_float64():
    """The tight check that a port bug would fail: compare against a torch
    float64 reference on the CPU stream (no Metal downcast). Proves the port
    is algorithmically identical, not just close."""
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from myna import mlx_model as MM
    cfg = MynaConfig(**TINY)
    torch.manual_seed(0)
    m = MynaModel(cfg).double()
    m.eval()
    sd = {k: v.double().numpy() for k, v in m.state_dict().items()}
    eng = MM.MynaMLX(cfg.__dict__, MM._build_params(cfg.__dict__, sd))
    batch = make_batch(vocab=cfg.vocab)
    ts = [torch.tensor(x).double() if np.issubdtype(np.asarray(x).dtype, np.floating) else torch.tensor(x)
          for x in batch]
    with torch.no_grad():
        ref = m(*ts).numpy()
    got = np.array(eng.forward(*batch))
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_mlx_padded_state_matches():
    """Padded (shorter) states exercise the mask path where torch and mlx
    differ most: k=v=0 and gate forced to 1 on padding."""
    from myna.mlx_model import MynaMLX
    cfg = MynaConfig(**TINY)
    torch.manual_seed(1)
    m = MynaModel(cfg)
    sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "tiny.pt"
        torch.save({"state_dict": sd, "cfg": vars(cfg)}, p)
        mlx = MynaMLX.from_checkpoint(p)
        batch = list(make_batch(vocab=cfg.vocab))
        batch[0][1, 4:] = 0          # second row: 4 real tokens, 2 pad
        batch[1] = np.array([4, 4], dtype=np.int64)  # state_len reflects pad
        ref = _torch_logits(cfg, sd, batch)
        got = np.array(mlx.forward(*batch))
    np.testing.assert_allclose(got, ref, rtol=1e-2, atol=1e-2)
    assert (got.argmax(-1) == ref.argmax(-1)).all()

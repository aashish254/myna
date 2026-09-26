"""Apple-Silicon MLX inference port of MynaModel (V1-C).

A faithful mirror of `model.py` + `trunk.py` in `mlx.core`, so the same
trained torch checkpoint answers questions on the Metal GPU/NPU path with no
torch dependency at inference. Weights are loaded straight from the torch
`state_dict` (Linear stores [out, in] in both frameworks, so it is a cast).

Correctness is the whole point: `tests/test_mlx_parity.py` checks the MLX
forward matches the torch forward on random weights to 1e-3, and the same
harness runs on the real checkpoint. Nothing here re-trains; it only serves.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------- rope

def _repeat_interleave_2(x: mx.array) -> mx.array:
    # x: [..., d/2] -> [..., d] repeating each element twice along last axis
    return mx.repeat(x, 2, axis=-1)


def _rep0(x: mx.array, n: int) -> mx.array:
    """repeat_interleave along axis 0: element i becomes n copies (matches torch)."""
    s = x.shape
    return mx.broadcast_to(x.reshape((s[0], 1) + s[1:]), (s[0], n) + s[1:]).reshape((s[0] * n,) + s[1:])


def apply_rope(x: mx.array, pos: mx.array, base: float = 10000.0) -> mx.array:
    # x: [B, H, L, d]; pos: [L] or [B, L]
    d = x.shape[-1]
    inv = mx.power(mx.array(base, x.dtype),
                   -mx.arange(0, d, 2, dtype=x.dtype) / d)          # [d/2]
    ang = pos[..., None].astype(x.dtype) * inv                       # [*, d/2] or [B,L,d/2]
    cos = _repeat_interleave_2(mx.cos(ang))
    sin = _repeat_interleave_2(mx.sin(ang))
    if pos.ndim == 1:                                                # [L, d] -> broadcast over B,H
        pass
    else:                                                            # [B, L, d] -> [B, 1, L, d]
        cos, sin = cos[:, None], sin[:, None]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rot = mx.concatenate((-x2, x1), axis=-1)
    return x * cos + rot * sin


# ---------------------------------------------------------------- scans

def gla_chunked(q, k, v, gate, init_S=None, chunk: int = 16):
    B, H, T, dk = q.shape
    dv = v.shape[-1]
    S = mx.zeros((B, H, dk, dv), dtype=q.dtype) if init_S is None else init_S
    outs = []
    for j in range(0, T, chunk):
        qs, ks, vs, gs = q[:, :, j:j + chunk], k[:, :, j:j + chunk], v[:, :, j:j + chunk], gate[:, :, j:j + chunk]
        C = qs.shape[2]
        loga = mx.cumsum(mx.log(mx.maximum(gs, 1e-8)), axis=-2)     # [B,H,C,dk]
        exponent = mx.minimum(loga[:, :, :, None, :] - loga[:, :, None, :, :], mx.array(0.0))  # [B,H,C,C,dk]
        tri = mx.tril(mx.ones((C, C), dtype=q.dtype)).reshape(1, 1, C, C, 1)
        D = mx.exp(exponent) * tri
        w = mx.einsum("bhtk,bhsk,bhtsk->bhts", qs, ks, D)
        o_intra = mx.einsum("bhts,bhsv->bhtv", w, vs)
        o_inter = mx.einsum("bhtk,bhkv->bhtv", qs * mx.exp(loga), S)
        outs.append(o_intra + o_inter)
        decay = mx.exp(loga[:, :, -1:, :] - loga)
        S = mx.exp(loga[:, :, -1, :])[..., None] * S + mx.einsum("bhsk,bhsv->bhkv", ks * decay, vs)
    return mx.concatenate(outs, axis=2), S


# ---------------------------------------------------------------- modules

class MynaMLX:
    """Inference engine. Holds an mlx param tree and a forward mirroring MynaModel."""

    def __init__(self, cfg: dict, params: dict):
        self.cfg = cfg
        self.p = params
        self.d_ptr = cfg["d_ptr"]

    @classmethod
    def from_checkpoint(cls, path) -> "MynaMLX":
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = dict(ck["cfg"])
        sd = {k: v.detach().float().numpy() for k, v in ck["state_dict"].items()}
        return cls(cfg, _build_params(cfg, sd))

    def _linear(self, scope, x):
        w = self.p[f"{scope}.weight"]
        b = self.p.get(f"{scope}.bias")
        y = x @ w.T
        return y if b is None else y + b

    def _embed(self, ids):
        return self.p["trunk.tok.weight"][ids]

    def _rms(self, scope, x):
        w = self.p[f"{scope}.w"]
        ms = mx.mean(x * x, axis=-1, keepdims=True)
        return x * mx.rsqrt(ms + 1e-6) * w

    def _gla(self, scope, x, pos, init_S=None, mask=None):
        B, T = x.shape[0], x.shape[1]
        dv, h = self.cfg["d_v"], self.cfg["n_heads"]
        q, k, v, g = self._split_params(scope, x, pos)
        if mask is not None:
            k = k * mask
            v = v * mask
            g = g * mask + (1.0 - mask)
        out, S = gla_chunked(q, k, v, g, init_S)
        out = out.swapaxes(1, 2).reshape(B, T, h * dv)
        return self._linear(f"{scope}.out_proj", out), S

    def _split_params(self, scope, x, pos):
        # scope names the GLA instance under a layer, e.g. "trunk.layers.0.fwd"
        B, T, _ = x.shape
        dk, dv, h = self.cfg["d_k"], self.cfg["d_v"], self.cfg["n_heads"]
        qkv = self._linear(f"{scope}.qkv", x)
        qkv = qkv.reshape(B, T, h, 2 * dk + dv).swapaxes(1, 2)
        q, k, v = qkv[..., :dk], qkv[..., dk:2 * dk], qkv[..., 2 * dk:]
        q = apply_rope(q, pos)
        k = apply_rope(k, pos)
        g = mx.sigmoid(self._linear(f"{scope}.gate", x))
        g = g.reshape(B, T, h, dk).swapaxes(1, 2)
        return q, k, v, g

    def _scan_state(self, li, h, pos, mask=None):
        base = f"trunk.layers.{li}"
        o, S = self._gla(f"{base}.fwd", self._rms(f"{base}.norm1", h), pos, mask=mask)
        h = h + o
        ff_h = self._rms(f"{base}.norm2", h)
        ff = self._linear(f"{base}.ff.2", nn.gelu(self._linear(f"{base}.ff.0", ff_h)))
        return h + ff, S

    def _scan_question(self, li, h, pos, init_S, mask=None):
        base = f"trunk.layers.{li}"
        n = self._rms(f"{base}.norm1", h)
        o_f, _ = self._gla(f"{base}.fwd", n, pos, init_S=init_S, mask=mask)
        n_flip = n[:, ::-1]
        pos_flip = pos[..., ::-1] if pos.ndim == 1 else pos[:, ::-1]
        m_flip = None if mask is None else mask[:, :, ::-1]
        o_b, _ = self._gla(f"{base}.bwd", n_flip, pos_flip, mask=m_flip)
        h = h + o_f + o_b[:, ::-1]
        ff_h = self._rms(f"{base}.norm2", h)
        ff = self._linear(f"{base}.ff.2", nn.gelu(self._linear(f"{base}.ff.0", ff_h)))
        return h + ff

    def forward(self, state_ids, state_len, q_ids, q_mask, span_mat, opt_valid, decide_idx):
        # boundary inputs are numpy arrays / python lists (inference only, no autograd)
        sid = mx.array(state_ids)
        qid = mx.array(q_ids)
        B, Ls = sid.shape
        N, Lq = qid.shape
        sm = (sid != 0).astype(mx.float32)[..., None]

        h = self._embed(sid) * sm
        pos_s = mx.arange(Ls)
        S_layers = []
        for li in range(self.cfg["n_layers"]):
            h, S = self._scan_state(li, h, pos_s, mask=sm[:, None])
            S_layers.append(S)

        # question branches for the whole batch: [B*N, Lq, d]
        qe = self._embed(qid)                                      # [N, Lq, d]
        hq = mx.broadcast_to(qe[None], (B, N, Lq, qe.shape[-1])).reshape(B * N, Lq, -1)
        qmask = mx.array(q_mask).astype(mx.float32)
        qmask = mx.broadcast_to(qmask[None], (B, N, Lq)).reshape(B * N, Lq)
        qm = qmask[..., None, :, None]
        state_len = mx.array(state_len)
        pos_q = _rep0(state_len, N)[..., None] + mx.arange(Lq)[None]  # [B*N, Lq]
        for li in range(self.cfg["n_layers"]):
            hq = self._scan_question(li, hq, pos_q, _rep0(S_layers[li], N), mask=qm)
        hq = self._rms("trunk.norm", hq).reshape(B, N, Lq, -1)

        d = hq.shape[-1]
        idx = mx.broadcast_to(mx.array(decide_idx)[None, :, None, None], (B, N, 1, d))
        decide = mx.take_along_axis(hq, idx, axis=2).squeeze(2)     # [B,N,d]
        pooled = mx.einsum("bnld,nol->bnod", hq, mx.array(span_mat))
        q = self._linear("wq", decide)
        k = self._linear("wk", pooled)
        logits = mx.einsum("bnd,bnod->bno", q, k) / math.sqrt(self.d_ptr)
        valid = mx.array(opt_valid)
        logits = mx.where(valid[None], logits, mx.array(-1e9, dtype=logits.dtype))
        mx.eval(logits)
        return logits


def _build_params(cfg, sd):
    """Map a torch state_dict (as numpy) into the flat mlx param namespace used above."""
    p = {}
    p["trunk.tok.weight"] = mx.array(sd["trunk.tok.weight"])
    for i in range(cfg["n_layers"]):
        for branch in ("fwd", "bwd"):
            s = f"trunk.layers.{i}.{branch}"
            p[f"{s}.qkv.weight"] = mx.array(sd[f"{s}.qkv.weight"])
            p[f"{s}.gate.weight"] = mx.array(sd[f"{s}.gate.weight"])
            p[f"{s}.gate.bias"] = mx.array(sd[f"{s}.gate.bias"])
            p[f"{s}.out_proj.weight"] = mx.array(sd[f"{s}.out_proj.weight"])
        for n in ("norm1", "norm2"):
            p[f"trunk.layers.{i}.{n}.w"] = mx.array(sd[f"trunk.layers.{i}.{n}.w"])
        p[f"trunk.layers.{i}.ff.0.weight"] = mx.array(sd[f"trunk.layers.{i}.ff.0.weight"])
        p[f"trunk.layers.{i}.ff.0.bias"] = mx.array(sd[f"trunk.layers.{i}.ff.0.bias"])
        p[f"trunk.layers.{i}.ff.2.weight"] = mx.array(sd[f"trunk.layers.{i}.ff.2.weight"])
        p[f"trunk.layers.{i}.ff.2.bias"] = mx.array(sd[f"trunk.layers.{i}.ff.2.bias"])
    p["trunk.norm.w"] = mx.array(sd["trunk.norm.w"])
    p["wq.weight"] = mx.array(sd["wq.weight"])
    p["wq.bias"] = mx.array(sd["wq.bias"])
    p["wk.weight"] = mx.array(sd["wk.weight"])
    p["wk.bias"] = mx.array(sd["wk.bias"])
    return p

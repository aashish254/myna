"""Myna trunk.

Shape of the model, and why it is fast:

- The observation (ticket, page, document) is scanned once through a stack of
  gated-linear-attention layers. Each layer carries a fixed-size matrix state
  `S` of shape [H, d_k, d_v]. Appending tokens to the observation continues
  the scan from the cached `S`; nothing before the append point is ever
  re-read. This is exact, not approximate (see tests/test_trunk_numerics.py).
- Questions do NOT go into the trunk with the state. Each question is a short
  branch that resumes every layer's scan from the cached state-end `S`, and
  additionally runs a full backward scan over its own tokens. Questions can
  read the whole state; they cannot read each other — isolation is
  architectural, not a mask.
- There is no vocabulary readout anywhere. Typed heads (choice/score/noul)
  are pointer probes over question-branch hidden states.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------- rotary


def rope_cos_sin(positions: torch.Tensor, dim: int, base: float = 10000.0):
    inv = base ** (-torch.arange(0, dim, 2, device=positions.device).float() / dim)
    ang = positions.float().unsqueeze(-1) * inv  # [*, dim/2]
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x: torch.Tensor, pos: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    # x: [B, H, L, d]; pos: [L] or [B, L] absolute positions; half-split rotation
    d = x.shape[-1]
    cos, sin = rope_cos_sin(pos, d, base)
    cos, sin = torch.repeat_interleave(cos, 2, dim=-1), torch.repeat_interleave(sin, 2, dim=-1)
    if cos.dim() == 2:
        pass  # [L, d] broadcasts over [B,H,L,d]
    else:
        cos, sin = cos[:, None], sin[:, None]  # [B, 1, L, d]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rot = torch.cat((-x2, x1), dim=-1)
    return x * cos + rot * sin


# ---------------------------------------------------------------- scans


def gla_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    init_S: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential scan with carried state. Returns (out [B,H,L,dv], final S [B,H,dk,dv]).

    `init_S` resumes from a cached prefix — the append path.
    """
    B, H, T, dk = q.shape
    dv = v.shape[-1]
    S = q.new_zeros(B, H, dk, dv) if init_S is None else init_S
    outs = []
    for t in range(T):
        S = gate[:, :, t].unsqueeze(-1) * S + k[:, :, t].unsqueeze(-1) * v[:, :, t].unsqueeze(-2)
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], S))
    return torch.stack(outs, dim=2), S


def gla_parallel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Exact quadratic reference form (no carried state; starts empty)."""
    loga = torch.cumsum(torch.log(gate.clamp_min(1e-8)), dim=-2)
    T = loga.shape[-2]
    exponent = (loga.unsqueeze(-2) - loga.unsqueeze(-3)).clamp(max=0.0)  # [B,H,T,T,dk]
    mask = torch.ones(T, T, device=q.device, dtype=q.dtype).tril().reshape(1, 1, T, T, 1)
    D = torch.exp(exponent) * mask
    w = torch.einsum("bhti,bhsi,bhtsi->bhts", q, k, D)
    return torch.einsum("bhts,bhsv->bhtv", w, v)


def gla_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    init_S: torch.Tensor | None = None,
    chunk: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked exact form: small [C,C,dk] matrices within a chunk, a
    chunk-level recurrence for the carried state. Same output as the other
    two forms, but the autograd graph is linear in sequence length instead of
    quadratic — this is the training path."""
    B, H, T, dk = q.shape
    S = q.new_zeros(B, H, dk, v.shape[-1]) if init_S is None else init_S
    outs = []
    for j in range(0, T, chunk):
        qs, ks, vs, gs = [t[:, :, j : j + chunk] for t in (q, k, v, gate)]
        C = qs.shape[2]
        loga = torch.cumsum(torch.log(gs.clamp_min(1e-8)), dim=-2)  # [B,H,C,dk]
        exponent = (loga.unsqueeze(-2) - loga.unsqueeze(-3)).clamp(max=0.0)
        mask = torch.ones(C, C, device=q.device, dtype=q.dtype).tril().reshape(1, 1, C, C, 1)
        w = torch.einsum("bhtk,bhsk,bhtsk->bhts", qs, ks, torch.exp(exponent) * mask)
        o_intra = torch.einsum("bhts,bhsv->bhtv", w, vs)
        o_inter = torch.einsum("bhtk,bhkv->bhtv", qs * torch.exp(loga), S)
        outs.append(o_intra + o_inter)
        decay = torch.exp(loga[:, :, -1:, :] - loga)  # contribution of each step to chunk end
        S = torch.exp(loga[:, :, -1, :]).unsqueeze(-1) * S + torch.einsum("bhsk,bhsv->bhkv", ks * decay, vs)
    return torch.cat(outs, dim=2), S


# ---------------------------------------------------------------- layers


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class GLA(nn.Module):
    """One gated-linear-attention module. The caller picks scan direction by
    flipping the sequence; this module is just projections + scan."""

    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int):
        super().__init__()
        self.h, self.dk, self.dv = n_heads, d_k, d_v
        self.qkv = nn.Linear(d_model, n_heads * (2 * d_k + d_v), bias=False)
        self.gate = nn.Linear(d_model, n_heads * d_k, bias=True)
        # Forget-gate init matters enormously: with the default bias 0 the
        # gate is sigmoid(0)=0.5, so a token's memory decays to 1e-6 within
        # 20 steps and the model literally cannot read its observation at
        # init (measured; training then never escapes chance). Start at
        # a=sigmoid(4.6)~0.99 with no input dependence: long memory is the
        # prior, input-dependent forgetting is learned.
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 4.6)
        self.out_proj = nn.Linear(n_heads * d_v, d_model, bias=False)

    def _split(self, x, pos):
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, self.h, 2 * self.dk + self.dv).transpose(1, 2)
        q, k, v = torch.split(qkv, [self.dk, self.dk, self.dv], dim=-1)
        q, k = apply_rope(q, pos), apply_rope(k, pos)
        g = torch.sigmoid(self.gate(x).view(B, T, self.h, self.dk).transpose(1, 2))
        return q, k, v, g

    def forward(self, x, pos, init_S=None, parallel=False, mask=None, chunk=None):
        # mask: [B, 1, L, 1] valid-token mask; padded tokens contribute
        # nothing to the memory (k=v=0) while keeping tensors rectangular.
        q, k, v, g = self._split(x, pos)
        if mask is not None:
            # padded tokens contribute nothing to the memory (k=v=0) and must
            # not decay it either (gate forced to 1)
            k, v = k * mask, v * mask
            g = g * mask + (1.0 - mask)
        if parallel:
            out, S = gla_chunked(q, k, v, g, init_S, chunk=chunk or 16)
        else:
            out, S = gla_recurrent(q, k, v, g, init_S)
        B, T = x.shape[0], x.shape[1]
        return self.out_proj(out.transpose(1, 2).reshape(B, T, self.h * self.dv)), S


class MynaLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.fwd = GLA(d_model, n_heads, d_k, d_v)
        self.bwd = GLA(d_model, n_heads, d_k, d_v)  # used inside question branches only
        self.norm2 = RMSNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def scan_state(self, h, pos, init_S=None, parallel=False, mask=None, chunk=None):
        """One layer of the causal state pass. Returns (new h, state-end S)."""
        o, S = self.fwd(self.norm1(h), pos, init_S, parallel=parallel, mask=mask, chunk=chunk)
        h = h + o
        return h + self.ff(self.norm2(h)), S

    def scan_question(self, h, pos, init_S, parallel=False, mask=None, chunk=None):
        """Question-branch pass: forward resumes from the cached state-end S;
        backward reads only the branch itself."""
        n = self.norm1(h)
        o_f, _ = self.fwd(n, pos, init_S, parallel=parallel, mask=mask, chunk=chunk)
        mf = None if mask is None else torch.flip(mask, [2])
        o_b, _ = self.bwd(torch.flip(n, [1]), torch.flip(pos, [-1]), parallel=parallel, mask=mf, chunk=chunk)
        h = h + o_f + torch.flip(o_b, [1])
        return h + self.ff(self.norm2(h))


class MynaTrunk(nn.Module):
    def __init__(
        self,
        vocab: int,
        d_model: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        d_k: int = 64,
        d_v: int = 64,
        d_ff: int = 1024,
    ):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList(MynaLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_layers))
        self.norm = RMSNorm(d_model)

    # ---- state passes -------------------------------------------------

    def encode_state(self, ids: torch.Tensor, parallel: bool = False, chunk: int | None = None):
        """Full causal pass over an observation. Returns (final h, per-layer
        state-ends S). The cached continuation state is S alone: fixed size,
        independent of observation length."""
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.tok(ids)
        S_cache = []
        for layer in self.layers:
            h, S = layer.scan_state(h, pos, init_S=None, parallel=parallel, chunk=chunk)
            S_cache.append(S)
        return self.norm(h), S_cache

    def append_state(self, delta_ids: torch.Tensor, S_cache, state_len: int, chunk: int | None = None):
        """Exact incremental pass: scan ONLY the new tokens, resuming each
        layer from its cached state-end S. The state below the append point is
        never re-read. Returns (per-layer delta hidden states, new state-ends);
        streaming callers keep only the new S."""
        pos = torch.arange(state_len, state_len + delta_ids.shape[1], device=delta_ids.device)
        hd = self.tok(delta_ids)
        delta_h, new_S = [hd], []
        for layer, S in zip(self.layers, S_cache):
            hd, S_next = layer.scan_state(hd, pos, init_S=S, parallel=True, chunk=chunk)
            delta_h.append(hd)
            new_S.append(S_next)
        return delta_h, new_S

    def encode_questions(self, questions: list[torch.Tensor], S_cache, state_len: int, chunk: int | None = None):
        """Each question is a branch off the cached state: every layer's
        forward scan resumes from the state-end S, so a question reads the
        whole observation and nothing of any other question."""
        outs = []
        for q in questions:
            pos = torch.arange(state_len, state_len + q.shape[1], device=q.device)
            h = self.tok(q)
            for layer, S in zip(self.layers, S_cache):
                h = layer.scan_question(h, pos, S, parallel=True, chunk=chunk)
            outs.append(self.norm(h))
        return outs

    def encode_question_batch(
        self,
        q_ids: torch.Tensor,
        q_mask: torch.Tensor,
        S_cache,
        state_len: int,
        chunk: int | None = None,
    ) -> torch.Tensor:
        """All questions of one request scanned together: [N, Lq] ids,
        padded, returned hidden states [N, Lq, d]."""
        N, Lq = q_ids.shape
        pos = state_len + torch.arange(Lq, device=q_ids.device)
        h = self.tok(q_ids)
        m = q_mask[:, None, :, None]
        for layer, S in zip(self.layers, S_cache):
            h = layer.scan_question(h, pos, S.expand(N, -1, -1, -1), parallel=True, mask=m, chunk=chunk)
        return self.norm(h)

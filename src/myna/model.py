"""Myna model: trunk + typed pointer heads.

All three answer types run through one mechanism: a pointer head scores
option spans against a DECIDE probe. `choice` reads the argmax, `score` the
ordinal expectation, `noul` the probability on the implicit "Yes" option.
No vocabulary readout, no generation, no parsing.

Question templates are fixed per workflow, so the whole batch shares the
question segment tensors — only the state varies per row.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trunk import MynaTrunk


@dataclass
class MynaConfig:
    vocab: int = 4096
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_k: int = 64
    d_v: int = 64
    d_ff: int = 1024
    d_ptr: int = 256

    def trunk(self) -> MynaTrunk:
        return MynaTrunk(self.vocab, self.d_model, self.n_layers, self.n_heads, self.d_k, self.d_v, self.d_ff)


class MynaModel(nn.Module):
    def __init__(self, cfg: MynaConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = cfg.trunk()
        self.wq = nn.Linear(cfg.d_model, cfg.d_ptr)
        self.wk = nn.Linear(cfg.d_model, cfg.d_ptr)

    def forward(
        self,
        state_ids: torch.Tensor,  # [B, Ls]
        state_len: torch.Tensor,  # [B] true (pre-padding) lengths
        q_ids: torch.Tensor,  # [N, Lq]
        q_mask: torch.Tensor,  # [N, Lq] 1 for real question tokens
        span_mat: torch.Tensor,  # [N, O, Lq] option-span pooling weights
        opt_valid: torch.Tensor,  # [N, O]
        decide_idx: torch.Tensor,  # [N] index of the DECIDE token
    ):
        B, Ls = state_ids.shape
        N, Lq = q_ids.shape
        dev = state_ids.device
        sm = (state_ids != 0).float()[:, :, None]

        h = self.trunk.tok(state_ids) * sm
        pos_s = torch.arange(Ls, device=dev)
        S_layers = []
        for layer in self.trunk.layers:
            h, S = layer.scan_state(h, pos_s, parallel=True, mask=sm.unsqueeze(1))
            S_layers.append(S)

        hq = self.trunk.tok(q_ids).expand(B, N, Lq, -1).reshape(B * N, Lq, -1)
        qm = q_mask.float()[None].expand(B, N, Lq).reshape(B * N, Lq)[:, None, :, None]
        pos_q = state_len.repeat_interleave(N)[:, None] + torch.arange(Lq, device=dev)
        for layer, S in zip(self.trunk.layers, S_layers):
            hq = layer.scan_question(hq, pos_q, S.repeat_interleave(N, dim=0), parallel=True, mask=qm)
        hq = self.trunk.norm(hq).view(B, N, Lq, -1)

        decide = hq[:, torch.arange(N, device=dev), decide_idx]  # [B,N,d]
        pooled = torch.einsum("bnld,nol->bnod", hq, span_mat)  # mean-pool each option span
        q = self.wq(decide)
        k = self.wk(pooled)
        logits = torch.einsum("bnd,bnod->bno", q, k) / (self.cfg.d_ptr**0.5)
        return logits.masked_fill(~opt_valid[None], -1e9)


def typed_loss(logits: torch.Tensor, gold: torch.Tensor, has_gold: torch.Tensor):
    """logits [B,N,O], gold [B,N] option index, has_gold [B,N] bool."""
    lp = F.log_softmax(logits, dim=-1)
    sel = lp.gather(-1, gold[..., None]).squeeze(-1)
    return -sel[has_gold].mean()

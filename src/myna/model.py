"""Myna model: trunk + typed pointer heads.

All three answer types run through one mechanism: a pointer head scores
option spans against a DECIDE probe. `choice` reads the argmax, `score` the
ordinal expectation, `noul` the probability on the implicit "Yes" option.
No vocabulary readout, no generation, no parsing.

Question segments come in two shapes: one set shared by the whole batch (a
fixed workflow) or one set per row, which is what the real suites need — boolq
and mnli put the example itself in the instruction text, so every row has a
different question. Only the state varies per row in the first case; the second
pads questions, options and question tokens to the batch maximum.
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

    def _question_head(self, S_layers, state_len, q_ids, q_mask, span_mat, opt_valid, decide_idx):
        """Pointer readout given per-layer state-ends S_layers and per-row state
        lengths. Factored out so the state prefix can be scanned detached.

        Question tensors may arrive per-batch-row ([B,N,...], so one batch mixes
        question-sets) or shared across the batch ([N,...], the v0 contract);
        the shared form is broadcast rather than duplicated.
        """
        B = state_len.shape[0]
        if q_ids.dim() == 2:
            N, Lq = q_ids.shape
            q_ids = q_ids.expand(B, N, Lq)
            q_mask = q_mask.expand(B, N, Lq)
            span_mat = span_mat.expand(B, N, *span_mat.shape[1:])
            opt_valid = opt_valid.expand(B, *opt_valid.shape)
            decide_idx = decide_idx.expand(B, N)
        N, Lq = q_ids.shape[1], q_ids.shape[2]
        dev = q_ids.device
        hq = self.trunk.tok(q_ids).reshape(B * N, Lq, -1)
        qm = q_mask.reshape(B * N, Lq)[:, None, :, None]
        pos_q = state_len.repeat_interleave(N, dim=0)[:, None] + torch.arange(Lq, device=dev)
        for layer, S in zip(self.trunk.layers, S_layers):
            hq = layer.scan_question(hq, pos_q, S.repeat_interleave(N, dim=0), parallel=True, mask=qm)
        hq = self.trunk.norm(hq).view(B, N, Lq, -1)

        d = hq.shape[-1]
        decide = hq.gather(2, decide_idx[..., None, None].expand(B, N, 1, d)).squeeze(2)  # [B,N,d]
        pooled = torch.einsum("bnld,bnol->bnod", hq, span_mat)
        q = self.wq(decide)
        k = self.wk(pooled)
        logits = torch.einsum("bnd,bnod->bno", q, k) / (self.cfg.d_ptr**0.5)
        return logits.masked_fill(~opt_valid, -1e9)

    def forward(
        self,
        state_ids: torch.Tensor,  # [B, Ls]
        state_len: torch.Tensor,  # [B] true (pre-padding) lengths
        q_ids: torch.Tensor,  # [N, Lq] shared, or [B, N, Lq] per row
        q_mask: torch.Tensor,  # matches q_ids; 1 for real question tokens
        span_mat: torch.Tensor,  # [N, O, Lq] or [B, N, O, Lq] option-span pooling weights
        opt_valid: torch.Tensor,  # [N, O] or [B, N, O]
        decide_idx: torch.Tensor,  # [N] or [B, N] index of the DECIDE token
    ):
        B, Ls = state_ids.shape
        dev = state_ids.device
        sm = (state_ids != 0).float()[:, :, None]

        h = self.trunk.tok(state_ids) * sm
        pos_s = torch.arange(Ls, device=dev)
        S_layers = []
        for layer in self.trunk.layers:
            h, S = layer.scan_state(h, pos_s, parallel=True, mask=sm.unsqueeze(1))
            S_layers.append(S)

        return self._question_head(S_layers, state_len, q_ids, q_mask, span_mat, opt_valid, decide_idx)

    def forward_truncated(
        self,
        state_ids: torch.Tensor,  # [B, Ls] long state; padding token id 0 masked out
        split: int,               # tokens [0, split) scanned under no_grad; [split, Ls) with grad
        state_len: torch.Tensor,  # [B] true lengths (>= split); used for question offsets
        q_ids: torch.Tensor, q_mask: torch.Tensor, span_mat: torch.Tensor,
        opt_valid: torch.Tensor, decide_idx: torch.Tensor,
        chunk: int = 16,
    ):
        """Long-context training step with truncated backprop: the prefix
        (< the needle) is scanned detached — it still contributes numerically
        to the carried state, exactly as at inference — and only the suffix
        keeps gradients. Bounds training memory to O(Ls - split) so 4k states
        are trainable on consumer hardware while the recurrence still reads the
        whole observation."""
        B, Ls = state_ids.shape
        dev = state_ids.device
        if split <= 0:
            return self.forward(state_ids, state_len, q_ids, q_mask, span_mat, opt_valid, decide_idx)
        pre, suf = state_ids[:, :split], state_ids[:, split:]
        with torch.no_grad():
            sm_pre = (pre != 0).float()[:, :, None]
            h = self.trunk.tok(pre) * sm_pre
            S_pre = []
            for layer in self.trunk.layers:
                h, S = layer.scan_state(h, torch.arange(split, device=dev), parallel=True,
                                        mask=sm_pre.unsqueeze(1), chunk=chunk)
                S_pre.append(S)
        sm_suf = (suf != 0).float()[:, :, None]
        pos_suf = torch.arange(split, Ls, device=dev)
        h = self.trunk.tok(suf) * sm_suf
        S_layers = []
        for layer, S0 in zip(self.trunk.layers, S_pre):
            h, S = layer.scan_state(h, pos_suf, init_S=S0, parallel=True,
                                    mask=sm_suf.unsqueeze(1), chunk=chunk)
            S_layers.append(S)
        return self._question_head(S_layers, state_len, q_ids, q_mask, span_mat, opt_valid, decide_idx)


def typed_loss(logits: torch.Tensor, gold: torch.Tensor, has_gold: torch.Tensor):
    """logits [B,N,O], gold [B,N] option index, has_gold [B,N] bool."""
    lp = F.log_softmax(logits, dim=-1)
    sel = lp.gather(-1, gold[..., None]).squeeze(-1)
    return -sel[has_gold].mean()

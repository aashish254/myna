"""BPE tokenizer + question-segment builder.

Question layout inside a branch:

    {instruction tokens} [OPT] {option 0 text} [OPT] {option 1 text} ... [DECIDE]

The pointer head mean-pools hidden states over each option span and reads the
answer from [DECIDE]. [PAD] is id 0 so padded tokens mask out of the scans.
"""

from __future__ import annotations

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

OPT = "[OPT]"
DECIDE = "[DECIDE]"
SPECIALS = ["[PAD]", OPT, DECIDE]


def train_tokenizer(texts, vocab_size: int = 4096) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(iter(texts), trainer)
    return tok


def opt_id(tok: Tokenizer) -> int:
    return tok.token_to_id(OPT)


def decide_id(tok: Tokenizer) -> int:
    return tok.token_to_id(DECIDE)


def encode_text(tok: Tokenizer, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False).ids


def build_question(tok: Tokenizer, instruction: str, options: list[str]):
    """Returns (ids, spans, decide_idx) with spans[i] = (start, end) token
    range of option i's text."""
    ids: list[int] = list(encode_text(tok, instruction))
    spans = []
    for opt in options:
        ids.append(opt_id(tok))
        start = len(ids)
        ids.extend(encode_text(tok, opt))
        spans.append((start, len(ids)))
    ids.append(decide_id(tok))
    return ids, spans, len(ids) - 1


def question_tensors(tok: Tokenizer, questions: list[tuple[str, list[str]]]):
    """One question set, shared across every row of a batch (the v0 contract).

    Returns dict with:
      q_ids [N, Lq], q_mask [N, Lq], span_mat [N, O, Lq] (mean-pool weights),
      opt_valid [N, O], decide_idx [N]
    """
    return {k: v[0] for k, v in batch_question_tensors(tok, [questions]).items()}


def batch_question_tensors(tok: Tokenizer, rows: list[list[tuple[str, list[str]]]]):
    """Batch-build tensors where each row carries ITS OWN question set.

    rows[b] is the list of (instruction, options) questions for batch row b.
    Everything is padded to the batch maximum on all three varying axes —
    questions N, options O, question tokens Lq — so the tensors stay
    rectangular:
      q_ids [B, N, Lq], q_mask [B, N, Lq], span_mat [B, N, O, Lq],
      opt_valid [B, N, O], decide_idx [B, N]

    A row with fewer than N questions leaves its spare slots all-zero; their
    logits are undefined and must be masked by the caller's gold mask, exactly
    as an option slot with opt_valid False.
    """
    built = [[build_question(tok, instr, opts) for instr, opts in row] for row in rows]
    b_n = [len(r) for r in built]
    n_q = max(b_n)
    max_opts = max(len(opts) for row in rows for _, opts in row)
    lq = max(len(ids) for row in built for ids, _, _ in row)
    q_ids = torch.zeros(len(built), n_q, lq, dtype=torch.int64)
    q_mask = torch.zeros(len(built), n_q, lq)
    span_mat = torch.zeros(len(built), n_q, max_opts, lq)
    opt_valid = torch.zeros(len(built), n_q, max_opts, dtype=torch.bool)
    decide_idx = torch.zeros(len(built), n_q, dtype=torch.int64)
    for n, row in enumerate(built):
        for m, (ids, spans, dec) in enumerate(row):
            q_ids[n, m, : len(ids)] = torch.tensor(ids)
            q_mask[n, m, : len(ids)] = 1.0
            decide_idx[n, m] = dec
            for o, (s, e) in enumerate(spans):
                span_mat[n, m, o, s:e] = 1.0 / (e - s)
                opt_valid[n, m, o] = True
    return {
        "q_ids": q_ids,
        "q_mask": q_mask,
        "span_mat": span_mat,
        "opt_valid": opt_valid,
        "decide_idx": decide_idx,
    }

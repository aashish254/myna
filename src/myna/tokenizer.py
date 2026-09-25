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
    """Batch-build tensors for a fixed question set.

    Returns dict with:
      q_ids [N, Lq], q_mask [N, Lq], span_mat [N, O, Lq] (mean-pool weights),
      opt_valid [N, O], decide_idx [N]
    """
    built = [build_question(tok, instr, opts) for instr, opts in questions]
    n_q, max_opts = len(built), max(len(opts) for _, opts in questions)
    lq = max(len(ids) for ids, _, _ in built)
    q_ids = torch.zeros(n_q, lq, dtype=torch.int64)
    q_mask = torch.zeros(n_q, lq)
    span_mat = torch.zeros(n_q, max_opts, lq)
    opt_valid = torch.zeros(n_q, max_opts, dtype=torch.bool)
    decide_idx = torch.zeros(n_q, dtype=torch.int64)
    for n, (ids, spans, dec) in enumerate(built):
        q_ids[n, : len(ids)] = torch.tensor(ids)
        q_mask[n, : len(ids)] = 1.0
        decide_idx[n] = dec
        for o, (s, e) in enumerate(spans):
            span_mat[n, o, s:e] = 1.0 / (e - s)
            opt_valid[n, o] = True
    return {
        "q_ids": q_ids,
        "q_mask": q_mask,
        "span_mat": span_mat,
        "opt_valid": opt_valid,
        "decide_idx": decide_idx,
    }

"""Needle corpus generator invariants (CPU, no model)."""

from __future__ import annotations

import random

from myna.longctx import DESKS, FILLER, make_needle
from myna.tokenizer import train_tokenizer


def _tok():
    texts = [" ".join(FILLER), " ".join(DESKS), " ".join(FILLER + DESKS)]
    return train_tokenizer(texts, vocab_size=512)


def test_needle_gold_is_the_only_evidence():
    tok = _tok()
    rng = random.Random(0)
    nd = make_needle(tok, 256, rng)
    # gold desk name appears exactly once (in the needle), other desks absent
    assert nd.state.count(nd.options[nd.gold]) == 1
    for d in nd.options:
        if d != nd.options[nd.gold]:
            assert d not in nd.state
    assert 0 <= nd.gold < len(nd.options)
    assert nd.instruction  # references the entity


def test_needle_targets_requested_length():
    from myna.tokenizer import encode_text
    tok = _tok()
    rng = random.Random(1)
    for target in (128, 1024, 2048):
        nd = make_needle(tok, target, rng)
        n = len(encode_text(tok, nd.state))
        # filler is added whole-sentence until >= target, so overshoot is bounded
        assert n >= target
        assert n - target < 30, (n, target)

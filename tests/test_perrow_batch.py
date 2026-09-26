"""Per-row question tensors (SPEC P2) — the batch contract that unblocks boolq/mnli.

The property under test is *row independence*: a row's logits must not move
when it joins a batch whose other rows ask different questions, and a
homogeneous batch must score exactly as it did when one tensor set was shared
across the batch. Both directions are pinned, plus the two samplers the
training loop uses to reach a mixed batch.

The torch-vs-MLX parity test is the witness for the shared form: `mlx_model`
still implements the pre-P2 broadcast math, so any drift in the broadcast path
shows up there.
"""

from __future__ import annotations

import random

import pytest
import torch

from myna.data import Example, Question
from myna.model import MynaConfig, MynaModel, typed_loss
from myna.real_data import flatten_groups
from myna.tokenizer import batch_question_tensors, question_tensors, train_tokenizer
from myna.train import build_batch, build_row_batch, draw_row_batch, question_tokens

CFG = MynaConfig(vocab=128, d_model=32, n_layers=2, n_heads=2, d_k=8, d_v=8, d_ff=48, d_ptr=16)
ATOL = 1e-6

# four shapes a real suite mixes into one train split: multi-question rows, a
# many-option row, and a row whose instruction text is unusually long.
SCHEMATA = {
    "triage": [
        Question("dept", "choice", "Which team should handle this ticket?",
                 ["billing", "technical", "shipping"]),
        Question("urgent", "noul", "Does the writer need help today?", ["no", "yes"]),
    ],
    "many": [
        Question("intent", "choice", "Pick the intent of this message.",
                 [f"intent number {i} about the account" for i in range(11)]),
    ],
    "long": [
        Question("sent", "choice",
                 "Read the whole message carefully and say whether the customer sounds "
                 "pleased with the service they received or not at all",
                 ["positive", "negative"]),
    ],
    "single": [Question("t", "score", "How severe?", ["low", "high"])],
}
STATES = {
    "triage": "the invoice on my card is wrong twice, please sort the billing out today",
    "many": "I want to move money and then check where my package is",
    "long": "absolutely lovely service, the courier even called before arriving",
    "single": "the whole checkout is down and we are losing money right now",
}


def _tok():
    texts = [q.instruction for qs in SCHEMATA.values() for q in qs]
    texts += [o for qs in SCHEMATA.values() for q in qs for o in q.options]
    texts += list(STATES.values())
    return train_tokenizer(texts, vocab_size=200)


TOK = _tok()


def _row(name, gold):
    return SCHEMATA[name], Example(STATES[name], name, gold)


def _model():
    torch.manual_seed(7)
    m = MynaModel(CFG)
    m.eval()
    return m


def _forward(m, b):
    with torch.no_grad():
        return m(b["state_ids"], b["state_len"], b["q_ids"], b["q_mask"],
                 b["span_mat"], b["opt_valid"], b["decide_idx"])


def _per_row(m, items):
    """Each row scored alone — the reference a mixed batch must reproduce."""
    return [_forward(m, build_row_batch([it], TOK, "cpu")) for it in items]

def test_batch_question_tensors_pad_to_the_batch_maximum():
    items = [_row("triage", (0, 1)), _row("many", (4,)), _row("single", (1,))]
    qt = batch_question_tensors(TOK, [[(q.instruction, q.options) for q in qs] for qs, _ in items])
    n_q = max(len(qs) for qs, _ in items)
    n_opt = max(len(q.options) for qs, _ in items for q in qs)
    assert qt["q_ids"].shape[0] == len(items) and qt["q_ids"].shape[1] == n_q
    assert qt["span_mat"].shape[1:] == (n_q, n_opt, qt["q_ids"].shape[2])
    assert qt["opt_valid"].shape == (len(items), n_q, n_opt)
    for n, (qs, _) in enumerate(items):
        for k, q in enumerate(qs):
            assert bool(qt["opt_valid"][n, k, : len(q.options)].all())
            assert not qt["opt_valid"][n, k, len(q.options):].any()
            for o in range(len(q.options)):
                # every option span pools its own tokens, and nothing else
                assert float(qt["span_mat"][n, k, o].sum()) == pytest.approx(1.0)
            assert qt["decide_idx"][n, k] == int(qt["q_mask"][n, k].sum()) - 1
        for k in range(len(qs), n_q):
            assert qt["q_mask"][n, k].sum() == 0, "spare question slots stay empty"


def test_shared_and_per_row_forms_of_one_set_agree():
    """The spec's equivalence box: a homogeneous batch scores identically with
    and without the leading batch dim."""
    m = _model()
    items = [_row("triage", (1, 0)), _row("triage", (0, 1)), _row("triage", (2, 0))]
    questions = SCHEMATA["triage"]
    a = _forward(m, build_batch([ex for _q, ex in items], TOK, questions, "cpu"))
    c = _forward(m, build_row_batch(items, TOK, "cpu"))
    assert a.shape == c.shape
    assert torch.allclose(a, c, atol=ATOL), (a - c).abs().max()


def test_rows_from_different_question_sets_do_not_interfere():
    m = _model()
    items = [_row("triage", (0, 1)), _row("many", (4,)), _row("long", (1,)), _row("single", (0,))]
    mixed = _forward(m, build_row_batch(items, TOK, "cpu"))
    for n, solo in enumerate(_per_row(m, items)):
        qs = items[n][0]
        n_opt = max(len(q.options) for q in qs)
        got = mixed[n, : len(qs), :n_opt]
        want = solo[0, : len(qs), :n_opt]
        assert torch.allclose(got, want, atol=ATOL), f"row {n} moved by {(got - want).abs().max()}"


def test_padding_produces_no_nans_and_no_logits():
    m = _model()
    items = [_row("triage", (0, 1)), _row("single", (1,))]
    b = build_row_batch(items, TOK, "cpu")
    out = _forward(m, b)
    assert out[1, 1:].numel() > 0, "row 1 must actually be padded on the question axis"
    assert torch.isfinite(out[b["opt_valid"]]).all()
    assert (out[~b["opt_valid"]] == -1e9).all(), "padded option slots stay masked out"


def test_loss_over_a_mixed_batch_is_the_cell_weighted_mean_of_its_rows():
    items = [_row("triage", (0, 1)), _row("many", (4,)), _row("single", (0,))]
    m = _model()
    b = build_row_batch(items, TOK, "cpu")
    assert [int(h.sum()) for h in b["has_gold"]] == [len(qs) for qs, _ in items]
    for n, (qs, ex) in enumerate(items):
        assert b["gold"][n, : len(qs)].tolist() == list(ex.gold)
        assert not b["has_gold"][n, len(qs):].any()
    mixed = typed_loss(_forward(m, b), b["gold"], b["has_gold"]).item()
    parts = []
    for it in items:
        solo = build_row_batch([it], TOK, "cpu")
        cells = int(solo["has_gold"].sum())
        parts.append((typed_loss(_forward(m, solo), solo["gold"], solo["has_gold"]).item(), cells))
    want = sum(l * c for l, c in parts) / sum(c for _l, c in parts)
    assert abs(mixed - want) < 1e-5, (mixed, want)
    assert mixed > 0


def test_flatten_groups_is_lossless():
    groups = {"a#1": (SCHEMATA["triage"], [Example(STATES["triage"], "a", (0, 1))]),
              "b#1": (SCHEMATA["single"], [Example(STATES["single"], "b", (1,))])}
    flat = flatten_groups(groups)
    assert [(qs, ex.state) for qs, ex in flat] == [
        (SCHEMATA["triage"], STATES["triage"]), (SCHEMATA["single"], STATES["single"])]


def _items(k=6):
    """A stand-in train split: the four schemata interleaved, every row distinct."""
    out = []
    for i in range(k):
        for name, gold in (("triage", (0, 1)), ("many", (4,)), ("long", (1,)), ("single", (0,))):
            out.append((SCHEMATA[name], Example(f"{STATES[name]} variant {i}", name, gold)))
    return out


def test_question_tokens_is_the_padding_the_batch_will_get():
    """The budget is only honest if the counter matches what build_question
    actually lays out: instruction + one [OPT] per option + [DECIDE]."""
    for name, qs in SCHEMATA.items():
        qt = question_tensors(TOK, [(q.instruction, q.options) for q in qs])
        assert question_tokens(TOK, qs) == int(qt["q_mask"].sum(-1).max()), name


def test_question_tokens_cache_hits_and_never_changes_the_answer():
    cache = {}
    items = _items(2)
    first = [question_tokens(TOK, qs, cache) for qs, _ in items]
    assert len(cache) == len(SCHEMATA) < len(items), "one entry per question set, not per row"
    assert [question_tokens(TOK, qs, cache) for qs, _ in items] == first


def _padded_cells(picked, tok=TOK):
    """What the picked batch really costs: rows x widest question count x
    longest question, both padded to the batch maximum."""
    return (len(picked) * max(len(qs) for qs, _ in picked)
            * max(question_tokens(tok, qs) for qs, _ in picked))


def test_draw_row_batch_budgets_the_question_branch():
    budget = 250
    picked = draw_row_batch(_items(), batch=8, max_q_cells=budget, rng=random.Random(0), tok=TOK)
    assert picked
    assert len({ex.state for _q, ex in picked}) == len(picked), "rows must be distinct"
    assert _padded_cells(picked) <= budget
    assert len(picked) < 8, "the budget must actually deny rows, not just rename them"
    # with an unlimited budget a full batch goes in
    assert len(draw_row_batch(_items(), batch=8, max_q_cells=10**6, rng=random.Random(1), tok=TOK)) == 8


def test_the_budget_tracks_the_widest_row_not_the_last_picked_one():
    """A sampler that compares against the previous row rather than the running
    maximum lets a 110-token question hide behind a 13-token one and books the
    batch for a fifth of what it will actually allocate."""
    # 700 cells: wide enough that any single row fits (draw_row_batch always
    # accepts its first row), tight enough that only two of these four do.
    budget = 700
    for seed in range(12):
        picked = draw_row_batch(_items(1), batch=4, max_q_cells=budget,
                                rng=random.Random(seed), tok=TOK)
        assert _padded_cells(picked) <= budget, (seed, _padded_cells(picked), len(picked))
        assert len(picked) < 4, seed


def test_a_budget_below_every_row_still_returns_one_row():
    picked = draw_row_batch(_items(1), batch=8, max_q_cells=1, rng=random.Random(0), tok=TOK)
    assert len(picked) == 1, "never empty — but nothing else may join a batch already over budget"
    assert build_row_batch(picked, TOK, "cpu")["gold"].shape == (1, len(picked[0][0]))

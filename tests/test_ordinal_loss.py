"""The ordinal cost for `score` cells (SPEC §5 P9: 9a).

`typed_loss` over an ordered legend is the one component in the stack that does not
believe the legend is ordered: `data.py` documents `options` as the ordinal sequence
and both the engine and the MLX bench read a `score` as the index-weighted
expectation, while cross-entropy charges "predicted 4, gold 3" exactly what it
charges "predicted 1, gold 5". So the tests here are shaped around that sentence
rather than around the new formula's arithmetic:

* the defect itself, pinned — two wrong predictions with the *same* probability on
  the gold rung, one a neighbour and one the far end, must be identical under `ce`
  (which is the bug) and strictly ordered under `emd`;
* the normalisation, pinned by its consequences — the worst reachable price is
  1.0 for every legend length, a one-rung miss is `1/(K-1)`, and at `K = 2` the
  number *is* the Brier score of the second option, which is the statistic §9.24
  already demands beside a noul argmax;
* the padded tail, pinned against the shortcut — the divisor is the cell's own `K`,
  not the batch's `O`, so a five-option legend sitting in a six-option batch must
  cost exactly what it costs alone;
* `ce` as the default, pinned byte-for-byte, because every published figure in this
  repo was trained with it and nothing here is allowed to move one.

`ref_loss` below is an independent implementation — softmax, a Python loop over
cells, the CDF built only up to the cell's own option count — so it agrees with
`typed_loss` for a reason other than being the same expression.
"""

from __future__ import annotations

import json
import math
import sys

import pytest
import torch
import torch.nn.functional as F

from myna.data import Example, Question
from myna.model import typed_loss
from myna.tokenizer import train_tokenizer
from myna.train import build_batch, build_row_batch

INF = -1e9  # what model.py's masked_fill puts on a padded option slot


def logits_from(probs):
    """logits whose softmax reproduces `probs` (which must sum to 1) over exactly
    `len(probs)` options.

    A zero-probability option is not absent, it is masked, so it gets the same -1e9
    `model.py`'s masked_fill puts there — that is what makes the tail terms fall out
    of the CDF sum instead of leaking into it.
    """
    p = torch.tensor(probs, dtype=torch.float32)
    return torch.where(p > 0, torch.log(p.clamp(min=1e-12)), torch.full_like(p, INF))


def mask(O, K, B=1, N=1):
    """opt_valid [B,N,O]: the cell has K real options padded to O."""
    ov = torch.zeros(B, N, O, dtype=torch.bool)
    ov[:, :, :K] = True
    return ov


def cell(lg, K, score=True, gold=0):
    """One cell as the loss sees it: [1,1,O] logits plus the three masks."""
    return (lg[None, None], torch.tensor([[gold]]), torch.ones(1, 1, dtype=torch.bool),
            torch.tensor([[score]]), mask(len(lg), K))


def ref_loss(logits, gold, has_gold, ordinal, opt_valid, score_loss):
    """The loss as an independent reader would compute it: one cell at a time."""
    p = F.softmax(logits, dim=-1)
    tot, cnt = 0.0, 0
    for b in range(gold.shape[0]):
        for n in range(gold.shape[1]):
            if not bool(has_gold[b, n]):
                continue
            cnt += 1
            g = int(gold[b, n])
            if score_loss == "ce" or not bool(ordinal[b, n]):
                tot += -math.log(float(p[b, n, g]))
                continue
            K = int(opt_valid[b, n].sum())
            fp = torch.cumsum(p[b, n, :K], 0)
            fg = (torch.arange(K) >= g).float()
            tot += float(((fp - fg) ** 2).sum()) / max(K - 1, 1)
    return tot / cnt


# --- the defect: cross-entropy cannot see the axis -----------------------------

def test_ce_cannot_tell_a_one_rung_miss_from_the_opposite_end():
    """gold 3 over a six-rung legend; the far mass is on 0 in one, on 4 in the
    other, and p(gold) is 0.1 in both."""
    near = logits_from([0.0, 0.0, 0.0, 0.1, 0.9, 0.0])
    far = logits_from([0.9, 0.0, 0.0, 0.1, 0.0, 0.0])
    a, b = cell(near, 6, gold=3), cell(far, 6, gold=3)
    ce_near, ce_far = float(typed_loss(*a, "ce")), float(typed_loss(*b, "ce"))
    emd_near, emd_far = float(typed_loss(*a, "emd")), float(typed_loss(*b, "emd"))

    assert ce_near == ce_far, "the nominal cost is blind to the axis: that is 9a's bug"
    assert emd_near < emd_far
    assert emd_near == pytest.approx(0.81 / 5, abs=1e-5)
    assert emd_far == pytest.approx(2.43 / 5, abs=1e-5)
    assert ce_near == pytest.approx(-math.log(0.1), abs=1e-4)


def test_emd_is_zero_when_the_prediction_sits_on_the_gold_rung():
    for K, g in ((2, 0), (2, 1), (6, 3), (6, 5), (11, 4)):
        lg = torch.where(torch.arange(K) == g, 30.0, -30.0)
        got = float(typed_loss(*cell(lg, K, gold=g), "emd"))
        assert got == pytest.approx(0.0, abs=1e-6), f"K={K} gold={g} priced {got}"


# --- the normalisation: one scale for every legend -----------------------------

def test_the_worst_price_is_one_and_does_not_depend_on_the_legend_length():
    """Delta at the opposite end is 1.0 for every K, because the divisor is the
    cell's own K-1 — that is the whole content of "one scale" (SPEC §5 P9: 9a)."""
    for K in range(2, 9):
        lg = logits_from([1.0] + [0.0] * (K - 1))
        got = float(typed_loss(*cell(lg, K, gold=K - 1), "emd"))
        assert got == pytest.approx(1.0, abs=1e-5), f"K={K} worst {got}"


def test_a_one_rung_miss_costs_the_reciprocal_of_the_legend():
    """gold 1, the mass on 2: exactly one CDF rung is wrong, so the price is
    1/(K-1) — it falls as the legend grows, which is the ordering the loss exists
    to express."""
    for K in range(3, 9):
        lg = logits_from([0.0, 0.1, 0.9] + [0.0] * (K - 3))
        got = float(typed_loss(*cell(lg, K, gold=1), "emd"))
        assert got == pytest.approx(0.81 / (K - 1), abs=1e-5), f"K={K} got {got}"


def test_a_two_option_cell_is_the_brier_score_of_the_second_option():
    """The identity that makes the divisor right rather than cosmetic: at K=2 the
    Cramér sum is (p[1] - y[1])^2, the binary Brier score §9.24 insists on quoting."""
    gen = torch.Generator().manual_seed(11)
    for g in (0, 1):
        lg = torch.randn(8, 1, 2, generator=gen)
        gold = torch.full((8, 1), g, dtype=torch.int64)
        hg = torch.ones(8, 1, dtype=torch.bool)
        ord_ = torch.ones(8, 1, dtype=torch.bool)
        got = typed_loss(lg, gold, hg, ord_, mask(2, 2, 8), "emd")
        p1 = F.softmax(lg, -1)[:, 0, 1]
        want = (p1 - float(g)) ** 2
        assert float(got) == pytest.approx(float(want.mean()), abs=1e-6)


def test_a_short_cell_costs_the_same_inside_a_long_batch():
    """The divisor is the cell's own K, not the batch's O: price it by O-1 and a
    five-option legend sitting in a six-option batch comes out at 4/5 of itself,
    which is a 20% discount for sharing a batch with a longer question."""
    probs = [0.0, 0.1, 0.0, 0.0, 0.9]  # gold 1, the far mass on 4 over a 5-rung legend
    alone = logits_from(probs)
    padded = torch.cat([alone, torch.full((1,), INF)])
    args = cell(alone, 5, gold=1)
    b = (padded[None, None], args[1], args[2], args[3], mask(6, 5))
    a, c = float(typed_loss(*args, "emd")), float(typed_loss(*b, "emd"))
    assert a == pytest.approx(c, abs=1e-7)
    assert c == pytest.approx(2.43 / 4, abs=1e-5)
    assert c != pytest.approx(2.43 / 5, abs=1e-3), "that is the O-1 divisor, not K-1"


# --- ce stays exactly what it was ----------------------------------------------

def test_the_default_is_the_published_objective_byte_for_byte():
    gen = torch.Generator().manual_seed(3)
    lg = torch.randn(4, 3, 7, generator=gen) - 0.5
    gold = torch.randint(0, 6, (4, 3), generator=gen)
    hg = torch.ones(4, 3, dtype=torch.bool)
    ord_ = torch.tensor([[True, False, True]] * 4)
    ov = mask(7, 6, 4, 3)
    manual = -F.log_softmax(lg, -1).gather(-1, gold[..., None]).squeeze(-1)[hg].mean()
    assert typed_loss(lg, gold, hg) == manual            # the 3-arg form callers use
    assert typed_loss(lg, gold, hg, ord_, ov) == manual  # masks present, ce default
    assert typed_loss(lg, gold, hg, ord_, ov, "ce") == manual
    assert typed_loss(lg, gold, hg, ord_, ov, "emd").item() != float(manual)


def test_only_score_cells_are_repriced():
    gen = torch.Generator().manual_seed(5)
    lg = torch.randn(3, 4, 6, generator=gen)
    gold = torch.randint(0, 5, (3, 4), generator=gen)
    hg = torch.ones(3, 4, dtype=torch.bool)
    ov = mask(6, 5, 3, 4)
    for row in ([False] * 4, [True] * 4, [True, False, False, True]):
        ord_ = torch.tensor([row] * 3)
        got = float(typed_loss(lg, gold, hg, ord_, ov, "emd"))
        want = ref_loss(lg, gold, hg, ord_, ov, "emd")
        assert got == pytest.approx(want, abs=1e-5)
    all_choice = torch.zeros(3, 4, dtype=torch.bool)
    assert float(typed_loss(lg, gold, hg, all_choice, ov, "emd")) == pytest.approx(
        float(typed_loss(lg, gold, hg)), abs=1e-7)


def test_the_mean_is_over_cells_not_over_rows():
    """Row 0 holds two cheap cells and row 1 one expensive one plus a padded slot.

    Mean over the three live cells is 0.5; mean over the two rows' means is 0.625;
    counting the gold-less slot at all gives 0.4375. The batch is rectangular and
    the loss is not, which is the whole reason `has_gold` exists.
    """
    half, far = math.log(0.5), INF
    lg = torch.tensor([[[half, half], [half, half]], [[far, 0.0], [0.0, 0.0]]])
    gold = torch.tensor([[1, 1], [0, 0]])
    hg = torch.tensor([[True, True], [True, False]])
    ord_ = torch.ones(2, 2, dtype=torch.bool)
    ov = mask(2, 2, 2, 2)
    got = float(typed_loss(lg, gold, hg, ord_, ov, "emd"))
    assert got == pytest.approx(0.5, abs=1e-6)
    assert got == pytest.approx(ref_loss(lg, gold, hg, ord_, ov, "emd"), abs=1e-6)
    assert got != pytest.approx(0.625, abs=1e-3), "that is the mean of the two row means"
    assert got != pytest.approx(0.4375, abs=1e-3), "that is the padded slot counted as a cell"


def test_emd_refuses_a_caller_that_has_not_plumbed_the_masks():
    """A silently-degraded objective is worse than a crash: `emd` that fell back to
    `ce` would print a run claiming an ordinal loss it never used."""
    lg = torch.randn(2, 2, 4)
    gold = torch.zeros(2, 2, dtype=torch.int64)
    hg = torch.ones(2, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="ordinal mask"):
        typed_loss(lg, gold, hg, score_loss="emd")
    with pytest.raises(ValueError, match="ordinal mask"):
        typed_loss(lg, gold, hg, torch.ones(2, 2, dtype=torch.bool), score_loss="emd")


def test_the_shared_and_per_row_forms_agree():
    """`ordinal` and `opt_valid` follow the question tensors' broadcast contract:
    [N] and [N,O] are the shared form, and they must price the same as [B,N]."""
    gen = torch.Generator().manual_seed(9)
    lg = torch.randn(4, 3, 5, generator=gen)
    gold = torch.randint(0, 4, (4, 3), generator=gen)
    hg = torch.ones(4, 3, dtype=torch.bool)
    ord_shared = torch.tensor([True, False, True])
    ov_shared = torch.zeros(3, 5, dtype=torch.bool)
    ov_shared[:, :4] = True
    shared = typed_loss(lg, gold, hg, ord_shared, ov_shared, "emd")
    per_row = typed_loss(lg, gold, hg, ord_shared.expand(4, -1), ov_shared.expand(4, -1, -1), "emd")
    assert float(shared) == float(per_row)
    assert float(shared) == pytest.approx(
        ref_loss(lg, gold, hg, ord_shared.expand(4, -1), ov_shared.expand(4, -1, -1), "emd"),
        abs=1e-5)


# --- the batch builders carry the mask -----------------------------------------

SCHEMATA = {
    "stars": [Question("stars", "score", "How many stars?",
                       ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]),
              Question("urgent", "noul", "Does the writer need help today?", ["no", "yes"])],
    "single": [Question("severity", "score", "How severe?", ["low", "high"])],
    "dept": [Question("dept", "choice", "Which team?", ["billing", "technical", "shipping"])],
}
STATES = {k: f"the invoice for the {k} case is wrong twice over" for k in SCHEMATA}
TOK = train_tokenizer([q.instruction for qs in SCHEMATA.values() for q in qs]
                      + [o for qs in SCHEMATA.values() for q in qs for o in q.options]
                      + list(STATES.values()), vocab_size=200)


def test_build_batch_marks_the_score_questions_only():
    qs = SCHEMATA["stars"]
    exs = [Example(STATES["stars"], "stars", (3, 1)), Example(STATES["stars"], "stars", (1, 0))]
    b = build_batch(exs, TOK, qs, "cpu")
    assert b["ordinal"].dtype == torch.bool and b["ordinal"].shape == (2,)
    assert b["ordinal"].tolist() == [True, False]
    assert b["ordinal"].shape[0] == b["gold"].shape[1]


def test_build_row_batch_marks_score_cells_and_leaves_padding_false():
    items = [(SCHEMATA["stars"], Example(STATES["stars"], "stars", (2, 0))),
             (SCHEMATA["dept"], Example(STATES["dept"], "dept", (1,))),
             (SCHEMATA["single"], Example(STATES["single"], "single", (1,)))]
    b = build_row_batch(items, TOK, "cpu")
    assert b["ordinal"].shape == b["has_gold"].shape == (3, 2)
    # row 0: score + noul; row 1: one choice; row 2: one score. Padded slots False.
    assert b["ordinal"].tolist() == [[True, False], [False, False], [True, False]]
    assert b["has_gold"].tolist() == [[True, True], [True, False], [True, False]]
    assert int((~b["has_gold"] & b["ordinal"]).sum()) == 0, \
        "a padded cell must not claim to be ordinal"


def test_the_builder_masks_feed_the_loss_in_the_order_the_loop_passes_them():
    """The call site is `typed_loss(logits, gold, has_gold, ordinal, opt_valid, flag)`;
    swap two of those and the prices are nonsense, so pin the convention itself."""
    items = [(SCHEMATA["stars"], Example(STATES["stars"], "stars", (2, 0))),
             (SCHEMATA["single"], Example(STATES["single"], "single", (1,)))]
    b = build_row_batch(items, TOK, "cpu")
    lg = torch.randn(2, 2, 5)
    gold = torch.tensor([[2, 0], [1, 0]])
    hg = b["has_gold"]
    for flag in ("ce", "emd"):
        got = typed_loss(lg, gold, hg, b["ordinal"], b["opt_valid"], flag)
        assert torch.isfinite(got) and float(got) > 0
        assert float(got) == pytest.approx(
            ref_loss(lg, gold, hg, b["ordinal"], b["opt_valid"], flag), abs=1e-5)


# --- the flag and the masks reach the loop's call site -------------------------

TRAIN_ROWS = [
    {"state": {"document": s}, "questions": {"stars": {
        "type": "score", "instructions": ["How many stars did this deserve?"],
        "criteria": ["1 star: terrible", "2 stars: poor", "3 stars: okay",
                     "4 stars: good", "5 stars: excellent"],
        "label": i}}, "_meta": {"source": "yelp"}}
    for s, i in [("the broth was salt water and the noodles were cold", 0),
                 ("best bowl of the year, i went back twice", 4),
                 ("fine food, nothing to praise or complain about", 2),
                 ("friendly staff but the portion was tiny", 1)]
]


def _write_suite(dirpath):
    dirpath.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", TRAIN_ROWS), ("development", TRAIN_ROWS[:2]),
                       ("test", TRAIN_ROWS[:2]), ("calibration", TRAIN_ROWS[:2])):
        with (dirpath / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return dirpath


def _one_update(tmp_path, monkeypatch, argv_extra):
    """Run the real loop for one update and record every typed_loss call site.

    The trainer does not seed torch, so two processes never share an init and "the
    printed loss changed" would pass even if the flag did nothing. Spying on the
    call site prices the actual claim: the loop hands `typed_loss` the flag it was
    given and the masks that make it mean anything.
    """
    import myna.train as T

    seen = []
    real = T.typed_loss

    def spy(logits, gold, has_gold, ordinal=None, opt_valid=None, score_loss="ce"):
        seen.append({"gold": gold.shape, "has_gold": has_gold.clone(), "ordinal": ordinal,
                     "opt_valid": opt_valid, "score_loss": score_loss})
        return real(logits, gold, has_gold, ordinal, opt_valid, score_loss)

    monkeypatch.setattr(T, "typed_loss", spy)
    monkeypatch.setattr(sys, "argv", [
        "myna.train", "--suite", str(_write_suite(tmp_path / "suite")),
        "--out", str(tmp_path / "out"), "--device", "cpu", "--steps", "1", "--batch", "2",
        "--vocab", "128", "--seed", "0", "--eval-every", "0", *argv_extra])
    T.main()
    return seen


def _call(seen):
    assert len(seen) == 1, f"one update, one call: got {len(seen)}"
    return seen[0]


def test_the_loop_passes_the_flag_it_was_given(tmp_path, monkeypatch, capsys):
    """Three invocations, one assertion each: the explicit values arrive intact, and
    no flag at all means `ce`, which is the clause "no published figure moves".
    """
    cases = [(["--score-loss", "ce"], "ce"), ([], "ce"), (["--score-loss", "emd"], "emd")]
    for i, (argv, want) in enumerate(cases):
        capsys.readouterr()
        call = _call(_one_update(tmp_path / str(i), monkeypatch, argv))
        assert call["score_loss"] == want, f"{argv} reached the loss as {call['score_loss']!r}"
        assert call["ordinal"] is not None and call["opt_valid"] is not None, \
            "emd without the masks raises inside typed_loss, so ce-only is what ran"


def test_the_score_cells_are_marked_in_the_real_batch(tmp_path, monkeypatch, capsys):
    """The suite's one question is a `score`, so an ordinal mask that was plumbed but
    empty would still run and still print a loss — it just would not be emd."""
    capsys.readouterr()
    call = _call(_one_update(tmp_path / "emd", monkeypatch, ["--score-loss", "emd"]))
    assert bool(call["ordinal"].any()), "no cell was marked ordinal: emd priced nothing"
    assert call["ordinal"].dtype == torch.bool
    assert call["ordinal"].dim() == 1, "the shared path builds [N], like the question tensors"
    assert call["ordinal"].shape[-1] == call["gold"][1]
    assert call["opt_valid"].dim() == 2 and call["opt_valid"].shape[-2] == call["gold"][1]


def test_the_row_batch_path_passes_per_row_masks(tmp_path, monkeypatch, capsys):
    capsys.readouterr()
    call = _call(_one_update(tmp_path / "row", monkeypatch,
                             ["--score-loss", "emd", "--row-batch"]))
    assert call["score_loss"] == "emd"
    assert call["ordinal"].dim() == 2 and tuple(call["ordinal"].shape) == call["gold"]
    assert bool(call["ordinal"].any())


def test_long_context_refuses_a_loss_it_cannot_price(tmp_path, monkeypatch, capsys):
    """The needle corpus has no score cells and `build_needle_batch` carries no
    ordinal mask, so `--score-loss emd` there would train on ce and print a run
    that claims an ordinal objective it never used."""
    monkeypatch.setattr(sys, "argv", ["myna.train", "--long-context", "512",
                                      "--score-loss", "emd", "--device", "cpu",
                                      "--steps", "1", "--out", str(tmp_path / "out")])
    import myna.train as T

    with pytest.raises(SystemExit) as got:
        T.main()
    assert "no score cells" in str(got.value)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

"""Needle corpus + recall-curve invariants (CPU, no model).

The generator's two jobs are to keep the needle the *only* evidence and to actually
produce the length it was asked for — a "16k" row built from a 9k state would be the
kind of claim that survives review because nobody counts the tokens.

`eval_lengths` is the other half, and it is what prints G4's accuracy column, so it
is tested against stub engines whose answers are known: a perfect reader has to read
1.000, a reader that ignores the state and always answers the same desk must land on
the share of golds that happen to be that desk (and *not* on 1.000), and the sample
size has to be what it claims. A curve whose scoring is wrong decays the same way
whether the model reads or not, which is exactly how a bug becomes a result.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.eval_needle import baseline_verdict  # noqa: E402
from myna.longctx import DESKS, FILLER, eval_lengths, make_needle  # noqa: E402
from myna.tokenizer import encode_text, train_tokenizer  # noqa: E402


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
    tok = _tok()
    rng = random.Random(1)
    for target in (128, 1024, 2048, 16384):
        nd = make_needle(tok, target, rng)
        n = len(encode_text(tok, nd.state))
        # filler is added whole-sentence until >= target, so overshoot is bounded
        assert n >= target
        assert n - target < 30, (n, target)


# ------------------------------------------------------------- the recall curve
class Stub:
    """An engine that answers `desk` from a rule, never from a gold label."""

    def __init__(self, tok):
        self.tok = tok
        self.calls = 0

    def _opts(self, questions):
        return list(next(iter(questions.values()))["criteria"])

    def _answer(self, probs, opts):
        return {"answers": {"desk": {"probabilities": probs}},
                "usage": {}, "policy": {"abstain_below": None, "abstained": []}}


class Reader(Stub):
    """Recovers the needle by string search — which is the point: if a state-reader
    cannot score 1.000 here, either the gold is not in the state or the harness is
    not comparing the same keys."""

    def predict(self, state, questions):
        self.calls += 1
        opts = self._opts(questions)
        (own,) = [d for d in opts if f"the {d} desk" in state]
        n = len(opts)
        return self._answer({d: (0.9 if d == own else 0.1 / (n - 1)) for d in opts}, opts)


class Constant(Stub):
    def predict(self, state, questions):
        self.calls += 1
        opts = self._opts(questions)
        n = len(opts)
        return self._answer({d: (0.9 if d == opts[0] else 0.1 / (n - 1)) for d in opts}, opts)


def golds(tok, lengths, n_per_length, seed=0):
    """Re-derive the gold sequence the way `eval_lengths` walks it, so the stub
    accuracies can be checked against an exact share rather than a plausibility."""
    rng = random.Random(seed)
    return [make_needle(tok, L, rng).gold for L in lengths for _ in range(n_per_length)]


def test_a_state_reader_scores_one_at_every_length():
    tok = _tok()
    rows = eval_lengths(Reader(tok), [128, 512], n_per_length=6, seed=3)
    assert [r["acc"] for r in rows] == [1.0, 1.0], rows
    assert [r["tokens"] for r in rows] == [128, 512]


def test_the_denominator_is_the_sample_the_harness_asked_for():
    tok = _tok()
    stub = Reader(tok)
    eval_lengths(stub, [128, 256, 512], n_per_length=7, seed=4)
    assert stub.calls == 21, "a row that skipped a question shrinks the sample silently"


def test_ignoring_the_state_lands_on_a_share_not_on_a_result():
    """The same needles, the same scorer, a model that answers one desk every time.

    Its accuracy must equal the share of that desk in the gold sequence — computed
    here from the generator, not asserted from a guess. If `eval_lengths` compared
    the wrong key, or dropped rows from the denominator, this equality breaks."""
    tok = _tok()
    lengths, n = [128, 512], 8
    g = golds(tok, lengths, n, seed=5)
    rows = eval_lengths(Constant(tok), lengths, n_per_length=n, seed=5)
    # one row per length, scored over that length's own needles — the gold is drawn
    # at the top of `make_needle`, but the filler loop consumes a different number
    # of draws per length, so the slices are not interchangeable.
    expect = [sum(1 for x in g[i * n:(i + 1) * n] if x == 0) / n for i in range(len(lengths))]
    assert [r["acc"] for r in rows] == pytest.approx(expect)
    assert max(expect) < 1.0, "the sample has no variance — the test proves nothing"


def test_a_missing_answer_is_a_crash_not_a_wrong_answer():
    """A dropped question that scored as "incorrect" would depress the curve and
    look like the model failing at long range."""
    class Silent(Stub):
        def predict(self, state, questions):
            return {"answers": {}, "usage": {}, "policy": {}}

    with pytest.raises(KeyError):
        eval_lengths(Silent(_tok()), [128], n_per_length=2, seed=6)


def test_the_curve_records_a_time_per_row_and_keeps_its_units():
    tok = _tok()
    rows = eval_lengths(Reader(tok), [128], n_per_length=3, seed=7)
    assert rows[0]["ms"] > 0, "a zero mean latency is a timer that never ran"


def test_the_needle_lands_everywhere_in_the_state_not_just_at_the_ends():
    """G4's whole question is *long-range* recall: a needle parked at the end of
    every state would measure how well the model reads its suffix, and the curve
    would look flat while testing nothing. The harness prints "at a random
    position", so the positions are checked to actually spread.
    """
    tok = _tok()
    rng = random.Random(21)
    frac, instructions = [], set()
    for _ in range(40):
        nd = make_needle(tok, 1024, rng)
        m = re.search(r"owned by the (.+?) desk\.", nd.state)
        assert m, nd.state[:120]
        frac.append(m.start() / len(nd.state))
        instructions.add(nd.instruction)
    assert min(frac) < 0.25 and max(frac) > 0.75, (min(frac), max(frac))
    assert len({round(f, 1) for f in frac}) >= 5, f"only {len(set(frac))} distinct spots"
    # the entity varies too: one fixed entity would let a checkpoint answer the
    # whole sweep from a single memorised question tensor
    assert len(instructions) > 1, instructions


def test_the_curve_says_when_it_cannot_be_read_as_decay():
    """`baseline_verdict` is the guard between a low curve and a meaningless one:
    accuracy that starts at the uniform floor cannot decay, it was never recall."""
    at_chance = [{"tokens": 128, "acc": 0.1875, "ms": 80.0},
                 {"tokens": 16384, "acc": 0.0625, "ms": 7000.0}]
    v = baseline_verdict(at_chance, 16, chance=1 / 6)
    assert v["readable"] is False
    assert "128" in v["note"] and "0.167" in v["note"]
    assert v["shortest_rung"] == 128 and v["acc"] == 0.1875

    readable = [{"tokens": 128, "acc": 0.94, "ms": 80.0},
                {"tokens": 16384, "acc": 0.86, "ms": 7000.0}]
    ok = baseline_verdict(readable, 16, chance=1 / 6)
    assert ok["readable"] is True and ok["note"] is None


def test_the_guard_needs_a_margin_not_a_hair():
    """0.17 vs a 0.167 floor is one sample of luck at n=16; calling that readable
    would license the whole decay reading off noise. The boundary is inclusive: a
    rung exactly `chance + margin` above the floor counts."""
    barely = baseline_verdict([{"tokens": 128, "acc": 0.17, "ms": 1.0}], 16, chance=1 / 6)
    assert barely["readable"] is False
    cleared = baseline_verdict([{"tokens": 128, "acc": 0.27, "ms": 1.0}], 16, chance=1 / 6)
    assert cleared["readable"] is True
    # 0.25 + 0.125 is exactly representable, so this rung sits on the line itself
    # rather than a float's approximation of it: inclusive means inclusive.
    on_the_line = baseline_verdict([{"tokens": 128, "acc": 0.375, "ms": 1.0}], 16,
                                  chance=0.25, margin=0.125)
    assert on_the_line["acc"] == on_the_line["chance"] + 0.125
    assert on_the_line["readable"] is True, "the margin is a floor, not a wall"


def test_the_reported_standard_error_is_the_sample_size_one():
    p = 1 / 6
    v = baseline_verdict([{"tokens": 128, "acc": 0.2, "ms": 1.0}], 40, chance=p)
    assert v["one_se"] == pytest.approx((p * (1 - p) / 40) ** 0.5, abs=1e-4)
    assert v["one_se"] < baseline_verdict([{"tokens": 128, "acc": 0.2, "ms": 1.0}],
                                         10, chance=p)["one_se"]


def test_the_default_ladder_reaches_the_length_the_window_claim_makes():
    """§2.1's window row says 16,384. The recall curve is the sentence that has to
    stand behind a length claim, so its default ladder has to contain that rung —
    otherwise the published sweep quietly stops one notch short."""
    from bench.eval_needle import LENGTHS

    assert max(LENGTHS) == 16384, LENGTHS
    assert set(LENGTHS) >= {1024, 4096, 8192, 16384}


def test_the_cli_refuses_a_bad_ladder_before_it_loads_a_checkpoint(tmp_path):
    """`--lengths 0` would make every row divide by the sample and report the same
    accuracy forever; the refusal has to come before the checkpoint load, because
    the path here is deliberately one that does not exist."""
    from bench.eval_needle import main

    with pytest.raises(SystemExit, match="--lengths"):
        main(["--ckpt", str(tmp_path / "no-such-run"), "--lengths", "0"])
    with pytest.raises(SystemExit, match="--lengths"):
        main(["--ckpt", str(tmp_path / "no-such-run"), "--lengths=-1"])

"""P5 5a: abstention is an engine behaviour and it says why (SPEC §5 P5, gate G5).

The gate this file exists to hold is *"a fast path that is uncertain must never be
silently load-bearing"* — a claim about what the caller sees, so every test here
reads the engine's own response and checks each commitment against the probability
vector printed beside it. Nothing asserts against a hand-written probability.

Two ideas carry most of the risk, and both are tested through the real head:

* **A confident "no" is not an abstention.** noul reports raw `p(Yes)` because the
  Brier score needs it, so pricing noul confidence as `p(Yes)` would abstain on the
  model's surest answers. The sharpened-head test builds that case out of a live
  two-way readout rather than a dict.
* **`abstain` must agree with `probabilities`.** A flag computed from one readout
  and a distribution from another is the same failure the latency harness found in
  its own `score` handling, so the sweep compares them row by row.

`Myna.temperature` is the scalar the readout divides its logits by, so 1e-3 makes
every head near one-hot and 1e3 makes it near uniform — a dial on the real head
without monkeypatching the thing under test.
"""

from __future__ import annotations

import random

import pytest
import torch

from myna.data import WORKFLOWS, generate
from myna.engine import Myna, abstain_check
from myna.model import MynaConfig, MynaModel
from myna.tokenizer import train_tokenizer

SUPPORT_QS = WORKFLOWS["support"][0]
QUESTIONS = {q.name: {"type": q.type, "instructions": q.instruction, "criteria": q.options}
             for q in SUPPORT_QS}
EXAMPLES = [e.state for e in generate(12, "support", random.Random(12))]


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    """An untrained checkpoint. Its probabilities are arbitrary, and every
    assertion below is about their *relationship* to the flag, so a random head
    is a witness rather than a caveat."""
    tok = train_tokenizer([e.state for e in generate(64, "support", random.Random(11))],
                          vocab_size=600)
    cfg = MynaConfig(vocab=tok.get_vocab_size(), d_model=64, n_layers=2, n_heads=4,
                     d_k=16, d_v=16, d_ff=128, d_ptr=32)
    torch.manual_seed(7)
    d = tmp_path_factory.mktemp("abstain-ckpt")
    torch.save({"state_dict": MynaModel(cfg).state_dict(), "cfg": vars(cfg),
                "temperature": 1.0}, d / "model.pt")
    tok.save(str(d / "tokenizer.json"))
    return d


def engine(ckpt, threshold=None, temperature=None) -> Myna:
    m = Myna(ckpt, abstain_below=threshold)
    if temperature is not None:
        m.temperature = temperature
    return m


def answers(m, which=0) -> dict:
    return m.observe(EXAMPLES[which]).ask(QUESTIONS)["answers"]


# ------------------------------------------------------------- the pure decision
@pytest.mark.parametrize("bad", [0.0, -0.2, 1.5, 2.0])
def test_a_floor_outside_the_unit_interval_is_refused(ckpt, bad):
    """0.0 abstains never and 1.5 abstains always; both read like a working
    deployment until the day the fallback bill arrives."""
    with pytest.raises(ValueError, match=r"must be in \(0, 1\]"):
        Myna(ckpt, abstain_below=bad)
    with pytest.raises(ValueError, match=r"must be in \(0, 1\]"):
        abstain_check([0.5, 0.5], bad, ["a", "b"])


def test_confidence_is_the_top_option_and_the_margin_the_runner_up_gap():
    g = abstain_check([0.1, 0.62, 0.28], 0.6, ["a", "b", "c"])
    assert g["confidence"] == pytest.approx(0.62) and g["abstain"] is False
    assert g["margin"] == pytest.approx(0.62 - 0.28)
    higher = abstain_check([0.1, 0.62, 0.28], 0.7, ["a", "b", "c"])
    assert higher["abstain"] is True and higher["confidence"] == pytest.approx(0.62)


def test_the_reason_carries_the_measured_numbers_not_a_fixed_string():
    g = abstain_check([0.44, 0.36, 0.2], 0.5, ["billing", "technical", "shipping"])
    assert g["abstain"]
    assert "billing" in g["reason"]
    assert "0.440" in g["reason"] and "0.080" in g["reason"] and "0.500" in g["reason"]


def test_the_reason_quotes_the_option_that_would_have_been_chosen():
    """A reason naming the runner-up would read like a plausible bug fix and
    point a human at the wrong row of the distribution."""
    g = abstain_check([0.2, 0.44, 0.36], 0.5, ["billing", "technical", "shipping"])
    assert "technical" in g["reason"] and "billing" not in g["reason"]


def test_a_confidence_sitting_exactly_on_the_floor_commits():
    """`<` versus `<=` at the boundary. It matters because the risk/coverage
    harness takes its floor from a real answer's confidence, so that one answer is
    always exactly on the line — and the kept set must be the complement of the
    abstain set, not one row short of it."""
    assert abstain_check([0.7, 0.3], 0.7, ["a", "b"])["abstain"] is False
    assert abstain_check([0.7, 0.3], 0.7 + 1e-9, ["a", "b"])["abstain"] is True


def test_a_two_way_vector_needs_both_of_its_labels():
    """The helper's one structural footgun: one label against two probabilities
    points the reason string at whichever option the caller happened to keep."""
    with pytest.raises(ValueError, match="probabilities for"):
        abstain_check([0.5, 0.5], 0.6, ["only-one"])


# ------------------------------------------------- never abstaining is the default
def test_no_threshold_means_every_question_is_committed(ckpt):
    out = engine(ckpt).observe(EXAMPLES[0]).ask(QUESTIONS)
    for name, a in out["answers"].items():
        assert a["abstain"] is False, name
        assert a["reason"] is None, name
    assert out["policy"] == {"abstain_below": None, "abstained": []}


def test_the_flag_and_the_printed_distribution_cannot_disagree(ckpt):
    """Sweep floors over the same questions: `abstain` must be exactly
    `confidence < floor`, and `confidence` must be the top of the distribution the
    same answer prints.

    The second half is the cross-check that matters: `probabilities` comes out of
    the softmax, `abstain` out of the gate, so a gate reading a stale or
    re-derived vector shows up here rather than in production. noul has no option
    vector to print — its distribution is the pair (`noul`, 1-`noul`), which is
    also where a p(Yes)-as-confidence bug would first disagree with itself."""
    def top(a):
        return max(a["probabilities"].values()) if "probabilities" in a \
            else max(a["noul"], 1.0 - a["noul"])

    base = answers(engine(ckpt))
    for floor in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.95):
        for name, a in answers(engine(ckpt, floor)).items():
            assert a["abstain"] is (a["confidence"] < floor), (name, floor)
            assert a["confidence"] == pytest.approx(top(base[name]), abs=1e-6), (name, floor)
            # a committed side in a two-way question is by definition at least as
            # likely as its alternative: confidence below 0.5 means it was read off
            # p(Yes). (A five-option choice can legitimately top out at 0.3.)
            if a["type"] == "noul":
                assert a["confidence"] >= 0.5 - 1e-6, (name, floor, a)


def test_raising_the_floor_never_returns_an_answer_it_had_withdrawn(ckpt):
    """Monotonicity: the flag must be a function of the confidence. A non-monotone
    gate means it is being computed from something else."""
    for which in range(4):
        lo = answers(engine(ckpt, 0.35), which)
        hi = answers(engine(ckpt, 0.75), which)
        for name in lo:
            if lo[name]["abstain"]:
                assert hi[name]["abstain"], (name, which)


# ------------------------------------------------ the noul side, at engine level
def test_a_confident_no_is_not_an_abstention(ckpt):
    """Sharpen the heads until noul is near one-hot, then look only at the rows
    that landed on "No".

    This is the test that catches pricing noul confidence as p(Yes): on those rows
    `noul` is ~0.0 while the committed side is ~1.0, so such an implementation
    abstains at a high floor and fails here."""
    rows = [answers(engine(ckpt, 0.9, temperature=1e-3), i)["churn_risk"]
            for i in range(len(EXAMPLES))]
    nos = [r for r in rows if r["noul"] < 0.5]
    yeses = [r for r in rows if r["noul"] >= 0.5]
    assert nos and yeses, "no two-sided noul sample — the test would be vacuous"
    for a in nos:
        assert a["abstain"] is False, a
        assert a["confidence"] > 0.9 and a["yes"] is False
        assert a["noul"] < 0.1, "the raw p(Yes) must survive for the Brier score"
    for a in yeses:
        assert a["abstain"] is False and a["yes"] is True


def test_a_coin_flip_noul_does_abstain(ckpt):
    """The other half of the same dial: flatten the logits and the gate has to
    fire, or G5's coverage curve would be a straight line at 100% coverage."""
    rows = [answers(engine(ckpt, 0.9, temperature=1e3), i)["churn_risk"]
            for i in range(3)]
    assert all(r["abstain"] for r in rows), rows[0]
    assert all(r["yes"] is None for r in rows)
    assert all(abs(r["noul"] - 0.5) < 0.05 for r in rows)


def test_an_abstained_choice_still_publishes_its_full_distribution(ckpt):
    """The fallback seam needs the vector — a seam that drops the answer cannot
    ask a stronger model to re-rank it, and the risk/coverage curve reads it too."""
    a = answers(engine(ckpt, 0.99))["department"]
    assert a["abstain"] is True and a["choice"] is None
    assert len(a["probabilities"]) == len(QUESTIONS["department"]["criteria"])
    assert sum(a["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    # the reason is the engine's own measurement, not a fixed string: the option
    # it would have chosen and its probability both have to be in there.
    top = max(a["probabilities"], key=a["probabilities"].get)
    assert top in a["reason"] and f"{a['confidence']:.3f}" in a["reason"]
    assert "0.990" in a["reason"]


def test_a_score_abstains_on_its_level_and_withdraws_its_expectation(ckpt):
    """`score` is the awkward type: its answer is an expectation over levels, which
    is defined even for a flat distribution. So abstention must be about the level
    decision, and the expectation goes with it rather than surfacing as 0.0."""
    flat = answers(engine(ckpt, 0.4, temperature=1e3))["urgency"]
    assert flat["abstain"] is True
    assert flat["level"] is None and flat["score"] is None
    sharp = answers(engine(ckpt, 0.9, temperature=1e-3))["urgency"]
    assert sharp["abstain"] is False
    assert sharp["level"] in QUESTIONS["urgency"]["criteria"]
    assert isinstance(sharp["score"], float)


def test_the_policy_and_the_abstention_count_ride_with_the_response(ckpt):
    out = engine(ckpt, 0.99).observe(EXAMPLES[0]).ask(QUESTIONS)
    assert out["policy"]["abstain_below"] == 0.99
    assert sorted(out["policy"]["abstained"]) == sorted(
        k for k, v in out["answers"].items() if v["abstain"])
    assert out["policy"]["abstained"], "a 0.99 floor on random heads should abstain on all three"


def test_the_engine_still_answers_every_question_when_it_abstains(ckpt):
    """An abstention is not a skipped call: in the latency harness a missing
    answer is a free question, and here it would read like a careful model."""
    out = engine(ckpt, 0.95).observe(EXAMPLES[0]).ask(QUESTIONS)
    assert set(out["answers"]) == set(QUESTIONS)


def test_an_abstaining_answer_is_a_type_error_at_a_naive_caller(ckpt):
    """The `None` is the point. `answer["choice"].upper()` raises, so a caller
    that never reads `abstain` fails at the seam instead of shipping a label the
    model itself refused to commit to."""
    a = answers(engine(ckpt, 0.99))["department"]
    assert a["choice"] is None
    with pytest.raises(AttributeError):
        a["choice"].upper()

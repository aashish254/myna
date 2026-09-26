"""P5 5b/5d: the risk/coverage harness measured against a synthetic oracle.

G5's pass condition is a *curve*, and a curve has no external truth to check
against — the numbers are the artifact. So the witness here is constructed the
other way round: the test knows which questions a model got right and how confident
it should look, feeds that to the harness, and requires the curve to say what the
oracle already knows. A harness that ranks on the wrong quantity, or that cuts the
curve at the wrong boundary, cannot survive these tests — which is the precondition
for trusting it on a real checkpoint (SPEC §7.1).

The boundary convention is the subtle one and gets its own test: the engine
abstains on `confidence < floor` while the curve keeps `confidence >= floor`. Those
must be complements at the *same* float, including for the rows whose confidence is
exactly the floor — a half-open/closed mismatch would show up as an
engine/curve disagreement that only the re-run catches, and only if it is tested.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.risk_coverage import (COVERAGE_RUNGS, at_or_above, collect,  # noqa: E402
                                 committed_probability, curve, engine_questions, g5_verdict,
                                 is_correct, verify_floor)
from myna.data import Example, Question  # noqa: E402
from myna.engine import abstain_check  # noqa: E402

QS = [Question("intent", "choice", "Which intent?", ["a", "b", "c"]),
      Question("urgent", "noul", "Is it urgent?", ["no", "yes"]),
      Question("severity", "score", "How severe?", ["0", "1", "2"])]


def ans_probs(probs):
    return {"type": "choice", "probabilities": dict(zip(["a", "b", "c"], probs)),
            "confidence": max(probs)}


# ------------------------------------------------------- what the curve ranks on
def test_confidence_is_the_top_of_the_printed_distribution():
    assert committed_probability(ans_probs([0.2, 0.7, 0.1])) == pytest.approx(0.7)


def test_a_sure_no_ranks_as_high_as_a_sure_yes():
    """Both halves of the noul trap: `p(Yes)=0.01` is the model's *strongest*
    possible answer, so ranking on p(Yes) would file it under abstentions."""
    sure_no = {"type": "noul", "noul": 0.01, "confidence": 0.99}
    sure_yes = {"type": "noul", "noul": 0.99, "confidence": 0.99}
    assert committed_probability(sure_no) == pytest.approx(0.99)
    assert committed_probability(sure_no) == committed_probability(sure_yes)


def test_an_answer_with_nothing_to_rank_on_is_an_error_not_a_zero():
    """0.0 would sort it to the bottom of the curve and quietly become "the model
    was unsure", which is a claim about a row that carries no evidence."""
    with pytest.raises(ValueError, match="no probability"):
        committed_probability({"type": "choice", "probabilities": {}})


def test_score_is_correct_on_its_argmax_level_not_its_expectation():
    """The expectation is defined for a flat distribution; treating it as the
    answer would make an undecided score partly correct by construction."""
    assert is_correct(ans_probs([0.2, 0.5, 0.3]), "choice", 1)
    assert not is_correct(ans_probs([0.2, 0.5, 0.3]), "choice", 2)
    noul_p = {"type": "noul", "noul": 0.49}
    assert is_correct(noul_p, "noul", 0) and not is_correct(noul_p, "noul", 1)


def test_a_level_tie_is_broken_the_way_the_engine_breaks_it():
    """`noul` 0.5 commits to "yes" (`p >= 0.5`), which is also what
    `Observation.ask` reports in `yes`. A harness that scored the tie the other way
    would disagree with the product on exactly the rows the fallback exists for."""
    tie = {"type": "noul", "noul": 0.5}
    assert is_correct(tie, "noul", 1) and not is_correct(tie, "noul", 0)


# ------------------------------------------------------------------- the curve
def points(pairs):
    """[(confidence, correct)] -> the shape `collect` returns."""
    return [{"source": "s", "question": "q", "type": "choice",
             "confidence": c, "correct": int(ok)} for c, ok in pairs]


def test_a_perfectly_informative_confidence_clears_the_gate_it_was_designed_for():
    """Oracle: 70 of 100 answers are both right and confident, 30 are both wrong
    and unconfident. Then every rung up to 70% coverage reads 1.000 accuracy, and
    G5's condition (>=0.95 at >=60%) is met at the 0.7 rung.

    The 70/30 split is the smallest one the gate can be *seen* to require: with
    50/50 the top 60% already contains 10 wrong answers, accuracy 0.833, and the
    verdict is not-met even though the confidence ranks perfectly. That is the
    gate working as written, not a harness failure — so it is the next test."""
    pts = points([(0.90 + i * 1e-4, True) for i in range(70)]
                 + [(0.10 + i * 1e-4, False) for i in range(30)])
    rows = curve(pts)
    for r in rows:
        if r["coverage"] <= 0.7:
            assert r["accuracy"] == 1.0, r
    v = g5_verdict(rows)
    assert v["pass"] is True and v["best_at_target"]["coverage"] == pytest.approx(0.7)


def test_a_perfectly_ranked_confidence_still_fails_when_accuracy_is_only_two_thirds():
    pts = points([(0.90 + i * 1e-4, True) for i in range(50)]
                 + [(0.10 + i * 1e-4, False) for i in range(50)])
    rows = curve(pts)
    for r in rows:
        if r["coverage"] <= 0.5:
            assert r["accuracy"] == 1.0, r
    v = g5_verdict(rows)
    assert v["pass"] is False, "0.95 at 60% coverage needs 57 of the top 60 to be right"


def test_an_uninformative_confidence_buys_nothing():
    """Random confidence, fixed accuracy: every rung reads the pooled accuracy, so
    the harness cannot manufacture coverage gains out of noise."""
    rng = random.Random(1)
    pts = points([(rng.random(), i % 2 == 0) for i in range(200)])
    pooled = sum(p["correct"] for p in pts) / len(pts)
    for r in curve(pts):
        assert abs(r["accuracy"] - pooled) < 0.08, r


def test_accuracy_rises_monotonically_as_coverage_falls_on_a_calibrated_head():
    pts = points([(0.5 + 0.5 * (i / 99.0), i >= 50) for i in range(100)])
    accs = [r["accuracy"] for r in curve(pts)]
    assert all(a >= b - 1e-9 for a, b in zip(accs[1:], accs[:-1])), accs
    assert accs[0] == pytest.approx(0.5) and accs[-1] == 1.0


def test_the_floor_column_is_the_confidence_of_the_last_kept_row():
    pts = points([(0.9, True), (0.8, True), (0.3, False), (0.2, False)])
    rows = {r["rung"]: r for r in curve(pts)}
    assert rows[0.5]["n"] == 2 and rows[0.5]["threshold"] == pytest.approx(0.8)
    assert rows[1.0]["threshold"] == pytest.approx(0.2)


def test_realized_coverage_is_what_the_sample_supports_not_the_rung_asked():
    """3 questions cannot be cut at 30%. The table prints the row count it
    actually averaged — so a small cell reads as the small cell it is instead of
    as a 30% slice. A rung below one row keeps one row rather than reporting an
    average of nothing."""
    rows = curve(points([(0.9, True), (0.5, False), (0.1, False)]))
    assert [r["n"] for r in rows] == [3, 3, 2, 2, 2, 2, 1, 1, 1, 1]
    assert all(r["coverage"] == r["n"] / 3 for r in rows)
    assert all(r["n"] >= 1 and r["accuracy"] is not None for r in rows)
    # the largest deviation is on the smallest rung, and it is one row's worth
    assert max(abs(r["coverage"] - r["rung"]) for r in rows) <= 2 / 3


def test_an_empty_sample_produces_no_cells_and_no_crash():
    rows = curve([])
    assert all(r["n"] == 0 and r["accuracy"] is None for r in rows)
    assert g5_verdict(rows)["pass"] is False


def test_an_anti_calibrated_head_makes_the_curve_fall_instead_of_rise():
    """Confidence high exactly where the model is wrong: the curve must go *down*
    as coverage tightens. A harness that sorted on the wrong side of the vector,
    or that averaged the wrong prefix, cannot produce this shape — and it must not
    reach the gate either."""
    pts = points([(0.90 + i * 1e-4, False) for i in range(30)]
                 + [(0.10 + i * 1e-4, True) for i in range(70)])
    rows = curve(pts)
    assert rows[0]["accuracy"] == pytest.approx(0.7)
    assert rows[-1]["accuracy"] == pytest.approx(0.0)
    accs = [r["accuracy"] for r in rows]
    assert all(a <= b + 1e-9 for a, b in zip(accs[1:], accs[:-1])), accs
    assert g5_verdict(rows)["pass"] is False


def test_the_gate_needs_the_coverage_as_much_as_the_accuracy():
    """0.95 accuracy on 10% of the rows is a model that refuses to answer. The
    verdict must say not-met, and say so because of the coverage half."""
    pts = points([(0.99, True) for _ in range(10)] + [(0.11, False) for _ in range(90)])
    rows = curve(pts, rungs=(1.0, 0.9, 0.5, 0.2, 0.1))
    v = g5_verdict(rows)
    assert v["best_at_target"] and v["best_at_target"]["coverage"] <= 0.15
    assert v["pass"] is False


# ------------------------------- the engine/curve boundary, which the re-run checks
def test_the_floor_belongs_to_the_kept_set_and_to_nobody_else():
    """The engine's rule is `abstain when confidence < floor`; the curve's kept set
    is its complement. Rows sitting *exactly* on the floor are the interesting
    ones — and the floor is read off one of them, so they always exist.

    The last assertion crosses into the engine itself: a `<` become `<=` there
    would abstain on its own boundary row, and this is the only place in the suite
    where the two sides of that comparison are checked against the same float."""
    pts = points([(0.7, True), (0.7, False), (0.7, True), (0.4, False)])
    assert [p["correct"] for p in at_or_above(pts, 0.7)] == [1, 0, 1]
    ok = verify_floor(pts, 0.7, engine_abstained=1)
    assert ok["curve_below_floor"] == 1 and ok["agree"] is True
    assert ok["realized_coverage"] == pytest.approx(0.75)
    # one row out of place is a different rule, not a rounding difference
    assert verify_floor(pts, 0.7, engine_abstained=2)["agree"] is False
    assert abstain_check([0.7, 0.3], 0.7, ["a", "b"])["abstain"] is False


def test_the_gate_boundaries_are_inclusive_on_both_halves():
    """G5's wording is ">= 0.95 at >= 60% coverage", so the oracle is built to land
    on both boundaries at once: 57 right and 3 wrong in the top 60 of 100.

    Read the other way (a `>` anywhere) the same curve fails the gate, which is
    what makes this the test that pins the wording rather than the arithmetic."""
    pts = points([(0.99 - i * 1e-4, True) for i in range(57)]
                 + [(0.80 - i * 1e-4, False) for i in range(3)]
                 + [(0.20 - i * 1e-4, False) for i in range(40)])
    rows = curve(pts)
    r60 = next(r for r in rows if r["rung"] == 0.6)
    assert r60["n"] == 60 and r60["accuracy"] == pytest.approx(0.95)
    v = g5_verdict(rows)
    assert v["pass"] is True and v["best_at_target"]["coverage"] == pytest.approx(0.6)
    # 56/60 is 0.933, one row short of the target, and the verdict must see it
    short = points([(0.99 - i * 1e-4, True) for i in range(56)]
                   + [(0.80 - i * 1e-4, False) for i in range(4)]
                   + [(0.20 - i * 1e-4, False) for i in range(40)])
    assert g5_verdict(curve(short))["pass"] is False


# ------------------------------------------------------------- collect(), stubbed
class StubObservation:
    def __init__(self, answers):
        self._answers = answers

    def ask(self, questions):
        return {"answers": {k: v for k, v in self._answers.items() if k in questions},
                "policy": {"abstain_below": None, "abstained": []}, "usage": {}}


class StubMyna:
    """Stands in for the engine so `collect`'s own guards are testable — the real
    checkpoint cannot be asked to return a skewed distribution or skip a question."""

    def __init__(self, answers):
        self.answers = answers

    def observe(self, state):
        return StubObservation(self.answers)


def groups_one_row():
    exs = [Example("a state", "banking77#00000000", (1, 1, 0))]
    return {"banking77#00000000": (QS, exs)}


def test_collect_scores_every_question_the_split_carries():
    answers = {"intent": ans_probs([0.2, 0.7, 0.1]),
               "urgent": {"type": "noul", "noul": 0.8, "confidence": 0.8},
               "severity": {"type": "score", "probabilities": {"0": 0.6, "1": 0.3, "2": 0.1},
                            "confidence": 0.6}}
    pts = collect(StubMyna(answers), groups_one_row())
    assert [p["question"] for p in pts] == ["intent", "urgent", "severity"]
    assert [p["source"] for p in pts] == ["banking77"] * 3
    assert [p["correct"] for p in pts] == [1, 1, 1], "gold indices are per question slot"
    assert [round(p["confidence"], 2) for p in pts] == [0.7, 0.8, 0.6]


def test_a_question_the_engine_skipped_is_an_error_not_a_missing_row():
    """Silently dropping it would shrink the denominator of every cell below it."""
    answers = {"intent": ans_probs([0.2, 0.7, 0.1])}
    with pytest.raises(AssertionError, match="engine skipped"):
        collect(StubMyna(answers), groups_one_row())


def test_a_confidence_that_disagrees_with_its_own_distribution_is_an_error():
    """The curve ranks on the distribution and the gate fires on `confidence`, so a
    pair that disagrees means the published floor is not the rule the engine runs."""
    answers = {"intent": {"type": "choice", "probabilities": {"a": 0.2, "b": 0.7, "c": 0.1},
                          "confidence": 0.2},
               "urgent": {"type": "noul", "noul": 0.8, "confidence": 0.8},
               "severity": {"type": "score", "probabilities": {"0": 0.6, "1": 0.3, "2": 0.1},
                            "confidence": 0.6}}
    with pytest.raises(AssertionError, match="engine confidence"):
        collect(StubMyna(answers), groups_one_row())


def test_the_spec_the_engine_is_asked_carries_no_gold():
    """`criteria` is the option text list and nothing else: passing the label index
    through would let a head that reads its input score itself."""
    specs = engine_questions(QS)
    assert specs["intent"] == {"type": "choice", "instructions": "Which intent?",
                               "criteria": ["a", "b", "c"]}
    assert all("label" not in s for s in specs.values())


def test_the_run_rungs_cover_the_gates_own_floor():
    """0.6 coverage is in G5's wording, so the table has to contain that rung —
    otherwise `best coverage at target` is measured against a rung that never ran."""
    assert 0.6 in COVERAGE_RUNGS and 1.0 in COVERAGE_RUNGS

"""P5 5c: the fallback seam routes, and the routing is visible per decision.

The seam is tested with two fakes rather than with laya: what has to be proven is
the *contract* (who gets asked what, and which name rides on the answer), and laya
is a 421M-param checkout on `PYTHONPATH` that would make the suite depend on it.
`DECIDERS` is checked by name so the registry cannot quietly lose an entry, and
`laya_spec` — the one part of the laya path that is pure translation — is checked
against the shim in `bench/eval_laya_real.py`, because two translations of the same
schema that drift mean the fallback was answering a different question than the
baseline was scored on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from myna.fallback import DECIDERS, Decider, Fallback, build, laya_spec


class Fake(Decider):
    """Answers every question with a fixed confidence, abstaining where told."""

    def __init__(self, name, abstain_on=(), confidences=None):
        self.name = name
        self.abstain_on = set(abstain_on)
        self.confidences = confidences or {}
        self.calls: list[list[str]] = []

    def predict(self, state, questions):
        self.calls.append(sorted(questions))
        out = {}
        for k in questions:
            conf = self.confidences.get(k, 0.01 if k in self.abstain_on else 0.99)
            out[k] = {"type": "choice", "choice": None if conf < 0.5 else "a",
                      "confidence": conf, "abstain": conf < 0.5, "probabilities": {"a": conf}}
        return {"answers": out, "usage": {}}


def _fb(primary, secondary):
    return Fallback(primary, secondary)


def test_only_the_abstained_questions_reach_the_fallback():
    fast = Fake("fast", abstain_on=["q1"])
    slow = Fake("slow")
    _fb(fast, slow).predict("state", {"q0": {}, "q1": {}, "q2": {}})
    assert fast.calls == [["q0", "q1", "q2"]]
    assert slow.calls == [["q1"]], "re-asking answered questions hides what abstention costs"


def test_a_confident_path_never_touches_the_fallback():
    fast, slow = Fake("fast"), Fake("slow")
    out = _fb(fast, slow).predict("state", {"q0": {}, "q1": {}})
    assert slow.calls == []
    assert {a["model"] for a in out["answers"].values()} == {"fast"}


def test_every_answer_names_the_engine_that_committed_to_it():
    fast = Fake("fast", abstain_on=["q1"])
    slow = Fake("slow")
    out = _fb(fast, slow).predict("state", {"q0": {}, "q1": {}})
    assert out["answers"]["q0"]["model"] == "fast"
    assert out["answers"]["q1"]["model"] == "slow"
    assert out["answers"]["q1"]["abstain"] is False, "the routed answer was replaced"
    assert out["usage"]["routed_to_secondary"] == 1, "the count a cost dashboard reads"


def test_the_label_follows_the_decider_not_a_hardcoded_name():
    """A literal "myna" in the router would survive any fake-test rewrite and
    fail the day a different engine sits in the fast lane."""
    fast = Fake("cheap-model", abstain_on=["q1"])
    slow = Fake("better-model")
    out = _fb(fast, slow).predict("s", {"q0": {}, "q1": {}})
    assert out["answers"]["q0"]["model"] == "cheap-model"
    assert out["answers"]["q1"]["model"] == "better-model"
    assert out["routing"] == {"primary": "cheap-model", "secondary": "better-model",
                              "abstained": ["q1"], "still_abstained": [], "rounds": 1}


def test_a_fallback_that_also_refuses_is_reported_not_dropped():
    """The answer that must survive worst: the fast engine abstained, the slow one
    agreed, and a caller that found nothing in `still_abstained` would read the
    row as "answered with a refusal" rather than "nobody will commit"."""
    fast = Fake("fast", abstain_on=["q1"])
    slow = Fake("slow", abstain_on=["q1"])
    out = _fb(fast, slow).predict("s", {"q0": {}, "q1": {}})
    assert set(out["answers"]) == {"q0", "q1"}
    assert out["answers"]["q1"]["abstain"] is True
    assert out["routing"]["still_abstained"] == ["q1"]
    assert out["answers"]["q1"]["model"] == "slow"


def test_the_second_engine_is_asked_once_not_until_it_agrees():
    fast = Fake("fast", abstain_on=["q1"])
    slow = Fake("slow", abstain_on=["q1"])
    _fb(fast, slow).predict("s", {"q1": {}})
    assert len(slow.calls) == 1


def test_one_engine_cannot_be_its_own_fallback():
    same = Fake("only")
    with pytest.raises(ValueError, match="different engines"):
        Fallback(same, same)


def test_a_round_budget_below_one_is_refused_at_construction():
    with pytest.raises(ValueError, match="max_rounds"):
        Fallback(Fake("fast"), Fake("slow"), max_rounds=0)


def test_the_registry_holds_both_engines_by_name():
    assert set(DECIDERS) >= {"myna", "laya"}


def test_an_unknown_decider_names_the_registry_instead_of_raising_keyerror():
    with pytest.raises(SystemExit, match="unknown decider 'mlx'"):
        build("mlx")


def test_the_fallback_dict_carries_no_abstain_key_when_nothing_refused():
    """`routing.abstained` empty with a non-empty answer set is the shape a
    dashboard reads as "the fast path is not load-bearing today"."""
    out = _fb(Fake("fast"), Fake("slow")).predict("s", {"q0": {}})
    assert out["routing"]["abstained"] == []
    assert out["usage"]["routed_to_secondary"] == 0


# ------------------------------------------------- the laya adapter's question shim
MIXED = {
    "intent": {"type": "choice", "instructions": "Which intent?",
               "criteria": {"a": "first", "b": "second"}, "label": "a"},
    # the shape imdb and yelp actually ship: a noul row carrying criteria
    "urgent": {"type": "noul", "instructions": "Is it urgent?",
               "criteria": {"true": "Yes, today", "false": "No"}, "label": True},
    "sev": {"type": "score", "instructions": "How severe?", "criteria": ["0", "1"],
            "label": 1},
}


def test_noul_reaches_laya_without_criteria():
    """The one schema difference that makes this function exist: laya's `noul`
    takes `instructions` and nothing else, so myna's `["no", "yes"]` list has to be
    dropped before the call or the fallback raises on the way to answering."""
    spec = laya_spec(MIXED)["urgent"]
    assert spec == {"type": "noul", "instructions": "Is it urgent?"}
    assert "criteria" not in spec


def test_no_gold_label_crosses_the_seam_in_any_shape():
    """Every type, not just the one that happened to carry a label in the fixture:
    a fallback that is handed the answer has been asked to decide nothing."""
    for qid, spec in laya_spec(MIXED).items():
        assert "label" not in spec, qid
    assert laya_spec(MIXED)["intent"]["criteria"] == {"a": "first", "b": "second"}
    assert laya_spec(MIXED)["sev"]["criteria"] == ["0", "1"]


def test_the_two_laya_shims_in_the_repo_cannot_drift():
    """`bench/eval_laya_real.py` translates the same schema for the accuracy
    witness. If the two ever disagree, the fallback is answering a different
    question than the one the baseline was scored on — and only one of them is in
    the shipped product path."""
    from bench.eval_laya_real import laya_questions

    assert laya_spec(MIXED) == laya_questions(MIXED)


def test_the_laya_side_calls_system_one_and_not_the_router(monkeypatch):
    """`Router` selects a model of its own, so a `model` label naming the router
    would not name the thing that answered — the same equal-footing rule §9.1 puts
    on the latency table, applied to the fallback path.

    laya is a checkout on `PYTHONPATH`, not a dependency, so the module is faked:
    what is under test is which entry point `LayaDecider` reaches and what it hands
    it."""
    import types

    seen = {}

    class FakeAgent:
        def __init__(self, ckpt_dir, device=None):
            seen["ckpt_dir"], seen["device"] = ckpt_dir, device

        def system_one(self, state, questions):
            seen["state"], seen["questions"] = state, questions
            return {"answers": {k: {"type": "choice", "abstain": False, "confidence": 0.9}
                                for k in questions}}

        def router(self, *a, **kw):  # pragma: no cover - must never be reached
            raise AssertionError("Router reaches its own model selection")

    fake = types.ModuleType("laya")
    fake.Agent = FakeAgent
    monkeypatch.setitem(sys.modules, "laya", fake)

    slow = build("laya", ckpt_dir="/tmp/laya-ckpt", device="cpu")
    out = slow.predict("a state", MIXED)
    assert seen["ckpt_dir"] == "/tmp/laya-ckpt" and seen["device"] == "cpu"
    assert set(out["answers"]) == set(MIXED)
    assert seen["questions"] == laya_spec(MIXED), "the fallback was asked something else"

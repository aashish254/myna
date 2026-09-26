"""Adapter tests: real-suite JSONL -> myna (Question, Example) shapes."""

import json

from myna.real_data import load_split, load_suite, suite_texts


def _row(state, questions, **meta):
    return {"state": state, "questions": questions, "_meta": {"source": "fake", **meta}}


CHOICE_Q = {
    "intent": {
        "type": "choice",
        "instructions": "Which intent?",
        "criteria": {"pay_bill": "I want to pay my bill", "card_loss": None, "top_up": "adding money"},
    }
}


def write_suite(tmp_path, rows_by_split):
    for name, rows in rows_by_split.items():
        with open(tmp_path / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return tmp_path


def test_grouping_and_gold_indices(tmp_path):
    rows = [
        _row("pay my bill please", {**CHOICE_Q, "intent": {**CHOICE_Q["intent"], "label": "pay_bill"}}),
        _row("lost my card", {**CHOICE_Q, "intent": {**CHOICE_Q["intent"], "label": "card_loss"}}),
    ]
    groups = load_split(write_suite(tmp_path, {"train": rows}) / "train.jsonl")
    assert len(groups) == 1, "same question set must group regardless of label"
    key = next(iter(groups))
    questions, exs = groups[key]
    assert key.startswith("fake#")
    q = questions[0]
    assert q.options == ["I want to pay my bill", "card loss", "adding money"]  # null desc -> humanized key
    assert [e.gold for e in exs] == [(0,), (1,)]


def test_noul_and_score_mapping(tmp_path):
    rows = [
        _row("is this urgent?", {"urgent": {"type": "noul", "instructions": "urgent?", "criteria": None, "label": True}}),
        _row("great product", {"tone": {"type": "score", "instructions": "tone?",
                                        "criteria": ["bad", "neutral", "great"], "label": 2}}),
    ]
    groups = load_split(write_suite(tmp_path, {"train": rows}) / "train.jsonl")
    assert len(groups) == 2
    for questions, exs in groups.values():
        if questions[0].type == "noul":
            assert questions[0].options == ["no", "yes"]
            assert exs[0].gold == (1,)
        else:
            assert questions[0].options == ["bad", "neutral", "great"]
            assert exs[0].gold == (2,)


def test_label_outside_criteria_is_skipped(tmp_path):
    good = _row("pay my bill", {"intent": {**CHOICE_Q["intent"], "label": "pay_bill"}})
    bad = _row("no such option", {"intent": {**CHOICE_Q["intent"], "label": "mystery"}})
    groups = load_split(write_suite(tmp_path, {"train": [good, bad]}) / "train.jsonl")
    _, exs = next(iter(groups.values()))
    assert len(exs) == 1


def test_structured_states_are_flattened(tmp_path):
    q = {"intent": {**CHOICE_Q["intent"], "label": "pay_bill"}}
    rows = [_row({"document": "pay my bill"}, q), _row(["part one", "part two"], q),
            _row({"a": {"inner": "deep text"}}, q)]
    groups = load_split(write_suite(tmp_path, {"train": rows}) / "train.jsonl")
    states = [e.state for _, exs in groups.values() for e in exs]
    assert states == ["pay my bill", "part one part two", "deep text"]


def test_structured_instructions_are_flattened(tmp_path):
    q = {"intent": {"type": "choice",
                    "instructions": {"question": "Which intent?", "focus": "Use only the information given."},
                    "criteria": {"pay_bill": "paying", "top_up": None}, "label": "top_up"}}
    groups = load_split(write_suite(tmp_path, {"train": [_row("x", q)]}) / "train.jsonl")
    questions, exs = next(iter(groups.values()))
    assert questions[0].instruction == "Which intent? Use only the information given."
    assert exs[0].gold == (1,)


def test_missing_splits_map_empty(tmp_path):
    write_suite(tmp_path, {"train": [_row("x", {"intent": {**CHOICE_Q["intent"], "label": "top_up"}})]})
    suite = load_suite(tmp_path)
    assert suite["dev"] == {} and suite["test"] == {}
    texts = suite_texts(suite["train"])
    assert "x" in texts
    assert any("adding money" in t for t in texts)  # option strings are tokenizer corpus too

"""The trainer half of the paraphrase gate (SPEC §5 P1: "wire into the trainer").

`tests/test_paraphrase.py` audits the table. It cannot see the wiring, and the
wiring is where four distinct bugs hide: a row re-worded under *another* row's
source, the tensors built from the original question list so the phrasing never
reaches the model, the memory budget priced at the phrasing the sampler happened
to draw (so one unlucky step allocates more than the run was sized for), and a
"held-out" eval pass that quietly scores the exact suite wording again.

Each is pinned here, and the whole path is run end to end by a subprocess so the
banner and `metrics.json` are witnessed rather than asserted about.
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from myna.data import Example, Question
from myna.model import MynaConfig, MynaModel
from myna.paraphrase import (
    EVAL_INDEX,
    Paraphraser,
    TRAIN_INDEXES,
    paraphrase_questions,
    variants,
)
from myna.tokenizer import batch_question_tensors, train_tokenizer
from myna.train import build_row_batch, evaluate, held_out_eval, question_tokens, worst_case_tokens

CFG = MynaConfig(vocab=128, d_model=32, n_layers=2, n_heads=2, d_k=8, d_v=8, d_ff=48, d_ptr=16)

AGNEWS_CORE = "What is the topic of this article?"
YELP_CORE = "How many stars did this reviewer give?"
AGNEWS = [Question("topic", "choice", AGNEWS_CORE,
                   ["world news", "sports", "business", "science and technology"])]
YELP = [Question("stars", "score", YELP_CORE, ["1 star: terrible", "5 stars: excellent"])]
STATES = {
    "agnews#1": "the prime minister signed the accord with neighbouring states on tuesday",
    "yelp#2": "the ramen broth was worth every single cent and i would go back tomorrow",
}
ITEMS = [(AGNEWS, Example(STATES["agnews#1"], "agnews#1", (0,))),
         (YELP, Example(STATES["yelp#2"], "yelp#2", (1,)))]


@pytest.fixture(scope="module")
def tok():
    texts = [q.instruction for q in AGNEWS + YELP]
    texts += [o for q in AGNEWS + YELP for o in q.options]
    texts += list(STATES.values())
    return train_tokenizer(texts, vocab_size=128)


class Spy(Paraphraser):
    """Records what the trainer asked for, so a mutation in `build_row_batch`
    shows up as a wrong key rather than as two equal-looking tensor dicts."""

    def __init__(self):
        super().__init__()
        self.keys = []
        self.drew = []

    def draw(self, group_key, questions, rng):
        qs = super().draw(group_key, questions, rng)
        self.keys.append(group_key)
        self.drew.append([q.instruction for q in qs])
        return qs


def _train_pool(core, source):
    v = variants(core, source)
    return {v[i] + "" for i in TRAIN_INDEXES} | {v[i].strip() for i in TRAIN_INDEXES}


def test_each_row_is_reworded_under_its_own_source(tok):
    spy, rng = Spy(), random.Random(0)
    build_row_batch(ITEMS, tok, "cpu", spy, rng)
    assert spy.keys == ["agnews#1", "yelp#2"], "one phrasing per row, keyed on that row's group"
    pool = _train_pool(AGNEWS_CORE, "agnews")
    assert set(spy.drew[0]) <= pool, "row 0 must draw from the agnews table"
    assert spy.drew[0][0] != AGNEWS_CORE, "the suite wording must never reach the model"
    pool_y = _train_pool(YELP_CORE, "yelp")
    assert set(spy.drew[1]) <= pool_y and spy.drew[1][0] != YELP_CORE
    # a source mix-up would put yelp phrasings in the agnews row and vice versa
    assert not set(spy.drew[0]) & pool_y and not set(spy.drew[1]) & pool


def test_drawn_phrasing_is_what_gets_tokenized(tok):
    spy, rng = Spy(), random.Random(3)
    b = build_row_batch(ITEMS, tok, "cpu", spy, rng)
    drawn = [[Question(q.name, q.type, instr, list(q.options))
              for q, instr in zip(src, drew)] for src, drew in zip((AGNEWS, YELP), spy.drew)]
    want = batch_question_tensors(tok, [[(q.instruction, q.options) for q in qs] for qs in drawn])
    assert torch.equal(b["q_ids"], want["q_ids"]), \
        "the tensors were built from something other than the drawn wording"
    assert torch.equal(b["span_mat"], want["span_mat"])
    assert torch.equal(b["decide_idx"], want["decide_idx"])


def test_row_batch_differs_from_the_unparaphrased_build(tok):
    with_pp = build_row_batch(ITEMS, tok, "cpu", Paraphraser(), random.Random(0))
    plain = build_row_batch(ITEMS, tok, "cpu", None, None)
    assert not torch.equal(with_pp["q_ids"], plain["q_ids"]), \
        "paraphrasing changed nothing in the tensors — the wiring is dead"
    assert torch.equal(with_pp["state_ids"], plain["state_ids"]), "states are not the lever"
    assert torch.equal(with_pp["gold"], plain["gold"]), "gold must not move"


def test_worst_case_prices_the_longest_phrasing_not_the_draw(tok):
    p, cache = Paraphraser(), {}
    n = worst_case_tokens(tok, p, ITEMS, cache)
    assert n == 2, "one entry per distinct question set"
    for key, qs in (("agnews#1", AGNEWS), ("yelp#2", YELP)):
        sets = p.variant_set(key, qs)
        lengths = [question_tokens(tok, v) for v in sets]
        assert max(lengths) > min(lengths), \
            "fixture has equal-length phrasings, so this test would prove nothing"
        for v, true_len in zip(sets, lengths):
            assert cache[id(v)] == max(lengths), "every phrasing must be priced at the worst"
        assert cache[id(qs)] == max(lengths), \
            "the base list is what draw_row_batch measures; pricing it truly under-budgets"


def test_evaluate_scores_the_held_out_phrasing(tok, monkeypatch):
    import myna.paraphrase as pp

    calls, real = [], pp.paraphrase_questions

    def spy(questions, source, index):
        calls.append((source, index))
        return real(questions, source, index)

    monkeypatch.setattr(pp, "paraphrase_questions", spy)
    model = MynaModel(CFG).eval()
    splits = {"agnews#1": (AGNEWS, [Example(STATES["agnews#1"], "agnews#1", (0,))])}
    m = evaluate(model, tok, splits, "cpu", 1.0, paraphrase_index=EVAL_INDEX)
    assert calls == [("agnews", EVAL_INDEX)], "the held-out pass must re-word every group"
    assert "agnews#1/topic" in m
    # and the default pass must not touch the wording at all
    calls.clear()
    evaluate(model, tok, splits, "cpu")
    assert calls == [], "dev/test are scored on the exact suite strings"


def test_held_out_and_exact_wordings_are_scored_separately(tok):
    """The two evals must be different numbers-by-construction, not the same
    tensor path relabelled: same weights, same rows, different instruction text."""
    model = MynaModel(CFG).eval()
    splits = {"yelp#2": (YELP, [Example(STATES["yelp#2"], "yelp#2", (1,))] * 4)}
    exact = evaluate(model, tok, splits, "cpu")
    unseen = evaluate(model, tok, splits, "cpu", paraphrase_index=EVAL_INDEX)
    assert set(exact) == set(unseen), "same keys, so a table can diff the two runs"
    logits_exact = _logits_for(model, tok, splits, YELP)
    logits_unseen = _logits_for(model, tok, splits,
                                paraphrase_questions(YELP, "yelp", EVAL_INDEX))
    assert not torch.allclose(logits_exact, logits_unseen), \
        "the model saw identical question text in both passes"


def _logits_for(model, tok, splits, questions):
    from myna.train import build_batch

    exs = list(splits["yelp#2"][1])
    b = build_batch(exs, tok, questions, "cpu")
    with torch.no_grad():
        return model(b["state_ids"], b["state_len"], b["q_ids"], b["q_mask"],
                     b["span_mat"], b["opt_valid"], b["decide_idx"])


def test_held_out_eval_reads_the_reserved_index(tok, monkeypatch):
    """One call site pins `EVAL_INDEX`; a `paraphrase_index=0` there would score a
    training wording and print it as a transfer result."""
    import myna.paraphrase as pp

    calls, real = [], pp.paraphrase_questions
    monkeypatch.setattr(pp, "paraphrase_questions",
                        lambda qs, src, i: (calls.append(i), real(qs, src, i))[1])
    model = MynaModel(CFG).eval()
    splits = {"yelp#2": (YELP, [Example(STATES["yelp#2"], "yelp#2", (1,))])}
    m = held_out_eval(model, tok, splits, "cpu", 1.0)
    assert calls == [EVAL_INDEX] and "yelp#2/stars" in m


# ---- end to end: the flag, the banner, and what lands in metrics.json ---------

TRAIN_ROWS = [
    {"state": {"document": s}, "questions": {"topic": {
        "type": "choice", "instructions": AGNEWS_CORE,
        "criteria": {"world_news": "about the world", "sports": "about sport",
                     "business": None, "sci_tech": "science or tech"},
        "label": lab}}, "_meta": {"source": "agnews"}}
    for s, lab in [("the ceasefire was signed in geneva", "world_news"),
                   ("united beat city 3 nil at old trafford", "sports"),
                   ("the central bank raised its benchmark rate", "business"),
                   ("researchers built a faster transistor", "sci_tech")]
]
EVAL_ROWS = [
    {"state": {"document": s}, "questions": {"stars": {
        "type": "score", "instructions": YELP_CORE,
        "criteria": ["1 star: terrible", "2 stars: poor", "3 stars: okay",
                     "4 stars: good", "5 stars: excellent"],
        "label": i}}, "_meta": {"source": "yelp"}}
    for s, i in [("the broth was salt water and the noodles were cold", 0),
                 ("best bowl of the year, i went back twice", 4),
                 ("fine food, nothing to complain about and nothing to praise", 2),
                 ("friendly staff but the portion was tiny", 1)]
]


def _write_suite(dirpath: Path):
    dirpath.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", TRAIN_ROWS), ("development", EVAL_ROWS),
                       ("test", EVAL_ROWS), ("calibration", EVAL_ROWS)):
        with (dirpath / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return dirpath


def _run_trainer(tmp_path, extra, suite=None, out=None):
    suite = suite if suite is not None else _write_suite(tmp_path / "suite")
    out = out if out is not None else tmp_path / "out"
    cmd = [sys.executable, "-m", "myna.train", "--suite", str(suite), "--out", str(out),
           "--device", "cpu", "--steps", "1", "--batch", "2", "--vocab", "128",
           "--max-q-cells", "4096", "--eval-every", "0"] + extra
    # import myna and re-use its own package root, so a mutation battery that runs
    # these tests against a patched copy of src sees the copy in the subprocess too
    import myna

    env = {**os.environ, "PYTHONPATH": str(Path(myna.__file__).resolve().parent.parent)}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=Path.cwd())
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout, json.loads((out / "metrics.json").read_text())


def test_the_draw_rate_divides_by_updates_that_ran_not_by_steps_asked_for(tmp_path):
    """`--steps` is an ask, not a dose. A `--resume` process runs the tail of it and a
    stopped one runs less than it asked, so a rate printed over `--steps` divides a real
    counter by updates that never happened — which is precisely how §9.56(x) produced a
    15.2% paraphrase-coverage gap that §9.58 retracts. The second run here asks for 5 and
    executes 3."""
    suite = _write_suite(tmp_path / "suite")
    out = tmp_path / "out"
    _run_trainer(tmp_path, ["--paraphrase", "on", "--row-batch", "--save-every", "1",
                            "--steps", "2"], suite=suite, out=out)
    o2, m2 = _run_trainer(tmp_path, ["--paraphrase", "on", "--row-batch", "--save-every", "1",
                                     "--steps", "5", "--resume"], suite=suite, out=out)
    assert m2["resumed_from_step"] == 2 and m2["steps_requested"] == 5
    assert m2["updates_executed"] == 3 == m2["last_step"] - m2["resumed_from_step"] + 1
    line = [ln for ln in o2.splitlines() if "phrasing draws reached the batches" in ln]
    assert len(line) == 1, "the loop must report consulting the table, once"
    assert "over 3 updates executed (of 5 requested, resumed from step 2)" in line[0]
    assert "over 5 steps" not in line[0], "the requested count is not the dose"
    assert "per update" in line[0]


@pytest.mark.parametrize("extra", [["--row-batch"], []])
def test_trainer_flag_produces_a_held_out_metric(tmp_path, extra):
    """Both batch paths, because they re-word at different places: the row path
    inside `build_row_batch`, the shared path in the loop itself."""
    out, metrics = _run_trainer(tmp_path, ["--paraphrase", "on"] + extra)
    assert "question sets re-worded, suite wording held out" in out
    assert "=== dev, HELD-OUT phrasing ===" in out
    draws = [line for line in out.splitlines() if "phrasing draws reached the batches" in line]
    assert len(draws) == 1, "the loop must report consulting the table, once"
    assert metrics["paraphrase_draws"] > 0, f"nothing was drawn: {draws}"
    assert metrics["paraphrase_sets"] == 1, "the four train rows share one question set"
    assert metrics["dev_unseen"] is not None
    assert metrics["dev_unseen"].keys() == metrics["dev"].keys()


def test_paraphrase_off_leaves_the_suite_wording_alone(tmp_path):
    out, metrics = _run_trainer(tmp_path, ["--paraphrase", "off"])
    assert "question sets re-worded" not in out
    assert "phrasing draws reached the batches" not in out
    assert metrics["paraphrase"] == "off"
    assert metrics["paraphrase_draws"] == 0
    assert metrics["dev_unseen"] is None


def test_long_context_refuses_the_flag(tmp_path):
    """The needle corpus is synthetic, so the table has no entries for it. Ignoring
    `--paraphrase on` there would print a run that claims a gate it never ran."""
    suite = _write_suite(tmp_path / "suite")
    import myna

    env = {**os.environ, "PYTHONPATH": str(Path(myna.__file__).resolve().parent.parent)}
    r = subprocess.run(
        [sys.executable, "-m", "myna.train", "--suite", str(suite), "--paraphrase", "on",
         "--long-context", "64", "--steps", "1", "--device", "cpu", "--vocab", "128",
         "--out", str(tmp_path / "out")],
        capture_output=True, text=True, env=env, cwd=Path.cwd())
    assert r.returncode != 0
    assert "not available with --long-context" in r.stderr + r.stdout

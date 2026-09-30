"""Tier 0's harness, gated: an ablation has to destroy the input it names.

`bench/diag_question_ablation.py` publishes a table of six arms against one trained
checkpoint, and the claims it carries are *mechanism* claims — the option spans carry the
labels, eight of the sixteen cells never read their row's state, three cells emit one constant.
Those sentences are all things the harness prints, so every one of them is a code path, and §7.1
says a claim printed by code is mutation-checked. Three things had to be pinned first, and
they are the three ways this harness could print a plausible table that measures nothing:

* **the guard is an input, not a banner.** The `as-scored` arm must reproduce the metrics file
  the trainer wrote, and the fixture makes that a real test rather than a tautology: the
  expected numbers come from `myna.train.evaluate()` — a second implementation of the same
  loop — while the harness reads `model.pt` off disk and runs its own `score()`. Two files,
  two code paths, one macro. Around it, each refusal is tested on its own: a key off by more
  than a row, a key this split cannot produce, a scored key absent from the metrics, and the
  one-row flips that are rounding — counted, and refused only past `ROW_SLACK_KEYS`.
* **coverage is the arm's own falsification.** An arm that rewrites nothing measures nothing,
  which is how `permute-instruction` came to be near the identity on the pilot (nine sources
  repeat one cue). So each arm is checked against hand-built expectations: how many slots it
  rewrote, that it rewrote *only* its own input kind (instruction arms leave option sets and
  states byte-identical, and vice versa), that a permute never hands a 3-option set a question
  from a different qname, that a cross-source cue really comes from another source and that a
  width with no partner keeps its own wording, and that `swap-state` is a permutation *pooled
  by source* — so a source's state multiset is preserved while a one-row source provably keeps
  its own state.
* **the printed table is the witness.** `runs/` is symlinked into the battery's scratch copy,
  so a test that reads the committed artifact stays green while the code that wrote it lies
  (§9.48). Every printing test here re-runs `main()` in-process and reads the table by column
  offset, then compares it to the JSON the same call wrote — including the `Δ` column, which
  must be a difference against `as-scored` and not against the floor, and the per-cell
  emission block, which must list every kept cell because `label_distributions` was called
  with `only_below_floor=False`.

The refusals a user actually hits are pinned too: a tokenizer that is not where the default
points (the defect this box found first), an empty split, a missing metrics file, `--split dev`
priced against the test block, and `--min-rows` moving a cell out of the table, the macro and
the floors together. The last tests bind the committed witness to the committed Kaggle report
instead — including the five below-floor pairs the harness's own docstring quotes, re-derived
from the artifact rather than trusted as prose (§9.30 run backwards).

**One fixture decision carries all of it.** The model here is *fitted* on a `train.jsonl` the
tests never score, not drawn from a seed. Sized as a random init, the instrument could not fail:
the probe measured 0 of 20 per-set keys moved for four of the five treatment arms (only
`blank-options` bit, because the pointer head reads the option spans whatever the weights are),
which means every mutation of "the ablated copy never reaches the scorer" and "the seed is
decoration" would survive and read as a mutation that names no claim. §9.48 called that an
instrument sized so small the bug cannot show; here it is an instrument so untrained the bug
cannot show. With the fit, all five arms cost cells real numbers, `--seed 3` scores a different
`swap-state` column from `--seed 0`, and the sensitivity profile is itself pinned so a blunt
fixture announces itself instead of quietly lowering the battery's coverage.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import bench.diag_question_ablation as DQ  # noqa: E402
from myna.model import MynaConfig, MynaModel, typed_loss  # noqa: E402
from myna.real_data import load_suite, suite_texts  # noqa: E402
from myna.report import accuracy_cells, cell_stats, roll_up  # noqa: E402
from myna.tokenizer import Tokenizer, train_tokenizer  # noqa: E402
from myna.train import build_batch, evaluate  # noqa: E402

REPORT = REPO / "runs" / "v1b_kaggle_3600b.report.json"
WITNESS = REPO / "runs" / "diag_question_ablation.json"
WITNESS_LOG = REPO / "runs" / "diag_question_ablation.log"

TEMPERATURE = 1.2
LAST_STEP = 7

# --------------------------------------------------------------------------- the fixture
#
# Twenty labelled question slots, thirty-three rows, seven cells, and one source with a
# single row. Every shape in here exists to make an arm *able* to fail: two agnews/topic sets
# of the same (qname, K) but different wording, so `permute-instruction` has a bucket of two
# to shuffle; five, six and four single-row sets whose qname is constant and whose instruction
# is the row, which is boolq's real shape and what makes a within-source permutation reachable
# at all; a 3-option legend and a 4-option choice that no other source offers, so the
# cross-source arm has a width it must leave alone; and dbpedia's one row, which `swap-state`
# cannot swap for anything. Two yelp score cells share a width (3) but not a qname, which is
# what makes "only sets of the same shape may exchange cues" a testable claim rather than a
# comment.
#
# **The model in this fixture is fitted, not drawn.** A random init answers from the option
# spans alone, so blanking a cue, permuting a cue or swapping a state changes *nothing* -- the
# probe that sized this fixture measured 0 of 20 per-set keys moved for four of the five
# treatment arms, and 19 of 20 for `blank-options`. A battery run against that instrument would
# score "the ablated copy never reaches the scorer" as a surviving mutant and read it as the
# mutant being unreachable, which is §9.48's defect arriving as a fixture problem: an
# instrument that cannot move cannot fail. So the suite carries a `train.jsonl` (below, never
# scored by any test) whose states say what their golds mean, and the tiny model is fitted on
# it for `TRAIN_EPOCHS` steps of `typed_loss` before `model.pt` is written. All five treatment
# arms then move keys, `--seed` moves `swap-state`, and every arm's claim is falsifiable.
#
# The two yelp score cells are what keep the *cue* load-bearing rather than an artifact of
# memorised row indices: `stars` maps terrible/fine/excellent onto 0/1/2 while `quality` labels
# the same three adjectives 2 under a different cue and legend, so answering either question
# needs the cue or the legend, which is exactly the input `blank-instruction`,
# `permute-instruction`, `cross-source-instruction` and `blank-options` each destroy. Train and
# test share a cell's cue wording (as the pilot's splits do) but never a state string or a
# row index, so nothing here can be answered by counting digits.

TOPIC_A_INSTR = "What is the topic of this note?"
TOPIC_B_INSTR = "Which section of the paper does this item belong to?"
# dev carries one different cue, so its group keys differ from test's. That is what makes a
# harness that read the wrong block of the metrics file fail loudly instead of quietly.
TOPIC_A_INSTR_DEV = "What is the topic of this article?"

TOPIC_CRIT_A = {"world": "world affairs", "sport": "games and scores"}
TOPIC_CRIT_B = {"world": "global news and diplomacy", "sport": "sports results"}
CATEGORY_CRIT = {"crime": "police and courts", "sport": "matches and teams",
                 "tech": "gadgets and software", "world": "diplomacy and war"}
STARS_LEGEND = ["1 star: would not return", "3 stars: it was fine", "5 stars: excellent"]
QUALITY_LEGEND = ["1 star: poor quality", "3 stars: average quality", "5 stars: great quality"]

STARS_INSTR = "How many stars did this reviewer give?"
QUALITY_INSTR = "Rate the quality of the goods."
CATEGORY_INSTR = "Which class does this headline belong to?"


def _row(state, source, questions):
    return {"state": state, "questions": questions, "_meta": {"source": source}}


def _choice(qname, instr, crit, label):
    return {qname: {"type": "choice", "instructions": instr, "criteria": crit, "label": label}}


def _noul(qname, instr, label):
    return {qname: {"type": "noul", "instructions": instr, "criteria": None, "label": label}}


def _score(qname, instr, legend, label):
    return {qname: {"type": "score", "instructions": instr, "criteria": legend, "label": label}}


def rows(topic_instr=TOPIC_A_INSTR):
    out = []
    # agnews/topic: two sets, same shape, different wording -> a permute bucket of two.
    # Golds 3 world / 2 sport then 4 world, so the cell's own majority floor is 7/9.
    for state, label in (("the chancellor opened the budget talks with the neighbours", "world"),
                         ("the substitute striker won the derby late on", "sport"),
                         ("the senators ratified the maritime treaty on friday", "world"),
                         ("the home side lifted the cup after extra time", "sport"),
                         ("the envoys returned to the embassy once the truce held", "world")):
        out.append(_row(state, "agnews",
                        _choice("topic", topic_instr, TOPIC_CRIT_A, label)))
    for state in ("the delegation crossed the demarcation line at dawn",
                  "a joint communique ended the two week standoff",
                  "the foreign office recalled its ambassador on tuesday",
                  "the observers were seated on both sides of the aisle"):
        out.append(_row(state, "agnews", _choice("topic", TOPIC_B_INSTR, TOPIC_CRIT_B, "world")))
    # agnews/is_business: one cue per row, so one set per row -- five singleton sets in one
    # permute bucket, and the case a within-*set* state shuffle could never reach.
    for state, cue, label in (
            ("the invoice for the printer was paid by bank transfer",
             "Is the money owed here?", True),
            ("the supplier emailed a quote for the replacement parts",
             "Does this describe a purchase?", True),
            ("the annual licence renewed itself without a reminder",
             "Was anything bought?", True),
            ("the deposit cleared before the contract was countersigned",
             "Did a payment go through?", True),
            ("the concert was rained off and the refunds queued",
             "Was the invoice settled on time?", False)):
        out.append(_row(state, "agnews", _noul("is_business", cue, label)))
    # boolq/answer: six rows, balanced golds, so its floor is exactly chance.
    for state, cue, label in (
            ("the committee shelved the budget after one reading",
             "Did the committee approve the budget?", False),
            ("the committee approved the budget unanimously at dawn",
             "Was the budget adopted?", True),
            ("the chair vetoed the budget on procedural grounds",
             "Did the members sign off on the budget?", False),
            ("the board passed the budget after a recount",
             "Was a majority found for the budget?", True),
            ("the panel threw out the budget before lunch",
             "Did the session end with a vote for the budget?", False),
            ("the council carried the budget with two abstentions",
             "Was the spending plan adopted?", True)):
        out.append(_row(state, "boolq", _noul("answer", cue, label)))
    # yelp: three cells in one source, which is what makes swap-state's pooling *by source*
    # the difference between an arm and a no-op.
    for state, cue, label in (
            ("the broth was worth every cent and the queue was short",
             "Would this reviewer return?", True),
            ("i would book the same table again without thinking",
             "Was the visit worth repeating?", True),
            ("the tasting menu justified the price twice over",
             "Would you come back for this?", True),
            ("the waiter never returned and the plate went cold",
             "Did the service win a second visit?", False)):
        out.append(_row(state, "yelp", _noul("recommend", cue, label)))
    # terrible/fine/excellent carry the answer here...
    for state, label in (("the food was terrible and the kitchen closed early", 0),
                         ("the food was fine and the room was loud", 1),
                         ("the food was excellent and the dessert doubled down", 2),
                         ("the service was terrible from start to finish", 0),
                         ("the coffee was fine but the bill was wrong", 1)):
        out.append(_row(state, "yelp", _score("stars", STARS_INSTR, STARS_LEGEND, label)))
    # ...and are overridden by the cue and legend here, which is what makes the cue
    # load-bearing without teaching the model to read a row index.
    for state in ("the terrible shipment arrived without the manual",
                  "the fine shipment arrived with a broken seal",
                  "the excellent shipment arrived a day early"):
        out.append(_row(state, "yelp", _score("quality", QUALITY_INSTR, QUALITY_LEGEND, 2)))
    # dbpedia/category: one row, and the only 4-option width in the fixture.
    out.append(_row("the arrest was announced by the police commissioner on tuesday",
                    "dbpedia",
                    _choice("category", CATEGORY_INSTR, CATEGORY_CRIT, "crime")))
    return out


# The fit split, and no test scores it. Same seven cells and the same keyword rules, with two
# restrictions that keep the fit honest: not one state string is shared with the scored split,
# and no row carries an index in its text -- so the only way a fitted model can answer these
# cells is by reading the state, which is what makes `swap-state` able to hurt it. Each cell's
# cue is the cue that cell is asked with in test (the pilot's splits do the same), except that
# every singleton-row cell gets its own distinct cue, so a cue that is load-bearing for one
# cell is load-bearing for the reason in the comment above and not because it numbers the rows.
TRAIN_TOPIC_A = [
    ("the foreign ministers opened the talks in the capital", "world"),
    ("parliament ratified the treaty shortly after midnight", "world"),
    ("the embassy reopened once the border stalemate ended", "world"),
    ("a ceasefire held through the second night of talks", "world"),
    ("the winger decided the final in stoppage time", "sport"),
    ("the home crowd sang until the equaliser came", "sport"),
    ("a rain delay shortened the semi-final replay", "sport"),
    ("the club signed the goalkeeper for three seasons", "sport"),
]
TRAIN_TOPIC_B = [
    "the chancellor flew out for the summit on sunday",
    "mediators reopened the crossing for civilians",
    "the council seated the observers from both capitals",
    "maritime talks were extended by a fortnight",
]
TRAIN_BUSINESS = [
    ("the vendor emailed an itemised bill for the parts", "Is a bill involved?", True),
    ("our card was charged for the annual licence", "Does this describe spending?", True),
    ("the match was abandoned when the floodlights failed", "Was the fixture called off?", False),
    ("the queue for the concession ran round the block", "Did the stand stay open?", False),
]
TRAIN_ANSWER = [
    ("the board carried the motion by eleven votes", "Was the motion approved?", True),
    ("treasurers signed the plan before recess", "Did the plan get signed?", True),
    ("the amendment was tabled without a seconder", "Did the amendment survive?", False),
    ("the chair ruled the proposal out of order", "Was the proposal accepted?", False),
]
TRAIN_RECOMMEND = [
    ("every dish justified the second visit", "Would you go back?", True),
    ("the kitchen sent a replacement unprompted", "Was the meal worth repeating?", True),
    ("the soup arrived cold and nobody apologised", "Would you book this again?", False),
    ("the manager refused to split the bill", "Did the staff put it right?", False),
]
TRAIN_STARS = [
    ("the starter was terrible and the main was salt", 0),
    ("the pasta was fine and the music was loud", 1),
    ("the tasting menu was excellent and generous", 2),
    ("the grill was terrible from the first plate", 0),
    ("the set lunch was fine for the price", 1),
    ("the breakfast was excellent and unhurried", 2),
]
# The same three adjectives `stars` scores, every one labeled 2 -- so the cue and the legend,
# not the adjective, carry this cell.
TRAIN_QUALITY = [
    "the terrible warranty claim took four weeks",
    "a fine bottle of sauce came free",
    "the excellent router arrived unboxed",
    "the terrible courier left it out in the rain",
    "a fine pair of gloves fit perfectly",
    "the excellent kettle boiled in a minute",
]
TRAIN_CATEGORY = [
    ("the magistrates remanded the two defendants", "crime"),
    ("police charged the driver after the pursuit", "crime"),
    ("the handset ships with a larger battery", "tech"),
    ("the compiler gained a faster backend this week", "tech"),
    ("the marathon was won by a single second", "sport"),
    ("the away side took the trophy on penalties", "sport"),
    ("the two presidents signed a maritime accord", "world"),
    ("the assembly voted to admit the observers", "world"),
]


def train_rows():
    out = [_row(s, "agnews", _choice("topic", TOPIC_A_INSTR, TOPIC_CRIT_A, l))
           for s, l in TRAIN_TOPIC_A]
    out += [_row(s, "agnews", _choice("topic", TOPIC_B_INSTR, TOPIC_CRIT_B, "world"))
            for s in TRAIN_TOPIC_B]
    out += [_row(s, "agnews", _noul("is_business", c, l)) for s, c, l in TRAIN_BUSINESS]
    out += [_row(s, "boolq", _noul("answer", c, l)) for s, c, l in TRAIN_ANSWER]
    out += [_row(s, "yelp", _noul("recommend", c, l)) for s, c, l in TRAIN_RECOMMEND]
    out += [_row(s, "yelp", _score("stars", STARS_INSTR, STARS_LEGEND, l))
            for s, l in TRAIN_STARS]
    out += [_row(s, "yelp", _score("quality", QUALITY_INSTR, QUALITY_LEGEND, 2))
            for s in TRAIN_QUALITY]
    out += [_row(s, "dbpedia", _choice("category", CATEGORY_INSTR, CATEGORY_CRIT, l))
            for s, l in TRAIN_CATEGORY]
    return out


def _set_key(r):
    """`load_split`'s group key without the md5: source plus the question JSON minus labels.
    Two rows only share a question-set if their cues, types and criteria are identical, which
    is why the fixture's cue wording is what decides how many sets it holds."""
    return (r["_meta"]["source"],
            json.dumps({qid: {k: v for k, v in q.items() if k != "label"}
                        for qid, q in r["questions"].items()},
                       sort_keys=False, ensure_ascii=False))


SETS = len({_set_key(r) for r in rows()})
ROW_N = len(rows())
# Every set in this fixture asks exactly one question, so a slot and a set are the same count
# here -- and `coverage()["question_slots"]` is therefore also what the guard compares.
SLOTS = SETS
# The slots whose (qname, K) shape has a same-source partner (permute) and whose K has a
# partner in another source (cross-source). Both are the K=2 families: topic's two sets,
# is_business's five, answer's six, recommend's four. stars and quality are yelp-only at
# K=3 under two qnames, and category is the only K=4.
PARTNERED = 2 + 5 + 6 + 4
CELLS = 7

# The fit is the expensive part of the fixture and the reason it is module-scoped: ~3 s on this
# box for `TRAIN_EPOCHS` full-batch passes over 40 rows, once for the whole file.
TRAIN_EPOCHS = 30
TRAIN_LR = 3e-3


def fit(model, groups, hf, epochs=TRAIN_EPOCHS):
    """One full-batch Adam pass per question-set per epoch -- the same `build_batch` and
    `typed_loss` the trainer uses, so the fixture's model is a small real model rather than a
    random one with a label on it."""
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    for _epoch in range(epochs):
        model.train()
        for _wf, (questions, exs) in groups.items():
            b = build_batch(exs, hf, questions, "cpu")
            logits = model(b["state_ids"], b["state_len"], b["q_ids"], b["q_mask"],
                           b["span_mat"], b["opt_valid"], b["decide_idx"])
            loss = typed_loss(logits, b["gold"], torch.ones_like(b["gold"], dtype=torch.bool))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return loss.item()


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    """A pilot-shaped suite, a checkpoint *fitted on the suite's own train split*, and the
    metrics file the *trainer's* own `evaluate()` wrote for it.

    The fit is not decoration and is the whole reason the battery can bite: an unfitted model
    answers from the option spans alone, so four of the five treatment arms score byte-identical
    numbers to `as-scored` and every mutation of the transform-and-score path survives as a
    false negative. See the fixture comment above.
    """
    root = tmp_path_factory.mktemp("tier0-suite")
    ckpt = tmp_path_factory.mktemp("tier0-ckpt")
    (root / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows()),
                                     encoding="utf-8")
    (root / "development.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows(TOPIC_A_INSTR_DEV)), encoding="utf-8")
    (root / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train_rows()),
                                      encoding="utf-8")
    suites = load_suite(str(root))
    # the tokenizer sees the train texts too, exactly as a real run tokenizes the corpus it is
    # about to train on -- and the option strings must never fragment at eval time.
    tok = train_tokenizer(suite_texts(suites["train"]) + suite_texts(suites["test"]),
                          vocab_size=128)
    # train.py saves model.pt and tokenizer.json into the run's out dir together, so the
    # harness's default --run-dir works here; `--tokenizer` is for the download that split
    # them, and that path is tested on its own.
    tok.save(str(ckpt / "tokenizer.json"))
    hf = Tokenizer.from_file(str(ckpt / "tokenizer.json"))
    torch.manual_seed(0)
    model = MynaModel(MynaConfig(vocab=tok.get_vocab_size(), d_model=32, n_layers=1,
                                 n_heads=2, d_k=16, d_v=16, d_ff=32, d_ptr=16))
    model.to("cpu")
    fit(model, suites["train"], hf)
    torch.save({"cfg": vars(model.cfg), "state_dict": model.state_dict(),
                "temperature": TEMPERATURE}, ckpt / "model.pt")

    test_groups, dev_groups = suites["test"], suites["dev"]
    doc = {"temperature": TEMPERATURE, "last_step": LAST_STEP,
           "test": evaluate(model, hf, test_groups, "cpu", TEMPERATURE),
           "dev": evaluate(model, hf, dev_groups, "cpu", TEMPERATURE),
           "qtypes": {f"{wf}/{q.name}": q.type
                      for wf, (qs, _e) in test_groups.items() for q in qs}}
    metrics = root / "trainer.metrics.json"
    metrics.write_text(json.dumps(doc, indent=2))
    return SimpleNamespace(root=root, ckpt=ckpt, tok=ckpt / "tokenizer.json",
                           metrics=metrics, groups=test_groups, dev_groups=dev_groups,
                           train_groups=suites["train"], hf=hf, model=model,
                           vocab=tok.get_vocab_size(), doc=doc)


def argv(fx, *extra):
    return ["--run-dir", str(fx.ckpt), "--suite", str(fx.root), "--metrics", str(fx.metrics),
            "--min-rows", "1", *extra]


def ref_keys(fx):
    """The metrics' accuracy keys: no `:brier`, no `:ece`."""
    return [k for k in fx.doc["test"]
            if not (k.endswith(":brier") or k.endswith(":ece"))]


def singleton_keys(fx, groups=None):
    """Keys whose question-set holds one row, where one row is one whole flip."""
    groups = groups if groups is not None else fx.groups
    return [k for k in ref_keys(fx) if len(groups[k.split("/")[0]][1]) == 1]


def doctored(fx, tmp_path, mutate, split="test", name="doctored.json"):
    doc = json.loads(fx.metrics.read_text())
    mutate(doc[split])
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return p


def run(fx, tmp_path, capsys, *extra, out="w.json"):
    o = tmp_path / out
    rc = DQ.main(argv(fx, "--out", str(o), *extra))
    return rc, capsys.readouterr().out, json.loads(o.read_text())


def table(printed):
    """The table block only: header line plus the cell rows, stopping before the macro block.
    Read by offset, never by substring (§9.34 -- a substring check passed on a render whose
    macro row sat six columns right of its header)."""
    block = printed.split("macro by arm")[0]
    lines = block.splitlines()
    head = max(i for i, ln in enumerate(lines) if ln.split()[:1] == ["cell"])
    return lines[head], [ln for ln in lines[head + 1:] if ln.strip()]


# ------------------------------------------------------------------ the guard, field by field


def test_the_as_scored_arm_reproduces_what_the_trainer_computed(fx, tmp_path, capsys):
    """Two implementations of the same loop, one number.

    `evaluate()` produced the metrics file; `score()` re-derives it from `model.pt` read back
    off disk. If those ever disagree the whole table is unanchored, so this is the equality
    the harness exists to hold -- and it is checked against an aggregation built here, not
    against a literal somebody typed.
    """
    rc, printed, w = run(fx, tmp_path, capsys)
    assert rc == 0, printed
    keys = ref_keys(fx)
    assert w["guard"]["keys_compared"] == len(keys) == SLOTS, sorted(keys)
    assert w["guard"]["tie_flips"] == 0
    assert f"guard green: the as-scored arm reproduces {SLOTS} committed keys " \
        f"(0 single-row tie flips)" in printed
    # the independent path: myna.report's roll-up of evaluate()'s own numbers
    cells, unmatched = accuracy_cells(fx.doc["test"], fx.groups)
    assert unmatched == [], "accuracy_cells found sets this split does not carry"
    kept = roll_up(cell_stats(fx.groups), cells, keep_min=1)
    assert len(kept) == CELLS
    assert w["guard"]["macro_as_scored"] == pytest.approx(
        sum(r["acc"] for r in kept) / len(kept), abs=1e-12)
    assert w["model"]["params"] == sum(p.numel() for p in
                                       MynaModel(MynaConfig(**w["model"]["cfg"])).parameters())
    assert w["model"]["temperature"] == TEMPERATURE
    assert f"temperature {TEMPERATURE}" in printed.splitlines()[1]
    assert f"last_step {LAST_STEP}" in printed.splitlines()[1]


def test_a_key_off_by_more_than_a_row_refuses_and_prints_no_table(fx, tmp_path, capsys):
    """Five rows in that set, so one row is 0.2 and a half-point move is a different
    measurement. `bad` is what stops the table, and the table is the claim."""
    def mutate(m):
        k = next(key for key in ref_keys(fx) if len(fx.groups[key.split("/")[0]][1]) == 5)
        m[k] += 0.5

    o = tmp_path / "x.json"
    assert DQ.main(argv(fx, "--metrics", str(doctored(fx, tmp_path, mutate)),
                        "--out", str(o))) == 2
    out = capsys.readouterr()
    assert "THE RUN DOES NOT REPRODUCE THE COMMITTED METRICS" in out.err, out.err
    assert "delta 0.500000 > 0.200000 = one row" in out.err, out.err
    assert "guard green" not in out.out and "worst-d" not in out.out
    assert not o.exists(), "the witness was written even though the guard refused the table"


def test_a_single_row_flip_is_counted_as_a_tie_flip_not_a_defect(fx, tmp_path, capsys):
    """On a one-row set the only scores are 0.0 and 1.0, so a flip is exactly one row and is
    CPU-against-T4 rounding, not a different measurement. The guard has to *count* it."""
    key = singleton_keys(fx)[0]

    def mutate(m):
        m[key] = 1.0 - m[key]

    rc, printed, w = run(fx, tmp_path, capsys, "--metrics",
                         str(doctored(fx, tmp_path, mutate)), out="tie.json")
    assert rc == 0, printed
    assert w["guard"]["tie_flips"] == 1
    assert "(1 single-row tie flips)" in printed


def test_row_slack_keys_plus_one_is_not_rounding(fx, tmp_path, capsys):
    """`ROW_SLACK_KEYS` is the difference between "a few ties broke differently on this box"
    and "this is not the run that produced the report". Set it to infinity and the second
    reads as the first."""
    keys = singleton_keys(fx)[: DQ.ROW_SLACK_KEYS + 1]
    assert len(keys) == DQ.ROW_SLACK_KEYS + 1

    def mutate(m):
        for k in keys:
            m[k] = 1.0 - m[k]

    o = tmp_path / "x.json"
    assert DQ.main(argv(fx, "--metrics", str(doctored(fx, tmp_path, mutate)),
                        "--out", str(o))) == 2
    err = capsys.readouterr().err
    assert f"{len(keys)} keys differ by a whole row" in err, err


def test_a_key_the_split_cannot_produce_is_named_by_key(fx, tmp_path, capsys):
    def mutate(m):
        m["agnews#deadbeef/topic"] = 0.5

    o = tmp_path / "x.json"
    assert DQ.main(argv(fx, "--metrics", str(doctored(fx, tmp_path, mutate)),
                        "--out", str(o))) == 2
    err = capsys.readouterr().err
    assert "agnews#deadbeef/topic is in the committed metrics but this split does not " \
        "produce it" in err, err


def test_a_scored_key_the_metrics_file_lost_is_named(fx, tmp_path, capsys):
    """The other direction: the harness scored a set the published metrics do not mention.
    Silence here is how a changed split would be reported as the old result."""
    def mutate(m):
        m.pop(singleton_keys(fx)[0])

    o = tmp_path / "x.json"
    assert DQ.main(argv(fx, "--metrics", str(doctored(fx, tmp_path, mutate)),
                        "--out", str(o))) == 2
    err = capsys.readouterr().err
    assert "are not in the committed metrics" in err, err


def test_the_sidecars_are_skipped_and_do_not_count_as_compared():
    """A `:brier`/`:ece` sidecar is not a second accuracy claim, so it must neither refuse the
    run nor inflate `keys_compared`. Unit-tested because no fixture can reach this and the
    row-slack branch in one `main()` call."""
    arm, totals = {"a#1/answer": 0.6}, {"a#1/answer": 10}
    bad, slack, compared = DQ.guard_against_metrics(
        arm, {"a#1/answer": 0.6, "a#1/answer:brier": 0.9, "a#1/answer:ece": 0.9}, totals)
    assert (bad, slack, compared) == ([], 0, 1), (bad, slack, compared)


def test_the_tolerance_and_the_row_slack_are_two_different_numbers():
    """Inside `FLOAT_TOL` is exact; a hair over it on a 100-row key is a tie flip; two whole
    rows is a defect. Three thresholds, so a mutant that widens one of them has to move a
    number that the other one still refuses."""
    arm, totals = {"a#1/topic": 0.6}, {"a#1/topic": 100}
    assert DQ.guard_against_metrics(arm, {"a#1/topic": 0.6 + DQ.FLOAT_TOL / 10},
                                    totals)[1] == 0
    assert DQ.guard_against_metrics(arm, {"a#1/topic": 0.6 + DQ.FLOAT_TOL * 10},
                                    totals)[1] == 1
    bad, slack, _ = DQ.guard_against_metrics(arm, {"a#1/topic": 0.62}, totals)
    assert slack == 0 and len(bad) == 1, (bad, slack)
    six = {f"a{i}/topic": 0.6 for i in range(6)}
    bad, slack, compared = DQ.guard_against_metrics(six, {k: 0.61 for k in six},
                                                    {k: 100 for k in six})
    assert compared == 6 and slack == 6 and len(bad) == 1, (bad, slack, compared)
    assert "keys differ by a whole row" in bad[0], bad
    five = {f"a{i}/topic": 0.6 for i in range(5)}
    bad, slack, _ = DQ.guard_against_metrics(five, {k: 0.61 for k in five},
                                             {k: 100 for k in five})
    assert bad == [] and slack == 5, bad


# ------------------------------------------------------------------------- the six arms


def test_an_arm_is_a_copy_the_untouched_split_stays_the_ruler(fx):
    """The docstring says an ablated copy is the input under test and never the ruler. That is
    only true because of the `deepcopy`: a harness that ablated in place would score arm 2
    against the split that arm 1 already broke, and the `as-scored` guard would still pass
    because it runs last."""
    def read(groups):
        return {wf: ([q.instruction for q in qs], [tuple(q.options) for q in qs],
                     [e.state for e in exs]) for wf, (qs, exs) in groups.items()}

    before = read(fx.groups)
    for arm in DQ.ARMS:
        DQ.ablate(fx.groups, arm, 0)
    assert read(fx.groups) == before


def test_an_unknown_arm_names_the_arms_that_exist():
    with pytest.raises(ValueError) as e:
        DQ.ablate({}, "swap-instruction", 0)
    msg = str(e.value)
    assert "unknown arm 'swap-instruction'" in msg, msg
    for arm in DQ.ARMS:
        assert arm in msg, msg
    assert set(DQ.ARMS) == {"as-scored", "blank-instruction", "permute-instruction",
                            "cross-source-instruction", "blank-options", "swap-state"}


def test_the_control_arm_changes_nothing_and_counts_every_input(fx):
    """`coverage()` is the harness's own falsification line, so its denominators are pinned
    here: every slot and every row accounted for, nothing rewritten."""
    cov = DQ.coverage(fx.groups, DQ.ablate(fx.groups, "as-scored", 0))
    assert cov == {"question_slots": SLOTS, "rows": ROW_N, "instructions_changed": 0,
                   "option_sets_changed": 0, "states_changed": 0}, cov
    assert (SLOTS, ROW_N, CELLS) == (20, 33, 7), "the fixture moved; so must the counts"


def test_blank_instruction_rewrites_every_cue_and_nothing_else(fx):
    g = DQ.ablate(fx.groups, "blank-instruction", 0)
    cov = DQ.coverage(fx.groups, g)
    assert (cov["instructions_changed"], cov["option_sets_changed"],
            cov["states_changed"]) == (SLOTS, 0, 0), cov
    for _wf, (qs, _exs) in g.items():
        for q in qs:
            assert q.instruction == DQ.NEUTRAL_INSTRUCTION
    # one neutral *sentence*, and a sentence this corpus never says -- an empty cue would
    # tokenize to nothing and the arm would be measuring a missing input, not a neutral one.
    assert DQ.NEUTRAL_INSTRUCTION.strip() and DQ.NEUTRAL_INSTRUCTION.endswith(".")
    assert all(DQ.NEUTRAL_INSTRUCTION != q.instruction
               for _wf, (qs, _e) in fx.groups.items() for q in qs)


def test_blank_options_keeps_the_option_count_the_pointer_head_reads(fx):
    """The claim is 'same count, no text'. `K` is what the head is built around, so an arm
    that emits two spans for a three-option legend still reads as a blanking arm on the
    coverage line, and only this assertion says otherwise."""
    g = DQ.ablate(fx.groups, "blank-options", 0)
    cov = DQ.coverage(fx.groups, g)
    assert (cov["option_sets_changed"], cov["instructions_changed"],
            cov["states_changed"]) == (SLOTS, 0, 0), cov
    for wf, (qs, _exs) in g.items():
        for q, oq in zip(qs, fx.groups[wf][0]):
            assert q.options == [f"option {i + 1}" for i in range(len(oq.options))]
            assert len(q.options) == len(oq.options)


def test_permute_instruction_only_exchanges_sets_of_the_same_shape(fx):
    """Two restrictions at once: within a source, and within the ordered (qname, K) shape.
    yelp holds `stars` and `quality`, both score, both 3 options -- if qname left the bucket
    key the arm would exchange them and the table would say "the wording moved" when the
    question did."""
    g = DQ.ablate(fx.groups, "permute-instruction", 0)
    cov = DQ.coverage(fx.groups, g)
    assert cov["states_changed"] == 0 and cov["option_sets_changed"] == 0, cov
    assert 0 < cov["instructions_changed"] <= PARTNERED, cov

    def shape(qs):
        return tuple((q.name, len(q.options)) for q in qs)

    buckets = {}
    for wf, (qs, _exs) in fx.groups.items():
        buckets.setdefault((wf.split("#", 1)[0], shape(qs)), []).append(wf)
    for wf, (qs, _exs) in g.items():
        mine = buckets[(wf.split("#", 1)[0], shape(qs))]
        donors = [k for k in mine if k != wf]
        for i, (q, oq) in enumerate(zip(qs, fx.groups[wf][0])):
            if len(mine) == 1:
                assert q.instruction == oq.instruction, (wf, q.name)
                continue
            if q.instruction != oq.instruction:
                assert any(q.instruction == fx.groups[k][0][i].instruction for k in donors), \
                    (wf, q.name, q.instruction)
    # the two 3-option yelp cells sit in different-shape buckets, so neither may move
    for wf, (qs, _exs) in g.items():
        if qs[0].name in ("stars", "quality", "category"):
            for q, oq in zip(qs, fx.groups[wf][0]):
                assert q.instruction == oq.instruction, (wf, qs[0].name)


def test_cross_source_instruction_uses_another_source_or_says_nothing(fx):
    """`permute-instruction` was near the identity on the pilot for the reason in the
    docstring, so this is the arm that actually asks the question. Two things must hold: the
    cue is not one the *same* source ever used, and a width with no partner in another source
    keeps its own cue instead of reporting itself as tested."""
    g = DQ.ablate(fx.groups, "cross-source-instruction", 0)
    cov = DQ.coverage(fx.groups, g)
    assert (cov["instructions_changed"], cov["option_sets_changed"],
            cov["states_changed"]) == (PARTNERED, 0, 0), cov

    own, pool = {}, {}
    for wf, (qs, _exs) in fx.groups.items():
        source = wf.split("#", 1)[0]
        own.setdefault(source, set()).update(q.instruction for q in qs)
        for q in qs:
            pool.setdefault(len(q.options), set()).add((source, q.instruction))
    for wf, (qs, _exs) in g.items():
        source = wf.split("#", 1)[0]
        for q, oq in zip(qs, fx.groups[wf][0]):
            partners = [s for s, _t in pool[len(oq.options)] if s != source]
            if q.instruction != oq.instruction:
                assert partners, (wf, q.name)
                assert q.instruction not in own[source], (wf, q.name, q.instruction)
                assert any(s != source and t == q.instruction
                           for s, t in pool[len(oq.options)]), (wf, q.name)
            else:
                assert not partners, (wf, q.name)
    # and the widths that have no partner are exactly the ones the arm cannot test
    untestable = sorted(f"{wf.split('#', 1)[0]}/{q.name}"
                        for wf, (qs, _e) in g.items() for q in qs
                        if not any(s != wf.split("#", 1)[0]
                                   for s, _t in pool[len(q.options)]))
    assert untestable == ["dbpedia/category", "yelp/quality", "yelp/stars"]


def test_swap_state_is_a_permutation_pooled_by_source(fx):
    """The arm that carries the prior-collapse finding, so its transform is the claim. Golds
    and workflows must survive (the score stays the score), each source's state multiset must
    be exactly preserved (a shuffle, not a rewrite), and dbpedia's single row has nowhere to
    swap to -- which is why the published line prints states_changed out of the row count
    rather than asserting they are equal."""
    g = DQ.ablate(fx.groups, "swap-state", 0)
    cov = DQ.coverage(fx.groups, g)
    assert (cov["instructions_changed"], cov["option_sets_changed"]) == (0, 0), cov
    assert cov["states_changed"] <= ROW_N - 1

    orig, new, moved, total = defaultdict(Counter), defaultdict(Counter), Counter(), Counter()
    for wf, (_qs, exs) in fx.groups.items():
        orig[wf.split("#", 1)[0]].update(e.state for e in exs)
    for wf, (_qs, exs) in g.items():
        source = wf.split("#", 1)[0]
        new[source].update(e.state for e in exs)
        for e, oe in zip(exs, fx.groups[wf][1]):
            assert e.gold == oe.gold and e.workflow == oe.workflow, wf
            total[source] += 1
            moved[source] += int(e.state != oe.state)
    assert orig == new, "swap-state did not permute the states, it invented or dropped some"
    assert moved["dbpedia"] == 0 and total["dbpedia"] == 1
    for source in ("agnews", "boolq", "yelp"):
        assert moved[source] > 0, f"{source} is a no-op arm, not a swap"
        assert total[source] > 1
    assert sum(moved.values()) == cov["states_changed"], \
        "coverage() does not count the rows the transform actually moved"


def test_the_states_a_source_offers_are_all_distinct_so_a_swap_is_visible(fx):
    """The fixture's own precondition: if two rows of a source shared a state string,
    `swap-state` could permute them and print `states_changed` short for a reason that has
    nothing to do with the code."""
    for source in ("agnews", "boolq", "yelp"):
        seen = [e.state for wf, (_q, exs) in fx.groups.items()
                if wf.split("#", 1)[0] == source for e in exs]
        assert len(seen) == len(set(seen)) > 1, source
        cues = [q.instruction for wf, (qs, _e) in fx.groups.items()
                if wf.split("#", 1)[0] == source for q in qs]
        assert len(cues) == len(set(cues)), f"{source} repeats a cue, so a rewrite of it " \
                                           "cannot be counted"


def test_an_arm_is_reproducible_and_the_seed_changes_it(fx):
    """`random.Random(f"{seed}|...")` is what makes the table one measurement instead of a
    fresh draw per run, and the seed argument is what makes two runs comparable. Both
    directions: the same seed twice is identical, and `--seed` is not decoration."""
    for arm in ("swap-state", "permute-instruction", "cross-source-instruction"):
        a, b = DQ.ablate(fx.groups, arm, 0), DQ.ablate(fx.groups, arm, 0)
        assert a.keys() == b.keys()
        assert all([q.instruction for q in a[k][0]] == [q.instruction for q in b[k][0]]
                   and [e.state for e in a[k][1]] == [e.state for e in b[k][1]]
                   for k in a), arm
    one, two = DQ.ablate(fx.groups, "swap-state", 1), DQ.ablate(fx.groups, "swap-state", 2)
    assert any([e.state for e in one[k][1]] != [e.state for e in two[k][1]] for k in one), \
        "swap-state ignores its seed, so `--seed` is decoration"


# ---------------------------------------------------------------- the instrument can move
#
# Everything above checks what each arm *does to the split*. This section checks that the
# difference survives into the printed numbers, which is the only half the battery can bite
# on: a mutant that scores `groups` instead of the ablated copy, or that calls `ablate` with a
# hardcoded seed, changes not one byte of an unfitted model's output. So these are
# precondition tests in the §9.48 sense -- if they ever fail, it is the instrument that has
# gone blunt and the mutation counts on this file stop meaning anything.


def test_the_fit_split_shares_no_row_with_the_scored_split(fx):
    """The fixture's own honesty condition: if a train row were also a test row, a fitted model
    could answer it from memory, every arm that destroys the state would look harmless, and the
    sensitivity below would be an artifact of memorisation rather than of reading."""
    train_states = {e.state for _wf, (_q, exs) in fx.train_groups.items() for e in exs}
    test_states = {e.state for _wf, (_q, exs) in fx.groups.items() for e in exs}
    assert not (train_states & test_states), sorted(train_states & test_states)[:3]
    # every cell the harness scores is taught, or its invariance is a fixture gap
    taught = {(wf.split("#", 1)[0], q.name) for wf, (qs, _e) in fx.train_groups.items()
              for q in qs}
    assert taught == set(cell_stats(fx.groups)) == set(cell_stats(fx.train_groups))
    # and the fit split cannot be answered by counting: no train or test state carries a digit
    assert not any(c.isdigit() for s in train_states | test_states for c in s), \
        "a state with a digit in it can be indexed into a label rather than read"


def test_the_fitted_model_reads_something_it_can_lose(fx, tmp_path, capsys):
    """The whole point of fitting the fixture: every one of the five treatment arms costs at
    least one cell a number, and at least one arm costs the *macro* ground. A random init scored
    byte-identical tables for four of the five arms and 0 of 20 per-set keys moved, so this is
    the line that says the battery's negative results are evidence rather than a blunt
    instrument -- and the exact mover set is pinned, because an arm that quietly stops moving is
    the §9.48 defect arriving as a fixture problem.

    It is worth pinning which cells move, since that profile is the fixture reproducing the
    pilot's own finding rather than an artifact: `yelp/quality` is the constant emitter (it
    answers its one label whatever it is shown, so it sits on its own 1.000 floor with nothing
    to lose), and `dbpedia/category`'s single row is scored 0.000 by a model that never learned
    the 4-way class, so no arm can move it further. Everything else is arm-sensitive, and
    `swap-state` costing five cells is the state-reading claim made checkable. If this profile
    ever changes on an *unmutated* tree the fit has moved -- re-measure it, do not loosen the
    assertion.
    """
    rc, _printed, w = run(fx, tmp_path, capsys)
    assert rc == 0
    profile = {arm: sorted(name for name, cols in w["cells"].items()
                           if cols[arm] is not None and cols[arm] != cols["as-scored"])
               for arm in DQ.ARMS if arm != "as-scored"}
    assert profile == {
        "blank-instruction": ["agnews/is_business", "agnews/topic", "boolq/answer",
                              "yelp/recommend", "yelp/stars"],
        "permute-instruction": ["agnews/is_business", "agnews/topic", "boolq/answer",
                               "yelp/recommend"],
        "cross-source-instruction": ["agnews/is_business", "agnews/topic", "boolq/answer",
                                     "yelp/recommend"],
        "blank-options": ["agnews/is_business", "agnews/topic", "boolq/answer", "yelp/quality",
                          "yelp/recommend", "yelp/stars"],
        "swap-state": ["agnews/is_business", "agnews/topic", "boolq/answer", "yelp/recommend",
                       "yelp/stars"],
    }, profile
    for arm in profile:
        cov = w["coverage"][arm]
        assert cov["instructions_changed"] + cov["option_sets_changed"] + \
            cov["states_changed"] > 0, \
            f"{arm} moved a number while rewriting nothing"
    worst = min(w["macro"][a] for a in DQ.ARMS if a != "as-scored")
    assert worst < w["macro"]["as-scored"], \
        f"no arm costs the macro anything ({w['macro']}) -- the model reads no input"


def test_the_witnesses_coverage_is_the_transform_recomputed_here(fx, tmp_path, capsys):
    """`coverage()` printed beside each arm is the line that says whether the arm fired, and
    the witness is where a reader checks it. So the stored count is recomputed from `ablate()`
    independently for all six arms, at the seed the run records -- a `covs[arm] =
    coverage(groups, groups)` mutant prints a rewriting arm and stores a zero."""
    rc, _printed, w = run(fx, tmp_path, capsys)
    assert rc == 0
    assert set(w["coverage"]) == set(DQ.ARMS)
    for arm in DQ.ARMS:
        fresh = DQ.coverage(fx.groups, DQ.ablate(fx.groups, arm, w["seed"]))
        assert w["coverage"][arm] == fresh, (arm, w["coverage"][arm], fresh)


def test_the_seed_reaches_the_scored_numbers_not_only_the_transform(fx, tmp_path, capsys):
    """`--seed` decides which state each row is swapped with, so two seeds are two
    measurements of the same arm. The transform is already shown to depend on the seed above;
    this says the *table* does, which is the half a harness could lose by scoring the same copy
    twice or by pinning the seed inside `main()`.
    """
    rc, _p0, w0 = run(fx, tmp_path, capsys, out="seed0.json")
    rc2, _p3, w3 = run(fx, tmp_path, capsys, "--seed", "3", out="seed3.json")
    assert (rc, rc2) == (0, 0)
    assert w0["seed"] == 0 and w3["seed"] == 3
    moved = [name for name, cols in w3["cells"].items()
             if cols["swap-state"] != w0["cells"][name]["swap-state"]]
    assert moved, "no cell's swap-state number depends on --seed"
    # the control arms are seed-invariant by construction, and saying so here is what stops a
    # mutant that re-draws the whole split per seed from looking like a fix
    for arm in ("as-scored", "blank-instruction", "blank-options"):
        assert all(cols[arm] == w0["cells"][name][arm] for name, cols in w3["cells"].items()), \
            arm


# ------------------------------------------------------------------------ main() printing


def test_the_printed_table_is_the_witness_cell_by_cell(fx, tmp_path, capsys):
    """Every printed accuracy, floor, uniform floor and worst-delta has to be the witness's
    own number, read by column offset. `n/a` means a cell an arm could not score, so on a
    complete six-arm run it must never appear -- and the table must carry all seven cells."""
    rc, printed, w = run(fx, tmp_path, capsys)
    assert rc == 0
    arms = w["arms"]
    head, body = table(printed)
    cols = head.split()
    assert cols[:4] == ["cell", "type", "n", "K"], head
    assert cols[4:4 + len(arms)] == arms, head
    assert cols[-3:] == ["floor", "unif", "worst-d"], head
    read = {}
    for ln in body:
        parts = ln.split()
        assert len(parts) == 4 + len(arms) + 3, ln
        name, typ, n, k = parts[0], parts[1], int(parts[2]), int(parts[3])
        vals = [float(v) for v in parts[4:4 + len(arms)]]
        floor, unif, worst = float(parts[-3]), float(parts[-2]), float(parts[-1])
        assert "n/a" not in ln, ln
        cell = w["cells"][name]
        assert [cell[a] for a in arms] == pytest.approx(vals, abs=5e-4), name
        assert worst == pytest.approx(min(vals) - vals[0], abs=5e-4), (name, worst)
        read[name] = (typ, n, k, floor, unif)
    assert len(read) == CELLS, sorted(read)
    for (source, qname), s in cell_stats(fx.groups).items():
        typ, n, k, floor, unif = read[f"{source}/{qname}"]
        assert typ == s["type"] and n == s["n"] and k == max(s["options"]), (source, qname)
        assert floor == pytest.approx(s["majority"], abs=5e-4)
        assert unif == pytest.approx(s["uniform"], abs=5e-4)


def test_the_emission_counts_are_the_predictions_not_the_gold(fx, tmp_path, capsys):
    """`collapse_share` is the number SPEC §9.47 quotes for the three constant emitters, and it
    is the *predicted* label's share -- not the gold majority's, which is printed in the same
    line and is 1.000 for a cell like `yelp/quality` that has one gold label. Both histograms are
    rebuilt here from `score()`'s own chunks and compared field by field, because a mutant that
    reads `g` where the code reads `p` would leave every printed line looking self-consistent."""
    _acc, preds, _tot = DQ.score(fx.model, fx.hf, fx.groups, "cpu", TEMPERATURE)
    ph, gh = defaultdict(Counter), defaultdict(Counter)
    for key, chunks in preds.items():
        wf, qname = key.split("/", 1)
        for p, g, _conf in chunks:
            ph[(wf.split("#", 1)[0], qname)].update(p)
            gh[(wf.split("#", 1)[0], qname)].update(g)
    rc, _printed, w = run(fx, tmp_path, capsys)
    assert rc == 0
    separates = 0
    for name, d in w["distributions"].items():
        cell = (name.split("/")[0], name.split("/")[1])
        n = sum(ph[cell].values())
        assert n == sum(gh[cell].values()) > 0, name
        assert d["n"] == n, name
        assert d["collapse_share"] == pytest.approx(ph[cell].most_common(1)[0][1] / n), name
        assert d["gold_majority_share"] == pytest.approx(gh[cell].most_common(1)[0][1] / n), name
        assert d["distinct_labels_predicted"] == len(ph[cell]), name
        assert d["predicted_top"][0][1] == ph[cell].most_common(1)[0][1], name
        assert d["gold_top"] == [[i, c] for i, c in gh[cell].most_common(3)], name
        separates += int(d["collapse_share"] != d["gold_majority_share"])
    assert separates >= 2, "every cell's predicted share equals its gold share here, so the " \
                           "two fields are indistinguishable -- widen the fixture"


def test_the_macro_block_prints_deltas_against_the_as_scored_arm(fx, tmp_path, capsys):
    """`Δ` is what the write-up quotes (blank-options -0.2225 is the headline of the whole
    finding). Printed against the floor instead of against `as-scored`, every arm would read
    as a margin over chance and an ablation would look like a leaderboard."""
    rc, printed, w = run(fx, tmp_path, capsys)
    assert rc == 0
    block = printed.split("macro by arm")[1]
    lines = [ln for ln in block.splitlines() if " vs floor " in ln]
    assert len(lines) == len(DQ.ARMS), lines
    base = w["macro"]["as-scored"]
    for arm in DQ.ARMS:
        ln = next(l for l in lines if l.strip().startswith(arm))
        level = float(ln.split(" vs floor ")[0].split()[-1])
        floor_ = float(ln.split(" vs floor ")[1].split()[0])
        delta = float(ln.split("Δ ")[1])
        assert level == pytest.approx(w["macro"][arm], abs=5e-5), arm
        assert floor_ == pytest.approx(w["macro_floor"], abs=5e-5), arm
        assert delta == pytest.approx(w["macro"][arm] - base, abs=5e-5), arm
    assert "as-scored" in block and "Δ +0.0000" in block, \
        "as-scored is its own baseline, so its delta is exactly zero"


def test_the_emission_block_lists_every_kept_cell_and_the_witness_agrees(fx, tmp_path,
                                                                        capsys):
    """`only_below_floor=False` is a deliberate choice in `main()`: a collapse sitting above
    its own floor is the same defect, and a filtered list hides it. So the block names all
    seven cells, and each line carries the witness's own collapse share, label count and
    state-delta."""
    rc, printed, w = run(fx, tmp_path, capsys)
    assert rc == 0
    lines = [ln.strip() for ln in printed.split("per-cell label emission")[1].splitlines()
             if ln.startswith("  ") and "/" in ln]
    assert len(lines) == CELLS, lines
    assert set(w["distributions"]) == {ln.split()[0] for ln in lines}
    for ln in lines:
        name = ln.split()[0]
        d = w["distributions"][name]
        assert float(ln.split("acc ")[1].split()[0]) == pytest.approx(d["acc"], abs=5e-4)
        assert float(ln.split("collapse ")[1].split()[0]) == pytest.approx(
            d["collapse_share"], abs=5e-4), name
        assert int(ln.split("labels ")[1].split("/")[0]) == d["distinct_labels_predicted"]
        assert ln.split("labels ")[1].split()[0].split("/")[1] == str(d["options"]), name
        swap = w["cells"][name]["swap-state"]
        assert float(ln.split("state-Δ ")[1]) == pytest.approx(swap - d["acc"], abs=5e-4), \
            (name, ln)
    # a constant emitter is a measurable shape, not a story: yelp/quality has one gold label
    # on all three rows, so its gold share is 1.000 and its floor is 1.000.
    d = w["distributions"]["yelp/quality"]
    assert d["gold_majority_share"] == 1.0 and d["options"] == 3 and d["type"] == "score"
    assert d["gold_top"] == [[2, 3]], d["gold_top"]
    # the witness carries its prediction counts too, and n is the cell's row count
    assert d["n"] == 3 and sum(c for _i, c in d["predicted_top"]) == 3
    assert len(d["predicted_top"]) <= 3


def test_each_arms_line_prints_its_own_macro_floor_and_rewrite_counts(fx, tmp_path, capsys):
    """The per-arm line is what `runs/diag_question_ablation.log` is read for, and the committed
    log test cannot check it: `runs/` is symlinked into the battery's scratch copy, so a mutant
    of this print statement leaves the committed artifact untouched and its test green (§9.48).
    Re-run in process: each arm's own macro, the floor all arms share, and the three rewrite
    counts with their denominators."""
    rc, printed, w = run(fx, tmp_path, capsys)
    assert rc == 0
    for arm in DQ.ARMS:
        ln = next(l for l in printed.splitlines() if l.startswith(arm))
        cov = w["coverage"][arm]
        assert f"macro {w['macro'][arm]:.4f} (floor {w['macro_floor']:.4f})" in ln, (arm, ln)
        assert (f"rewrote {cov['instructions_changed']:>4d}/{cov['question_slots']} cues, "
                f"{cov['option_sets_changed']:>4d} option sets, "
                f"{cov['states_changed']:>4d}/{cov['rows']} states") in ln, (arm, ln)
    assert "0 single-row tie flips" in printed


def test_the_guard_note_prints_the_count_it_computed_not_a_round_zero(fx, tmp_path, capsys):
    """`guard green: … N committed keys (M single-row tie flips)` is the line that makes the
    table quotable, and both numbers are computed. A note that hardcodes either one is a banner
    that survives the defect it is supposed to announce, so this pins the whole sentence against
    the witness fields -- and the tie-flip half is exercised for real by a doctored metric."""
    rc, printed, w = run(fx, tmp_path, capsys)
    assert rc == 0
    assert (f"guard green: the as-scored arm reproduces {w['guard']['keys_compared']} "
            f"committed keys ({w['guard']['tie_flips']} single-row tie flips) · macro "
            f"{w['guard']['macro_as_scored']:.10f}") in printed


def test_an_arm_the_run_skipped_leaves_its_column_out(fx, tmp_path, capsys):
    """`--arms` is how the battery keeps a mutant affordable, so the table has to survive a
    subset: the header lists only the arms run, the state-Δ column has no number to print
    when `swap-state` was not one of them, and the witness lists only what it holds."""
    rc, printed, w = run(fx, tmp_path, capsys, "--arms", "as-scored", "blank-options",
                         out="subset.json")
    assert rc == 0
    assert w["arms"] == ["as-scored", "blank-options"]
    assert set(w["cells"]["yelp/stars"]) == {"as-scored", "blank-options"}
    assert set(w["coverage"]) == {"as-scored", "blank-options"}
    head, _body = table(printed)
    assert head.split()[4:6] == ["as-scored", "blank-options"], head
    assert "blank-instruction" not in head
    tail = [ln for ln in printed.split("per-cell label emission")[1].splitlines()
            if ln.startswith("  ") and "/" in ln]
    assert len(tail) == CELLS
    assert all(ln.rstrip().endswith("state-Δ") for ln in tail), tail[:2]


def test_min_rows_moves_a_cell_out_of_the_table_the_macro_and_the_floors_together(
        fx, tmp_path, capsys):
    """`--min-rows` is derived from the rows on disk, and the roll-up it feeds is the same one
    `myna.report` publishes: the floors in the macro line are the *kept* cells' floors, never
    the full split's. yelp/quality (3 rows) and dbpedia/category (1) leave all three places at
    once."""
    rc, printed, w = run(fx, tmp_path, capsys, "--min-rows", "4", out="keep.json")
    assert rc == 0 and w["keep_min"] == 4
    assert "yelp/quality" not in printed and "dbpedia/category" not in printed
    assert len(w["cells"]) == 5
    cells, _ = accuracy_cells(fx.doc["test"], fx.groups)
    kept = roll_up(cell_stats(fx.groups), cells, keep_min=4)
    everything = roll_up(cell_stats(fx.groups), cells, keep_min=1)
    assert len(kept) == 5 and len(everything) == CELLS
    assert w["macro"]["as-scored"] == pytest.approx(
        sum(r["acc"] for r in kept) / len(kept), abs=1e-12)
    assert w["macro_floor"] == pytest.approx(
        sum(r["majority"] for r in kept) / len(kept), abs=1e-12)
    assert w["macro_floor"] != pytest.approx(
        sum(r["majority"] for r in everything) / len(everything), abs=1e-9), \
        "the floor was carried over from the unfiltered scope"


# --------------------------------------------------------------------------------- inputs


def test_a_min_rows_that_keeps_no_cell_refuses_with_its_own_number(fx, tmp_path, capsys):
    """`--min-rows` is published and `myna.report`'s default is 30, so a split whose biggest
    cell is smaller is a reachable call. Every roll-up is empty there, `macro()` answers with
    `None`, and the per-arm print dies on a format string -- which is a traceback where the
    harness should have said which number it could not keep. The refusal also proves the default
    came from `MIN_CELL_ROWS` and not from a literal typed next door."""
    from myna.report import MIN_CELL_ROWS
    o = tmp_path / "x.json"
    assert DQ.main(["--run-dir", str(fx.ckpt), "--suite", str(fx.root),
                    "--metrics", str(fx.metrics), "--out", str(o)]) == 2
    err = capsys.readouterr().err
    assert f"--min-rows {MIN_CELL_ROWS} keeps no cell" in err, err
    assert "largest cell is 9 rows over 7 cells" in err, err
    assert not o.exists()
    assert DQ.main(argv(fx, "--min-rows", "400", "--out", str(o))) == 2
    assert "--min-rows 400 keeps no cell" in capsys.readouterr().err
    # and the boundary itself still prints a table: at 3 the one-row cell leaves and the 3-row
    # cell stays, so the refusal is about keeping *nothing*, not about small numbers.
    rc, printed, w = run(fx, tmp_path, capsys, "--min-rows", "3", out="three.json")
    assert rc == 0 and w["keep_min"] == 3
    assert "dbpedia/category" not in printed and "yelp/quality" in printed
    assert len(w["cells"]) == CELLS - 1


def test_a_tokenizer_named_by_a_flag_binds_that_file_not_the_default_path(fx, tmp_path, capsys):
    """The two Kaggle downloads reached this box as two directories, so `--tokenizer` exists --
    and a digest computed from `run_dir/tokenizer.json` while a different file did the scoring
    would bind the witness to an artifact that was never used. Here the two files are
    *behaviourally* identical (the same vocab, re-serialised), so the guard cannot catch it and
    only the digest line can."""
    raw = json.loads(fx.tok.read_text())
    twin = tmp_path / "tokenizer-twin.json"
    twin.write_text(json.dumps(raw, indent=4, sort_keys=True))
    assert twin.read_bytes() != fx.tok.read_bytes(), \
        "the two serialisations came out byte-identical, so this test proves nothing"
    elsewhere = tmp_path / "ckpt-with-twin"
    elsewhere.mkdir()
    shutil.copyfile(fx.ckpt / "model.pt", elsewhere / "model.pt")
    shutil.copyfile(twin, elsewhere / "tokenizer.json")
    o = tmp_path / "twin.json"
    assert DQ.main(["--run-dir", str(elsewhere), "--tokenizer", str(fx.tok),
                    "--suite", str(fx.root), "--metrics", str(fx.metrics),
                    "--min-rows", "1", "--out", str(o)]) == 0
    w = json.loads(o.read_text())
    assert w["model"]["tokenizer_path"] == str(fx.tok)
    assert w["model"]["tokenizer_sha256"] == hashlib.sha256(fx.tok.read_bytes()).hexdigest()
    assert w["model"]["tokenizer_sha256"] != \
        hashlib.sha256((elsewhere / "tokenizer.json").read_bytes()).hexdigest()
    assert w["guard"]["tie_flips"] == 0, "the two tokenizer files did not score alike"


def test_a_metrics_file_with_no_step_recorded_says_so(fx, tmp_path, capsys):
    """`model.pt` from the committed run carries no step at all, so the number in the printed
    line comes from the metrics file -- and when that file has no `last_step` either, the
    harness must not print a bare `0`, which reads as "step zero of a run" rather than "not
    recorded". §9.37's rule that a witness names where each figure came from."""
    doc = json.loads(fx.metrics.read_text())
    doc.pop("last_step")
    m = tmp_path / "nostep.json"
    m.write_text(json.dumps(doc))
    rc, printed, w = run(fx, tmp_path, capsys, "--metrics", str(m), out="nostep.json")
    assert rc == 0
    assert w["last_step"] == "not recorded"
    assert "last_step not recorded (from nostep.json" in printed.splitlines()[1], printed[:400]
    assert w["model"]["step"] == -1


def test_the_split_selects_which_block_of_the_metrics_file_is_the_ruler(fx, tmp_path,
                                                                       capsys):
    """`--split dev` is scored against the file's `dev` block. The fixture gives dev one
    different cue, so its keys differ from test's: hand it the test block and the guard names
    a key the split does not produce, which is the only way a hardcoded "test" is detectable
    at all."""
    rc, printed, w = run(fx, tmp_path, capsys, "--split", "dev", out="dev.json")
    assert rc == 0, printed
    assert w["split"] == "dev" and w["guard"]["tie_flips"] == 0

    def put_the_test_block_in_dev(doc):
        doc["dev"] = dict(json.loads(fx.metrics.read_text())["test"])

    o = tmp_path / "x.json"
    assert DQ.main(argv(fx, "--split", "dev", "--metrics",
                        str(doctored(fx, tmp_path, put_the_test_block_in_dev, split="dev",
                                     name="wrongblock.json")), "--out", str(o))) == 2
    err = capsys.readouterr().err
    assert "does not produce it" in err, err


def test_a_tokenizer_that_is_not_where_the_default_points_refuses_by_name(fx, tmp_path):
    """The defect this harness actually shipped with: `model.pt` and `tokenizer.json` reached
    this box as two Kaggle downloads, and a default that resolves to a missing file is a
    published command that has never been run. Both paths name the file, and one of them names
    the flag that fixes it."""
    lonely = tmp_path / "ckpt-no-tok"
    lonely.mkdir()
    shutil.copyfile(fx.ckpt / "model.pt", lonely / "model.pt")
    with pytest.raises(SystemExit) as e:
        DQ.main(["--run-dir", str(lonely), "--suite", str(fx.root), "--min-rows", "1",
                 "--metrics", str(fx.metrics), "--out", str(tmp_path / "x.json")])
    msg = str(e.value)
    assert "no tokenizer at" in msg and str(lonely / "tokenizer.json") in msg, msg
    assert "--tokenizer" in msg, msg
    with pytest.raises(SystemExit) as e:
        DQ.main(argv(fx, "--tokenizer", str(tmp_path / "nope.json"),
                     "--out", str(tmp_path / "x.json")))
    assert "no tokenizer at" in str(e.value)


def test_an_empty_split_and_a_missing_metrics_file_say_which_one_it_was(fx, tmp_path,
                                                                       capsys):
    """Two boundary errors, each with its own words. A `return 2` that prints nothing is how a
    wrong `--split` turns into a silent nothing-table."""
    o = tmp_path / "x.json"
    assert DQ.main(argv(fx, "--out", str(o), "--split", "calibration")) == 2
    assert "calibration split of" in capsys.readouterr().err
    assert not o.exists()
    assert DQ.main(argv(fx, "--metrics", str(tmp_path / "nope.json"),
                        "--out", str(o))) == 2
    assert "are not on disk" in capsys.readouterr().err
    assert not o.exists()


def test_the_witness_binds_the_inputs_it_scored(fx, tmp_path, capsys):
    """§9.37: a witness binds its unit *and* its inputs. The digests, the tokenizer's path
    (which is not inside `--run-dir` on this box), the metrics file the guard read, the
    `last_step` that came from that file rather than from the weights, and the flags actually
    parsed all have to be in the artifact."""
    o = tmp_path / "w.json"
    assert DQ.main(argv(fx, "--tokenizer", str(fx.tok), "--out", str(o))) == 0
    printed = capsys.readouterr().out
    w = json.loads(o.read_text())
    assert w["model"]["tokenizer_sha256"] == hashlib.sha256(fx.tok.read_bytes()).hexdigest()
    assert w["model"]["model_sha256"] == hashlib.sha256(
        (fx.ckpt / "model.pt").read_bytes()).hexdigest()
    assert w["model"]["tokenizer_path"] == str(fx.tok)
    assert w["model"]["cfg"]["n_layers"] == 1 and w["model"]["cfg"]["vocab"] == fx.vocab
    assert w["metrics"] == str(fx.metrics) and w["last_step"] == LAST_STEP
    assert w["run_dir"] == str(fx.ckpt) and w["suite"] == str(fx.root)
    assert w["split"] == "test" and w["device"] == "cpu" and w["seed"] == 0
    assert w["keep_min"] == 1
    # the checkpoint records no step at all, and the artifact says so instead of inventing one
    assert w["model"]["step"] == -1
    assert "the checkpoint records no step" in printed.splitlines()[1]
    assert f"model {w['model']['model_sha256'][:12]}" in printed
    assert f"tokenizer {w['model']['tokenizer_sha256'][:12]}" in printed
    assert f"wrote {o}" in printed
    assert w["guard"]["float_tol"] == DQ.FLOAT_TOL
    assert w["guard"]["row_slack_keys"] == DQ.ROW_SLACK_KEYS
    assert set(w["macro"]) == set(w["arms"]) == set(DQ.ARMS)
    assert w["macro_uniform"] == pytest.approx(
        sum(r["uniform"] for r in roll_up(cell_stats(fx.groups),
                                          accuracy_cells(fx.doc["test"], fx.groups)[0],
                                          keep_min=1)) / CELLS, abs=1e-12)
    # `cmd` names the flags this call parsed, not the process's argv -- under pytest the two
    # differ, and a witness whose cmd is a pytest invocation is a receipt for nothing.
    given = argv(fx, "--tokenizer", str(fx.tok), "--out", str(o))
    tokens = shlex.split(w["cmd"])
    assert tokens[0] == sys.executable, w["cmd"]
    assert tokens[1].endswith("bench/diag_question_ablation.py"), w["cmd"]
    assert tokens[2:] == given, w["cmd"]
    assert "--tokenizer" in w["cmd"] and "--run-dir" in w["cmd"]


# ------------------------------------------------------- the committed artifacts have to agree

SKIP = pytest.mark.skipif(not (WITNESS.exists() and REPORT.exists()),
                          reason="the Tier 0 witness or the Kaggle report is not on disk")


@SKIP
def test_the_committed_witness_is_the_kaggle_runs_own_number():
    """The reason Tier 0 was run at all: G1's 0.4893 became checkable on this box instead of
    inherited from a transcript. The as-scored column is the committed report's cell-by-cell
    accuracy, the guard says zero tie flips, and the macro is the report's published macro."""
    w = json.loads(WITNESS.read_text())
    report = json.loads(REPORT.read_text())
    assert w["guard"]["tie_flips"] == 0
    assert w["guard"]["keys_compared"] == 716, w["guard"]
    assert w["arms"] == list(DQ.ARMS)
    assert w["coverage"]["as-scored"] == {"question_slots": 716, "rows": 1176,
                                         "instructions_changed": 0, "option_sets_changed": 0,
                                         "states_changed": 0}
    assert w["guard"]["macro_as_scored"] == pytest.approx(report["macro"]["acc"], abs=1e-6)
    assert w["macro_floor"] == pytest.approx(report["macro"]["majority"], abs=1e-6)
    assert w["macro_uniform"] == pytest.approx(report["macro"]["uniform"], abs=1e-6)
    by_label = {f"{c['source']}/{c['question']}": c for c in report["cells"]}
    assert set(w["cells"]) == set(by_label), sorted(set(w["cells"]) ^ set(by_label))
    for name, cols in w["cells"].items():
        assert cols["as-scored"] == pytest.approx(by_label[name]["acc"], abs=1e-6), name
        assert all(cols[a] is not None for a in w["arms"]), name
    assert w["model"]["params"] == 16926848, "the published run's parameter count moved"


@SKIP
def test_the_committed_docstring_pairs_are_the_committed_reports_own_numbers():
    """The harness's docstring names five below-floor cells with three digits each, and the
    README quotes that sentence. Prose carrying figures, so the figures come out of the
    artifact -- §9.30 applied to a docstring."""
    report = json.loads(REPORT.read_text())
    by_label = {f"{c['source']}/{c['question']}": c for c in report["cells"]}
    text = " ".join(DQ.__doc__.split())
    pairs = re.findall(r"([a-z0-9]+/[a-z_]+) (\d\.\d{3}) vs (\d\.\d{3})",
                       text[text.index("G1 measures"):text.index("Before buying")])
    assert len(pairs) == 5, pairs
    for name, acc, floor in pairs:
        c = by_label[name]
        assert f"{c['acc']:.3f}" == acc and f"{c['majority']:.3f}" == floor, name
        assert c["acc"] < c["majority"], name
    below = sorted(n for n, c in by_label.items()
                   if c["acc"] is not None and c["acc"] < c["majority"])
    assert below == sorted(n for n, _a, _f in pairs), below


@SKIP
def test_the_committed_witness_is_what_the_spec_prose_says():
    """SPEC §9.47's three readings are eight cells named, three floors quoted to four decimals
    and a "six of them by more than 0.06" -- all of them counts out of this artifact. Pinned
    because prose quoting an artifact without a test is where §9.30's drift starts, and this one
    is the finding the next three months of the project is scheduled around."""
    w = json.loads(WITNESS.read_text())
    cells = w["cells"]
    stateless = sorted(n for n, c in cells.items() if c["swap-state"] == c["as-scored"])
    assert stateless == ["agnews/is_scitech", "agnews/is_sports", "agnews/is_world",
                        "amazon/stars", "banking77/intent", "contrastive/decision",
                        "mnli/relation", "sst5/sentiment"], stateless
    assert len(stateless) == len(cells) // 2 == 8
    hurt = {n: cells[n]["swap-state"] - cells[n]["as-scored"] for n in cells
            if abs(cells[n]["swap-state"] - cells[n]["as-scored"]) > 0.06}
    assert sorted(hurt) == ["agnews/topic", "dbpedia14/category", "imdb/positive",
                            "trec/answer_type", "yelp/rating", "yelp/recommend"], sorted(hurt)
    assert round(cells["agnews/topic"]["swap-state"]
                - cells["agnews/topic"]["as-scored"], 6) == -0.060345
    # the three constant emitters: one label predicted on every row, scoring exactly their own
    # majority floor, which is the sentence "an association, not evidence" turned into arithmetic
    for name in ("agnews/is_scitech", "agnews/is_sports", "agnews/is_world"):
        d = w["distributions"][name]
        assert d["type"] == "noul" and d["distinct_labels_predicted"] == 1, name
        assert d["collapse_share"] == 1.0, name
        assert f"{d['acc']:.4f}" == f"{d['majority']:.4f}", (name, d["acc"], d["majority"])
        assert cells[name]["swap-state"] == cells[name]["as-scored"], name
    # and the arm itself is not a no-op: 1,164 of the 1,176 states were rewritten
    cov = w["coverage"]["swap-state"]
    assert (cov["states_changed"], cov["rows"]) == (1164, 1176), cov
    assert w["coverage"]["permute-instruction"]["instructions_changed"] == 260
    assert w["coverage"]["cross-source-instruction"]["instructions_changed"] == 526


@SKIP
def test_the_committed_log_and_the_committed_json_are_one_run():
    """The log is what a reader reads and the JSON is what a test reads: two files, one
    measurement (§9.37). The command line on the log's first line is the receipt that the
    flags published in the docstring regenerate the artifact, including the `--tokenizer`
    this box needs."""
    w = json.loads(WITNESS.read_text())
    log = WITNESS_LOG.read_text()
    lines = log.splitlines()
    # Harness prints "$ uv run python bench/diag_question_ablation.py ..." followed by the
    # run header (last_step/metadata), so the first line contains the command.
    assert any("$ uv run python bench/diag_question_ablation.py" in ln for ln in lines[:5]), \
        f"log starts: {lines[:3]}"
    assert "--tokenizer" in lines[0] or any("--tokenizer" in ln for ln in lines[:4])
    assert "EXIT=0" in lines[-1], lines[-3:]
    # The harness prints "$ ..." then "run: last_step ...", so second = lines[1].
    second = lines[1]
    assert "last_step 3599" in second, second
    for arm, level in w["macro"].items():
        line = next(ln for ln in log.splitlines() if ln.startswith(arm))
        assert f"macro {level:.4f} (floor {w['macro_floor']:.4f})" in line, (arm, line)
    assert f"guard green: the as-scored arm reproduces {w['guard']['keys_compared']} " \
        "committed keys (0 single-row tie flips)" in log
    for arm, cov in w["coverage"].items():
        line = next(ln for ln in log.splitlines() if ln.startswith(arm))
        assert f"rewrote {cov['instructions_changed']:>4d}/{cov['question_slots']}" in line, \
            (arm, line)

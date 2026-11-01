"""The row-level outcomes dump, gated: a table of predictions that reproduces the published cells.

`bench/row_outcomes.py` exists because every committed witness in this repository stops at a
mean. `runs/v1b_kaggle_3600b.metrics.json` is `{group#sig/question: accuracy}` and
`runs/v1b_kaggle_3600b.report.json` collapses that to 16 numbers, and a mean has no resample
left in it — so no interval, no paired test and no McNemar could be computed from either. This
harness manufactures the missing object: one record per row per question, with the row's
`_meta` kept so a competitor's identical pass can be joined to it.

Three claims, and each is a code path rather than a comment:

* **The mirror loader and `load_split` see the same rows in the same order.** `_meta` is
  discarded by `myna.real_data.load_split`, so the identity table is rebuilt by calling the
  same two functions the loader calls (`parse_questions` for the skip rule, `_signature` for
  the group key). It is load-bearing in the worst way: `outcomes()` pairs `ids[wf][j]` with the
  j-th prediction, so a row skipped by one reading and not the other shifts every later row
  onto someone else's predictions and still prints 16 plausible cells. The fixture carries a
  row one of the two must skip, in the middle of a group, and a test names its index.
* **The chunk contract is checked, not assumed.** `score()` appends one tuple of parallel lists
  per 64-row chunk, in the order the loop took them. `outcomes()` re-derives the sizes from
  `range(0, len(data), CHUNK)` — the same expression, not a copy of the number — and refuses
  when they disagree, so a future width change fails here instead of mis-aligning 1,440
  predictions. Tested by handing `outcomes()` a wrongly-chunked `preds` and reading the message.
* **The dump prices the published table.** `guard_against_report()` compares whole `roll_up`
  rows — accuracy, size, both floors, the type, the set count — under the ablation harness's own
  `FLOAT_TOL` and one-row slack, and a failure means nothing is written. A dump that is a
  *different* measurement than the report is worse than no dump, because it looks like the report.

The end-to-end test runs `main()` against a real checkpoint and a report built by
`myna.train.evaluate()` + `myna.report` — two code paths this file does not share — and the
fixture is a random init rather than a fitted one: nothing here claims the model *reads*
anything, only that the arithmetic of the dump matches the arithmetic of the report, and a
model that must be fitted would make that claim slower to check, not stronger.

`fx` is imported by `tests/test_significance.py`, which reads this file's output as its input.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import bench.row_outcomes as RO  # noqa: E402
from myna.model import MynaConfig, MynaModel  # noqa: E402
from myna.real_data import load_suite, suite_texts  # noqa: E402
from myna.report import (accuracy_cells, cell_stats, g1_verdict, macro,  # noqa: E402
                         roll_up, strata)
from myna.tokenizer import Tokenizer, train_tokenizer  # noqa: E402
from myna.train import evaluate  # noqa: E402

# --------------------------------------------------------------------------- the fixture suite
#
# Thirty-one rows, five sources, ten question slots, eight (source, question) cells — seven of
# them at or over `MIN_ROWS` and `trec/answer_type` below it. The geometry is the pilot's,
# shrunk: `agnews` asks three questions of every row, which is what makes five `agnews/*` cells
# share one sample rather than being five samples (the reason the resample unit is the row);
# `yelp` asks two; `boolq` appears as two question-sets of three and two rows, so a cell can
# have `sets` > 1 and `accuracy_cells` has more than one mean to weight; `banking77` is one
# five-way cell with a perfectly flat label histogram (each of five labels twice), which is the
# shape whose majority floor moves the most when rows are resampled; and `trec`'s two rows are
# the below-floor cell that `--min-rows` drops from the table, the macro and both floors
# together.
#
# `_meta.id` is unique on all 31 scored rows. `_meta.row` is *not*: three rows carry a row number
# belonging to another row, which is the collision the real split has 316 times over and the
# reason the cluster key is `id`. The unparseable row (`bad`) is a choice label outside its
# criteria — `load_split` skips it, and the mirror must skip it too.

MIN_ROWS = 3
G1_TARGET = 0.55
G1_MARGIN = 0.05
TEMPERATURE = 1.0
BAD_ROW_INDEX = 19  # inside trec's run of rows on disk, between boolq and trec's two scored rows

TOPIC_CRIT = {"world": "diplomacy and war", "sport": "matches and teams",
              "tech": "gadgets and software", "biz": "markets and firms"}
INTENT_CRIT = {"transfer": "move money", "balance": "ask the balance", "card": "about a card",
               "pin": "change a pin", "rate": "the exchange rate"}
ANSWER_TYPE_CRIT = {"person": "who", "place": "where", "num": "how many"}
STARS_LEGEND = ["1 star: would not return", "3 stars: it was fine", "5 stars: excellent"]

TOPIC_INSTR = "Which desk files this story?"
WORLD_INSTR = "Is this about the world desk?"
SPORTS_INSTR = "Is this about the sports desk?"
RECOMMEND_INSTR = "Would this reviewer come back?"
STARS_INSTR = "How many stars did this reviewer give?"
ANSWER_INSTR_A = "Did the committee approve the budget?"
ANSWER_INSTR_B = "Was the spending plan adopted?"
INTENT_INSTR = "Which intent does this message ask for?"
TYPE_INSTR = "What kind of thing does this question ask about?"

AGNEWS_STATES = [
    ("the chancellor opened the budget talks with the neighbours", "world", True, False),
    ("the substitute striker won the derby late on", "sport", False, True),
    ("the envoys returned to the embassy once the truce held", "world", True, False),
    ("the home side lifted the cup after extra time", "sport", False, True),
    ("the handset ships with a larger battery this quarter", "tech", False, False),
    ("the treasury reported a smaller deficit than forecast", "biz", False, False),
    ("a joint communique ended the two week standoff at dawn", "world", True, False),
    ("the away side took the trophy on penalties", "sport", False, True),
]
YELP_STATES = [
    ("the broth was worth every cent and the queue was short", True, 2),
    ("the waiter never returned and the plate went cold", False, 0),
    ("the tasting menu justified the price twice over", True, 2),
    ("i would book the same table again without thinking", True, 2),
    ("the coffee was fine but the bill was wrong", False, 1),
    ("the kitchen sent a replacement unprompted", True, 2),
]
BOOLQ_A = [("the committee shelved the budget after one reading", False),
           ("the board passed the budget after a recount", True),
           ("the panel threw out the budget before lunch", False)]
BOOLQ_B = [("the chair vetoed the budget on procedural grounds", False),
           ("the council carried the budget with two abstentions", True)]
BANKING_STATES = [
    ("how much do i have left in the account", "balance"),
    ("move fifty pounds to the savings", "transfer"),
    ("what is the cost of a coffee in tokyo", "rate"),
    ("i lost my card and need a new one", "card"),
    ("set my pin to something i can remember", "pin"),
    ("what is the current balance on checking", "balance"),
    ("send the rent to my landlord", "transfer"),
    ("how many euros to the dollar today", "rate"),
    ("block the stolen card immediately", "card"),
    ("can i choose my own four digit pin", "pin"),
]
TREC_STATES = [("who wrote the wealth of nations", "person"),
               ("how many hearts does an octopus have", "num")]


def _meta(i, source, row_no=None):
    return {"id": f"r{i:03d}", "source": source, "row": (i if row_no is None else row_no),
            "row_sha256": f"sha{i:02d}", "text_sha256": f"t{i:02d}",
            "group_id": f"g{source}", "variant": "v1", "repo": "decision-v2-pilot",
            "revision": "0000000", "split": "test"}


def suite_rows():
    """The scored rows, in file order, plus the row both readings must skip.

    Three `_meta.row` values are re-used by a different row (see the block at the end): same
    integer, different row, which is exactly the shape that would merge two rows of a resample.
    The skipped row shares its question JSON — and therefore its group key — with trec's two
    scored rows, because `_signature` drops the label: it is inside the group's span on disk and
    belongs to none of it in the loader.
    """
    out, i = [], 0

    def add(state, source, questions):
        nonlocal i
        out.append({"state": state, "questions": questions, "_meta": _meta(i, source)})
        i += 1

    for state, topic, world, sports in AGNEWS_STATES:
        add(state, "agnews", {
            "topic": {"type": "choice", "instructions": TOPIC_INSTR, "criteria": TOPIC_CRIT,
                      "label": topic},
            "is_world": {"type": "noul", "instructions": WORLD_INSTR, "criteria": None,
                         "label": world},
            "is_sports": {"type": "noul", "instructions": SPORTS_INSTR, "criteria": None,
                          "label": sports}})
    for state, rec, stars in YELP_STATES:
        add(state, "yelp", {
            "recommend": {"type": "noul", "instructions": RECOMMEND_INSTR, "criteria": None,
                          "label": rec},
            "stars": {"type": "score", "instructions": STARS_INSTR, "criteria": STARS_LEGEND,
                      "label": stars}})
    for j, (state, lab) in enumerate(BOOLQ_A):
        add(state, "boolq", {"answer": {"type": "noul", "instructions": ANSWER_INSTR_A,
                                        "criteria": None, "label": lab}})
    for state, lab in BOOLQ_B:
        add(state, "boolq", {"answer": {"type": "noul", "instructions": ANSWER_INSTR_B,
                                       "criteria": None, "label": lab}})
    # the row both readings must skip: a choice label its criteria do not offer
    out.append({"state": "the envoy signed the accord on the eighth of june",
                "questions": {"answer_type": {"type": "choice", "instructions": TYPE_INSTR,
                                              "criteria": ANSWER_TYPE_CRIT, "label": "date"}},
                "_meta": _meta(99, "trec")})
    for state, lab in TREC_STATES:
        add(state, "trec", {"answer_type": {"type": "choice", "instructions": TYPE_INSTR,
                                            "criteria": ANSWER_TYPE_CRIT, "label": lab}})
    for state, lab in BANKING_STATES:
        add(state, "banking77", {"intent": {"type": "choice", "instructions": INTENT_INSTR,
                                            "criteria": INTENT_CRIT, "label": lab}})
    # three deliberate `_meta.row` / `_meta.row_sha256` collisions, so `cluster_key_check` in the
    # witness shows both counts below `distinct_ids` and the choice of cluster key is a
    # measurement rather than a story. The pilot has 316 and 219 of these; the values reused here
    # are other rows' numbers, so they are *different rows sharing an identity*, which is the
    # only kind of collision that would silently merge two rows of one resample.
    for dst, src in ((1, 9), (2, 22), (15, 14)):
        for field in ("row", "row_sha256"):
            out[dst]["_meta"][field] = out[src]["_meta"][field]
    return out


N_ROWS = 31       # 8 agnews + 6 yelp + 5 boolq + 10 banking77 + 2 trec
N_QUESTIONS = 53  # 8*3 + 6*2 + 5 + 10 + 2; the skipped row asks nothing
CELLS_KEPT = 7
CELLS_ALL = 8


def build_report(model, hf, groups):
    """`myna.report`'s own aggregation of `evaluate()`'s own numbers — the ruler `main()` must
    reproduce, written by code this harness does not share a single line of."""
    accs = evaluate(model, hf, groups, "cpu", TEMPERATURE)
    cells, unmatched = accuracy_cells(accs, groups)
    assert unmatched == []
    kept = roll_up(cell_stats(groups), cells, MIN_ROWS)
    scored = [r for r in kept if r["acc"] is not None]
    return {"split": "test", "suite": "fixture", "file": "test.jsonl", "min_rows": MIN_ROWS,
            "cells": kept, "unmatched_sets": unmatched, "strata": strata(groups),
            "macro": {"acc": macro(scored, "acc"), "majority": macro(scored, "majority"),
                      "uniform": macro(scored, "uniform"), "cells_scored": len(scored),
                      "cells_kept": len(kept)},
            "g1": g1_verdict(macro(scored, "acc"), macro(scored, "majority"),
                             target=G1_TARGET, margin=G1_MARGIN)}


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    """A suite on disk, a checkpoint, the report built from the *trainer's* `evaluate()`, and
    the dump `main()` writes from its own re-read of `model.pt`."""
    root = tmp_path_factory.mktemp("sig-suite")
    ckpt = tmp_path_factory.mktemp("sig-ckpt")
    rows = suite_rows()
    (root / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                     encoding="utf-8")
    suites = load_suite(str(root))
    groups = suites["test"]
    tok = train_tokenizer(suite_texts(suites["test"]), vocab_size=64)
    tok.save(str(ckpt / "tokenizer.json"))
    hf = Tokenizer.from_file(str(ckpt / "tokenizer.json"))
    torch.manual_seed(7)
    model = MynaModel(MynaConfig(vocab=tok.get_vocab_size(), d_model=16, n_layers=1,
                                 n_heads=2, d_k=8, d_v=8, d_ff=16, d_ptr=8))
    model.to("cpu")
    torch.save({"cfg": vars(model.cfg), "state_dict": model.state_dict(),
                "temperature": TEMPERATURE}, ckpt / "model.pt")
    report = build_report(model, hf, groups)
    report_path = root / "fixture.report.json"
    report_path.write_text(json.dumps(report, indent=1) + "\n")
    dump = tmp_path_factory.mktemp("sig-dump") / "myna.rowoutcomes.json"
    rc = RO.main(["--run-dir", str(ckpt), "--suite", str(root), "--report", str(report_path),
                  "--min-rows", str(MIN_ROWS), "--out", str(dump)])
    assert rc == 0, "the fixture dump did not reproduce the fixture report"
    return SimpleNamespace(root=root, ckpt=ckpt, groups=groups, rows=rows,
                           report=report, report_path=report_path, dump_path=dump,
                           dump=json.loads(dump.read_text()), model=model, hf=hf)


# ------------------------------------------------------------------------ the mirror loader


def test_the_mirror_and_the_loader_agree_on_every_group(fx):
    """Same keys, same row count per key. This is the precondition for `ids[wf][j]` to be the
    j-th example, and the skip rule is what makes it non-trivial: the bad row is inside the
    `trec` group's file span, not at the end."""
    ids, n_rows_file, n_skipped = RO.row_identities(fx.root / "test.jsonl")
    assert n_rows_file == len(fx.rows) == 32
    assert n_skipped == 1, "the unparseable row was not skipped — so nothing here is tested"
    assert set(ids) == set(fx.groups)
    assert RO.check_identities(fx.groups, ids) == []
    assert sum(len(v) for v in ids.values()) == N_ROWS == 31


def test_a_row_that_one_reading_skips_shifts_every_later_prediction(fx):
    """The shape of the failure the mirror exists to avoid: drop one row from the identity table
    and `check_identities` must name the group and both counts, because every row after it in
    that group would otherwise be printed with its neighbour's outcome."""
    ids, _n, _s = RO.row_identities(fx.root / "test.jsonl")
    wf = next(k for k, v in ids.items() if k.startswith("agnews"))
    victim = ids[wf][2]
    ids[wf] = ids[wf][:2] + ids[wf][3:]
    bad = RO.check_identities(fx.groups, ids)
    assert len(bad) == 1 and wf in bad[0], bad
    assert "the loader made 8 examples and the mirror found 7" in bad[0], bad[0]
    # a group neither reading has in common is named by both sides, not asserted away
    ids2, _, _ = RO.row_identities(fx.root / "test.jsonl")
    ids2["agnews#deadbeef"] = [victim]
    bad = RO.check_identities(fx.groups, ids2)
    assert any("which groups exist" in line for line in bad), bad


def test_the_unparseable_row_is_in_the_file_and_in_neither_group(fx):
    """The skipped row carries `_meta.id` `r099` — a number no scored row has, so its absence
    here says the skip removed one row rather than shifting the ids. `rows_in_file` counts it and
    `rows_skipped` names it, which is the pair a reader uses to check the 32 → 31 arithmetic."""
    ids, _n, _s = RO.row_identities(fx.root / "test.jsonl")
    assert fx.rows[BAD_ROW_INDEX]["_meta"]["id"] == "r099"
    dumped = {r["meta"]["id"] for r in fx.dump["rows"]}
    assert "r099" not in dumped
    assert "r099" not in {m["id"] for v in ids.values() for m in v}
    assert len(dumped) == N_ROWS
    assert fx.dump["rows_in_file"] == 32 and fx.dump["rows_skipped"] == 1


# ---------------------------------------------------------------------------------- outcomes


def test_a_prediction_chunk_the_width_changed_is_refused_by_number():
    """`outcomes()` re-derives `sizes` from the same `range()` the scoring loop uses. Give it
    chunks that came from a different width and it must refuse *with both lists*, because a
    silent re-flatten would pair row 0 with row 64's prediction and every cell would still be
    the right size."""
    from myna.real_data import Question
    qs = [Question("topic", "choice", "which desk?", ["a", "b", "c"])]
    data = [SimpleNamespace(gold=g) for g in range(5)]
    groups = {"agnews#aaaa": (qs, data)}
    ids = {"agnews#aaaa": [{"id": f"r{i}", "source": "agnews"} for i in range(5)]}
    good = {"agnews#aaaa/topic": [(list(range(5)), list(range(5)), [0.5] * 5)]}
    rows, hits, n, tot = RO.outcomes(groups, good, ids)
    assert tot == 5 and n["agnews#aaaa/topic"] == 5
    assert hits["agnews#aaaa/topic"] == 5
    # one wrong row, inside the chunk, so a hit count is read as an arithmetic fact
    misses = {"agnews#aaaa/topic": [([0, 1, 2, 0, 4], list(range(5)), [0.5] * 5)]}
    _r, hits, _n, _t = RO.outcomes(groups, misses, ids)
    assert hits["agnews#aaaa/topic"] == 4
    # now a 2+3 split: it sums to the same 5 rows and is the wrong contract
    wrong = {"agnews#aaaa/topic": [([0, 1], [0, 1], [0.5, 0.5]),
                                   ([2, 3, 4], [2, 3, 4], [0.5] * 3)]}
    with pytest.raises(SystemExit) as e:
        RO.outcomes(groups, wrong, ids)
    assert "returned chunks [2, 3] but this group of 5 rows chunks as [5] at width 64" \
        in str(e.value), e.value


def test_the_flattened_order_is_the_loops_order_not_the_sorted_one():
    """`data[i:i+64]` walks the group in loader order, and the identity list is in the same
    order. Give every row a gold nothing else has — its own index — so any re-flatten, sort or
    column-wise merge puts someone else's number on someone else's identity, visibly."""
    from myna.real_data import Question
    qs = [Question("topic", "choice", "which desk?", ["a", "b"])]
    n_rows = 130
    data = [SimpleNamespace(gold=i) for i in range(n_rows)]
    groups = {"agnews#bbbb": (qs, data)}
    ids = {"agnews#bbbb": [{"id": f"r{i:03d}", "source": "agnews"} for i in range(n_rows)]}
    preds_col = list(range(n_rows))
    preds_col[5] = 999
    chunks = [(preds_col[a:b], list(range(a, b)), [0.9] * (b - a))
              for a, b in ((0, 64), (64, 128), (128, 130))]
    preds = {"agnews#bbbb/topic": chunks}
    rows, hits, sizes, _tot = RO.outcomes(groups, preds, ids)
    assert [len(c[0]) for c in chunks] == [64, 64, 2]
    assert sizes["agnews#bbbb/topic"] == 130
    assert hits["agnews#bbbb/topic"] == 129
    assert [r["questions"]["topic"]["gold"] for r in rows] == list(range(n_rows))
    assert [r["meta"]["id"] for r in rows] == [f"r{i:03d}" for i in range(n_rows)]
    assert rows[5]["questions"]["topic"] == {"type": "choice", "pred": 999, "gold": 5,
                                            "k": 2, "ok": 0, "conf": 0.9}
    assert rows[128]["questions"]["topic"]["ok"] == 1 and rows[129]["idx"] == 129
    assert rows[0]["questions"]["topic"]["conf"] == 0.9
    assert rows[64]["idx"] == 64 and rows[64]["group"] == "agnews#bbbb"


def test_every_dumped_question_carries_what_a_paired_test_needs(fx):
    """`k` (the option count, which the uniform floor is priced from), `gold`, `pred`, `ok` and
    a confidence — and `ok` must be `pred == gold` and nothing else, because `significance.py`
    sums `ok` and never re-derives it."""
    seen = 0
    for r in fx.dump["rows"]:
        for name, q in r["questions"].items():
            seen += 1
            assert set(q) == {"type", "pred", "gold", "k", "ok", "conf"}, (r, name)
            assert q["ok"] == int(q["pred"] == q["gold"])
            assert isinstance(q["k"], int) and q["k"] >= 2
            assert q["type"] in ("noul", "choice", "score")
    assert seen == fx.dump["n_questions"] == N_QUESTIONS == 53


def test_the_meta_block_is_exactly_the_declared_fields(fx):
    """The witness is read by `significance.py`, by the registry and by a paper table, and it
    carries no state text on purpose — `_meta` whole would double the file for fields nobody
    joins on. So the field list is a contract, and it is this file's own constant."""
    assert set(fx.dump["rows"][0]["meta"]) == set(RO.META_FIELDS)
    assert all(set(r["meta"]) == set(RO.META_FIELDS) for r in fx.dump["rows"])


def test_the_cluster_key_is_the_one_field_that_is_unique(fx):
    """`_meta.id` is unique over 31 rows; `_meta.row` and `_meta.row_sha256` each repeat three
    times here, and in the real split they collide 316 and 219 times. The witness prints all
    three distinct-counts beside each other so a reader can check the choice of key instead of
    trusting the docstring."""
    chk = fx.dump["cluster_key_check"]
    assert chk == {"distinct_ids": 31, "distinct_row": 28, "distinct_row_sha256": 28}, chk
    assert fx.dump["cluster_key"] == "meta.id"
    assert max(r["meta"]["row"] for r in fx.dump["rows"]) == 30
    for field in ("row", "row_sha256"):
        seen = [r["meta"][field] for r in fx.dump["rows"]]
        assert len({v for v in seen if seen.count(v) > 1}) == 3, field


# --------------------------------------------------------------------- the guard against the report


def _cell_rows(source="agnews", question="topic", n=5, acc=0.6, majority=0.4,
               uniform=0.25, qtype="choice", sets=3):
    return [{"source": source, "question": question, "n": n, "acc": acc,
             "majority": majority, "uniform": uniform, "type": qtype, "sets": sets}]


def _recompute(hits, n=5, **kw):
    """A `roll_up` row with the accuracy rebuilt from hits, which is what `main()` feeds in. An
    explicit `acc=` in `kw` overrides, so a test can move one field and hold the rest."""
    row = {"source": "agnews", "question": "topic", "n": n, "acc": hits / n,
           "majority": 0.4, "uniform": 0.25, "type": "choice", "sets": 3}
    return [dict(row, **kw)]


def test_an_exact_match_returns_no_bad_and_no_flips():
    bad, flips, compared = RO.guard_against_report(_recompute(3, 5), _cell_rows(acc=0.6))
    assert (bad, flips, compared) == ([], 0, 1)


def test_a_half_point_move_on_a_five_row_cell_is_a_different_measurement():
    bad, flips, _c = RO.guard_against_report(_recompute(0, 5), _cell_rows(acc=0.6))
    assert len(bad) == 1 and "these rows give 0.000000, the committed report gives 0.600000" \
        in bad[0], bad
    assert "0.200000 = one row" in bad[0]
    assert flips == 0


def test_a_one_row_cell_flip_is_rounding_and_is_counted_as_such():
    """A 1-row cell has only 0.0 and 1.0 available, so a whole-cell delta is one row and is the
    same thing the ablation harness grants a CPU-against-T4 tie. Counted, not absorbed."""
    bad, flips, _c = RO.guard_against_report(_recompute(0, 1, acc=0.0),
                                             _cell_rows(n=1, acc=1.0))
    assert bad == [] and flips == 1, (bad, flips)


def test_the_tie_slack_stops_at_one_and_a_half_rows():
    """`1.5 * one_row` is the boundary between "a tie broke differently" and "this is not the run
    that produced the report", and it is measured from both sides."""
    n = 100
    bad, flips, _ = RO.guard_against_report(_recompute(49, n), _cell_rows(n=n, acc=0.50))
    assert (bad, flips) == ([], 1)
    bad, flips, _ = RO.guard_against_report(_recompute(47, n), _cell_rows(n=n, acc=0.50))
    assert len(bad) == 1 and flips == 0, (bad, flips)


def test_every_field_of_the_row_is_compared_not_just_the_accuracy():
    """A dump whose cells have the right accuracy and the wrong `n`, floor, type or set count is
    describing a different split, and the accuracy alone would not say so."""
    for field, value in (("n", 6), ("majority", 0.5), ("uniform", 0.3),
                         ("type", "noul"), ("sets", 2)):
        # `acc` is rebuilt from hits/n, so moving the denominator needs the accuracy held
        # fixed by hand — otherwise this case would test the acc branch as well.
        moved = {field: value}
        if field == "n":
            moved["acc"] = 0.6
        bad, _f, _c = RO.guard_against_report(_recompute(3, **moved), _cell_rows(acc=0.6))
        assert len(bad) == 1 and f"{field} is {value!r} here and" in bad[0], (field, bad)


def test_a_cell_only_one_side_has_is_named_on_both_directions():
    bad, _f, compared = RO.guard_against_report(
        _cell_rows() + _cell_rows(source="yelp", question="stars", qtype="score"),
        _cell_rows())
    assert compared == 1 and len(bad) == 1 and "1 only here ['yelp/stars']" in bad[0], bad
    bad, _f, compared = RO.guard_against_report(
        _cell_rows(), _cell_rows() + _cell_rows(source="boolq", question="answer",
                                                qtype="noul"))
    assert compared == 1 and "1 only in the report ['boolq/answer']" in bad[0], bad


# ------------------------------------------------------------------- the harness end to end


def test_the_dump_prices_the_report_it_was_asked_to_reproduce(fx):
    """Two implementations, one number: the report came from `evaluate()`, the dump came from
    `score()` reading `model.pt` off disk, and the guard is the equality between them — cell by
    cell, on every field, plus the macro to 1e-9."""
    d = fx.dump
    assert d["guard"] == {"report": str(fx.report_path), "report_macro": d["guard"]["report_macro"],
                          "cells_compared": CELLS_KEPT, "tie_flips": 0,
                          "float_tol": RO.FLOAT_TOL, "row_slack_keys": RO.ROW_SLACK_KEYS,
                          "ok": True}
    assert d["guard"]["report_macro"] == fx.report["macro"]["acc"]
    assert d["macro"] == fx.report["macro"], (d["macro"], fx.report["macro"])
    assert d["n_rows"] == N_ROWS and d["n_sets"] == len(fx.groups) == 6
    by_label = {(c["source"], c["question"]): c for c in d["cells"]}
    for c in fx.report["cells"]:
        assert by_label[(c["source"], c["question"])]["acc"] == pytest.approx(c["acc"], abs=1e-12)
        assert by_label[(c["source"], c["question"])]["hits"] == pytest.approx(
            round(c["acc"] * c["n"]), 6)
    # hits × cells is the geometry a reader checks the macro against, and the per-cell `hits` is
    # counted from the dumped rows, not from score()'s sums
    for key, c in by_label.items():
        assert c["hits"] == sum(q["ok"] for r in d["rows"] if r["meta"]["source"] == key[0]
                                for name, q in r["questions"].items() if name == key[1])


def test_the_dropped_cell_leaves_the_table_the_macro_and_both_floors_together(fx):
    """`trec/answer_type` holds two rows and `--min-rows 3` drops it. The published macro is
    over the kept cells, so the floors move with it — and a harness that carried the unfiltered
    floor into the kept table would print a margin that is not G1's margin."""
    assert fx.dump["min_rows"] == MIN_ROWS
    names = {(c["source"], c["question"]) for c in fx.dump["cells"]}
    assert ("trec", "answer_type") not in names and len(names) == CELLS_KEPT
    assert fx.dump["macro"]["cells_kept"] == CELLS_KEPT
    assert fx.dump["macro"]["cells_scored"] == CELLS_KEPT
    # the same roll-up, at the report's own scope and at every cell the split can make
    accs = evaluate(fx.model, fx.hf, fx.groups, "cpu", TEMPERATURE)
    cells, _un = accuracy_cells(accs, fx.groups)
    everything = roll_up(cell_stats(fx.groups), cells, 1)
    kept = [r for r in everything if r["n"] >= MIN_ROWS]
    assert len(everything) == CELLS_ALL and len(kept) == CELLS_KEPT
    assert ("trec", "answer_type") in {(r["source"], r["question"]) for r in everything}
    for field in ("acc", "majority", "uniform"):
        assert fx.dump["macro"][field] == pytest.approx(
            macro(kept, field), abs=1e-12), (field, fx.dump["macro"])
        unfiltered = macro([r for r in everything if r["acc"] is not None], field)
        assert fx.dump["macro"][field] != pytest.approx(unfiltered, abs=1e-9), \
            f"{field} was carried over from the unfiltered scope"


def test_the_printed_line_reports_the_row_identity_it_found(fx, capsys):
    """The line a reader uses to decide the cluster key was safe is computed from the dump, so
    the same three counts have to be in the text and in the JSON."""
    capsys.readouterr()
    rc = RO.main(["--run-dir", str(fx.ckpt), "--suite", str(fx.root),
                  "--report", str(fx.report_path), "--min-rows", str(MIN_ROWS),
                  "--out", str(fx.root / "again.json")])
    out = capsys.readouterr().out
    assert rc == 0
    assert ("row identity: 31 rows → 31 distinct _meta.id (vs 28 distinct _meta.row and "
            "28 distinct _meta.row_sha256") in out, out
    assert f"guard: {CELLS_KEPT} cells reproduced fixture.report.json exactly (tie flips 0" in out
    assert "macro" in out and f"{len(fx.groups)} question-sets" in out


def test_a_report_the_dump_disagrees_with_is_refused_and_nothing_is_written(fx, tmp_path,
                                                                            capsys):
    """The guard is not a banner: a doctored cell accuracy has to stop the file, name both
    numbers, and leave no artifact behind — a dump that looks like the published table and is
    not it is the failure this harness exists to make impossible."""
    doc = json.loads(fx.report_path.read_text())
    victim = next(c for c in doc["cells"] if c["n"] >= 5)
    victim["acc"] += 0.3
    doctored = tmp_path / "doctored.json"
    doctored.write_text(json.dumps(doc))
    o = tmp_path / "never.json"
    assert RO.main(["--run-dir", str(fx.ckpt), "--suite", str(fx.root),
                    "--report", str(doctored), "--min-rows", str(MIN_ROWS),
                    "--out", str(o)]) == 3
    err = capsys.readouterr().err
    assert f"these rows give {victim['acc'] - 0.3:.6f}, the committed report gives " in err, err
    assert "nothing was written" in err
    assert not o.exists()


def test_a_doctored_floor_is_refused_even_though_the_accuracy_matches(fx, tmp_path, capsys):
    """The accuracy is the number G1 reads, and the floor is the number G1 reads *against*.
    Nudging only the floor leaves `acc` untouched, so a guard that compared accuracies alone
    would write a dump whose margin is someone else's."""
    doc = json.loads(fx.report_path.read_text())
    for c in doc["cells"]:
        c["majority"] += 0.01
    doctored = tmp_path / "floors.json"
    doctored.write_text(json.dumps(doc))
    o = tmp_path / "never.json"
    assert RO.main(["--run-dir", str(fx.ckpt), "--suite", str(fx.root),
                    "--report", str(doctored), "--min-rows", str(MIN_ROWS),
                    "--out", str(o)]) == 3
    err = capsys.readouterr().err
    assert "majority is" in err and "in the report" in err, err
    assert not o.exists()


def test_a_macro_that_only_moves_with_the_guard_would_still_be_caught(fx, tmp_path, capsys):
    """The macro check is separate from the cell check on purpose: a report whose 16 cells are
    right and whose headline mean is wrong is exactly the artifact a reader quotes."""
    doc = json.loads(fx.report_path.read_text())
    doc["macro"]["acc"] += 1e-6
    doctored = tmp_path / "macro.json"
    doctored.write_text(json.dumps(doc))
    o = tmp_path / "never.json"
    assert RO.main(["--run-dir", str(fx.ckpt), "--suite", str(fx.root),
                    "--report", str(doctored), "--min-rows", str(MIN_ROWS),
                    "--out", str(o)]) == 3
    err = capsys.readouterr().err
    assert "macro over these rows is" in err, err


def test_a_missing_split_or_report_names_the_file_it_wanted(fx, tmp_path, capsys):
    o = tmp_path / "x.json"
    assert RO.main(["--run-dir", str(fx.ckpt), "--suite", str(tmp_path / "nodir"),
                    "--report", str(fx.report_path), "--out", str(o)]) == 2
    assert "has no test.jsonl" in capsys.readouterr().err
    assert RO.main(["--run-dir", str(fx.ckpt), "--suite", str(fx.root),
                    "--report", str(tmp_path / "nope.json"), "--out", str(o)]) == 2
    assert "is not on disk" in capsys.readouterr().err
    assert not o.exists()


def test_the_witness_binds_the_inputs_it_scored(fx):
    """§9.37: the digests, the run dir, the report it was guarded against, and the flags that
    produced it are all fields, and `cmd` is this call's argv rather than the process's."""
    d = fx.dump
    assert d["engine"] == "myna" and d["split"] == "test" and d["file"] == "test.jsonl"
    assert d["run_dir"] == str(fx.ckpt) and d["suite"] == str(fx.root)
    assert d["model"]["temperature"] == TEMPERATURE
    assert (d["model"]["model_sha256"], d["model"]["tokenizer_sha256"]) == (
        hashlib.sha256((fx.ckpt / "model.pt").read_bytes()).hexdigest(),
        hashlib.sha256((fx.ckpt / "tokenizer.json").read_bytes()).hexdigest())
    assert d["cmd"].startswith("uv run python bench/row_outcomes.py")
    for flag in (f"--run-dir {fx.ckpt}", f"--suite {fx.root}",
                 f"--report {fx.report_path}", f"--min-rows {MIN_ROWS}",
                 f"--out {fx.dump_path}"):
        assert flag in d["cmd"], (flag, d["cmd"])
    assert d["cmd"].endswith(f"--out {fx.dump_path}")
    assert d["guard"]["report"] == str(fx.report_path)
    assert d["strata"] == fx.report["strata"]
    assert d["unmatched_sets"] == []


def test_the_dump_is_reproducible_byte_for_byte(fx):
    """A witness two runs cannot produce alike is a transcript, not a measurement. Same
    checkpoint, same suite, same flags → same bytes, and the one field allowed to differ is the
    `cmd` that names the file being written."""
    a, b = fx.root / "a.json", fx.root / "b.json"
    for p in (a, b):
        assert RO.main(["--run-dir", str(fx.ckpt), "--suite", str(fx.root),
                        "--report", str(fx.report_path), "--min-rows", str(MIN_ROWS),
                        "--out", str(p)]) == 0
    ta, tb = json.loads(a.read_text()), json.loads(b.read_text())
    assert ta["cmd"] != tb["cmd"], "the witness must name the file it actually wrote"
    ta.pop("cmd")
    tb.pop("cmd")
    assert ta == tb
    # the digests are the part of the reproduction a reader cannot re-derive from the file
    assert ta["model"] == tb["model"] and len(ta["rows"]) == len(tb["rows"]) == N_ROWS

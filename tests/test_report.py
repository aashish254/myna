"""The stratified reporting harness (SPEC §5 P3 3g).

A table of accuracies is only evidence if the numbers beside it are derived from
the same rows. Three claims are pinned here:

* the **floors** — `majority` from that cell's own label histogram, `uniform`
  row-weighted over the option counts of the sets that ask it (they differ: kev
  randomizes distractors per row, so agnews/topic appears with 4 and with 5
  options);
* the **weighting** — `evaluate()` reports one accuracy per question-*set*, so an
  unweighted cell mean lets a 2-row set outvote a 10-row one;
* the **strata** — shared-instruction vs per-row-instruction, *derived* from the
  instruction strings. Keying it on the adapter's group signature would call nine
  sources per-row, because the signature includes those randomized descriptions.

Then the whole CLI runs as a subprocess on the real pilot split, and its majority
macro must land on SPEC §4.2's published 0.4331 — a floor computed twice by two
independent pieces of code, agreeing, is a floor you can quote.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from myna.data import Example, Question
from myna.real_data import load_split
from myna.report import (LBL, MIN_CELL_ROWS, accuracy_cells, brier_cells, cell_stats,
                         g1_verdict, laya_cells, macro, main, roll_up, strata, table_text)

REPO = Path(__file__).resolve().parent.parent
PILOT = REPO / "data" / "decision-v2-pilot"

AG = Question("topic", "choice", "What is the topic?", ["world", "sport", "business", "tech"])
AG5 = Question("topic", "choice", "What is the topic?",
               ["world", "sport", "business", "tech", "weather"])
TREC = Question("answer_type", "choice", "What kind of thing is asked for?", ["hum", "loc"])
# boolq's real shape: the row's own question *is* the instruction, so every row is a
# question-set of one. Four shared sets would make it look like agnews, and the
# strata test would be testing nothing.
BOOLQ_ROWS = [(1,), (0,), (1,), (0,), (1,), (0,), (1,), (1,)]
BOOLQ = {f"boolq#{i:04d}": ([Question("answer", "noul", f"Did event {i} happen?",
                                      ["yes", "no"])], [g])
         for i, g in enumerate(BOOLQ_ROWS)}


def _groups(spec):
    """spec: {group_key: (questions, [gold tuple, ...])} -> the adapter's shape."""
    return {k: (qs, [Example(f"state {i}", k, gold) for i, gold in enumerate(golds)])
            for k, (qs, golds) in spec.items()}


SPLIT = _groups({
    "agnews#aaaa": ([AG], [(0,), (0,), (0,), (1,), (1,), (1,), (1,), (2,), (2,), (2,)]),
    "agnews#bbbb": ([AG5], [(0,), (3,)]),
    "trec#0000": ([TREC], [(0,)] * 40),
    **BOOLQ,
})


def test_majority_floor_is_the_cells_own_label_histogram():
    c = cell_stats(SPLIT)
    topic = c[("agnews", "topic")]
    assert topic["n"] == 12 and topic["sets"] == 2
    # labels over the 12 rows: 0 four times, 1 four times, 2 three, 3 one
    assert topic["majority"] == pytest.approx(4 / 12), \
        "the floor is the mode over the pooled cell, not the mode of its largest set"
    assert c[("boolq", "answer")]["n"] == 8
    assert c[("boolq", "answer")]["majority"] == pytest.approx(5 / 8)
    assert topic["type"] == "choice" and c[("boolq", "answer")]["type"] == "noul"
    assert c[("trec", "answer_type")]["majority"] == 1.0, "a pure cell has a floor of 1.0"


def test_uniform_floor_weights_the_option_counts_it_actually_sees():
    c = cell_stats(SPLIT)
    # 10 rows asked over 4 options, 2 rows over 5 -> weighted mean of 1/4 and 1/5.
    assert c[("agnews", "topic")]["uniform"] == pytest.approx((10 / 4 + 2 / 5) / 12)
    assert c[("agnews", "topic")]["options"] == {4: 10, 5: 2}
    assert c[("boolq", "answer")]["uniform"] == pytest.approx(0.5)


def test_the_option_mix_is_reported_not_absorbed():
    c = cell_stats(SPLIT)
    assert len(c[("agnews", "topic")]["options"]) == 2, "the fixture must actually mix"
    assert len(c[("boolq", "answer")]["options"]) == 1


def test_cell_accuracy_is_weighted_by_rows_not_by_sets():
    metrics = {"agnews#aaaa/topic": 1.0, "agnews#bbbb/topic": 0.0, "boolq#0000/answer": 0.5}
    acc, unmatched = accuracy_cells(metrics, SPLIT)
    assert acc[("agnews", "topic")]["acc"] == pytest.approx(10 / 12), \
        "an unweighted mean of the two sets is 0.5 — a 2-row set outvoting a 10-row one"
    assert acc[("agnews", "topic")]["n"] == 12 and acc[("agnews", "topic")]["sets"] == 2
    assert unmatched == [], "all three keys are in the split, so nothing may go unweighted"


def test_sidecar_metric_keys_never_become_accuracy_rows():
    metrics = {"boolq#0000/answer": 0.9, "boolq#0000/answer:brier": 0.01,
               "boolq#0000/answer:ece": 0.02}
    acc, _ = accuracy_cells(metrics, SPLIT)
    assert list(acc) == [("boolq", "answer")]
    assert acc[("boolq", "answer")]["acc"] == pytest.approx(0.9), \
        "the brier sidecar must not join the mean (0.9 and 0.01 would give 0.455)"


def test_a_scored_set_that_is_not_in_the_split_is_reported():
    acc, unmatched = accuracy_cells({"agnews#zzzz/topic": 1.0}, SPLIT)
    assert unmatched == ["agnews#zzzz/topic"], \
        "silently weighting an unknown set at 0 rows hides a split mismatch"


def test_strata_separates_a_repeated_schema_from_a_per_row_question():
    s = strata(SPLIT)
    assert s["boolq"]["class"] == "per-row-instruction"
    assert s["agnews"]["class"] == "shared-instruction"
    assert s["trec"]["class"] == "shared-instruction"
    assert s["boolq"]["distinct_instructions"] == 8 and s["boolq"]["sets"] == 8
    assert s["boolq"]["repeated_instruction_share"] == 0.0
    assert s["agnews"]["repeated_instruction_share"] == 1.0


def test_strata_sees_two_measures_where_the_adapter_sees_one():
    """Instruction repetition and question-set reuse are different facts, and only
    the first one means "groupable". Here both of agnews's sets hold more than one
    row, so its two shares coincide at 1.0; the real pilot split is where they
    part — agnews repeats one instruction across all 116 test rows while 113
    exact-signature sets hold them, because the randomized option descriptions
    split every row off on its own. Pinned against the real data below."""
    s = strata(SPLIT)
    assert s["boolq"]["multi_row_set_share"] == 0.0, "every boolq set is one row deep"
    assert s["boolq"]["repeated_instruction_share"] == 0.0
    assert s["agnews"]["multi_row_set_share"] == 1.0 and s["agnews"]["sets"] == 2
    # Two sets, one instruction: the counts answer different questions, so neither
    # may be reported as the other.
    assert s["agnews"]["distinct_instructions"] == 1 and s["agnews"]["sets"] == 2
    assert s["boolq"]["distinct_instructions"] == s["boolq"]["sets"] == 8
    for source, d in s.items():
        assert 0.0 <= d["repeated_instruction_share"] <= 1.0, f"{source}: {d}"
        assert 0.0 <= d["multi_row_set_share"] <= 1.0, f"{source}: {d}"


def test_strata_share_is_a_fraction_of_slots_not_a_sum_of_rows():
    two = _groups({"x#aaaa": ([AG, BOOLQ["boolq#0000"][0][0]], [(0, 1)] * 5),
                   "x#bbbb": ([AG, BOOLQ["boolq#0000"][0][0]], [(1, 0)] * 5)})
    d = strata(two)["x"]
    assert d["rows"] == 10 and d["slots"] == 20, "two questions per row is two slots"
    assert d["distinct_instructions"] == 2
    assert d["repeated_instruction_share"] == 1.0


def test_min_rows_keeps_cells_and_says_which_it_dropped():
    stats = cell_stats(SPLIT)
    acc, _ = accuracy_cells({"boolq#0000/answer": 1.0}, SPLIT)
    assert {(r["source"], r["question"]) for r in roll_up(stats, acc, keep_min=5)} == \
        {("agnews", "topic"), ("boolq", "answer"), ("trec", "answer_type")}
    kept = {(r["source"], r["question"]) for r in roll_up(stats, acc, keep_min=10)}
    assert kept == {("agnews", "topic"), ("trec", "answer_type")}, "boolq has 8 rows"
    assert roll_up(stats, acc) == roll_up(stats, acc, keep_min=MIN_CELL_ROWS)
    boolq_cell = [r for r in roll_up(stats, acc, keep_min=5) if r["source"] == "boolq"][0]
    assert boolq_cell["n_scored"] == 1 and boolq_cell["n"] == 8, \
        "one scored row of a cell of eight: the two counts must stay distinct"


def test_macro_averages_over_cells_not_over_rows():
    """§4.2's macro is one number per (source, question) cell. Averaging rows
    instead would let trec's 40 hide boolq's 8 — and the published floor would stop
    being the floor the table is read against."""
    stats = cell_stats(SPLIT)
    acc, _ = accuracy_cells({"boolq#0000/answer": 1.0, "trec#0000/answer_type": 0.0}, SPLIT)
    scored = [r for r in roll_up(stats, acc, keep_min=5) if r["acc"] is not None]
    assert [(r["source"], r["n"]) for r in scored] == [("boolq", 8), ("trec", 40)]
    assert macro(scored, "acc") == pytest.approx(0.5)
    assert macro(scored, "acc") != pytest.approx(8 / 48), "that is the row-weighted number"


def test_macro_is_none_when_nothing_was_scored():
    stats = cell_stats(SPLIT)
    rows = roll_up(stats, {}, keep_min=5)
    assert macro(rows, "acc") is None
    assert macro(rows, "majority") is not None, "the split still has floors"


def test_g1_needs_both_the_target_and_the_margin():
    v = g1_verdict(0.72, 0.60)
    assert v["meets_target"] and not v["meets_margin"] and not v["pass"], \
        "0.72 over a 0.60 floor is not the gate; the gate exists to stop that sale"
    ok = g1_verdict(0.70, 0.4331)
    assert ok["pass"] and ok["meets_target"] and ok["meets_margin"]
    below = g1_verdict(0.55, 0.4331)
    assert not below["meets_target"] and not below["meets_margin"] and not below["pass"]
    assert g1_verdict(None, 0.43)["pass"] is False, "no model number is never a pass"
    assert g1_verdict(0.99, None)["pass"] is False, "no floor is never a pass either"


def test_the_table_prints_the_strata_with_their_own_macro():
    stats = cell_stats(SPLIT)
    acc, _ = accuracy_cells({"boolq#0000/answer": 1.0, "agnews#aaaa/topic": 0.5,
                             "agnews#bbbb/topic": 0.0}, SPLIT)
    rows = roll_up(stats, acc, keep_min=5)
    text = table_text(rows, strata(SPLIT), split="test")
    assert "shared-instruction" in text and "per-row-instruction" in text
    ag = [ln for ln in text.splitlines() if ln.startswith("agnews/topic")][0]
    # (0.5 x 10 rows + 0.0 x 2 rows) / 12 = 0.4167 — the weighted cell accuracy
    assert "0.417" in ag, ag
    assert "0.333" in ag, "the majority floor must ride in the same line as the model"
    assert "0.083" in ag, "the last column is model minus MAJORITY, not minus chance"
    assert "trec/answer_type" in text, "a kept cell with no model number still prints floors"
    assert "macro: 1 cell(s)" in text


def test_laya_columns_join_on_the_cell_not_on_the_row_count():
    """laya's witness JSON reports one accuracy per (source, qid) for its own sample
    size; the join is by cell identity, and a 0.0 must still print as a number."""
    stats = cell_stats(SPLIT)
    acc, _ = accuracy_cells({"boolq#0000/answer": 1.0, "agnews#aaaa/topic": 0.5,
                             "agnews#bbbb/topic": 0.0}, SPLIT)
    rows = roll_up(stats, acc, keep_min=5)
    laya = {("agnews", "topic"): {"acc": 0.95, "n": 40},
            ("boolq", "answer"): {"acc": 0.0, "n": 40}}
    text = table_text(rows, strata(SPLIT), laya=laya, split="test")
    ag = [ln for ln in text.splitlines() if ln.startswith("agnews/topic")][0]
    bq = [ln for ln in text.splitlines() if ln.startswith("boolq/answer")][0]
    assert "0.950" in ag, ag
    assert "0.000" in bq, "a competitor's zero is a measurement, not a missing cell"


def test_a_perfect_laya_macro_still_prints_when_its_other_cell_is_zero():
    """The stratum macro averages over the cells the *model* scored, so a 0.0 in that
    set is a data point; dropping it would silently raise the competitor's number."""
    stats = cell_stats(SPLIT)
    acc, _ = accuracy_cells({"agnews#aaaa/topic": 0.5, "agnews#bbbb/topic": 0.0,
                             "trec#0000/answer_type": 0.0}, SPLIT)
    rows = roll_up(stats, acc, keep_min=5)
    laya = {("agnews", "topic"): {"acc": 0.95, "n": 40},
            ("trec", "answer_type"): {"acc": 0.0, "n": 40}}
    text = table_text(rows, strata(SPLIT), laya=laya, split="test")
    ln = [l for l in text.splitlines() if "macro: 2 cell(s)" in l][0]
    assert "0.475" in ln, ln


# ---- the CLI, on the real split: the floors must reproduce SPEC §4.2 -----------

def _cli(argv, expect_zero=True):
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    r = subprocess.run([sys.executable, "-m", "myna.report", *argv],
                       capture_output=True, text=True, env=env, cwd=REPO)
    if expect_zero:
        assert r.returncode == 0, r.stderr[-3000:]
    return r


@pytest.fixture()
def empty_metrics(tmp_path):
    p = tmp_path / "metrics.json"
    p.write_text(json.dumps({"test": {}, "dev": {}}))
    return str(p)


@pytest.mark.skipif(not (PILOT / "test.jsonl").exists(), reason="pilot corpus not on disk")
def test_the_majority_floor_reproduces_the_published_04331(tmp_path, empty_metrics):
    """§4.2's floors came out of an ad-hoc pass with no script behind them.
    Recomputing them here — through the adapter, over the shipped split — and
    landing on the same number is the witness that both measured the same thing,
    and that the floors G1 is judged against are real."""
    r = _cli(["--suite", str(PILOT), "--split", "test", "--metrics", empty_metrics])
    line = [ln for ln in r.stdout.splitlines() if "majority floor" in ln]
    assert len(line) == 1, r.stdout
    assert "0.433" in line[0], line[0]
    assert "uniform floor 0.332" in line[0], \
        line[0] + " — row-weighted over the mixed option counts, not priced at one set"
    assert "1440 rows over 16 cells" in r.stdout and "16 kept at n>=30" in r.stdout
    assert "6 cell(s) mix option counts" in r.stdout
    # With nothing scored the floors are still a property of the split, so they must
    # print over all kept cells rather than vanish as an em dash.
    assert "no cell in" in r.stdout, "say that the table has floors and no model"
    assert "floors over all 16 kept cells" in line[0], line[0]
    # 9c's `elif` is not decoration: with no model number the clip macro is None, and
    # "the clip column is the model column" would read as a model beating every floor.
    assert "the clip column is the model column" not in r.stdout, \
        "nothing was scored, so there is nothing to compare a clip against"


@pytest.mark.skipif(not (PILOT / "test.jsonl").exists(), reason="pilot corpus not on disk")
def test_a_scored_set_from_another_split_never_becomes_a_model_number(tmp_path):
    """A metrics file scored on a different suite has weights of zero here: the note
    is the report, and the cell must not print an accuracy derived from no rows."""
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"test": {"agnews#ffff/topic": 0.5}}))
    out = tmp_path / "s.json"
    r = _cli(["--suite", str(PILOT), "--split", "test", "--metrics", str(metrics),
              "--out", str(out)])
    assert "1 scored question-sets are not in test.jsonl" in r.stdout, r.stdout
    assert "no cell in" in r.stdout, "an all-unknown metrics block is still no model"
    j = json.loads(out.read_text())
    assert j["unmatched_sets"] == ["agnews#ffff/topic"]
    assert all(c["acc"] is None for c in j["cells"]), "zero rows is not evidence"
    assert j["g1"]["pass"] is False


@pytest.mark.skipif(not (PILOT / "test.jsonl").exists(), reason="pilot corpus not on disk")
def test_min_rows_names_every_cell_it_dropped(tmp_path, empty_metrics):
    r = _cli(["--suite", str(PILOT), "--split", "test", "--metrics", empty_metrics,
              "--min-rows", "100"])
    assert "kept at n>=100" in r.stdout, r.stdout
    dropped_line = [ln for ln in r.stdout.splitlines() if "dropped" in ln][0]
    assert "dropped 10:" in dropped_line, dropped_line
    assert "imdb/positive" in dropped_line and "boolq/answer" in dropped_line
    assert "agnews/topic" not in dropped_line, "116 rows, so it is kept"


def test_a_min_rows_of_zero_fails_loudly(tmp_path, empty_metrics):
    r = _cli(["--suite", str(PILOT), "--split", "test", "--metrics", empty_metrics,
              "--min-rows", "0"], expect_zero=False)
    assert r.returncode != 0 and "--min-rows must be positive" in r.stderr + r.stdout
    assert "Traceback" not in r.stderr, "a refusal, not a crash"


@pytest.mark.skipif(not (PILOT / "test.jsonl").exists(), reason="pilot corpus not on disk")
def test_the_laya_witness_rides_in_the_same_table(tmp_path, capsys):
    """The competitor is read from its own JSON, for the same cell keys, and the
    banner has to say which split and sample size it came from — otherwise a 40-row
    laya sample is quoted beside myna's 1440 with no caveat."""
    art = REPO / "runs" / "myna-v1-rich"
    witness = REPO / "runs" / "laya_decision_v2_test.json"
    if not (art / "metrics.json").exists() or not witness.exists():
        pytest.skip("the v1-rich or laya artifact is not on disk")
    assert main(["--suite", str(PILOT), "--split", "test", "--metrics", str(art),
                 "--laya", str(witness), "--out", str(tmp_path / "s.json")]) == 0
    out = capsys.readouterr().out
    assert "test split, n=546" in out, out[:400]
    assert "0.673" in out, "the laya macro over the shared-instruction stratum"


def test_a_competitor_cell_asked_two_ways_is_merged_by_rows(tmp_path):
    """The competitor keys its own rows on `(source, qid, type)`, and one qid can be a
    choice in some records and a noul in others. myna's cell key carries no type, so the
    two rows are *one* cell here — and keeping whichever row came last prices a 16-answer
    figure as if it were the 40-answer cell's (SPEC §9.32, which this replaced).
    """
    rows = [{"source": "contrastive", "qid": "decision", "type": "choice", "acc": 0.25,
             "n": 24},
            {"source": "contrastive", "qid": "decision", "type": "noul", "acc": 1.0,
             "n": 16},
            {"source": "agnews", "qid": "topic", "type": "choice", "acc": 0.5, "n": 40},
            {"source": "trec", "qid": "answer_type", "type": "choice", "acc": 0.0, "n": 0},
            {"source": "trec", "qid": "answer_type", "type": "choice", "acc": 1.0, "n": 0}]
    p = tmp_path / "laya.json"
    p.write_text(json.dumps({"split": "test", "n": 80, "n_per_source": 40, "rows": rows}))
    cells, _data = laya_cells(p)
    c = cells[("contrastive", "decision")]
    assert c["acc"] == pytest.approx(0.55), \
        f"last-wins again: {c['acc']} is the noul row alone, not (0.25*24 + 1.0*16)/40"
    assert c["n"] == 40 and c["rows"] == 2, c
    assert [q["type"] for q in c["parts"]] == ["choice", "noul"], "the note needs both"
    assert cells[("agnews", "topic")]["rows"] == 1
    assert cells[("trec", "answer_type")]["acc"] == pytest.approx(0.5), \
        "two rows that report no sample size are averaged, not turned into a zero"


# ---- the competitor line: one cell set, or no gap at all -----------------------

CONTROL = REPO / "runs" / "scratch_metrics_test.json"
LAYA_WITNESS = REPO / "runs" / "laya_decision_v2_test.json"


@pytest.mark.skipif(not (PILOT / "test.jsonl").exists(), reason="pilot corpus not on disk")
def test_the_competitor_line_averages_one_cell_set_not_two(tmp_path):
    """SPEC §8's three rows are only comparable if the gap is over *the same cells*.

    Each harness's own summary is a different statistic on a different sample: laya's
    `overall_acc` is row-weighted over its own 40-rows-per-source draw, myna's MACRO is
    unweighted over 16 (source, question) cells of all 1,440 test rows. Subtracting one
    from the other is the mistake this pins shut — so the line must print the
    intersection's numbers and the row-weighted overall must appear nowhere in the run.
    """
    if not (CONTROL.exists() and LAYA_WITNESS.exists()):
        pytest.skip("the control witness or the laya witness is not on disk")
    committed = (REPO / "runs" / "report_scratch_vs_laya.log").read_text().splitlines()
    want = [ln for ln in committed if ln.strip().startswith("myna ")]
    assert len(want) == 1, committed[-6:]
    out = tmp_path / "joined.json"
    r = _cli(["--suite", str(PILOT), "--split", "test", "--metrics", str(CONTROL),
              "--laya", str(LAYA_WITNESS), "--out", str(out)])
    assert "MACRO over the 16 of 16 cells BOTH scored" in r.stdout, r.stdout[-800:]
    assert want[0] in r.stdout, \
        f"the committed witness says {want[0]!r}; this run says otherwise"
    both = json.loads(out.read_text())["macro"]["both"]
    assert both["gap"] == pytest.approx(both["myna"] - both["laya"])
    assert both["cells"] == 16 and both["cells_scored"] == 16, both
    assert "per-cell, unweighted" in r.stdout and "row-weighted overall" in r.stdout
    assert "contrastive/decision is 2 rows of that JSON over 40 answers" in r.stdout, \
        "a cell assembled from two competitor rows has to say so where the gap is printed"
    assert "0.6319" not in r.stdout and "overall_acc" not in r.stdout, \
        "the row-weighted overall leaked into a per-cell comparison"
    assert not any(ln.startswith("  note:") and "absent from the laya JSON" in ln
                   for ln in r.stdout.splitlines()), \
        "both sides scored all 16 cells, so the note must stay silent"


@pytest.mark.skipif(not (PILOT / "test.jsonl").exists(), reason="pilot corpus not on disk")
def test_a_cell_the_competitor_never_scored_moves_the_floor_too(tmp_path):
    """Drop one laya cell and the intersection shrinks — *including its floors*.

    The floor beside a competitor number has to be the floor of the cells in that
    number. Reusing the 16-cell floor next to a 15-cell gap would flatter whichever side
    lost the harder cell, which is exactly the shape of the error §9.30 buried.
    """
    if not (CONTROL.exists() and LAYA_WITNESS.exists()):
        pytest.skip("the control witness or the laya witness is not on disk")
    data = json.loads(LAYA_WITNESS.read_text())
    dropped = ("imdb", "positive")
    data["rows"] = [row for row in data["rows"]
                    if (row["source"], row["qid"]) != dropped]
    assert len(data["rows"]) == 16, "the witness changed shape; find the row again"
    laya = tmp_path / "laya_minus_one.json"
    laya.write_text(json.dumps(data))
    out = tmp_path / "joined.json"
    r = _cli(["--suite", str(PILOT), "--split", "test", "--metrics", str(CONTROL),
              "--laya", str(laya), "--out", str(out)])
    assert "MACRO over the 15 of 16 cells BOTH scored" in r.stdout, r.stdout[-800:]
    assert "note: 1 scored cell(s) are absent from the laya JSON" in r.stdout
    j = json.loads(out.read_text())
    both = j["macro"]["both"]
    shared = [c for c in j["cells"] if c["acc"] is not None
              and (c["source"], c["question"]) != dropped]
    assert both["cells"] == 15 and both["cells_scored"] == 16, both
    assert both["majority"] == pytest.approx(
        sum(c["majority"] for c in shared) / len(shared)), \
        "the floor beside the gap is not the gap's own floor"
    assert both["myna"] == pytest.approx(
        sum(c["acc"] for c in shared) / len(shared)), \
        "myna's side of the gap is averaged over cells laya never scored"
    by_cell = {}
    for row in data["rows"]:
        by_cell.setdefault((row["source"], row["qid"]), []).append(row)

    def merged(key):
        parts = by_cell[key]
        tot = sum(p["n"] for p in parts)
        return (sum(p["acc"] * p["n"] for p in parts) / tot if tot
                else sum(p["acc"] for p in parts) / len(parts))

    assert both["laya"] == pytest.approx(
        sum(merged((c["source"], c["question"])) for c in shared) / len(shared)), \
        "laya's side of the gap is not the merged value of the cells it shares with mine"
    assert both["majority"] != pytest.approx(j["macro"]["majority"]), \
        "the 15-cell floor came out identical to the 16-cell one: nothing was dropped"


@pytest.mark.skipif(not (PILOT / "test.jsonl").exists(), reason="pilot corpus not on disk")
def test_boolq_and_mnli_are_the_only_per_row_sources(tmp_path, empty_metrics):
    r = _cli(["--suite", str(PILOT), "--split", "test", "--metrics", empty_metrics])
    per_row = [ln.split()[0] for ln in r.stdout.splitlines()
               if "per-row-instruction" in ln and "of slots" in ln]
    assert sorted(per_row) == ["boolq", "mnli"], r.stdout[-1200:]
    assert "per-row-instruction — 2 source(s): boolq, mnli" in r.stdout
    shared = [ln for ln in r.stdout.splitlines() if "shared-instruction  " in ln]
    assert len(shared) == 9, shared
    # The two measures part on real data, and the class follows the instruction one:
    # agnews repeats one instruction across all its rows while its 113 exact-signature
    # sets almost never hold more than one row.
    ag = [ln for ln in r.stdout.splitlines() if ln.strip().startswith("agnews")
          and "of slots" in ln][0]
    assert "1.00 of slots repeat an instruction" in ag, ag
    assert "8 distinct over 300 slots / 116 rows" in ag, \
        ag + " — the instruction count and the set count are different numbers"
    assert "113 exact-signature sets" in ag, ag
    assert "0.05 of rows in a reused set" in ag, ag
    assert "per-row-instruction" in ag.split()[1] or "shared-instruction" in ag


@pytest.mark.skipif(not (PILOT / "test.jsonl").exists(), reason="pilot corpus not on disk")
def test_a_real_metrics_file_lands_in_the_right_cells(tmp_path):
    art = REPO / "runs" / "myna-v1-rich"
    if not (art / "metrics.json").exists():
        pytest.skip("the v1-rich artifact is not on disk")
    out = tmp_path / "strata.json"
    r = _cli(["--suite", str(PILOT), "--metrics", str(art), "--split", "test",
              "--out", str(out)])
    assert "G1:" in r.stdout and "not met" in r.stdout, "the void v1 numbers must not pass"
    j = json.loads(out.read_text())
    cells = {(c["source"], c["question"]): c for c in j["cells"]}
    assert len(cells) == 16
    assert all(c["acc"] is not None for c in cells.values()), \
        "every cell of the frozen split is scored, or the macro is over a subset"
    assert cells[("imdb", "positive")]["majority"] == pytest.approx(43 / 80)
    assert cells[("imdb", "positive")]["n"] == 80 and cells[("imdb", "positive")]["sets"] == 1
    # The chance floor of a mixed cell is reported with its whole option distribution
    # and priced at the widest set it appears in, so a reader can audit the weighting.
    # (JSON coerces the option-count keys to strings; the weights are the point.)
    assert cells[("agnews", "topic")]["options_by_size"] == {"4": 104, "5": 12}
    assert cells[("agnews", "topic")]["options"] == 5
    assert j["g1"]["pass"] is False and j["strata"]["boolq"]["class"] == "per-row-instruction"
    assert j["unmatched_sets"] == [], "the artifact was scored on exactly this split"
    assert "floors over all" not in r.stdout, "nothing was dropped, so no second basis"


def test_a_split_that_is_not_there_fails_loudly(tmp_path, empty_metrics):
    empty = tmp_path / "nowhere"
    empty.mkdir()
    r = _cli(["--suite", str(empty), "--split", "test", "--metrics", empty_metrics],
             expect_zero=False)
    assert r.returncode != 0
    assert "test.jsonl" in r.stderr + r.stdout, "name the file the floors need"
    assert "the floors and the row weights" in r.stderr + r.stdout, "a refusal, not a stack trace"
    assert "Traceback" not in r.stderr, r.stderr[-800:]


def test_a_metrics_block_that_is_not_there_fails_loudly(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"dev": {"x/y": 1.0}}))
    r = _cli(["--suite", str(PILOT), "--split", "test", "--metrics", str(metrics)],
             expect_zero=False)
    assert r.returncode != 0 and "no 'test' block" in r.stderr + r.stdout


def test_main_accepts_a_run_directory(tmp_path):
    """The checkpoint directory is what a Kaggle job leaves behind; asking for the
    file inside it is a detail the caller should not have to know."""
    art = REPO / "runs" / "myna-v1-rich"
    if not (art / "metrics.json").exists():
        pytest.skip("the v1-rich artifact is not on disk")
    assert main(["--suite", str(PILOT), "--split", "test", "--metrics", str(art),
                 "--out", str(tmp_path / "s.json")]) == 0
    assert (tmp_path / "s.json").exists()


# ---- 9c's clip column and 9b's Brier column (SPEC §5 P9) ----------------------
#
# Two of the three numbers a cell can be judged on already ride in this table; 9c
# adds `max(model, majority)` *beside* the model rather than instead of it, and the
# sentence that labels the difference is the artifact under test. What a mutation
# battery would catch and a count of passing tests would not: a clip printed in
# place of the accuracy, a clip macro sold as a gain, and a missing `:brier` sidecar
# read as a Brier of zero.

def _cli_metrics(tmp_path, metrics):
    p = tmp_path / "metrics.json"
    p.write_text(json.dumps({"test": metrics}))
    return str(p)


def _pilot_metrics(tmp_path, price, brier=None):
    """A metrics file with the pilot's *own* set keys, priced per question type.

    The keys come from the split so the row weights are the real ones; the prices are
    constants, so every expected figure below is hand arithmetic rather than the
    module's own output.
    """
    groups = load_split(PILOT / "test.jsonl")
    m = {}
    for wf, (qs, _ex) in groups.items():
        for q in qs:
            m[f"{wf}/{q.name}"] = price(q.type)
            if brier and q.type == "noul":
                m[f"{wf}/{q.name}:brier"] = brier
    return _cli_metrics(tmp_path, m)


def test_clip_is_the_bigger_number_and_its_worth_is_the_difference():
    stats = cell_stats(SPLIT)
    acc, _ = accuracy_cells({"boolq#0000/answer": 0.5, "agnews#aaaa/topic": 0.5,
                             "agnews#bbbb/topic": 0.5}, SPLIT)
    rows = {(r["source"], r["question"]): r for r in roll_up(stats, acc, keep_min=5)}
    bq = rows[("boolq", "answer")]
    assert bq["majority"] == pytest.approx(5 / 8), "five 1s in eight rows"
    assert bq["acc"] == pytest.approx(0.5) and bq["clip"] == pytest.approx(0.625)
    assert bq["clip_worth"] == pytest.approx(0.125), \
        "the clip's worth is what the constant predictor has that the model does not"
    ag = rows[("agnews", "topic")]
    assert ag["acc"] == pytest.approx(0.5) and ag["clip"] == pytest.approx(0.5)
    assert ag["clip_worth"] == 0.0, "a cell above its floor is not owed a clip"
    assert rows[("trec", "answer_type")]["clip"] is None, \
        "no model number means no clip, not a clip equal to the floor"


def test_the_clip_and_brier_columns_ride_beside_the_model_not_over_it():
    stats = cell_stats(SPLIT)
    acc, _ = accuracy_cells({"boolq#0000/answer": 0.5}, SPLIT)
    rows = roll_up(stats, acc, keep_min=5, briers={("boolq", "answer"): 0.375})
    text = table_text(rows, strata(SPLIT))
    hdr = [ln for ln in text.splitlines() if ln.startswith("source/question")][0]
    assert "clip" in hdr and "brier" in hdr
    ln = [l for l in text.splitlines() if l.startswith("boolq/answer")][0]
    assert "0.500" in ln and "0.625" in ln, "model and clip on the same line"
    assert "0.375" in ln, ln
    assert "0.563" not in ln, "the line carries both numbers, never their midpoint"
    assert "clip = max(model, maj)" in text and "oracle" in text, \
        "the column is labelled where it is printed, not only in the docstring"


COLS = ("model", "clip", "laya", "maj", "unif", "brier", "-maj")


def _edges(hdr):
    """Where each numeric column *ends*, plus the boundary that opens the first one.

    Every field before the numbers is right-aligned, so a header word ends exactly on
    its field; the label is left-aligned, so its end says nothing. Without the opening
    boundary the first numeric column swallows the whole row prefix.
    """
    return [hdr.index("sets") + len("sets"),
            *(hdr.index(nm) + len(nm) for nm in COLS)]


def _columns(ln, edges):
    """Read one table line by *position*, the way a reader does."""
    out, prev = {}, edges[0]
    for nm, e in zip(COLS, edges[1:]):
        out[nm] = ln[prev:e].strip()
        prev = e
    return out


def test_the_macro_row_lands_in_the_same_columns_as_the_cells(tmp_path, capsys):
    """A header word and the number under it are one claim. The committed artifact
    had the macro row's label overrun its field by six characters, so every figure in
    it sat six columns right of the header it belonged to — no assertion about values
    can see that, and a reader's eye reads a column by position.

    The pilot split, not the small fixture: its longest real cell name is 20
    characters, which is what makes the label field's width load-bearing. A Brier is
    priced so the column left of the delta is occupied too, and one noul cell is read
    whose delta is negative — that is the pair of figures that can collide.
    """
    metrics = _pilot_metrics(tmp_path, lambda t: 0.5, brier=0.4)
    assert main(["--suite", str(PILOT), "--split", "test", "--metrics", metrics]) == 0
    out = capsys.readouterr().out.splitlines()
    hdr = [ln for ln in out if ln.startswith("source/question")][0]
    edges = _edges(hdr)
    assert len(set(edges)) == 8, edges
    body = [ln for ln in out
            if ln.startswith(("contrastive/decision", "banking77/intent",
                              "agnews/is_business", "  macro:"))]
    assert len(body) == 5, \
        f"the 20-char name, a 16-char name, a priced noul cell, two macro rows: {body}"
    for ln in body:
        ends = [m.end() for m in re.finditer(r"-?\d\.\d{3}", ln)]
        assert ends, ln
        assert all(e in edges for e in ends), (ln, edges, ends)
        # Alignment is not enough: a delta that fills its field exactly butts against
        # the figure to its left, and "0.312-0.125" reads as one number. The width the
        # delta gets is what keeps a minus sign off its neighbour.
        assert not re.search(r"\d-\d\.\d", ln), f"two figures touch: {ln}"
    # The one width the data sets rather than the design: the label field has to
    # outlast the longest real cell name, or the table's first column stops being one.
    widest = max(len(f"{s}/{q}") for s, q in cell_stats(load_split(PILOT / "test.jsonl")))
    assert LBL > widest, f"LBL {LBL} against the split's own widest cell name {widest}"


def test_the_clip_keeps_whichever_number_wins(tmp_path, capsys):
    """`clip` is a maximum, so one row has to show the model and another the floor —
    at one price for every cell the two orders both occur on this split. A column
    that always printed the floor, or always the model, passes any single-row check.
    """
    metrics = _pilot_metrics(tmp_path, lambda t: 0.5)
    assert main(["--suite", str(PILOT), "--split", "test", "--metrics", metrics]) == 0
    out = capsys.readouterr().out.splitlines()
    edges = _edges([ln for ln in out if ln.startswith("source/question")][0])

    def cells(prefix):
        return [_columns(ln, edges) for ln in out if ln.startswith(prefix)]

    # yelp/rating's floor is 0.275, so a 0.5 model keeps its own number...
    yr = [r for r in cells("yelp/rating")][0]
    assert yr["model"] == "0.500" and yr["clip"] == "0.500" and yr["maj"] == "0.275", yr
    # ...and agnews/is_business's floor is 0.698, so the same model does not.
    ab = [r for r in cells("agnews/is_business")][0]
    assert ab["model"] == "0.500" and ab["clip"] == "0.698" and ab["maj"] == "0.698", ab
    loser = [ln for ln in out if "agnews/is_business 0.500 < 0.698" in ln]
    assert len(loser) == 1, "the named cell and its worth belong to the floor-loser line"
    # The macro row has to carry the clip too, and it is not the majority macro: eight
    # of the shared group's 14 cells keep 0.500, four take an agnews floor (0.698,
    # 0.653, 0.809, 0.778) and two take 0.537, which averages to 0.572 — against the
    # 0.431 floor and the 0.500 model in the same two columns.
    mr = _columns([ln for ln in out if ln.startswith("  macro: 14")][0], edges)
    assert mr["model"] == "0.500" and mr["clip"] == "0.572" and mr["maj"] == "0.431", mr


def test_a_missing_brier_sidecar_is_an_em_dash_not_a_zero():
    stats = cell_stats(SPLIT)
    acc, _ = accuracy_cells({"boolq#0000/answer": 0.5}, SPLIT)
    ln = [l for l in table_text(roll_up(stats, acc, keep_min=5), strata(SPLIT)).splitlines()
          if l.startswith("boolq/answer")][0]
    assert "—" in ln and "0.000" not in ln, ln


def test_a_floor_losing_cell_is_named_and_its_delta_called_arithmetic(tmp_path, capsys):
    """The clip is an oracle statistic, so the only honest print names the cells and
    refuses the gain. This is the sentence a later run will be quoted on."""
    metrics = _pilot_metrics(tmp_path, lambda t: 0.0)
    assert main(["--suite", str(PILOT), "--split", "test", "--metrics", metrics]) == 0
    out = capsys.readouterr().out
    assert "16 of 16 scored cell(s) answer below their own majority floor" in out, out
    assert "worth +0.250" in out, "sst5's floor is a quarter of the rows: name the worth"
    assert "of arithmetic and 0 of model" in out, out
    assert "leaderboard decision" in out, "the clip is not offered as a fix"
    assert "MACRO over the 16 scored cells: model 0.000" in out, \
        "the clip macro must not become the headline number"
    assert "0.433" in out, "the majority macro is still the floor, printed as the floor"


def test_a_model_no_floor_beats_says_the_clip_column_buys_nothing(tmp_path, capsys):
    metrics = _pilot_metrics(tmp_path, lambda t: 1.0)
    assert main(["--suite", str(PILOT), "--split", "test", "--metrics", metrics]) == 0
    out = capsys.readouterr().out
    assert "no scored cell answers below its own majority floor" in out, out
    assert "1.000 = 1.000" in out, "clip == model, and the line says so rather than silence"


def test_brier_is_row_weighted_and_a_key_from_another_split_carries_no_weight():
    """The weights differ by 10 to 2 here on purpose: with equal-size sets the
    weighted and unweighted means coincide, and a dropped weight would survive."""
    br = brier_cells({"agnews#aaaa/topic:brier": 0.1, "agnews#bbbb/topic:brier": 0.3,
                      "zzz#ffff/topic:brier": 0.9}, SPLIT)
    assert br[("agnews", "topic")] == pytest.approx((0.1 * 10 + 0.3 * 2) / 12), br
    assert ("zzz", "topic") not in br, "a sidecar with no rows in the split is not a cell"
    assert brier_cells({"boolq#0000/answer": 0.1}, SPLIT) == {}, \
        "a plain accuracy key is not a Brier sidecar"


def test_the_noul_line_counts_sidecars_and_reads_against_the_coin_flip(tmp_path, capsys):
    metrics = _pilot_metrics(tmp_path, lambda t: 0.5, brier=0.25)
    assert main(["--suite", str(PILOT), "--split", "test", "--metrics", metrics]) == 0
    out = capsys.readouterr().out
    assert "noul: 8 of 8 scored binary cells carry a :brier sidecar" in out, out
    assert "0.250 against 0.250 for a coin flip" in out, \
        "a Brier equal to chance is the point of printing the reference"
    assert "§9.24" in out, "the sentence says why an argmax alone is not enough"


def test_a_metrics_file_with_no_sidecars_calls_its_brier_column_missing_data(tmp_path,
                                                                            capsys):
    metrics = _pilot_metrics(tmp_path, lambda t: 0.5)
    assert main(["--suite", str(PILOT), "--split", "test", "--metrics", metrics]) == 0
    out = capsys.readouterr().out
    assert "noul: 0 of 8 scored binary cells carry a :brier sidecar" in out, out
    assert "missing data and not a model that scores 0" in out, out


def test_the_json_report_carries_the_clip_and_brier_as_fields(tmp_path):
    out = tmp_path / "r.json"
    metrics = _pilot_metrics(tmp_path, lambda t: 0.1, brier=0.4)
    assert main(["--suite", str(PILOT), "--split", "test", "--metrics", metrics,
                 "--out", str(out)]) == 0
    j = json.loads(out.read_text())
    # `clip` is per-cell, so a model price of 0.1 keeps its own number everywhere
    # except banking77/intent, whose floor is 0.043: the average of those maxima is
    # the macro, and it is not the floor macro. Both halves of that difference are
    # what the test would miss if the column were `majority` renamed.
    prices = [max(0.1, c["majority"]) for c in j["cells"]]
    assert j["macro"]["clip"] == pytest.approx(sum(prices) / len(prices)), j["macro"]
    assert j["macro"]["clip_worth"] == pytest.approx(j["macro"]["clip"] - j["macro"]["acc"]), \
        "the clip is the model plus its own worth, cell by cell: an unclipped difference " \
        "would make a cell above its floor subtract from the total"
    assert len(j["below_majority_floor"]) == 15, \
        "one cell beats its own floor at 0.1, and it is the 77-way one"
    assert not any(e["cell"] == "banking77/intent" for e in j["below_majority_floor"])
    assert all(e["clip_worth"] > 0 for e in j["below_majority_floor"])
    assert j["macro"]["brier_noul"] == pytest.approx(0.4)
    assert j["brier"]["noul_cells"] == 8 and j["brier"]["priced_cells"] == 8
    cell = [c for c in j["cells"] if c["source"] == "sst5"][0]
    assert cell["clip"] == pytest.approx(cell["majority"]) and cell["brier"] is None

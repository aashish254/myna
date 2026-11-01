"""The vegaml control arm, checked at the seam: a relabel must not become a recomputation.

`bench/eval_vegaml_row_outcomes.py` scores an engine that is not this repository's, and
`bench/vegaml_control.py` puts that dump into the second-engine slot of the frozen
`bench/significance.py`. Both are thin, and thin files are where a silent wrong answer lives,
so what is asserted here is arithmetic the test computes itself rather than a number the code
printed.

Two claims carry the unit and each gets its own kind of proof:

* **The scoring shim is the only place a control's correctness is decided.** vegaml answers a
  `score` question by positional index while the published competitor column answers by level
  text, so the index rule is tested against the real suite's own golds
  (`test_score_labels_are_positional`) rather than against a fixture that assumes them, and
  every one of the three answer types is checked with a hand-written answer whose `ok` the test
  decides before the code runs.
* **The control harness renames; it does not re-derive.**
  `test_wrapper_is_a_relabel_of_the_same_arithmetic` runs `pack`/`statize`/`boot` directly and
  compares them with the wrapper's JSON, key for key, so a future edit that makes
  `vegaml_macro` mean something other than "the second-engine column" fails here instead of
  shipping a silently different statistic under a new name.

The abstain field is the one place this control can do something neither myna nor laya can, and
the harness scores it forced-choice on purpose; `test_abstain_is_recorded_not_applied` pins that
choice, because a later "just exclude the abstentions" edit would change the control's accuracy
without changing a single line of the scoring code.

Two arms beyond the fixture's default make the verdicts falsifiable rather than merely printed:
`strong_ok=0` puts the control *below* myna, which is the only way the difference claim's
second side gets exercised (`control-above-myna` fails there while `myna-vs-control` stands), and
`weak_every=1` makes the control answer every row exactly as myna did, so the gap is exactly 0,
both claims fail, and neither verdict word may appear on the winning side. The myna-side guard is
parametrised over all four statistics it checks, each doctored by 1e-6 — the slack a loosened
threshold would swallow.

The control dump here is **synthetic and labelled so**, for the same reason
`tests/test_significance.py`'s competitor is: `control_dump()` copies myna's own rows and marks
every third one, so the arithmetic under test is pairing, relabelling and adjudication — not
vegaml's ability. `runs/vegaml_control.json` is the real dump read by the same `main()`.

Two families are therefore read off the source and off the committed artifacts instead of off that
fixture, because no cheap test can run 1,176 inferences: the crash-resume pair
(`test_the_raw_sidecar_writes_every_field_from_raw_reads`,
`test_from_raw_and_a_live_pass_bind_the_same_names`) pins the two halves of `--from-raw` against
each other — the first pass died in exactly that seam after paying for its rows — and
`test_two_independent_passes_of_the_control_answer_identically` compares the two real full passes
of vegaml, field by field over all 1,440 outcomes, so "the control was run once" is a repeatable
claim rather than a sample of one run.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "bench"))

import bench.significance as SIGN  # noqa: E402
import bench.vegaml_control as CTRL  # noqa: E402
from eval_vegaml_row_outcomes import (TYPE_MAP, boolean_disagree, cell_table,  # noqa: E402
                                      outcome, score_pred, vegaml_questions)
from test_row_outcomes import MIN_ROWS, fx  # noqa: E402

SUITE = REPO / "data" / "decision-v2-pilot" / "test.jsonl"

#: One plausible answer per suite type, in the shape `vegaml/_api.py:281-298` returns.
ANSWER = {
    "choice": {"probs": {"a": 0.2, "b": 0.7, "c": 0.1}, "confidence": 0.7, "choice": "b",
               "prediction_set": ["b"], "abstain": False, "unbound": False},
    "score": {"probs": {"0": 0.1, "1": 0.1, "2": 0.6, "3": 0.1, "4": 0.1},
              "confidence": 0.6, "score": 2, "expected_score": 2.0,
              "prediction_set": [2], "abstain": False, "unbound": False},
    "noul": {"probs": {"False": 0.3, "True": 0.7}, "confidence": 0.7, "boolean": 0.7,
             "prediction_set": [True], "abstain": False, "unbound": False},
}


# --------------------------------------------------------------------- the real suite, read back
def test_score_labels_are_positional():
    """Every `score` gold in the published suite is an index into its own criteria list.

    This is the fact the control's score rule stands on: `vegaml/_api.py:275` labels a score
    answer `list(range(k))`, so comparing that with the gold is only correct if the gold is also
    a position. If a future corpus row stores a level *value* (a `5` for five stars) this test
    fails and the rule has to be re-written — rather than scoring that cell at chance and
    reporting it as a control.
    """
    seen = 0
    with SUITE.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            for qid, q in json.loads(line)["questions"].items():
                if q["type"] != "score":
                    continue
                crit = q["criteria"]
                assert isinstance(crit, list), f"{qid}: score criteria is {type(crit).__name__}"
                assert isinstance(q["label"], int) and not isinstance(q["label"], bool), \
                    f"{qid}: score label {q['label']!r} is not a positional int"
                assert 0 <= q["label"] < len(crit), \
                    f"{qid}: label {q['label']} is outside range({len(crit)})"
                seen += 1
    assert seen >= 200, f"only {seen} score outcomes scanned — the suite is not what this tests"


def test_suite_types_are_all_mapped():
    """No question type reaches the control unmapped, and `noul` is the only rename."""
    types = set()
    with SUITE.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                types |= {q["type"] for q in json.loads(line)["questions"].values()}
    assert types == set(TYPE_MAP), f"suite types {sorted(types)} vs map {sorted(TYPE_MAP)}"
    assert TYPE_MAP["noul"] == "boolean"


# -------------------------------------------------------------------------------------------- shim
def test_vegaml_questions_renames_and_strips_gold():
    qs = {"x": {"type": "noul", "instructions": "does it hold?", "label": True, "src": "boolq"},
          "y": {"type": "choice", "instructions": "which?",
                "criteria": {"a": None, "b": "second"}, "label": "a", "src": "s"},
          "z": {"type": "score", "instructions": {"question": "q", "focus": "f"},
                "criteria": ["l0", "l1"], "label": 1}}
    out = vegaml_questions(qs)
    assert out["x"] == {"type": "boolean", "instructions": "does it hold?"}
    assert out["y"]["criteria"] == {"a": None, "b": "second"}
    assert out["z"]["instructions"] == {"question": "q", "focus": "f"}, "a dict instruction " \
        "passes through; vegaml serialises it (`_api.py:194`)"
    for qid, d in out.items():
        assert "label" not in d and "src" not in d, f"{qid} leaks its gold to the engine"


def test_vegaml_questions_refuses_an_unmapped_type():
    with pytest.raises(SystemExit) as e:
        vegaml_questions({"x": {"type": "regression", "instructions": "i", "label": 1}})
    assert "no vegaml counterpart" in str(e.value)


def test_no_criteria_field_is_sent_when_the_suite_has_none():
    """A noul question carries no criteria; `criteria: None` would read as an empty option set."""
    out = vegaml_questions({"x": {"type": "noul", "instructions": "i", "label": False}})
    assert "criteria" not in out["x"]


# ------------------------------------------------------------------ scoring, one type at a time
def test_outcome_choice_matches_the_label_field():
    q = {"type": "choice", "label": "b", "criteria": {"a": None, "b": None, "c": None}}
    assert outcome(q, ANSWER["choice"]) == (1, "b", 0.7, 0)
    assert outcome({**q, "label": "c"}, ANSWER["choice"])[0] == 0


def test_outcome_score_is_the_positional_index_rule():
    q = {"type": "score", "label": 2, "criteria": ["l0", "l1", "l2", "l3", "l4"]}
    assert outcome(q, ANSWER["score"]) == (1, 2, 0.6, 0)
    assert outcome({**q, "label": 4}, ANSWER["score"])[0] == 0
    assert score_pred(ANSWER["score"]) == 2


def test_outcome_boolean_uses_the_half_line():
    assert outcome({"type": "noul", "label": True}, ANSWER["noul"]) == (1, True, 0.7, 0)
    assert outcome({"type": "noul", "label": False}, ANSWER["noul"])[0] == 0
    at_half = dict(ANSWER["noul"], boolean=0.5)
    assert outcome({"type": "noul", "label": True}, at_half)[1] is True, "p >= 0.5 answers yes"


def test_disagreement_between_probs_and_the_answer_field_is_counted():
    """The engine's headline field and its own probability table can disagree — and must say so."""
    bad = dict(ANSWER["choice"], choice="c")
    q = {"type": "choice", "label": "c", "criteria": {"a": None, "b": None, "c": None}}
    ok, pred, _conf, disagree = outcome(q, bad)
    assert (ok, pred, disagree) == (1, "c", 1)
    assert outcome({"type": "score", "label": 4}, dict(ANSWER["score"], score=4))[3] == 1


def test_boolean_disagreement_needs_both_the_side_and_the_confidence():
    """`boolean` must favour the argmax of `probs`, and `confidence` must equal its max."""
    assert boolean_disagree(ANSWER["noul"], True) is False
    against_its_table = {"probs": {"False": 0.7, "True": 0.3}, "confidence": 0.7, "boolean": 0.6}
    assert boolean_disagree(against_its_table, True) is True, "probs say no, the field says yes"
    stale_conf = dict(ANSWER["noul"], confidence=0.55)
    assert boolean_disagree(stale_conf, True) is True, "confidence is not p.max()"
    # and the same table read as a `no` is consistent, so the check is not simply "flag it"
    assert boolean_disagree(against_its_table, False) is False


def test_outcome_boolean_flags_the_field_that_contradicts_its_probs():
    a = {"probs": {"False": 0.7, "True": 0.3}, "confidence": 0.7, "boolean": 0.6}
    ok, pred, _conf, disagree = outcome({"type": "noul", "label": True}, a)
    assert (pred, disagree) == (True, 1)
    assert ok == 1, "the field is still what the control answered"


def test_abstain_is_recorded_not_applied():
    """An abstaining answer is still scored, because neither myna nor laya can decline."""
    q = {"type": "choice", "label": "b", "criteria": {"a": None, "b": None}}
    assert outcome(q, dict(ANSWER["choice"], abstain=True))[0] == 1, \
        "the harness would have to change a scoring rule to honour an abstain flag"


def test_confidence_is_the_engines_not_a_recomputed_max():
    q = {"type": "choice", "label": "b", "criteria": {"a": None, "b": None}}
    assert outcome(q, dict(ANSWER["choice"], confidence=None))[2] is None


def test_cell_table_merges_two_types_into_one_cell():
    """`contrastive/decision` is one cell asked two ways — the published geometry (SPEC §9.32)."""
    rows = [{"meta": {"source": "contrastive", "id": 1},
             "questions": {"decision": {"type": "noul", "ok": 1},
                           "decision_alt": {"type": "choice", "ok": 0}}},
            {"meta": {"source": "contrastive", "id": 2},
             "questions": {"decision": {"type": "choice", "ok": 1}}}]
    c = cell_table(rows)
    assert [x["question"] for x in c] == ["decision", "decision_alt"]
    d = c[0]
    assert (d["n"], d["hits"], d["acc"]) == (2, 2, 1.0)
    assert d["types"] == {"choice": {"acc": 1.0, "n": 1}, "noul": {"acc": 1.0, "n": 1}}, \
        "each type is weighted by its own instances, not the cell's"
    alt = c[1]
    assert (alt["n"], alt["hits"], alt["acc"]) == (1, 0, 0.0), "a miss may not count as a hit"
    assert alt["types"] == {"choice": {"acc": 0.0, "n": 1}}


# ------------------------------------------------------------------------- the control fixture
def vegaml_pin():
    return {"version": "0.15.0", "path": "/tmp/vegaml", "commit": "0" * 40,
            "init_sha256": "a" * 16, "python": "3.13.1", "torch": "2.15.0",
            "transformers": "5.22.1", "numpy": "2.5.4", "repo": "nandakishorm/vega-U",
            "checkpoint_dir": "/tmp/vega-U", "checkpoint_files": {},
            "reader": "Qwen3.5-0.8B", "reader_hub_id": "Qwen/Qwen3.5-0.8B",
            "device": "cpu", "threads": 1, "mode": "engine", "views": 1, "alpha": "0.1",
            "reader_params": 752393024, "engine_params": 14342512,
            "backbone_used": "Qwen/Qwen3.5-0.8B"}


def control_dump(myna, skip_ids=(), abstain_ids=(), weak_every=3, strong_ok=1,
                 unbound_ids=()):
    """A vegaml-shaped dump over the fixture's own rows: every `weak_every`-th row is the
    control's weak one and the rest answer with `strong_ok`.

    Row-level on purpose, for the same reason `test_significance.py` marks whole rows: the
    cluster key has to be doing something. `abstain_ids` flags a row's answers as abstained so
    the control-health block has a non-zero rate to carry. `weak_every` and `strong_ok` are knobs,
    not decoration: `strong_ok=0` puts the control *below* myna, and `weak_every=1` makes it answer
    every row exactly as myna did — the two arms the two-sided claim and the verdict words need in
    order to be falsifiable at all.
    """
    rows = []
    for r in myna["rows"]:
        if r["meta"]["id"] in skip_ids:
            continue
        strong = r["idx"] % weak_every != 0
        rec = {}
        for name, q in r["questions"].items():
            rec[name] = {"type": q["type"], "ok": strong_ok if strong else q["ok"],
                         "gold": q["gold"], "pred": q["gold"], "conf": 0.6,
                         "abstain": r["meta"]["id"] in abstain_ids,
                         "set_n": 1, "agree": None,
                         "unbound": r["meta"]["id"] in unbound_ids}
        rows.append({"meta": r["meta"], "group": r["group"], "idx": r["idx"], "questions": rec})
    n_q = sum(len(r["questions"]) for r in rows)
    n_ab = sum(1 for r in rows for q in r["questions"].values() if q["abstain"])
    return {"cmd": "fixture", "engine": "vegaml", "vegaml": vegaml_pin(), "warnings": [],
            "suite": myna["suite"], "split": myna["split"], "n_rows": len(rows),
            "n_questions": n_q, "errors": [], "cluster_key": "meta.id",
            "myna_n_rows": myna["n_rows"], "paired_cells": len(myna["cells"]),
            "cells": cell_table(rows), "reader_tokens": n_q * 40, "argmax_disagreements": 0,
            "abstain": {"n": n_ab, "rate": n_ab / max(n_q, 1),
                        "accuracy_of_abstained": None, "mean_set_size": 1.0,
                        "rule": "scored forced-choice; the flag is recorded, not applied"},
            "seconds": 1.0, "rows": rows}


@pytest.fixture(scope="module")
def ctl(fx, tmp_path_factory):
    """Run the wrapper for real over the fixture dumps and hand back its JSON, MD and inputs."""
    root = tmp_path_factory.mktemp("vegaml-ctl")
    dump = control_dump(fx.dump, abstain_ids={r["meta"]["id"] for r in fx.dump["rows"]
                                             if r["idx"] % 5 == 0})
    p = root / "vegaml.rowoutcomes.json"
    p.write_text(json.dumps(dump))
    out, md = root / "control.json", root / "control.md"
    rc = CTRL.main(["--myna", str(fx.dump_path), "--control", str(p),
                    "--report", str(fx.report_path), "--B", "3000", "--seed", "7",
                    "--min-rows", str(MIN_ROWS), "--out", str(out), "--md", str(md)])
    assert rc == 0, "the control harness refused to write on a clean fixture"
    return SimpleNamespace(dump=dump, dump_path=p, res=json.loads(out.read_text()), out=out,
                           md=md.read_text(), root=root)


# ------------------------------------------------------------------------------- the wrapper
def run_ctl(fx, tmp_path, dump, name="c", B=500, seed=3, extra=()):
    """Run `main()` for real and hand back all three of its faces: exit code, JSON, prose.

    The console is captured rather than read from the artifact because the verdict words are
    printed there *and* in the markdown, and `bench/mutation_vegaml_control.py` hardcodes each of
    them separately — a battery needs a test on each side it mutates.
    """
    p = tmp_path / f"{name}.rowoutcomes.json"
    p.write_text(json.dumps(dump))
    out, md = tmp_path / f"{name}.res.json", tmp_path / f"{name}.res.md"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = CTRL.main(["--myna", str(fx.dump_path), "--control", str(p), "--report",
                        str(fx.report_path), "--B", str(B), "--seed", str(seed),
                        "--min-rows", str(MIN_ROWS), "--out", str(out), "--md", str(md),
                        *extra])
    return SimpleNamespace(rc=rc, dump_path=p, out=out, res=json.loads(out.read_text())
                           if out.exists() else None,
                           md=md.read_text() if md.exists() else "", console=buf.getvalue())


def test_wrapper_is_a_relabel_of_the_same_arithmetic(ctl, fx):
    """`vegaml_macro` is significance.py's second-engine column, not a new statistic."""
    packed = SIGN.pack(fx.dump, ctl.dump, fx.report["cells"], MIN_ROWS)
    point = SIGN.statize(*[packed[k].sum(0) for k in ("nm", "hm", "nl", "hl", "um")],
                         packed["lab"].sum(0))
    assert ctl.res["point"]["vegaml_macro"] == pytest.approx(point["laya_macro"], abs=1e-12)
    assert (ctl.res["point"]["gap_myna_minus_vegaml"]
            == pytest.approx(point["gap_myna_minus_laya"], abs=1e-12))
    assert ctl.res["relabel"] == {"laya_macro": "vegaml_macro",
                                  "gap_myna_minus_laya": "gap_myna_minus_vegaml"}
    assert not [k for k in ctl.res["point"] if k.startswith("laya")]
    assert not [k for k in ctl.res["point"] if k.startswith("_")], \
        "`statize` returns per-replicate arrays under `_` keys; publishing them is a schema leak"


def test_myna_side_survives_the_control(ctl, fx):
    """Swapping the second engine cannot move myna's macro, floors or margin."""
    for k, want in (("myna_macro", fx.report["macro"]["acc"]),
                    ("majority_floor", fx.report["macro"]["majority"]),
                    ("uniform_floor", fx.report["macro"]["uniform"]),
                    ("margin_over_floor", fx.report["g1"]["margin_observed"])):
        assert ctl.res["point"][k] == pytest.approx(want, abs=1e-9), k


def test_resample_is_the_same_draw_at_the_same_seed(ctl, fx):
    """The replicate digest depends on row count and seed only — so both arms share one draw."""
    other = control_dump(fx.dump, weak_every=2)
    p = ctl.root / "other.json"
    p.write_text(json.dumps(other))
    out = ctl.root / "other-out.json"
    assert CTRL.main(["--myna", str(fx.dump_path), "--control", str(p), "--report",
                      str(fx.report_path), "--B", "3000", "--seed", "7", "--min-rows",
                      str(MIN_ROWS), "--out", str(out), "--md", str(out) + ".md"]) == 0
    a, b = ctl.res, json.loads(out.read_text())
    assert a["bootstrap"]["first_replicate_digest"] == b["bootstrap"]["first_replicate_digest"]
    assert a["point"]["vegaml_macro"] != b["point"]["vegaml_macro"], \
        "the two fixtures would be the same measurement and the digest proves nothing"


def test_mcnemar_pooled_is_the_sum_of_the_cells(ctl):
    per = ctl.res["per_cell"]
    mc = ctl.res["mcnemar"]["vs_vegaml"]
    assert (sum(c["mcnemar"]["b"] for c in per),
            sum(c["mcnemar"]["c"] for c in per)) == (mc["b"], mc["c"])
    assert mc["n"] == mc["b"] + mc["c"]
    assert sum(c["n"] for c in per) == ctl.res["paired_question_instances"]
    for c in per:
        assert c["gap"] == pytest.approx(c["myna"] - c["vegaml"], abs=1e-12), \
            "the Δ column is the difference of the two columns printed beside it"


def test_mcnemar_counts_are_the_paired_discordance(ctl, fx):
    """`b` and `c` recomputed by the test from the two dumps, with no help from the harness.

    McNemar reads only the off-diagonal, so this is the one place the control's per-instance
    outcomes have to line up with myna's row for row. The fixture's control is never right where
    myna is wrong and never wrong where myna is right except on the rows it copies, so `b` is a
    number this test can predict: it is the count of paired instances on a *strong* row where
    myna answered correctly — zero, because a strong row is answered correctly by construction.

    The counting is restricted to the published cells, and that restriction is itself the claim:
    `pack()` keys on `(source, question)` from `report["cells"]` with `n >= min_rows`, so the
    fixture's below-floor `trec/answer_type` questions are in both dumps and in no cell. Counting
    them here would make the harness look two-pairs wrong on a fixture that is fine.
    """
    eligible = {(c["source"], c["question"]) for c in fx.report["cells"]
                if c["n"] >= MIN_ROWS}
    by_id = {r["meta"]["id"]: r for r in fx.dump["rows"]}
    b = c = paired = excluded = 0
    for cr in ctl.dump["rows"]:
        mr = by_id[cr["meta"]["id"]]
        source = cr["meta"]["source"]
        for name, cq in cr["questions"].items():
            mq = mr["questions"].get(name)
            if mq is None:
                continue
            if (source, name) not in eligible:
                excluded += 1
                continue
            paired += 1
            b += int(mq["ok"] and not cq["ok"])
            c += int(cq["ok"] and not mq["ok"])
    mc = ctl.res["mcnemar"]["vs_vegaml"]
    assert (b, c) == (mc["b"], mc["c"]), "the harness counts a different discordance than this"
    assert b == 0 and c > 0, "the fixture stopped discriminating the two sides"
    assert mc["p"] is not None and mc["p"] <= 1.0
    assert paired == ctl.res["paired_question_instances"]
    assert excluded > 0, "no fixture question is outside a kept cell, so this arm tests nothing"


def test_direction_of_the_fixture_control(ctl):
    """The fixture lets the control answer every third row wrong, so it sits above myna."""
    assert ctl.res["point"]["vegaml_macro"] > ctl.res["point"]["myna_macro"]
    gap = ctl.res["ci95"]["gap_myna_minus_vegaml"]
    by_id = {c["id"]: c for c in ctl.res["claims"]}
    assert by_id["control-above-myna"]["survives"] is (gap[1] < 0)
    assert by_id["myna-vs-control"]["survives"] is (gap[0] > 0 or gap[1] < 0)
    for cl in by_id.values():
        assert cl["ci95"] == ctl.res["ci95"][cl["statistic"]], \
            "a claim adjudicated on an interval other than the one it prints"


def test_a_control_below_myna_stands_the_difference_and_not_the_parity(fx, tmp_path):
    """The two-sided rule is the only thing that makes `myna-vs-control` a claim at all.

    `control-above-myna` reads one side of the interval; if `myna-vs-control` read the same side
    the two sentences would be the same sentence and the harness could report "they differ" only
    when the control won. Here the fixture's control answers *every* strong row wrongly, so the
    gap is positive and excludes 0 from above.
    """
    r = run_ctl(fx, tmp_path, control_dump(fx.dump, weak_every=3, strong_ok=0), name="below",
                B=3000, seed=7)
    assert r.rc == 0
    gap = r.res["ci95"]["gap_myna_minus_vegaml"]
    assert gap[0] > 0, f"the fixture control is not below myna: {[round(x, 4) for x in gap]}"
    assert r.res["point"]["vegaml_macro"] < r.res["point"]["myna_macro"]
    by_id = {c["id"]: c for c in r.res["claims"]}
    assert by_id["myna-vs-control"]["survives"] is True
    assert by_id["control-above-myna"]["survives"] is False
    assert "**stands**" in r.md and "**does not stand**" in r.md
    assert "REWRITTEN  control-above-myna" in r.console
    assert "SURVIVES  myna-vs-control" in r.console


def test_a_control_that_answers_exactly_like_myna_stands_nothing(fx, tmp_path):
    """The degenerate arm: identical outcomes, so the gap is 0 and no claim survives.

    This is the arm that proves the verdict words are computed. Both sentences are printed as
    false here, and a harness that hardcoded either word — in the markdown or on the console —
    says something else about a control that is myna.
    """
    r = run_ctl(fx, tmp_path, control_dump(fx.dump, weak_every=1), name="same")
    assert r.rc == 0
    assert r.res["point"]["gap_myna_minus_vegaml"] == pytest.approx(0.0, abs=1e-12)
    assert r.res["mcnemar"]["vs_vegaml"] == {"b": 0, "c": 0, "n": 0, "p": None}
    assert all(c["survives"] is False for c in r.res["claims"])
    assert "**stands**" not in r.md
    assert "p = n/a (no discordant pairs)" in r.md, \
        "zero discordant pairs must read as an absence of evidence, not as `None`"
    assert "SURVIVES" not in r.console and r.console.count("REWRITTEN") == 2
    for c in r.res["per_cell"]:
        assert c["gap"] == pytest.approx(0.0, abs=1e-12)


def test_the_seed_reaches_the_resample(ctl, fx, tmp_path):
    """The digest pins the draw, so a second seed has to move it and the answers must not.

    `--B` is held at the fixture's 3,000 on purpose: `boot()` hashes the first *block*, whose
    shape is `min(1000, B)` rows, so a run that changes the seed and the replicate count
    together cannot attribute the moved digest to either. That is how the "ignore the seed and
    draw at 20261031" mutant survived its first battery — the comparison varied B with the seed.
    """
    r = run_ctl(fx, tmp_path, ctl.dump, name="seed8", B=3000, seed=8)
    assert r.rc == 0
    assert r.res["bootstrap"]["first_replicate_digest"] != \
        ctl.res["bootstrap"]["first_replicate_digest"], "the seed flag went nowhere"
    assert r.res["point"] == ctl.res["point"], "a resample seed may not move a point estimate"


def test_the_control_provenance_and_health_are_carried_from_the_dump(fx, tmp_path):
    """Every field of the witness block is read off the dump, not written by this harness.

    The fixture's own control is deliberately clean — no errors, no warnings, no disagreements —
    so a hardcoded zero in any of those lines is invisible until a dump that *has* them is put
    through. This one does, under values no real vegaml pass reports either.
    """
    dump = control_dump(fx.dump)
    dump["engine"] = "vegaml"
    dump["errors"] = [{"id": 3, "source": "yelp", "error": "RuntimeError: no answer"}]
    dump["warnings"] = ["UserWarning: temperature 0.31 outside the clamp"]
    dump["argmax_disagreements"] = 3
    dump["reader_tokens"] = 999_999
    dump["seconds"] = 1234.5
    dump["vegaml"] = dict(dump["vegaml"], version="9.9.9", init_sha256="b" * 16,
                          reader="Other/Reader", mode="ttt", views=2, device="mps",
                          torch="0.0.1", transformers="1.2.3")
    r = run_ctl(fx, tmp_path, dump, name="prov")
    assert r.rc == 0
    c = r.res["control"]
    assert (c["version"], c["package_sha256"], c["reader"], c["mode"], c["views"],
            c["device"], c["torch"], c["transformers"]) == (
        "9.9.9", "b" * 16, "Other/Reader", "ttt", 2, "mps", "0.0.1", "1.2.3")
    h = r.res["control_health"]
    assert h["errors"] == 1, "a control that failed to answer some rows must say so"
    assert h["argmax_disagreements"] == 3
    assert h["reader_tokens"] == 999_999 and h["inference_seconds"] == 1234.5
    assert h["warnings"] == ["UserWarning: temperature 0.31 outside the clamp"]
    assert h["abstain"] == dump["abstain"]
    assert "9.9.9" in r.md and "`Other/Reader`" in r.md


def test_the_unbound_flag_is_tallied_from_the_dump_rows(fx, tmp_path):
    """`unbound` is the control's second decline flag and the prose quotes its count.

    The fixture marks one set of rows abstained, another unbound, and lets them overlap, because
    the published sentence makes two claims at once — a rate for `unbound` and how much of it is
    the same rows `abstain` already caught. The expected numbers are recounted from the dump here
    rather than pasted, so a tally that reads the wrong field, drops the overlap or counts rows
    instead of question outcomes fails on the arithmetic.
    """
    rows = fx.dump["rows"]
    ab = {r["meta"]["id"] for r in rows if r["idx"] % 5 == 0}
    un = {r["meta"]["id"] for r in rows if r["idx"] % 3 == 0}
    assert ab and un and (ab & un), "the fixture stopped exercising the overlap"
    dump = control_dump(fx.dump, abstain_ids=ab, unbound_ids=un)
    r = run_ctl(fx, tmp_path, dump, name="unbound")
    assert r.rc == 0
    qs = [q for row in dump["rows"] for q in row["questions"].values()]
    want = sum(1 for q in qs if q["unbound"])
    both = sum(1 for q in qs if q["unbound"] and q["abstain"])
    got = r.res["control_health"]["unbound"]
    assert got["n"] == want == len([q for q in qs if q["unbound"]]), got
    assert got["of"] == len(qs), "the denominator is question outcomes, not rows"
    assert got["also_abstained"] == both, "the overlap is the sentence's whole point"
    assert want > both > 0, "neither side of the overlap would be tested"
    assert f"{got['n']} of {got['of']} outcomes, {got['also_abstained']} of them" in r.md, \
        "the count is in the JSON and not in the prose that quotes it"


def test_control_health_block_is_carried_from_the_dump(ctl):
    h = ctl.res["control_health"]
    assert h["abstain"] == ctl.dump["abstain"]
    assert h["abstain"]["n"] > 0, "the fixture would not be testing the abstain path at all"
    assert h["abstain"]["rule"] == "scored forced-choice; the flag is recorded, not applied"
    assert h["reader_tokens"] == ctl.dump["reader_tokens"]
    assert h["warnings"] == []


def test_md_text_names_the_relabel_and_the_abstain_rule(ctl):
    assert "laya_macro" in ctl.md and "vegaml_macro" in ctl.md
    assert "both of the control's decline flags are recorded, never applied" in ctl.md
    assert "`vegaml` 0.15.0" in ctl.md
    assert "[0." in ctl.md, "the MD prints an interval per statistic"
    for k in ("myna_macro", "majority_floor", "uniform_floor", "margin_over_floor",
              "vegaml_macro", "gap_myna_minus_vegaml"):
        assert f"| `{k}` |" in ctl.md, f"{k} is in the JSON and not in the prose"
    p, lo, hi = (ctl.res["point"]["vegaml_macro"], *ctl.res["ci95"]["vegaml_macro"])
    assert f"| {p:.4f} | [{lo:.4f}, {hi:.4f}] |" in ctl.md, \
        "the prose rounds to something other than the four decimals the published row quotes"


def test_md_text_prints_the_mcnemar_counts_from_the_json(ctl):
    """The pooled McNemar sentence is the JSON's `b`, `c` and p, formatted, not a paraphrase.

    `mcnemar_exact` itself is `significance.py`'s and is already mutated in
    `bench/mutation_significance.py`; what is *this* harness's face is the paragraph that prints
    its result. A p-value hardcoded here would survive a broken count next door, and the prose is
    the half a reader quotes.
    """
    mc = ctl.res["mcnemar"]["vs_vegaml"]
    assert ctl.md.count(f"pooled over {mc['n']} discordant pairs") == 1
    assert f"myna right and the control wrong {mc['b']}, the reverse {mc['c']}, " \
           f"exact two-sided p = {format(mc['p'], '.3g')}." in ctl.md
    assert f"pooled over {mc['n']} discordant pairs" in ctl.md
    assert "n/a" not in ctl.md.split("## McNemar")[1].split("## The control's own")[0], \
        "the fixture has discordant pairs, so the prose may not report an absence of evidence"


def test_two_dumps_sharing_no_row_id_are_refused(fx, tmp_path):
    """Pairing is an intersection, and an empty one must stop the run rather than resample it.

    `control_dump` copies myna's own ids, so the only way to reach this branch is to rename them
    — which is also the shape of the real failure this catches: a control pass over a different
    split, or a dump whose `meta.id` came from another file. Twenty thousand replicates of an
    empty paired set print an interval around nothing.
    """
    dump = control_dump(fx.dump)
    for cr in dump["rows"]:
        cr["meta"] = dict(cr["meta"], id="vegaml-" + str(cr["meta"]["id"]))
    r = run_ctl(fx, tmp_path, dump, name="disjoint")
    assert r.rc == 3
    assert "share no row id" in r.console, r.console[-400:]
    assert not r.out.exists() and not (tmp_path / "disjoint.res.md").exists()


def test_engine_guard_refuses_a_foreign_dump(fx, tmp_path):
    """A laya dump in this slot would print laya's numbers under vegaml's name."""
    laya = control_dump(fx.dump)
    laya["engine"] = "laya"
    p = tmp_path / "foreign.json"
    p.write_text(json.dumps(laya))
    assert CTRL.main(["--myna", str(fx.dump_path), "--control", str(p), "--report",
                      str(fx.report_path), "--out", str(tmp_path / "o.json"),
                      "--md", str(tmp_path / "o.md")]) == 3
    assert not (tmp_path / "o.json").exists(), "a refused run must leave no witness behind"


def test_missing_input_is_exit_2(fx, tmp_path):
    assert CTRL.main(["--myna", str(tmp_path / "nope.json"), "--control",
                      str(tmp_path / "nope2.json"), "--report", str(fx.report_path)]) == 2


def test_a_subset_that_prices_no_kept_cell_is_refused_then_allowed(fx, tmp_path, ctl):
    """`--allow-subset` is a pairing waiver, and the two `trec` rows are the honest test.

    `min_rows` already keeps `trec/answer_type` out of the macro, so dropping those two rows
    shrinks the paired set without moving a single statistic — which leaves the pairing guard as
    the only thing standing between this run and a written witness. If it writes, it has to say
    how many rows left.
    """
    skip = tuple(r["meta"]["id"] for r in fx.dump["rows"] if r["meta"]["source"] == "trec")
    assert len(skip) == 2
    dump = control_dump(fx.dump, skip_ids=skip)
    refused = run_ctl(fx, tmp_path, dump, name="short")
    assert refused.rc == 3, "a paired set smaller than the split was allowed by default"
    assert refused.out.exists() is False, "a refused run must leave no witness behind"
    assert f"the paired set is {fx.dump['n_rows'] - 2} rows" in refused.console
    allowed = run_ctl(fx, tmp_path, dump, name="short", extra=["--allow-subset"])
    assert allowed.rc == 0, allowed.console
    assert allowed.res["bootstrap"]["n_rows"] == fx.dump["n_rows"] - 2
    assert sorted(allowed.res["dropped"]["rows"]) == sorted(skip)
    assert (f"2 rows are in the myna dump but not the control's (first {sorted(skip)[:3]}); "
            f"they leave the paired set") in allowed.console, allowed.console
    # and the statistics are untouched, because those two rows price no kept cell
    assert allowed.res["point"]["myna_macro"] == pytest.approx(
        fx.report["macro"]["acc"], abs=1e-12)
    assert allowed.res["paired_question_instances"] == ctl.res["paired_question_instances"], \
        "dropping the rows of a below-floor cell changed how many questions paired"


def test_allow_subset_waives_the_pairing_guard_and_not_the_statistic_guard(fx, tmp_path):
    """Losing a row that *does* feed a kept cell moves myna's own accuracy, and that is a second
    refusal with a different reason. One flag must not clear both.

    This is the guard that makes the wrapper's relabel safe: `vegaml_macro` sits in the same
    `statize()` output as myna's macro, and a pass that cannot reproduce myna's four numbers is
    reading a myna dump that changed underneath it — not a control finding.
    """
    victim = next(r["meta"]["id"] for r in fx.dump["rows"] if r["meta"]["source"] == "agnews")
    r = run_ctl(fx, tmp_path, control_dump(fx.dump, skip_ids=(victim,)), name="victim",
                extra=["--allow-subset"])
    assert r.rc == 3
    assert "the paired set is" not in r.console, "the flag waived the wrong guard"
    assert "myna_macro from these rows is" in r.console, r.console[-400:]
    assert not r.out.exists()


@pytest.mark.parametrize("block,field", [("macro", "acc"), ("macro", "majority"),
                                         ("macro", "uniform"), ("g1", "margin_observed")])
def test_myna_guard_fires_on_each_reported_statistic(fx, tmp_path, block, field):
    """The wrapper's 1e-9 check is on myna's own four numbers, and each one is checked.

    A control pass cannot move them: if it appears to, the myna dump changed under the
    instrument. Each of the four is doctored by 1e-6 — a slack a loosened threshold would swallow
    — and the harness has to refuse on all four, not only on the macro it prints first.
    """
    bad = json.loads(json.dumps(fx.report))
    bad[block][field] = bad[block][field] + 1e-6
    rp = tmp_path / f"bad.{block}.{field}.report.json"
    rp.write_text(json.dumps(bad))
    p = tmp_path / f"bad-{block}-{field}.rowoutcomes.json"
    p.write_text(json.dumps(control_dump(fx.dump)))
    rc = CTRL.main(["--myna", str(fx.dump_path), "--control", str(p), "--report", str(rp),
                    "--B", "500", "--seed", "3", "--min-rows", str(MIN_ROWS),
                    "--out", str(tmp_path / "o.json"), "--md", str(tmp_path / "o.md")])
    assert rc == 3, f"a {block}.{field} doctored by 1e-6 was accepted"
    assert not (tmp_path / "o.json").exists()


def test_dump_help_advertises_every_registry_flag():
    """`bench/reproduce.py` asks the file behind a row whether it takes that row's flags."""
    rc = subprocess.run([sys.executable, str(REPO / "bench" / "eval_vegaml_row_outcomes.py"),
                         "--help"], capture_output=True, text=True, cwd=str(REPO))
    assert rc.returncode == 0, rc.stderr[-400:]
    for flag in ("--out", "--raw", "--from-raw", "--limit", "--probe-out", "--device",
                 "--threads", "--mode", "--views", "--alpha", "--myna-dump", "--repo"):
        assert flag in rc.stdout, flag


def test_the_raw_sidecar_writes_every_field_from_raw_reads():
    """The two halves of crash-resume may not be edited apart — measured, not reasoned.

    The first full pass died at `KeyError: 'warnings'` after 888 s of someone else's model: the
    aggregation step read a field the sidecar writer never wrote, and `seen_n` was bound on one
    branch only. Both are the same defect class — a `--from-raw` recovery path that only works on
    a sidecar written by a *different* version of the file — and neither shows up in any unit test
    because no cheap test can run 1,176 inferences. So the contract is read off the source.
    """
    src = (REPO / "bench" / "eval_vegaml_row_outcomes.py").read_text()
    reads = set(re.findall(r'\braw\["([a-z_]+)"\]', src))
    assert reads, "the from-raw branch no longer reads the sidecar by name"
    start = src.index('raw_path.write_text(json.dumps({')
    block = src[start:src.index('"rows": rows', start) + len('"rows": rows')]
    written = set(re.findall(r'"([a-z_]+)":', block))
    assert reads <= written, f"--from-raw reads {sorted(reads - written)}, the sidecar writes none"


def test_from_raw_and_a_live_pass_bind_the_same_names():
    """Every name the shared aggregation tail uses is assigned on *both* branches.

    `n_rows_in_file` came from `seen_n`, which only the `--from-raw` branch set, and the live pass
    crashed at the payload after paying for its rows. A syntax check cannot see it — Python
    resolves a local at run time — so this walks the two branch bodies of the `if args.from_raw:`
    statement and compares the names each one assigns.
    """
    import ast

    src = (REPO / "bench" / "eval_vegaml_row_outcomes.py").read_text()
    main = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    branch = next(n for n in ast.walk(main) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Attribute) and n.test.attr == "from_raw")

    def assigned(statements):
        out = set()
        for stmt in statements:
            for n in ast.walk(stmt):
                if isinstance(n, (ast.Assign, ast.AugAssign)):
                    targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                    for t in targets:
                        if isinstance(t, (ast.Tuple, ast.List)):
                            out |= {e.id for e in t.elts if isinstance(e, ast.Name)}
                        elif isinstance(t, ast.Name):
                            out.add(t.id)
        return out

    both = assigned(branch.body) & assigned(branch.orelse)
    need = {"rows", "errors", "pin", "warned", "seen_n", "n_tokens", "disagree",
            "inference_s", "cmd"}
    assert need <= both, (f"--from-raw binds {sorted(need - assigned(branch.body))} less, "
                          f"a live pass binds {sorted(need - assigned(branch.orelse))} less")


# --------------------------------------------------------------- two independent full passes
PASS1 = REPO / "runs" / "vegaml_0150_pass1.rowoutcomes.raw.json"
PASS2 = REPO / "runs" / "vegaml_0150_test.rowoutcomes.json"


def test_two_independent_passes_of_the_control_answer_identically():
    """The control arm is deterministic on this box, and that is a measured claim, not an assumption.

    Two full passes over the same 1,176 rows — the first paid for and then survived only as a
    crash-resume sidecar, the second written clean after both binding defects were fixed — are
    compared here field by field on every one of the 1,440 question outcomes: the type, the gold,
    the answer, `ok`, the confidence, the abstain flag, the conformal set size and `unbound`. They
    agree on all of it, and the cell table `cell_table()` builds from pass 1's rows equals the one
    pass 2 wrote.

    Why this is worth a test rather than a sentence: the paper quotes one number for the control,
    from one run. Either that run is repeatable or its interval is a description of a single
    sample of a stochastic engine, and `vegaml`'s answer carries a temperature-driven sampling path
    and a conformal set, so the question is live. Timing is the one thing deliberately *not*
    compared — pass 1 took 880.9 s of inference and pass 2 928.9 s, and a wall clock is not a
    measurement of the engine.

    The witness is a committed artifact (`bench/eval_vegaml_row_outcomes.py` writes it before
    aggregating, which is how it survived the first pass's crash), so this reads real answers from
    a real engine rather than a fixture.
    """
    a = json.loads(PASS1.read_text(encoding="utf-8"))
    b = json.loads(PASS2.read_text(encoding="utf-8"))
    assert a["engine"] == b["engine"] == "vegaml"
    assert a["vegaml"]["init_sha256"] == b["vegaml"]["init_sha256"] == "daaa84b1fa864909"
    assert (a["n_rows"], b["n_rows"]) == (1176, 1176)
    by_id = {r["meta"]["id"]: r for r in a["rows"]}
    assert set(by_id) == {r["meta"]["id"] for r in b["rows"]}
    fields = ("type", "ok", "gold", "pred", "conf", "abstain", "set_n", "unbound")
    diffs, n = [], 0
    for r in b["rows"]:
        other = by_id[r["meta"]["id"]]
        assert set(other["questions"]) == set(r["questions"]), r["meta"]["id"]
        for name, q in r["questions"].items():
            n += 1
            for f in fields:
                if q[f] != other["questions"][name][f]:
                    diffs.append((r["meta"]["id"], name, f, other["questions"][name][f], q[f]))
    assert n == 1440, f"only {n} outcomes compared — the dumps are not what this test reads"
    assert diffs == [], f"{len(diffs)} outcomes differ between the two passes: {diffs[:5]}"
    assert cell_table(a["rows"]) == b["cells"]
    ab = {"n": sum(1 for r in a["rows"] for q in r["questions"].values() if q["abstain"]),
          "rate": None}
    ab["rate"] = ab["n"] / n
    assert (ab["n"], b["abstain"]["n"]) == (631, 631)
    assert ab["rate"] == pytest.approx(b["abstain"]["rate"], abs=1e-12)
    assert a["argmax_disagreements"] == b["argmax_disagreements"] == 0

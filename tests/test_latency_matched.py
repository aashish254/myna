"""G2's harness has to be able to lie, so these tests are the lie detector.

`bench/bench_latency_matched.py` publishes a *ratio*. Three ways it can be wrong
without any of the timers noticing, all of them realistic:

* the two engines are handed different inputs, and the faster-looking one simply
  answered fewer questions (a skipped question is a free call);
* `laya` is silently truncating its state at `max_len` while myna reads all of
  it, so the row compares a short answer against a long one;
* the per-call cost fit is run over a grid that held one axis fixed, and a
  zero-variance column is collinear with the intercept — the solver then splits
  the two arbitrarily and prints a confident slope of 0.0 next to a *negative*
  fixed cost. That happened on the first draft of this harness, and it is the
  reason `fit_cost` refuses rather than rounding.

Everything here is pure: no model, no timing, no laya. The numbers are synthetic
and known in advance.
"""

import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.bench_latency_matched import (Q_CHOICE, Q_NOUL, fit_cost, fit_text,  # noqa: E402
                                         laya_tokens_per_row, main, markdown,  # noqa: E402
                                         matched_questions, non_degenerate, ratio_rows,  # noqa: E402
                                         require_full_answers, saturation_note, stats,  # noqa: E402
                                         state_text, table, timed)
from myna.engine import Myna  # noqa: E402


# ------------------------------------------------------------- the matched inputs
def test_the_question_ladder_alternates_like_layas_own_bench():
    qs = matched_questions(5)
    assert list(qs) == ["q0", "q1", "q2", "q3", "q4"]
    assert qs["q0"]["type"] == "choice" and qs["q1"]["type"] == "noul"
    assert matched_questions(0) == {}


def test_options_per_question_agree_across_the_two_engines():
    """"Matched options/question" is a claim about two codebases, so it is tested
    against the expansion myna actually feeds its pointer head."""
    assert len(Myna._options(None, Q_CHOICE)) == 3
    assert len(Myna._options(None, Q_NOUL)) == 2
    assert len(Q_CHOICE["criteria"]) == 3          # what laya renders
    assert "criteria" not in Q_NOUL                 # its schema wants it absent


def test_both_engines_are_handed_the_very_same_objects():
    """Fairness by construction: one str, one dict, no per-engine copy that could
    drift. A mutation that rebuilt the dict per side must fail this."""
    text = state_text(3)
    qs = matched_questions(4)
    assert matched_questions(4) is not qs           # fresh dict per call, not shared state
    assert text.count("Stripe payouts") == 3
    for spec in qs.values():
        assert any(spec is o for o in (Q_CHOICE, Q_NOUL))   # identical objects, not equal copies


# ------------------------------------------------------------- the usage accounting
def test_laya_usage_is_a_sum_over_question_rows():
    assert laya_tokens_per_row(640, 10) == 64.0
    assert laya_tokens_per_row(72, 1) == 72.0
    assert laya_tokens_per_row(0, 0) == 0.0         # n=0 must not raise


def test_a_plateau_in_the_per_row_sequence_is_called_truncation():
    growing = saturation_note([80.0, 150.0], 1024)
    assert growing is None, "a state that still costs more is still being read"
    flat = saturation_note([150.0, 150.0], 1024)
    assert flat and "150.0 -> 150.0" in flat and "1024" in flat
    over = saturation_note([1024.0], 1024)
    assert over and ">= max_len 1024" in over


def test_shrinking_per_row_sequence_is_also_truncation():
    assert saturation_note([200.0, 180.0], 1024)


# ------------------------------------------------------------- the answer guard
def test_an_engine_that_answers_fewer_questions_fails_the_run():
    qs = matched_questions(3)
    with pytest.raises(AssertionError) as e:
        require_full_answers({"answers": {"q0": {"confidence": 0.6}}}, qs, "myna")
    assert "answered 1 of 3" in str(e.value) and "q1" in str(e.value)


def test_a_confidently_dead_head_is_not_a_measurement():
    qs = matched_questions(2)
    ans = {k: {"confidence": 1.0} for k in qs}
    with pytest.raises(AssertionError, match="only 0/1 confidences"):
        require_full_answers({"answers": ans}, qs, "laya")


def test_score_answers_are_judged_on_their_distribution():
    """`score` carries probabilities and no confidence field; reading only
    `confidence` would call every score answer degenerate."""
    assert non_degenerate({"a": {"type": "score", "probabilities": {"0": 0.7, "1": 0.3}}})
    assert not non_degenerate({"a": {"type": "score", "probabilities": {"0": 1.0, "1": 0.0}}})
    assert not non_degenerate({})


# ------------------------------------------------------------- the timers
def test_stats_describe_the_samples_it_was_given():
    s = stats([float(i) for i in range(1, 20)])
    assert s["n"] == 19 and s["p50_ms"] == 10.0 and s["min_ms"] == 1.0
    assert s["mean_ms"] == 10.0 and s["p95_ms"] == 18.0
    assert stats([5.0, 1.0, 9.0, 3.0])["p50_ms"] == 5.0   # order must not matter


def test_warmup_runs_are_not_timed_but_do_run():
    calls = []

    def fn():
        calls.append(1)
        return len(calls)

    st, out = timed(fn, warmup=3, reps=4)
    assert len(calls) == 7 and st["n"] == 4 and out == 7


# ------------------------------------------------------------- the cost fit
def test_fit_recovers_coefficients_it_was_given():
    rows = [(L, Q, 5.0 + 0.2 * L + 3.0 * Q) for L in (64, 256, 512, 1024) for Q in (1, 10)]
    f = fit_cost(rows)
    assert f["identifiable"] and f["terms"] == ["fixed", "L", "Q"]
    assert math.isclose(f["fixed_ms"], 5.0, abs_tol=1e-6)
    assert math.isclose(f["per_state_token_us"], 200.0, abs_tol=1e-3)
    assert math.isclose(f["per_question_ms"], 3.0, abs_tol=1e-6)
    assert f["r2"] == pytest.approx(1.0, abs=1e-6)


def test_a_held_constant_axis_is_dropped_not_fitted():
    """The first draft fitted the state scan with Q fixed at 1: the Q column was
    all ones, collinear with the intercept, and the answer came out as a fixed
    cost of -44.9 ms with 0.0 ms per question."""
    rows = [(L, 1, 4.0 + 0.5 * L) for L in (64, 128, 256, 512, 1024)]
    f = fit_cost(rows)
    assert f["terms"] == ["fixed", "L"]
    assert math.isclose(f["per_question_ms"], 0.0, abs_tol=1e-9)
    assert f["fixed_ms"] == pytest.approx(4.0, abs=1e-6)
    assert "note" not in f


def test_too_few_rows_for_the_terms_is_refused_with_a_reason():
    f = fit_cost([(64, 1, 10.0), (128, 10, 40.0), (256, 1, 60.0)])
    assert not f["identifiable"] and "3 rows cannot pin 3 coefficients" in f["reason"]
    assert "fixed_ms" not in f
    assert "not identifiable" in fit_text("myna_ask", f)


def test_a_negative_intercept_is_flagged_rather_than_published_plain():
    f = fit_cost([(L, 1, 0.01 * L * L) for L in (64, 128, 256, 512, 1024, 2048)])
    assert f["identifiable"] and f["fixed_ms"] < 0
    assert "super-linearly" in f["note"] and "super-linearly" in fit_text("x", f)


def test_fit_reports_which_axes_it_could_attribute():
    rows = [(L, Q, 1.0 + 0.1 * L) for L in (64, 128, 256, 512, 1024) for Q in (1,)]
    assert fit_cost(rows)["terms"] == ["fixed", "L"]


def test_a_single_state_length_still_yields_a_honest_per_question_slope():
    """The mirror image of the case above: `--ladder 512` varies only the question
    count, so the state axis is the constant one and must be dropped. Both axes
    need the guard; a fit that drops only one of them is half a fix."""
    rows = [(512.0, Q, 8.0 + 2.0 * Q) for Q in (1, 5, 10, 20, 50)]
    f = fit_cost(rows)
    assert f["terms"] == ["fixed", "Q"]
    assert f["per_state_token_us"] == 0.0
    assert f["per_question_ms"] == pytest.approx(2.0, abs=1e-6)
    assert f["fixed_ms"] == pytest.approx(8.0, abs=1e-6)
    assert "note" not in f


# ------------------------------------------------------------- the published table
def test_the_ratio_is_computed_from_the_p50s_and_nothing_else():
    laya = {1: {"p50_ms": 40.0}, 10: {"p50_ms": 160.0}}
    e2e = {1: {"p50_ms": 10.0}, 10: {"p50_ms": 40.0}}
    ask = {1: {"p50_ms": 5.0}, 10: {"p50_ms": 20.0}}
    rows = ratio_rows(laya, e2e, ask)
    assert [r["speedup_e2e"] for r in rows] == [4.0, 4.0]
    assert [r["speedup_stream"] for r in rows] == [8.0, 8.0]
    assert rows[0]["laya_system_one_p50_ms"] == 40.0


def test_table_renders_every_column_of_every_row():
    t = table([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], "t")
    assert t.splitlines()[1] == "| a | b |"
    assert "| 2 | y |" in t
    assert "(no rows)" in table([], "t")


def _meta(**over):
    m = {"platform": "Darwin", "device": "cpu", "ns": [1], "myna_state_tokens": 87,
         "intra_threads": 8, "inter_threads": 1, "warmup": 3, "reps": 10,
         "myna_ckpt": "runs/myna-v0", "myna_params": 14_450_000,
         "myna_dtype": "torch.float32",
         "laya_version": "0.3.20", "laya_ckpt": "/x", "laya_params": 421_000_000,
         "laya_max_len": 1024, "laya_head_max_len": 256,
         "laya_dtype": "torch.float16", "laya_amp_enabled": False,
         "laya_autocast": "torch.float32"}
    m.update(over)
    return m


def test_the_markdown_names_both_models_and_their_windows():
    res = {"meta": _meta(), "ratio": ratio_rows({1: {"p50_ms": 40.0}}, {1: {"p50_ms": 10.0}},
                                                {1: {"p50_ms": 5.0}}),
           "ladder": [], "fit": {"myna_e2e": {"identifiable": True, "fixed_ms": 5.0,
                                              "per_state_token_us": 200.0, "per_question_ms": 3.0,
                                              "r2": 0.99, "n_rows": 10,
                                              "terms": ["fixed", "L", "Q"]}}}
    md = markdown(res)
    assert "421.00M params" in md and "max_len 1024" in md
    assert "14.45M params" in md
    # "matched dtype" may only be written from the dtypes the run reported
    assert "weights torch.float32" in md and "weights torch.float16" in md
    assert "autocast off" in md
    assert "**200.0 µs**" in md and "R² 0.99" in md
    assert "Accuracy is not measured here" in md


def test_an_untruncated_row_is_blank_not_none():
    """"None" in a published column reads like a value the harness measured."""
    res = {"meta": _meta(), "ratio": None,
           "ladder": [{"state_tokens_myna": 66, "questions": 1, "laya_p50_ms": 115.4,
                       "laya_truncated": ""}],
           "fit": {"myna_ask": fit_cost([(L, 1, 4.0 + 0.5 * L)
                                         for L in (64, 128, 256, 512, 1024)])}}
    md = markdown(res)
    assert "| None |" not in md and "66" in md and "115.4" in md


def test_a_run_without_laya_withdraws_the_ratio_instead_of_projecting_it():
    res = {"meta": _meta(laya_version=None, laya_ckpt=None, laya_params=None,
                         laya_max_len=None, laya_head_max_len=None),
           "ratio": None, "ladder": [],
           "fit": {"myna_ask": {"identifiable": False, "n_rows": 2, "n_params": 3,
                                "terms": ["fixed", "L", "Q"], "reason": "2 rows cannot pin 3"}}}
    md = markdown(res)
    assert "not measured in this run" in md and "ratio is reported" in md
    assert "not identifiable" in md
    assert "p50, same box" not in md


# ------------------------------------------------------------- the CLI surface
def test_a_laya_less_run_writes_ratio_null_and_exits_zero(tmp_path):
    """The small end-to-end smoke: `--no-laya` must still produce both artifacts,
    with `"ratio": null` rather than a myna-only number dressed as a comparison."""
    import json
    out, md = tmp_path / "l.json", tmp_path / "l.md"
    assert main(["--no-laya", "--ns", "1", "--ladder", "64", "--reps", "1",
                 "--warmup", "0", "--state-reps", "1", "--out", str(out), "--md", str(md)]) == 0
    res = json.loads(out.read_text())
    assert res["ratio"] is None and res["meta"]["laya_version"] is None
    assert res["meta"]["inference_only"] is True
    assert "not measured" in md.read_text()


def test_a_run_records_the_box_and_its_own_command(tmp_path):
    """§9.23: a millisecond is only restatable with the load beside it, and the
    two committed P4 runs predate that convention (SPEC §9.30), so the harness
    now writes both itself rather than relying on whoever typed the table."""
    import json
    out, md = tmp_path / "l.json", tmp_path / "l.md"
    assert main(["--no-laya", "--ns", "1", "--ladder", "64", "--reps", "1",
                 "--warmup", "0", "--state-reps", "1", "--out", str(out), "--md", str(md)]) == 0
    m = json.loads(out.read_text())["meta"]
    assert len(m["load_avg_after"]) == 3 and all(
        isinstance(x, float) and x >= 0 for x in m["load_avg_after"])
    # a stubbed `[0.0, 0.0, 0.0]` is the failure this guards against, so compare
    # against the box rather than against zero: the 1-minute average still
    # carries the run that just finished
    live = os.getloadavg()
    assert all(abs(r - l) < 0.9 for r, l in zip(m["load_avg_after"], live)), \
        f"recorded {m['load_avg_after']} is not what this box reports ({live})"
    assert "--no-laya" in m["cmd"], "the command that made the numbers is not recorded"
    text = md.read_text()
    assert "load average after the run" in text and "--no-laya" in text


def test_help_documents_the_matched_axes():
    import subprocess
    r = subprocess.run([sys.executable, "-m", "bench.bench_latency_matched", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    for flag in ("--intra-threads", "--ladder", "--no-laya", "--state-reps"):
        assert flag in r.stdout


# --------------------------------------------------- one process, enforced not assumed
def test_a_second_copy_cannot_share_the_output_path(tmp_path):
    """A duplicate run used to be possible: it is how a contaminated table gets
    published, and the output itself carries no sign of it."""
    from bench.bench_latency_matched import acquire_lock
    out = tmp_path / "l.json"
    holder = acquire_lock(out)
    assert holder is not None and (tmp_path / "l.json.lock").exists()
    with pytest.raises(SystemExit) as e:
        acquire_lock(out)
    assert "not a measurement" in str(e.value)
    holder.close()  # closing the descriptor releases the flock
    assert acquire_lock(out) is not None


def test_main_refuses_before_it_loads_a_model_when_the_lock_is_held(tmp_path):
    """The guard has to run before the thread pinning and before any timing, or
    the contended process still produces a table."""
    from bench.bench_latency_matched import acquire_lock
    out = tmp_path / "l.json"
    holder = acquire_lock(out)
    with pytest.raises(SystemExit):
        main(["--no-laya", "--ns", "1", "--ladder", "64", "--reps", "1", "--warmup", "0",
              "--state-reps", "1", "--out", str(out), "--md", str(tmp_path / "l.md")])
    assert not out.exists(), "the refused run wrote an artifact anyway"
    holder.close()

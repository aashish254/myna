"""The `--long-context` trainer's own discipline (SPEC §5 P3, §6; TODO 7b's preconditions).

`--long-context` trains the largest states in the repo, and until 7b's preconditions it
did so with no memory plan, no cadence, no resume, no stop rule and no `metrics.json` —
the loop saved a weights-only snapshot on the eval cadence and nothing else. A run of it
could be killed at 90% and leave no optimizer, no step counter and no record of the dose
that ran, which is exactly the failure SPEC §6 was written after. These are the claims
`tests/test_memory_plan.py` makes of the suite path, re-witnessed here rather than assumed
to travel, plus the two that exist only on this branch of the trainer: the gradient window
as the thing being sized, and the box's own swap as a stop signal.

Runs are at CPU scale (256-token states, batch 2) and in a subprocess, so what is checked
is the line the operator reads and the `metrics.json` the run leaves behind — not a
helper's return value. The gate is `bench/mutation_train_longctx.py`.
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from myna.train import (
    LONG_CTX_P95_SAMPLES,
    LONG_CTX_SAVE_CEILING,
    MI,
    STATE_MIB_PER_POSITION,
    parse_swapusage,
    rows_that_fit,
    swap_abort_reason,
)

LC_BASE = ["--long-context", "256", "--grad-tokens", "64", "--vocab", "128",
           "--device", "cpu", "--batch", "2", "--eval-every", "0"]


def _run_lc(tmp_path, extra, expect_zero=True):
    out = tmp_path / "out"
    cmd = [sys.executable, "-m", "myna.train", *LC_BASE, "--out", str(out),
           "--steps", "3"] + extra
    import myna

    env = {**os.environ, "PYTHONPATH": str(Path(myna.__file__).resolve().parent.parent)}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=Path.cwd())
    if expect_zero:
        assert r.returncode == 0, r.stderr[-3000:]
    return r, out


def _lc_args(tmp_path, **over):
    ns = argparse.Namespace(long_context=128, grad_tokens=64, batch=1, steps=3, lr=6e-4,
                            seed=0, out=str(tmp_path / "out"), vocab=128, config="v0",
                            init=None, resume=False, warm_start=None, save_every=2,
                            eval_every=0, mem_safety=0.6, free_gib=8.0, stop_factor=1e9)
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_a_long_context_run_leaves_the_ladder_its_inputs_and_its_dose(tmp_path):
    r, out = _run_lc(tmp_path, ["--save-every", "2"])
    for name in ("model.pt", "model_last.pt", "tokenizer.json", "metrics.json"):
        assert (out / name).exists(), f"{name} missing: the run trained and left nothing"
    m = json.loads((out / "metrics.json").read_text())
    assert m["steps_requested"] == 3 and m["updates_executed"] == 3 and m["last_step"] == 2
    assert m["stopped"] is None and m["resumed_from_step"] == 0
    assert m["long_context"] == 256 and m["grad_tokens"] == 64 and m["save_every"] == 2
    assert m["state_tokens_p95"] > 0 and m["stop_factor"] == 1.5 and m["seed"] == 0
    blob = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
    assert set(blob) == {"state_dict", "cfg", "temperature"} and blob["temperature"] == 1.0, \
        "the needle ladder loads model.pt, so it has to be the file the run writes"


def test_the_rolling_snapshot_lands_on_the_cadence_and_says_so(tmp_path):
    """A cadence nobody can see is a cadence nobody can mutation-check, so the
    rolling saves print, and the printed steps are the witness: with `--save-every 2`
    over 5 updates they are 1 and 3, and the run then guarantees step 4."""
    r, out = _run_lc(tmp_path, ["--steps", "5", "--save-every", "2"])
    steps = [int(ln.split(" at step ")[1].split(" ")[0])
             for ln in r.stdout.splitlines() if ln.startswith("snapshot:")]
    assert steps == [1, 3], steps
    blob = torch.load(out / "model_last.pt", map_location="cpu", weights_only=False)
    assert blob["step"] == 4, "the final save is the guaranteed one, so it ends on the last update"
    assert "optimizer" in blob and "scheduler" in blob


def test_the_long_context_default_cadence_is_the_25_the_rule_asks_for(tmp_path):
    _run_lc(tmp_path, ["--steps", "2"])
    m = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert m["save_every"] == 25 <= LONG_CTX_SAVE_CEILING, m["save_every"]


def test_a_cadence_above_the_ceiling_is_refused_rather_than_clamped(tmp_path):
    """Clamping would train a different run than the one asked for; the operator
    asking for a snapshot every 500 updates of a 4k-token job is a mistake worth a
    line of stdout."""
    r, _ = _run_lc(tmp_path / "bad", ["--steps", "2", "--save-every", "51"], expect_zero=False)
    assert r.returncode != 0
    assert "above the 50-update ceiling" in r.stdout + r.stderr, r.stderr[-800:]


def test_warm_start_is_refused_because_this_path_takes_its_weights_from_init(tmp_path):
    r, _ = _run_lc(tmp_path / "ws", ["--warm-start", str(tmp_path / "nope.pt")],
                   expect_zero=False)
    assert r.returncode != 0
    assert "--warm-start is not available with --long-context" in r.stdout + r.stderr


def test_resume_continues_the_long_context_loop_with_the_saved_schedule(tmp_path):
    r1, _ = _run_lc(tmp_path, ["--steps", "4", "--save-every", "1"])
    assert "resume:" not in r1.stdout
    r2, out2 = _run_lc(tmp_path, ["--steps", "6", "--save-every", "1", "--resume"])
    line = [ln for ln in r2.stdout.splitlines() if ln.startswith("resume:")][0]
    assert "step 3" in line and "continuing at 4 of 6" in line, line
    assert "not a bit-identical replay" in line
    m = json.loads((out2 / "metrics.json").read_text())
    assert m["resumed_from_step"] == 4 and m["last_step"] == 5
    assert m["updates_executed"] == 2, "the dose is what ran, not what was asked for"


def test_resume_carries_the_snapshot_optimizer_and_schedule_into_the_loop(tmp_path):
    """Every counter in `metrics.json` is the snapshot's, so a resume that loads the weights
    and leaves Adam's moments behind still prints a correct dose — it just continues as a
    fresh optimizer wearing the old step number. Two figures say which happened: the
    schedule's `T_max` is the one the *first* command built (4, not the 6 this one asks
    for), and the Adam step count is the restored updates plus the two that ran here."""
    _run_lc(tmp_path, ["--steps", "4", "--save-every", "1"])
    snap = torch.load(tmp_path / "out" / "model_last.pt", map_location="cpu", weights_only=False)
    assert snap["step"] == 3 and snap["scheduler"]["T_max"] == 4, snap["scheduler"]
    _run_lc(tmp_path, ["--steps", "6", "--save-every", "1", "--resume"])
    blob = torch.load(tmp_path / "out" / "model_last.pt", map_location="cpu", weights_only=False)
    assert blob["scheduler"]["T_max"] == 4, "T_max 6 would be a schedule built from scratch"
    adam = next(iter(blob["optimizer"]["state"].values()))
    assert int(adam["step"]) == 6, f"expected 4 restored + 2 executed, got {adam['step']}"


def test_resume_without_a_snapshot_fails_loud_here_too(tmp_path):
    r, _ = _run_lc(tmp_path / "fresh", ["--resume"], expect_zero=False)
    assert r.returncode != 0
    assert "no " in r.stdout + r.stderr and "model_last.pt" in r.stdout + r.stderr


def test_a_drifted_long_context_step_stops_exits_nonzero_and_leaves_the_snapshot(tmp_path):
    r, out = _run_lc(tmp_path, ["--steps", "25", "--stop-factor", "0.0001",
                                "--save-every", "0"], expect_zero=False)
    assert r.returncode != 0, "a truncated run must not exit 0"
    stop = [ln for ln in r.stdout.splitlines() if ln.startswith("STOP at step")]
    assert len(stop) == 1 and "snapshot written to" in stop[0], r.stdout[-1500:]
    assert "run ended at step" in r.stdout + r.stderr
    m = json.loads((out / "metrics.json").read_text())
    assert m["stopped"] and "step time" in m["stopped"]
    assert m["updates_executed"] == m["last_step"] + 1
    assert m["last_step"] < 24, "the stop must actually cut the run short"


def test_swap_under_the_ceiling_is_not_an_abort_and_over_it_is():
    assert parse_swapusage("total = 7168.00M  used = 1024.00M  free = 6144.00M  (encrypted)") \
        == 1024 * MI
    assert parse_swapusage("used =  42.50G") == int(42.5 * 1024 * MI)
    assert parse_swapusage("total = 0.00M  free = 0.00M") is None, "no used figure, no claim"
    assert parse_swapusage("") is None
    assert swap_abort_reason(1024 * MI) is None
    assert swap_abort_reason(0) is None
    assert swap_abort_reason(None) is None, "a box that cannot answer is not a box that paged"
    assert swap_abort_reason(int(4.0 * 1024 * MI)) is None, "the ceiling is inclusive"
    reason = swap_abort_reason(int(4.5 * 1024 * MI))
    assert reason and "4.5 GiB > 4.0 GiB" in reason, reason


def test_the_swap_probe_stays_quiet_off_macos(monkeypatch):
    import myna.train as t

    monkeypatch.setattr(sys, "platform", "linux")
    assert t.swap_used_bytes() is None, "a Linux run must not read a Mac-only budget"


def test_the_loop_asks_the_swap_probe_and_stops_on_its_answer(tmp_path, monkeypatch):
    """The wiring, not the arithmetic: a box reporting 5 GiB of swap ends the run,
    writes `stopped`, and exits non-zero, because an abort rule nobody calls is a
    comment. In-process, because no subprocess can be handed a fake sysctl."""
    import myna.train as t

    monkeypatch.setattr(t, "swap_used_bytes", lambda: int(5 * 1024 * MI))
    args = _lc_args(tmp_path)
    with pytest.raises(SystemExit, match="swap used"):
        t.train_long_context(args, random.Random(args.seed), "cpu")
    m = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert m["stopped"] and "swap used 5.0 GiB" in m["stopped"], m["stopped"]
    assert m["updates_executed"] == 1 and m["last_step"] == 0


def test_the_long_context_plan_is_sized_by_grad_tokens_not_the_whole_state(tmp_path):
    """The point of truncated backprop is that the prefix keeps no backward
    activations, so pricing the *state* would halve the batch the box can carry —
    and this headroom is the arithmetic that says which axis the code used."""
    gib = "0.2083"
    r, out = _run_lc(tmp_path, ["--steps", "1", "--free-gib", gib])
    m = json.loads((out / "metrics.json").read_text())
    window, p95 = m["grad_window_tokens"], m["state_tokens_p95"]
    assert p95 > 2 * window, f"the fixture does not separate the axes: p95 {p95}, window {window}"
    free = int(float(gib) * 1024 * MI)
    assert rows_that_fit(free, window, 0.6) == 2, "window-sizing carries this batch"
    assert rows_that_fit(free, p95, 0.6) == 1, "state-sizing would not"
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("memory plan:")][0]
    assert "fits" in line and "exceeds" not in line, line
    assert m["batch"] == 2 == m["batch_requested"]
    assert abs(m["mem_plan_free_gib"] - float(gib)) < 1e-3


def test_an_over_large_batch_is_clamped_to_the_window_plan_and_says_so(tmp_path):
    gib = "0.2083"
    r, out = _run_lc(tmp_path, ["--steps", "1", "--batch", "3", "--free-gib", gib])
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("memory plan:")][0]
    assert "exceeds it, using 2" in line, line
    m = json.loads((out / "metrics.json").read_text())
    assert m["batch"] == 2 and m["batch_requested"] == 3, \
        "the printed plan, the batch used and the record must be one number"


def test_no_headroom_reading_on_this_path_says_so_instead_of_guessing(tmp_path):
    r, out = _run_lc(tmp_path, ["--steps", "1"])
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("memory plan:")][0]
    assert "no headroom reading for cpu" in line and "--free-gib" in line, line
    m = json.loads((out / "metrics.json").read_text())
    assert m["mem_plan_free_gib"] is None and m["batch"] == 2


def test_a_safety_factor_that_buys_no_window_is_refused(tmp_path):
    r, _ = _run_lc(tmp_path / "zero", ["--steps", "1", "--free-gib", "1", "--mem-safety", "0"],
                   expect_zero=False)
    assert r.returncode != 0
    assert "gradient-window tokens per row" in r.stdout + r.stderr, r.stderr[-800:]


def test_the_plan_prints_both_axes_and_matches_its_own_record(tmp_path):
    r, out = _run_lc(tmp_path, ["--steps", "1"])
    m = json.loads((out / "metrics.json").read_text())
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("long-context plan:")][0]
    assert (f"{m['detached_tokens_p95']} detached scan + {m['grad_window_tokens']} "
            f"gradient-carrying") in line, line
    assert f"split at {m['split']}" in line
    assert f"{m['projected_step_gib']:.1f} GiB retained per step" in line
    assert m["grad_window_tokens"] == min(m["grad_tokens"], m["state_tokens_p95"])
    assert m["detached_tokens_p95"] == max(0, m["state_tokens_p95"] - m["grad_window_tokens"])
    assert m["projected_step_gib"] == m["batch"] * m["grad_window_tokens"] * \
        STATE_MIB_PER_POSITION / 1024


def test_the_long_context_p95_is_a_property_of_the_generator_not_the_seed(tmp_path):
    """Same claim as the suite path's, for the same reason: a replication that
    changed the batch would be a different experiment wearing this one's name."""
    _run_lc(tmp_path / "s0", ["--steps", "1", "--seed", "0"])
    _run_lc(tmp_path / "s1", ["--steps", "1", "--seed", "1"])
    a = json.loads((tmp_path / "s0" / "out" / "metrics.json").read_text())
    b = json.loads((tmp_path / "s1" / "out" / "metrics.json").read_text())
    assert a["state_tokens_p95"] == b["state_tokens_p95"], (a["state_tokens_p95"],
                                                            b["state_tokens_p95"])


def test_the_recorded_p95_is_a_tail_and_not_one_lucky_sample(tmp_path):
    """`p95_sample` records `len(probe)` rather than the constant, and that is the whole
    witness here: this generator pads every state to the same token count (measured — min
    == max == 266 over the probe at a 256-token target), so the *number* cannot tell a
    sixty-state sample from a one-state one. The count of states drawn can, and a plan
    priced off one lucky state is the thing §6 says not to pass off as a measurement."""
    _run_lc(tmp_path, ["--steps", "1"])
    m = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert m["p95_sample"] == LONG_CTX_P95_SAMPLES, m["p95_sample"]
    assert m["p95_sample"] >= 30, "a p95 below 30 rows is not a tail anyone should size to"
    assert m["state_tokens_min"] <= m["state_tokens_p95"] <= m["state_tokens_max"], m


def test_long_context_scaled_reaches_the_config_that_exists(tmp_path):
    """`--config scaled` named a symbol that was never imported on this path, so it
    raised NameError instead of the refusal it was supposed to be able to print."""
    r, out = _run_lc(tmp_path, ["--steps", "1", "--batch", "1", "--config", "scaled",
                                "--long-context", "128", "--save-every", "0"])
    m = json.loads((out / "metrics.json").read_text())
    assert m["params"] > 13_830_272, f"scaled did not build the bigger trunk: {m['params']}"

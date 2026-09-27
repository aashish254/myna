"""The trainer's memory plan, stop rule and checkpoint cadence (SPEC §5 P3: 3d-3f).

Why a test file and not a careful read: every number here is a *budget*, and a
budget that is wrong in the optimistic direction fails silently — the run just
dies later, on Kaggle, hours in. §6's killed M5 job is the precedent: it was
sized from a projected rate rather than a measured one. So each of the four
claims below is witnessed against the code that makes it, and the three that
print something are additionally run in a subprocess so the line and
`metrics.json` are checked rather than paraphrased:

* sizing comes from `rows_that_fit` on a measured p95, and an over-large
  `--batch` is **clamped** rather than attempted;
* a step time that drifts above `--stop-factor` x its own median stops the loop,
  writes a snapshot, and exits **non-zero** (a truncated run that exits 0 reads
  as a finished one);
* `--resume` continues from the snapshot's step with the optimizer and scheduler,
  so the cosine decay is not restarted;
* with no headroom reading (CPU/MPS) the plan says so instead of inventing one.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from myna.data import Example, Question
from myna.model import MynaConfig, MynaModel
from myna.tokenizer import train_tokenizer
from myna.train import (
    MI,
    QUESTION_MIB_PER_CELL,
    STATE_MIB_PER_POSITION,
    free_device_bytes,
    load_snapshot,
    median,
    rows_that_fit,
    save_snapshot,
    state_token_p95,
    stop_reason,
)

CFG = MynaConfig(vocab=128, d_model=32, n_layers=2, n_heads=2, d_k=8, d_v=8, d_ff=48, d_ptr=16)
CORE = "What is the topic of this article?"
QS = [Question("topic", "choice", CORE,
               ["world news", "sports", "business", "science and technology"])]


def _ex(state, label=0):
    return Example(state, "agnews#1", (label,))


# ---- the p95 that sizes everything -------------------------------------------

def test_p95_tracks_the_tail_not_the_mean():
    import random

    tok = train_tokenizer([q.instruction for q in QS] + [o for o in QS[0].options]
                          + ["a short one", "word " * 200], vocab_size=128)
    # 19 short rows + 3 long ones: the long share sits just above 5%, which is
    # exactly where a quantile is supposed to start costing
    short = [(QS, _ex("the ceasefire was signed in geneva"))] * 19
    long_ = [(QS, _ex("word " * 150))] * 3
    p95 = state_token_p95(tok, short + long_, rng=random.Random(0))
    plain = state_token_p95(tok, short, rng=random.Random(0))
    assert p95 > plain, "the batch pads to its longest row, so the tail is the cost"
    assert state_token_p95(tok, [], rng=random.Random(0)) == 0, "empty split must not crash"


def test_p95_samples_when_the_split_is_bigger_than_the_limit():
    """limit exists so startup stays fast on 57,904 rows; it must not silently
    size the run off the first N rows, which is the pool's own order."""
    import random

    tok = train_tokenizer([q.instruction for q in QS] + [o for o in QS[0].options]
                          + ["filler word"], vocab_size=128)
    items = [(QS, _ex(f"state number {i} filler word")) for i in range(120)]
    got = {state_token_p95(tok, items, limit=50, rng=random.Random(i)) for i in range(4)}
    assert len(got) == 1, f"a p95 over a sample must be stable: {got}"
    assert state_token_p95(tok, items, limit=50, rng=random.Random(0)) == \
        state_token_p95(tok, items, limit=0, rng=random.Random(0))


# ---- rows_that_fit: the arithmetic the clamp is made of -----------------------

def test_free_device_bytes_reads_cuda_headroom_not_total(tmp_path, monkeypatch):
    """The CUDA half is the one Kaggle uses, and this box has no CUDA, so it is
    witnessed against a patched driver call instead of skipped: a run that read
    *total* bytes instead of free would size the batch for memory the caching
    allocator does not have."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (8 * 1024 * MI, 24 * 1024 * MI))
    assert free_device_bytes("cuda") == 8 * 1024 * MI
    assert free_device_bytes("cpu") is None, "the device string, not the driver, decides"

    def boom():
        raise RuntimeError("no context")

    monkeypatch.setattr(torch.cuda, "mem_get_info", boom)
    assert free_device_bytes("cuda") is None, "an unreadable driver is a None, not a crash"


def test_rows_that_fit_is_the_documented_division():
    tokens, safety = 1000, 0.6
    free = 4 * 1024 * MI  # 4 GiB
    assert STATE_MIB_PER_POSITION == 0.8, (
        "the MiB-per-state-position figure is measured by bench/mem_profile.py; changing it "
        "means re-running that profile, not editing a test")
    # 4 GiB * 0.6 = 2457.6 MiB; a row costs 1000 * 0.8 = 800 MiB -> 3 rows
    assert rows_that_fit(free, tokens, safety) == int(free * safety / MI //
                                                      (tokens * STATE_MIB_PER_POSITION)) == 3
    assert rows_that_fit(free, tokens, safety=1.0) == 5
    assert rows_that_fit(free, tokens * 2, safety) == 1, "never zero: a batch of 1 is the floor"
    assert rows_that_fit(free, tokens * 100, safety) == 1, \
        "a row that cannot fit still returns 1 — the stop rule, not this, handles that"


def test_rows_that_fit_refuses_to_invent_a_budget():
    assert rows_that_fit(None, 1000) is None, "no headroom reading must not print a batch"
    assert rows_that_fit(0, 1000) is None
    assert rows_that_fit(1024 * MI, 0) is None
    assert rows_that_fit(1024 * MI, -5) is None


def test_the_question_branch_is_reserved_before_the_states_are_sized():
    """`--row-batch` keeps a second, separately measured allocation live in the same
    step (1.6 MiB per question cell). Sizing the batch as though the states were
    alone is the arithmetic that filled the M5, so the reserve comes out first —
    and if it takes the whole budget, the answer is 0, which the caller turns into
    a refusal rather than a batch of 1 that still cannot run."""
    free, tokens = 8 * 1024 * MI, 300  # 4800 MiB of usable budget at safety 0.6
    plain = rows_that_fit(free, tokens, 0.6)
    reserve = 1024 * QUESTION_MIB_PER_CELL * MI  # 1.6 GiB
    with_reserve = rows_that_fit(free, tokens, 0.6, reserve_bytes=reserve)
    assert plain == 20 and with_reserve == 13, \
        f"1.6 GiB out of a 4.8 GiB budget, at 240 MiB per row: {plain} -> {with_reserve}"
    assert plain - with_reserve == 7
    assert rows_that_fit(free, tokens, 0.6, reserve_bytes=free) == 0, "the whole budget is gone"
    assert rows_that_fit(free, tokens, 0.6, reserve_bytes=int(free * 0.6) + MI) == 0


def test_free_device_bytes_is_none_off_cuda_and_never_guessed():
    assert free_device_bytes("cpu") is None
    assert free_device_bytes("mps") is None, (
        "torch has no MPS headroom call; free system RAM is not one process's budget")
    if not torch.cuda.is_available():
        assert free_device_bytes("cuda") is None


def test_median_both_parity_cases():
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([4.0, 1.0, 2.0, 3.0]) == 2.5


# ---- the stop rule -----------------------------------------------------------

def test_stop_needs_a_window_before_it_can_judge():
    fast = [0.1] * 20
    assert stop_reason(fast, None, 1000) is None, "20 samples is not yet 21"
    assert stop_reason(fast + [0.1], None, 1000) is None, "steady state must never stop"
    assert stop_reason([0.1] * 5 + [9.0], None, 1000) is None, \
        "five samples cannot condemn a sixth: without a window the first slow " \
        "step of a run stops it, which reads as a crash and is not the rule"


def test_stop_fires_on_time_drift_and_names_both_numbers():
    times = [0.1] * 20 + [0.5]
    r = stop_reason(times, None, 1000, factor=1.5)
    assert r is not None and "step time" in r and "median" in r and "last 20" in r
    # a factor of 5 calls the same spike normal, so the rule really is the factor
    assert stop_reason(times, None, 1000, factor=5.0) is None
    # and it really is the median: a window whose *largest* member is 0.4 must
    # still stop on 0.5, which a min/max/last comparison would get wrong
    assert stop_reason([0.1, 0.4] * 10 + [0.5], None, 1000, factor=1.5) is not None
    # no factor passed: 1.5 is the margin the Kaggle run uses unless a flag says
    # otherwise, so the function's own default has to be the same number as the
    # trainer's — a `factor=10.0` default here would make the whole rule decorative
    assert stop_reason([0.1] * 20 + [0.5], None, 0) is not None, "the default factor"
    assert stop_reason([1.0] * 20 + [1.4], None, 0) is None, "1.4x must be inside the margin"
    assert stop_reason([1.0] * 20 + [1.6], None, 0) is not None, "1.6x must be outside it"


def test_stop_uses_only_the_recent_window_so_recovery_is_not_penalised():
    """A single slow eval pass early in the run must not make every later step
    look fast-and-suspicious: the median is taken over the trailing window."""
    times = [9.0] + [0.1] * 21
    assert stop_reason(times, None, 1000, factor=1.5) is None
    # the converse: a job that warms *down* to 0.1s/update must still be judged
    # against 0.1s, so a 0.5s spike at update 41 is a real regression
    slowed = [9.0] * 20 + [0.1] * 20 + [0.5]
    assert stop_reason(slowed, None, 1000, factor=1.5) is not None, \
        "the window must be trailing, not the whole history"


def test_stop_fires_when_headroom_drops_under_the_projection():
    projected = 2000 * MI
    assert stop_reason([0.1], projected // 2, projected, factor=1.5) is not None
    assert "projected" in stop_reason([0.1], projected // 2, projected)
    # inside the 1.25x band is already too tight: a step that *just* fits is the
    # one the allocator then oversubscribes
    assert stop_reason([0.1], int(projected * 1.1), projected, factor=1.5) is not None
    assert stop_reason([0.1], int(projected * 1.25), projected, factor=1.5) is None, \
        "exactly the headroom factor is the boundary, not inside it"
    assert stop_reason([0.1], int(projected * 2), projected, factor=1.5) is None
    assert stop_reason([0.1], 10 * MI, 0, factor=1.5) is None, \
        "no projection means nothing to compare against; free RAM alone is not a reason"


def test_no_headroom_reading_disables_only_the_headroom_half():
    """CPU/MPS report None, which must not disable the drift check — the drift
    check is the one that would have caught §6's paging."""
    assert stop_reason([0.1] * 20, None, 0, factor=1.5) is None
    assert stop_reason([0.1] * 20 + [9.0], None, 0, factor=1.5) is not None


# ---- snapshot round-trip -----------------------------------------------------

def test_snapshot_carries_optimizer_scheduler_and_step(tmp_path):
    tok = train_tokenizer([q.instruction for q in QS] + [o for o in QS[0].options], vocab_size=128)
    model = MynaModel(CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    for g in opt.param_groups:
        g["lr"] = 3e-4
    sched.last_epoch = 5  # as if five cosine steps had already been taken
    # Adam moments, so the round-trip check below proves they survive the save
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    opt.step()
    path = save_snapshot(tmp_path, model, opt, sched, tok, CFG, step=17, temperature=1.37)
    assert path.exists() and (tmp_path / "tokenizer.json").exists()

    model2 = MynaModel(CFG)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda s: 1.0)
    step = load_snapshot(path, model2, opt2, sched2)
    assert step == 17, "the loop must start after the snapshot, not inside it"
    for k in model.state_dict():
        assert torch.equal(model.state_dict()[k], model2.state_dict()[k]), k
    assert [g["lr"] for g in opt2.param_groups] == [g["lr"] for g in opt.param_groups], \
        "optimizer state carries the schedule's current position"
    assert sched2.state_dict() == sched.state_dict(), "a resumed run must not restart the decay"
    assert opt2.state and opt.state, "Adam moments must survive or the resume restarts warm-up"
    assert load_snapshot(path, MynaModel(CFG)) == 17, "weights-only load must not need an opt"


# ---- end to end: the lines the operator sees ----------------------------------

TRAIN_ROWS = [
    {"state": {"document": s}, "questions": {"topic": {
        "type": "choice", "instructions": CORE,
        "criteria": {"world_news": "about the world", "sports": "about sport",
                     "business": None, "sci_tech": "science or tech"},
        "label": lab}}, "_meta": {"source": "agnews"}}
    for s, lab in [("the ceasefire was signed in geneva", "world_news"),
                   ("united beat city 3 nil at old trafford", "sports"),
                   ("the central bank raised its benchmark rate", "business"),
                   ("researchers built a faster transistor", "sci_tech")]
]
EVAL_ROWS = [
    {"state": {"document": s}, "questions": {"topic": {
        "type": "choice", "instructions": CORE,
        "criteria": {"world_news": "about the world", "sports": "about sport",
                     "business": None, "sci_tech": "science or tech"},
        "label": lab}}, "_meta": {"source": "agnews"}}
    for s, lab in [("peace talks resumed at the UN", "world_news"),
                   ("the striker scored a hat trick", "sports")]
]


def _write_suite(dirpath: Path):
    dirpath.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", TRAIN_ROWS), ("development", EVAL_ROWS),
                       ("test", EVAL_ROWS), ("calibration", EVAL_ROWS)):
        with (dirpath / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return dirpath


def _run(tmp_path, extra, expect_zero=True):
    suite = _write_suite(tmp_path / "suite")
    out = tmp_path / "out"
    cmd = [sys.executable, "-m", "myna.train", "--suite", str(suite), "--out", str(out),
           "--device", "cpu", "--steps", "2", "--batch", "2", "--vocab", "128",
           "--eval-every", "0", "--paraphrase", "off"] + extra
    import myna

    env = {**os.environ, "PYTHONPATH": str(Path(myna.__file__).resolve().parent.parent)}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=Path.cwd())
    if expect_zero:
        assert r.returncode == 0, r.stderr[-3000:]
    return r, out


def out_metrics(tmp_path):
    """`_run` always writes under `<tmp_path>/out`, so a test can read the record
    without rebuilding the path in three places."""
    return (Path(tmp_path) / "out" / "metrics.json").read_text()


def _p95_measured(tmp_path):
    """One run with plenty of stated headroom, to read the p95 the trainer itself
    computed. The sizing tests then ask for a *specific* fit and check the trainer
    arrives at it, instead of hoping the fixture lands on the right side of a
    boundary."""
    r, out = _run(tmp_path / "probe", ["--free-gib", "8", "--batch", "2", "--steps", "1"])
    m = json.loads((out / "metrics.json").read_text())
    assert "fits" in [ln for ln in r.stdout.splitlines() if "memory plan:" in ln][0]
    return m["state_tokens_p95"]


def _gib_for_fit(p95, fit, safety=0.6, reserve_mib=0.0):
    """Free bytes (GiB) whose plan yields exactly `fit` rows, and the batch just
    above it — so a clamp that only fires at 2x the fit is caught, not missed.
    The midpoint of the band is what makes this robust: the plan is `fit` for any
    headroom in [fit, fit+1) x row-cost, so the request sits half a row away from
    either boundary rather than on one."""
    cost_mib = p95 * STATE_MIB_PER_POSITION
    mid_mib = ((fit + 0.5) * cost_mib + reserve_mib) / safety
    assert rows_that_fit(int(mid_mib * MI), p95, safety,
                         reserve_bytes=int(reserve_mib * MI)) == fit, "helper agrees with the code"
    return mid_mib / 1024


def test_batch_is_clamped_to_the_plan_and_the_clamp_prints(tmp_path):
    p95 = _p95_measured(tmp_path)
    gib = _gib_for_fit(p95, 5)
    r, out = _run(tmp_path, ["--free-gib", f"{gib:.6f}", "--batch", "6", "--steps", "1"])
    line = [ln for ln in r.stdout.splitlines() if "memory plan:" in ln]
    assert len(line) == 1, r.stdout
    assert "exceeds it, using 5" in line[0], line[0]
    m = json.loads((out / "metrics.json").read_text())
    assert m["batch"] == 5, "the printed plan and the batch actually used must be one number"
    assert abs(m["mem_plan_free_gib"] - gib) < 1e-3 and m["mem_safety"] == 0.6
    assert m["state_tokens_p95"] == p95, "the plan must be built from the measured tail"


def test_a_batch_at_exactly_the_plan_is_taken_without_shrinking(tmp_path):
    """The boundary the clamp is defined by: batch == fit must not be reduced, and
    a plan one row larger must reduce it. Only these two together pin the compare."""
    p95 = _p95_measured(tmp_path)
    gib = _gib_for_fit(p95, 5)
    r, out = _run(tmp_path, ["--free-gib", f"{gib:.6f}", "--batch", "5", "--steps", "1"])
    line = [ln for ln in r.stdout.splitlines() if "memory plan:" in ln][0]
    assert "fits" in line and "exceeds" not in line, line
    assert json.loads((out / "metrics.json").read_text())["batch"] == 5


def test_the_safety_factor_changes_the_plan_by_the_factor():
    """The clamp is `--mem-safety` x free, so the plan must move when the flag
    does; a plan that divides by MI but forgets the factor is silently optimistic."""
    free, tokens = 8 * 1024 * MI, 300  # a row costs 240 MiB
    # 8192/240 = 34.1, 4096/240 = 17.1, 2048/240 = 8.5 — the floors, not roundings
    assert rows_that_fit(free, tokens, safety=1.0) == 34
    assert rows_that_fit(free, tokens, safety=0.5) == 17
    assert rows_that_fit(free, tokens, safety=0.25) == 8


def test_no_headroom_reading_says_so_instead_of_guessing(tmp_path):
    r, out = _run(tmp_path, ["--batch", "2", "--steps", "1"])
    line = [ln for ln in r.stdout.splitlines() if "memory plan:" in ln][0]
    assert "no headroom reading for cpu" in line and "--batch taken as given" in line
    m = json.loads((out / "metrics.json").read_text())
    assert m["mem_plan_free_gib"] is None and m["batch"] == 2, \
        "the metrics must record that the plan came from the flag, not a measurement"


def test_drift_stops_the_run_and_exits_nonzero(tmp_path):
    """--stop-factor 1e-4 makes the 21st update a violation by construction, so
    the check is witnessed rather than trusted; the window is what it waits for."""
    r, out = _run(tmp_path, ["--steps", "30", "--save-every", "0",
                             "--stop-factor", "0.0001"], expect_zero=False)
    assert r.returncode != 0, "a truncated run must not exit 0"
    stop_lines = [ln for ln in r.stdout.splitlines() if ln.startswith("STOP at step")]
    assert len(stop_lines) == 1, r.stdout[-2000:]
    assert "snapshot written to" in stop_lines[0]
    assert (out / "model_last.pt").exists(), "the stop must leave the snapshot behind"
    assert "run ended at step" in r.stderr + r.stdout
    m = json.loads((out / "metrics.json").read_text())
    assert m["stopped"] and "step time" in m["stopped"]
    assert m["last_step"] == int(stop_lines[0].split()[3].rstrip(":")), \
        f"metrics and the printed step disagree: {m['last_step']} vs {stop_lines[0]}"
    assert m["last_step"] < 29, "the stop must actually cut the run short"


def test_the_defaults_that_run_on_kaggle_are_in_the_record(tmp_path):
    """The two rules nobody passes a flag for are the two that decide whether a
    metered run survives: a default `--stop-factor` of 10 stops nothing, and a
    default `--save-every` of 10,000 loses the job. So the defaults must be visible
    in `metrics.json`, not only in `--help`."""
    _run(tmp_path, ["--steps", "2"])
    m = json.loads(out_metrics(tmp_path))
    assert m["stop_factor"] == 1.5 and m["save_every"] == 25
    _run(tmp_path / "override", ["--steps", "2", "--stop-factor", "3", "--save-every", "5"])
    m2 = json.loads(out_metrics(tmp_path / "override"))
    assert m2["stop_factor"] == 3.0 and m2["save_every"] == 5


def test_the_default_factor_lets_a_normal_run_finish(tmp_path):
    """The counterpart to the test above: without the artificial factor, 6 updates
    never trip the rule, so the stop is not a hair trigger."""
    r, out = _run(tmp_path, ["--steps", "6", "--save-every", "2"])
    assert "STOP at step" not in r.stdout
    m = json.loads((out / "metrics.json").read_text())
    assert m["stopped"] is None and m["last_step"] == 5


def test_resume_continues_after_the_snapshot_step(tmp_path):
    r1, out = _run(tmp_path, ["--steps", "4", "--save-every", "2"])
    assert "resume:" not in r1.stdout
    assert json.loads((out / "metrics.json").read_text())["resumed_from_step"] == 0
    r2, out2 = _run(tmp_path, ["--steps", "6", "--save-every", "2", "--resume"])
    line = [ln for ln in r2.stdout.splitlines() if ln.startswith("resume:")][0]
    assert "step 3" in line and "continuing at 4 of 6" in line, line
    assert "not a bit-identical replay" in line
    m = json.loads((out2 / "metrics.json").read_text())
    assert m["resumed_from_step"] == 4 and m["last_step"] == 5
    assert m["stopped"] is None


def test_resume_without_a_snapshot_fails_loud(tmp_path):
    """The check must be the trainer's own words: letting torch raise inside
    `load_snapshot` exits non-zero too, but prints a traceback that reads like a
    bug rather than like "you asked to continue a run that has no snapshot"."""
    r, _out = _run(tmp_path, ["--resume"], expect_zero=False)
    assert r.returncode != 0
    assert "to continue from" in r.stderr + r.stdout
    assert "Traceback" not in r.stderr, "an expected condition must not surface as a crash"


# ---- --warm-start: the weights carry, the schedule does not -----------------

def _weights(path):
    return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]


def test_warm_start_actually_moves_the_weights(tmp_path):
    """The discriminator a no-op flag cannot survive.

    Two cold runs of the same command must be bit-identical — that is the seeding
    `--seed` always promised and did not give. Given that, a warm-started run that
    still lands on the cold weights has loaded nothing, and one that lands
    elsewhere can only have got there from the checkpoint."""
    _r1, out = _run(tmp_path, ["--steps", "2", "--save-every", "2"])
    src = out / "model_last.pt"
    assert src.exists()
    _r2, warm = _run(tmp_path / "warm", ["--steps", "2", "--warm-start", str(src)])
    _r3, cold = _run(tmp_path / "cold", ["--steps", "2"])
    _r4, cold2 = _run(tmp_path / "cold2", ["--steps", "2"])
    w, c, c2 = (_weights(p / "model.pt") for p in (warm, cold, cold2))
    assert w.keys() == c.keys() == c2.keys()
    assert all(torch.equal(c[k], c2[k]) for k in c), \
        "two cold runs must be bit-identical or no A/B from this trainer means anything"
    assert any(not torch.equal(w[k], c[k]) for k in w), \
        "a warm start that lands on identical weights loaded nothing"


def test_warm_start_keeps_the_step_counter_and_the_optimizer(tmp_path):
    _r, out = _run(tmp_path, ["--steps", "2", "--save-every", "2"])
    r2, out2 = _run(tmp_path / "w", ["--steps", "3", "--warm-start", str(out / "model_last.pt")])
    line = [ln for ln in r2.stdout.splitlines() if ln.startswith("warm-start:")][0]
    assert "trained at step 1" in line, line          # the source really was opened
    assert "scheduler are fresh" in line
    assert "resume:" not in r2.stdout                 # not the other path
    m = json.loads((out2 / "metrics.json").read_text())
    assert m["resumed_from_step"] == 0 and m["last_step"] == 2, \
        "a warm start restarts the schedule, so it owns steps 0..2"


def test_warm_start_and_resume_together_is_refused(tmp_path):
    """They mean opposite things about the optimizer state, and whichever one won
    silently, a future reader of the log would not be able to tell what ran."""
    _r, out = _run(tmp_path, ["--steps", "2", "--save-every", "2"])
    r, _o = _run(tmp_path / "both",
                 ["--warm-start", str(out / "model_last.pt"), "--resume"],
                 expect_zero=False)
    assert r.returncode != 0
    text = r.stderr + r.stdout
    assert "different things" in text and "Traceback" not in text


def test_warm_start_with_a_missing_file_fails_loud(tmp_path):
    r, _o = _run(tmp_path, ["--warm-start", str(tmp_path / "nowhere.pt")], expect_zero=False)
    assert r.returncode != 0
    assert "no " in r.stderr + r.stdout and "Traceback" not in r.stderr


def test_row_batch_subtracts_the_question_branch_from_the_plan(tmp_path):
    """Same arithmetic as the state clamp, but the step holds a second allocation
    too, so a plan that ignores `--max-q-cells` over-sizes the batch by exactly the
    question branch. 128 cells x 1.6 MiB = 0.2 GiB comes out before rows are counted."""
    p95 = _p95_measured(tmp_path)
    reserve_mib = 128 * QUESTION_MIB_PER_CELL
    gib = _gib_for_fit(p95, 5, reserve_mib=reserve_mib)
    r, out = _run(tmp_path, ["--row-batch", "--max-q-cells", "128",
                             "--free-gib", f"{gib:.6f}", "--batch", "6", "--steps", "1"])
    line = [ln for ln in r.stdout.splitlines() if "memory plan:" in ln][0]
    assert "less 0.2 GiB of question branch" in line, line
    assert "exceeds it, using 5" in line, line
    assert json.loads(out_metrics(tmp_path))["batch"] == 5
    # without the reserve the same headroom would have been priced a row or two higher
    looser = rows_that_fit(int(gib * 1024 * MI), p95, 0.6)
    assert looser > 5, f"the reserve changed nothing: {looser} vs 5"


def test_a_question_branch_that_eats_the_budget_is_refused(tmp_path):
    """"Refuse to exceed it" has a second case: the branch alone does not fit. A
    batch of 1 printed at that point would still OOM on the first forward."""
    r, _out = _run(tmp_path, ["--row-batch", "--max-q-cells", "2048", "--free-gib", "4",
                              "--batch", "8", "--steps", "1"], expect_zero=False)
    both = r.stdout + r.stderr
    assert "memory plan:" in both and "question branch" in both, both[-800:]
    assert "--max-q-cells 2048" in both and "lower --max-q-cells" in both
    assert "not more than" in both, "the message must say which side of the arithmetic failed"


def test_the_kaggle_flag_set_runs_here_at_cpu_scale(tmp_path):
    """3a's local half: the exact command the Kaggle entrypoint will launch has to
    be proven to parse, run and record here, so the first time the flag set is
    exercised is not on a metered GPU. `--device cpu` is deliberate — the only
    difference on the box should be the device string."""
    r, out = _run(tmp_path, ["--row-batch", "--max-q-cells", "2048", "--accum-groups", "2",
                             "--group-sample", "uniform", "--paraphrase", "on",
                             "--free-gib", "10", "--mem-safety", "0.5", "--save-every", "1",
                             "--stop-factor", "1.5", "--warmup", "1", "--n-train", "4",
                             "--steps", "3", "--batch", "2", "--seed", "7"])
    assert "memory plan:" in r.stdout and "paraphrase:" in r.stdout
    m = json.loads((out / "metrics.json").read_text())
    assert m["batch"] >= 1 and m["stopped"] is None and m["last_step"] == 2
    assert m["paraphrase_draws"] > 0, "the row-batch path must consult the phrasing table"
    assert m["state_tokens_p95"] > 0 and m["mem_safety"] == 0.5
    assert (out / "model_last.pt").exists() and (out / "model.pt").exists()


def test_snapshot_lands_on_the_cadence_not_only_at_the_end(tmp_path):
    """3f is "cadence <= 50 updates", so prove a snapshot exists *mid*-run: with
    --save-every 2 over 5 steps the last one is written at step 3, and the run's
    final save writes only model.pt. If the cadence line were never reached,
    model_last.pt would be absent or hold step 4."""
    _run(tmp_path, ["--steps", "5", "--save-every", "2"])
    blob = torch.load(tmp_path / "out" / "model_last.pt", map_location="cpu",
                      weights_only=False)
    assert blob["step"] == 3, f"expected a mid-run snapshot at step 3, got {blob['step']}"
    assert "optimizer" in blob and "scheduler" in blob
    assert (tmp_path / "out" / "model.pt").exists()


def test_the_default_cadence_is_within_the_50_update_rule(tmp_path):
    """3f is a number, so it is witnessed by running with no `--save-every` at all
    and reading back the step the default left behind: 25 updates, so 26 steps end
    on a snapshot at step 24."""
    _run(tmp_path, ["--steps", "26", "--stop-factor", "1e9"])
    blob = torch.load(tmp_path / "out" / "model_last.pt", map_location="cpu",
                      weights_only=False)
    assert 0 < blob["step"] < 25, f"default cadence wrote step {blob['step']}"
    assert blob["step"] == 24

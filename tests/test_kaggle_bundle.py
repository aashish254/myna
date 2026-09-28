"""The Kaggle bundle: `kaggle/run.py`, its requirements, and the dataset packager.

Nothing here talks to Kaggle — the job runs on the user's credentials, and the
box is not this laptop (SPEC §6: the MacBook keeps the measurements, Kaggle gets
the training). What *is* checkable locally is the thing that actually breaks
cloud runs: the entrypoint that composes the command. So these tests run
`kaggle/run.py` for real, on CPU, against a tiny suite, and assert on what it
prints, writes and exits.

The rule each test protects:

* no `EXPERIMENT_NAME`, no run — an unnamed checkpoint is not evidence;
* the corpus is found or the failure names what was searched, never a silent
  fallback to the synthetic data;
* the command it builds carries the flags this loop measured (row-batch, the
  memory plan, the paraphrase table, the cadence), and `--resume` appears exactly
  when a snapshot exists *and still has a schedule left*: a snapshot whose
  `metrics.json` reached the requested step is a refusal that names `--warm-start`,
  because continuing a spent cosine decay trains at lr ~0 (§5 P9 9e). `--free-gib`
  reaches the plan only when the caller pins it, since the number is the box's.
* the trainer's non-zero stop exit reaches the notebook as non-zero;
* the staged dataset is byte-identical to the corpus the gates were sized on — the
  build itself fails on a lossy copy, a same-size edit is still a problem, and the
  manifest records the corpus rather than the stage — and packaging never uploads;
* the stage names the account that owns it, and never guesses: no owner, no stage.
  The constant that used to supply one named an account that is not the one holding
  the GPU quota, and the hand-edited metadata on disk is the evidence.
* the notebook cells the launch docs paste are *generated*, and the generator is held
  to the box: every flag it emits is one `run.py` accepts, every `/kaggle/...` path it
  emits is a shape the box has, its setup cell stages a mount made of symlinks, and the
  GPU-hours it prices cells at come from a committed log rather than from a guess.
"""

import ast
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from test_memory_plan import _write_suite  # noqa: E402  (tests/ is on pytest's sys.path)

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "kaggle" / "run.py"
PKG = REPO / "kaggle" / "package_dataset.py"
CAMPAIGN = REPO / "kaggle" / "campaign.py"
REQ = REPO / "kaggle" / "requirements.txt"
PILOT = REPO / "data" / "decision-v2-pilot"
# The one Kaggle run that completed, and the artifact its wall-clock price comes from.
V1B_LOG = REPO / "runs" / "v1b_kaggle_3600b.train.log"
SPLITS = ("train.jsonl", "development.jsonl", "test.jsonl", "calibration.jsonl")


def _run_py(argv, name="smoke", env_extra=None, expect_zero=True):
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    env.pop("EXPERIMENT_NAME", None)
    if name is not None:
        env["EXPERIMENT_NAME"] = name
    env.update(env_extra or {})
    r = subprocess.run([sys.executable, str(RUN), *argv], capture_output=True, text=True,
                       env=env, cwd=REPO)
    if expect_zero:
        assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    return r


# ---- the interpreter constraint, checked without needing a second interpreter --

def _files(*dirs):
    return [p for d in dirs for p in (REPO / d).rglob("*.py")]


def test_every_source_file_parses_under_python_311():
    """3c: the supported floor is Python 3.11 while this MacBook's venv is 3.13. Checked
    with `feature_version` rather than a version sniff, so CI on any interpreter catches
    3.12-only syntax (PEP 701 f-strings are the usual offender). What is deliberately *not*
    asserted anywhere in this file is the image's actual version: the one box that ran
    printed 3.12 in its session output, that output is not a committed witness, and the
    answer to "which Python is this box on" is `run.py --check`, which the generated setup
    cell runs."""
    bad = []
    for p in _files("src", "tests", "bench", "kaggle"):
        try:
            ast.parse(p.read_text(), filename=str(p), feature_version=(3, 11))
        except SyntaxError as e:
            bad.append(f"{p.relative_to(REPO)}: {e}")
    assert not bad, "not 3.11-compatible:\n" + "\n".join(bad)


def test_the_project_does_not_claim_it_needs_312_or_later():
    """>=3.13 in `pyproject.toml` is what made a rented box a silent fallback:
    uv would refuse the interpreter and the job would run on whatever was left."""
    text = (REPO / "pyproject.toml").read_text()
    line = [ln for ln in text.splitlines() if ln.startswith("requires-python")]
    assert len(line) == 1, line
    version = line[0].split("=", 1)[1].strip().strip('"')
    minor = int(version.split(">=")[1].split(".")[1])
    assert minor <= 11, f"{version} still excludes the 3.11 floor this package claims"


def test_requirements_pin_the_runtime_the_trainer_imports():
    text = REQ.read_text()
    for dep in ("torch", "numpy", "tokenizers", "pytest"):
        assert any(ln.strip().startswith(dep) for ln in text.splitlines()
                   if not ln.strip().startswith("#")), f"{dep} missing from requirements.txt"
    # the CUDA wheel must not be forced here: the Kaggle image already has one, and
    # a CPU wheel installed over it is how a GPU job quietly trains on CPU.
    assert "cu121" not in text and "+cpu" not in text


# ---- the entrypoint's contract ------------------------------------------------

def test_dry_run_prints_the_exact_command_and_nothing_else(tmp_path):
    suite = _write_suite(tmp_path / "suite")
    r = _run_py(["--corpus", str(suite), "--dry-run", "--out-root", str(tmp_path / "runs")])
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, r.stdout
    cmd = lines[0]
    assert "myna.train" in cmd and "--resume" not in cmd, "nothing to resume yet"
    for flag in ("--row-batch", "--paraphrase on", "--accum-groups 8", "--max-q-cells 2048",
                 "--group-sample pool", "--save-every 25", "--stop-factor 1.5",
                 "--mem-safety 0.6", "--eval-every 250", "--steps 10000", "--batch 32",
                 "--seed 0", "--vocab 8192", "--device auto"):
        assert flag in cmd, f"{flag} missing from the command:\n{cmd}"
    assert f"{tmp_path / 'runs' / 'smoke'}" in cmd
    assert not (tmp_path / "runs" / "smoke").exists(), "a dry run must not create the run"


def test_dry_run_launches_nothing_even_with_the_flags_a_real_run_needs(tmp_path):
    """Cheap on purpose: `--steps 1` means that if the dry-run short-circuit ever
    breaks, this test spends one update rather than ten thousand."""
    suite = _write_suite(tmp_path / "suite")
    root = tmp_path / "runs"
    r = _run_py(["--corpus", str(suite), "--dry-run", "--out-root", str(root),
                 "--steps", "1", "--batch", "1", "--vocab", "128"])
    assert len([ln for ln in r.stdout.splitlines() if ln.strip()]) == 1
    assert not root.exists(), f"--dry-run created {root}"


def test_resume_is_added_only_when_a_snapshot_exists(tmp_path):
    suite = _write_suite(tmp_path / "suite")
    out = tmp_path / "runs" / "smoke"
    out.mkdir(parents=True)
    cmd1 = _run_py(["--corpus", str(suite), "--dry-run", "--out-root", str(tmp_path / "runs")]).stdout
    assert "--resume" not in cmd1
    (out / "model_last.pt").write_bytes(b"x")  # any file: the flag is chosen on existence
    cmd2 = _run_py(["--corpus", str(suite), "--dry-run", "--out-root", str(tmp_path / "runs")]).stdout
    assert "--resume" in cmd2, "re-running the cell must continue the job, not restart it"


def test_the_free_gib_pin_reaches_the_plan_but_is_never_invented(tmp_path):
    """`--free-gib` is the T4 workaround (SPEC §6): the plan reads free bytes before
    the model is resident, so an unpinned plan over-batches. The entrypoint passes it
    through and does not supply a number, because the right number belongs to the box
    and a guessed one is exactly as loud as the bug it fixes."""
    suite = _write_suite(tmp_path / "suite")
    root = str(tmp_path / "runs")
    plain = _run_py(["--corpus", str(suite), "--dry-run", "--out-root", root]).stdout
    assert "--free-gib" not in plain
    pinned = _run_py(["--corpus", str(suite), "--dry-run", "--out-root", root,
                      "--free-gib", "9"]).stdout
    assert "--free-gib 9" in pinned, pinned


def test_a_finished_snapshot_is_not_resumed(tmp_path):
    """The snapshot file is the same whether the job was killed or completed, so this
    is the one place that can tell them apart — and resuming a finished cosine
    schedule trains at lr ~0 (SPEC §5 P9 9e)."""
    suite = _write_suite(tmp_path / "suite")
    out = tmp_path / "runs" / "smoke"
    out.mkdir(parents=True)
    (out / "model_last.pt").write_bytes(b"x")
    argv = ["--corpus", str(suite), "--dry-run", "--out-root", str(tmp_path / "runs"),
            "--steps", "3600"]
    (out / "metrics.json").write_text(json.dumps({"last_step": 750}))
    assert "--resume" in _run_py(argv).stdout, "a killed job is what --resume is for"
    (out / "metrics.json").write_text(json.dumps({"last_step": 3600}))
    r = _run_py(argv, expect_zero=False)
    msg = r.stdout + r.stderr
    assert r.returncode != 0, "a spent schedule must not be continued silently"
    assert "--warm-start" in msg and "lr ~0" in msg, msg
    assert "myna.train" not in msg, "the refusal composes nothing: `--resume` names itself"


def test_warm_start_is_a_flag_and_never_a_pair(tmp_path):
    suite = _write_suite(tmp_path / "suite")
    out = tmp_path / "runs" / "smoke"
    init = tmp_path / "init" / "model.pt"
    init.parent.mkdir(parents=True)
    init.write_bytes(b"x")
    argv = ["--corpus", str(suite), "--dry-run", "--out-root", str(tmp_path / "runs"),
            "--warm-start", str(init)]
    cmd = _run_py(argv).stdout
    assert f"--warm-start {init}" in cmd and "--resume" not in cmd, cmd
    out.mkdir(parents=True)
    (out / "model_last.pt").write_bytes(b"x")
    r = _run_py(argv, expect_zero=False)
    assert r.returncode != 0 and "--resume" in r.stdout + r.stderr, \
        "a snapshot plus --warm-start is the pair the trainer refuses"


def test_no_experiment_name_is_a_refusal(tmp_path):
    suite = _write_suite(tmp_path / "suite")
    r = subprocess.run([sys.executable, str(RUN), "--corpus", str(suite), "--dry-run",
                        "--out-root", str(tmp_path / "runs")],
                       capture_output=True, text=True, cwd=REPO,
                       env={k: v for k, v in os.environ.items()
                            if k not in ("EXPERIMENT_NAME", "PYTHONPATH")})
    assert r.returncode != 0
    assert "EXPERIMENT_NAME" in r.stderr + r.stdout


def test_a_half_mounted_corpus_fails_loud(tmp_path):
    suite = _write_suite(tmp_path / "suite")
    (suite / "calibration.jsonl").unlink()
    r = _run_py(["--corpus", str(suite), "--dry-run"], expect_zero=False)
    assert r.returncode != 0
    assert "calibration.jsonl" in r.stderr + r.stdout, "name the file that is missing"
    assert "synthetic" not in r.stderr, "must not fall back to generated data"


def test_the_runs_root_is_kaggle_working(tmp_path, monkeypatch):
    """`/kaggle/working` is the only path the platform persists, so it is the
    default; the flag is the escape hatch this loop uses to test the entrypoint."""
    sys.path.insert(0, str(REPO / "kaggle"))
    import run as krun

    assert krun.WORKING == Path("/kaggle/working"), \
        "a run written anywhere else is not saved when the notebook dies"
    monkeypatch.setattr(krun, "WORKING", tmp_path / "working")
    assert krun.run_dir("v1", None) == tmp_path / "working" / "runs" / "v1", \
        "run_dir must read the module constant, not a literal of its own"
    assert krun.run_dir("v2", str(tmp_path / "elsewhere")) == tmp_path / "elsewhere" / "v2", \
        "the name is a directory of its own: two runs must not share a checkpoint path"


def test_corpus_discovery_follows_a_symlinked_kaggle_mount(tmp_path, monkeypatch):
    """Kaggle mounts each dataset as a *symlink* under `/kaggle/input`, and
    `Path.glob("**")` does not descend into a symlinked directory. Measured on 3.9,
    3.11 and 3.13: pathlib found 0 files where `glob.glob(recursive=True)` found 2
    across one fixture. So the documented cell without `--corpus` would die on the
    box and work on every laptop — and it stayed invisible because every other test
    in this file passes `--corpus` explicitly, which is the branch that *does* work."""
    sys.path.insert(0, str(REPO / "kaggle"))
    import run as krun

    uploaded = _write_suite(tmp_path / "uploaded")
    owner = tmp_path / "input" / "datasets" / "someone"
    owner.mkdir(parents=True)
    mount = owner / "decision-v2-pilot"
    os.symlink(uploaded, mount)
    working = tmp_path / "empty-working"
    working.mkdir()
    monkeypatch.setattr(krun, "WORKING", working)
    monkeypatch.setattr(krun, "INPUT", tmp_path / "input")
    assert krun.find_corpus(None) == mount, \
        "a corpus behind a symlink is still the corpus: discovery must follow it"
    # and the failure still names what it searched, rather than inventing a path
    monkeypatch.setattr(krun, "INPUT", tmp_path / "nothing_mounted")
    with pytest.raises(SystemExit) as exc:
        krun.find_corpus(None)
    assert "Searched" in str(exc.value), "an unfound corpus has to say where it looked"


def test_the_entrypoint_actually_trains_and_records(tmp_path):
    """End to end, on CPU, two updates: the point is that the composed command
    runs, tees a log, and leaves a record a human can read next week."""
    suite = _write_suite(tmp_path / "suite")
    root = tmp_path / "runs"
    r = _run_py(["--corpus", str(suite), "--out-root", str(root), "--device", "cpu",
                 "--steps", "2", "--batch", "2", "--vocab", "128", "--paraphrase", "off",
                 "--save-every", "1"], name="kaggle-smoke")
    out = root / "kaggle-smoke"
    assert "memory plan:" in r.stdout and "run: kaggle-smoke" in r.stdout
    rec = json.loads((out / "run.json").read_text())
    assert rec["name"] == "kaggle-smoke" and rec["python"] == ".".join(map(str, sys.version_info[:3]))
    assert "--row-batch" in rec["command"] and rec["defaults"]["steps"] == 10_000
    assert "memory plan:" in (out / "train.log").read_text(), "the log must hold the run"
    m = json.loads((out / "metrics.json").read_text())
    assert m["last_step"] == 1 and m["stopped"] is None
    assert (out / "model.pt").exists() and (out / "tokenizer.json").exists()
    # and a snapshot the next cell could resume from, at the step it stopped on
    import torch

    blob = torch.load(out / "model_last.pt", map_location="cpu", weights_only=False)
    assert blob["step"] == 1 and "optimizer" in blob and "scheduler" in blob


def test_a_stopped_run_propagates_nonzero(tmp_path):
    suite = _write_suite(tmp_path / "suite")
    r = _run_py(["--corpus", str(suite), "--out-root", str(tmp_path / "runs"),
                 "--device", "cpu", "--steps", "30", "--batch", "2", "--vocab", "128",
                 "--stop-factor", "0.0001", "--save-every", "0"], expect_zero=False)
    assert r.returncode != 0
    both = r.stdout + r.stderr
    assert "STOP at step" in both and "training ended with code" in both, both[-800:]


def test_check_reports_the_interpreter_it_is_running_under(tmp_path):
    suite = _write_suite(tmp_path / "suite")
    r = _run_py(["--corpus", str(suite), "--check", "--device", "cpu",
                 "--out-root", str(tmp_path / "runs")])
    j = json.loads(r.stdout[r.stdout.index("{"):r.stdout.rindex("}") + 1])
    assert j["python"] == ".".join(map(str, sys.version_info[:3]))
    assert j["corpus"] == str(suite) and j["out"].endswith("/runs/smoke")
    assert j["myna"].endswith("src/myna"), "the code the entrypoint imports must be this repo's"


# ---- the notebook cell generator ----------------------------------------------

def _campaign():
    import importlib.util
    spec = importlib.util.spec_from_file_location("myna_campaign", str(CAMPAIGN))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _campaign_render(*argv):
    r = subprocess.run([sys.executable, str(CAMPAIGN), *argv], capture_output=True,
                       text=True, cwd=REPO)
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-1500:])
    return r.stdout


def _runner_flags(text):
    """The `--flags` on the generated `!python ... run.py` shell commands only. The
    cells' prose mentions this generator's own flags (`--owner`, `--include-dead`),
    and those are not the runner's contract."""
    flags, in_cmd = set(), False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("!python"):
            in_cmd = True
        if not in_cmd:
            continue
        for tok in shlex.split(s.rstrip(" \\").replace("!python", "python")):
            if tok.startswith("--"):
                flags.add(tok)
        if not s.endswith("\\"):
            in_cmd = False
    return flags


def test_every_flag_the_generator_emits_the_runner_accepts():
    """A cell naming a flag `kaggle/run.py` does not have dies in argparse, in front of
    a human who has already chosen a GPU. `--context-len` and `--calibrate` did exactly
    that in the version of this file the docs were generated from."""
    out = _campaign_render("--owner", "pilot", "--include-dead")
    help_text = subprocess.run([sys.executable, str(RUN), "--help"], capture_output=True,
                               text=True, cwd=REPO, check=True).stdout
    accepted = {t for t in re.findall(r"--[a-z][a-z0-9-]*", help_text)}
    emitted = _runner_flags(out)
    assert emitted, "the generator emitted no runner command at all"
    assert emitted <= accepted, f"run.py rejects: {sorted(emitted - accepted)}"


def test_no_generated_cell_names_a_path_the_box_does_not_have():
    """The two paths every hand-typed cell in this repo got wrong: a corpus mount with
    no `datasets/<owner>` in it, and a repo at `/kaggle/working/myna` that nothing
    creates. Checked as a rule about the paths rather than a list of forbidden strings:
    every `/kaggle/input/...` names the owner, every `/kaggle/working/...` names the
    directory the setup cell creates."""
    owner = "pilot"
    out = _campaign_render("--owner", owner, "--include-dead")
    for tok in re.findall(r"/kaggle/[a-zA-Z0-9_./-]+", out):
        if tok.startswith("/kaggle/input/"):
            assert tok.startswith(f"/kaggle/input/datasets/{owner}/"), tok
        if tok.startswith("/kaggle/working/"):
            # the two directories that exist there: the one the setup cell creates and
            # the one `run.py` writes. Anything else is the old bug's shape.
            assert tok.startswith(("/kaggle/working/myna", "/kaggle/working/runs")), tok
    assert "/kaggle/working/myna/kaggle/run.py" in out, "the cells must run the staged copy"
    assert "/kaggle/input/decision-v2-pilot" not in out
    assert "!python kaggle/run.py" not in out, "a relative path from an empty directory"


def test_the_setup_cell_stages_a_mount_that_is_all_symlinks(tmp_path, capsys):
    """The generator's own discovery, run against a fake Kaggle mount built the way
    Kaggle builds one: the dataset directories are symlinks. Nothing else in this file
    reaches the cell text — every other test hands `find_corpus` a `--corpus`."""
    camp = _campaign()
    content = tmp_path / "uploaded" / "pilot"
    (content / "myna-code" / "kaggle").mkdir(parents=True)
    (content / "myna-code" / "src" / "myna").mkdir(parents=True)
    (content / "myna-code" / "kaggle" / "run.py").write_text("# entrypoint\n")
    (content / "myna-code" / "src" / "myna" / "__init__.py").write_text("")
    (content / "decision-v2-pilot").mkdir()
    for split in SPLITS:
        (content / "decision-v2-pilot" / split).write_text("{}\n")
    input_root = tmp_path / "input" / "datasets" / "pilot"
    input_root.mkdir(parents=True)
    for name in ("myna-code", "decision-v2-pilot"):
        (input_root / name).symlink_to(content / name)
    staged = tmp_path / "working" / "myna"
    cell = camp.MOUNT_CELL.replace("@INPUT@", str(tmp_path / "input")).replace(
        "@STAGED@", str(staged))
    ns: dict = {}
    exec(compile(cell, "<setup cell>", "exec"), ns)
    printed = capsys.readouterr().out
    assert "staged from:" in printed and "no kaggle/run.py" not in printed, printed
    assert (staged / "kaggle" / "run.py").exists(), staged
    assert ns["corpora"] == [str(tmp_path / "input" / "datasets" / "pilot"
                                 / "decision-v2-pilot")], ns["corpora"]

    # and a mount with the code but no corpus is a refusal, not a silent empty run
    bare = tmp_path / "bare" / "datasets" / "pilot"
    bare.mkdir(parents=True)
    (bare / "myna-code").symlink_to(content / "myna-code")
    cell2 = camp.MOUNT_CELL.replace("@INPUT@", str(tmp_path / "bare")).replace(
        "@STAGED@", str(tmp_path / "working2" / "myna"))
    with pytest.raises(SystemExit) as e:
        exec(compile(cell2, "<setup cell>", "exec"), {})
    assert "corpus" in str(e.value), str(e.value)


def test_the_priced_cost_is_measured_and_its_witness_is_committed():
    """SPEC §9.30: a figure in prose without an artifact behind it is refused. The
    5.6 s/update the cells price themselves with comes from one committed log line, and
    a first draft of the launch docs said 1.5 h for the same 3600 steps."""
    camp = _campaign()
    log = V1B_LOG.read_text().splitlines()
    last = [ln for ln in log if ln.startswith("step  3599")]
    assert last, f"{V1B_LOG} has no final step line"
    assert "20225s" in last[-1], last[-1]
    assert camp.SECONDS_PER_STEP == pytest.approx(20225 / 3600)
    assert camp.hours(3600) == pytest.approx(5.62, abs=0.01)
    assert camp.hours(10_000) == pytest.approx(15.6, abs=0.05)


def test_the_measured_dead_ablation_is_not_in_the_default_lane():
    default = _campaign_render("--owner", "pilot")
    assert "--name extended_ce_s0" in default and "--name extended_ce_s1" in default
    assert "--name ablation" not in default
    assert "Not here (measured, not resolved)" in default
    dead = _campaign_render("--owner", "pilot", "--include-dead")
    for name in ("ablation_ce_s0", "ablation_ce_s1", "ablation_emd_s0", "ablation_emd_s1"):
        assert f"--name {name}" in dead, name
    assert "ordinal_ab.json" in dead, "the dead cells must carry the reason to the box"


def test_the_nb_route_writes_a_pushable_kernel_and_pushes_nothing(tmp_path):
    """Pasting is how the wrong paths got into the docs, so the generator can also
    write the kernel. The metadata is what `kaggle kernels push` reads: if it names the
    wrong account, mounts nothing, runs on CPU, or is public, the run is wrong before it
    starts — and `isPrivate` is the one that leaks a corpus."""
    out_dir = tmp_path / "kernel"
    r = subprocess.run([sys.executable, str(CAMPAIGN), "--nb", str(out_dir),
                        "--owner", "pilot", "--slug", "myna-campaign"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "Nothing pushed" in r.stdout, r.stdout
    meta = json.loads((out_dir / "kernel-metadata.json").read_text())
    assert meta["id"] == "pilot/myna-campaign" and meta["code_file"] == "myna-campaign.ipynb"
    assert meta["title"].lower().replace(" ", "-") == "myna-campaign", meta["title"]
    assert meta["is_private"] is True and meta["enable_gpu"] is True
    assert meta["enable_internet"] is True, "the setup cell pip-installs; internet OFF dies"
    assert meta["dataset_sources"] == ["pilot/myna-code", "pilot/decision-v2-pilot"]
    nb = json.loads((out_dir / meta["code_file"]).read_text())
    src = [c["source"] for c in nb["cells"]]
    assert "import os, shutil" in src[1] and "pip" in src[2]
    assert "--name extended_ce_s0" in src[-2] and "--name extended_ce_s1" in src[-1]
    for cell in (src[1], src[2]):
        ast.parse(cell)  # the setup cells are python; the experiment cells are shell
    for k in ("competition_sources", "kernel_sources", "model_sources"):
        assert meta[k] == [], f"{k}: mounting anything else is not this lane's design"

    no_owner = subprocess.run([sys.executable, str(CAMPAIGN), "--nb", str(tmp_path / "x")],
                              capture_output=True, text=True, cwd=REPO,
                              env={**os.environ, "KAGGLE_USERNAME": ""})
    assert no_owner.returncode != 0 and "KAGGLE_USERNAME" in no_owner.stderr + no_owner.stdout
    assert not (tmp_path / "x").exists(), "a kernel with no account is not written"


# ---- the dataset packager -----------------------------------------------------

def test_the_stage_names_the_account_and_never_guesses_it(tmp_path):
    """The owner was a module constant, and it named an account that is not the one
    the pilot was uploaded to — the hand-edited `dataset-metadata.json` under
    `.kaggle-dataset/` is the proof somebody worked around it after generation.
    `kaggle datasets create` rejects an owner that is not the authenticated user, so
    guessing is a failed upload, and a *silent* guess is a dataset in the wrong
    account. Synthesized source, no `data/` needed: this runs in a worktree too.
    """
    src = tmp_path / "corpus"
    _write_suite(src)
    (src / "tokenizer-8192.json").write_text("{}\n")
    dest = tmp_path / "staged"
    no_env = {k: v for k, v in os.environ.items() if k != "KAGGLE_USERNAME"}

    r = subprocess.run([sys.executable, str(PKG), "--dest", str(dest), "--source", str(src)],
                       capture_output=True, text=True, cwd=REPO, env=no_env)
    assert r.returncode != 0, "an unstaged owner must not produce a dataset"
    assert "KAGGLE_USERNAME" in r.stdout + r.stderr, "name the knob that supplies it"
    assert not dest.exists(), "a refused build leaves no half-made stage"

    b = subprocess.run([sys.executable, str(PKG), "--dest", str(dest), "--source", str(src),
                        "--owner", "quota-holder"], capture_output=True, text=True, cwd=REPO,
                       env=no_env)
    assert b.returncode == 0, b.stdout + b.stderr
    meta = json.loads((dest / "dataset-metadata.json").read_text())
    assert meta["ownerSlug"] == "quota-holder", meta
    assert meta["datasetSlug"] == "decision-v2-pilot", meta

    v = subprocess.run([sys.executable, str(PKG), "--dest", str(dest), "--source", str(src),
                        "--verify-only", "--owner", "someone-else"],
                       capture_output=True, text=True, cwd=REPO, env=no_env)
    assert v.returncode != 0 and "someone-else" in v.stdout, \
        "verify-only has to notice a stage that points at the wrong account"


@pytest.mark.skipif(not (PILOT / "train.jsonl").exists(), reason="pilot corpus not on disk")
def test_packaging_stages_the_pilot_byte_for_byte(tmp_path):
    dest = tmp_path / "staged"
    r = subprocess.run([sys.executable, str(PKG), "--dest", str(dest), "--owner", "pilot"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "hashes equal" in r.stdout
    assert "kaggle datasets create" in r.stdout, "print the upload command..."
    assert "uploaded" not in r.stdout.lower(), "...and do not run it"
    src_hashes = {n: json.loads((dest / "SOURCE_SHA256.json").read_text())["files"][n]["sha256"]
                  for n in ("train.jsonl", "development.jsonl", "test.jsonl",
                            "calibration.jsonl")}
    for name, sha in src_hashes.items():
        import hashlib
        h = hashlib.sha256((PILOT / name).read_bytes()).hexdigest()
        assert h == sha, f"{name} is not the corpus the gates were sized on"
    meta = json.loads((dest / "dataset-metadata.json").read_text())
    assert meta["datasetSlug"] == "decision-v2-pilot"
    assert {"train.jsonl", "calibration.jsonl"} <= {res["path"] for res in meta["resources"]}
    # re-verification of the staged copy is a separate, scriptable step
    v = subprocess.run([sys.executable, str(PKG), "--dest", str(dest), "--verify-only"],
                       capture_output=True, text=True, cwd=REPO)
    assert v.returncode == 0 and v.stdout.startswith("OK"), v.stdout


@pytest.mark.skipif(not (PILOT / "train.jsonl").exists(), reason="pilot corpus not on disk")
def test_packaging_notices_a_corrupt_stage(tmp_path):
    dest = tmp_path / "staged"
    assert subprocess.run([sys.executable, str(PKG), "--dest", str(dest), "--owner", "pilot"],
                          capture_output=True, text=True, cwd=REPO).returncode == 0
    with open(dest / "test.jsonl", "a") as f:
        f.write('{"state": {}, "questions": {}}\n')
    v = subprocess.run([sys.executable, str(PKG), "--dest", str(dest), "--verify-only"],
                       capture_output=True, text=True, cwd=REPO)
    assert v.returncode != 0
    assert "test.jsonl" in v.stdout and v.stdout.startswith("PROBLEMS"), v.stdout


@pytest.mark.skipif(not (PILOT / "train.jsonl").exists(), reason="pilot corpus not on disk")
def test_a_same_size_edit_of_the_stage_is_still_a_problem(tmp_path):
    """The test above appends, which changes the size — so a `check()` that only
    compared sizes would sail through it. One byte swapped in place leaves every
    size intact and is visible only in the sha, and that is the realistic failure:
    a re-download that truncated and padded would be size-clean."""
    dest = tmp_path / "staged"
    assert subprocess.run([sys.executable, str(PKG), "--dest", str(dest), "--owner", "pilot"],
                          capture_output=True, text=True, cwd=REPO).returncode == 0
    p = dest / "test.jsonl"
    raw = p.read_bytes()
    p.write_bytes(bytes([raw[0] ^ 0x01]) + raw[1:])
    assert p.stat().st_size == len(raw) and p.read_bytes() != raw
    v = subprocess.run([sys.executable, str(PKG), "--dest", str(dest), "--verify-only"],
                       capture_output=True, text=True, cwd=REPO)
    assert v.returncode != 0, "a content change of the same size was called verified"
    assert "test.jsonl" in v.stdout and "sha256" in v.stdout, v.stdout


@pytest.mark.skipif(not (PILOT / "train.jsonl").exists(), reason="pilot corpus not on disk")
def test_a_lossy_copy_fails_the_build_and_the_manifest_still_records_the_corpus(
        tmp_path, monkeypatch, capsys):
    """Two halves of the same promise, both invisible to a test that only ever
    stages a *successful* copy. (1) `build()`'s own verification has to fail the
    run, not just the later `--verify-only`, or the packager prints "verified"
    over a corrupt dataset and the box mounts it. (2) `SOURCE_SHA256.json` is what
    the kernel checks the mount against, so it must be a record of the *corpus* —
    hashing the staged copy against itself would make a bad copy self-consistent
    and therefore invisible forever."""
    sys.path.insert(0, str(REPO / "kaggle"))
    import package_dataset as pkg

    real = shutil

    class LossyCopy:
        """Only `copy2` is lossy; `rmtree` passes through, so the global shutil is
        never touched for anything else running in this process."""

        @staticmethod
        def copy2(src, dst, *a, **kw):
            real.copy2(src, dst, *a, **kw)
            Path(dst).write_bytes(Path(dst).read_bytes() + b"\n")

        rmtree = staticmethod(real.rmtree)

    monkeypatch.setattr(pkg, "shutil", LossyCopy)
    dest = tmp_path / "staged"
    rc = pkg.main(["--dest", str(dest), "--owner", "pilot"])
    out = capsys.readouterr().out
    assert rc != 0, "a stage that does not match the corpus must not exit 0"
    assert "PROBLEM" in out and "verified" not in out, out
    manifest = json.loads((dest / "SOURCE_SHA256.json").read_text())
    for name in ("train.jsonl", "test.jsonl"):
        want = hashlib.sha256((PILOT / name).read_bytes()).hexdigest()
        assert manifest["files"][name]["sha256"] == want, \
            f"{name}: the manifest recorded the staged bytes, not the corpus's"
        assert manifest["files"][name]["bytes"] == (PILOT / name).stat().st_size


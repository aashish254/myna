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
  when a snapshot exists;
* the trainer's non-zero stop exit reaches the notebook as non-zero;
* the staged dataset is byte-identical to the corpus the gates were sized on — the
  build itself fails on a lossy copy, a same-size edit is still a problem, and the
  manifest records the corpus rather than the stage — and packaging never uploads.
"""

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from test_memory_plan import _write_suite  # noqa: E402  (tests/ is on pytest's sys.path)

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "kaggle" / "run.py"
PKG = REPO / "kaggle" / "package_dataset.py"
REQ = REPO / "kaggle" / "requirements.txt"
PILOT = REPO / "data" / "decision-v2-pilot"


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
    """3c: the Kaggle image is Python 3.11 while the MacBook venv is 3.13. Checked
    with `feature_version` rather than a version sniff, so CI on any interpreter
    catches 3.12-only syntax (PEP 701 f-strings are the usual offender)."""
    bad = []
    for p in _files("src", "tests", "bench", "kaggle"):
        try:
            ast.parse(p.read_text(), filename=str(p), feature_version=(3, 11))
        except SyntaxError as e:
            bad.append(f"{p.relative_to(REPO)}: {e}")
    assert not bad, "not 3.11-compatible:\n" + "\n".join(bad)


def test_the_project_does_not_claim_it_needs_312_or_later():
    """>=3.13 in `pyproject.toml` is what made the Kaggle image a silent fallback:
    uv would refuse the interpreter and the job would run on whatever was left."""
    text = (REPO / "pyproject.toml").read_text()
    line = [ln for ln in text.splitlines() if ln.startswith("requires-python")]
    assert len(line) == 1, line
    version = line[0].split("=", 1)[1].strip().strip('"')
    minor = int(version.split(">=")[1].split(".")[1])
    assert minor <= 11, f"{version} still excludes the Kaggle image's 3.11"


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


# ---- the dataset packager -----------------------------------------------------

@pytest.mark.skipif(not (PILOT / "train.jsonl").exists(), reason="pilot corpus not on disk")
def test_packaging_stages_the_pilot_byte_for_byte(tmp_path):
    dest = tmp_path / "staged"
    r = subprocess.run([sys.executable, str(PKG), "--dest", str(dest)],
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
    assert subprocess.run([sys.executable, str(PKG), "--dest", str(dest)],
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
    assert subprocess.run([sys.executable, str(PKG), "--dest", str(dest)],
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
    rc = pkg.main(["--dest", str(dest)])
    out = capsys.readouterr().out
    assert rc != 0, "a stage that does not match the corpus must not exit 0"
    assert "PROBLEM" in out and "verified" not in out, out
    manifest = json.loads((dest / "SOURCE_SHA256.json").read_text())
    for name in ("train.jsonl", "test.jsonl"):
        want = hashlib.sha256((PILOT / name).read_bytes()).hexdigest()
        assert manifest["files"][name]["sha256"] == want, \
            f"{name}: the manifest recorded the staged bytes, not the corpus's"
        assert manifest["files"][name]["bytes"] == (PILOT / name).stat().st_size


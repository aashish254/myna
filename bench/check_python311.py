"""Prove the package runs on the Kaggle image's Python, not just on this venv (P3 3c).

`pyproject.toml` said `requires-python = ">=3.13"` while the supported floor is 3.11 and a
notebook keeps whatever interpreter its image ships. That combination does not fail in an
interesting way: pip/uv resolve
past it, the notebook keeps the interpreter it already has, and the run either
dies on a syntax error an hour in or — worse — trains fine here and silently
diverges there because a stdlib call changed.

So this script runs the *same* entrypoint a Kaggle cell would, under a real 3.11:
imports the whole package, answers `run.py --check`, and completes two updates of
training on CPU. It fails loud (exit 2) if no 3.11 interpreter is discoverable —
an unrunnable compatibility check is not a passed one.

    python bench/check_python311.py
    MYNA_PY311=/path/to/python3.11 python bench/check_python311.py
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MINOR = 11  # the Kaggle image's interpreter, per SPEC §5 P3 3c


def find_py311() -> str:
    env = os.environ.get("MYNA_PY311")
    cands = [env] if env else []
    cands += [shutil.which(f"python3.{MINOR}"), f"/opt/homebrew/bin/python3.{MINOR}",
              f"/usr/bin/python3.{MINOR}", str(Path.home() / f".local/share/uv/python"
                                               f"/cpython-3.{MINOR}-macos-aarch64-none/bin"
                                               f"/python3.{MINOR}")]
    for c in cands:
        if c and Path(c).exists():
            r = subprocess.run([c, "-c", "import sys, torch; print(sys.version_info[1], "
                                "torch.__version__)"], capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.split()[0] == str(MINOR):
                print(f"interpreter: {c}  (python 3.{r.stdout.split()[0]}, torch "
                      f"{r.stdout.split()[1]})")
                return c
    raise SystemExit(2 if not env else 1)


def run(py, argv, cwd=REPO, env_extra=None):
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), **(env_extra or {})}
    r = subprocess.run([py, *argv], capture_output=True, text=True, cwd=cwd, env=env)
    label = " ".join(argv[:3])
    if r.returncode != 0:
        print(f"FAIL  {label}\n{r.stdout[-1500:]}\n{r.stderr[-2500:]}")
        raise SystemExit(1)
    print(f"ok    {label}")
    return r


def main(argv=None):
    # the log is a witness, so it names the command and the interpreter it
    # actually used rather than leaving the reader to guess which python ran it
    print("$ " + shlex.join(["python", "bench/check_python311.py",
                              *(argv if argv is not None else sys.argv[1:])]))
    py = find_py311()
    claim = [ln for ln in (REPO / "pyproject.toml").read_text().splitlines()
             if ln.startswith("requires-python")]
    assert len(claim) == 1 and f">=3.{MINOR}" in claim[0], \
        f"{claim} — the package must *claim* 3.{MINOR}, not merely happen to run on it"
    print(f"claim:    {claim[0]}")
    print(f"checking 3.{MINOR} compatibility (this interpreter is "
          f"{'3.' + str(sys.version_info[1])})")
    run(py, ["-c", "import sys; assert sys.version_info[1] == " + str(MINOR) +
             ", sys.version; import myna.train, myna.paraphrase, myna.real_data, "
             "myna.model, myna.serve, myna.data, myna.tokenizer; "
             "print('package imports clean')"])
    for rel in sorted((REPO / "kaggle").glob("*.py")) + sorted((REPO / "src").rglob("*.py")):
        run(py, ["-m", "py_compile", str(rel)])
    with tempfile.TemporaryDirectory() as tmp:
        suite = Path(tmp) / "suite"
        sys.path.insert(0, str(REPO / "tests"))
        from test_memory_plan import _write_suite  # the same fixture the 3.13 tests use

        _write_suite(suite)
        out = Path(tmp) / "out"
        run(py, ["-m", "myna.train", "--suite", str(suite), "--out", str(out),
                 "--device", "cpu", "--steps", "2", "--batch", "2", "--vocab", "128",
                 "--eval-every", "0", "--paraphrase", "off"])
        m = json.loads((out / "metrics.json").read_text())
        assert m["last_step"] == 1, m
        print(f"ok    trained 2 updates and wrote metrics (dev macro on 3.{MINOR})")
        r = run(py, [str(REPO / "kaggle" / "run.py"), "--corpus", str(suite), "--check",
                     "--device", "cpu", "--out-root", str(Path(tmp) / "runs")],
                env_extra={"EXPERIMENT_NAME": "py311"})
        j = json.loads(r.stdout[r.stdout.index("{"):r.stdout.rindex("}") + 1])
        assert j["python"].startswith(f"3.{MINOR}."), j
        assert j["myna"] == str(REPO / "src" / "myna"), "it must import *this* repo"
        print(f"ok    kaggle/run.py --check under 3.{MINOR}: {j['python']}, torch {j['torch']}")
    print(f"\nPASS: the package imports, compiles and trains on python 3.{MINOR}")
    return 0


if __name__ == "__main__":
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__)
        raise SystemExit(0)
    try:
        sys.exit(main())
    except SystemExit as e:
        if e.code == 2:
            print(f"No python 3.{MINOR} with torch found. Set MYNA_PY311 to one — a "
                  "compatibility check that silently skipped is worse than no check.")
        raise

"""The command surface itself: `--help` must render on the box a reader has.

SPEC §2.3 G7 promises "one command per result", and a command whose usage text
raises is worse than no command: the reader cannot tell whether the flags moved
or the install is broken. Two real defects are pinned here.

* argparse formats every `help=` string through `%`, so a literal percent sign —
  "~1.4% of updates" in `--group-sample` — made `python -m myna.train --help`
  die with `TypeError: %o format`. It has to be written `%%`.
* `myna.serve` imported `uvicorn` before `parse_args()`, so `--help` required
  the optional `serve` extra. An optional dependency must not gate the usage text.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

import myna

SRC = str(Path(myna.__file__).resolve().parent.parent)
ROOT = Path.cwd()


def _help(*argv):
    r = subprocess.run([sys.executable, *argv, "--help"], capture_output=True, text=True,
                       cwd=ROOT, env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin",
                                      "HOME": str(Path.home())})
    return r


@pytest.mark.parametrize("module,flag", [
    ("myna.train", "--paraphrase"),
    ("myna.train", "--max-q-cells"),
    ("myna.train", "--score-loss"),
    ("myna.train", "--warm-start"),
    ("myna.rlcd", "--score"),
    ("myna.serve", "--port"),
    ("myna.serve", "--abstain-below"),
    ("myna.report", "--min-rows"),
    ("myna.onnx_export", "--q-len"),
])
def test_module_help_renders(module, flag):
    r = _help("-m", module)
    assert r.returncode == 0, f"{module} --help died: {r.stderr[-400:]}"
    assert flag in r.stdout, f"{module} --help does not list {flag}"


@pytest.mark.parametrize("script,expect", [
    ("bench/diag_learn.py", "--pool"),
    ("bench/diag_overfit.py", "--n"),
    ("bench/mutation_report.py", "Mutation battery"),
    ("bench/mutation_memory_plan.py", "Mutation battery"),
    ("bench/bench_latency_matched.py", "--intra-threads"),
    ("bench/mutation_latency_matched.py", "Mutation battery"),
    ("bench/risk_coverage.py", "--no-verify"),
    ("bench/mutation_p5.py", "Mutation battery"),
    ("bench/eval_needle.py", "--lengths"),
    ("bench/mutation_longctx.py", "Mutation battery"),
    ("bench/mutation_onnx.py", "Mutation battery"),
    ("bench/bench_mlx.py", "--quantize"),
    ("bench/diag_mlx_int8_gem.py", "--iters"),
    ("bench/mutation_mlx.py", "Mutation battery"),
    ("bench/eval_scratch.py", "--seeds"),
    ("bench/mutation_scratch.py", "Mutation battery"),
    ("bench/gates.py", "--check"),
    ("bench/mutation_gates.py", "Mutation battery"),
    ("bench/mutation_ordinal.py", "Mutation battery"),
])
def test_bench_script_help_renders(script, expect):
    """The diag scripts took `sys.argv[1]` positionally, so `--help` was parsed as a
    step count and died in `int()` — the same class of defect §9.13 found in
    `myna.train`. A reader cannot tell a moved flag from a broken install."""
    r = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True,
                       cwd=ROOT, env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin",
                                      "HOME": str(Path.home())})
    assert r.returncode == 0, f"{script} --help died: {r.stderr[-400:]}"
    assert expect in r.stdout, r.stdout[-400:]
    assert "Traceback" not in r.stderr, r.stderr[-400:]


def test_percent_in_a_help_string_survives_argparse_formatting():
    """`%%` is how argparse prints one percent; a bare `%` raises instead."""
    r = _help("-m", "myna.train")
    assert r.returncode == 0, r.stderr[-600:]
    assert "1.4%" in r.stdout and "65%" in r.stdout, "the measured rates must still print"


@pytest.mark.parametrize("script", [
    "bench/mutation_paraphrase.py",
    "bench/mutation_latency_matched.py",
    "bench/mutation_p5.py",
    "bench/mutation_longctx.py",
    "bench/mutation_onnx.py",
    "bench/mutation_mlx.py",
    "bench/mutation_scratch.py",
    "bench/mutation_gates.py",
    "bench/mutation_ordinal.py",
])
def test_mutation_help_does_not_run_the_battery(script):
    """Each run is ~30 pytest passes; `--help` must not be one of them."""
    r = subprocess.run([sys.executable, script, "--help"],
                       capture_output=True, text=True, cwd=ROOT,
                       env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin", "HOME": str(Path.home())})
    assert r.returncode == 0, r.stderr[-400:]
    assert "Mutation battery" in r.stdout
    # the battery prints one progress line per mutation ("[ 3/29] caught …"); the
    # docstring it prints for --help mentions the tallies, so match that shape
    assert not re.search(r"^\[\s*\d+/\d+\]", r.stdout + r.stderr, re.M), \
        "`--help` ran the battery"

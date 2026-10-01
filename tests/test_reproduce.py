"""8b: the registry that ties every published table row to one command and one witness.

A registry is only worth having if it can go red, so these tests do not check that the
rows look plausible — they check the six assertions `--check` makes, each one broken in
isolation:

* a row that quotes no figure at all, which no artifact could ever contradict;
* a witness that is missing, or present but not committed (`runs/*` is half-ignored by
  this repo's gitignore, which is exactly how an artifact stops existing for everyone
  but the machine that made it);
* a figure the docs quote that the artifact no longer contains — the §9.30 failure, in
  which a number copied out of a file was copied again after the file changed;
* a command the docs stopped printing;
* a command whose own tool no longer accepts the flags it publishes — the only
  assertion here that asks an interpreter rather than a file (§9.43);
* a gated row that grew a witness, which is how a projection gets read as a measurement.

And one meta-test: the README block is generated from the registry, so a row added here
without being published there fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench"))

import reproduce as R  # noqa: E402


def row(**over):
    """A copy of a real row with fields replaced, so each test breaks exactly one thing."""
    base = next(r for r in R.ROWS if r["id"] == "needle")
    merged = {**base, **over}
    return {k: (list(v) if isinstance(v, list) else v) for k, v in merged.items()}


def problems(r):
    return R.check_row(r)[0]


def notes(r):
    return R.check_row(r)[1]
def test_the_registry_is_well_formed():
    assert len({r["id"] for r in R.ROWS}) == len(R.ROWS), "row ids must be unique"
    for r in R.ROWS:
        assert r["status"] in R.STATUSES, r["id"]
        assert r["cmd"].strip(), r["id"]
        assert r["table"].strip(), r["id"]
        # a row with no quoted figure is a row nobody can falsify
        assert r["quotes"] or r["status"] == R.KAGGLE, f"{r['id']} quotes no number"


def test_the_committed_tree_passes():
    assert R.main(["--check"]) == 0


def test_a_figure_the_witness_does_not_print_fails_the_row():
    # the §9.30 mutation: the prose moved, the artifact did not
    assert problems(row(quotes=[("runs/needle_myna-v0.json", '"acc": 0.2345')])) == \
        ['runs/needle_myna-v0.json no longer contains \'"acc": 0.2345\'']


def test_a_witness_that_is_not_committed_fails_the_row():
    # .gitignore re-includes runs/*.md but not the artifact *directories* under runs/,
    # so a checkpoint's weights exist on this laptop and nowhere else. A row that
    # points its witness there is a row nobody else can audit.
    r = row(witness=["runs/myna-v0/model.pt"], quotes=[])
    assert any("not committed" in p for p in problems(r)), problems(r)


def test_a_missing_witness_fails_the_row():
    assert "witness missing: runs/never_ran.md" in problems(
        row(witness=["runs/never_ran.md"], quotes=[]))


def test_a_command_the_docs_dropped_fails_the_row():
    r = row(cmd="uv run python bench/eval_needle.py --ckpt runs/myna-v0 --n 999")
    assert any(p.startswith("no doc quotes the command") for p in problems(r))


def test_a_row_that_names_no_witness_at_all_fails():
    assert "no witness named" in problems(row(witness=[]))


# --- the fourth assertion: the command still runs (§9.43) --------------------------
# The three above are answered by reading files, so all three stay true when a tool
# renames a flag: the artifact was written before the rename, the docs quote the old
# line, and the witness is committed. These ask the interpreter instead.

def test_a_flag_the_tool_dropped_fails_the_row():
    r = row(cmd="uv run python bench/eval_needle.py --ckpt runs/myna-v0 --recall-everything")
    assert any("--recall-everything" in p for p in problems(r)), problems(r)


def test_a_command_whose_module_is_not_there_fails_the_row():
    r = row(cmd="uv run python -m myna.no_such_tool --out runs/x.md")
    assert any("does not answer --help" in p for p in problems(r)), problems(r)


def test_an_env_prefixed_command_is_still_asked_its_flags():
    # `PYTHONPATH="…" .venv/bin/python -m …` is how the laya row is written, so the
    # assignment must not be mistaken for the command.
    r = row(cmd='PYTHONPATH="<laya checkout>" .venv/bin/python '
                "-m bench.bench_latency_matched --window-nope 8")
    assert any("--window-nope" in p for p in problems(r)), problems(r)


def test_a_gated_row_is_not_asked_to_parse_as_a_shell_line():
    gated = next(r for r in R.ROWS if r["status"] == R.KAGGLE)
    assert not any("--help" in p or "parse" in p for p in problems(gated))


def test_the_flag_question_reaches_every_python_command_and_only_those():
    """A coverage assertion, not a behaviour one: the prefix-stripping above and the
    `uv run` skip are what make 24 of the 31 rows checkable, and a checker that quietly
    recognised fewer targets would still print 31/31."""
    asked = [r["id"] for r in R.ROWS
             if r["status"] != R.KAGGLE and R.python_target(r["cmd"])]
    assert len(asked) == 24, asked
    assert "latency-matched" in asked and "v0-accuracy" in asked
    assert "scope-pricing" in asked, "a `-m bench.*` row must be asked its flags"
    assert "anti-prior-audit" in asked, "the row P10 added owes the flag question too"
    assert "antiprior-off-macro" in asked, "the off control is a python command, not a grep"
    for skipped in ("browser-g3-fp32", "report-floors",
                    "kaggle-wall-clock", "kaggle-dev-tail"):
        assert skipped not in asked, skipped


def test_a_launcher_prefix_is_not_treated_as_the_interpreter():
    kind, target, flags = R.python_target("uv run python -m myna.report --split dev")
    assert (kind, target) == ("module", "myna.report")
    assert flags == ["--split"]


def test_a_tool_that_never_answers_its_help_is_a_problem_not_a_traceback(monkeypatch):
    # §9.40's shape inside §9.43's check: a `--help` that hangs must cost the timeout and
    # report, not end the gate by exception. `timeout=0` raises on the spot.
    monkeypatch.setattr(R, "HELP_TIMEOUT", 0)
    monkeypatch.setattr(R, "_HELP", {})
    r = row(cmd="uv run python -m myna.report --split dev")
    assert any("--help" in p and "0s" in p for p in problems(r)), problems(r)


def test_a_value_flag_loses_its_value_before_the_comparison():
    assert R.python_target("uv run python -m myna.report --split=dev")[2] == ["--split"]


# --- §9.51: MLX loads on macOS and nowhere else, and the gate must know which is which.
# The predicate is tested as a table first, then wired: a note off macOS, red on macOS,
# and red for every import error that is not exactly "this platform cannot load MLX".

MLX_LOAD_TAILS = [
    "ModuleNotFoundError: No module named 'mlx'",
    "ImportError: libmlx.so: cannot open shared object file: No such file or directory",
]
NOT_A_PLATFORM_LIMIT = [
    "ModuleNotFoundError: No module named 'numpy'",
    "ImportError: cannot import name 'quantized_' from 'mlx.core' (/…/myna/mlx_model.py)",
    "--help never answered within 0s",
    "no output",
]


def test_the_mlx_load_failure_is_a_platform_limit_only_off_macos():
    for tail in MLX_LOAD_TAILS:
        assert R.mlx_unloadable_here(tail, "linux"), tail
        assert R.mlx_unloadable_here(tail, "win32"), tail
        assert not R.mlx_unloadable_here(tail, "darwin"), tail
    for tail in NOT_A_PLATFORM_LIMIT:
        assert not R.mlx_unloadable_here(tail, "linux"), tail


def test_an_unloadable_mlx_row_is_a_note_and_not_a_failure(monkeypatch):
    r = row(cmd="uv run python -m bench.bench_mlx --ckpt runs/myna-v0 --reps 20")
    monkeypatch.setattr(R, "_HELP", {("module", "bench.bench_mlx"): (1, MLX_LOAD_TAILS[1])})
    monkeypatch.setattr(sys, "platform", "linux")
    assert not any("--help" in p for p in problems(r)), problems(r)
    noted = notes(r)
    assert any("bench.bench_mlx cannot be imported on linux" in n for n in noted), noted


def test_the_same_failure_on_macos_is_red_because_mlx_can_load_there(monkeypatch):
    # The mutation this arm exists for: widening the limit to every platform turns a
    # broken local MLX install into a printed excuse, and the 4 rows would pass unseen.
    r = row(cmd="uv run python -m bench.bench_mlx --ckpt runs/myna-v0 --reps 20")
    for tail in MLX_LOAD_TAILS:
        monkeypatch.setattr(R, "_HELP", {("module", "bench.bench_mlx"): (1, tail)})
        monkeypatch.setattr(sys, "platform", "darwin")
        assert any("does not answer --help" in p for p in problems(r)), (tail, problems(r))
        assert not any("cannot be imported" in n for n in notes(r))


def test_a_missing_base_dependency_off_macos_still_fails_the_row(monkeypatch):
    r = row(cmd="uv run python -m bench.bench_mlx --ckpt runs/myna-v0 --reps 20")
    monkeypatch.setattr(R, "_HELP",
                        {("module", "bench.bench_mlx"):
                         (1, "ModuleNotFoundError: No module named 'torch'")})
    monkeypatch.setattr(sys, "platform", "linux")
    assert any("does not answer --help" in p for p in problems(r)), problems(r)


def test_the_registry_names_which_rows_the_platform_limit_reaches():
    """The limit is only honest if it says how far it reaches: four rows publish MLX
    commands, so CI skips exactly four `--help` assertions there, not a growing set."""
    ids = sorted(r["id"] for r in R.ROWS
                 if "bench_mlx" in r["cmd"] or "mlx_int8_gem" in r["cmd"])
    assert ids == ["arch-params", "mlx-int8", "mlx-int8-keepgate", "mlx-kernel"], ids
    for i in ids:
        r = next(x for x in R.ROWS if x["id"] == i)
        assert R.python_target(r["cmd"]), f"{i} must be a python target or the limit is moot"


def test_check_exits_nonzero_when_a_row_is_broken():
    # --check printing FAIL is not enough: a runner that always exits 0 turns the gate
    # into a report, which is the shape every other §9.30 drift took
    broken = row(id="broken", quotes=[("runs/never_ran.json", '"nope"')])
    saved = R.ROWS
    try:
        R.ROWS = saved + [broken]
        assert R.main(["--check", "--row", "broken"]) == 1
    finally:
        R.ROWS = saved


def test_the_generated_index_is_not_evidence(tmp_path, monkeypatch):
    """README's block is `ROWS` echoed back by `readme_block()` below, so a matcher
    that read it would find every command and the doc-quote assertion could never
    fail. Blank the one prose line that carries this command and the row has to go red
    even though the index still prints it verbatim."""
    r = next(x for x in R.ROWS if x["id"] == "mlx-int8-keepgate")
    for name in R.DOCS:
        text = (R.REPO / name).read_text()
        if name == "README.md":
            assert text.count(r["cmd"]) == 2, "expected the prose line and the index"
            text = text.replace(r["cmd"], "uv run python -m bench.bench_mlx  # blanked", 1)
        (tmp_path / name).write_text(text)
    monkeypatch.setattr(R, "REPO", tmp_path)
    assert any(p.startswith("no doc quotes the command") for p in problems(r))


def test_a_row_that_quotes_no_figure_is_refused():
    # a witness alone is not a claim: without a literal from it the row can never be
    # contradicted by the artifact, which is the whole purpose of the row
    assert "row quotes no figure from its witness" in problems(row(quotes=[]))
    # the gated row is exempt because it has no artifact to quote from yet
    assert not any("quotes no figure" in p for p in
                   problems(row(status=R.KAGGLE, witness=[], quotes=[])))


def test_a_gated_row_may_not_grow_a_witness():
    r = row(status=R.KAGGLE, witness=["runs/needle_myna-v0.md"], quotes=[])
    assert "a gated row must not point at a witness" in problems(r)


def test_the_gated_row_is_the_only_row_without_a_witness():
    for r in R.ROWS:
        if r["status"] == R.KAGGLE:
            assert not r["witness"], f"{r['id']} is gated but names a witness"
        else:
            assert r["witness"], f"{r['id']} runs somewhere and names no witness"


def test_wrapped_and_commented_commands_still_match():
    # README breaks the onnx command across two lines and puts `# ...` after several
    # others; a matcher that only saw whole lines would call those undocumented
    joined = R.norm(R.strip_comment(
        'uv run python -m myna.onnx_export --ckpt runs/myna-v0 --out runs/onnx \\ \n'
        '    --scan-chunk 16 --n-chunks 4 --suite data/decision-v2-pilot   # → report'))
    assert joined in R.doc_text(), joined
    assert R.strip_comment('uv run python bench/x.py  # keep "the # in quotes"') == \
        'uv run python bench/x.py'


def test_readme_prints_every_row_and_nothing_else():
    readme = (REPO / "README.md").read_text()
    block = R.readme_block()
    assert block in readme, "README's reproduce block is out of date with the registry"
    # every row's command is published; the gated one is published as a refusal
    for r in R.ROWS:
        assert r["cmd"] in block or r["status"] == R.KAGGLE
        assert r["table"] in block


def test_list_names_each_row_once(capsys):
    assert R.main(["--list"]) == 0
    out = capsys.readouterr().out
    for r in R.ROWS:
        assert out.count(f"| `{r['id']}` |") == 1


@pytest.mark.parametrize("id,status", [("needle", R.HERE), ("latency-matched", R.LAYA)])
def test_run_refuses_without_yes(id, status, monkeypatch):
    """--dry-run is part of the test, not a convenience: the guard is one of the
    mutations the battery applies, and a battery that executed the real command when
    the guard was removed would overwrite the committed witnesses it is checking."""
    called = []
    monkeypatch.setattr(R.subprocess, "call", lambda *a, **k: called.append(a) or 0)
    assert next(r for r in R.ROWS if r["id"] == id)["status"] == status
    with pytest.raises(SystemExit) as e:
        R.main(["--run", id, "--dry-run"])
    assert "committed witnesses" in str(e.value)
    assert called == []


def test_dry_run_prints_the_command_and_touches_nothing(capsys):
    assert R.main(["--run", "needle", "--yes", "--dry-run"]) == 0
    out = capsys.readouterr().out
    row = next(r for r in R.ROWS if r["id"] == "needle")
    assert out.splitlines()[0] == "$ " + row["cmd"]
    assert row["witness"][0] in out


def test_run_refuses_a_kaggle_row_even_with_yes():
    gated = next(r for r in R.ROWS if r["status"] == R.KAGGLE)
    with pytest.raises(SystemExit) as e:
        R.main(["--run", gated["id"], "--yes", "--dry-run"])
    assert "gated on Kaggle" in str(e.value)


def test_run_refuses_a_training_row_even_with_yes():
    retrain = next(r for r in R.ROWS if r["status"] == R.RETRAIN)
    with pytest.raises(SystemExit) as e:
        R.main(["--run", retrain["id"], "--yes", "--dry-run"])
    assert "does not happen on this box" in str(e.value)


def test_an_unknown_row_is_refused_not_silently_skipped():
    for argv in (["--check", "--row", "nope"], ["--run", "nope"]):
        with pytest.raises(SystemExit) as e:
            R.main(argv)
        assert "no row" in str(e.value)


def test_the_make_target_checks_and_only_reruns_when_a_row_is_named():
    """The README promises `make repro` and the gate is only a gate if the target runs
    the check rather than the runs; the run target in turn only reaches an artifact
    when a row is named and `--yes` is on the same line."""
    mk = (REPO / "Makefile").read_text()

    def recipe(target):
        return mk.split(f"\n{target}:")[1].split("\n\n")[0]

    assert "bench/reproduce.py --check" in recipe("repro")
    assert "--run" not in recipe("repro")
    assert "--yes --dry-run" in recipe("repro-show")
    assert "--yes --dry-run" not in recipe("repro-run") and "--yes" in recipe("repro-run")
    # a missing ROW must abort before anything runs, not invoke `--run` with no id
    assert "$(if $(ROW)" in mk


def test_every_harness_that_writes_a_witness_records_its_command():
    """§9.30 forward: the artifacts that already exist were checked by hand, and the
    only way to keep the next one from being checked by hand is to make the harness
    print its own command. This asserts the *source* does, so a future witness cannot
    be written by a harness that forgot."""
    import re
    srcs = sorted((REPO / "bench").glob("*.py")) + [REPO / "src" / "myna" / "onnx_export.py",
                                                    REPO / "src" / "myna" / "train.py"]
    for f in srcs:
        text = f.read_text()
        if not re.search(r"(write_text|json\.dump|open\([^)]*[\"']w)", text):
            continue
        if f.name.startswith("mutation_"):
            continue  # the batteries print to stdout, they are not witnesses
        assert "cmd" in text or "getloadavg" in text, f"{f.relative_to(REPO)} writes no provenance"


def test_a_real_run_leaves_its_command_at_the_top_of_its_witness(tmp_path):
    """The source claiming it is not enough — the assertion has to land on a file a
    run actually wrote. Two harnesses, both sized down to a second of work, one
    markdown-first and one markdown-plus-json, because they write provenance
    differently (§9.30: the class of defect is a harness that stopped)."""
    import subprocess
    sys.path.insert(0, str(REPO / "tests"))
    from test_memory_plan import _write_suite  # the same fixture the 3.13 tests use

    suite = _write_suite(tmp_path / "suite")
    md = tmp_path / "risk.md"
    out = subprocess.run([sys.executable, "-m", "bench.risk_coverage", "--suite", str(suite),
                          "--limit", "1", "--no-verify", "--out", str(md)],
                         cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-900:]
    first = md.read_text().splitlines()[0]
    assert first.startswith("$ python -m bench.risk_coverage")
    for flag in (f"--suite {suite}", "--limit 1", "--no-verify", f"--out {md}"):
        assert flag in first, first
    # and the machine-readable twin carries it too, so neither witness is hand-written
    js = json.loads(Path(str(md).rsplit(".", 1)[0] + ".json").read_text())
    assert js["cmd"] == first[2:]


def test_a_needle_run_prints_its_command_into_both_witnesses(tmp_path):
    import subprocess
    md = tmp_path / "needle.md"
    out = subprocess.run([sys.executable, "bench/eval_needle.py", "--ckpt", "runs/myna-v0",
                          "--device", "cpu", "--n", "1", "--lengths", "64",
                          "--out", str(md)],
                         cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-900:]
    lines = md.read_text().splitlines()
    assert lines[2] == (f"`python bench/eval_needle.py --ckpt runs/myna-v0 --device cpu "
                        f"--n 1 --lengths 64 --out {md}`"), lines[:4]
    assert json.loads(Path(str(md).rsplit(".", 1)[0] + ".json").read_text())["cmd"] == \
        lines[2].strip("`")

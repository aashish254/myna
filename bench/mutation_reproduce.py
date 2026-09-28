"""Mutation battery for the reproduction registry (SPEC §5 P8 8b, §7.1 discipline).

    python bench/mutation_reproduce.py

`bench/reproduce.py` publishes one claim: *every table row in this repo has a command
that regenerates it and a committed witness that still prints the figure the prose
quotes.* Nothing external can check that — the registry and the docs are both in this
repo, so a weakened checker and a drifting doc look identical from inside. Hence the
same witness as every other gate here: inject each lie the checker could tell, and
require `tests/test_reproduce.py` to fail.

Five families, in the order the defect actually arrives.

**A check that reports nothing.** The quote assertion losing its `bad.append`, the
tracked-witness branch becoming unreachable, the doc-quote test inverted to always
pass. Each one turns the gate into a printer: `--check` still says 19/19, and §9.30
happens again with a green light on.

**A matcher that only matches the easy case.** `norm()` folding a backslash continuation
badly, `strip_comment()` leaving the comment in. README breaks the onnx command over two
lines and puts `# ...` after four others, so a matcher that silently fails on both would
call those rows undocumented — the failure direction is *stricter*, and it is caught for
the same reason: a gate that cannot tell a real command from a wrapped one is not a gate.

**A row that owes nothing.** A gated Kaggle row allowed to point at a witness; a row
with no quoted figure accepted. Both are how a projection gets published as a
measurement — the registry's reason to exist is that the `g1-v1` row has no file.

**A runner that runs what it should refuse.** `--run` without `--yes`, a gated row, a
training row: each guard removed in turn. Every one of these mutations is paired with a
test that also passes `--dry-run`, because the scratch repo symlinks `runs/` and a
battery that launched the real harness would overwrite the committed witnesses it is
supposed to be checking.

**A harness that stops naming itself.** `emit("$ " + cmd)` emptied in the risk/coverage
harness, and the `cmd` key dropped from the needle's json. The static half of the test
reads the source, the dynamic half runs the harness against a two-row fixture suite and
reads the file back, so the lie has to survive both.

Scratch-repo mechanics per §9.18: `src`, `bench`, `tests`, every top-level markdown file
plus `pyproject.toml` and `Makefile` are copied (`pythonpath = ["src"]` is resolved against the rootdir, so
a mutated file reached through `PYTHONPATH` loses to the unmutated one, §9.12), `data`,
`runs` and `.git` are symlinked (the checker asks `git ls-files` whether a witness is
committed, which needs a repository), the baseline must be green before any mutation
runs, and the battery aborts if the *first* mutation survives.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
TESTS = ["tests/test_reproduce.py"]
# Every top-level file the tests read has to be in the scratch, and naming them is how
# this battery went red on itself: `reproduce.DOCS` gained two launch docs and this list
# did not, and `Makefile` was never here even though
# `test_the_make_target_checks_and_only_reruns_when_a_row_is_named` reads it. A glob of
# the markdown files plus the two build files travels with the repo instead.
SCRATCH_FILES = sorted(p.name for p in ROOT.iterdir() if p.suffix == ".md") + [
    "pyproject.toml", "Makefile"]

RP = "bench/reproduce.py"
RC = "bench/risk_coverage.py"
ND = "bench/eval_needle.py"

MUTATIONS = [
    # --- a check that reports nothing ------------------------------------------
    ("a figure the witness no longer prints is still a published figure", RP,
     "            bad.append(f\"{f} no longer contains {value!r}\")",
     "            pass"),
    ("a witness that is not committed counts as a witness", RP,
     "        elif not tracked(f):",
     "        elif False:"),
    ("a row may name a witness that does not exist", RP,
     "        if not (REPO / f).exists():",
     "        if False:"),
    ("a command the docs dropped is still published", RP,
     "    if norm(r[\"cmd\"]) not in doc_text() and r[\"status\"] != KAGGLE:",
     "    if False:"),
    ("a row with no witness at all passes", RP,
     "    if not r[\"witness\"] and r[\"status\"] not in (KAGGLE,):",
     "    if False:"),
    ("--check always exits zero", RP,
     "        return 1 if fails else 0",
     "        return 0"),
    # --- a matcher that only matches the easy case ------------------------------
    ("a command wrapped with a backslash stops being recognised", RP,
     "    return re.sub(r\"\\s+\", \" \", re.sub(r\"\\\\\\s*\\n\\s*\", \" \", s)).strip()",
     "    return re.sub(r\"\\s+\", \" \", s).strip()"),
    ("an inline doc comment becomes part of the command", RP,
     "        elif ch == \"#\":",
     "        elif ch == \"#\" and False:"),
    ("the docs are read one file at a time, so only README counts", RP,
     "    raw = \"\\n\".join((REPO / name).read_text() for name in DOCS)",
     "    raw = \"\\n\".join((REPO / name).read_text() for name in DOCS[:1])"),
    ("the registry's own generated index counts as a doc quoting the command", RP,
     "    raw = GENERATED.sub(\"\\n\", raw)",
     "    pass"),
    # --- a row that owes nothing ------------------------------------------------
    ("a gated Kaggle row is allowed to grow a witness", RP,
     "    if r[\"status\"] == KAGGLE and r[\"witness\"]:",
     "    if False:"),
    ("a row without a falsifiable figure is accepted", RP,
     "    if not r[\"quotes\"] and r[\"status\"] != KAGGLE:",
     "    if False:"),
    # --- the runner's promise ---------------------------------------------------
    # every --run mutation is paired with a test that passes --dry-run: removing a
    # guard must not be able to launch a real harness, because the scratch repo
    # symlinks runs/ and a real run would overwrite the committed witnesses.
    ("--run overwrites a committed witness without asking", RP,
     "        if not args.yes:",
     "        if False:"),
    ("a gated Kaggle row can be run locally", RP,
     "        if me[\"status\"] == KAGGLE:",
     "        if False:"),
    ("a training row can be run on this box", RP,
     "        if me[\"status\"] == RETRAIN:",
     "        if False:"),
    ("the README block stops carrying a row's command", RP,
     "        out.append(r[\"cmd\"] if r[\"status\"] != KAGGLE",
     "        out.append(\"\" if r[\"status\"] != KAGGLE"),
    ("the README block stops naming the table a command belongs to", RP,
     "        out.append(f\"# {r['table']}  [{r['status']}]\")",
     "        out.append(f\"# [{r['status']}]\")"),
    # --- a harness that stops naming itself -------------------------------------
    ("the risk/coverage witness stops printing its own command", RC,
     "    emit(\"$ \" + cmd)",
     "    emit(\"\")"),
    ("the risk/coverage json stops carrying it", RC,
     "    out_json.write_text(json.dumps({\"cmd\": cmd,",
     "    out_json.write_text(json.dumps({\"not_cmd\": None,"),
    ("the needle markdown stops printing its command", ND,
     "        f.write(f\"`{cmd}`\\n\\n\")",
     "        pass"),
    ("the needle json stops carrying its command", ND,
     "    js.write_text(json.dumps({\"cmd\": cmd,",
     "    js.write_text(json.dumps({\"not_cmd\": None,"),
]


def make_scratch(tmp):
    repo = Path(tmp) / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir()
    for name in ("src", "bench", "tests"):
        shutil.copytree(ROOT / name, repo / name,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for name in SCRATCH_FILES:
        shutil.copy(ROOT / name, repo / name)
    # runs/ holds the witnesses and data/ the frozen split; .git is what makes a
    # witness "committed", so the checker needs a repository to ask the question in
    for name in ("data", "runs", ".git"):
        if (ROOT / name).exists():
            (repo / name).symlink_to(ROOT / name)
    return repo


def pytest_in(repo):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-q", "--no-header",
                           "-p", "no:cacheprovider"],
                          capture_output=True, text=True, cwd=repo, env=env, timeout=1800)


def run_one(rel, old, new, tmp):
    repo = make_scratch(tmp)
    path = repo / rel
    text = path.read_text()
    n = text.count(old)
    if n != 1:
        return f"BAD-PATTERN ({n} matches)", ""
    path.write_text(text.replace(old, new))
    r = pytest_in(repo)
    if r.returncode == 0:
        return "SURVIVED", ""
    fails = [ln for ln in r.stdout.splitlines()
             if ln.startswith("FAILED") or ln.startswith("ERROR")]
    detail = fails[0] if fails else (r.stdout.strip().splitlines() or [""])[-1]
    return "caught", detail[:110]


def main():
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__)
        return 0
    survivors, bad = [], []
    with tempfile.TemporaryDirectory() as tmp:
        base = make_scratch(tmp)
        r0 = pytest_in(base)
        if r0.returncode != 0:
            print("baseline (unmutated) copy is RED -- the battery proves nothing")
            print(r0.stdout[-4000:])
            return 2
        print(f"baseline copy: green ({len(TESTS)} test file, {len(MUTATIONS)} mutations)")
        print(f"against a scratch copy of src/bench/tests + {len(SCRATCH_FILES)} top-level "
              "files (every *.md, pyproject.toml, Makefile), with\n"
              "data/runs/.git symlinked (§9.18), so no live tree is being edited.\n")
        for i, (label, rel, old, new) in enumerate(MUTATIONS, 1):
            verdict, detail = run_one(rel, old, new, tmp)
            print(f"[{i:2d}/{len(MUTATIONS)}] {verdict:11s} {label}"
                  + (f"\n            {detail}" if detail and verdict == "caught" else ""),
                  flush=True)
            if verdict == "SURVIVED":
                survivors.append(label)
                if i == 1:
                    print("  first mutation survived: the battery is not testing the "
                          "mutated code; aborting", flush=True)
                    return 2
            elif verdict.startswith("BAD-PATTERN"):
                bad.append((label, verdict))
        print(f"\n{len(MUTATIONS) - len(survivors) - len(bad)}/{len(MUTATIONS)} mutations caught")
        for s in survivors:
            print(f"  SURVIVED: {s}")
        for b in bad:
            print(f"  {b}")
        return 1 if survivors or bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

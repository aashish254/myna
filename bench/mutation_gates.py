"""Mutation battery for the release-gate checker (SPEC §7.1, §9.18, §9.31).

`bench/gates.py` publishes seven verdicts, and a verdict is three words — the least
falsifiable artifact in the repo. Each entry below is one way those three words could stop
being tied to anything, applied to a frozen copy and caught by `tests/test_gates.py`.

The battery is shaped differently from the others because the harness under test reads the
*docs* and the *git index*, not just its own arguments. A scratch copy therefore needs
`README.md`/`SPEC.md`/`PLAN.md`/`TODO.md` present (the registry's own doc-quote check reads
all four) and `.git` symlinked, so `git ls-files --error-unmatch` still answers for the
committed witnesses. `runs/` and `data/` are symlinked as elsewhere; nothing here writes a
witness, because the fixture suite runs entirely on the committed artifacts.

Four of the twenty-one are worth naming, because they are the ones a reader would not
guess to break:

* `if bad:` → `if False:` inside `registry_green()` — G7's proof *is* the checker, so a
  checker that reports itself green while rows drift is a gate resting on itself. That is
  §9.31 applied to this file, and it is the only entry whose victim is another gate.
* `drift, _ = check_row(r)` → `drift = []` — the seam that stops a gate being greener than
  the witness behind it. Without it the registry could drift and the verdict table would
  keep printing "met".
* `return value == p["expect"]` → `return True` — the artifact stops being asked. Every
  proof would still print its own sentence, which is what makes this the dangerous one.
* the four §9.44 entries together — SPEC §2.2 is the gate table's one *hand-copied* cell,
  README's block is generated and `gates.py`'s own counts now are counted, so the only
  prose left that can fall behind is the one nobody reads. Two of the four ask it to agree
  with `ROWS`; the other two are the shapes that check would take if it were theatre: the
  comparison that never reports, and the table read from the wrong column so a count it
  cannot find is a count that always matches.

One entry survived its first pass and the fix was to the *test*, not the rule, so it is
worth the sentence: dropping `r["status"] not in UNREPRODUCIBLE` from the `with_witness`
filter was invisible, because the test broke a `met` verdict by citing `g1-v1` — and the
registry forbids a `gated-kaggle` row from carrying quotes, so `r["quotes"]` filtered it
out anyway and both halves of the condition could not be told apart. The discriminating
citation is a `retrain` row, whose witness *is* committed and figure-checked and which this
box still cannot regenerate; `test_a_met_verdict_cannot_rest_on_a_run_this_box_cannot_repeat`
cites both shapes and asserts they fail for the same named reason.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
TESTS = ["tests/test_gates.py"]
GA = "bench/gates.py"

MUTATIONS = [
    # --- the artifact stops being asked ---------------------------------------
    ("a proof reads its field and then ignores the value", GA,
     '    return value == p["expect"], f"{p[\'path\']} is {value!r}, not {p[\'expect\']!r}"',
     '    return True, "proof not read"'),
    ("a proof key path stops walking lists", GA,
     "        if isinstance(cur, list):", "        if False:"),
    ("a proof file no longer has to be committed", GA,
     '            elif not _tracked(p["file"]):', "            elif False:"),
    # --- the verdict stops owing the reader a field ---------------------------
    ("a met verdict may rest on a field that prints false", GA,
     '        if isinstance(p["expect"], bool):', "        if False:"),
    ("a met verdict stops needing a field that prints true", GA,
     '    if g["verdict"] == MET and not any(p["expect"] is True for p in g["proofs"]):',
     "    if False:"),
    ("a not-met verdict stops needing a field that prints false", GA,
     '    if g["verdict"] == NOT_MET and not any(p["expect"] is False for p in g["proofs"]):',
     "    if False:"),
    # --- the registry underneath stops propagating ----------------------------
    ("a cited registry row stops being re-checked", GA,
     "        drift, _ = check_row(r)", "        drift = []"),
    ("a cited row no longer has to be locally reproducible", GA,
     '    with_witness = [r for r in cited if r["quotes"] and r["status"] not in UNREPRODUCIBLE]',
     '    with_witness = [r for r in cited if r["quotes"]]'),
    ("a void artifact can back a verdict again (§9.23)", GA,
     '            if "void" in r["note"]:', "            if False:"),
    ("a gate may cite nothing at all", GA,
     '    if not g["rows"]:', "    if False:"),
    # --- "open" stops meaning a missing artifact ------------------------------
    ("open no longer has to name a run this box cannot do", GA,
     '        if not [r for r in cited if r["status"] in UNREPRODUCIBLE]:', "        if False:"),
    # --- the prose stops being a second place that has to agree ---------------
    ("SPEC §2.2's verdict word stops being compared", GA,
     '    elif got != g["verdict"]:', "    elif False:"),
    ("a gate with no §2.2 row at all passes", GA,
     "    if got is None:", "    if False:"),
    # --- §9.44: §2.2's hand-copied counts stop answering to the registry ------
    ("the gate that rests on the registry stops reading its counts back out of §2.2", GA,
     '    if any(p["file"] == REGISTRY for p in g["proofs"]):', "    if False:"),
    ("the live-count comparison runs and never reports", GA,
     "            if tok not in cell:", "            if False:"),
    ("§2.2 is read for the wrong column, so no count can ever be found in it", GA,
     "            out[m.group(1)] = [c.strip() for c in "
     "line.strip().strip(\"|\").split(\"|\")][-1]",
     "            out[m.group(1)] = [c.strip() for c in "
     "line.strip().strip(\"|\").split(\"|\")][2]"),
    ("the registry's size stops being counted and every status reads zero", GA,
     '        return sum(1 for r in ROWS if r["status"] == status)', "        return 0"),
    ("the release headline counts every verdict as met", GA,
     '    return {v: sum(1 for g in GATES if g["verdict"] == v) for v in VERDICTS}',
     '    return {v: sum(1 for g in GATES if g["verdict"] in VERDICTS) for v in VERDICTS}'),
    ("README's generated block loses its fences", GA,
     '    return "\\n".join([BEGINS,', '    return "\\n".join(["",'),
    # --- §9.31: the checker checking itself -----------------------------------
    ("G7's proof reports green while registry rows drift", GA,
     "    if bad:", "    if False:"),
    ("--check's exit code stops reflecting the failures", GA,
     "        return 1 if fails else 0", "        return 0"),
]


def make_scratch(tmp):
    """Copy the code, link everything the checker reads out of the repo.

    `.git` is a symlink rather than a copy: `git ls-files` reads the index, not the
    filesystem, so the linked index answers for witness paths that exist at the same
    relative location under `runs/` (also linked). The docs are linked because the
    registry's own doc-quote check — which `check_gate` runs through every cited row —
    reads all of them, and the `Makefile` is linked because
    `test_make_gates_is_the_check_and_the_prose_promises_it_by_that_name` reads it: a
    scratch built from `*.md`/`*.toml` alone had no Makefile in it, and this battery's
    baseline was red for exactly that reason without anybody being told.
    """
    repo = Path(tmp) / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir()
    for name in ("src", "tests", "bench"):
        shutil.copytree(ROOT / name, repo / name,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for name in ("data", "runs", ".git"):
        if (ROOT / name).exists():
            (repo / name).symlink_to(ROOT / name)
    for name in sorted(p.name for p in ROOT.iterdir()
                       if p.suffix in (".md", ".toml", ".cfg")) + ["Makefile"]:
        (repo / name).symlink_to(ROOT / name)
    return repo


def pytest_in(repo):
    env = {k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}
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
    fails = [ln for ln in r.stdout.splitlines() if ln.startswith("FAILED")
             or ln.startswith("ERROR")]
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
        print(f"baseline copy: green ({len(TESTS)} test file)\n")
        for i, (label, rel, old, new) in enumerate(MUTATIONS, 1):
            verdict, detail = run_one(rel, old, new, tmp)
            print(f"[{i:2d}/{len(MUTATIONS)}] {verdict:11s} {label}"
                  + (f"\n            {detail}" if detail and verdict == "caught" else ""),
                  flush=True)
            if verdict == "SURVIVED":
                survivors.append(label)
                if i == 1:
                    print("  first mutation survived: the harness is not testing the "
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
    sys.exit(main())

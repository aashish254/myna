"""Mutation battery for the 9d scope table (`bench/scope_pricing.py`, SPEC §7.1).

    python bench/mutation_scope_pricing.py

`bench/scope_pricing.py` publishes a four-row table and one JSON witness, and 9d's scope
decision is argued from those numbers in SPEC, TODO and README. Every figure in it is a
claim no external truth can check: the harness reads a committed report and prints a
*smaller* measurement of it, which is precisely the shape §9.41 turned into a correction —
one run, two aggregations, the flattering one quoted.

So the witness is the same as everywhere else in this repo: mutate each lie the harness can
tell, and require a test to fail. The lies cluster in four families, and each family is how
this table would actually rot.

**The control equality stops being a check.** The only thing that makes the three filtered
rows "the same measurement on a different scope" is that the unfiltered row reproduces the
report's own published `macro` block to the digit and its `g1.pass` too. Widen `TOLERANCE`,
change one comparison to read the wrong report field, drop one of the three compared fields,
or delete the verdict comparison, and the table still prints four tidy rows — now priced off
an artifact the harness cannot tie to. `TOLERANCE = 1e-12` and `if False:` are the two ends
of the same hole: a guard that is present and a guard that bites.

**The floor stops belonging to its own row.** A scope row must carry the majority floor the
*surviving* cells imply. Read `maj` (or `acc`, or `uniform`) off `cells` instead of `sub`
and every drop becomes a win, which is the marketing sentence this file exists to make
impossible. Same family: the two 9d options collapsing into a predicate that keys on
`source` where it says `(source, question)`, `>=` becoming `>` in the below-floor scope
(three cells sit *exactly* on their own floor, so that boundary moves 11 → 8), and
`g1_verdict(acc, maj)` called with its arguments swapped.

**The witness forgetting how it was made.** §9.30: the `cmd` line, the `report` path, the
`published` headline row, the set of `scopes`, and the `below_floor_cells` list — the last
one computed by a second expression that has to agree with the subset predicate, because the
README quotes both.

**The printed table labelling a number wrongly.** `myna`/`floor`/`margin`/`verdict` are the
column headers; print the uniform floor under `macro`, print a shortfall under `margin`, or
print `pass` for a row that fails one half of G1, and the JSON is still right while the
artifact a human reads is not.

Same scratch-repo mechanics as `mutation_p5.py`: copy `src` + `bench` + `tests` (pyproject
pins `pythonpath = ["src"]` relative to the rootdir, so a mutated file reached through
PYTHONPATH silently loses to the unmutated one, §9.12), symlink `data` and `runs`, require a
green baseline, abort if the *first* mutation survives, and report any pattern that did not
match exactly once. Only `bench/scope_pricing.py` is mutated: the roll-up it imports
(`myna.report.macro`, `g1_verdict`) is `bench/mutation_report.py`'s subject, and mutating it
here would count the same claim twice.

Deliberately absent: `sort_keys`/column widths/`:>8.4f` rounding and the `SUBSETS` label
strings. Those move digits or words without moving a claim, and counting them would inflate
the coverage figure — the rule `mutation_p5.py` states for `INFER_CHUNK`.

The one thing this battery does *not* carry, and cannot, is a mutation of its own subject:
`bench/scope_pricing.py`'s `main()` runs only under `if __name__ == "__main__"`, and pytest
imports it as `bench.scope_pricing`. That is the same limit §9.46's printed-table tests live
inside, and it is why the tests call `main(argv)` directly rather than the script.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
TESTS = ["tests/test_scope_pricing.py"]

SP = "bench/scope_pricing.py"

MUTATIONS = [
    # --- the control equality: the tie between this table and the committed run ----------
    ("the report-comparison tolerance is widened to a tenth", SP,
     "TOLERANCE = 1e-12",
     "TOLERANCE = 0.05"),
    ("the field comparison never fires", SP,
     "        if abs(control[field] - want) > TOLERANCE:",
     "        if False:"),
    ("the majority floor is no longer compared to the report", SP,
     '    for field, want in (("myna", published["acc"]), ("majority_floor", published["majority"]),\n'
     '                        ("uniform_floor", published["uniform"])):',
     '    for field, want in (("myna", published["acc"]),\n'
     '                        ("uniform_floor", published["uniform"])):'),
    ("the level is checked against the report's uniform floor", SP,
     '("myna", published["acc"])',
     '("myna", published["uniform"])'),
    ("the verdict comparison is dropped", SP,
     '    if control["pass"] is not doc["g1"]["pass"]:',
     "    if False:"),
    # --- the floor belongs to its own row -----------------------------------------------
    ("a subset row is priced against the published floor", SP,
     '    maj = macro(sub, "majority")',
     '    maj = macro(cells, "majority")'),
    ("a subset row reports the published level, not its own", SP,
     '    acc = macro(sub, "acc")',
     '    acc = macro(cells, "acc")'),
    ("the uniform floor is carried over instead of recomputed", SP,
     '        "uniform_floor": macro(sub, "uniform"),',
     '        "uniform_floor": macro(cells, "uniform"),'),
    ("option A keys on source, so it drops nothing", SP,
     '     lambda c, dead: (c["source"], c["question"]) not in dead),',
     '     lambda c, dead: c["source"] not in dead),'),
    ("the below-floor scope excludes cells sitting exactly on their floor", SP,
     '("every cell below its own floor out", lambda c, dead: c["acc"] >= c["majority"]),',
     '("every cell below its own floor out", lambda c, dead: c["acc"] > c["majority"]),'),
    ("the dropped count is the kept count", SP,
     '        "dropped": len(cells) - len(sub),',
     '        "dropped": len(sub),'),
    ("the verdict is computed with level and floor swapped", SP,
     '    v = g1_verdict(acc, maj)',
     '    v = g1_verdict(maj, acc)'),
    ("the margin shortfall is measured against the level", SP,
     '        "shortfall_to_margin": round(0.15 - v["margin_observed"], 6),',
     '        "shortfall_to_margin": round(0.15 - acc, 6),'),
    # --- the witness: §9.30 provenance, and the list that must agree with the predicate --
    ("the witness stops naming the command that writes it", SP,
     '        "cmd": shlex.join(["python", "-m", "bench.scope_pricing", *flags]),',
     '        "cmd": "see the reproduction registry",'),
    ("the published headline row is replaced by the most filtered row", SP,
     '        "published": control,',
     '        "published": rows[-1][1],'),
    ("the control row disappears from the witness's scopes", SP,
     '        "scopes": {label: r for label, r in rows},',
     '        "scopes": {label: r for label, r in rows[1:]},'),
    ("the named cells print question/source instead of source/question", SP,
     '        "named_cells_out_of_scope": [f"{s}/{q}" for s, q in sorted(NAMED)],',
     '        "named_cells_out_of_scope": [f"{q}/{s}" for s, q in sorted(NAMED)],'),
    ("the listed below-floor cells include the ones exactly on their floor", SP,
     '                              if c["acc"] < c["majority"]],',
     '                              if c["acc"] <= c["majority"]],'),
    # --- the printed table: the artifact a human reads ----------------------------------
    ("the headline calls the uniform floor the macro", SP,
     '    lines = [f"G1 on the published scope: macro {published[\'myna\']:.4f} · "',
     '    lines = [f"G1 on the published scope: macro {published[\'uniform_floor\']:.4f} · "'),
    ("the headline calls the uniform floor the majority floor", SP,
     "             f\"floor {published['majority_floor']:.4f} · \"",
     "             f\"floor {published['uniform_floor']:.4f} · \""),
    ("the margin column prints a shortfall", SP,
     "                     f\"{r['margin_observed']:>+9.4f}{r['shortfall_to_margin']:>12.4f}\"",
     "                     f\"{r['shortfall_to_margin']:>9.4f}{r['shortfall_to_margin']:>12.4f}\""),
    ("the verdict column is printed from the cell count", SP,
     "                     f\"{'pass' if r['pass'] else 'not met'}\")",
     "                     f\"{'pass' if r['cells'] > 10 else 'not met'}\")"),
    ("the printed table drops the published scope row", SP,
     "    for label, r in rows:",
     "    for label, r in rows[1:]:"),
    # --- the boundary: a report that is not there --------------------------------------
    ("a missing report is not reported", SP,
     "    if not path.exists():",
     "    if False:"),
]


def make_scratch(tmp):
    repo = Path(tmp) / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir()
    for name in ("src", "bench", "tests"):
        shutil.copytree(ROOT / name, repo / name,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    shutil.copy(ROOT / "pyproject.toml", repo / "pyproject.toml")
    for name in ("data", "runs"):
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
        summary = (r0.stdout.strip().splitlines() or [""])[-1]
        print(f"baseline copy: green ({len(TESTS)} test file"
              f"{'s' if len(TESTS) != 1 else ''}) {summary}\n")
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

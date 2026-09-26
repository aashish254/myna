"""Mutation battery for the G4 long-context recall harness (SPEC §5 P7, §7.1).

    python bench/mutation_longctx.py

`myna.longctx.eval_lengths` prints the one number G4 is read from: needle accuracy
at 1k / 4k / 8k / 16k tokens. A curve has no external truth to assert against, and
this one has a specific temptation attached to it — a flat line at 16k is the claim
§2.1's window row already makes, so a harness that quietly stops testing what it
says it tests would produce exactly the number people want to see.

The families below are the ways that happens without anything looking broken:

* **the needle stops being the only evidence**, or stops moving: a needle pinned to
  the end of every state measures suffix reading and prints "16k recall";
* **the scoring compares the wrong things**: option index instead of option text,
  min instead of max, a denominator that is the number of lengths;
* **the run lies about its own reproducibility**: a seed that is parsed, printed and
  then ignored is the worst of these, because the command line looks like a witness.

`test_the_curve_records_a_time_per_row_and_keeps_its_units` is the one claim here
that is *not* mutation-checked: the battery has no way to catch a mean reported as a
total without asserting on wall-clock magnitudes, and a timing assertion in a suite
that runs on shared cores (§9.23) would be flaky in exactly the direction that
matters. It is a mean over the samples the run took, and nothing more is claimed.

Same scratch-repo mechanics as the other batteries: copy `src` + `bench` + `tests`
(pyproject's `pythonpath = ["src"]` outranks PYTHONPATH relative to the rootdir,
§9.12), green baseline required, abort on a first-mutation survivor, and any pattern
that does not match exactly once is reported rather than skipped.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
TESTS = ["tests/test_longctx.py"]

LC = "src/myna/longctx.py"
EN = "bench/eval_needle.py"

MUTATIONS = [
    # --- the needle has to stay the only evidence, and it has to move -----------
    ("the needle is always pinned to the end of the state", LC,
     "    pos = rng.randrange(len(sentences) + 1)",
     "    pos = len(sentences)"),
    ("the needle is always pinned to the front", LC,
     "    pos = rng.randrange(len(sentences) + 1)",
     "    pos = 0"),
    ("insertion ignores the chosen position", LC,
     "    sentences.insert(pos, needle)",
     "    sentences.append(needle)"),
    ("the desk named in the needle is not the gold", LC,
     '    needle = f"note that {entity} is owned by the {DESKS[gold]} desk".capitalize() + "."',
     '    needle = f"note that {entity} is owned by the {DESKS[0]} desk".capitalize() + "."'),
    ("every needle asks about the same entity", LC,
     "    entity = entity if entity is not None else rng.choice(ENTITIES)",
     "    entity = ENTITIES[0]"),
    ("the gold is constant, so any flat answer scores 1.000", LC,
     "    gold = rng.randrange(len(DESKS))",
     "    gold = 0"),
    ("the state stops being built to the requested length", LC,
     "    while ntok < target_tokens:",
     "    while ntok < target_tokens // 2:"),
    # --- what counts as right ---------------------------------------------------
    ("the wrong option is scored correct", LC,
     "            correct += int(pick == nd.options[nd.gold])",
     "            correct += int(pick != nd.options[nd.gold])"),
    ("the argmax becomes an argmin", LC,
     "            pick = max(pred, key=pred.get)",
     "            pick = min(pred, key=pred.get)"),
    ("one row is averaged over the whole ladder", LC,
     "        acc = correct / n_per_length",
     "        acc = correct / (n_per_length * len(lengths))"),
    ("the question is asked over three of the six desks", LC,
     '                          "criteria": {d: d for d in nd.options}}}',
     '                          "criteria": {d: d for d in nd.options[:3]}}}'),
    # --- the run reporting something it did not do ------------------------------
    ("the seed is parsed, printed, and then ignored", LC,
     "    rng = random.Random(seed)",
     "    rng = random.Random(0)"),
    ("only the first length is measured, the rest are printed anyway", LC,
     "    for L in lengths:",
     "    for L in lengths[:1]:"),
    ("a zero or negative rung reaches the model", EN,
     "    if not args.lengths or min(args.lengths) < 1:",
     "    if False:"),
    ("the default ladder drops the rung G4 is about", EN,
     "LENGTHS = [1024, 4096, 8192, 16384]",
     "LENGTHS = [1024, 4096, 8192, 12288]"),
    # --- the guard that decides whether the curve means anything ----------------
    ("a hair above the floor counts as reading the state", EN,
     '    readable = first["acc"] >= chance + margin',
     '    readable = first["acc"] >= chance'),
    ("the boundary turns exclusive on the rung exactly at the margin", EN,
     '    readable = first["acc"] >= chance + margin',
     '    readable = first["acc"] > chance + margin'),
    ("the guard reads the longest rung instead of the shortest", EN,
     "    first = rows[0]",
     "    first = rows[-1]"),
    ("the standard error is computed as if n multiplied the noise", EN,
     "    se = (chance * (1 - chance) / max(n_per_length, 1)) ** 0.5",
     "    se = (chance * (1 - chance) * max(n_per_length, 1)) ** 0.5"),
    ("the note is printed when the curve is fine and hidden when it is not", EN,
     '            "note": (None if readable else',
     '            "note": (None if not readable else'),
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
        print(f"baseline copy: green ({len(TESTS)} test file)\n")
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

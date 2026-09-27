"""Mutation battery for the 8c control harness (SPEC §7.1, §9.18).

`bench/eval_scratch.py` prints the one upstream-corpus number this box can produce
without training a model, and every word of its provenance is load-bearing: *random
init*, *the suite's own tokenizer*, *seed 0*, *no gradient step*. A mutation battery is
the only witness available for a harness whose output **is** the artifact, so each entry
below is a way one of those words could stop meaning what it says.

Same discipline as the other batteries: `src`, `tests` and `bench` are copied into a
scratch repo (pyproject pins `pythonpath = ["src"]` against the rootdir, so a mutated
file reached through `PYTHONPATH` silently loses to the real one — §9.12), `data` and
`runs` are symlinked because the tests read both, one textual mutation is applied, and
`tests/test_scratch.py` runs there. The fixture suite in that file is twenty rows, so a
full pass costs seconds rather than the minutes a pilot control run takes.

Two entries are deliberately *absent*, and saying so is the point:

* `model.to(device)` → `pass` is not detectable on a CPU-only box, and this battery runs
  on one. The mutation is real in principle and invisible here.
* `temperature=1.0` → any other positive value cannot move a single figure the harness
  publishes: it divides logits, so `argmax` — and therefore every accuracy — is
  unchanged, and `tests/test_scratch.py` pins exactly that invariance rather than
  pretending to detect it. The `:ece`/`:brier` sidecars do move, and `--metrics-out`
  drops them.

Four entries needed the *instrument* changed rather than the code, and that is worth the
sentence because it was not obvious. "Which of the two draws did this artifact come from"
and "the stored figure is rounded" survived their first runs for one reason each: the
original twelve-row fixture had question-sets of 1, 2 and 4 rows, so every per-set accuracy
was a multiple of 0.1 and a file rounded to one decimal was byte-identical to one that was
not; and seeds 0 and 1 — different weights, same argmaxes — produced *identical* accuracy
maps, so `mean`, `max(macros)`, `seeds[0]` and `seeds[-1]` all denoted the same numbers and
no assertion could tell them apart. The fixture is twenty rows with a 7-row question-set
now, and the two tests that publish a per-seed artifact run three draws. Both conditions
are asserted as guards in `tests/test_scratch.py`, so if the data ever stops expressing a
bug the test says so instead of the battery quietly going green.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
TESTS = ["tests/test_scratch.py"]
SC = "bench/eval_scratch.py"

MUTATIONS = [
    # --- what the weights are --------------------------------------------------
    ("the seed stops drawing weights, so the two-sample control is one draw", SC,
     "    torch.manual_seed(seed)", "    pass"),
    ("the control is built at the config's vocab instead of the suite tokenizer's", SC,
     "        model = MynaModel(MynaConfig(vocab=vocab))",
     "        model = MynaModel(MynaConfig())"),
    ("a checkpoint on the wrong vocab is scored anyway", SC,
     "    if ckpt and model.cfg.vocab != vocab:", "    if False:"),
    # --- which run the published table is --------------------------------------
    ("the markdown table is the last seed's while the metrics file claims the first", SC,
     "    basis = cells_by_seed[args.seeds[0]]", "    basis = cells_by_seed[args.seeds[-1]]"),
    ("the published mean becomes the best seed", SC,
     "    mean = statistics.fmean(macros)", "    mean = max(macros)"),
    # --- the metrics artifact: the trainer's shape, or a claim ------------------
    ("the calibration sidecars leak into the accuracy map", SC,
     "                         if not (k.endswith(\":brier\") or k.endswith(\":ece\"))},",
     "                         if not k.endswith(\":brier\")},"),
    ("the metrics file holds the last seed's numbers", SC,
     "        first = metrics_by_seed[args.seeds[0]]",
     "        first = metrics_by_seed[args.seeds[-1]]"),
    ("the metrics file's seed field names a seed that was not written", SC,
     "            \"seed\": args.seeds[0], \"seeds_run\": args.seeds,",
     "            \"seed\": args.seeds[-1], \"seeds_run\": args.seeds,"),
    ("the stored accuracies are rounded to one decimal", SC,
     "            args.split: {k: round(v, 6) for k, v in first.items()",
     "            args.split: {k: round(v, 1) for k, v in first.items()"),
    ("the witness loses the command that produced it", SC,
     "        f.write(f\"`{cmd}`\\n\\n\")", "        f.write(\"scratch control\\n\\n\")"),
    # --- the refusals ----------------------------------------------------------
    ("a repeated seed counts as two samples", SC,
     "    dupes = sorted({s for s in args.seeds if args.seeds.count(s) > 1})", "    dupes = []"),
    ("a missing tokenizer stops being the suite's problem", SC,
     "    if not tok.exists():", "    if False:"),
    ("a resume file is accepted as a checkpoint", SC,
     "    if ckpt and not (ckpt / \"model.pt\").exists():", "    if False:"),
]


def make_scratch(tmp):
    """The tests import `bench.eval_scratch` and read `data/` + `runs/`, so the copy
    needs all of them — and `runs/` is symlinked, which is why no mutation here is
    allowed to *write* a committed witness (the fixture suite keeps its outputs in
    pytest's tmp dirs)."""
    repo = Path(tmp) / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir()
    for name in ("src", "tests", "bench"):
        shutil.copytree(ROOT / name, repo / name,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    shutil.copy(ROOT / "pyproject.toml", repo / "pyproject.toml")
    for name in ("data", "runs"):
        if (ROOT / name).exists():
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

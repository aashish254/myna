"""Mutation battery for the Kaggle bundle (SPEC §7.1: every gate mutation-checked).

The bundle is the piece this loop cannot run for real — no credentials, no GPU,
no remote — which is exactly why it needs the battery. Everything it promises is
about *not doing something silent*: not running without an experiment name, not
sizing states while a question branch is live, not staging a corpus whose bytes
drifted, not uploading on someone's behalf. A test suite that passes against an
entrypoint that quietly drops `--row-batch` would be worse than no suite.

Same harness as `mutation_paraphrase.py`: `src`, `tests`, `kaggle` and
`pyproject.toml` are copied into a scratch repo (the real one's
`pythonpath = ["src"]` outranks PYTHONPATH — SPEC §9.12) and pytest runs there, so
the mutation under test is the one being imported. `data/` is symlinked because
the packaging tests hash the real pilot corpus.

Three holes are named rather than probed: `if args.dry_run: return 0` mutating to
`if False` *is* caught, by a test that asks for one update so the mutant costs one
update; and `--corpus` discovery inside `/kaggle/input` cannot be witnessed here
because that path only exists on the box — the explicit-flag half is covered, the
glob is not. Third, the worst upload bug — a literal `subprocess.run(["kaggle",
...])` in the packager — is never installed, because the `kaggle` package *is*
present in this venv and a battery has no right to risk publishing to a real
account. The "claims it uploaded" mutant stands in for it, and the same test
catches both halves of the promise: the command must be printed, and the word
"uploaded" must not appear.

The first pass of this battery caught 21/25, and all four survivors were the
tests' fault, not the code's: `WORKING` was only exercised through a monkeypatch
(so a literal `/tmp` default was invisible); `check()`'s sha branch was
distinguishable from a size comparison only by a same-size edit nobody made; and
the build-time verification and the manifest's provenance were never reached,
because every test staged a *successful* copy. Three tests now cover exactly
those, and they are the reason the packager's promises are worth quoting.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
TESTS = ["tests/test_kaggle_bundle.py"]
RUN = "kaggle/run.py"
PKG = "kaggle/package_dataset.py"

MUTATIONS = [
    # --- what the composed command must contain --------------------------------
    ("the entrypoint forgets --row-batch (boolq/mnli would not train)", RUN,
     '"--paraphrase", args.paraphrase, "--row-batch"]', '"--paraphrase", args.paraphrase]'),
    ("the entrypoint trains on the suite wording", RUN,
     'ap.add_argument("--paraphrase", choices=["off", "on"], default="on")',
     'ap.add_argument("--paraphrase", choices=["off", "on"], default="off")'),
    ("accumulation collapses to one set per update (the interference bug)", RUN,
     '"accum_groups": 8,', '"accum_groups": 1,'),
    ("group sampling goes back to uniform (measured: starves the data-rich sets)", RUN,
     '"group_sample": "pool",', '"group_sample": "uniform",'),
    ("the cadence on the box becomes 500 updates", RUN,
     '"save_every": 25,', '"save_every": 500,'),
    ("the drift tolerance on the box becomes 10x", RUN,
     '"stop_factor": 1.5,', '"stop_factor": 10.0,'),
    ("the memory safety margin disappears", RUN,
     '"mem_safety": 0.6,', '"mem_safety": 1.0,'),
    ("the question branch is unbudgeted", RUN,
     '"max_q_cells": 2048,', '"max_q_cells": 8192,'),
    ("the seed is left to chance", RUN,
     '"seed": 0,', '"seed": None,'),
    ("--eval-every is off, so a long run reports nothing mid-way", RUN,
     '"eval_every": 250,', '"eval_every": 0,'),
    # --- the entrypoint's refusals --------------------------------------------
    ("a run without a name silently picks one", RUN,
     "    if not args.name:", "    if False:"),
    ("a half-mounted corpus is accepted", RUN,
     '            raise SystemExit(f"--corpus {p} is missing {\', \'.join(missing)}")',
     "            pass"),
    ("the dry run runs the job", RUN,
     "    if args.dry_run:\n        return 0", "    if args.dry_run and False:\n        return 0"),
    ("--resume is never added, so re-running the cell restarts the job", RUN,
     '    if (out / "model_last.pt").exists():\n        cmd += ["--resume"]',
     "    if False:\n        cmd += [\"--resume\"]"),
    ("--resume is added blindly, even with nothing to resume", RUN,
     '    if (out / "model_last.pt").exists():\n        cmd += ["--resume"]',
     '    if True:\n        cmd += ["--resume"]'),
    ("runs share one directory", RUN,
     'return (Path(out_root) if out_root else WORKING / "runs") / name',
     "return Path(out_root) if out_root else WORKING"),
    ("the output goes somewhere Kaggle does not persist", RUN,
     'WORKING = Path("/kaggle/working")', 'WORKING = Path("/tmp/myna-runs")'),
    ("the stop rule's non-zero exit is swallowed", RUN,
     "    if code:", "    if False:"),
    ("--check never verifies the imports", RUN,
     "        import myna.train  # the real import, not a syntax check (3c)", "        pass"),
    # --- the packager ----------------------------------------------------------
    ("the staged copy is hashed against itself, not against the corpus", PKG,
     'manifest = {"dataset": source.name, "files": inventory(source)}',
     'manifest = {"dataset": source.name, "files": inventory(dest)}'),
    ("verification is reduced to file sizes", PKG,
     "        if sha256(a) != sha256(b):", "        if False:"),
    ("a corrupt stage still reports OK", PKG,
     "    problems = check(dest, source)\n    if args.json:", "    problems = []\n    if args.json:"),
    ("the packager claims it uploaded", PKG,
     '    print(f"upload when you decide to:  kaggle datasets create -p {dest}")',
     '    print(f"uploaded {dest} to kaggle datasets")'),
    ("the dataset metadata omits a split", PKG,
     'if name.endswith(".jsonl") else "json"} for name in FILES + EXTRA]}, indent=2) + "\\n")',
     'if name.endswith(".jsonl") else "json"} for name in FILES[:2]]}, indent=2) + "\\n")'),
    ("the dataset is published to the wrong slug", PKG,
     'DATASET_ID = "aashish254/decision-v2-pilot"', 'DATASET_ID = "someone-else/pilot"'),
]


def make_scratch(tmp):
    repo = Path(tmp) / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir()
    for name in ("src", "tests", "kaggle"):
        shutil.copytree(ROOT / name, repo / name,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    shutil.copy(ROOT / "pyproject.toml", repo / "pyproject.toml")
    (repo / "data").symlink_to(ROOT / "data")
    return repo


def pytest_in(repo):
    env = {k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-q", "-x", "--no-header",
                           "-p", "no:cacheprovider"],
                          capture_output=True, text=True, cwd=repo, env=env, timeout=1800)


def run_one(rel, old, new, tmp):
    repo = make_scratch(tmp)
    path = repo / rel
    text = path.read_text()
    n = text.count(old)
    if n != 1:
        return f"BAD-PATTERN ({n})", ""
    path.write_text(text.replace(old, new))
    r = pytest_in(repo)
    if r.returncode == 0:
        return "SURVIVED", ""
    fails = [ln for ln in r.stdout.splitlines() if ln.startswith("FAILED")]
    return "caught", (fails[0] if fails else r.stdout.strip().splitlines()[-1])[:110]


def main():
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
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__)
        raise SystemExit(0)
    sys.exit(main())

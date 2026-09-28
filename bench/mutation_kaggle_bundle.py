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

One hole is named rather than probed: the worst upload bug — a literal `subprocess.run(
["kaggle`, ...])` in the packager — is never installed, because the `kaggle` package
*is* present in this venv and a battery has no right to risk publishing to a real
account. The "claims it uploaded" mutant stands in for it, and the same test
catches both halves of the promise: the command must be printed, and the word
"uploaded" must not appear.

The other hole used to be described as cheap, and that was wrong. `if args.dry_run:
return 0` mutating to `if args.dry_run and False: return 0` *is* probed, and on
2026-09-28 the battery stopped at 13/40 with a real 10,000-update CPU training run
inside it: the mutant is caught by a test that asserts on the printed command, but the
first such test asked for the *default* dose, so breaking the short-circuit made it
launch what it was checking the print of. The 30-minute timeout then fired, killed
pytest and left the trainer as an orphan on a fan-noising MacBook. Two fixes, both in
this pair of files: `MINI_DOSE` in the test helper caps every uncapped `--dry-run`
(so the mutant now costs one update, which is what the old comment claimed), and
`pytest_in` runs pytest in its own process group so a timeout kills the trainer too.
The cost of the lesson is the two `DEFAULTS` mutants below — a capped print cannot
check the printed defaults, so those moved to a test that reads the module.

`--corpus` discovery used to be on that list too, on the reasoning that
`/kaggle/input` only exists on the box. It is probed now: `find_corpus` reads the
mount from a module constant, so a test can hand it a *symlinked* dataset directory
— which is how Kaggle actually presents one, and which `Path.glob("**")` refuses to
descend into (measured on 3.9, 3.11 and 3.13: pathlib 0 hits, `glob(recursive=True)`
2). The mutant that reverts `_mount_glob` to the pathlib form is caught there, and
was caught by nothing before, because every other test passes `--corpus` explicitly.

The first pass of this battery caught 21/25, and all four survivors were the
tests' fault, not the code's: `WORKING` was only exercised through a monkeypatch
(so a literal `/tmp` default was invisible); `check()`'s sha branch was
distinguishable from a size comparison only by a same-size edit nobody made; and
the build-time verification and the manifest's provenance were never reached,
because every test staged a *successful* copy. Three tests now cover exactly
those, and they are the reason the packager's promises are worth quoting.
"""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
TESTS = ["tests/test_kaggle_bundle.py"]
# The unmutated copy takes ~36s, so ten minutes is ~16x the honest run: enough for a
# mutant that costs one real update, short enough that a hang is reported rather than
# slept through.
PYTEST_TIMEOUT = 600
RUN = "kaggle/run.py"
PKG = "kaggle/package_dataset.py"
CAMP = "kaggle/campaign.py"

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
    # The printed command now carries a capped dose (MINI_DOSE in the test file), so
    # the defaults themselves are asserted off the module. These two mutants exist to
    # prove that test bites: without them, capping the print would have quietly
    # removed the only check on 10,000 steps and a 32-row batch.
    ("the default dose stops being the 10,000 updates the launch docs price", RUN,
     '"steps": 10_000,', '"steps": 500,'),
    ("the default batch forgets it is an upper bound", RUN,
     '"batch": 32,', '"batch": 512,'),
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
    ("corpus discovery stops following the mount's symlinks", RUN,
     'return sorted(Path(p) for p in glob.glob(str(root / "**" / "train.jsonl"),\n'
     '                                             recursive=True))',
     'return sorted(root.glob("**/train.jsonl"))'),
    ("the dry run runs the job", RUN,
     "    if args.dry_run:\n        return 0", "    if args.dry_run and False:\n        return 0"),
    ("--resume is never added, so re-running the cell restarts the job", RUN,
     '    return ["--resume"]', "    return []"),
    ("--resume is added blindly, even with nothing to resume", RUN,
     '    if not (out / "model_last.pt").exists():\n        return []',
     "    if False:\n        return []"),
    ("a finished run is resumed, so the next cell trains at lr ~0", RUN,
     "        if isinstance(reached, int) and reached >= steps:", "        if False:"),
    ("the memory plan's free-bytes pin is dropped", RUN,
     "    if args.free_gib is not None:", "    if False:"),
    ("--warm-start never reaches the trainer", RUN,
     '        cmd += ["--warm-start", str(args.warm_start)]', "        pass"),
    ("--warm-start and --resume are emitted together", RUN,
     '        if (out / "model_last.pt").exists():\n'
     '            raise SystemExit(f"--warm-start with a snapshot at {out}',
     '        if False:\n'
     '            raise SystemExit(f"--warm-start with a snapshot at {out}'),
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
     "    problems = check(dest, source, owner)\n    if args.json:",
     "    problems = []\n    if args.json:"),
    ("the packager claims it uploaded", PKG,
     '    print(f"upload when you decide to:  kaggle datasets create -p {dest}")',
     '    print(f"uploaded {dest} to kaggle datasets")'),
    ("the dataset metadata omits a split", PKG,
     'if name.endswith(".jsonl") else "json"} for name in FILES + EXTRA]}, indent=2) + "\\n")',
     'if name.endswith(".jsonl") else "json"} for name in FILES[:2]]}, indent=2) + "\\n")'),
    ("the staged metadata names a hardcoded account", PKG,
     '{"ownerSlug": owner, "datasetSlug": SLUG,', '{"ownerSlug": "someone-else", "datasetSlug": SLUG,'),
    # --- the notebook cell generator ---------------------------------------------
    ("the setup cell cannot see a symlinked mount", CAMP,
     'os.walk("@INPUT@", followlinks=True)', 'os.walk("@INPUT@")'),
    ("a corpus path with no owner in it", CAMP,
     'f"/kaggle/input/datasets/{owner}/decision-v2-pilot"', '"/kaggle/input/decision-v2-pilot"'),
    ("the cells run the repo from a directory nothing creates", CAMP,
     'STAGED = "/kaggle/working/myna"', 'STAGED = "/kaggle/myna"'),
    ("the generator prices a cell off the guessed rate", CAMP,
     "SECONDS_PER_STEP = 20225 / 3600", "SECONDS_PER_STEP = 1.5"),
    ("the open lane buys 31 GPU hours off a remembered slope", CAMP,
     "+0.0072 per 1,000 updates, which at ", "flat from step 2250, which at "),
    ("the generator emits a flag the runner does not accept", CAMP,
     'f"    --free-gib {free_gib:g}"', 'f"    --context-len {free_gib:g}"'),
    ("the measured-dead ablation goes back into the default lane", CAMP,
     'if v["open"] or include_dead]', "if True]"),
    ("the generated kernel is public", CAMP,
     '"is_private": True,', '"is_private": False,'),
    ("the generated kernel mounts no corpus", CAMP,
     '"dataset_sources": [f"{owner}/myna-code", f"{owner}/decision-v2-pilot"]',
     '"dataset_sources": [],'),
    ("the kernel writes its id without the account it was given", CAMP,
     'f"{owner}/{slug}"', '"someone-else/myna-campaign"'),
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
    # the packager tests hash the real pilot corpus; the cell generator prices its
    # runs off the committed Kaggle log. Neither belongs in a scratch copy twice.
    (repo / "runs").symlink_to(ROOT / "runs")
    return repo


def pytest_in(repo):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    cmd = [sys.executable, "-m", "pytest", *TESTS, "-q", "-x", "--no-header",
           "-p", "no:cacheprovider"]
    # `start_new_session`, so the timeout can kill the *group*. A test that launches
    # training outlives pytest otherwise: on 2026-09-28 the `the dry run runs the job`
    # mutant left a 10,000-update CPU run burning the MacBook after the battery had
    # already died on TimeoutExpired — a battery that reports by orphaning a trainer
    # is a hazard, not a check. A timeout is a caught mutant either way.
    proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out = proc.communicate(timeout=PYTEST_TIMEOUT)[0]
        return subprocess.CompletedProcess(cmd, proc.returncode, out, "")
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        out = proc.communicate()[0]
        out += f"\nTIMEOUT after {PYTEST_TIMEOUT}s -- the whole group was killed"
        return subprocess.CompletedProcess(cmd, 99, out, "")


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

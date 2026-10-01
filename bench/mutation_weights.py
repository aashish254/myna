"""Mutation battery for the install path (SPEC §5 P11).

    python bench/mutation_weights.py

`myna.weights` publishes one claim no external truth can check: *the bytes this
returns are the bytes that number was measured on*. `myna.report` prints a macro
over any `model.pt` it is handed, so every failure mode below is a confident
figure attached to the wrong weights rather than an outage — which is why the
witness is a battery and not a code review.

The lies cluster in three families, each of which is how the feature would
actually rot:

* **Verification going silent.** The digest comparison made unconditional, the
  published digest never read, the refused file left on disk, the manifest not
  written (so the next call has nothing to check against), `cached()` satisfied by
  a file's existence. Each leaves the happy path green.
* **The write being non-atomic.** An empty body kept, a `.part` left beside a
  checkpoint that was refused, `--check` writing into the directory it is auditing.
* **The documented shape changing.** `DEFAULT_TAG` pointing at an arm the registry
  does not quote, a third-party import (the README's install line stops being
  true), the cache rooted inside the package (one `git add -f` from shipping 65 MB
  in every wheel), the manifest's two-space gap becoming one space, which is the
  difference between `sha256sum -c` reading it and silently hashing nothing.

`myna.serve` is here for the same reason in miniature: the two guards added in P11
are the difference between a reader getting `myna-weights --dest DIR` and getting a
torch `FileNotFoundError`, or a traceback about uvicorn.

Same scratch-repo mechanics as `mutation_p5.py`: copy `src` + `bench` + `tests`
(pyproject pins `pythonpath = ["src"]` relative to the rootdir, §9.12), symlink
`data` and `runs`, require a green baseline, abort if the *first* mutation survives,
and report any pattern that did not match exactly once. Deliberately absent: nothing
mutates `_sha256`'s chunk size or the timeouts — those change speed, not a claim.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
TESTS = ["tests/test_weights.py", "tests/test_serve.py", "tests/test_cli_help.py"]

W = "src/myna/weights.py"
SV = "src/myna/serve.py"

MUTATIONS = [
    # --- verification going silent --------------------------------------------
    ("GitHub's published digest is never read, so any bytes are the right ones", W,
     '"sha256": (a.get("digest") or "").removeprefix("sha256:") or None}',
     '"sha256": None}'),
    ("the digest comparison is made unconditional", W,
     '        if spec["sha256"] is not None and digests[name] != spec["sha256"]:',
     '        if spec["sha256"] is not None and False:'),
    ("bytes the release refused stay on disk", W,
     "            path.unlink(missing_ok=True)",
     "            pass"),
    ("a cache is trusted on the files existing alone, not on their hashes", W,
     "    return all((dirn / name).is_file()\n"
     "               and claimed.get(name) == _sha256(dirn / name) for name in ASSETS)",
     "    return all((dirn / name).is_file() for name in ASSETS)"),
    ("a directory with no manifest counts as cached", W,
     "    if not claimed:\n        return False",
     "    if not claimed:\n        return True"),
    ("the manifest is never written, so later reuse has nothing to check", W,
     '    (dirn / MANIFEST).write_text("".join(f"{digests[name]}  {name}\\n" '
     "for name in ASSETS))",
     "    pass"),
    ("the manifest's gap becomes one space, which `sha256sum -c` cannot parse", W,
     'f"{digests[name]}  {name}\\n"',
     'f"{digests[name]} {name}\\n"'),
    ("a release missing an asset is fetched anyway", W,
     "    if missing:\n        raise WeightsError(f\"release {tag} has no ",
     "    if False:\n        raise WeightsError(f\"release {tag} has no "),
    ("the default checkpoint becomes an arm the registry does not quote", W,
     'DEFAULT_TAG = "antiprior_off_s0-weights"',
     'DEFAULT_TAG = "v1b-checkpoint"'),
    # --- the write not being atomic -------------------------------------------
    ("an empty body is kept as a checkpoint", W,
     "    if tmp.stat().st_size == 0:",
     "    if False:"),
    ("a download that died mid-body leaves its partial file behind", W,
     "        tmp.unlink(missing_ok=True)\n        reason =",
     "        reason ="),
    ("--check writes into the directory it is auditing", W,
     "    if args.check:\n        ok = cached(dest)",
     "    if args.check:\n        fetch_weights(args.tag, dest=args.dest, repo=args.repo)\n"
     "        ok = cached(dest)"),
    # --- the documented shape changing ----------------------------------------
    ("the fetcher gains a third-party dependency", W,
     "import argparse\nimport hashlib",
     "import argparse\nimport requests\nimport hashlib"),
    ("the cache moves inside the package", W,
     '    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")',
     "    base = str(Path(__file__).resolve().parent.parent)"),
    # --- serve's two guards ----------------------------------------------------
    ("a wrong --ckpt is left to torch to report", SV,
     "    if missing:\n        raise SystemExit(",
     "    if False:\n        raise SystemExit("),
    ("the missing serve extra is a traceback again", SV,
     "    except ImportError:",
     "    except KeyError:"),
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
        print(f"baseline copy: green ({len(TESTS)} test files)\n")
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

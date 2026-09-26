"""Mutation battery for the P3 memory-plan / stop-rule / resume gate (SPEC §7.1).

Same discipline as `mutation_paraphrase.py`: copy `src` + `tests` into a scratch
repo, apply one textual mutation to the copy, run the gate's test files against
that copy, and report whether they caught it. Anything that survives is a hole in
the gate, so the script exits non-zero on a survivor and aborts immediately if the
*first* mutation survives — that is the signature of a harness that is not testing
the code it mutates (SPEC §9.12).

The mutations are chosen against the failure modes this loop actually paid for: a
projected rate passed off as a budget (§6, the killed M5 run), a batch the box
could not hold, a truncated run that exits 0, and a resume that quietly restarts
the learning-rate decay.

Three mutations are deliberately *absent*, and saying so is the point:
`if free is None: free = free_device_bytes(device)` and `projected = planned * p95 * ... + reserve`
cannot be distinguished from their mutants on this box — the first only matters
where CUDA is present (its own reading is unit-witnessed against a patched driver,
above), the second is equal to its mutant because the clamp assigns
`args.batch = planned` before it is read, and the projected figure is only consumed
where `free_device_bytes` answers. A battery that claims to cover what it cannot
reach is worse than one that names the gap.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
TESTS = ["tests/test_memory_plan.py", "tests/test_device_and_calibration.py"]
TR = "src/myna/train.py"

MUTATIONS = [
    # --- the p95 and the per-position price the whole plan is built from ------
    ("p95 becomes the median: the tail stops being paid for", TR,
     "return lens[min(int(0.95 * (len(lens) - 1)), len(lens) - 1)] if lens else 0",
     "return lens[len(lens) // 2] if lens else 0"),
    ("p95 takes the pool's head instead of a random sample", TR,
     "exs = (rng or random.Random(0)).sample(exs, limit)",
     "exs = exs[:limit]"),
    ("state price swapped for the option-cell price", TR,
     "def rows_that_fit(free_bytes, tokens_p95, safety=0.6, mib_per_position=STATE_MIB_PER_POSITION,",
     "def rows_that_fit(free_bytes, tokens_p95, safety=0.6, mib_per_position=QUESTION_MIB_PER_CELL,"),
    ("the measured 0.8 MiB/position quietly becomes 1.6", TR,
     "STATE_MIB_PER_POSITION = 0.8", "STATE_MIB_PER_POSITION = 1.6"),
    # --- what counts as headroom ---------------------------------------------
    ("a missing reading is filled in with a guess", TR,
     "    if not free_bytes or tokens_p95 <= 0:\n        return None",
     "    if tokens_p95 <= 0:\n        return 8"),
    ("the safety factor is not applied", TR,
     "budget = free_bytes * safety - reserve_bytes", "budget = free_bytes - reserve_bytes"),
    ("a plan that rounds to zero is allowed to say 0", TR,
     "return max(1, int(", "return max(0, int("),
    ("a question branch that eats the budget still gets a batch", TR,
     "    if budget <= 0:\n        return 0", "    if budget <= 0:\n        return 1"),
    ("the reserve is added to the budget instead of taken out of it", TR,
     "budget = free_bytes * safety - reserve_bytes", "budget = free_bytes * safety + reserve_bytes"),
    ("free_device_bytes keys on the wrong device", TR,
     'if device == "cuda" and torch.cuda.is_available():',
     'if device == "mps" and torch.cuda.is_available():'),
    ("cuda reports total bytes as if they were free", TR,
     "            return int(free)\n        except RuntimeError:",
     "            return int(_total)\n        except RuntimeError:"),
    ("an unreadable driver crashes the run instead of reporting None", TR,
     "        except RuntimeError:\n            return None",
     "        except ZeroDivisionError:\n            return None"),
    # --- the stop rule ---------------------------------------------------------
    ("the drift rule judges a spike against the whole history, not the window", TR,
     "step_times[-1] > factor * median(step_times[-window - 1:-1])",
     "step_times[-1] > factor * median(step_times[:-1])"),
    ("the window is short enough to stop on the first slow step", TR,
     "if len(step_times) >= window + 1 and", "if len(step_times) >= 2 and"),
    ("the rule's own window shrinks to nothing", TR,
     "def stop_reason(step_times, free, projected_bytes, factor=1.5, window=20, headroom=1.25):",
     "def stop_reason(step_times, free, projected_bytes, factor=1.5, window=2, headroom=1.25):"),
    ("the shipped --stop-factor default becomes 10, so nothing ever stops", TR,
     'ap.add_argument("--stop-factor", type=float, default=1.5,',
     'ap.add_argument("--stop-factor", type=float, default=10.0,'),
    ("the shipped --save-every default loses all but the last snapshot", TR,
     'ap.add_argument("--save-every", type=int, default=25,',
     'ap.add_argument("--save-every", type=int, default=10_000,'),
    ("the shipped --mem-safety default spends the whole budget", TR,
     'ap.add_argument("--mem-safety", type=float, default=0.6,',
     'ap.add_argument("--mem-safety", type=float, default=1.0,'),
    ("the tolerance is 10x, so drift never stops anything", TR,
     "def stop_reason(step_times, free, projected_bytes, factor=1.5,",
     "def stop_reason(step_times, free, projected_bytes, factor=10.0,"),
    ("headroom ignored: stopping only when already at zero", TR,
     "free < projected_bytes * headroom", "free < projected_bytes"),
    ("the headroom factor is generous instead of tight", TR,
     "def stop_reason(step_times, free, projected_bytes, factor=1.5, window=20, headroom=1.25):",
     "def stop_reason(step_times, free, projected_bytes, factor=1.5, window=20, headroom=0.5):"),
    ("the median is replaced by its maximum", TR,
     "return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])", "return max(s)"),
    ("the projection is compared without its guard, so 0 stops everything", TR,
     "if free is not None and projected_bytes and free < projected_bytes * headroom:",
     "if free is not None and not projected_bytes and free < projected_bytes * headroom:"),
    # --- the clamp actually clamps --------------------------------------------
    ("the clamp only fires at twice the fit", TR,
     "elif args.batch > fit:", "elif args.batch > fit * 2:"),
    ("the question branch is never priced", TR,
     "reserve = args.max_q_cells * QUESTION_MIB_PER_CELL * MI if args.row_batch else 0",
     "reserve = 0"),
    ("the refusal does not name the knob to turn", TR,
     'f"costs at --max-q-cells {args.max_q_cells}; lower --max-q-cells "',
     'f"costs; the step is too large "'),
    ("the plan line hides that a reserve was taken", TR,
     'reserved = f", less {reserve / 1024 / MI:.1f} GiB of question branch" if reserve else ""',
     'reserved = ""'),
    ("the clamp prints a shrink it does not perform", TR,
     "        planned = fit\n", "        planned = args.batch\n"),
    ("the planned batch never reaches the loop", TR,
     "    args.batch = planned\n", "    planned = planned\n"),
    ("the no-reading branch prints a plan it cannot support", TR,
     '    if fit is None:\n        print(f"memory plan: no headroom reading',
     '    if False:\n        print(f"memory plan: no headroom reading'),
    # --- stop: the exit code, the snapshot, the metrics ------------------------
    ("step times are never recorded, so the drift rule is dead", TR,
     "step_times.append(time.time() - t_update)", "pass"),
    ("the stop skips writing the snapshot", TR,
     "            save_snapshot(args.out, model, opt, sched, tok, cfg, step)\n            stopped = reason",
     "            stopped = reason"),
    ("the stop reports a reason and keeps training", TR,
     '                  f"{Path(args.out) / \'model_last.pt\'}", flush=True)\n            break',
     '                  f"{Path(args.out) / \'model_last.pt\'}", flush=True)\n            continue'),
    ("metrics.json is told the run finished", TR,
     '"stopped": stopped, "last_step": last_step,', '"stopped": None, "last_step": last_step,'),
    ("a truncated run exits 0", TR,
     "    if stopped:\n        # a truncated run", "    if False:\n        # a truncated run"),
    # --- cadence and resume ----------------------------------------------------
    ("the cadence counts steps from 0, so it fires one update late", TR,
     "if args.save_every and (step + 1) % args.save_every == 0:",
     "if args.save_every and step % args.save_every == 0:"),
    ("the snapshot drops the optimizer state", TR,
     '"step": step, "optimizer": opt.state_dict(), "scheduler": sched.state_dict()},',
     '"step": step, "scheduler": sched.state_dict()},'),
    ("the snapshot drops the scheduler state", TR,
     '"step": step, "optimizer": opt.state_dict(), "scheduler": sched.state_dict()},',
     '"step": step, "optimizer": opt.state_dict()},'),
    ("the snapshot records no step, so a resume restarts the decay", TR,
     '"step": step, "optimizer"', '"step": 0, "optimizer"'),
    ("resume replays the snapshotted update", TR,
     "start_step = load_snapshot(ckpt, model, opt, sched) + 1",
     "start_step = load_snapshot(ckpt, model, opt, sched)"),
    ("a missing snapshot falls through to torch's own error", TR,
     'raise SystemExit(f"--resume: no {ckpt} to continue from")', "pass"),
    ("load_snapshot ignores the stored step", TR,
     'return int(blob.get("step", 0))', "return 0"),
    ("resume loads the weights but not the optimizer", TR,
     "    if opt is not None and blob.get(\"optimizer\"):\n        opt.load_state_dict(blob[\"optimizer\"])",
     "    if False:\n        opt.load_state_dict(blob[\"optimizer\"])"),
    ("the plan's free bytes never reach metrics", TR,
     '"mem_plan_free_gib": None if free is None else free / 1024 / MI,',
     '"mem_plan_free_gib": None,'),
]


def make_scratch(tmp):
    """A copy of the repo, because `pyproject.toml` pins `pythonpath = ["src"]`
    relative to the rootdir: a mutated `src` reached through PYTHONPATH loses to
    the real one (SPEC §9.12 — that mistake reported 0/29 caught)."""
    repo = Path(tmp) / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir()
    for name in ("src", "tests"):
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
        return f"BAD-PATTERN ({n} matches)", ""
    path.write_text(text.replace(old, new))
    r = pytest_in(repo)
    if r.returncode == 0:
        return "SURVIVED", ""
    fails = [ln for ln in r.stdout.splitlines() if ln.startswith("FAILED")]
    detail = fails[0] if fails else r.stdout.strip().splitlines()[-1:] and r.stdout.strip().splitlines()[-1]
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

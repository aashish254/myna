"""Mutation battery for the `--long-context` trainer's P3 discipline (SPEC §7.1).

    python bench/mutation_train_longctx.py

`tests/test_memory_plan.py` gates the suite trainer, and `bench/mutation_memory_plan.py`
gates what that gate catches. The needle trainer is a *different loop* in the same file —
it had no plan, no cadence, no resume, no stop rule and no `metrics.json` at all until 7b's
preconditions — so its claims get their own gate (`tests/test_train_longctx.py`) and their
own battery here. Copying the suite's mutants across would have proven the suite path twice.

What this battery is about is the three things only this branch of the trainer has:

* **the gradient window as the unit being priced.** Truncated backprop keeps no backward
  activations before the split, so the state length is the wrong axis; sizing on it halves
  the batch the box can carry and the run still finishes, which is the silent direction;
* **the swap ceiling.** The M5's budget is a system budget (§6), so a run that pages is
  already failing; the reading is Darwin-only and the abort is wired into this loop alone;
* **the guaranteed final save.** `model.pt` is what the needle ladder loads, and a normal
  finish used to write nothing but a weights-only `model_last.pt` on the eval cadence.

Two mutations are deliberately *absent*, and saying so is the point. The stop branch's
`save_snapshot(...)` call is byte-identical to the suite path's, so no pattern in this file
can reach it — and on this path a removed stop-save is caught by the tests anyway, because
the guaranteed final save writes the same file; the claim that is newly witnessed is the
non-zero exit, not that save. And `projected = planned * window * ...` cannot be
distinguished from a `projected` built on the full state length by any test that does not
have a CUDA headroom reading to compare against: on CPU and MPS `free_device_bytes` answers
None, so the headroom half of the stop rule is inert here by design (SPEC §6: a projected
rate is not a budget). The suite battery's own header names the same three gaps.

The first pass of this battery reported 22/24, and both survivors became code changes rather
than deletions. "the tail is estimated from one state" survived because `metrics.json`
recorded `LONG_CTX_P95_SAMPLES` — the constant the source asks for, not the number of states
it drew — and this generator pads every state to the same token count, so the tail figure
cannot report its own sample; the field now carries `len(probe)` with the spread beside it.
"resume loads the weights but not the optimizer" survived because every counter in
`metrics.json` comes from the snapshot, so a weights-only resume prints a perfectly correct
dose; the test now reads the schedule's `T_max` (the first command's, not this one's) and
Adam's own step count (restored updates plus executed ones).

Same scratch-repo mechanics as the other batteries: copy `src` + `bench` + `tests`, green
baseline required, abort on a first-mutation survivor, and any pattern that does not match
exactly once is reported rather than skipped.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
TESTS = ["tests/test_train_longctx.py"]

TR = "src/myna/train.py"

MUTATIONS = [
    # --- what gets priced ------------------------------------------------------
    ("the plan prices the whole state instead of the gradient window", TR,
     "    window = min(args.grad_tokens, p95)",
     "    window = p95"),
    ("the plan prints the state length as if it were the window", TR,
     'f"scan + {window} gradient-carrying per row (split at {split})',
     'f"scan + {p95} gradient-carrying per row (split at {split})'),
    ("the record stores the requested length as the measured tail", TR,
     '                   "detached_tokens_p95": detached, "state_tokens_p95": p95,',
     '                   "detached_tokens_p95": detached, "state_tokens_p95": args.long_context,'),
    ("the tail is sampled from the experiment seed, so a replication changes the batch", TR,
     "    probe_rng = random.Random(0)",
     "    probe_rng = random.Random(args.seed)"),
    ("the tail is estimated from one state and the record still says sixty", TR,
     "             for _ in range(LONG_CTX_P95_SAMPLES)]",
     "             for _ in range(1)]"),
    # --- the saves -------------------------------------------------------------
    ("a normal finish leaves no model_last.pt at all", TR,
     "    save_snapshot(out, model, opt, sched, tok, cfg, last_step)\n",
     ""),
    ("the run writes the ladder a different filename than the one it loads", TR,
     '    torch.save({"state_dict": model.state_dict(), "cfg": vars(cfg), "temperature": 1.0},\n'
     '               out / "model.pt")',
     '    torch.save({"state_dict": model.state_dict(), "cfg": vars(cfg), "temperature": 1.0},\n'
     '               out / "model_final.pt")'),
    ("the cadence stops firing, and only prints", TR,
     "        if args.save_every and (step + 1) % args.save_every == 0:\n"
     "            path = save_snapshot(args.out, model, opt, sched, tok, cfg, step)",
     "        if False:\n"
     "            path = save_snapshot(args.out, model, opt, sched, tok, cfg, step)"),
    ("the cadence counts from 0, so it fires one update late", TR,
     "        if args.save_every and (step + 1) % args.save_every == 0:\n"
     "            path = save_snapshot",
     "        if args.save_every and step % args.save_every == 0:\n"
     "            path = save_snapshot"),
    ("the snapshot line names a step it did not write", TR,
     '            print(f"snapshot: {path} at step {step} (every {args.save_every} updates)",',
     '            print(f"snapshot: {path} at step {step + 1} (every {args.save_every} updates)",'),
    ("the 50-update ceiling becomes a 5000-update suggestion", TR,
     "LONG_CTX_SAVE_CEILING = 50",
     "LONG_CTX_SAVE_CEILING = 5000"),
    ("the ceiling is checked and then ignored", TR,
     "    if args.save_every and args.save_every > LONG_CTX_SAVE_CEILING:",
     "    if False:"),
    # --- resume ----------------------------------------------------------------
    # Both loops carry a byte-identical resume block, so every pattern here keeps the
    # `ema = None` that only the long-context one is followed by — otherwise the
    # battery would mutate the suite path and credit this gate with catching it.
    ("resume replays the snapshotted update", TR,
     "        ckpt = Path(args.out) / \"model_last.pt\"\n"
     "        if not ckpt.exists():\n"
     "            raise SystemExit(f\"--resume: no {ckpt} to continue from\")\n"
     "        start_step = load_snapshot(ckpt, model, opt, sched) + 1\n"
     "        print(f\"resume: {ckpt} at step {start_step - 1} -> continuing at {start_step} of \"\n"
     "              f\"{args.steps}; data RNG re-seeded from --seed {args.seed}, so this is a \"\n"
     "              f\"continuation, not a bit-identical replay\", flush=True)\n"
     "    ema = None",
     "        ckpt = Path(args.out) / \"model_last.pt\"\n"
     "        if not ckpt.exists():\n"
     "            raise SystemExit(f\"--resume: no {ckpt} to continue from\")\n"
     "        start_step = load_snapshot(ckpt, model, opt, sched)\n"
     "        print(f\"resume: {ckpt} at step {start_step - 1} -> continuing at {start_step} of \"\n"
     "              f\"{args.steps}; data RNG re-seeded from --seed {args.seed}, so this is a \"\n"
     "              f\"continuation, not a bit-identical replay\", flush=True)\n"
     "    ema = None"),
    ("resume loads the weights but not the optimizer", TR,
     "        start_step = load_snapshot(ckpt, model, opt, sched) + 1\n"
     "        print(f\"resume: {ckpt} at step {start_step - 1} -> continuing at {start_step} of \"\n"
     "              f\"{args.steps}; data RNG re-seeded from --seed {args.seed}, so this is a \"\n"
     "              f\"continuation, not a bit-identical replay\", flush=True)\n"
     "    ema = None",
     "        start_step = load_snapshot(ckpt, model) + 1\n"
     "        print(f\"resume: {ckpt} at step {start_step - 1} -> continuing at {start_step} of \"\n"
     "              f\"{args.steps}; data RNG re-seeded from --seed {args.seed}, so this is a \"\n"
     "              f\"continuation, not a bit-identical replay\", flush=True)\n"
     "    ema = None"),
    # --- the stop rule's own wiring --------------------------------------------
    ("the loop never asks the swap probe", TR,
     "            reason = swap_abort_reason(swap_used_bytes())",
     "            reason = None"),
    ("the swap reading is taken and never aborts", TR,
     "    if swap_bytes is None or swap_bytes <= ceiling_bytes:\n        return None",
     "    if True:\n        return None"),
    ("the swap ceiling is 4 TiB, so a paging box keeps running", TR,
     "LONG_CTX_SWAP_ABORT_GIB = 4.0",
     "LONG_CTX_SWAP_ABORT_GIB = 4096.0"),
    ("a sysctl line with no `used` figure is read as zero", TR,
     "    if not m:\n        return None",
     "    if not m:\n        return 0"),
    ("the swap probe reads a Mac-only budget on every platform", TR,
     '    if sys.platform != "darwin":\n        return None',
     "    if False:\n        return None"),
    ("a truncated run exits 0", TR,
     '    print(f"saved checkpoint to {out}", flush=True)\n    if stopped:',
     '    print(f"saved checkpoint to {out}", flush=True)\n    if False:'),
    ("metrics.json is told the run finished", TR,
     '                   "last_step": last_step, "resumed_from_step": start_step, "stopped": stopped,',
     '                   "last_step": last_step, "resumed_from_step": start_step, "stopped": None,'),
    ("the record says the requested dose ran", TR,
     '                   "steps_requested": args.steps, "updates_executed": updates_executed,\n'
     '                   "last_step": last_step, "resumed_from_step": start_step, "stopped": stopped,',
     '                   "steps_requested": args.steps, "updates_executed": args.steps,\n'
     '                   "last_step": last_step, "resumed_from_step": start_step, "stopped": stopped,'),
    # --help and the flags this path refuses ------------------------------------
    ("--warm-start is accepted and silently ignored", TR,
     '    if args.warm_start:\n        raise SystemExit("--warm-start is not available with '
     '--long-context: this path takes "',
     "    if False:\n        raise SystemExit(\"--warm-start is not available with "
     "--long-context: this path takes \""),
    ("--config scaled names the symbol that was never imported here", TR,
     "            cfg = MynaConfigScaledT4(vocab=vocab)",
     "            cfg = MynaConfigScaled(vocab=vocab)"),
]


def make_scratch(tmp):
    """A copy of the repo, because `pyproject.toml` pins `pythonpath = ["src"]`
    relative to the rootdir: a mutated `src` reached through PYTHONPATH loses to
    the real one (SPEC §9.12 — that mistake reported 0/29 caught)."""
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
    raise SystemExit(main())

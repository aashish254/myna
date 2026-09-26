"""Mutation battery for the G2 latency harness (SPEC §5 P4, §9.18 discipline).

`bench/bench_latency_matched.py` publishes a *ratio* between two vendors' code on
one box. There is no external truth to assert against — the printed table is the
artifact — so the only available witness is that each figure moves when, and only
when, the condition it claims to hold actually holds. Each mutation below is a
plausible way the harness could flatter myna or punish laya without any timer
noticing: answering fewer questions, fitting a slope over an axis it held
constant, calling a plateau "just more state", dropping the truncation label,
dividing the wrong way round, or reporting a ratio at all when laya was never
loaded.

Same scratch-repo mechanics as `mutation_report.py`: `src` + `bench` + `tests`
are copied (`pyproject.toml` pins `pythonpath = ["src"]` relative to the rootdir,
so a mutated file reached through PYTHONPATH silently loses to the real one,
§9.12), `runs/` is symlinked because the CLI smoke test loads `runs/myna-v0`.
Exit non-zero on any survivor or any pattern that did not match exactly once, and
abort on a first-mutation survivor.

Deliberately *absent*: nothing mutates `torch.set_num_threads`, the ladder's
`(1, 10)` question pair, or the rounding in `stats`. Those change the numbers a
run prints without changing any claim the harness makes, and a battery that
counted them would inflate the coverage figure — the count here is of
can-a-lie-get-published, not of-does-the-output-differ. Also absent: the
section-2 bookkeeping inside `main()` (the `or ""` that keeps a clean
`laya_truncated` column out of the artifact), because the CLI smoke test runs
without laya and never reaches it; saying so is cheaper than a fake mutation.

The first pass of this battery scored 30/31. The survivor was real and is now
pinned: `fit_cost` guarded the question axis against a zero-variance column but
not the state axis, so `--ladder 512` would have fitted a per-token slope out of
a single length. That is the same defect the guard exists for, one axis over.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
TESTS = ["tests/test_latency_matched.py"]
LM = "bench/bench_latency_matched.py"

MUTATIONS = [
    # --- the inputs both engines must share -----------------------------------
    ("the question ladder alternates the other way, so q0 is not a choice", LM,
     '    return {f"q{i}": (Q_CHOICE if i % 2 == 0 else Q_NOUL) for i in range(n)}',
     '    return {f"q{i}": (Q_CHOICE if i % 2 == 1 else Q_NOUL) for i in range(n)}'),
    ("every question becomes a 3-option choice: options/question stops matching", LM,
     '    return {f"q{i}": (Q_CHOICE if i % 2 == 0 else Q_NOUL) for i in range(n)}',
     '    return {f"q{i}": Q_CHOICE for i in range(n)}'),
    ("the choice option set shrinks to 2, so laya gets easier work", LM,
     '            "criteria": {"billing": "payments", "technical": "bugs and integrations",\n'
     '                         "sales": "pricing"}}',
     '            "criteria": {"billing": "payments", "technical": "bugs and integrations"}}'),
    ("the state is no longer scaled, so 'same str object' stops being the same content", LM,
     "    return STATE_BODY * reps", "    return STATE_BODY"),
    # --- the answer guard: a skipped question is a free call -------------------
    ("missing answers stop being an error", LM,
     "    missing = set(questions) - set(ans)", "    missing = set()"),
    ("the failure message reports the request size, not what came back", LM,
     '        raise AssertionError(f"{who} answered {len(ans)} of {len(questions)} questions; "',
     '        raise AssertionError(f"{who} answered {len(questions)} of {len(questions)} questions; "'),
    ("the degeneracy check is switched off", LM,
     "    if not non_degenerate(ans):", "    if False and not non_degenerate(ans):"),
    ("an empty answer dict counts as non-degenerate", LM,
     "    return bool(ps) and any(0.0 < p < 1.0 for p in ps)", "    return bool(ps)"),
    ("score answers stop being read, so every score looks dead", LM,
     '        ps += [float(p) for p in a.get("probabilities", {}).values()]',
     "        pass"),
    # --- laya's window, measured from its own token counts --------------------
    ("per-row tokens stop being divided by the row count", LM,
     "    return round(input_tokens / max(1, n_questions), 1)", "    return round(float(input_tokens), 1)"),
    ("the divide loses its guard, so an empty call would crash the run", LM,
     "    return round(input_tokens / max(1, n_questions), 1)",
     "    return round(input_tokens / n_questions, 1)"),
    ("a plateau no longer reads as truncation", LM,
     "    if len(row_tokens) >= 2 and row_tokens[-1] <= row_tokens[-2]:",
     "    if len(row_tokens) >= 2 and row_tokens[-1] < row_tokens[-2]:"),
    ("growing rows get flagged as truncated, killing every honest row", LM,
     "    if len(row_tokens) >= 2 and row_tokens[-1] <= row_tokens[-2]:",
     "    if len(row_tokens) >= 2 and row_tokens[-1] >= row_tokens[-2]:"),
    ("the at-window test moves out of reach", LM,
     "    if row_tokens and row_tokens[-1] >= max_len:", "    if row_tokens and row_tokens[-1] >= max_len * 2:"),
    # --- the timers ------------------------------------------------------------
    ("warm-up calls are dropped, so the first cold pass is the measurement", LM,
     "    for _ in range(warmup):", "    for _ in range(0):"),
    ("percentiles computed on unsorted samples", LM,
     "    t = sorted(times_ms)", "    t = list(times_ms)"),
    ("the percentile index is one past the rank it means", LM,
     "        return t[min(n - 1, max(0, int(round(p / 100 * (n - 1)))))]",
     "        return t[min(n - 1, max(0, int(p / 100 * n)))]"),
    # --- the cost fit: the mutation that already happened once -----------------
    ("a constant state axis is fitted anyway (collinear with the intercept)", LM,
     "    if len(set(Ls)) > 1:", "    if True:"),
    ("a constant question axis is fitted anyway — the -44.9 ms fixed cost", LM,
     "    if len(set(Qs)) > 1:", "    if True:"),
    ("the identifiability guard stops counting degrees of freedom", LM,
     "    if len(rows) < len(names) + 2:", "    if len(rows) < len(names):"),
    ("per-token cost reported in ms as if it were µs", LM,
     '                "per_state_token_us": round(coef.get("L", 0.0) * 1000, 2),',
     '                "per_state_token_us": round(coef.get("L", 0.0), 2),'),
    ("the fit targets the question count instead of the measured time", LM,
     "    y = torch.tensor([r[2] for r in rows], dtype=torch.float64)",
     "    y = torch.tensor([r[1] for r in rows], dtype=torch.float64)"),
    ("a negative intercept stops being flagged", LM,
     '    if coef["fixed"] < 0:', "    if False:"),
    ("an unidentifiable fit prints numbers anyway", LM,
     '    if not v.get("identifiable"):', "    if False and not v.get(\"identifiable\"):"),
    # --- the published ratio ---------------------------------------------------
    ("the end-to-end speedup is inverted", LM,
     '                    "myna_predict_p50_ms": b, "speedup_e2e": round(a / b, 2),',
     '                    "myna_predict_p50_ms": b, "speedup_e2e": round(b / a, 2),'),
    ("the streaming speedup is computed against the wrong baseline", LM,
     '                    "myna_ask_only_p50_ms": c, "speedup_stream": round(a / c, 2)})',
     '                    "myna_ask_only_p50_ms": c, "speedup_stream": round(a / b, 2)})'),
    ("a ratio is printed even when laya never ran", LM,
     "    if laya_by_n:", "    if True:"),
    ("the empty table crashes instead of saying there were no rows", LM,
     "    if not rows:", "    if False:"),
    ("the disclaimer that no accuracy was measured is deleted", LM,
     '          "`latency_matched.json`. Accuracy is not measured here and G1 is not open to "',
     '          "`latency_matched.json`. Accuracy is measured here and G1 is open to "'),
    ("the markdown claims a laya version that was never loaded", LM,
     '    if m["laya_version"]:', "    if True:"),
    ("matched dtype becomes a hard label instead of the reported weight dtype", LM,
     '                     f"weights {m.get(\'laya_dtype\', \'unknown\')}, autocast "',
     '                     f"weights torch.float32, autocast "'),
    # --- one process is a condition, not a description -------------------------
    ("the single-instance guard is never called, so two copies can time together", LM,
     "    acquire_lock(args.out)", "    pass"),
    ("the guard keeps its print but loses its exit", LM,
     "    except OSError:", "    except FileNotFoundError:"),
    ("the lock turns shared: a second copy is admitted", LM,
     "        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)",
     "        fcntl.flock(fh, fcntl.LOCK_SH | fcntl.LOCK_NB)"),
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

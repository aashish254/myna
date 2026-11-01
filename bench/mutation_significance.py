#!/usr/bin/env python
"""Mutation battery for the paired significance harness (`bench/significance.py`).

    uv run python bench/mutation_significance.py

The claims here are printed by code: "the macro reaches G1's target", "laya's macro is above myna's",
"myna's margin over its own floor excludes 0", and the per-cell McNemar table with its exact p-values.
SPEC §9.47 says a claim a program prints is a claim a mutation must be able to break -- and if a broken
harness prints the same four sentences as a working one, the intervals have measured nothing.

The lies cluster in five families.

**The guard stops being an input.** It is the only thing tying the witness to the committed report,
and `tests/test_significance.py` makes it a real test rather than a tautology by expecting the
point estimate from the JSON dump while the harness reads `model.pt` off disk. Loosen the 1e-9 check
to 1e-6 and a cell that moved by a hundredth is "rounding". Remove the `--allow-subset` refusal
when the paired set shrinks and the run writes a witness that does not price the published statistic.
And `if bad:` → `if False:` prints the CI around a different mean -- which is the lie the whole guard
exists to make impossible.

**The resample unit stops being the row.** `pack()` collapses each row to one vector of per-cell sums
and `boot()` multiplies an integer multinomial weight matrix against it, so a replicate can only ever
contain whole rows. Resampling questions instead of rows would give a different digest, a different
number, and the replay test says so. A question-level rule on purpose leaves the cluster key doing
nothing and the interval too narrow -- which is the mistake everyone falls into first when writing
a bootstrap. And the floors frozen at the point estimate would put an interval on the wrong quantity --
the gap would inherit none of the floor's own sampling error, which matters most on `banking77/intent`,
where the floor is 0.043 from 116 rows over 77 labels.

**The pairing stops being symmetric.** `pack()` intersects on `_meta.id` and counts what it dropped,
so McNemar reads only discordance. Removing the intersection makes both sides score every row they have,
which is unpaired and cannot be compared; swapping the two engines changes the sign of the gap but not
the interval width; indexing `lab[i, j, 0]` rather than `lab_ix[key][q["gold"]]` turns a label histogram
into all-zeros and forces the uniform floor to zero everywhere. And counting a pair as concordant even
when both are wrong, or counting a laya-only answer as paired when myna did not answer it, inflates the
discordance pool without adding information.

**The print face is misaligned with the JSON.** The headline CI columns read by column offset, so
changing `:.4f` to `:.2f` passes the tests and a paper reader sees a rounded number; computing `ci(m) - ci(l)`
for the gap instead of `ci(acc_m - acc_l)` gives a wider interval and the wrong coverage; reading the
p-value line by substring instead of by column index lets a `Δ` that prints a level instead of a difference
pass by containing the right digits somewhere; hardcoding "(0 single-row tie flips)" instead of reading
the count from the guard note; and printing "STANDS" whenever the gap is negative (not just when the
upper bound is) would flip a verdict word without changing a confidence value.

**The inputs.** The checkpoint weights loaded but never applied (so the numbers come from the random init),
the tokenizer path read from disk while `--tokenizer`'s file scored the split (this box holds the two
in different directories, so the lie is reachable), and the model temperature taken from the default
instead of the blob (so a doctored checkpoint that records no temperature uses the wrong clamp).

Same scratch-repo mechanics as `mutation_diag_question_ablation.py`: copy `src` + `bench` + `tests` (the
pyproject pins `pythonpath = ["src"]` relative to the rootdir, so a mutated file reached only
through PYTHONPATH silently loses to the unmutated one, §9.12), symlink `data` and `runs`, require a green baseline, abort if the *first* mutation survives, and report any pattern that did not match exactly once.

Deliberately absent: `boot()`'s block size of 1000 (any block that covers a full replicate leaves the summary numbers exactly where they were); `mcnemar_exact()`'s use of `math.lgamma` vs `math.log(math.gamma())`; and computing gap CI as difference of means (the old and new strings are identical, so it survives by definition).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
TESTS = ["tests/test_row_outcomes.py", "tests/test_significance.py"]

AP = "bench/significance.py"
RO = "bench/row_outcomes.py"

MUTATIONS = [
    # --- the guard stops being an input ------------------------------------------
    ("the 1e-9 statistic guard loosens to 1e-6", AP,
     "if want is not None and abs(point[k] - want) > 1e-9:",
     "if want is not None and abs(point[k] - want) > 1e-6:"),
    # --- the resample unit stops being the row -----------------------------------
    ("the weights are drawn from a fixed seed 0", AP,
     "rng = np.random.default_rng(seed)",
     "rng = np.random.default_rng(0)"),
    ("the weights are multinomial over cells, not rows", AP,
     "w = rng.multinomial(N, [1.0 / N] * N, size=hi - lo).astype(np.float32)",
     "w = rng.multinomial(C, [1.0 / C] * C, size=hi - lo).astype(np.float32)"),
    ("the floors are frozen at the point estimate", AP,
     "floor = np.where(nm > 0, lab.max(axis=-1) / np.maximum(nm, 1), np.nan)",
     "# floor frozen at point: np.full((B, C), 0.4331)"),
    ("uniform floor divides by k instead of 1/k", AP,
     'um[i, j] += 1.0 / q["k"]',
     'um[i, j] += q["k"]'),
    ("the gap CI is computed as difference of means", AP,
     '"gap_myna_minus_laya": _nanmean(np.where(both, acc_m - acc_l, np.nan)),',
     '"gap_myna_minus_laya": _nanmean(np.where(both, acc_m, np.nan)) - _nanmean(np.where(both, acc_l, np.nan)),'),
    ("the replicate digest is computed from w[lo]", AP,
     'drawn.append(hashlib.sha256(w.tobytes()).hexdigest()[:16])',
     'drawn.append(hashlib.sha256(w[lo].tobytes()).hexdigest()[:16])'),
    # --- the pairing stops being symmetric ---------------------------------------
    ("the paired set is myna minus laya", AP,
     'ids = sorted(set(by_myna) & set(by_laya))',
     'ids = sorted(set(by_myna) - set(by_laya))'),
    ("the gap CI is two intervals subtracted", AP,
     '"gap_myna_minus_laya": _nanmean(np.where(both, acc_m - acc_l, np.nan)),',
     '"gap_myna_minus_laya": [float(ci(acc_m[:, j])[0] - ci(acc_l[:, j])[1]) for j in range(C)]'),
    ("McNemar counts concordant as discordant too", AP,
     'pairs.append((j, q["ok"], a["ok"], q["gold"]))',
     'pairs.append((j, q["ok"] ^ a["ok"], q["ok"] ^ a["ok"], q["gold"]))'),
    ("a laya-only answer is counted as paired", AP,
     'if a is None:\n                laya_only += 1\n                continue',
     'if a is None: pass'),
    ("the gold label index is 0", AP,
     'lab[i, j, lab_ix[key][q["gold"]]] += 1',
     'lab[i, j, 0] += 1'),
    # --- the print face ----------------------------------------------------------
    ("the gap CI is computed as difference of means", AP,
     '"gap_myna_minus_laya": _nanmean(np.where(both, acc_m - acc_l, np.nan)),',
     '"gap_myna_minus_laya": _nanmean(np.where(both, acc_m, np.nan)) - _nanmean(np.where(both, acc_l, np.nan)),'),
    ("the McNemar p-value column reads a substring", RO,
        'assert cols[7] == "n/a (no discordant pairs)" if j["mcnemar"]["p"] is None else',
        'assert str(j["mcnemar"]["p"])[:4] in cols[7]'),
    ("the verdict word always says REWRITTEN", AP,
     'print(f"  {\'STANDS   \' if c[\'survives\'] else \'REWRITTEN\'} {c[\'claim\']} ·"',
     'print(f"  REWRITTEN      {c[\'claim\']} ·")'),
    ("the survival boolean is inverted", AP,
     '"survives": bool(holds)',
     '"survives": not bool(holds)'),
    # --- the claiming rules                                                      ---
    ("the floor rule uses >= 0", AP,
     'add("floor", f"myna\'s {report[\'split\']}-split macro is above its own majority floor",\n        "margin_over_floor", "CI excludes 0", m[0] > 0)',
     'add("floor", f"myna\'s {report[\'split\']}-split macro is above its own majority floor",\n        "margin_over_floor", "CI ≥ 0", m[0] >= 0)'),
    ("the g1-margin rule uses >", AP,
     'add("g1-margin", f"the margin over the floor reaches G1\'s +{g1[\'margin\']}",\n        "margin_over_floor", f"CI lower bound ≥ {g1[\'margin\']}", m[0] >= g1["margin"])',
     'add("g1-margin", f"the margin over the floor reaches G1\'s +{g1[\'margin\']}",\n        "margin_over_floor", f"CI lower bound > {g1[\'margin\']}", m[0] > g1["margin"])'),
    ("the competitor rule uses the wrong side", AP,
     'add("competitor", "laya\'s macro is above myna\'s on the same rows and cells",\n        "gap_myna_minus_laya", "CI excludes 0 on the negative side", gap[1] < 0)',
     'add("competitor", "laya\'s macro is below myna\'s on the same rows and cells",\n        "gap_myna_minus_laya", "CI excludes 0 on the positive side", gap[0] > 0)'),
    # --- the inputs --------------------------------------------------------------
    ("the checkpoint weights are never applied", RO,
     'model.load_state_dict(blob["state_dict"])',
     "pass"),
    ("the temperature comes from the default", RO,
     '"temperature": meta["temperature"]',
     '"temperature": 1.0'),
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

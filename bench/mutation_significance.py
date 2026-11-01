#!/usr/bin/env python
"""Mutation battery for the paired significance harness (`bench/significance.py`).

    uv run python bench/mutation_significance.py

The claims here are printed by code: "the macro reaches G1's target", "laya's macro is above
myna's", "myna's margin over its own floor excludes 0", and the per-cell McNemar table with its
exact p-values. SPEC §7.1 says a claim a program prints is a claim a mutation must be able to
break -- and if a broken harness prints the same four verdict sentences as a working one, the
intervals have measured nothing and the pre-registered rule adjudicated a fiction.

The lies cluster in six families.

**The guards stop being inputs.** The statistic guard is the only thing tying the interval to the
published number, so loosening `1e-9` to `1e-6` makes a macro that moved by a hundredth "the same
statistic", and dropping `margin_over_floor` from the checked list leaves G1's second half
unguarded while the macro still matches. `if off:` → `if False:` prints a CI around a different
mean, which is the lie the guard exists to make impossible. Removing the `--allow-subset` refusal
prices an interval on a row set smaller than the split the published macro is over, and removing
the empty-intersection stop turns a pairing failure into 20,000 replicates of nothing.

**The resample unit stops being the row.** `pack()` collapses each row to one vector of per-cell
sums and `boot()` multiplies an integer multinomial weight against it, so a replicate can only
contain whole rows. Drawing over cells instead of rows, or freezing the seed, or taking the digest
of one block's first row, each changes what the witness's `first_replicate_digest` claims to pin.
Freezing the floors at the point estimate puts an interval on the wrong quantity: the margin would
inherit none of the floor's own sampling error. Both percentile mutants are about the *coverage*
promise, not the arithmetic -- a full-range interval and a 92.5% interval both print as "95% CI".

**The pairing stops being symmetric.** Intersecting on `_meta.id` is what makes McNemar read only
discordance; the union lets one engine answer a row the other never saw, and the set-difference
drops the whole competitor. Taking the first cell's majority label for every cell quietly changes
the floor claim into a claim about `agnews/is_business`. And a witness that writes `"dropped_rows":
[]` while rows did leave the pairing is a witness that misdescribes its own input.

**The claiming rules.** `>` versus `>=` at each threshold, the floor rule allowed to sit at exactly
0, the competitor rule turned to the other side, and the target rule judged on the CI's *upper*
bound -- the last is the one that would let the sentence "the macro reaches 0.70" survive on the
strength of an interval whose centre is 0.4893.

**The printing face.** Two-decimal rounding in a table whose prose quotes four; the per-cell `Δ`
column printing laya's level instead of the difference; every p-value printed as "n/a (no
discordant pairs)" whatever the counts are; and each verdict word hardcoded -- the markdown line
always saying *stands*, the console line always saying REWRITTEN. Both faces are pinned against
the JSON's own `survives` flags, so a hardcoded word on either side is caught by the test that
reads that side, and the two together are what make the prose and the artifact one claim.

**The dump the intervals are made of (`bench/row_outcomes.py`).** An interval inherits every defect
of the outcomes it resamples. The mirror may disagree with the loader about which groups exist or
how many rows each has; a changed chunk width silently mis-aligns 1,440 predictions against 1,176
rows; `distinct_row` may be reported as the row count when 316 rows collide; the field-by-field
comparison against the report may check only `n`; and the guard that refuses to write a dump which
does not reproduce the committed report may print anyway.

Same scratch-repo mechanics as `mutation_diag_question_ablation.py`: copy `src` + `bench` + `tests`
(the pyproject pins `pythonpath = ["src"]` relative to the rootdir, so a mutated file reached only
through PYTHONPATH silently loses to the unmutated one, §9.12), symlink `data` and `runs`, require a
green baseline, abort if the *first* mutation survives, and report any pattern that did not match
exactly once. Duplicate labels abort the run before the first pytest: two entries with one name are
indistinguishable in the printed count, which is how an earlier pass of this battery reported a
survivor twice and never noticed.

Deliberately absent, because each survives by being behaviourally identical rather than by being
untested: `boot()`'s block size of 1000 (any block that still covers a full replicate leaves every
summary number where it was, and the digest is taken on the whole weight matrix); `_nanmean`'s use
of `np.nanmean` versus a hand-rolled sum-and-count, which agree on this data; `statize()`'s `keep`
versus `both` masks for the two macros, which differ only on cells a replicate emptied -- already
pinned cell by cell by `test_a_cell_a_replicate_empties_is_dropped_from_that_replicate_and_counted`;
`mcnemar_exact()`'s largest-term-downward summation versus `math.comb`, which is exactly what
`test_mcnemar_matches_an_exact_binomial_computed_with_fractions` compares it against, so mutating
it here would pin the same thing twice; and `score()`'s temperature divisor at the call site in
`bench/row_outcomes.py`. The last was measured rather than reasoned: it was in the first version of
this battery, ran, and survived, because a positive scalar leaves every argmax -- and therefore
every `ok` this harness resamples -- exactly where it was, and only the `conf` column moves. Two
entries dropped from that first version were not omissions but inventions:
`model.load_state_dict(blob["state_dict"])` and `"temperature": meta["temperature"]` do not exist
in `bench/row_outcomes.py` -- the checkpoint load lives in `diag_question_ablation.load_run`, where
`mutation_diag_question_ablation.py` already mutates it, and this harness only imports `score()`.
Three of the first pass's survivors were gaps in the *tests*, not in the code, and are closed by
them: `1e-6` was the mutant's own threshold rather than a value inside it, `g1.margin_observed`
could be invented without any statistic disagreeing, and a witness could print
`"unpaired_myna_questions": 0` beside a pairing that had lost a question.
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
    # --- the guards stop being inputs --------------------------------------------
    ("the 1e-9 statistic guard loosens to 1e-6", AP,
     '           if want is not None and abs(point[k] - want) > 1e-9]',
     '           if want is not None and abs(point[k] - want) > 1e-6]'),
    ("G1's margin is not one of the statistics the guard checks", AP,
     '              ("uniform_floor", report["macro"]["uniform"]),\n'
     '              ("margin_over_floor", report["g1"]["margin_observed"])]',
     '              ("uniform_floor", report["macro"]["uniform"])]'),
    ("a statistic that disagrees with the report still prints its interval", AP,
     "    if off:",
     "    if False:"),
    ("the pairing guard lets a subset through by default", AP,
     '    if not args.allow_subset and N != myna["n_rows"]:',
     "    if False:"),
    ("a missing input is read as an empty dict instead of named", AP,
     '    for p in (args.myna, args.laya, args.report):\n        if not Path(p).exists():',
     '    for p in (args.myna, args.laya, args.report):\n        if False:'),
    ("two dumps sharing no row id continue instead of stopping", AP,
     '    if not packed["ids"]:',
     "    if False:"),
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
    ("the interval is the full range of the replicates", AP,
     "    lo, hi = np.percentile(samples, [2.5, 97.5])",
     "    lo, hi = np.percentile(samples, [0.0, 100.0])"),
    ("the interval is a 92.5% interval printed as 95%", AP,
     "    lo, hi = np.percentile(samples, [2.5, 97.5])",
     "    lo, hi = np.percentile(samples, [2.5, 95.0])"),
    ("the replicate digest is computed from w[lo]", AP,
     'drawn.append(hashlib.sha256(w.tobytes()).hexdigest()[:16])',
     'drawn.append(hashlib.sha256(w[lo].tobytes()).hexdigest()[:16])'),
    ("the witness names a digest it never drew", AP,
     'f"(first replicate digest `{res[\'bootstrap\'][\'first_replicate_digest\']}`). "',
     'f"(first replicate digest `0000000000000000`). "'),
    # --- the pairing stops being symmetric ---------------------------------------
    ("the paired set is myna minus laya", AP,
     'ids = sorted(set(by_myna) & set(by_laya))',
     'ids = sorted(set(by_myna) - set(by_laya))'),
    ("the paired set is the union, so one side answers alone", AP,
     'ids = sorted(set(by_myna) & set(by_laya))',
     'ids = sorted(set(by_myna) | set(by_laya))'),
    ("McNemar counts concordant-as-discordant on myna's side", AP,
     "        d[0] += int(ok_m and not ok_l)",
     "        d[0] += int(ok_m or not ok_l)"),
    ("the floor label is the first cell's majority for every cell", AP,
     "        floor_ok = int(gold == maj[j])",
     "        floor_ok = int(gold == maj[0])"),
    ("a laya-only answer is counted as paired", AP,
     'if a is None:\n                laya_only += 1\n                continue',
     'if a is None: pass'),
    ("the gold label index is 0", AP,
     'lab[i, j, lab_ix[key][q["gold"]]] += 1',
     'lab[i, j, 0] += 1'),
    ("the witness claims nothing was unpaired", AP,
     '"unpaired_myna_questions": packed["dropped_questions"],',
     '"unpaired_myna_questions": 0,'),
    ("the witness claims no row left the pairing", AP,
     '"dropped_rows": packed["dropped_rows"],',
     '"dropped_rows": [],'),
    # --- the claiming rules -------------------------------------------------------
    ("the floor rule uses >= 0", AP,
     'add("floor", f"myna\'s {report[\'split\']}-split macro is above its own majority floor",\n'
     '        "margin_over_floor", "CI excludes 0", m[0] > 0)',
     'add("floor", f"myna\'s {report[\'split\']}-split macro is above its own majority floor",\n'
     '        "margin_over_floor", "CI ≥ 0", m[0] >= 0)'),
    ("the g1-margin rule uses >", AP,
     'add("g1-margin", f"the margin over the floor reaches G1\'s +{g1[\'margin\']}",\n'
     '        "margin_over_floor", f"CI lower bound ≥ {g1[\'margin\']}", m[0] >= g1["margin"])',
     'add("g1-margin", f"the margin over the floor reaches G1\'s +{g1[\'margin\']}",\n'
     '        "margin_over_floor", f"CI lower bound > {g1[\'margin\']}", m[0] > g1["margin"])'),
    ("the target rule is judged on the upper bound", AP,
     '        "myna_macro", f"CI lower bound ≥ {g1[\'target\']}", a[0] >= g1["target"])',
     '        "myna_macro", f"CI upper bound ≥ {g1[\'target\']}", a[1] >= g1["target"])'),
    ("the competitor rule uses the wrong side", AP,
     'add("competitor", "laya\'s macro is above myna\'s on the same rows and cells",\n'
     '        "gap_myna_minus_laya", "CI excludes 0 on the negative side", gap[1] < 0)',
     'add("competitor", "laya\'s macro is below myna\'s on the same rows and cells",\n'
     '        "gap_myna_minus_laya", "CI excludes 0 on the positive side", gap[0] > 0)'),
    ("the survival boolean is inverted", AP,
     '"survives": bool(holds)',
     '"survives": not bool(holds)'),
    # --- the printing face --------------------------------------------------------
    ("the headline table rounds the interval to two decimals", AP,
     '        L.append(f"| `{k}` | {p:.4f} | [{lo:.4f}, {hi:.4f}] | {what[k]} |")',
     '        L.append(f"| `{k}` | {p:.2f} | [{lo:.2f}, {hi:.2f}] | {what[k]} |")'),
    ("the per-cell Δ column prints laya's level", AP,
     'f"{c[\'gap\']:+.4f} | {c[\'mcnemar\'][\'b\']}',
     'f"{c[\'laya\']:+.4f} | {c[\'mcnemar\'][\'b\']}'),
    ("every McNemar p-value prints as an absence of evidence", AP,
     '    return "n/a (no discordant pairs)" if x is None else f"{x:.3g}"',
     '    return "n/a (no discordant pairs)"'),
    ("the markdown verdict word always says stands", AP,
     'f"- **{\'stands\' if c[\'survives\'] else \'does not stand\'}** — {c[\'claim\']}. "',
     'f"- **stands** — {c[\'claim\']}. "'),
    ("the console verdict word always says REWRITTEN", AP,
     'print(f"  {\'STANDS   \' if c[\'survives\'] else \'REWRITTEN\'} {c[\'claim\']} · "',
     'print(f"  REWRITTEN {c[\'claim\']} · "'),
    # --- the dump the intervals are made of (bench/row_outcomes.py) ---------------
    ("the mirror and the loader can disagree about which groups exist", RO,
     "    if set(ids) != set(groups):",
     "    if False:"),
    ("a group whose row count differs is accepted", RO,
     "        if len(ids[key]) != len(groups[key][1]):",
     "        if False:"),
    ("a changed chunk width mis-aligns predictions silently", RO,
     "            if [len(c[0]) for c in chunks] != sizes:",
     "            if False:"),
    ("the cluster key is reported unique whatever it collides with", RO,
     '                   "distinct_row": len({r["meta"]["row"] for r in rows}),',
     '                   "distinct_row": len(rows),'),
    ("the report guard prints a dump that disagrees with it", RO,
     '    if bad:\n        for line in bad:\n'
     '            print(f"GUARD FAIL: {line}", file=sys.stderr)\n'
     '        print("nothing was written", file=sys.stderr)\n        return 3',
     '    if False:\n        for line in bad:\n'
     '            print(f"GUARD FAIL: {line}", file=sys.stderr)\n'
     '        print("nothing was written", file=sys.stderr)\n        return 3'),
    ("a field other than accuracy is not compared to the report", RO,
     '        for field in ("n", "majority", "uniform", "type", "sets"):',
     '        for field in ("n",):'),
    ("the dumped rows' macro may differ from the report by a row", RO,
     '    if abs(macro_acc - report["macro"]["acc"]) > 1e-9:',
     '    if abs(macro_acc - report["macro"]["acc"]) > 1e-3:'),
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
    labels = [m[0] for m in MUTATIONS]
    dupes = sorted({l for l in labels if labels.count(l) > 1})
    if dupes:
        print(f"duplicate mutation labels: {dupes} — a counted battery cannot tell two "
              f"identical entries from one, and a survivor would be reported twice")
        return 2
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

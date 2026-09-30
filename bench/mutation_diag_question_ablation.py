"""Mutation battery for the Tier 0 input ablation (`bench/diag_question_ablation.py`).

    python bench/mutation_diag_question_ablation.py

Tier 0's claims are mechanism claims, and they are printed by code: "the option descriptions
are the answer channel", "eight of sixteen cells never read their row's state", "three cells
emit one label and finish on their own floor". SPEC §9.47 quotes all three, with numbers, from
this harness's table and its per-arm lines. §7.1 says a claim a program prints is a claim a
mutation must be able to break -- and if a broken harness prints the same six-arm table as a
working one, the ablation has measured nothing and the next three months of the project are
scheduled around a fiction.

The lies cluster in four families.

**The guard stops being an input.** It is the only thing tying the table to the committed run,
and `tests/test_diag_question_ablation.py` makes it a real test rather than a tautology by
expecting `myna.train.evaluate()`'s numbers while the harness reads `model.pt` off disk. Loosen
`FLOAT_TOL` and a key that moved by a hundredth is "rounding". Turn the one-row slack into an
absolute `1.5` and every wrong split passes. Move `>` to `>=` on `ROW_SLACK_KEYS` and a
six-flip table that should refuse prints anyway -- the second half of that pair is checked, so
the boundary is measured from both sides. Drop the `:brier`/`:ece` skip and the guard compares
calibration sidecars against accuracy keys and refuses a *correct* run (the same class of defect
as A6 running the other way: `keys_compared` counting the whole metrics file instead of the keys
actually compared). Remove either "this key is missing" branch and the run cannot tell a changed
split from a stable one. And `if bad:` → `if False:` prints the table regardless -- which is the
lie the whole guard exists to make impossible.

**The transform stops reaching the numbers.** `score(model, tok, g, ...)` → `groups` is the
single worst entry here: six arms print, six columns fill the table, and every one of them is
the untouched split. The same family: `ablate(groups, arm, args.seed)` → `0`, which makes
`--seed` decoration and two runs incomparable; `coverage(groups, g)` → `coverage(groups,
groups)`, which prints "rewrote 0/716 cues" beside a moving arm and quietly deletes the line that
tells a reader whether an arm fired at all. Then per-arm: the permute bucket key losing the
qname (so `stars` is handed `quality`'s cue and the arm measures question-identity, not
wording), the cross-source arm allowed to donate a source its own cue, its empty-partner guard
removed (a source with no same-width partner would crash instead of reporting itself untested),
`blank-options` emitting two spans whatever the legend holds (the pointer head's `K` is the claim
of that arm), `swap-state` pooled by question-set instead of by source (most sets hold one row, so
the arm becomes the identity while the table still reads "the state is load-bearing"), and that
arm's rebuild indexing `texts[pos]` where the drawn order wants `texts[order[pos]]` -- the same
lie from the other end: the shuffle still runs and the arm still reports a state column, but every
row is handed back its own state, so the table reads "the state is load-bearing" about a transform
that touched nothing. `copy.deepcopy` →
`copy.copy` is the quietest: arm 2 would be measured on the split arm 1 already broke, and since
the guard runs *last*, `as-scored` would still reproduce the committed metrics.

**The printing face.** `worst-d` computed against the floor instead of against `as-scored` (an
ablation table that reads as a leaderboard), the macro block's `Δ` against the floor for the same
reason, the header listing all six arms when `--arms` ran two, the per-cell `state-Δ` printing
the swap arm's level instead of its delta, `collapse_share` read out of the gold histogram
(a cell whose gold is one label then "collapses" whatever the model does -- the field SPEC §9.47
quotes for three named cells), `only_below_floor=True` at the call site hiding collapses that
happen to sit above their floor, the per-arm line denominating the rewrite count in itself, and
the guard note hardcoding `(0 single-row tie flips)`.

**The inputs.** `load_run` is where a run stops being *this* run: the tokenizer digest taken
from `run_dir/tokenizer.json` while `--tokenizer`'s file did the scoring (this box holds the two
in different directories, so the lie is reachable from the published command), the `--tokenizer`
refusal removed (a `FileNotFoundError` traceback instead of a named path), the checkpoint's
weights loaded but never applied, and `step` defaulting to `0` instead of `-1` -- which prints as
"step zero of a training run" for a `model.pt` that records no step at all.

Same scratch-repo mechanics as `mutation_anti_prior.py`: copy `src` + `bench` + `tests` (the
pyproject pins `pythonpath = ["src"]` relative to the rootdir, so a mutated file reached only
through PYTHONPATH silently loses to the unmutated one, §9.12), symlink `data` and `runs`,
require a green baseline, abort if the *first* mutation survives, and report any pattern that
did not match exactly once. The fixture is the load-bearing part of this battery and it is worth
naming, because it is why the first version of these tests would have scored 20/35 for the wrong
reason: the model in `tests/test_diag_question_ablation.py` is **fitted** on a `train.jsonl` the
tests never score. Sized as a random init it answers from the option spans alone, so four of the
five treatment arms produced byte-identical tables to `as-scored` and the entries above about the
transform not reaching the numbers were uncatchable -- a survivor that names no claim, which is
the noise rule `mutation_p5.py` states for `INFER_CHUNK`. `test_the_fitted_model_reads_something_
it_can_lose` pins the sensitivity profile cell by cell, so if a future torch build makes the fit
blunt again the battery says so in the baseline rather than quietly scoring fewer mutants.

Deliberately absent, because each would survive by being behaviourally identical rather than by
being untested: `score()`'s 64-row chunk width and the `temperature` divisor (both leave every
argmax -- hence every accuracy this harness publishes -- exactly where they were; the second is
pinned as invariant by `test_scratch.py`'s equivalent claim, and re-pinning it here would mutate
a thing the code is *correct* to be blind to); `as-scored`'s own `deepcopy` (it changes nothing,
so no downstream number can notice); `if len(keys) < 2: continue` in the permute bucket (a
one-member bucket shuffles to itself, and its rotation fallback is `order[1:] + order[:1]`, which
is that same list); permute reading its cue texts from `out` rather than `groups` (the two are
identical until the assignment loop, three lines later); `swap-state`'s identity-rotation
fallback (the drawn orders are not identity for any source in this fixture -- verified, not
assumed, by the same coverage assertions); and that arm's old `.get(j, exs[j].state)` default,
which the first draft of this battery mutated on the claim that a lone-row source would crash: it
survived, because every `(wf, j)` of a source is in `touched` by construction and no index ever
reached the fallback, so the honest fix was to delete the unreachable default from the harness
rather than score a survivor that names no claim; `roll_up(cell_stats(g), …)` in place of the untouched
`stats` (every arm preserves gold labels and option counts, which is all `stats` contributes to
the roll-up); and `main()`'s `if unmatched:` warning branch, which the guard already refuses
two lines later.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
TESTS = ["tests/test_diag_question_ablation.py"]

AP = "bench/diag_question_ablation.py"

MUTATIONS = [
    # --- the guard stops being an input ------------------------------------------
    ("a key that moved by a hundredth becomes rounding", AP,
     "        if delta <= FLOAT_TOL:",
     "        if delta <= FLOAT_TOL * 1000:"),
    ("the one-row slack is an absolute number, not a row", AP,
     "        if delta <= 1.5 * one_row:",
     "        if delta <= 1.5:"),
    ("a sixth tie flip tips the whole table", AP,
     "    if slack > ROW_SLACK_KEYS:",
     "    if slack >= ROW_SLACK_KEYS:"),
    ("a committed key this split never produced is absorbed", AP,
     '            bad.append(f"{key} is in the committed metrics but this split does not '
     'produce it")\n            continue',
     "            continue"),
    ("a scored key the metrics file lost is never mentioned", AP,
     "    extra = set(arm_acc) - set(metrics)",
     "    extra = set()"),
    ("the calibration sidecars are compared as accuracies", AP,
     '        if key.endswith(":brier") or key.endswith(":ece"):',
     "        if False:"),
    ("the guard counts the metrics file rather than the keys it compared", AP,
     "    return bad, slack, len([k for k in metrics if k in arm_acc])",
     "    return bad, slack, len(metrics)"),
    ("the table prints whatever the guard found", AP,
     "    if bad:",
     "    if False:"),
    ("the ruler is the test block whatever --split said", AP,
     '    published = published_raw[args.split]',
     '    published = published_raw["test"]'),
    # --- the transform stops reaching the numbers --------------------------------
    ("an arm is ablated and the untouched split is scored", AP,
     "        acc, preds, tot = score(model, tok, g, args.device, meta[\"temperature\"])",
     "        acc, preds, tot = score(model, tok, groups, args.device, meta[\"temperature\"])"),
    ("the seed is pinned inside the arms loop", AP,
     "        g = ablate(groups, arm, args.seed)",
     "        g = ablate(groups, arm, 0)"),
    ("the coverage line measures the split against itself", AP,
     "        covs[arm] = coverage(groups, g)",
     "        covs[arm] = coverage(groups, groups)"),
    ("the ablated copy is a shallow one, so arms stack on each other", AP,
     "    out = copy.deepcopy(groups)",
     "    out = copy.copy(groups)"),
    ("an unknown arm names nothing and scores the untouched split", AP,
     '    raise ValueError(f"unknown arm {arm!r}; the arms are {ARMS}")',
     "    return out"),
    ("the permute bucket forgets which question a set asks", AP,
     "            buckets[(source, tuple((q.name, len(q.options)) for q in qs))].append(wf)",
     "            buckets[(source, tuple(len(q.options) for q in qs))].append(wf)"),
    ("a cross-source cue may come from the same source", AP,
     '                others = [(s, text) for s, text in pool[len(q.options)] if s != source]',
     '                others = [(s, text) for s, text in pool[len(q.options)]]'),
    ("a width with no partner is swapped anyway", AP,
     "                if not others:\n                    continue",
     "                if False:\n                    continue"),
    ("blanked options stop matching the option count the head reads", AP,
     '                q.options = [f"option {i + 1}" for i in range(len(q.options))]',
     '                q.options = [f"option {i + 1}" for i in range(2)]'),
    ("the blanking arm keeps every cue", AP,
     "                q.instruction = NEUTRAL_INSTRUCTION",
     "                pass"),
    ("states are swapped inside a question-set, not across a source", AP,
     '            by_source[wf.split("#", 1)[0]].append((wf, j))',
     "            by_source[wf].append((wf, j))"),
    ("the swap-state rebuild is the identity, so the arm scores the split it read", AP,
     "            for pos, (wf, j) in enumerate(slots):\n                touched[wf][j] = texts[order[pos]]",
     "            for pos, (wf, j) in enumerate(slots):\n                touched[wf][j] = texts[pos]"),
    ("every state is reported as rewritten", AP,
     "            if e.state != ae.state:\n                ns += 1",
     "            if True:\n                ns += 1"),
    ("a rewritten cue is counted as a rewritten option set", AP,
     "            if q.instruction != aq.instruction:\n                ni += 1",
     "            if q.instruction != aq.instruction:\n                no += 1"),
    # --- the printing face -------------------------------------------------------
    ("worst-d is a margin over the floor, not a loss against as-scored", AP,
     '        worst = "" if base is None or any(v is None for v in vals) else '
     'f"{min(vals) - base:+.3f}"',
     '        worst = "" if base is None or any(v is None for v in vals) else '
     'f"{min(vals) - s[\'majority\']:+.3f}"'),
    ("the macro block's delta is against the floor too", AP,
     'f"{macro(arm_rows_all[arm], \'majority\'):.4f}  Δ {d - base_macro:+.4f}")',
     'f"{macro(arm_rows_all[arm], \'majority\'):.4f}  Δ '
     '{d - macro(arm_rows_all[arm], "majority"):+.4f}")'),
    ("the header lists arms the run never scored", AP,
     '            + " ".join(f"{a:>10s}" for a in order) + "  floor  unif  worst-d")',
     '            + " ".join(f"{a:>10s}" for a in ARMS) + "  floor  unif  worst-d")'),
    ("a cell outside --min-rows is printed anyway", AP,
     "        if s[\"n\"] < keep_min:\n            continue",
     "        if False:\n            continue"),
    ("state-delta prints the swapped arm's level", AP,
     '        dsw = "" if swap is None or base["acc"] is None else f"{swap - base[\'acc\']:+.3f}"',
     '        dsw = "" if swap is None or base["acc"] is None else f"{swap:+.3f}"'),
    ("a collapse above its own floor is filtered out of the block", AP,
     '    dist = label_distributions(all_preds["as-scored"], scored, only_below_floor=False)',
     '    dist = label_distributions(all_preds["as-scored"], scored, only_below_floor=True)'),
    ("collapse_share is the gold's majority, not the model's", AP,
     '            "collapse_share": (p.most_common(1)[0][1] / n) if n and p else None,',
     '            "collapse_share": (g.most_common(1)[0][1] / n) if n and g else None,'),
    ("the per-arm line denominates the rewrite count in itself", AP,
     'f"rewrote {covs[arm][\'instructions_changed\']:>4d}/{covs[arm][\'question_slots\']}"',
     'f"rewrote {covs[arm][\'instructions_changed\']:>4d}/'
     '{covs[arm][\'instructions_changed\']}"'),
    ("the guard note announces zero tie flips whatever it counted", AP,
     'f"({slack} single-row tie flips) · macro {base_macro:.10f}")',
     'f"(0 single-row tie flips) · macro {base_macro:.10f}")'),
    # --- the inputs --------------------------------------------------------------
    ("the tokenizer digest ignores which file scored the split", AP,
     '                         "tokenizer_sha256": hashlib.sha256(\n'
     '                             tok_path.read_bytes()).hexdigest(),',
     '                         "tokenizer_sha256": hashlib.sha256(\n'
     '                             (run_dir / "tokenizer.json").read_bytes()).hexdigest(),'),
    ("the witness names the default path, not the tokenizer it used", AP,
     '                         "tokenizer_path": str(tok_path),',
     '                         "tokenizer_path": str(run_dir / "tokenizer.json"),'),
    ("a missing tokenizer raises instead of naming itself", AP,
     "    if not tok_path.exists():",
     "    if False:"),
    ("the checkpoint's weights are read and never applied", AP,
     '    model.load_state_dict(blob["state_dict"])',
     "    pass"),
    ("a checkpoint with no step recorded becomes step zero", AP,
     '                         "step": int(blob.get("step", -1)),',
     '                         "step": int(blob.get("step", 0)),'),
    ("a set's accuracy is divided by its sets, not its rows", AP,
     "                    totals[key] += len(exs)",
     "                    totals[key] += 1"),
    ("the prediction chunks are never kept for the emission block", AP,
     '                    preds[key].append((pred[:, n].tolist(), b["gold"][:, n].tolist(),',
     '                    preds[key].append(([], [],'),
    ("the default --min-rows is zero, not the report's floor", AP,
     "    keep_min = args.min_rows or MIN_CELL_ROWS",
     "    keep_min = args.min_rows or 0"),
    ("a --min-rows that keeps nothing runs and crashes on an empty macro", AP,
     '    if not any(s["n"] >= keep_min for s in stats.values()):',
     "    if False:"),
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

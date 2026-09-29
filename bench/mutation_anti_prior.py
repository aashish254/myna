"""Mutation battery for anti-prior batching (`--anti-prior`, SPEC §5 P10).

    python bench/mutation_anti_prior.py

`--anti-prior` re-weights each mini-batch by the inverse label prior of the
(source, question) cells its rows answer, and the claim it carries is written in
SPEC, in `myna.train`'s help text and in the run's `metrics.json`. Every part of
that claim is a number the code computes, which is exactly the shape §7.1 says must
be mutation-checked: if a lie in the machinery prints the same banner as the truth,
the flag is not tested.

The lies cluster in four families, and each is how this mechanism would rot.

**The prior is measured in the wrong place.** The whole design rests on the cell
being `(source, question name)`, pooled across question-sets, and on it being the
*same* aggregation `myna.report` takes the majority floor from (§9.41). Look the
prior up by group key instead and every lookup misses, so all weights come out 1.0
and the run flattens nothing while reporting skewed cells. Divide the label
histogram by the wrong denominator and the shares stop summing to 1, which makes
"the constant answer is worth 0.5" a sentence about a number that is not a share.
Turn `with_labels` off in `cell_priors`, or on in `cell_stats`, and either the
weights cannot be computed at all or the published report JSON grows a key it never
had — the shape every committed artifact's readers depend on.

**The weights stop opposing the prior.** `1/share` becoming `share` is the single
worst lie available here: it *sharpens* the shortcut the flag exists to remove, and
the printed audit lines still look like marginals moving. Same family: the shipped
combination rule (`prod`) silently becoming one of the four kept in `COMBINE` — the
`bench/anti_prior_audit.py` A/B chose it by measurement (0.6184 against mean's
0.7112), so a default that drifts is a decision quietly unmade; the per-source
rescale dropped or taken from one source's mean, which moves the *task mix* while
the help text promises it does not; `flatten_weighted` pairing every row with the
first weight of its group, which re-prices rows against the wrong label and cannot
be seen from any printed number.

**The drawer never reaches a batch.** Both loop paths take a drawer; a flag wired to
only one of them prints the same banner and trains on the old marginals. Same family:
the pick moved *before* the batch-full check, which shifts the rng stream on the
off path and quietly re-orders the data behind every published figure; the off path
switched to sampling with replacement, which repeats one row across a batch and
drives that step's loss to ~0; a `WeightedDraw` whose scale factor, empty-table
guard or distinctness rule goes away.

**The flag's face.** `ANTI_PRIOR_SKEW` at 1.1 makes the reach zero; the reach banner
denominated in skewed cells instead of cells overstates what the corpus offers; the
audit's `>= skew` filter removed puts already-flat cells into the report;
`majority_drawn` read off the natural prior instead of the drawn counts prints a
flattening that was never measured; `max_source_shift` computed against zero draws
calls a hundred-point nothing; the `--long-context` refusal removed lets a run claim
a batching mechanism it cannot run; and `metrics.json` recording `"on"` unconditionally
labels an off run as a treated one.

**What the audit prints.** This is the family the first version of the tests let through.
Both arms carry the trainer's identical `cell majority X -> Y` banner, so if the per-arm
headers go — or a header says `off` over the treated arm's lines — the reader has one
run-on list of marginals and no control. Same family: a marginals table printing the
natural majority twice, so every cell reads as an exact hit; the reach sentence
denominated in skewed cells instead of cells, so six of sixteen becomes six of six; the
mix table's last column switching from treatment-minus-control to treatment-minus-rows,
which is the exact substitution the block exists to avoid (the cell budget moves the mix
in both arms, so only a between-arm difference belongs to the weights); `rows_by_source`
counting sets instead of rows, which moves the denominators of both the weight-sum check
and the mix; the batch-size stat divided by updates instead of mini-batches, so the dose
reads full when the batcher returned short; the `--compare` table's `natural` row read out
of a weighted draw, which makes the shipped rule look like the baseline; the sentence
naming the shipped rule swapped to a rule the same table shows flattening least; the
disclaimer that this is not an accuracy claim replaced by one that says it is; and the
audit drawing through the shared-set batcher instead of `draw_row_batch`, which prices a
mechanism at a dose no run used.

Same scratch-repo mechanics as `mutation_scope_pricing.py`: copy `src` + `bench` +
`tests` (pyproject pins `pythonpath = ["src"]` relative to the rootdir, so a mutated
file reached through PYTHONPATH silently loses to the unmutated one, §9.12), symlink
`data` and `runs`, require a green baseline, abort if the *first* mutation survives,
and report any pattern that did not match exactly once. The mechanism *and* the audit
harness are both mutated. The harness is not incidental to this claim: its printed
marginals are the figures SPEC quotes for P10b, and its per-arm tables are the only
place a reader can tell the control from the treatment. Arithmetic is imported from the
four functions above, so a mutated source shows up there too; what a source mutation
cannot reach is the harness's own face — which lines it labels which arm, which column
it puts in the last slot of the mix table, whether the rule it names as shipped is the
rule its table shows flattening most. §9.46 is why this family exists: tests written
beside a harness inherit its blind spots, and the first version of these tests passed
while the two arms printed as one run-on list.

That section also explains a scratch-repo hazard specific to it. `runs/` is *symlinked*
into the copy, so a test that reads the committed witness sees the real artifact even
when the code that wrote it is mutated — a green test on a lying harness. So the
printing tests re-run `main()` inside the test and assert against output produced there,
and the witness tests stay as the artifact binding rather than the behaviour check.

One mutation is deliberately absent: `flatten_weighted(groups, combine)` dropping its
`combine` argument. Nothing calls it with a non-default rule (the `--compare` A/B goes
through `balance_weights`), so it would survive, and a surviving mutation that names no
claim is noise in the coverage figure — the rule `mutation_p5.py` states for `INFER_CHUNK`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
TESTS = ["tests/test_anti_prior.py"]

RD = "src/myna/real_data.py"
RP = "src/myna/report.py"
TR = "src/myna/train.py"
AP = "bench/anti_prior_audit.py"

MUTATIONS = [
    # --- the prior is measured in the wrong place --------------------------------
    ("the cell prior is looked up by question-set, so every lookup misses", RD,
     "cells = [priors.get((source, q.name), {}) for q in questions]",
     "cells = [priors.get((key, q.name), {}) for q in questions]"),
    ("the label histogram is divided by the number of sets, not rows", RD,
     '{lab: n / c["n"] for lab, n in c["labels"].items()}',
     '{lab: n / c["sets"] for lab, n in c["labels"].items()}'),
    ("cell_priors asks cell_stats not to keep the histogram", RD,
     "cell_stats(groups, with_labels=True)",
     "cell_stats(groups, with_labels=False)"),
    ("cell_stats keeps labels whether or not they were asked for", RP,
     '        if with_labels:\n            c["labels"] = dict(labels)',
     '        if True:\n            c["labels"] = dict(labels)'),
    # --- the weights stop opposing the prior -------------------------------------
    ("a row is weighted by its label's frequency instead of its inverse", RD,
     "terms = [1.0 / cells[i][ex.gold[i]]",
     "terms = [cells[i][ex.gold[i]]"),
    ("the shipped combination rule drifts from product to max", RD,
     'def balance_weights(groups, combine="prod"):',
     'def balance_weights(groups, combine="max"):'),
    ("the per-source rescale is dropped", RD,
     "means = {s: (sum(ws) / len(ws) if ws else 1.0) for s, ws in per_source.items()}",
     "means = {s: 1.0 for s in per_source}"),
    ("every source is rescaled by the first source's mean", RD,
     "return {key: [w / means[source_of_group(key)] for w in ws] for key, ws in raw.items()}",
     "return {key: [w / means[next(iter(means))] for w in ws] for key, ws in raw.items()}"),
    ("a row is paired with its group's first weight", RD,
     "weights.append(ws[key][j])",
     "weights.append(ws[key][0])"),
    # --- the drawer never reaches a batch ---------------------------------------
    ("the row batcher ignores the drawer it was handed", TR,
     "i = rng.randrange(len(items)) if drawer is None else drawer.index(rng)",
     "i = rng.randrange(len(items))"),
    ("the unweighted path is forced through a drawer", TR,
     "i = rng.randrange(len(items)) if drawer is None else drawer.index(rng)",
     "i = drawer.index(rng)"),
    ("the pick is made before the batch-full check, shifting the rng stream", TR,
     "    for _ in range(batch * 40):\n        if len(out) == batch:\n            break\n"
     "        i = rng.randrange(len(items)) if drawer is None else drawer.index(rng)",
     "    for _ in range(batch * 40):\n"
     "        i = rng.randrange(len(items)) if drawer is None else drawer.index(rng)\n"
     "        if len(out) == batch:\n            break"),
    ("the unweighted shared-set path samples with replacement", TR,
     "    if drawer is None:\n        return rng.sample(pool, min(batch, len(pool)))",
     "    if drawer is None:\n        return [pool[rng.randrange(len(pool))] for _ in range(batch)]"),
    ("the weighted shared-set path stops dropping repeats", TR,
     "        i = drawer.index(rng)\n        if i in seen:\n            continue",
     "        i = drawer.index(rng)\n        if False:\n            continue"),
    ("the draw table is read without scaling by its mass", TR,
     "return bisect.bisect_right(self.cum, rng.random() * self.total, hi=self.n - 1)",
     "return bisect.bisect_right(self.cum, rng.random(), hi=self.n - 1)"),
    ("a draw table with no mass is accepted", TR,
     '        if not cum or run <= 0:\n'
     '            raise ValueError(f"draw table has no positive mass ({len(cum)} weights)")',
     "        if False:\n"
     '            raise ValueError(f"draw table has no positive mass ({len(cum)} weights)")'),
    ("the shared-set path is handed no drawer", TR,
     "drawer=group_drawers[wf] if anti else None)",
     "drawer=None)"),
    ("the row-batch path is handed no drawer", TR,
     "q_len_cache, drawer=row_drawer if anti else None)",
     "q_len_cache, drawer=None)"),
    ("the shared-set path builds no per-set drawers", TR,
     "if anti and not args.row_batch else {}",
     "if anti and args.row_batch else {}"),
    # --- the audit and the flag's face ------------------------------------------
    ("the shared-set batches are drawn but never counted", TR,
     "                if audit is not None:\n                    audit.add(questions, exs)",
     "                if False:\n                    audit.add(questions, exs)"),
    ("the row batches are drawn but never counted", TR,
     "                if audit is not None:\n                    audit.add_items(items)",
     "                if False:\n                    audit.add_items(items)"),
    ("flat cells are reported as if they had been flattened", TR,
     "        if not c or max(nat.values()) < skew:",
     "        if not c:"),
    ("the drawn majority is read off the natural prior", TR,
     '"majority_drawn": max(dr.values()),',
     '"majority_drawn": max(nat.values()),'),
    ("the reach banner denominates in skewed cells instead of cells", TR,
     'f"{len(skewed)} of {len(audit.priors)} cells carry a majority label at or above "',
     'f"{len(skewed)} of {len(skewed)} cells carry a majority label at or above "'),
    ("the reach threshold is set where no cell can reach it", RD,
     "ANTI_PRIOR_SKEW = 0.55",
     "ANTI_PRIOR_SKEW = 1.1"),
    ("a drawn share out of zero draws is reported as a mix shift", TR,
     "        if self.rows:  # a drawn share out of zero draws is not a mix, it is a 1.0 shift",
     "        if True:  # a drawn share out of zero draws is not a mix, it is a 1.0 shift"),
    ("the long-context combination is accepted instead of refused", TR,
     '        if args.anti_prior == "on":\n'
     "            # the needle batches are generated row by row inside the step",
     '        if False:\n            # the needle batches are generated row by row inside the step'),
    ("metrics.json says on whatever the flag said", TR,
     '"anti_prior": args.anti_prior, "anti_prior_audit": anti_prior,',
     '"anti_prior": "on", "anti_prior_audit": anti_prior,'),
    # --- what the audit harness prints ------------------------------------------
    ("both arms of the audit print under the control arm's label", AP,
     "(\"weighted: --anti-prior on, combine='prod', the treated arm\", lines_w)):",
     "(\"weighted: --anti-prior off, the control arm\", lines_w)):"),
    ("the control arm's label is carried by the treated arm's lines", AP,
     "(\"uniform: --anti-prior off, the control arm\", lines_u),",
     "(\"uniform: --anti-prior off, the control arm\", lines_w),"),
    ("the arm headers lose which arm they name", AP,
     'print(f"--- {label} ---", flush=True)',
     'print("--- ---", flush=True)'),
    ("a marginals table prints the natural majority as the realized one", AP,
     "{want:8.4f} {got:9.4f}",
     "{want:8.4f} {want:9.4f}"),
    ("the audit's reach sentence denominates in skewed cells", AP,
     "{len(skewed)} of {len(priors)}",
     "{len(skewed)} of {len(skewed)}"),
    ("the mix table's last column becomes treatment minus rows", AP,
     "mix_worst = max(mix_worst, abs(gw - gu))",
     "mix_worst = max(mix_worst, abs(gw - want))"),
    ("the between-arm column prints each arm's own gap to the rows", AP,
     "weighted-uniform {gw - gu:+5.2f}",
     "weighted-uniform {gu - want:+5.2f}"),
    ("rows are counted per question set, so every denominator is a set count", AP,
     "+ len(exs)",
     "+ 1"),
    ("the mean batch stat divides by updates instead of mini-batches", AP,
     "round(rows / (args.updates * args.sets_per_update), 2)",
     "round(rows / args.updates, 2)"),
    ("the shipped rule named under the table is not the rule the table picked", AP,
     'print("shipped rule: prod \u2014 mean/max/geo/sum are kept in COMBINE only so this "',
     'print("shipped rule: max \u2014 prod/geo/sum/mean are kept in COMBINE only so this "'),
    ("the audit's marginals are labelled the accuracy result", AP,
     'print("NOTE: nothing here is an accuracy claim. The test split\'s majority floors are "',
     'print("NOTE: these marginals are the accuracy result. The test split\'s floors are "'),
    ("the audit prices the shared-set batcher instead of the row batcher", AP,
     "picked = draw_row_batch(items, args.batch, args.max_q_cells, rng, tok, cache,\n"
     "                                drawer=drawer)",
     "picked = draw_batch(items, args.batch, rng, drawer=drawer)"),
    ("the compare table's natural row is read out of the prod draw", AP,
     'f"{\'natural\':8s} " + " ".join(f"{max(priors[c].values()):16.4f}" for c in order)',
     'f"{\'natural\':8s} " + " ".join(f"{majority_of(audits[\'prod\'].drawn[c]):16.4f}"'
     ' for c in order)'),
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

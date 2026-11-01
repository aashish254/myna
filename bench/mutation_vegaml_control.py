#!/usr/bin/env python
"""Mutation battery for the vegaml control arm (`bench/vegaml_control.py`, `bench/eval_vegaml_row_outcomes.py`).

    uv run python bench/mutation_vegaml_control.py

The control arm is two thin files, and thin files are where a silent wrong answer lives. One
scores an engine that is not this repository's; the other puts that dump into the second-engine
slot of `bench/significance.py` and prints intervals, claims and a markdown witness. Every number
the paper would quote about vegaml is printed by those two files, so SPEC §7.1 asks the question
in its usual form: which of them can a mutation break?

The lies cluster in six families.

**The guards stop being inputs.** The engine check is the only thing that keeps a laya dump from
being relabelled as vegaml's; the `--allow-subset` refusal prices an interval on a row set smaller
than the split; the 1e-9 check on myna's four statistics is what ties the control's arithmetic to
the published report, and it has four branches — one per statistic — each of which can quietly
stop being asked. A doctored `margin_observed`, a loosened threshold, an `if off:` that never
fires, or a missing input read as `{}` are five ways to print an interval that is not G1's number.

**The slot.** `pack(myna, control, …)` is a positional call: swap it and myna's numbers arrive
under the control's name while every guard still sees *a* consistent pair, because the myna-side
check compares the swapped-in control against the report and the report is myna's. The relabel map
is the other half: drop the gap entry and `point["gap_myna_minus_vegaml"]` is a KeyError; rename
`myna_macro` as well and the harness publishes myna's macro as the competitor's; let `statize`'s
`_`-prefixed per-replicate arrays through and the witness carries numpy in a JSON field.

**The adjudication.** `myna-vs-control` is a two-sided rule and `control-above-myna` is not; read
both from one side and the two sentences become the same sentence, which is the exact reframing
this repository was told not to do about laya. The interval a claim is judged on can also be
replaced by the point estimate, the per-cell Δ column can print a level instead of a difference,
and the pooled McNemar can count concordance as discordance — `b` and `c` then describe a pairing
that never happened while still summing to `n`.

**The witness face.** The markdown table's rows, the four-decimal promise, the two verdict words
(each printed on both faces, independently), the drop list, `bootstrap.n_rows`, and every field of
the provenance block. `control_health` is the sharpest of these, because a fixture whose control is
clean — no errors, no warnings, no argmax disagreements — cannot tell `len(control["errors"])` from
a hardcoded `0`; the test that carries it is the one that hands the harness a dirty dump. The
control's second decline flag is mutated in the same family: `unbound` is tallied out of the dump's
own question outcomes, so a denominator that quietly becomes rows, an overlap that becomes
"everything" and a prose count the JSON never computed are three separate lies.

**The dump's scoring rules (`bench/eval_vegaml_row_outcomes.py`).** The control's accuracy is these
few lines: the `noul`→`boolean` rename, `criteria` only when the suite has it, gold kept out of the
payload, `score` compared as a *positional index* (the one rule that genuinely differs from the
published laya column), `p >= 0.5`, the engine's `choice` field rather than an argmax over its own
probs, and the abstain flag recorded rather than applied. `cell_table` is the geometry the pairing
reads, so `hits += 1` and one shared type-part are both cell-level lies.

**Crash-resume, learned the hard way.** The first full pass paid 880.9 s of inference and then died
twice in the code *after* it: `n_rows_in_file` read a local the live branch never bound, and
`--from-raw` asked the sidecar for a `warnings` key the writer does not emit. No unit test can run
1,176 inferences to find that, so two tests read the source instead — one compares the sidecar's
written keys with the keys the recovery branch subscripts, the other walks the `if args.from_raw:`
statement's two bodies and requires each to bind every name the shared aggregation tail uses. Both
are mutated here, because a static test that no mutant has ever tried to fool is a sentence about
checking.

Deliberately absent, because the only way to reach it is an engine answering 1,176 rows: the rest
of `eval_vegaml_row_outcomes.main()` — the geometry guard, the read-back guard, the `want`/`got` set
comparison, `version_pin`'s shard resolution and the probe's throughput arithmetic. `runs/vegaml_0150_test.rowoutcomes.json`
is what exercises those paths, and the order matters: the first pass reached the geometry guard only
after the two binding defects below were fixed, which is what this file exists to make impossible.

Two entries the first pass of this battery declared unreachable are in it now, because a fixture
reached them after all. The empty-pairing stop is reachable by renaming the control's row ids —
`control_dump` copies myna's ids by construction, so `test_two_dumps_sharing_no_row_id_are_refused`
overwrites them, which is also the shape of the real failure (a control pass over a different split).
And `md_text`'s McNemar paragraph is the harness's own face even though `mcnemar_exact` and `_p` are
`significance.py`'s: `test_md_text_prints_the_mcnemar_counts_from_the_json` compares the sentence with
the JSON's `b`, `c` and p, and the degenerate arm has to print `n/a (no discordant pairs)` rather than
`None`.

Same scratch-repo mechanics as `mutation_significance.py`: copy `src` + `bench` + `tests` (the
pyproject pins `pythonpath = ["src"]` relative to the rootdir, so a mutated file reached only
through PYTHONPATH silently loses to the unmutated one, §9.12), symlink `data` and `runs`, require
a green baseline, abort if the *first* mutation survives, and report any pattern that did not match
exactly once. Duplicate labels abort the run before the first pytest.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
TESTS = ["tests/test_vegaml_control.py"]

VC = "bench/vegaml_control.py"
EV = "bench/eval_vegaml_row_outcomes.py"

MUTATIONS = [
    # --- the guards stop being inputs --------------------------------------------
    ("a dump from any engine is accepted into the vegaml slot", VC,
     '    if control.get("engine") not in ("vegaml",):',
     "    if False:"),
    ("a paired set smaller than the split is not refused", VC,
     '    if not args.allow_subset and N != myna["n_rows"]:',
     "    if False:"),
    ("the 1e-9 check on myna's macro, floors and uniform loosens to 1e-4", VC,
     '           if abs(float(point[k]) - report["macro"][w]) > 1e-9]',
     '           if abs(float(point[k]) - report["macro"][w]) > 1e-4]'),
    ("the 1e-9 check on G1's margin loosens to 1e-4", VC,
     '            float(point["margin_over_floor"]) - report["g1"]["margin_observed"]) > 1e-9:',
     '            float(point["margin_over_floor"]) - report["g1"]["margin_observed"]) > 1e-4:'),
    ("G1's margin is not one of the statistics the guard checks", VC,
     '    if report["g1"]["margin_observed"] is not None and abs(\n'
     '            float(point["margin_over_floor"]) - report["g1"]["margin_observed"]) > 1e-9:\n'
     '        off.append(("margin_over_floor", float(point["margin_over_floor"]),\n'
     '                    report["g1"]["margin_observed"]))',
     "    pass  # margin dropped from the guard"),
    ("a statistic that disagrees with the report still prints its interval", VC,
     "    if off:",
     "    if False:"),
    ("a missing input is read as an empty dict instead of named", VC,
     '    for p in (args.myna, args.control, args.report):\n        if not Path(p).exists():',
     '    for p in (args.myna, args.control, args.report):\n        if False:'),
    ("two dumps sharing no row id are resampled instead of refused", VC,
     '    if not packed["ids"]:',
     "    if False:"),
    # --- the slot ----------------------------------------------------------------
    ("the control is packed into the myna slot", VC,
     '    packed = pack(myna, control, report["cells"], min_rows)',
     '    packed = pack(control, myna, report["cells"], min_rows)'),
    ("the relabel map stops renaming the gap column", VC,
     'RELABEL = {"laya_macro": "vegaml_macro", "gap_myna_minus_laya": "gap_myna_minus_vegaml"}',
     'RELABEL = {"laya_macro": "vegaml_macro"}'),
    ("the relabel map renames myna's own macro into the control's column", VC,
     'RELABEL = {"laya_macro": "vegaml_macro", "gap_myna_minus_laya": "gap_myna_minus_vegaml"}',
     'RELABEL = {"myna_macro": "vegaml_macro", "gap_myna_minus_laya": "gap_myna_minus_vegaml"}'),
    ("statize's per-replicate arrays are published beside the statistics", VC,
     '        if k.startswith("_"):\n            continue',
     '        if False:\n            continue'),
    ("the point estimate keeps the instrument's competitor names", VC,
     "    point_r = relabel(point)",
     '    point_r = {k: float(v) for k, v in point.items() if not k.startswith("_")}'),
    ("the interval map is not relabelled", VC,
     "    ci_map = {RELABEL.get(k, k): ci(scal[k]) for k in scal}",
     "    ci_map = {k: ci(scal[k]) for k in scal}"),
    # --- the resample ------------------------------------------------------------
    ("the resample ignores the seed it was passed", VC,
     "    scal, acc_m, acc_l, kept_cells, digest = boot(packed, args.B, args.seed)",
     "    scal, acc_m, acc_l, kept_cells, digest = boot(packed, args.B, 20261031)"),
    ("the witness names a digest it never drew", VC,
     '                         "first_replicate_digest": digest,',
     '                         "first_replicate_digest": "0000000000000000",'),
    ("the bootstrap block reports the published row count, not the paired one", VC,
     '           "bootstrap": {"B": args.B, "seed": args.seed, "n_rows": N,',
     '           "bootstrap": {"B": args.B, "seed": args.seed, "n_rows": myna["n_rows"],'),
    ("the witness claims no row left the pairing", VC,
     '           "dropped": {"rows": packed["dropped_rows"],',
     '           "dropped": {"rows": [],'),
    ("the paired instance count is myna's own question count", VC,
     '           "paired_question_instances": len(packed["pairs"]),',
     '           "paired_question_instances": myna["n_questions"],'),
    # --- the adjudication --------------------------------------------------------
    ("the difference claim reads only the negative side", VC,
     '"threshold": "CI excludes 0", "survives": gap[0] > 0 or gap[1] < 0}',
     '"threshold": "CI excludes 0", "survives": gap[1] < 0}'),
    ("the parity claim reads the positive side", VC,
     '"threshold": "CI excludes 0 on the negative side", "survives": gap[1] < 0}',
     '"threshold": "CI excludes 0 on the negative side", "survives": gap[0] > 0}'),
    ("a claim is adjudicated on the point estimate instead of its interval", VC,
     '         "ci95": gap, "threshold": "CI excludes 0", "survives": gap[0] > 0 or gap[1] < 0}',
     '         "ci95": [point_r["gap_myna_minus_vegaml"]] * 2, "threshold": "CI excludes 0",'
     ' "survives": gap[0] > 0 or gap[1] < 0}'),
    ("the per-cell Δ column prints the difference the wrong way round", VC,
     '                         "vegaml": lh / n, "gap": (mh - lh) / n,',
     '                         "vegaml": lh / n, "gap": (lh - mh) / n,'),
    ("McNemar's b counts pairs the control also got right", VC,
     "        d[0] += int(ok_m and not ok_c)",
     "        d[0] += int(ok_m and ok_c)"),
    ("McNemar's c counts pairs myna also got right", VC,
     "        d[1] += int(ok_c and not ok_m)",
     "        d[1] += int(ok_c and ok_m)"),
    ("the pooled count loses the side where the control won", VC,
     "        pooled[1] += int(ok_c and not ok_m)",
     "        pooled[1] += 0"),
    # --- the witness face --------------------------------------------------------
    ("the markdown table drops the gap row", VC,
     '    for k in ("myna_macro", "majority_floor", "uniform_floor", "margin_over_floor",\n'
     '              "vegaml_macro", "gap_myna_minus_vegaml"):',
     '    for k in ("myna_macro", "majority_floor", "uniform_floor", "margin_over_floor",\n'
     '              "vegaml_macro"):'),
    ("the markdown table rounds to two decimals", VC,
     '        L.append(f"| `{k}` | {p[k]:.4f} | [{lo:.4f}, {hi:.4f}] |")',
     '        L.append(f"| `{k}` | {p[k]:.2f} | [{lo:.2f}, {hi:.2f}] |")'),
    ("the markdown verdict word always says stands", VC,
     '        L.append(f"- **{\'stands\' if cl[\'survives\'] else \'does not stand\'}** — {cl[\'claim\']} "',
     '        L.append(f"- **stands** — {cl[\'claim\']} "'),
    ("the console verdict word always says SURVIVES", VC,
     '        print(f"  {\'SURVIVES\' if cl[\'survives\'] else \'REWRITTEN\'}  {cl[\'id\']}: {cl[\'claim\']} "',
     '        print(f"  SURVIVES  {cl[\'id\']}: {cl[\'claim\']} "'),
    ("the markdown's McNemar p is a number this file invented", VC,
     '          f"p = {_p(mc[\'p\'])}.",',
     '          f"p = 0.001.",'),
    # --- provenance and health, read off the dump -------------------------------
    ("the control's version is the one this file expects", VC,
     '                       "version": control["vegaml"]["version"],',
     '                       "version": "0.15.0",'),
    ("a control that failed to answer rows reports no failures", VC,
     '                              "errors": len(control.get("errors") or []),',
     '                              "errors": 0,'),
    ("the argmax disagreements are reported as none", VC,
     '                              "argmax_disagreements": control.get("argmax_disagreements"),',
     '                              "argmax_disagreements": 0,'),
    ("the engine's warnings are dropped from the witness", VC,
     '                              "warnings": control.get("warnings")}}',
     '                              "warnings": []}}'),
    ("the abstain block is a clean summary instead of the dump's", VC,
     '           "control_health": {"abstain": control.get("abstain"),',
     '           "control_health": {"abstain": {"n": 0},'),
    ("the unbound rate is divided by rows instead of question outcomes", VC,
     '    return {"n": len(un), "of": len(qs),',
     '    return {"n": len(un), "of": len(rows),'),
    ("every unbound outcome is reported as one that also declined", VC,
     '            "also_abstained": sum(1 for q in un if q.get("abstain")),',
     '            "also_abstained": len(un),'),
    ("the markdown prints an overlap count the JSON never computed", VC,
     """          f"than a confidence one: {ub['n']} of {ub['of']} outcomes, {ub['also_abstained']} of \"""",
     """          f"than a confidence one: {ub['n']} of {ub['of']} outcomes, 0 of \""""),
    # --- the scoring rules of the control's own dump ----------------------------
    ("the shim hands the engine its gold", EV,
     '        d = {"type": TYPE_MAP[t], "instructions": q["instructions"]}',
     '        d = {"type": TYPE_MAP[t], "instructions": q["instructions"], "label": q["label"]}'),
    ("noul reaches vegaml under the suite's own name", EV,
     'TYPE_MAP = {"choice": "choice", "score": "score", "noul": "boolean"}',
     'TYPE_MAP = {"choice": "choice", "score": "score", "noul": "noul"}'),
    ("an unmapped question type is dropped instead of refused", EV,
     '            raise SystemExit(f"{qid}: question type {t!r} has no vegaml counterpart in "',
     "            continue  # mutant: the type is skipped quietly"),
    ("a question with no criteria is sent an empty option set", EV,
     '        if q.get("criteria") is not None:',
     "        if True:"),
    ("the score gold is read as a level text, not the index it is", EV,
     '        ok, disagree = int(pred == int(q["label"])), int(pred != arg)',
     '        ok, disagree = int(pred == q["criteria"][int(q["label"])]), int(pred != arg)'),
    ("the score readout argmins its own distribution", EV,
     '    return int(max(ans["probs"], key=lambda k: ans["probs"][k]))',
     '    return int(max(ans["probs"], key=lambda k: -ans["probs"][k]))'),
    ("the boolean half-line becomes a strict majority", EV,
     '        pred = bool(p >= 0.5)',
     '        pred = bool(p > 0.5)'),
    ("the choice readout is an argmax over probs, not the engine's field", EV,
     '        pred, arg = a["choice"], max(a["probs"], key=a["probs"].get)',
     '        pred, arg = max(a["probs"], key=a["probs"].get), a["choice"]'),
    ("the boolean check stops comparing the field with its own probs", EV,
     "    wrong_side = str(best) != str(pred)",
     "    wrong_side = False"),
    ("the boolean check stops comparing confidence with the table's max", EV,
     "    wrong_conf = conf is not None and abs(float(conf) - max(probs.values())) > 1e-9",
     "    wrong_conf = False"),
    ("the confidence column is flattened", EV,
     '    return ok, pred, (None if conf is None else float(conf)), disagree',
     '    return ok, pred, (None if conf is None else 0.5), disagree'),
    ("a cell counts every instance as a hit", EV,
     '            c["hits"] += q["ok"]',
     '            c["hits"] += 1'),
    ("every type in a cell is folded into the choice part", EV,
     '            p = c["parts"][q["type"]]',
     '            p = c["parts"]["choice"]'),
    # --- crash-resume: the two halves of `--from-raw` ---------------------------
    ("the sidecar stops writing the warnings the reader asks for", EV,
     '                                       "warnings": warned,\n',
     '                                       # warnings not written by the mutant\n'),
    ("the live pass leaves a name the shared aggregation tail reads unbound", EV,
     "        inference_s, seen_n = round(t0, 1), len(seen)",
     "        inference_s = round(t0, 1)"),
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
    missing = [(rel, label) for label, rel, old, _new in MUTATIONS
               if Path(ROOT / rel).read_text().count(old) != 1]
    if missing:
        print("the battery's patterns do not match the files on disk — nothing is run:\n"
              + "\n".join(f"  {rel}: {label}" for rel, label in missing))
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
            print(f"[{i:2d}/{len(MUTATIONS)}] {verdict:11s} {rel.split('/')[-1]}: {label}"
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

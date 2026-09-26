"""Mutation battery for the 3g stratified-reporting gate (SPEC §7.1).

Same discipline as `mutation_memory_plan.py`: copy `src` + `tests` into a scratch
repo (because `pyproject.toml` pins `pythonpath = ["src"]` relative to the rootdir
— a mutated `src` reached through PYTHONPATH silently loses to the real one,
SPEC §9.12), apply one textual mutation, run `tests/test_report.py` against that
copy, and report whether it caught the lie. Exit non-zero on any survivor, and
abort on the *first* survivor: that is the signature of a harness that is not
testing the code it mutates.

Why this gate needs it more than most. A table of floors and accuracies has no
assertion you can run against an external truth — the numbers *are* the artifact —
so the only witness available is that each published figure changes when, and only
when, the row data that produces it changes. Every mutation below is a plausible
mis-reporting: a floor that is a mode of the wrong thing, chance priced at the
first set instead of weighted over all of them, a 2-row group outvoting a 10-row
one, a stratum keyed on the randomized option descriptions, a competitor's 0.0
silently dropped from an average, a cell scored from a different split showing a
model number built on zero rows.

One mutation is deliberately *absent*: `v = g1_verdict(ma, mm if scored else None)`
cannot be distinguished from `g1_verdict(ma, mm)` on any input in this file, because
a missing model number already fails the verdict through `macro_acc is None`. And
`labels = c.pop("labels")` → `c["labels"]` is genuinely equivalent (the key is
private and nothing downstream reads it). Naming those is the point: a battery that
claims to cover what it cannot reach is worse than one that admits the gap.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
TESTS = ["tests/test_report.py"]
RP = "src/myna/report.py"

MUTATIONS = [
    # --- the majority floor: what is a mode of? ---------------------------------
    ("a cell's row count becomes its last set's row count", RP,
     "            c[\"n\"] += len(examples)", "            c[\"n\"] = len(examples)"),
    ("the set count reports rows, so 'sets' stops meaning sets", RP,
     "            c[\"sets\"] += 1", "            c[\"sets\"] += len(examples)"),
    ("the label histogram counts each distinct label once", RP,
     "                c[\"labels\"][e.gold[i]] += 1", "                c[\"labels\"][e.gold[i]] = 1"),
    ("the majority floor becomes the minority label", RP,
     "        c[\"majority\"] = (max(labels.values()) / c[\"n\"]) if c[\"n\"] and labels else 0.0",
     "        c[\"majority\"] = (min(labels.values()) / c[\"n\"]) if c[\"n\"] and labels else 0.0"),
    ("the majority floor becomes 1.0, i.e. it always wins", RP,
     "        c[\"majority\"] = (max(labels.values()) / c[\"n\"]) if c[\"n\"] and labels else 0.0",
     "        c[\"majority\"] = (sum(labels.values()) / c[\"n\"]) if c[\"n\"] and labels else 0.0"),
    # --- the uniform floor: chance over which option counts? --------------------
    ("the option mix counts sets instead of rows", RP,
     "            c[\"options\"][len(q.options)] += len(examples)",
     "            c[\"options\"][len(q.options)] += 1"),
    ("chance keeps only the last set's option count", RP,
     "            c[\"chance\"] += len(examples) / max(len(q.options), 1)",
     "            c[\"chance\"] = len(examples) / max(len(q.options), 1)"),
    ("chance ignores the option count entirely", RP,
     "            c[\"chance\"] += len(examples) / max(len(q.options), 1)",
     "            c[\"chance\"] += len(examples)"),
    ("the uniform floor is priced at the widest set, unweighted", RP,
     "        c[\"uniform\"] = (c[\"chance\"] / c[\"n\"]) if c[\"n\"] else 0.0",
     "        c[\"uniform\"] = (1.0 / max(c[\"options\"])) if c[\"options\"] else 0.0"),
    ("the floors are swapped in the row the table prints", RP,
     "                     \"sets\": s[\"sets\"], \"majority\": s[\"majority\"], \"uniform\": s[\"uniform\"],",
     "                     \"sets\": s[\"sets\"], \"majority\": s[\"uniform\"], \"uniform\": s[\"majority\"],"),
    # --- weighting: which set gets to vote? -------------------------------------
    ("a scored set's weight becomes 1 row, so singletons outvote", RP,
     "        n = len(groups[wf][1]) if wf in groups else 0", "        n = 1 if wf in groups else 0"),
    ("the cell mean drops its row weights", RP,
     "        out[k] = {\"acc\": (sum(v * n for v, n in pairs) / tot) if tot",
     "        out[k] = {\"acc\": (sum(v for v, _n in pairs) / len(pairs)) if tot"),
    ("the n a cell reports is the number of sets, not their rows", RP,
     "                  \"n\": tot, \"sets\": len(pairs)}", "                  \"n\": len(pairs), \"sets\": len(pairs)}"),
    ("a scored set missing from the split is absorbed silently", RP,
     "            unmatched.append(key)", "            pass"),
    ("the key is split the wrong way round", RP,
     "        wf, qname = key.split(\"/\", 1)", "        qname, wf = key.split(\"/\", 1)"),
    ("only the Brier sidecar is filtered, not the ECE one", RP,
     "        if key.endswith(\":brier\") or key.endswith(\":ece\"):",
     "        if key.endswith(\":brier\"):"),
    # --- strata: instruction strings vs the randomized signature ----------------
    ("strata key on the question name instead of its instruction", RP,
     "            p[\"instr_rows\"][q.instruction] += len(examples)",
     "            p[\"instr_rows\"][q.name] += len(examples)"),
    ("a slot counts as one even when the row asks five questions", RP,
     "            p[\"instr_rows\"][q.instruction] += len(examples)",
     "            p[\"instr_rows\"][q.instruction] += 1"),
    ("the per-source row count becomes its last set's", RP,
     "        p[\"rows\"] += len(examples)", "        p[\"rows\"] = len(examples)"),
    ("reused-set rows count once per set", RP,
     "            p[\"multi_rows\"] += len(examples)", "            p[\"multi_rows\"] += 1"),
    ("every instruction counts as repeated", RP,
     "        repeated = sum(n for n in p[\"instr_rows\"].values() if n > 1)",
     "        repeated = sum(n for n in p[\"instr_rows\"].values() if n >= 1)"),
    ("repetition counts distinct instructions instead of slots", RP,
     "        repeated = sum(n for n in p[\"instr_rows\"].values() if n > 1)",
     "        repeated = sum(1 for n in p[\"instr_rows\"].values() if n > 1)"),
    ("the share divides by rows, so it can exceed 1.0 (the first pass did)", RP,
     "        share = repeated / slots if slots else 0.0", "        share = repeated / p[\"rows\"] if p[\"rows\"] else 0.0"),
    ("the set-reuse share divides by sets", RP,
     "                       \"multi_row_set_share\": p[\"multi_rows\"] / p[\"rows\"] if p[\"rows\"] else 0.0,",
     "                       \"multi_row_set_share\": p[\"multi_rows\"] / p[\"sets\"] if p[\"sets\"] else 0.0,"),
    ("the stratum threshold becomes permissive enough to fold mnli in", RP,
     "                       \"class\": \"shared-instruction\" if share >= 0.5 else \"per-row-instruction\"}",
     "                       \"class\": \"shared-instruction\" if share >= 0.05 else \"per-row-instruction\"}"),
    ("distinct instructions reported as the set count", RP,
     "                       \"distinct_instructions\": len(p[\"instr_rows\"]),",
     "                       \"distinct_instructions\": p[\"sets\"],"),
    ("the set count reported as the instruction count", RP,
     "        out[source] = {\"rows\": p[\"rows\"], \"sets\": p[\"sets\"], \"slots\": slots,",
     "        out[source] = {\"rows\": p[\"rows\"], \"sets\": len(p[\"instr_rows\"]), \"slots\": slots,"),
    # --- keeping, dropping, averaging ------------------------------------------
    ("the minimum-rows filter is inverted", RP,
     "        if s[\"n\"] < keep_min:", "        if s[\"n\"] > keep_min:"),
    ("the shipped floor of 30 rows per cell becomes 4", RP,
     "MIN_CELL_ROWS = 30", "MIN_CELL_ROWS = 4"),
    ("a cell scored only by unknown sets shows a model number", RP,
     "                     \"acc\": a[\"acc\"] if a and a[\"n\"] else None,",
     "                     \"acc\": a[\"acc\"] if a else None,"),
    ("n_scored and n collapse into each other", RP,
     "                     \"n_scored\": a[\"n\"] if a else 0})",
     "                     \"n_scored\": s[\"n\"] if a else 0})"),
    ("a macro drops zeros, so the competitor's 0.0 stops counting", RP,
     "    vals = [r[field] for r in rows if r[field] is not None]", "    vals = [r[field] for r in rows if r[field]]"),
    ("an empty macro reports 0.0 instead of 'nothing was scored'", RP,
     "    vals = [r[field] for r in rows if r[field] is not None]\n    return sum(vals) / len(vals) if vals else None",
     "    vals = [r[field] for r in rows if r[field] is not None]\n    return sum(vals) / len(vals) if vals else 0.0"),
    ("the stratum laya macro drops zeros too", RP,
     "    vals = [v for v in (values.get((r[\"source\"], r[\"question\"])) for r in rows) if v is not None]",
     "    vals = [v for v in (values.get((r[\"source\"], r[\"question\"])) for r in rows) if v]"),
    # --- the verdict and the table ---------------------------------------------
    ("-maj is computed against chance, not against the majority floor", RP,
     "            d = None if r[\"acc\"] is None else r[\"acc\"] - r[\"majority\"]",
     "            d = None if r[\"acc\"] is None else r[\"acc\"] - r[\"uniform\"]"),
    ("G1's target is lowered to 0.50", RP,
     "def g1_verdict(macro_acc: float | None, macro_majority: float | None, target=0.70,",
     "def g1_verdict(macro_acc: float | None, macro_majority: float | None, target=0.50,"),
    ("G1's margin over the floor is lowered to 0.05", RP,
     "               margin=0.15) -> dict:", "               margin=0.05) -> dict:"),
    ("G1 becomes the target alone", RP,
     "            \"pass\": hit and macro_acc >= target and (macro_acc - macro_majority) >= margin,",
     "            \"pass\": hit and macro_acc >= target,"),
    ("a missing floor counts as a floor of nothing at all", RP,
     "    hit = macro_acc is not None and macro_majority is not None",
     "    hit = macro_acc is not None or macro_majority is not None"),
    ("every number in the table rounds to one decimal", RP,
     "def _fmt(x, nd=3):\n    return \"  —  \" if x is None else f\"{x:.{nd}f}\"",
     "def _fmt(x, nd=3):\n    return \"  —  \" if x is None else f\"{x:.1f}\""),
    ("the laya column prints the sample size instead of the accuracy", RP,
     "    laya_acc = {k: v[\"acc\"] for k, v in (laya or {}).items()}",
     "    laya_acc = {k: v[\"n\"] for k, v in (laya or {}).items()}"),
    # --- the CLI ----------------------------------------------------------------
    ("a run scored on test is read out of the dev block", RP,
     "    key = \"test\" if args.split in (\"test\",) else (\"dev\" if args.split in (\"dev\", \"development\")",
     "    key = \"dev\" if args.split in (\"test\",) else (\"dev\" if args.split in (\"dev\", \"development\")"),
    ("a run directory is no longer accepted, only the file inside it", RP,
     "    if metrics_path.is_dir():\n        metrics_path = metrics_path / \"metrics.json\"",
     "    if False:\n        metrics_path = metrics_path / \"metrics.json\""),
    ("a missing split turns into a stack trace instead of a refusal", RP,
     "    if not path.exists():\n        raise SystemExit", "    if False:\n        raise SystemExit"),
    ("--min-rows 0 is allowed, so every cell joins the G1 average", RP,
     "    if not (0 < args.min_rows):\n        raise SystemExit", "    if args.min_rows < 0:\n        raise SystemExit"),
    ("the report never says the table has no model in it", RP,
     "    if not scored:\n        print(f\"no cell in {metrics_path.name}",
     "    if scored:\n        print(f\"no cell in {metrics_path.name}"),
    ("the mixed-option-count note only fires on three or more counts", RP,
     "    mixed = [r for r in rows if len(r[\"options_by_size\"]) > 1]",
     "    mixed = [r for r in rows if len(r[\"options_by_size\"]) > 2]"),
    ("an unweighted question-set is quietly kept out of the note", RP,
     "    if unmatched:\n        print(f\"note: {len(unmatched)} scored question-sets",
     "    if False:\n        print(f\"note: {len(unmatched)} scored question-sets"),
    ("the floors vanish with the model instead of describing the split", RP,
     "    basis = scored if scored else rows", "    basis = scored"),
    ("the report stops saying which basis the floors were averaged over", RP,
     "    extra = \"\" if len(scored) == len(rows) else (", "    extra = \"\" if len(scored) <= len(rows) else ("),
    ("the laya banner drops the split it was measured on", RP,
     "        print(f\"laya: {args.laya} ({ldata['split']} split, n={ldata['n']}, \"",
     "        print(f\"laya: {args.laya} (n={ldata['n']}, \""),
    ("the JSON artifact loses the strata it printed", RP,
     "                                   \"strata\": cls, \"cells\": rows,", "                                   \"cells\": rows, \"cells\": rows,"),
    ("the JSON artifact claims nothing went unmatched", RP,
     "                                   \"dropped_cells\": dropped, \"unmatched_sets\": unmatched,",
     "                                   \"dropped_cells\": dropped, \"unmatched_sets\": [],"),
    ("the JSON artifact records a verdict of pass", RP,
     "                                   \"g1\": v}, indent=2) + \"\\n\")",
     "                                   \"g1\": {**v, \"pass\": True}}, indent=2) + \"\\n\")"),
]


def make_scratch(tmp):
    """`tests/test_report.py` runs the CLI with `cwd` at the repo root and reads
    `data/decision-v2-pilot` and `runs/` through it, so the copy needs both."""
    repo = Path(tmp) / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir()
    for name in ("src", "tests"):
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
    fails = [ln for ln in r.stdout.splitlines() if ln.startswith("FAILED")
             or ln.startswith("ERROR")]
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
    sys.exit(main())

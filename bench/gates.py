#!/usr/bin/env python
"""The release gate: G1-G7 each met, not met, or open, and the artifact that says so.

SPEC §5's closing line makes the release condition "G1-G7 all pass or all explicitly
marked not met", so 8d is a verdict table rather than more measurement. A verdict table is
the easiest artifact to lie with — three words per row, no numbers — so every cell here is
tied to something that can contradict it:

* the `rows` a gate cites are `bench/reproduce.py` rows, imported and re-checked rather
  than restated, so a gate cannot be greener than the witness behind it;
* the `proofs` are read out of committed JSON artifacts by key path, and a boolean proof
  must *agree* with the verdict it is cited for — a `met` gate may not rest on a field
  that prints `false`, and a `not met` gate may not rest on a feeling;
* `met` and `not met` each owe at least one committed witness carrying a figure;
* `open` is only allowed for a gate that cites a row this box cannot reproduce at all
  (`gated-kaggle` or `retrain`). "open" names a missing artifact, not missing effort.

That last distinction is load-bearing and easy to lose. G1 already has a committed number
below its threshold — the untrained control's 0.346 — and is still `open`, because the run
that decides it happens on a corpus this box does not train on. Calling that `not met`
would report the control as if it were the checkpoint; calling the whole gate set `met`
because three rows are green would be the marketing this file exists to refuse.

The prose is checked, never generated: `--check` parses SPEC §2.2's status column and
reds if any cell's leading words disagree with this file, so a verdict has to be changed
in two places that are not allowed to disagree. README's block *is* generated from here,
and per §9.31 that block is therefore not evidence — §2.2 and the artifacts are, and those
are what `--check` reads.

    uv run python bench/gates.py --check
    uv run python bench/gates.py --print
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))

from reproduce import (KAGGLE, RETRAIN, ROWS, check_row)  # noqa: E402

MET, NOT_MET, OPEN = "met", "not met", "open"
VERDICTS = (NOT_MET, MET, OPEN)          # NOT_MET first: "not met" contains "met"
UNREPRODUCIBLE = (KAGGLE, RETRAIN)       # statuses no local run can regenerate
SPEC_GATE_TABLE = "SPEC.md"
BEGINS, ENDS = "<!-- gates:release:begin -->", "<!-- gates:release:end -->"

# A proof may name the registry itself instead of a file: G7's claim *is* that the
# checker is green, so it is read live rather than copied out of a log that would go
# stale on the next commit.
REGISTRY = "@registry"


def proof(file, path, expect, what):
    """One falsifiable backing for a verdict.

    `path` is a slash-separated key path into a committed JSON artifact (or, for
    `REGISTRY`, an unused placeholder). `expect` is the exact value the field must still
    hold. `what` is the sentence the reader gets when it does not.
    """
    return {"file": file, "path": path, "expect": expect, "what": what}


def gate(id, name, condition, verdict, rows, proofs, reads, note):
    return {"id": id, "name": name, "condition": condition, "verdict": verdict,
            "rows": list(rows), "proofs": list(proofs), "reads": reads, "note": note}


GATES = [
    gate("G1", "Accuracy beats the witness",
         "macro decision-v2 test ≥ 0.70 on the frozen split, per-source table published, "
         "≥ +0.15 over the 0.4331 majority floor",
         OPEN,
         ["g1-v1", "scratch-control", "scratch-vs-laya", "void-vs-laya"],
         [proof("runs/report_scratch_vs_laya.json", "g1/pass", False,
                "the only myna architecture number committed here is the control, and it "
                "does not clear the threshold")],
         "nothing that decides it. What is committed is the untrained control at **0.346 / "
         "0.351** (`runs/scratch_decision_v2_test.md`) and the void pre-`385e06c` checkpoint "
         "at **0.338**, and `runs/report_scratch_vs_laya.json` prints `\"pass\": false` for "
         "the control's own G1 check.",
         "open on a missing *artifact*, not a missing *run*. `v1b-kaggle-3600b` trained 3,600 "
         "updates on Kaggle and reported its own metrics; no witness of it is committed in "
         "this repo, which is exactly the case §9.30 refuses a figure for. TODO 3i publishes "
         "the verdict from its own artifact — and until then the two numbers above are the "
         "architecture at chance and at chance-after-a-gradient-step, neither of which is "
         "G1's subject."),

    gate("G2", "Latency claim survives equal-footing re-measurement",
         "same box, same window, same question count, direct `Agent` call not `Router`; "
         "published ratio recomputed from that or withdrawn",
         MET,
         ["latency-matched"],
         [proof("runs/latency_matched.json", "meta/inference_only", True,
                "the matched run is timed inference-only on both engines"),
          proof("runs/latency_matched.json", "meta/myna_dtype", "torch.float32",
                "myna ran fp32, not a favoured dtype"),
          proof("runs/latency_matched.json", "meta/laya_dtype", "torch.float32",
                "laya ran the same fp32, autocast off")],
         "one M5 process, both engines fp32, `Agent.system_one` called directly, ladder "
         "capped inside laya's own 1024 window: **3.04×/3.21×/4.12×/3.60×** end-to-end and "
         "**6.78×/4.67×/4.75×/3.65×** ask-only at 1/5/10/50 questions, each cell the minimum "
         "of two committed runs.",
         "met as a ratio, which is all G2 asks. It carries no accuracy implication: the model "
         "that is faster is the one that still fails G1, and the millisecond columns behind "
         "these ratios are load-dependent on a box other sessions share (§9.23) — quote the "
         "ratio, never the milliseconds."),

    gate("G3", "Deployment works for real",
         "ONNX browser build answers a live page's decisions; bytes + p50 + cold-load "
         "measured in Chrome",
         MET,
         ["onnx-parity", "browser-g3-fp32", "browser-selftest"],
         [proof("runs/onnx_parity.json", "pass", True,
                "the torch↔onnxruntime parity report still says it passed"),
          proof("runs/browser_g3.json", "rows/0/report/passed", 6,
                "Chrome 153 passes 6 of 6 checks on the first isolation mode")],
         "the exported artifact answers three typed decisions in real Chrome 153 with the "
         "same probabilities the torch engine prints — 6/6 parity checks, both isolation "
         "modes, desktop and mobile widths — and its bytes, cold-load and p50 are measured "
         "there with the box's load average printed beside them.",
         "met as a mechanics gate, and the caveat is part of the verdict rather than a "
         "footnote: what ships is v0, which fails G1 and G5; the ≤ 20 MB int8 route is open "
         "on bytes and closed on agreement (§9.28); the Apple-silicon int8 route closes for "
         "the opposite reason — 17.40 MiB at 9.24e-03 and no latency win at all, so fp32 is "
         "the artifact on both paths (§9.29)."),

    gate("G4", "Long-context is *correct*, not just cheap",
         "needle-style typed decision ≥ 0.90 at 4k and ≥ 0.85 at 16k state",
         NOT_MET,
         ["needle"],
         [proof("runs/needle_myna-v0.json", "baseline/readable", False,
                "the harness itself refuses to let this curve be read as decay")],
         "v0's needle curve prints **0.188 at 128 tokens** against a 0.167 uniform floor — at "
         "chance on the *shortest* rung — so the longer rows are the decay of nothing, and "
         "`runs/needle_myna-v0.md` prints `G4: not measured by this run` in place of a table "
         "a reader could quote.",
         "not met, and not met by a run that cannot answer the question. The 16k *state* is "
         "measured, fixed at 576 KiB and cheap (§9.22); the 16k *decision* needs the 4k "
         "truncated-backprop checkpoint, which is 7b / `KAGGLE`. Those are two different "
         "claims and only the first one is settled."),

    gate("G5", "Useful confidence, with abstention",
         "risk/coverage curve on `calibration.jsonl`; ≥ 0.95 accuracy at ≥ 60% coverage on "
         "banking77 + dbpedia14 + trec",
         NOT_MET,
         ["risk-coverage"],
         [proof("runs/risk_coverage.json", "g5/pass", False,
                "the curve's own pass field is false, in the artifact that publishes it")],
         "the curve's own `g5.pass` is `false`: accuracy climbs **0.349 → 0.535** as coverage "
         "falls 1.00 → 0.20 over 448 rows / 568 questions, **no rung reaches 0.95** (the max "
         "is 0.596, at 10% coverage), and G5's three named sources measure 0.000 / 0.025 / "
         "0.150.",
         "not met on the level, met on the machinery, and the gate is the level. "
         "`Myna(abstain_below=t)` withholds the commitment with a measured reason and the "
         "fallback seam labels which engine committed — a confidence that ranks the answers "
         "of a model that cannot answer is routing, not the product G5 describes (§9.24)."),

    gate("G6", "No regression on what already worked",
         "synthetic v0 test ≥ 0.94",
         OPEN,
         ["v0-accuracy"],
         [],
         "v0's measured test macro is **0.9523** — the macro of the nine `=== test ===` rows "
         "in `runs/train-v0.log`, recomputed from its rows because the log prints no overall "
         "line — so the level clears today.",
         "open rather than met, because the gate is written against the *next* checkpoint: it "
         "says \"no regression\", and there is no trained v1 artifact in this repo to "
         "regress. A draft of this row cited 0.951, a figure no artifact prints (§9.30), "
         "which is why the value here is computed from the log's own rows."),

    gate("G7", "Reproducibility",
         "one command per result, seeds pinned, witness JSONs committed, `pytest` green",
         MET,
         ["report-floors", "scratch-control", "needle"],
         [proof(REGISTRY, "", True,
                "every published table row is bound to a committed witness that still "
                "contains the figure the prose quotes")],
         "`make repro` is green at this tick: 22 registry rows bind every published table "
         "cell to one command and one committed witness, and the 38 quoted figures are "
         "re-read out of those files rather than out of the prose. The seeds live inside the "
         "printed commands (`--seed 0`, `--seeds 0 1`), not in sentences about them.",
         "met as a binding, not as a rebuild, and the gap is disclosed rather than absorbed: "
         "no target *executes* the registry end to end — 14 rows re-run on this box, 2 retrain "
         "a checkpoint, 2 need a laya checkout on `PYTHONPATH`, 3 need Google Chrome and 1 is "
         "`KAGGLE`. That is TODO 8b, which stays unticked for exactly this reason."),
]

VERDICT_WORD = {MET: "met", NOT_MET: "not met", OPEN: "open"}


def rows_by_id():
    out = {}
    for r in ROWS:
        out.setdefault(r["id"], r)
    assert len(out) == len(ROWS), "duplicate registry row ids"
    return out


def json_path(doc, path):
    """Value at a slash-separated key path. Raises KeyError/LookupError if it is not there."""
    cur = doc
    for key in [k for k in path.split("/") if k]:
        if isinstance(cur, list):
            cur = cur[int(key)]
        else:
            cur = cur[key]
    return cur


def registry_green():
    """(True, detail) when no registry row has drifted from its witness."""
    bad = {}
    for r in ROWS:
        problems, _ = check_row(r)
        if problems:
            bad[r["id"]] = problems
    if bad:
        return False, "; ".join(f"{i}: {'; '.join(p)}" for i, p in bad.items())
    return True, (f"{len(ROWS)}/{len(ROWS)} registry rows hold, "
                  f"{sum(len(r['quotes']) for r in ROWS)} figures tied to a committed witness")


def read_proof(p):
    """(holds, shown) for one proof, with the artifact's own value in `shown`."""
    if p["file"] == REGISTRY:
        ok, detail = registry_green()
        return ok, detail
    path = REPO / p["file"]
    try:
        value = json_path(json.loads(path.read_text()), p["path"])
    except FileNotFoundError:
        return False, f"{p['file']} is not on disk"
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
        return False, f"{p['file']} has no {p['path']!r} ({e.__class__.__name__})"
    return value == p["expect"], f"{p['path']} is {value!r}, not {p['expect']!r}"


def spec_verdicts() -> dict:
    """The verdict word SPEC §2.2's status column leads with, per gate, read from disk.

    Leading words only: §2.2's cells carry prose after the verdict, and the whole point is
    that the cell *starts* with the verdict a reader skims for.
    """
    text = (REPO / SPEC_GATE_TABLE).read_text()
    out = {}
    for line in text.splitlines():
        m = re.match(r"^\|\s*\*\*(G\d)\*\*\s*\|", line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        body = cells[-1].lstrip("*").lstrip()
        v = next((w for w in VERDICTS if body.lower().startswith(w)), None)
        if v is None:
            out[m.group(1)] = f"UNPARSEABLE({body[:24]!r})"
        else:
            out[m.group(1)] = v
    return out


def check_gate(g, by_id, prose):
    """(problems, notes) for one gate. Empty problems means the verdict holds."""
    bad, notes = [], []
    if g["verdict"] not in VERDICTS:
        bad.append(f"verdict {g['verdict']!r} is not one of {VERDICTS}")
    if not g["rows"]:
        bad.append("cites no registry row: a verdict nothing measured")
    cited = []
    for rid in g["rows"]:
        r = by_id.get(rid)
        if r is None:
            bad.append(f"cites a registry row that does not exist: {rid}")
            continue
        cited.append(r)
        drift, _ = check_row(r)
        for d in drift:
            bad.append(f"cited row {rid}: {d}")
    for p in g["proofs"]:
        ok, shown = read_proof(p)
        if not ok:
            bad.append(f"proof {p['file']}:{p['path'] or '-'} does not hold ({shown})")
        if p["file"] != REGISTRY:
            if not (REPO / p["file"]).exists():
                bad.append(f"proof file missing: {p['file']}")
            elif not _tracked(p["file"]):
                bad.append(f"proof file not committed: {p['file']}")
        if isinstance(p["expect"], bool):
            if g["verdict"] == MET and p["expect"] is False:
                bad.append(f"a met gate rests on a false field ({p['file']}:{p['path']})")
            if g["verdict"] == NOT_MET and p["expect"] is True:
                bad.append(f"a not-met gate rests on a true field ({p['file']}:{p['path']})")

    # What each verdict owes the reader.
    with_witness = [r for r in cited if r["quotes"] and r["status"] not in UNREPRODUCIBLE]
    if g["verdict"] in (MET, NOT_MET):
        if not with_witness:
            bad.append(f"a {g['verdict']} verdict must cite a locally reproducible "
                       "row carrying a figure")
        for r in with_witness:
            if "void" in r["note"]:
                bad.append(f"a {g['verdict']} verdict may not rest on void artifact "
                           f"{r['id']} (§9.23)")
    if g["verdict"] == MET and not any(p["expect"] is True for p in g["proofs"]):
        bad.append("a met verdict needs a field that prints true")
    if g["verdict"] == NOT_MET and not any(p["expect"] is False for p in g["proofs"]):
        bad.append("a not-met verdict needs a field that prints false")
    if g["verdict"] == OPEN:
        if not [r for r in cited if r["status"] in UNREPRODUCIBLE]:
            bad.append("open must name the run this box cannot do, as a registry row "
                       "of status gated-kaggle or retrain")
        if g["proofs"]:
            notes.append("carries proof(s) of what *is* here; the deciding artifact is what "
                         "is not")
    got = prose.get(g["id"])
    if got is None:
        bad.append(f"{SPEC_GATE_TABLE} §2.2 has no {g['id']} row to agree with")
    elif got != g["verdict"]:
        bad.append(f"{SPEC_GATE_TABLE} §2.2 says {got!r}, this file says {g['verdict']!r}")
    return bad, notes


_TRACK_CACHE = {}


def _tracked(rel: str) -> bool:
    if rel not in _TRACK_CACHE:
        r = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO,
                           capture_output=True, text=True)
        _TRACK_CACHE[rel] = r.returncode == 0
    return _TRACK_CACHE[rel]


def print_table() -> str:
    """The operator view: one line per gate, condition and verdict. README's `table()`
    drops the condition because §2.2 owns that sentence."""
    return "\n".join(f"{g['id']:<4} {g['verdict']:<8} {g['condition']}" for g in GATES)


def tally():
    return {v: sum(1 for g in GATES if g["verdict"] == v) for v in VERDICTS}


def release_state() -> str:
    t = tally()
    return (f"**{t[MET]} of {len(GATES)} met, {t[NOT_MET]} not met, {t[OPEN]} open — the "
            "release gate is NOT clear.**")


def table() -> str:
    """The README block. Generated, and therefore not evidence (§9.31): `--check` reads
    SPEC §2.2 and the artifacts, never this. The pass condition is deliberately left out
    — it is §2.2's sentence to print, and a second copy is a second thing that can drift."""
    lines = ["| gate | verdict | what the committed artifact prints | what it does not claim |",
             "|---|---|---|---|"]
    for g in GATES:
        lines.append(f"| **{g['id']}** — {g['name']} | **{g['verdict']}** | "
                     f"{g['reads']} | {g['note']} |")
    return "\n".join(lines)


def readme_block() -> str:
    return "\n".join([BEGINS,
                      "## Where this stands: the release gate, all seven rows",
                      "",
                      release_state(),
                      "",
                      "Every verdict below is checked against a committed artifact — the registry",
                      "row beside it and the field read out of that artifact by",
                      "`uv run python bench/gates.py --check` — and the same words head the",
                      "status column of [SPEC.md](SPEC.md) §2.2. The gate that decides whether",
                      "this is a product (G1) and the gate that decides what it can read (G4)",
                      "are the two that cannot be closed on this box.",
                      "",
                      table(),
                      "",
                      ENDS]) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="verify every verdict against its registry rows, its proofs and "
                         "SPEC §2.2's status column")
    ap.add_argument("--print", dest="print_table", action="store_true",
                    help="print the release-gate table")
    ap.add_argument("--readme-block", action="store_true",
                    help="print the exact text README's generated block must contain")
    args = ap.parse_args(argv)

    if args.print_table:
        print(release_state())
        print()
        print(print_table())
        return 0

    if args.readme_block:
        print(readme_block())
        return 0

    if args.check:
        by_id, prose = rows_by_id(), spec_verdicts()
        fails = 0
        for g in GATES:
            bad, notes = check_gate(g, by_id, prose)
            print(f"{'ok  ' if not bad else 'FAIL'} {g['id']:<4} {g['verdict']:<8} "
                  f"{len(g['rows'])} row(s), {len(g['proofs'])} proof(s)")
            for n in notes:
                print(f"     · {n}")
            for e in bad:
                print(f"     ! {e}")
            fails += bool(bad)
        extra = set(prose) - {g["id"] for g in GATES}
        if extra:
            print(f"     ! SPEC §2.2 has gate rows no verdict is written for: {sorted(extra)}")
            fails += 1
        print(f"\n{len(GATES) - fails}/{len(GATES)} verdicts hold — {release_state()}")
        return 1 if fails else 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

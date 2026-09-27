#!/usr/bin/env python
"""One reproduction command per published table row, and the witness it came from.

SPEC §9.30 killed eight headline cells because a number was copied out of an artifact
into prose and then copied again. The copy is the failure mode, so this registry makes
the *pair* (command, artifact) the unit of record: a row is only allowed to exist if its
witness is committed and still contains the figure the docs quote from it.

    uv run python bench/reproduce.py --list          # markdown, for README
    uv run python bench/reproduce.py --check         # the gate; exit 1 on any drift
    uv run python bench/reproduce.py --run G3-browser --yes

Three kinds of row, and the difference is not cosmetic:

* **here** — runs on this box against this repo, witness committed.
* **external-laya / external-chrome** — needs a dependency that is not in the tree (a
  laya checkout on `PYTHONPATH`, Google Chrome). The command still runs, but only where
  that thing is installed, and `--check` says so rather than calling it a failure.
* **gated-kaggle** — the figure does not exist yet; it waits on a checkpoint trained on
  a corpus this box does not train on. Those rows must have *no* witness: a gated number
  with a file behind it is how a projection gets read as a measurement.

`--check` is deliberately three independent assertions, because they break separately:
the witness is committed (`git ls-files`), the docs quote this command verbatim, and the
witness still contains the value the registry claims for it.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = ("README.md", "SPEC.md", "PLAN.md", "TODO.md")

# The generated block is fenced in the docs by these two comment lines, and `doc_text`
# cuts everything between them out. Without the cut, the registry would be checking
# itself: it prints every command into that block, so "a doc quotes this command"
# would be true by construction.
BEGINS, ENDS = "<!-- reproduce:registry:begin -->", "<!-- reproduce:registry:end -->"
GENERATED = re.compile(re.escape(BEGINS) + r".*?" + re.escape(ENDS) + r"\n?", re.S)

# status values, and what each one owes the reader
HERE, LAYA, CHROME, KAGGLE, RETRAIN = ("here", "external-laya", "external-chrome",
                                       "gated-kaggle", "retrain")


def row(id, table, cmd, status, witness, quotes=(), note=""):
    """One published table row.

    `quotes` are `(file, value)` pairs: the file must contain the value as a literal.
    Keep them short and exact (`"0.9523"`, not a sentence) — they are the seam between
    the prose and the artifact, and a prose edit that changes the number has to change
    the registry too, which is the whole point.
    """
    return {"id": id, "table": table, "cmd": cmd, "status": status,
            "witness": list(witness), "quotes": list(quotes), "note": note}


# Every row here corresponds to a table or headline figure in README/SPEC/PLAN.
ROWS = [
    row("arch-params", "README architecture table · SPEC §2.1 (params, weights on disk)",
        "uv run python -m bench.bench_mlx --ckpt runs/myna-v0 --reps 20 --drift-rows 12 "
        "--out runs/bench_mlx_int8.md",
        HERE,
        ["runs/bench_mlx_int8.md", "runs/bench_mlx_int8.json"],
        [("runs/bench_mlx_int8.json", '"n_params": 14449280'),
         ("runs/bench_mlx_int8.json", '"file_bytes": 57805835')],
        "14.45M params and 55.13 MiB are read from this json, not from the checkpoint."),

    row("latency-matched", "README 'Measured on an Apple M5' · SPEC §2.1 G2",
        'PYTHONPATH="<laya checkout>" .venv/bin/python -m bench.bench_latency_matched',
        LAYA,
        ["runs/latency_matched.md", "runs/latency_matched.json"],
        [("runs/latency_matched.md", "| 1 | 100.691 | 33.083 | 3.04 | 14.856 | 6.78 |"),
         ("runs/latency_matched.md", "fixed **10.115 ms**")],
        "G2's ratios are per-row minima over two runs; the other run is "
        "runs/latency_matched_run1_superseded.md."),

    row("v0-accuracy", "README 'Trained v0' · SPEC §2.1 G6",
        "uv run python -m myna.train --steps 9000 --batch 32 --device mps --out runs/myna-v0",
        RETRAIN,
        ["runs/train-v0.log"],
        [("runs/train-v0.log", "support/urgency"),
         ("runs/train-v0.log", "=== test ===")],
        "dev 0.960 / test 0.952 are the macro over the nine per-question rows this log "
        "prints; the log prints no overall line, so the macro is arithmetic on its rows. "
        "Re-running it is training a model, which does not happen on this box."),

    row("rlcd", "PLAN v1 · TODO 4 (RLCD scoring pass)",
        "uv run python -m myna.rlcd --ckpt runs/myna-v0 --steps 800 --out runs/myna-v0-rlcd",
        RETRAIN,
        ["runs/rlcd_v0.log"],
        [("runs/rlcd_v0.log", "before   acc 0.9626"),
         ("runs/rlcd_v0.log", "after    acc 0.9657")],
        "a before/after pair inside one regeneration of the dev split (--n-eval 600, "
        "seed 0), so it is a delta, not a level comparable to G6's dev figure."),

    row("risk-coverage", "README abstention table · SPEC §2.2 G5",
        "uv run python -m bench.risk_coverage --ckpt runs/myna-v0 --out runs/risk_coverage.md",
        HERE,
        ["runs/risk_coverage.md", "runs/risk_coverage.json"],
        [("runs/risk_coverage.md", "calibration: 448 rows / 568 questions over 168 question-sets"),
         ("runs/risk_coverage.md", "0.349")],
        "G5 is NOT MET on this checkpoint and the witness says so in words; the curve is "
        "a routing measurement, not an accuracy one."),

    row("needle", "SPEC §2.2 G4 · README context column",
        "uv run python bench/eval_needle.py --ckpt runs/myna-v0 --device cpu --n 16 "
        "--seed 0 --lengths 128 1024 4096 8192 16384",
        HERE,
        ["runs/needle_myna-v0.md", "runs/needle_myna-v0.json"],
        [("runs/needle_myna-v0.md", "G4: **not measured by this run**"),
         ("runs/needle_myna-v0.json", '"acc": 0.1875')],
        "the defaults are --n 40 and no 128-token rung; both matter, so they are in the "
        "command. 0.188 at 128 tokens against a 0.167 floor is the cell that kills the "
        "decay reading."),

    row("laya-real-suite", "SPEC §2.1 G1 competitor row · README real-corpus table",
        "uv run python bench/eval_laya_real.py --split test --n-per-source 40 --seed 0 "
        "--out runs/laya_decision_v2_test.json",
        LAYA,
        ["runs/laya_decision_v2_test.json"],
        [('runs/laya_decision_v2_test.json', '"overall_acc": 0.6318681318681318')],
        "--suite is a local kev checkout and is deliberately NOT in the recorded command; "
        "SPEC §4.2 names the suite and its pinned revisions in prose."),

    row("onnx-parity", "SPEC §2.2 G3 (torch↔onnxruntime) · README browser section",
        "uv run python -m myna.onnx_export --ckpt runs/myna-v0 --out runs/onnx "
        "--scan-chunk 16 --n-chunks 4 --suite data/decision-v2-pilot",
        HERE,
        ["runs/onnx_parity.json", "runs/onnx_export.log"],
        [("runs/onnx_parity.json", '"pass": true'),
         ("runs/onnx_export.log", "chunk 256 scanned in 16-token tiles (3.0 MiB peak per call)")],
        "the artifact dir itself (runs/onnx) is build output and is not committed; the "
        "parity report and the export log are."),

    row("onnx-parity-widest", "SPEC §5 P6 6b (the 1,067-token request)",
        "uv run python -m myna.onnx_export --ckpt runs/myna-v0 --out runs/onnx-wide "
        "--questions 2 --q-len 1152 --scan-chunk 64 --report runs/onnx_parity_widest.json",
        HERE,
        ["runs/onnx_parity_widest.json", "runs/onnx_export_widest.log"],
        [("runs/onnx_parity_widest.json", '"measured": true'),
         ("runs/onnx_parity_widest.json", '"tokens": 1067')],
        "6a's report says `measured: false` for this request because the graph was "
        "narrower than it; widening the graph is the whole of 6b's parity claim."),

    row("browser-selftest", "SPEC §5 P6 (the node gate that backs every browser cell)",
        "npm run selftest",
        CHROME,
        ["runs/browser_selftest.log"],
        [("runs/browser_selftest.log", "7 checks")],
        "runs in node, not Chrome; it is the build gate the exported artifact has to pass "
        "before browser_g3 measures anything."),

    row("browser-g3-fp32", "README in-tab table, left column · SPEC §2.2 G3",
        "npm run g3",
        CHROME,
        ["runs/browser_g3.json"],
        [("runs/browser_g3.json", '"artifact_mib": 59.32'),
         ("runs/browser_g3.json", '"wire_mib": 73.09'),
         ("runs/browser_g3.json", "589824")],
        "589824 B is the 576 KiB state cell in the README's architecture table — the same "
        "number Chrome prints on all four rows."),

    row("browser-g3-int8", "README in-tab table, right column",
        "npm run g3 -- --artifact runs/onnx_int8 --out runs/browser_g3_int8.json",
        CHROME,
        ["runs/browser_g3_int8.json"],
        [("runs/browser_g3_int8.json", '"artifact_mib": 18.56'),
         ("runs/browser_g3_int8.json", '"p50_ms": 412.5')],
        "the int8 column exists to be refused: 18.56 MiB, and it fails 2 of Chrome's 6 "
        "parity checks (SPEC §9.28)."),

    row("int8-quantize", "SPEC §5 P6 (the int8 attempt, and the node gate going red)",
        "uv run python bench/quantize_int8.py",
        HERE,
        ["runs/int8_quantize.log"],
        [("runs/int8_quantize.log", "weights.bin        13.60 MiB   (fp32 54.32)"),
         ("runs/int8_quantize.log", "RED (5 checks ok)")],
        "the gate is red in the witness and stays red in the witness; that line is the "
        "conclusion, not a build failure to fix."),

    row("mlx-int8", "README 'On an Apple silicon Mac' · SPEC §5 P6 6c",
        "uv run python -m bench.bench_mlx --ckpt runs/myna-v0 --reps 20 --drift-rows 12 "
        "--out runs/bench_mlx_int8.md",
        HERE,
        ["runs/bench_mlx_int8.md", "runs/bench_mlx_int8.json",
         "runs/bench_mlx_int8_run2.md", "runs/bench_mlx_int8_run2.json"],
        [("runs/bench_mlx_int8.json", '"load_avg": ['),
         ("runs/bench_mlx_int8_run2.json", '"argv"')],
        "the published ratios are per-row minima over these two runs, so both are "
        "witnesses of the same row. The third command on this path is the keep-gate run, "
        "below."),

    row("mlx-int8-keepgate", "SPEC §5 P6 6c (the one row that flips when the gates are kept fp32)",
        "uv run python -m bench.bench_mlx --ckpt runs/myna-v0 --reps 3 --lengths 128 "
        "--drift-rows 12 --keep trunk.tok.weight .gate.weight --fp32-out "
        "runs/myna-v0-mlx-fp32-keepgate --int8-out runs/myna-v0-mlx-int8-keepgate "
        "--out runs/bench_mlx_int8_keepgate.md",
        HERE,
        ["runs/bench_mlx_int8_keepgate.md", "runs/bench_mlx_int8_keepgate.json"],
        [("runs/bench_mlx_int8_keepgate.json", '"reps": 3')],
        "--reps 3 is not a typo for speed: this run exists to flip one row of the ladder, "
        "and 20 reps of it would be a different claim on a hotter box."),

    row("mlx-kernel", "README 'and it prices that at the kernel' · SPEC §5 P6 6c",
        "uv run python bench/diag_mlx_int8_gem.py --iters 200 --trials 7 "
        "--out runs/mlx_int8_gem_probe.md",
        HERE,
        ["runs/mlx_int8_gem_probe.md", "runs/mlx_int8_gem_probe.log"],
        [("runs/mlx_int8_gem_probe.md", "median of 7 trials"),
         ("runs/mlx_int8_gem_probe.log", "load 4.18/4.29/4.55")],
        "the default is --trials 9; the committed probe ran 7, and the header prints "
        "whichever it ran."),

    row("report-floors", "SPEC §4.2 (the majority-label floors the accuracy gates are set against)",
        "uv run pytest tests/test_report.py -q",
        HERE,
        ["tests/test_report.py", "data/decision-v2-pilot"],
        [("tests/test_report.py", "0.4331")],
        "there is no metrics-less CLI for the floors: myna.report needs a checkpoint's "
        "metrics.json, so the frozen split's floors are pinned where they are computed."),

    row("py311", "SPEC §5 P3 3c (the Kaggle image is python 3.12/3.11, this box is 3.13)",
        "uv run python bench/check_python311.py",
        HERE,
        ["runs/python311_check.log"],
        [("runs/python311_check.log", "PASS: the package imports, compiles and trains on python 3.11")],
        "needs a 3.11 interpreter with torch (MYNA_PY311); it refuses to report a silent skip."),

    row("g1-v1", "SPEC §2.1 G1 myna row · TODO 3i",
        "KAGGLE: train on decision-v2 per kaggle/PLAN, then "
        "uv run python bench/eval_laya_real.py's myna twin on the test split",
        KAGGLE,
        [],
        [],
        "no witness exists and none may be added by a local run: the corpus is trained on "
        "Kaggle, so a figure here would be a projection wearing a measurement's clothes."),
]

STATUSES = (HERE, LAYA, CHROME, KAGGLE, RETRAIN)


def norm(s: str) -> str:
    """Fold a shell line continuation and any whitespace run to a single space, so a
    command wrapped across two lines in a markdown code block still matches. The
    backslash has to go before the whitespace collapses — `\\` + newline + indent is
    three whitespace-separated pieces, not one."""
    return re.sub(r"\s+", " ", re.sub(r"\\\s*\n\s*", " ", s)).strip()


def strip_comment(line: str) -> str:
    """Drop a trailing `# ...` doc comment from a shell line (quotes are not in these)."""
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


def doc_text() -> str:
    """Every doc, comments dropped and line continuations folded, as one string. The
    fold happens across the whole file rather than per line, which is the only way a
    command wrapped with a backslash in a markdown code block matches a registry line.

    The registry's own generated block is cut out first: it is this file's `ROWS`
    echoed back, so a matcher that read it would find every command in it and the
    doc-quote assertion could never fail. What has to be provable is that the prose
    next to the published table prints the command.
    """
    raw = "\n".join((REPO / name).read_text() for name in DOCS)
    raw = GENERATED.sub("\n", raw)
    return norm("\n".join(strip_comment(ln) for ln in raw.splitlines()))


def tracked(rel: str) -> bool:
    r = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO,
                       capture_output=True, text=True)
    return r.returncode == 0


def check_row(r) -> tuple[list[str], list[str]]:
    """(problems, notes) for one row. Empty problems means the row holds.

    The three assertions are independent on purpose: a row can have its witness
    committed and its figure intact while the docs stopped quoting the command, and
    that is exactly the drift §9.30 is about.
    """
    bad, notes = [], []
    if not r["quotes"] and r["status"] != KAGGLE:
        # a row that names no figure is a row no artifact can contradict
        bad.append("row quotes no figure from its witness")
    if norm(r["cmd"]) not in doc_text() and r["status"] != KAGGLE:
        bad.append(f"no doc quotes the command: {r['cmd'][:72]}")
    if not r["witness"] and r["status"] not in (KAGGLE,):
        bad.append("no witness named")
    if r["status"] == KAGGLE and r["witness"]:
        bad.append("a gated row must not point at a witness")
    for f in r["witness"]:
        if not (REPO / f).exists():
            bad.append(f"witness missing: {f}")
        elif not tracked(f):
            bad.append(f"witness not committed: {f}")
    for f, value in r["quotes"]:
        p = REPO / f
        if not p.exists():
            bad.append(f"quote file missing: {f}")
        elif value not in p.read_text():
            bad.append(f"{f} no longer contains {value!r}")
    if r["status"] == LAYA:
        notes.append("needs a laya checkout on PYTHONPATH, so it only runs where laya is")
    if r["status"] == CHROME:
        notes.append("needs Google Chrome, so it only runs on a box that has it")
    return bad, notes


def listing() -> str:
    lines = ["| row | where it is published | status | reproduce |",
             "|---|---|---|---|"]
    for r in ROWS:
        lines.append(f"| `{r['id']}` | {r['table']} | {r['status']} | `{r['cmd']}` |")
    return "\n".join(lines)


def readme_block() -> str:
    """The exact text README.md's 'every table' section must contain, one comment +
    one command per row, inside a fenced shell block between the two markers. A test
    pins README to this function, so adding a row without publishing its command is a
    red test rather than an undocumented number."""
    out = [BEGINS, "```bash"]
    for r in ROWS:
        out.append(f"# {r['table']}  [{r['status']}]")
        # a gated row has no command to print, and printing prose where a shell
        # line belongs would make the block uncopy-pasteable
        out.append(r["cmd"] if r["status"] != KAGGLE
                   else "# no local command: " + r["note"].split(".")[0] + ".")
    out.append("```")
    out.append(ENDS)
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="print the registry as markdown")
    ap.add_argument("--check", action="store_true",
                    help="verify every row: witness committed, docs quote the command, "
                         "the witness still contains the quoted figure")
    ap.add_argument("--run", metavar="ID", help="execute one row's command")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --run: print the command and the witnesses it overwrites, "
                         "and stop")
    ap.add_argument("--yes", action="store_true",
                    help="required with --run: a run overwrites a committed witness whose "
                         "cells the README quotes")
    ap.add_argument("--row", metavar="ID", help="restrict --check to one row")
    args = ap.parse_args(argv)

    rows = [r for r in ROWS if not args.row or r["id"] == args.row]
    if args.row and not rows:
        raise SystemExit(f"no row {args.row!r}; try --list")

    if args.list:
        print(listing())
        return 0

    if args.run:
        me = next((r for r in ROWS if r["id"] == args.run), None)
        if me is None:
            raise SystemExit(f"no row {args.run!r}; try --list")
        if me["status"] == KAGGLE:
            raise SystemExit(f"{args.run} is gated on Kaggle — there is no local command to run")
        if me["status"] == RETRAIN:
            raise SystemExit(f"{args.run} trains a model, which does not happen on this box")
        if not args.yes:
            raise SystemExit(
                f"{args.run} writes {', '.join(me['witness']) or 'a new file'} — those are "
                f"committed witnesses the README quotes cells from. Re-run with --yes to "
                f"overwrite them, and expect --check to go red until the new numbers are "
                f"carried into the prose. --dry-run prints what would happen.")
        if args.dry_run:
            print("$ " + me["cmd"])
            print(f"would overwrite: {', '.join(me['witness']) or '(nothing committed)'}")
            return 0
        print("$ " + me["cmd"], flush=True)
        return subprocess.call(me["cmd"], shell=True, cwd=str(REPO))

    if args.check:
        fails = 0
        for r in rows:
            bad, notes = check_row(r)
            print(f"{'ok  ' if not bad else 'FAIL'} {r['id']:<22} {r['status']:<16} "
                  f"{len(r['quotes'])} quoted figure(s)")
            for n in notes:
                print(f"     · {n}")
            for e in bad:
                print(f"     ! {e}")
            fails += bool(bad)
        print(f"\n{len(rows) - fails}/{len(rows)} rows hold; "
              f"{sum(len(r['quotes']) for r in rows)} figures tied to a committed witness, "
              f"{sum(1 for r in rows if r['status'] == KAGGLE)} gated with no witness")
        return 1 if fails else 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

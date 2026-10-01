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

`--check` is deliberately four independent assertions, because they break separately:
the witness is committed (`git ls-files`), the docs quote this command verbatim, the
witness still contains the value the registry claims for it, and the tool behind the
command still answers `--help` with every flag the row publishes. The last one is the
difference between a registry of *quotas* and a registry of *commands* (§9.43).
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
DOCS = ("README.md", "SPEC.md", "PLAN.md", "TODO.md",
        # the launch docs carry published figures now (the GPU-hour price, the G1
        # measurement), so they owe the registry the same quote as the others. Before
        # this they were outside `--check` entirely, which is how a hand-typed cell
        # with two wrong paths survived for a session.
        "KAGGLE_LAUNCH_INSTRUCTIONS.md", "EASY_LAUNCH.md")

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

    row("scratch-control", "README upstream table, control row · SPEC §8's disclosure",
        "uv run python bench/eval_scratch.py --split test --seeds 0 1 "
        "--out runs/scratch_decision_v2_test.md "
        "--metrics-out runs/scratch_metrics_test.json",
        HERE,
        ["runs/scratch_decision_v2_test.md", "runs/scratch_decision_v2_test.json",
         "runs/scratch_metrics_test.json"],
        [("runs/scratch_decision_v2_test.md", "mean **0.348**, spread 0.005"),
         ("runs/scratch_decision_v2_test.json", '"macro_mean": 0.34841306765899954'),
         ("runs/scratch_decision_v2_test.json", '"uniform_floor": 0.3321192559493052')],
        "The row exists to be beaten. A random init at the 0.332 chance floor is the "
        "architecture's floor and not its ceiling; --metrics-out is what lets myna.report "
        "read the control through the same code path a checkpoint comes in by."),

    row("scratch-vs-laya", "README upstream table, the gap line · SPEC §8",
        "uv run python -m myna.report --suite data/decision-v2-pilot --split test "
        "--metrics runs/scratch_metrics_test.json --laya runs/laya_decision_v2_test.json "
        "--out runs/report_scratch_vs_laya.json",
        HERE,
        ["runs/report_scratch_vs_laya.json", "runs/report_scratch_vs_laya.log",
         "runs/laya_decision_v2_test.json"],
        [("runs/report_scratch_vs_laya.log", "myna 0.346 · laya 0.667 · gap -0.321"),
         ("runs/report_scratch_vs_laya.json", '"laya": 0.6668154761904762'),
         # 9c's column, tied to the artifact that printed it: the clip is a *pair* of
         # numbers (what it is, and what part of it is not the model), so both halves
         # are quoted rather than the flattering one.
         ("runs/report_scratch_vs_laya.json", '"clip": 0.43648613706976636'),
         ("runs/report_scratch_vs_laya.json", '"clip_worth": 0.09032601120535727'),
         ("runs/report_scratch_vs_laya.log", "13 of 16 scored cell(s) answer below "
                                             "their own majority floor")],
        "16 cells both sides scored, unweighted. laya's row-weighted overall (0.6319, in "
        "its own JSON) is a different statistic over a different sample and never appears "
        "in this line; its contrastive/decision cell is two rows of that JSON merged by "
        "rows (§9.32). The clip macro is not the majority macro: a cell above its floor "
        "keeps its own number, which is why 0.4364 > 0.4331 while 13 cells lose."),

    row("void-vs-laya", "README upstream table's disclosure row · SPEC §5 P1, §9.23",
        "uv run python -m myna.report --suite data/decision-v2-pilot --split test "
        "--metrics runs/myna-v1-rich/metrics.json --laya runs/laya_decision_v2_test.json "
        "--out runs/report_void_vs_laya.json",
        HERE,
        ["runs/report_void_vs_laya.json", "runs/report_void_vs_laya.log"],
        [("runs/report_void_vs_laya.log", "myna 0.338 · laya 0.667 · gap -0.329")],
        "The input is the one local checkpoint that ever saw the upstream corpus, and it "
        "is uncommitted and void as evidence (§5 P1: trained before the 385e06c batching "
        "fix). It witnesses a disclosure — 0.338 against a control of 0.346 — and not a "
        "ceiling in either direction. Its own report now prints the 9c clip column and "
        "9b's Brier column, and neither is quoted: the artifact is committed, the "
        "metrics file behind it is not, so §9.30 keeps both out of the prose."),

    row("g1-v1", "SPEC §2.1 G1 myna row · TODO 3i",
        "KAGGLE: train on decision-v2 per kaggle/PLAN, then "
        "uv run python bench/eval_laya_real.py's myna twin on the test split",
        KAGGLE,
        [],
        [],
        "the run this row named is now committed as `v1b-kaggle-macro`, so G1's verdict no "
        "longer rests here. What stays gated is raising the figure: that is a second GPU "
        "spend, not a local re-run, and a number for it written here would be a projection "
        "wearing a measurement's clothes."),

    row("ordinal-ab", "SPEC §5 P9 9e (the `ce` vs `emd` verdict) · §9.35, §9.36",
        "python bench/ordinal_ab.py --ab-dir /tmp/ab2",
        RETRAIN,
        ["runs/ordinal_ab.json"],
        [("runs/ordinal_ab.json", '"churn_floor": 0.013982'),
         ("runs/ordinal_ab.json", '"resolution_ratio": 1.7591'),
         ("runs/ordinal_ab.json", '"resolution_ratio": 0.5194'),
         ("runs/ordinal_ab.json", '"dev": "2/3"'),
         ("runs/ordinal_ab.json", '"p95": 267')],
        "four CPU arms from one shared init. The arms' own `metrics.json` (≈295 KB each) and "
        "`model.pt` are not committed, so the artifact carries the rolled-up per-cell "
        "accuracies the verdict was computed from — and the p95 it records is the §9.35 "
        "confound, visible in the witness rather than asserted in prose. Quote the per-pair "
        "figures; `pooled_over_seeds` is in the file and is the wrong unit (§9.36)."),

    row("v1b-kaggle-macro", "SPEC §2.1 G1 myna row · §5 P8 · KAGGLE_LAUNCH_INSTRUCTIONS.md",
        "uv run python -m myna.report --suite data/decision-v2-pilot --split test "
        "--metrics runs/v1b_kaggle_3600b.metrics.json "
        "--laya runs/laya_decision_v2_test.json "
        "--out runs/v1b_kaggle_3600b.report.json",
        HERE,
        ["runs/v1b_kaggle_3600b.report.json", "runs/v1b_kaggle_3600b.metrics.json"],
        [("runs/v1b_kaggle_3600b.report.json", '"acc": 0.4893271976084137'),
         ("runs/v1b_kaggle_3600b.report.json", '"laya": 0.6668154761904762'),
         ("runs/v1b_kaggle_3600b.report.json", '"majority": 0.4330647953941198'),
         ("runs/v1b_kaggle_3600b.metrics.json", '"mem_plan_free_gib": 9.0'),
         ("runs/v1b_kaggle_3600b.metrics.json", '"batch": 10')],
        "The first valid GPU run, and now the witness G1 was open on: the metrics file "
        "the box wrote, committed next to the roll-up computed from it. 0.489 is the "
        "per-cell unweighted mean over the 16 cells both harnesses score — NOT the "
        "0.3983 a diagnostic printed for the same run under a different roll-up, which "
        "is why the unit is in this note (§9.37). `\"batch\": 10` is the memory plan "
        "clamping `--batch 32` against the pinned 9.0 GiB, so the row also witnesses "
        "the T4 regime."),

    row("antiprior-off-macro", "SPEC §5 P10 10c (the fresh `off` control) · §9.53(iii)",
        "uv run python -m myna.report --suite data/decision-v2-pilot --split test "
        "--metrics runs/antiprior_off_s0.metrics.json "
        "--laya runs/laya_decision_v2_test.json "
        "--out runs/antiprior_off_s0.report.json",
        HERE,
        ["runs/antiprior_off_s0.report.json", "runs/antiprior_off_s0.metrics.json"],
        [("runs/antiprior_off_s0.report.json", '"macro_acc": 0.5338735348381732'),
         ("runs/antiprior_off_s0.report.json", '"margin_observed": 0.10080873944405344'),
         ("runs/antiprior_off_s0.report.json", '"macro_majority": 0.4330647953941198'),
         ("runs/antiprior_off_s0.metrics.json", '"anti_prior": "off"'),
         ("runs/antiprior_off_s0.metrics.json", '"last_step": 3314'),
         ("runs/antiprior_off_s0.metrics.json", '"batch": 10')],
        "The control that made the +0.0446 visible: same corpus, same seed, same 3,600-update "
        "request, `--anti-prior off`, on the tree after the data-loader work. Its 0.5339 is 0.4893's "
        "twin and the gap between them belongs to the data path, not to the flag (§9.53(iii)) — which "
        "is the whole reason the pair has a fresh `off` arm instead of reusing V1-B's number. A re-run "
        "differs from the committed report in exactly two provenance leaves (`cmd`, and `metrics_file` "
        "naming the pre-flatten `runs/antiprior_off_s0/metrics.json`); every measured leaf, the whole "
        "`g1` block and all 16 cells, compares equal. `"
        "\"last_step\": 3314` of 3,600 requested is the truncated dose, and `myna.report` prints it as "
        "a VOID line rather than as a clean macro."),

    row("tier0-off-control", "SPEC §5 P10 10c (Tier 0 over the `off` control) · §9.55(iii)",
        "uv run python bench/diag_question_ablation.py --run-dir runs/antiprior_off_s0 "
        "--metrics runs/antiprior_off_s0.metrics.json "
        "--out runs/tier0_antiprior_off_s0.json",
        RETRAIN,
        ["runs/tier0_antiprior_off_s0.json", "runs/tier0_antiprior_off_s0.log"],
        [("runs/tier0_antiprior_off_s0.log", "macro 0.5339 (floor 0.4331)"),
         ("runs/tier0_antiprior_off_s0.log", "macro 0.4315 (floor 0.4331)"),
         ("runs/tier0_antiprior_off_s0.log", "macro 0.4076 (floor 0.4331)"),
         ("runs/tier0_antiprior_off_s0.log", "VOID: this run STOPPED at step 3314 of 3600"),
         ("runs/tier0_antiprior_off_s0.log", "reproduces 716 committed keys"),
         ("runs/tier0_antiprior_off_s0.json", '"macro_as_scored": 0.5338735348381732'),
         ("runs/tier0_antiprior_off_s0.json", '"swap-state": 0.4075622883568221'),
         ("runs/tier0_antiprior_off_s0.json", '"collapse_share": 1.0')],
        "The emitter-count baseline 10c's own verdict rule asks for, read off the control's weights. "
        "The harness prints its blocker before its numbers — the dose is truncated, so "
        "\"No claim may be read from them\" and every figure here describes these bytes, not a trained "
        "checkpoint. The as-scored arm reproduces all 716 keys of `antiprior_off_s0.report.json`, which "
        "is what ties the six arms to the published 0.5339. Against `tier0-ablation` on V1-B: the "
        "blank-instruction macro rises 0.3341 → 0.4315 and blank-options 0.2669 → 0.3674, both by more "
        "than as-scored's 0.4893 → 0.5339, so the prompt-reliance gaps close (instruction 0.1553 → "
        "0.1024, options 0.2225 → 0.1665) while the state gap widens (0.0720 → 0.1263). `grep -c "
        "\"collapse_share\": 1.0` is 3 in V1-B's witness and 4 in this one: `agnews/is_business` joined, "
        "answering label 0 on 43/43 rows and landing on its own 0.698 floor (0.6512 → 0.6977). Cells "
        "whose accuracy moves when their state is swapped go 8 → 10. Note this is the same "
        "checkpoint as `antiprior-off-macro`, ablated — the two rows are the macro and its mechanism."),

    row("antiprior-on-macro", "SPEC §5 P10 10c (the `on` arm, dose-matched) · §9.56",
        "uv run python -m myna.report --suite data/decision-v2-pilot --split test "
        "--metrics runs/antiprior_on_s0c.metrics.json "
        "--laya runs/laya_decision_v2_test.json "
        "--out runs/antiprior_on_s0c.report.json",
        HERE,
        ["runs/antiprior_on_s0c.report.json", "runs/antiprior_on_s0c.metrics.json"],
        [("runs/antiprior_on_s0c.report.json", '"macro_acc": 0.4784612299508689'),
         ("runs/antiprior_on_s0c.report.json", '"margin_observed": 0.04539643455674913'),
         ("runs/antiprior_on_s0c.report.json", '"macro_majority": 0.4330647953941198'),
         ("runs/antiprior_on_s0c.metrics.json", '"anti_prior": "on"'),
         ("runs/antiprior_on_s0c.metrics.json", '"last_step": 3314'),
         ("runs/antiprior_on_s0c.metrics.json", '"steps_requested": 3315')],
        "10c's verdict, and it is a cost: 0.4784612299508689 against `antiprior-off-macro`'s "
        "0.5338735348381732 is **−0.0554123048873043** from the flag alone. Both arms execute 3,315 "
        "updates — `\"last_step\": 3314` of `\"steps_requested\": 3315` here, the completed run "
        "(`\"stopped\": null`), against 3,314 of 3,600 for the truncated control (§9.56(ii)) — on this "
        "box, same seed, flags identical but for `--anti-prior` and `--out` (§9.55). G1 moves the wrong "
        "way on both halves: margin over the floor 0.10080873944405344 → 0.04539643455674913, and the "
        "laya gap widens −0.13294194135230297 → −0.18835424623960728. The shipped bytes round-trip: the "
        "`antiprior_on_s0c-weights` Release's `model.pt` hashes 4a720e4d7e15e3621edb05d8927f1e47aeaa442"
        "dcae23b1c8e52c903440d44a9, the same digest `runs/antiprior_on_s0c/model.pt` carries locally."),

    row("tier0-on-arm", "SPEC §5 P10 10c (Tier 0 over the `on` arm) · §9.56",
        "uv run python bench/diag_question_ablation.py --run-dir runs/antiprior_on_s0c "
        "--metrics runs/antiprior_on_s0c.metrics.json "
        "--out runs/tier0_antiprior_on_s0c.json",
        RETRAIN,
        ["runs/tier0_antiprior_on_s0c.json", "runs/tier0_antiprior_on_s0c.log"],
        [("runs/tier0_antiprior_on_s0c.json", '"macro_as_scored": 0.4784612299508689'),
         ("runs/tier0_antiprior_on_s0c.json", '"swap-state": 0.3860646597087993'),
         ("runs/tier0_antiprior_on_s0c.json", '"blank-options": 0.32090059476922256'),
         ("runs/tier0_antiprior_on_s0c.log", "macro 0.4785 (floor 0.4331)"),
         ("runs/tier0_antiprior_on_s0c.log", "macro 0.3209 (floor 0.4331)"),
         ("runs/tier0_antiprior_on_s0c.log", "guard green: the as-scored arm reproduces 716 committed keys"),
         ("runs/tier0_antiprior_on_s0c.json", '"collapse_share": 0.9183673469387755')],
        "The mechanism half of the same weights, and the count 10c's own judgement rule asked for. "
        "`grep -c '\"collapse_share\": 1.0'` is **0** here, against **4** in `tier0-off-control` and "
        "**3** in V1-B's `tier0-ablation`: the highest share in the whole 16 is `agnews/is_scitech` at "
        "0.918, so every constant emitter stopped emitting, which is the batching rule doing exactly "
        "what SPEC §10a designed and bought nothing — the macro above is 0.0554 lower. Cells whose "
        "accuracy moves when their state is swapped go 8 → 10 → **15** of 16; only `banking77/intent` "
        "is still inert. The four treated-by-name cells that were emitters fall to 0.674 / 0.918 / "
        "0.872 / 0.911 while their accuracies drop (is_business 0.698 → 0.651, is_scitech 0.653 → "
        "0.571, is_sports 0.809 → 0.766, is_world 0.778 → 0.733). Prompt-reliance gaps shrink as the "
        "shortcut goes: instruction 0.1024 → 0.0199, options 0.1665 → 0.1576, while the state gap "
        "narrows 0.1263 → 0.0924. Unlike `tier0-off-control` this run prints **no VOID line** — "
        "`stopped: null` and the requested step is reached — so it is the pair's first untruncated "
        "Tier 0 record, and its as-scored arm rewrites 0 keys while reproducing 716 committed ones "
        "with 0 tie flips."),

    row("v1b-kaggle-dev", "SPEC §9.41 (the dbpedia14 dev→test delta) · TODO 9h",
        "uv run python -m myna.report --split dev --metrics runs/v1b_kaggle_3600b.metrics.json "
        "--out runs/v1b_kaggle_3600b.dev.report.json > runs/v1b_kaggle_3600b.dev.report.log",
        HERE,
        ["runs/v1b_kaggle_3600b.dev.report.json", "runs/v1b_kaggle_3600b.dev.report.log"],
        [("runs/v1b_kaggle_3600b.dev.report.json", '"acc": 0.5355719231874607'),
         ("runs/v1b_kaggle_3600b.dev.report.json", '"sets": 37'),
         ("runs/v1b_kaggle_3600b.dev.report.log",
          "dbpedia14/category    choice   116   37 0.647")],
        "The dev half of the same committed metrics file, so a dev→test delta has a witness "
        "on both sides. `\"sets\": 37` beside `n: 116` is the point of the row: 36 of those "
        "sets hold one row, which is why the per-set roll-up of this cell (0.5324, and 0.2047 "
        "on test) is a variance amplifier rather than a measurement, and why §9.41's numbers "
        "are the row-weighted ones — 0.647 dev, 0.457 test, 0.5356 macro."),

    row("scope-pricing", "SPEC §5 P9 9d · TODO 9d",
        "uv run python -m bench.scope_pricing",
        HERE,
        ["runs/scope_pricing.json"],
        [("runs/scope_pricing.json", '"margin_observed": 0.056262402214293905'),
         ("runs/scope_pricing.json", '"margin_observed": 0.06676294243209946'),
         ("runs/scope_pricing.json", '"margin_observed": 0.09488116563239712')],
        "9d is the one open decision in the repo, and this prices it instead of arguing it: "
        "the two named cells out buys +0.0105 of margin against a floor that rises 0.0335 "
        "with them, and no scope in the table reaches 0.70. The subsets are the report's own "
        "`cells` rows through `myna.report.macro`, and the unfiltered row must reproduce the "
        "report's published 0.4893 / 0.4331 / `g1.pass` to the digit or the script refuses — "
        "one roll-up, not a third one (§9.37, §9.45)."),

    row("kaggle-wall-clock", "KAGGLE_LAUNCH_INSTRUCTIONS.md cost table · kaggle/campaign.py",
        'grep -E "^step" runs/v1b_kaggle_3600b.train.log | tail -1',
        HERE,
        ["runs/v1b_kaggle_3600b.train.log"],
        [("runs/v1b_kaggle_3600b.train.log", "20225s")],
        "3,600 updates took 20,225 s of wall clock on a T4 — 5 h 37 m, 5.62 s/update "
        "with evaluations included. `campaign.py` prices every cell from this constant, "
        "and the launch docs used to promise 1.5 h for the same run. The 250-step "
        "`dev-mid acc` line beside it is what says whether a run had converged: 0.4122 "
        "at step 2250, 0.4355 at 3500 (§9.38)."),

    row("kaggle-dev-tail", "SPEC §2.1 G1 · §9.38 · KAGGLE_LAUNCH_INSTRUCTIONS.md dose lane",
        'grep -E "dev-mid acc" runs/v1b_kaggle_3600b.train.log | uniq | tail -5',
        HERE,
        ["runs/v1b_kaggle_3600b.train.log"],
        [("runs/v1b_kaggle_3600b.train.log", "dev-mid acc 0.4355"),
         ("runs/v1b_kaggle_3600b.train.log", "dev-mid acc 0.4283"),
         ("runs/v1b_kaggle_3600b.train.log", "dev-mid acc 0.4122")],
        "The tail of the curve that prices the next lane. `uniq` because the box wrote "
        "every progress line twice (§9.38's witness is the doubled log, and a grep that "
        "did not fold it would show two rungs per step). The remembered \"flat from step "
        "2250\" is here disproved by the artifact: 0.4122 → 0.4355 is +0.0233 over the "
        "last 1,250 updates, and 0.4355 is the run's high, so the curve was still rising "
        "when its cosine schedule spent out."),

    row("anti-prior-audit", "SPEC §5 P10 · `campaign.py`'s P10 pair · TODO 10",
        "uv run python bench/anti_prior_audit.py --compare "
        "--out runs/anti_prior_audit.json",
        HERE,
        ["runs/anti_prior_audit.json", "runs/anti_prior_audit.log"],
        [("runs/anti_prior_audit.log",
          "cells at or above skew 0.55: 6 of 16 — agnews/is_business 0.7545, "
          "agnews/is_sports 0.7483, agnews/is_scitech 0.7467, agnews/is_world 0.7390, "
          "boolq/answer 0.6243, yelp/recommend 0.6055"),
         ("runs/anti_prior_audit.log", "worst relative deviation 0.00e+00"),
         ("runs/anti_prior_audit.log", "largest between-arm mix difference: 0.13 pts"),
         ("runs/anti_prior_audit.log",
          "shipped rule: prod — mean/max/geo/sum are kept in COMBINE only so this table "
          "stays reproducible"),
         ("runs/anti_prior_audit.json", '"rows": 57904'),
         ("runs/anti_prior_audit.json", '"updates": 3600'),
         ("runs/anti_prior_audit.json", '"weight_sum_max_rel_deviation": 0.0'),
         ("runs/anti_prior_audit.json", '"majority_drawn": 0.6246683046683047')],
        "What `--anti-prior on` changes about the data, measured without a model: the real "
        "batcher (`myna.train.draw_row_batch`) run at the published dose (batch 10, 2,048 "
        "question cells per forward, 8 sets, 3,600 updates, seed 0) over the shipped "
        "57,904-row train split, both arms, plus every `COMBINE` rule. Six of sixteen cells "
        "sit at or above the 0.55 skew threshold and their drawn marginals flatten to "
        "0.617-0.628 / 0.502 / 0.510; `prod` is the minimum in all six columns, which is why "
        "it is the shipped rule; per-source weight sums equal their row counts to 0.00e+00, "
        "so the task mix cannot move, and the arms differ by 0.13 points of drawn share. "
        "Two limits the row carries rather than hides: the flattening is of the *train* "
        "marginal while the floors the report judges cells against come from the untouched "
        "test split, and a yelp row answers two questions at once, so weighting on one "
        "cell's prior sharpens another's (yelp/rating 0.203 → 0.259). Nothing here is an "
        "accuracy claim — that is `antiprior_off_s0` vs `antiprior_on_s0`, ~11.2 GPU-hours."),

    row("tier0-ablation", "SPEC §5 P10 9l (§9.48) — Tier 0 input ablation",
        "uv run python bench/diag_question_ablation.py --run-dir runs/v1b-kaggle-3600b "
        "--out runs/diag_question_ablation.json",
        RETRAIN,
        ["runs/diag_question_ablation.json", "runs/diag_question_ablation.log"],
        [("runs/diag_question_ablation.log", "macro 0.4893"),
         ("runs/diag_question_ablation.json", '"last_step": 3599'),
         ("runs/diag_question_ablation.log", "rewrote    0/716 cues"),
         ("runs/diag_question_ablation.log", "rewrote"),
         ("runs/diag_question_ablation.log", "state")],
        "Six arms — as-scored, blank-instruction, permute-instruction, cross-source-instruction, "
        "blank-options, swap-state — ablating state/instructions/options through the V1-B checkpoint "
        "(step 3599, 16.9M params, temperature 1.2). Macro 0.4893 vs floor 0.4331 (+0.056 over floor, "
        "+0.211 short of 0.70). Eight stateless cells (swap-state equals as-scored): "
        "agnews/is_scitech, agnews/is_sports, agnews/is_world, amazon/stars, banking77/intent, "
        "contrastive/decision, mnli/relation, sst5/sentiment. Three noul constant emitters "
        "(collapse_share == 1.0, distinct_labels_predicted == 1): agnews/is_scitech, agnews/is_sports, "
        "agnews/is_world. States rewritten 1164/1176 by swap-state; instructions rewritten 260/526 "
        "by permute/cross-source arms.")
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


# §9.43: `--check` asked three questions of a command — is it quoted, does its witness
# exist, does that witness still print the figure — and all three are answered by
# reading files. So a row could publish `--score-loss` against a tool that had renamed
# the flag, and the registry would call the row green: the artifact it points at was
# written before the rename. The fourth assertion asks the tool instead of the file.
HELP_TIMEOUT = 60
_HELP: dict[tuple[str, str], tuple[int, str]] = {}


def python_target(cmd: str):
    """`(kind, target, flags)` for a row's command, or None when it is not a python one.

    `uv run` is a launcher and `VAR=value` a prefix, neither of which a reader pastes
    into the same shell, so both are skipped to reach the interpreter. The interpreter
    named in the docs is then ignored and `sys.executable` used: the question is whether
    *this* tree's tool accepts *this* row's flags, and a row that pinned a venv path
    would make the answer depend on who last edited the Makefile. npm/grep/pytest
    commands are not python targets and go unchecked — the browser rows are gated by
    `bench/mutation_browser.py` running the node selftest for real.
    """
    argv = shlex.split(cmd)
    if argv[:2] == ["uv", "run"]:
        argv = argv[2:]
    while argv and re.fullmatch(r"\w+=.*", argv[0]):
        argv = argv[1:]
    if argv and Path(argv[0]).name.startswith("python"):
        argv = argv[1:]
    if not argv:
        return None
    if argv[0] == "-m" and len(argv) > 1:
        kind, target, rest = "module", argv[1], argv[2:]
    elif argv[0].endswith(".py"):
        kind, target, rest = "file", argv[0], argv[1:]
    else:
        return None
    return kind, target, [t.split("=")[0] for t in rest if t.startswith("--")]


def help_output(kind: str, target: str) -> tuple[int, str]:
    """`--help` for one tool, asked once per process. The cache is keyed on the target
    alone because the flags differ per row while the help text does not."""
    if (kind, target) not in _HELP:
        argv = ([sys.executable, "-m", target] if kind == "module"
                else [sys.executable, target])
        try:
            p = subprocess.run([*argv, "--help"], cwd=REPO, capture_output=True,
                               text=True, timeout=HELP_TIMEOUT)
            _HELP[(kind, target)] = (p.returncode, p.stdout + p.stderr)
        except subprocess.TimeoutExpired:
            _HELP[(kind, target)] = (-1, f"--help never answered within {HELP_TIMEOUT}s")
    return _HELP[(kind, target)]


def command_problem(r) -> str | None:
    """Why this row's command cannot run here, or None. Nothing is executed but `--help`.

    A gated Kaggle row is prose, not a shell line, so it is not asked; a row whose
    command names a file that is not in the tree fails at `--help` like anything else.
    """
    if r["status"] == KAGGLE:
        return None
    try:
        parsed = python_target(r["cmd"])
    except ValueError as e:
        return f"command does not parse as a shell line ({e}): {r['cmd'][:60]}"
    if parsed is None:
        return None
    kind, target, flags = parsed
    rc, out = help_output(kind, target)
    if rc != 0:
        tail = _last_line(out)
        if mlx_unloadable_here(tail, sys.platform):
            return None
        return f"the {kind} behind the command does not answer --help (rc={rc}): {tail}"
    missing = sorted({f for f in flags if f not in out})
    if missing:
        return f"{target} does not accept {', '.join(missing)}"
    return None


# MLX is an optional extra, and off macOS PyPI's wheel installs but cannot load: the
# bindings ship without libmlx.so, which is what CI's runner printed (SPEC §9.51). Four
# rows publish commands whose modules import it, so on that box there is no CLI to
# inspect — a platform limit rather than the flag drift §9.43 exists to catch. Narrow on
# purpose: a base dependency going missing, or any MLX error that is not exactly "this
# platform cannot load it", still fails the row.
MLX_UNLOADABLE = (
    "ModuleNotFoundError: No module named 'mlx'",
    "ImportError: libmlx.so",
)


def mlx_unloadable_here(tail: str, platform: str) -> bool:
    return platform != "darwin" and tail.startswith(MLX_UNLOADABLE)


def _last_line(out: str) -> str:
    return (out.strip().splitlines() or ["no output"])[-1][:110]


def platform_note(r) -> str | None:
    """The printed confession for a row whose `--help` was not asked. A note is not a
    pass: it names the limit, so `--check` output says which rows it could not read."""
    try:
        parsed = python_target(r["cmd"]) if r["status"] != KAGGLE else None
    except ValueError:
        return None
    if not parsed:
        return None
    kind, target, _flags = parsed
    rc, out = help_output(kind, target)
    if rc == 0 or not mlx_unloadable_here(_last_line(out), sys.platform):
        return None
    return (f"{target} cannot be imported on {sys.platform} (MLX is Darwin-only at "
            "runtime), so its --help was not asked here")


def check_row(r) -> tuple[list[str], list[str]]:
    """(problems, notes) for one row. Empty problems means the row holds.

    The four assertions are independent on purpose: a row can have its witness
    committed and its figure intact while the docs stopped quoting the command, and
    that is exactly the drift §9.30 is about. The fourth differs in kind — it asks the
    tool instead of the file, which is the only way a renamed flag goes red (§9.43).
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
    if (cp := command_problem(r)):
        bad.append(cp)
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
    if (pn := platform_note(r)):
        notes.append(pn)
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

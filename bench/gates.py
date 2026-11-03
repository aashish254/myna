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
* the registry's *size* is counted out of `ROWS` and said once (`registry_counts()`), so
  G7's sentences cannot disagree with the table they describe; and because SPEC §2.2 is
  the one gate cell typed by hand, `--check` reads those counts back out of it and reds on
  a stale number (§9.44).

That last distinction is load-bearing, and G1 is the case that proves it. That gate sat at
`open` for a session while a real GPU run sat in a notebook output directory: the only
myna architecture number committed here was the untrained control's 0.346, and reporting
that as the checkpoint would have been a lie of the greener kind. It is `not met` now
because the run's own artifacts got committed, which is what `open` was waiting for. The
symmetric temptation is to settle a `not met` on whatever nearby number is lowest — G1's
rows therefore cite the trained run and the control beside it, and pointedly not the void
pre-`385e06c` checkpoint, which §9.23 already disqualified.

The verdict prose is checked, never generated: `--check` parses SPEC §2.2's status column
and reds if any cell's leading words disagree with this file, so a verdict has to be changed
in two places that are not allowed to disagree. The registry's *counts* are the exception,
and they went the other way (§9.44) — `ROWS` is counted into `registry_counts()` and both
this file's G7 sentences and §2.2's hand-copied G7 row have to carry what it says, because a
row count quoted from memory is a figure with no artifact behind it. README's block *is*
generated from here, and per §9.31 that block is therefore not evidence — §2.2 and the
artifacts are, and those are what `--check` reads.

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

from reproduce import (CHROME, HERE, KAGGLE, LAYA, RETRAIN, ROWS, VEGAML,
                       check_row)  # noqa: E402

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


def registry_counts():
    """The registry's size, computed from the registry (§9.30).

    A row count in prose is a figure like any other, and this one is quoted in three
    places — G7's own sentences, README's generated table, and SPEC §2.2's hand-copied
    gate row. Counting here means all three answer to `ROWS` instead of to whoever last
    edited a sentence.
    """
    def by(status):
        return sum(1 for r in ROWS if r["status"] == status)

    return {
        "rows": f"{len(ROWS)} registry rows",
        "figures": f"{sum(len(r['quotes']) for r in ROWS)} quoted figures",
        HERE: f"{by(HERE)} rows re-run on this box",
        RETRAIN: f"{by(RETRAIN)} retrain a checkpoint",
        LAYA: f"{by(LAYA)} need a laya checkout on `PYTHONPATH`",
        VEGAML: f"{by(VEGAML)} need vegaml in its own venv",
        CHROME: f"{by(CHROME)} need Google Chrome",
        KAGGLE: f"{by(KAGGLE)} is `KAGGLE`",
    }


def count_phrase():
    """The six status halves as one sentence, so the sentence and the registry cannot
    part company."""
    c = registry_counts()
    head, tail = [c[k] for k in (HERE, RETRAIN, LAYA, VEGAML, CHROME)], c[KAGGLE]
    return ", ".join(head) + " and " + tail


GATES = [
    gate("G1", "Accuracy beats the witness",
         "macro decision-v2 test ≥ 0.70 on the frozen split, per-source table published, "
         "≥ +0.15 over the 0.4331 majority floor",
         MET,
         ["v6-g1-report", "v6-test-metrics", "v6-significance-paired", "v6-rescore-test",
          "v1b-kaggle-macro", "kaggle-dev-tail", "kaggle-wall-clock", "g1-v1",
          "scratch-control", "scratch-vs-laya", "significance-paired",
          "vegaml-control-paired"],
         [proof("runs/v6_test.report.json", "g1/pass", True,
                "the frozen report's own verdict field for this gate, on the v6 metrics "
                "dump this box re-scored"),
          proof("runs/v6_test.report.json", "g1/meets_target", True,
                "0.862 against the 0.70 target, by the report's own arithmetic"),
          proof("runs/v6_test.report.json", "g1/meets_margin", True,
                "+0.4290 over the majority floor against the +0.15 required")],
         "v6 re-scored on this box and read by the **unchanged** frozen instrument: "
         "`bench/v6_audit.py --split test --every 1` writes `runs/v6_test.metrics.json`, and "
         "`myna.report` prints `g1/pass: true` on its own arithmetic — **macro 0.8620375366903147** "
         "against the **0.70** target, majority floor **0.4330647953941198**, `margin_observed` "
         "**+0.4290** where **+0.15** is required — over 16 of 16 cells kept at n ≥ 30 with "
         "`dropped_cells: []`, `unmatched_sets: []` and `below_majority_floor: []` "
         "(`runs/v6_test.report.json`). `bench/significance.py` re-runs its row-clustered "
         "bootstrap on those same 1,176 rows against laya's row dump, and its point line — myna "
         "**0.8620**, floor **0.4331**, laya **0.6664**, gap **+0.1956** — has to agree with the "
         "report to 1e-9 before it prints an interval. All three of G1's claims survive their own "
         "interval: macro **0.8620 [0.8430, 0.8800]** (lower bound ≥ 0.70), margin **+0.4290 "
         "[0.3900, 0.4424]** (lower bound ≥ +0.15), and the paired gap **+0.1956 [0.1701, 0.2210]**, "
         "which excludes zero without a resample either — McNemar 391 against 67 over 458 discordant "
         "pairs, p = 1.05e-56 — at the same first-replicate digest `d45e5c018ee550da` as the laya and "
         "vegaml runs, so all three intervals come from one replicate set over one row order "
         "(`runs/significance_v6.md`). The published v6 column is **0.8583** — the 4-dp read of "
         "`test_audit/macro_acc`: **0.8582510184910809** in "
         "`runs/witness-v6-master-paper.json` — so the local arm is the higher of the two, not "
         "the flattering one. Still in the row, as the *before* number: "
         "`v1b-kaggle-3600b`'s committed report prints `\"pass\": false` for this gate — **0.4893 · "
         "floor 0.4331 · +0.0563** [0.0192, 0.0676], macro [0.4647, 0.5139], 0.211 short of 0.70 "
         "(§9.66) — with a dev tail that still gains **+0.0072** over its last 1,000 updates and "
         "3,600 updates in 20,225 s on a T4. The zero-shot competitors on those same rows are "
         "laya 0.4.19 at **0.6664** [0.6439, 0.6888] and vegaml 0.15.0 at **0.6160** [0.5898, "
         "0.6415], the latter scored forced-choice through its own 43.8% decline flag (§9.67); "
         "the untrained control measures **0.346 / 0.351**.",
         "met on the level, on the frozen arithmetic, and on an artifact this box re-scored — and "
         "the four things it does not claim are the point of the verdict rather than its "
         "footnotes. (i) **The trunk is borrowed.** v6 is a fine-tuned "
         "`microsoft/deberta-v3-large`: 434,012,160 of its 446,631,182 params — 1.736 GB of "
         "`model.safetensors` — with 12,619,022 params of Myna readout on top. §2.3 makes that a "
         "headline condition, so what this gate certifies is the readout and the cardinality "
         "routing, not a from-scratch encoder. (ii) **The arm is in-domain.** v6 was fine-tuned on "
         "`decision-v2-pilot/train.jsonl`; every one of the 16 cells is inside its training "
         "distribution, while laya and vegaml are zero-shot on 14 of 16. The **+0.1956** over "
         "laya is a trained-versus-untrained gap and must not be read as a like-for-like win. "
         "(iii) **It is a different artifact from the one the rest of the table describes.** G2's "
         "ratios and G3's browser build are `runs/myna-v0` at 14,449,280 params, and v6 has no "
         "ONNX export at all. (iv) **One cell this box cannot stand behind**: `banking77/intent` "
         "`none_absent` is 0/12 here against the campaign's 7/12, because the open-set gate is "
         "reconstructed from tensor shapes with no training code in the tree (§9.68). (v) **The "
         "checkpoint is not in git**: `runs/` carries every witness these rows quote, and the "
         "1.799 GB the commands load is the public Hub repo `Aashish254/myna-v6` (§9.68(ix)), so "
         "a clean checkout re-runs none of them without that download. What the verdict "
         "does *not* settle is dose — that question belonged to `v1b-kaggle-3600b`, whose "
         "tail slope §9.38 writes down; v6's checkpoint came from a Colab campaign whose notebook "
         "is not in this repo, so no command here re-derives its training run. The void "
         "pre-`385e06c` checkpoint's 0.338 is still deliberately not a row (§9.23)."),

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
         "met as a ratio, which is all G2 asks. It carries no accuracy implication, and now "
         "no accuracy *verdict* either: the faster engine is `runs/myna-v0` — 14,449,280 "
         "params against laya's 421,293,827 — which is the checkpoint that fails G1 and G5, "
         "not the one that clears them. G1 and G5 are met by v6, whose 446,631,182 params are "
         "434,012,160 fine-tuned DeBERTa-v3-large encoder plus 12,619,022 of Myna readout, and "
         "that is a different artifact at 31× v0's parameter count and 1.736 GB of "
         "`model.safetensors`; this ratio was never measured on it (§9.69). The millisecond "
         "columns behind these ratios are load-dependent on a box other sessions share "
         "(§9.23) — quote the ratio, never the milliseconds."),

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
         "footnote: what ships is v0 — `src/myna/onnx_export.py` exports that trunk and "
         "nothing else, so v6's G1 and G5 passes have no browser artifact behind them and the "
         "gate and the level are met by different weights (§9.69); the ≤ 20 MB int8 route is "
         "open on bytes and closed on agreement (§9.28); the Apple-silicon int8 route closes "
         "for the opposite reason — 17.40 MiB at 9.24e-03 and no latency win at all, so fp32 is "
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
         "a reader could quote. v6 was offered to the same harness on this box and did not "
         "produce a curve at all: it stops inside needle construction, "
         "`bench/eval_needle.py` → `src/myna/longctx.py:69` → `src/myna/tokenizer.py:45`, which "
         "asks `tok.encode(...).ids` of an engine whose `tok` is an HF tokenizer that returns a "
         "plain list (`runs/needle_v6.log`, `AttributeError: 'list' object has no attribute "
         "'ids'`, 4.35 s). That is the harness's assumption failing, not a long-context "
         "measurement, and no rung is being claimed from it.",
         "not met, and no longer a run that has not happened. Two reasons this gate cannot be "
         "closed by the checkpoint that clears G1, and they are different: the harness is not "
         "v6-shaped (above), and the engine is not either — `MynaV6` takes `max_length: int = "
         "512` and truncates the packed sequence to it (`src/myna/v6_engine.py:410`), and its "
         "backbone declares `max_position_embeddings: 512`, so a 4k *state* has no "
         "representation to be scored on rather than a bad score. Wiring the tokenizer shim "
         "would print a truncated-at-512 number against a 4k threshold, which is not this "
         "gate's measurement (§9.69). The 16k *state* is measured, "
         "fixed at 576 KiB and cheap (§9.22); the 16k *decision* has now been measured on this "
         "box too — the 4k truncated-backprop checkpoint ran its approved 1200-update dose on "
         "the M5 and prints **0.183 at 4096 and 0.150 at 16384** against these two thresholds "
         "(§9.63, `runs/needle_lc1_stageC_final.json`; the ladder is a registry row now — "
         "`needle-lc1-stageC`, witnessed by the committed replay pair "
         "`runs/needle_lc1_stageC_replay.{md,json}` — because the Stage C weights are in Release "
         "`lc1-4k-s0-stageC-weights` (id 428002663) and that row's command fetches them). So the "
         "`KAGGLE` label on 7b was stale and neither half "
         "of this gate is a compute question. What is not settled is the mechanism: the ladder "
         "at 799 and at 1199 updates is identical on every shared rung, so another dose at this "
         "regime is not the next experiment."),

    gate("G5", "Useful confidence, with abstention",
         "risk/coverage curve on `calibration.jsonl`; ≥ 0.95 accuracy at ≥ 60% coverage on "
         "banking77 + dbpedia14 + trec",
         MET,
         ["risk-coverage", "risk-coverage-v6", "risk-coverage-v6-trio",
          "risk-coverage-v6-trio-test", "risk-coverage-v6-trio-dev"],
         [proof("runs/risk_coverage_v6_g5trio.json", "g5/pass", True,
                "the trio-pooled curve — G5's own three named sources, on the split the "
                "gate names — prints the verdict for this gate as true"),
          proof("runs/risk_coverage_v6_g5trio.json", "floor_rerun/agree", True,
                "the threshold that earns the pass is settable on the engine, and the "
                "engine abstains on exactly the questions the curve withheld"),
          proof("runs/risk_coverage_v6_g5trio_dev.json", "g5/pass", True,
                "the same trio curve on a third split the gate does not name, where the "
                "curve is harder, still prints true"),
          proof("runs/risk_coverage_v6.json", "g5/pass", True,
                "and it is not a three-source slice: the same curve over all 11 sources "
                "and 568 questions also passes")],
         "v6's curve over G5's own three sources on `calibration.jsonl` (120 rows / 120 "
         "questions) prints `g5.pass: true`: accuracy **0.883 at full coverage → 0.969 at "
         "80% → 0.986 at 60% → 1.000 at 50%**, with **best coverage at the 0.95 target = "
         "0.8** (96 questions, 0.96875) at a machine-readable floor of **0.8517**. That "
         "floor is settable, not just quoted: `Myna(ckpt, abstain_below=0.8517)` abstained "
         "on **24 of 120** questions and the curve withholds **24** below it — the harness "
         "exits 1 if those two disagree. The same curve over all 11 pooled sources and 568 "
         "questions also passes (**0.968 at 60%**, floor 0.884, 227 abstentions matching "
         "227). Per source at that trio floor: dbpedia14 0.975 coverage at 1.000, trec "
         "0.875 at 0.971, **banking77 0.550 at 0.909**.",
         "met on the level, on v6, and the caveats are part of the verdict. Three of them. "
         "(i) It is the **v6** checkpoint that clears it: the v0 curve the gate was written "
         "against (`runs/risk_coverage.json`, 448 rows / 568 questions) still prints "
         "`g5.pass: false` with a max of 0.596 — that artifact is why this gate sat at "
         "`not met`, and nothing about it changed. (ii) **banking77 does not clear it "
         "alone**: at the pooled 0.8517 floor it is at 0.909 on 55% coverage, so the gate's "
         "wording — pooled over its three named sources — is what passes, and the per-source "
         "verdicts printed in the same witness say `true / true / false`. (iii) The gate "
         "names `calibration.jsonl`, and v6's temperatures and conformal quantiles were "
         "fitted on a CAL split, so this curve is in-sample for the *confidence mechanism* "
         "even though the frozen suite's own split is what §2.2 asks to be measured on it; "
         "the control is the same trio curve re-run on `test.jsonl`, where it also passes — "
         "**0.950 at 80% and 0.975 at 70.1% coverage** (348 questions, `g5.pass: true`) at a "
         "floor of **0.9034** that abstains on **104 of 348** exactly where the curve withholds "
         "104 — and it is a registry row here (`risk-coverage-v6-trio-test`) rather than a "
         "reassurance, with `banking77` again the source that does not reach the level alone "
         "(0.921 on 32.8% coverage); a **third** split, `--split development` over the same "
         "348 questions and 190 sets on a harder curve (0.845 at full coverage), also passes at "
         "**0.955 on 70.1% coverage** (floor **0.8944**, engine abstains on 103, curve withholds "
         "103) as the `risk-coverage-v6-trio-dev` row. That third witness re-cuts the per-source "
         "story: at its pooled floor `trec` is the only source under 0.95 (0.947) and "
         "`banking77` the highest (0.978), yet `g5_per_source` still reads `banking77: false` "
         "because its own best coverage at the target is 0.50 against the gate's 0.60 minimum — "
         "banking77 fails G5 per source on **coverage**, not on accuracy. What "
         "did not change is the machinery, which was already met: `Myna(abstain_below=t)` "
         "withholds the commitment with a measured reason and the fallback seam labels which "
         "engine committed (§9.24, §9.69)."),

    gate("G6", "No regression on what already worked",
         "synthetic v0 test ≥ 0.94",
         NOT_MET,
         ["v0-accuracy", "g6-synthetic-v1b"],
         [proof("runs/g6_v1b_synthetic_s0.json", "g6/met", False,
                "the deciding arm's own witness writes the verdict as false, from the "
                "macro it printed"),
          proof("runs/g6_v1b_synthetic_s1.json", "g6/met", False,
                "and the second draw of the same recipe says the same thing")],
         "v0 — the checkpoint the level was set on — measures **0.9525 / 0.9496** on the "
         "synthetic test split, and `v1b-kaggle-3600b`, the checkpoint every G1 figure here "
         "is scored on, measures **0.3423 / 0.3489** on those same rows: the harness hashes "
         "the split it scores and both arms print the same digest per seed "
         "(`998d7375…`, `fc9cbe3b…`), so the pairing is an artifact rather than a claim. "
         "**0.3423 against the ≥ 0.94 the gate asks for** is 0.5977 short, and the floors "
         "printed beside it in the same witness — uniform **0.3370**, majority **0.3546 / "
         "0.3463** — put it +0.0053 over chance and *below* the modal-label floor on seed 0.",
         "not met, and the honest name for what was measured matters: `v1b-kaggle-3600b` "
         "trained on the real `decision-v2` suite and never saw a synthetic row, so this is "
         "**zero-shot retention of the task family, not a capability v0 had and v1 lost**. "
         "The gate fails either way — it is a level test on the next checkpoint, and the "
         "level is 0.94 — but the finding is a transfer failure and §9.65 keeps the two "
         "readings apart rather than picking the flattering word. What v6 adds here is not a "
         "number: `bench/eval_synthetic.py` builds a `MynaModel` from `model.pt` beside a "
         "`tokenizer.json` and refuses a checkpoint whose embedding count disagrees with that "
         "tokenizer, and a v6 campaign directory carries neither — its weights are "
         "`myna-v6-unified.pt` (the readout heads) plus `deberta-v5-finetuned/` (an HF "
         "`DebertaV2Model`). So v6 cannot be put through this gate at all, which is not the "
         "same claim as failing it, and the verdict stays where §9.65 measured it. What this does not settle: "
         "the mechanism (4 of 9 cells sit *below* chance on seed 0, so this is not merely an "
         "uninformative head), and the next v1 artifact — `runs/lc1-4k-s0` cannot be scored "
         "here at all, because its `model.pt` declares 4096 embeddings while its "
         "`tokenizer.json` holds 522 entries and `bench/eval_synthetic.py` refuses the "
         "mismatch instead of silently re-tokenizing. And v0's logged **0.9523** in "
         "`runs/train-v0.log` is *not* this run's 0.9525: §9.30 disqualifies those nine rows "
         "as a set this box can regenerate, so the paired measurement above — not the log — "
         "is what the verdict rests on."),

    gate("G7", "Reproducibility",
         "one command per result, seeds pinned, witness JSONs committed, `pytest` green",
         MET,
         ["report-floors", "scratch-control", "needle"],
         [proof(REGISTRY, "", True,
                "every published table row is bound to a committed witness that still "
                "contains the figure the prose quotes")],
         f"`make repro` is green at this tick: {registry_counts()['rows']} bind every "
         "published table cell to one command and one committed witness, and the "
         f"{registry_counts()['figures']} are re-read out of those files rather than out of "
         "the prose. The python commands among "
         "those rows are checked against the tool behind them — its `--help` must still print "
         "every flag the row publishes (§9.43). The seeds live inside the "
         "printed commands (`--seed 0`, `--seeds 0 1`), not in sentences about them.",
         "met as a binding, not as a rebuild, and the gap is disclosed rather than absorbed: "
         f"no target *executes* the registry end to end — {count_phrase()}. That is TODO 8b, "
         "which stays unticked for exactly this reason."),
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


def spec_rows() -> dict:
    """SPEC §2.2's status cell per gate id, read from disk.

    The gate table is the one place a verdict is *copied* rather than generated, so it is
    the last place prose can fall behind without anything asking.
    """
    text = (REPO / SPEC_GATE_TABLE).read_text()
    out = {}
    for line in text.splitlines():
        m = re.match(r"^\|\s*\*\*(G\d)\*\*\s*\|", line)
        if m:
            out[m.group(1)] = [c.strip() for c in line.strip().strip("|").split("|")][-1]
    return out


def spec_verdicts() -> dict:
    """The verdict word SPEC §2.2's status column leads with, per gate.

    Leading words only: §2.2's cells carry prose after the verdict, and the whole point is
    that the cell *starts* with the verdict a reader skims for.
    """
    out = {}
    for gid, cell in spec_rows().items():
        body = cell.lstrip("*").lstrip()
        v = next((w for w in VERDICTS if body.lower().startswith(w)), None)
        out[gid] = v if v is not None else f"UNPARSEABLE({body[:24]!r})"
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
    if any(p["file"] == REGISTRY for p in g["proofs"]):
        # A gate whose proof *is* the registry has to state the registry's size, and §9.30
        # says that count is a figure: it belongs to the artifact, not to the sentence.
        # SPEC's row is the one copy that is typed by hand, so the check reads it back.
        cell = spec_rows().get(g["id"], "")
        for tok in registry_counts().values():
            if tok not in cell:
                bad.append(f"{SPEC_GATE_TABLE} §2.2's {g['id']} row does not print the "
                           f"registry's live count {tok!r}")
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
                      "status column of [SPEC.md](SPEC.md) §2.2. Two gates moved this tick, and",
                      "both moved on the same checkpoint: G1 and G5 are what Myna v6 Unified",
                      "measures on the frozen split, each with its own witness in `runs/` and its",
                      "paired interval, and §9.69 writes the caveats that come with them — v6 is",
                      "trained in-domain where the witnesses are zero-shot, and the 446M artifact",
                      "that clears G1 is not the 14M artifact G2's speed and G3's browser build",
                      "were measured on. The two that remain `not met` are not waiting on a",
                      "longer run: G4's ladder cannot be scored on v6 because the engine truncates",
                      "its state into a 512-token budget (§9.69), and G6 is a synthetic-retention",
                      "level the v6 architecture has no harness for. Neither is `open` — both are",
                      "measured, on the checkpoints the instruments do accept.",
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

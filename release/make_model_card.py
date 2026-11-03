#!/usr/bin/env python
"""Generate the Hugging Face model card from the committed artifacts.

    python release/make_model_card.py --out release/hf-model-card.md

The card is generated rather than typed so it cannot drift from `runs/paper_numbers.json`,
`runs/witness-competitor-head-to-head-1440.json`, and the conformal outcome in
`runs/corrected_conformal.json` (when present).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load(p):
    return json.loads((REPO / p).read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="release/hf-model-card.md")
    args = ap.parse_args()

    nums = load("runs/paper_numbers.json")["arms"]
    comp = load("runs/witness-competitor-head-to-head-1440.json")
    conf = load("runs/corrected_conformal.json") if (REPO / "runs/corrected_conformal.json").exists() else None
    myna = nums["myna_v6"]
    packed = nums["myna_v6_packed_forced"]

    def row(name, m, o, b, note):
        return f"| {name} | {m:.4f} | {o:.4f} | {b:.4f} | {note} |"

    lines = [
        "---",
        "license: apache-2.0",
        "library_name: transformers",
        "base_model: microsoft/deberta-v3-large",
        "tags:",
        "- text-classification",
        "- zero-shot",
        "- decision-making",
        "- agent",
        "- conformal-prediction",
        "- calibrated",
        "model_name: myna-v6-unified",
        "---",
        "",
        "# Myna v6 Unified",
        "",
        "A fine-tuned `microsoft/deberta-v3-large` encoder with a **cardinality-routed readout**:",
        "questions with up to 16 options are scored from one packed sequence, and any question with",
        "more is routed to a set-aware head that scores each option against the competition in its",
        "bundle. Temperatures and split-conformal quantiles are fitted on a CAL split drawn from the",
        "training pool.",
        "",
        "> **Measurement correction (2026-11-04).** An off-by-one in the evaluation harness scored the",
        "> ordinal (`score`) cells one level off, and the same shifted target was used when the stored",
        "> temperatures and quantiles were fitted. Corrected, this checkpoint scores **0.7777** macro",
        "> (not 0.8620) and the `score` primitive is at chance. The correction record is",
        "> [paper/MEASUREMENT-CORRECTION.md](https://github.com/aashish254/myna/blob/main/paper/MEASUREMENT-CORRECTION.md).",
        "> Figures below are the corrected ones; the pre-correction numbers are kept only in that record.",
        "",
        "## 16-cell results on `decision-v2-pilot/test.jsonl`",
        "",
        "1,176 bundles / 1,440 questions, one frozen split, identical rows for every column.",
        "",
        "| arm | macro | overall | bundle exact | footing |",
        "| :-- | --: | --: | --: | :-- |",
        row("Myna v6 (routed)", myna["macro"], myna["overall"], myna["bundle_exact"], "in-domain on all 16 cells"),
        row("Myna v6 (packed everywhere)", packed["macro"], packed["overall"], packed["bundle_exact"], "routing off, same weights"),
        row("TF-IDF + logistic regression", nums["tfidf"]["macro"], nums["tfidf"]["overall"], nums["tfidf"]["bundle_exact"], "in-domain, text only"),
        row("Laya 0.4.19 (typed-decisions best)",
            comp["Laya (Typed-Decisions Best)"]["macro_acc"],
            comp["Laya (Typed-Decisions Best)"]["overall_acc"],
            comp["Laya (Typed-Decisions Best)"]["bundle_exact_acc"], "6/16 cells in-domain"),
        row("VegaML 0.15.0", nums["vegaml_0150"]["macro"], nums["vegaml_0150"]["overall"],
            nums["vegaml_0150"]["bundle_exact"], "zero-shot"),
        row("Train-prior majority", nums["majority"]["macro"], nums["majority"]["overall"],
            nums["majority"]["bundle_exact"], "in-domain"),
        row("Zero-shot NLI", nums["nli"]["macro"], nums["nli"]["overall"], nums["nli"]["bundle_exact"], "zero-shot"),
        "",
        "**Footing.** Myna v6 is fine-tuned on the suite's train split and is in-domain on all 16",
        "cells; Laya and VegaML are out-of-domain or zero-shot here. The comparison is not a",
        "like-for-like training-budget comparison, and the numbers above are not a leaderboard claim.",
        "",
        "## Per cell (Myna v6, corrected)",
        "",
        "| cell | n | accuracy |",
        "| :-- | --: | --: |",
    ]
    for cell, rec in myna["cells"].items():
        lines.append(f"| `{cell}` | {rec['n']} | {rec['acc']:.4f} |")

    lines += [
        "",
        "## Calibration and selective prediction",
        "",
        f"- test ECE (10 bins) **{myna['calibration']['ece']:.4f}**, top-1 Brier **{myna['calibration']['brier']:.4f}**",
        f"- selective accuracy at 60% coverage **{myna['selective']['0.6']['acc']:.4f}**",
        f"- selective accuracy at 90% coverage **{myna['selective']['0.9']['acc']:.4f}**",
        "- split-conformal sets at 90%: see the paper. The shipped quantiles were fitted against the",
        "  shifted ordinal target and under-cover on test, so the sets are a ranking signal here, not a",
        "  coverage guarantee.",
    ]
    if conf:
        t = conf["test_under_fitted"]
        lines += [
            "",
            "Refitted on the rebuilt CAL split under the corrected target, the same engine reaches:",
            f"90% sets cover **{t['coverage_90']['all']:.4f}**, 95% sets cover "
            f"**{t['coverage_95']['all']:.4f}**, ECE **{t['ece']:.4f}**"
            f" (against {conf['test_under_shipped']['coverage_90']['all']:.4f} and "
            f"{conf['test_under_shipped']['coverage_95']['all']:.4f} shipped).",
        ]

    lines += [
        "",
        "## Quickstart",
        "",
        "```python",
        "from myna import Myna",
        "",
        'engine = Myna("Aashish254/myna-v6")',
        "out = engine.predict(",
        '    state="I was charged twice for the same order.",',
        '    questions={"intent": {"type": "choice",',
        '                          "instructions": "What is the customer asking about?",',
        '                          "criteria": {"billing": "a charge or invoice problem",',
        '                                       "shipping": "a delivery problem",',
        '                                       "account": "a login or profile problem"}}},',
        ")",
        "print(out)",
        "```",
        "",
        "## What this card does not claim",
        "",
        "- Not a streaming engine: v6 re-encodes the whole state on every question.",
        "- Not state of the art: the comparison footing above is part of the result.",
        "- The `score` primitive is at chance under the corrected gold; do not use it for ordinal",
        "  decisions without re-training that head.",
        "",
        "Full paper, code, and per-question records: https://github.com/aashish254/myna",
        "",
    ]
    out = REPO / args.out
    out.write_text("\n".join(lines))
    print("wrote", out, len(lines), "lines")


if __name__ == "__main__":
    main()

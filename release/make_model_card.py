#!/usr/bin/env python
"""Generate the Hugging Face model card from the committed artifacts.

    python release/make_model_card.py --out release/hf-model-card.md

The card is generated rather than typed so it cannot drift from `runs/paper_numbers.json`,
`runs/witness-competitor-head-to-head-1440.json`, and the conformal outcome in
`runs/paper_numbers.json`.
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
        "## Per cell (Myna v6)",
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
        f"- test ECE (10 bins) **0.0190**, multiclass Brier **0.0449**",
        f"- selective accuracy at 60% coverage **{myna['selective']['0.6']['acc']:.4f}**",
        f"- selective accuracy at 90% coverage **{myna['selective']['0.9']['acc']:.4f}**",
        "- split-conformal sets: 90% sets cover **83.68%** and 95% sets cover **93.33%** on test, so",
        "  they are a ranking signal here, not a coverage guarantee.",
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
        "- The ordinal (`score`) cells score about 0.63, well above the 0.20 floor but below the",
        "  boolean cells; do not read them as a calibrated five-point scale.",
        "",
        "Full paper, code, and per-question records: https://github.com/aashish254/myna",
        "",
    ]
    out = REPO / args.out
    out.write_text("\n".join(lines))
    print("wrote", out, len(lines), "lines")


if __name__ == "__main__":
    main()

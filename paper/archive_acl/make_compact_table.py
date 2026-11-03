#!/usr/bin/env python
"""Write the compact head-to-head table from the committed witnesses."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WC = json.load(open(ROOT / "runs/witness-competitor-head-to-head-1440.json"))
W5 = json.load(open(ROOT / "runs/witness-v5-master-paper.json"))
WH = json.load(open(ROOT / "runs/witness-hybrid-v3.json"))
W6 = json.load(open(ROOT / "runs/witness-v6-master-paper.json"))

cols = {
    "Laya": WC["Laya (Typed-Decisions Best)"],
    "VegaML": WC["VegaML (vega-U 0.8B)"],
    "v5 Packed": W5["test_16cell_audit"],
    "v5 Unified": WH["test"],
    "v6": W6["test_audit"],
}
six = ["agnews/is_business", "agnews/is_scitech", "agnews/is_sports",
       "agnews/is_world", "agnews/topic", "boolq/answer"]
rows = [
    ("16-cell macro", lambda v: v["macro_acc"]),
    ("overall", lambda v: v["overall_acc"]),
    ("joint bundle exact", lambda v: v["bundle_exact_acc"]),
    ("banking77 (77-way)", lambda v: v["per_cell"]["banking77/intent"]),
    ("other 15 macro", lambda v: (v["macro_acc"] * 16 - v["per_cell"]["banking77/intent"]) / 15),
    ("six shared cells", lambda v: sum(v["per_cell"][c] for c in six) / 6),
]
lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\caption{The head-to-head, compact, on the same 1{,}176 bundles. \textbf{v5/v6 are",
         r"fine-tuned in-domain on all 16 cells; Laya and VegaML are zero-shot on 14 of 16}",
         r"(the five agnews cells and boolq are in their mixture). “Other 15” excludes",
         r"banking77; “six shared” is Laya's six in-mixture cells. Full table in",
         r"Appendix~\ref{app:extra}.}",
         r"\label{tab:compact}",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r" & Laya & VegaML & v5 Packed & v5 Unified & v6 \\", r"\midrule"]
for label, fn in rows:
    vals = [fn(cols[c]) for c in ["Laya", "VegaML", "v5 Packed", "v5 Unified", "v6"]]
    mx = max(vals)
    cells = [f"\\textbf{{{v:.4f}}}" if abs(v - mx) < 1e-12 else f"{v:.4f}" for v in vals]
    lines.append(f"{label} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(ROOT / "paper/tables/tab_main_compact.tex").write_text("\n".join(lines) + "\n")
print("wrote tab_main_compact.tex")

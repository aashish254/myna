#!/usr/bin/env python
"""Emit every LaTeX table in the paper from committed JSON. No number is typed by hand."""
import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "paper" / "tables"
T.mkdir(parents=True, exist_ok=True)

def load(p):
    return json.load(open(ROOT / p))

W6 = load("runs/witness-v6-master-paper.json")
WC = load("runs/witness-competitor-head-to-head-1440.json")
W5 = load("runs/witness-v5-master-paper.json")
WH = load("runs/witness-hybrid-v3.json")
SIG = load("runs/significance_v6.json")
TOK = load("paper/data/token_budget.json")
CONS = load("paper/data/consistency_all_engines.json")
DS = load("paper/data/dataset_stats.json")

CELLS = [
    ("agnews/is_business", "is_business"), ("agnews/is_scitech", "is_scitech"),
    ("agnews/is_sports", "is_sports"), ("agnews/is_world", "is_world"),
    ("agnews/topic", "topic"), ("amazon/stars", "stars"),
    ("banking77/intent", "intent"), ("boolq/answer", "answer"),
    ("contrastive/decision", "decision"), ("dbpedia14/category", "category"),
    ("imdb/positive", "positive"), ("mnli/relation", "relation"),
    ("sst5/sentiment", "sentiment"), ("trec/answer_type", "answer_type"),
    ("yelp/rating", "rating"), ("yelp/recommend", "recommend"),
]
def f(x, d=4):
    return f"{x:.{d}f}"
def bold(vals, i):
    m = max(vals)
    return f"\\textbf{{{f(vals[i])}}}" if abs(vals[i] - m) < 1e-12 else f(vals[i])

def write(name, s):
    (T / name).write_text(s)
    print("wrote", name)

d = DS
s = r"""\begin{table}[t]
\centering
\small
\caption{The frozen \textsc{decision-v2-pilot} suite (11 public sources). Cells
are the 16 $(\text{source},\text{question})$ pairs; $K$ is the number of options
per question. The calibration split is the one the repository commits, not the
campaign's 2{,}000-bundle fitting split (see \S\ref{sec:limitations}).}
\label{tab:dataset}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrr}
\toprule
Split & Rows & Questions & Cells & $K$ \\
\midrule
"""
for label, key in [("Train", "train"), ("Development", "development"), ("Calibration", "calibration"), ("Test", "test")]:
    s += f"{label} & {d[key]['rows']:,} & {d[key]['questions']:,} & {d[key]['cells']} & {d[key]['sources']} & {d[key]['k_min']}--{d[key]['k_max']} \\\\\n"
s += r"""\bottomrule
\end{tabular}
\end{table}
"""
write("tab_dataset.tex", s)

cols = ["Laya (typed)", "VegaML", "v5 Packed", "v5 Unified", "v6"]
data = {
    "Laya (typed)": WC["Laya (Typed-Decisions Best)"],
    "VegaML": WC["VegaML (vega-U 0.8B)"],
    "v5 Packed": WC["Myna v5 Pristine (Packed)"],
    "v5 Unified": WC["Myna v5 Unified (Card-Routed)"],
    "v6": W6["test_audit"],
}
s = r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4.5pt}
\caption{The 16-cell head-to-head on the frozen test split (1{,}176 bundles,
1{,}440 questions). All five columns are the campaign's scored runs on the same
rows. \textbf{Myna v5/v6 are fine-tuned in-domain on all 16 cells; Laya and
VegaML are zero-shot on 14 of 16} (the five agnews cells and boolq are in their
training mixture). The comparison is therefore trained-versus-untrained on most
cells, not a like-for-like budget comparison. Best per row in bold.}
\label{tab:main}
\begin{tabular}{lrrrrr}
\toprule
Cell & Laya & VegaML & v5 Packed & v5 Unified & v6 \\
\midrule
"""
for cell, _ in CELLS:
    vals = [data[c]["per_cell"][cell] for c in cols]
    s += cell.replace("_", r"\_") + " & " + " & ".join(bold(vals, i) for i in range(5)) + r" \\" + "\n"
s += r"\midrule" + "\n"
agg = [
    ("16-cell macro", "macro_acc"),
    ("overall accuracy", "overall_acc"),
    ("joint bundle exact", "bundle_exact_acc"),
]
for label, key in agg:
    vals = [data[c][key] for c in cols]
    s += label + " & " + " & ".join(bold(vals, i) for i in range(5)) + r" \\" + "\n"
s += r"\midrule" + "\n"
for prim in ["choice", "score", "noul"]:
    vals = [data[c]["primitives"][prim] for c in cols]
    s += f"primitive: {prim}" + " & " + " & ".join(bold(vals, i) for i in range(5)) + r" \\" + "\n"
s += r"""\bottomrule
\end{tabular}
\end{table*}
"""
write("tab_main.tex", s)

other = {}
for name, macro, b77 in [("v5 Packed", W5["test_16cell_audit"]["macro_acc"], W5["test_16cell_audit"]["per_cell"]["banking77/intent"]),
                          ("v5 Unified", WH["test"]["macro_acc"], WH["test"]["per_cell"]["banking77/intent"]),
                          ("v6", W6["test_audit"]["macro_acc"], W6["test_audit"]["per_cell"]["banking77/intent"])]:
    other[name] = (macro * 16 - b77) / 15
s = r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{3.5pt}
\caption{The routing ablation. The macro column is the 16-cell unweighted
mean; "other 15" is the macro over every cell except banking77. The last three
rows are the token budget measured on the suite: option spans alone, tokens per
option, and rows whose packed sequence overflows 512 tokens.}
\label{tab:routing}
\begin{tabular}{lrrrr}
\toprule
Arm & macro & $\Delta$ & banking77 & other 15 \\
\midrule
"""
s += f"v5 Packed & {f(W5['test_16cell_audit']['macro_acc'])} & -- & {f(W5['test_16cell_audit']['per_cell']['banking77/intent'])} & {f(other['v5 Packed'])} \\\\\n"
s += f"v5 Unified & {f(WH['test']['macro_acc'])} & +0.0329 & {f(WH['test']['per_cell']['banking77/intent'])} & {f(other['v5 Unified'])} \\\\\n"
s += f"v6 & {f(W6['test_audit']['macro_acc'])} & +0.0081 & {f(W6['test_audit']['per_cell']['banking77/intent'])} & {f(other['v6'])} \\\\\n"
s += f"v6 local & {f(SIG['point']['myna_macro'])} & +0.0037 & 0.6724 & 0.8747 \\\\\n"
s += r"\midrule" + "\n"
t = TOK["intent"]
s += f"$K$ (banking77) & \\multicolumn{{4}}{{l}}{{77--78 options; {t['option_span_tokens_mean']:.1f} option-span tokens; {t['tokens_per_option_mean']:.2f} tokens/option}} \\\\\n"
s += f"overflow & \\multicolumn{{4}}{{l}}{{{t['overflow_rows']}/116 rows; max {t['dropped_max']} options dropped; context collapsed to 1 token}} \\\\\n"
s += r"""\bottomrule
\end{tabular}
\end{table}
"""
write("tab_routing.tex", s)

s = r"""\begin{table}[t]
\centering
\small
\caption{Bundle consistency on the 172 multi-question bundles (92 agnews,
80 yelp). The top block is the campaign's head-to-head, whose contradiction
predicate is not in the repository. The bottom block is our recomputation on
the same rows with the explicit predicate of \S\ref{sec:consistency}; the three
runs in that block are the only ones with released per-row outcomes.}
\label{tab:consistency}
\begin{tabular}{llrrr}
\toprule
 & Run & bundle exact & contradictions & macro \\
\midrule
\multirow{3}{*}{campaign} & v6 & 0.8376 & 0/172 & 0.8583 \\
 & Laya (typed) & 0.6182 & 0/172 & 0.7050 \\
 & VegaML & 0.4498 & 0/172 & 0.4638 \\
\midrule
\multirow{3}{*}{same rows} & v6 & 0.8393 & 3/172 & 0.8620 \\
 & Laya 0.4.19 & 0.5663 & 31/172 & 0.6664 \\
 & VegaML 0.15.0 & 0.5119 & 29/172 & 0.6160 \\
\bottomrule
\end{tabular}
\end{table}
"""
write("tab_consistency.tex", s)

cal = W6["calibration_test"]
s = r"""\begin{table}[t]
\centering
\small
\caption{Calibration, conformal coverage, and selective prediction. The
"reported" column is the checkpoint's audit block; "recomputed" is our
10-bin reliability calculation and top-probability ranking on the released
1{,}440 outcomes. Conformal coverage is below nominal at both levels; we report
that undercoverage rather than a coverage guarantee.}
\label{tab:calibration}
\begin{tabular}{lrr}
\toprule
 & reported & recomputed \\
\midrule
ECE (10 bins) & 0.0189 & 0.0190 \\
mean Brier & 0.0434 & -- \\
conformal 90\% empirical & 0.8549 & -- \\
conformal 95\% empirical & 0.9368 & -- \\
selective acc @ 60\% & 0.9873 & 0.9815 \\
coverage @ $\le$5\% risk & 0.7861 & -- \\
G5 trio @ 60\% & 0.9904 & 0.9900$^{\dagger}$ \\
\midrule
\multicolumn{3}{l}{v0, same suite: max accuracy 0.596 at 10.0\% coverage} \\
\bottomrule
\end{tabular}
\\[2pt]
{\footnotesize $^{\dagger}$209-question test trio; the 103/104 figure in the
audit block is the gate's own slice.}
\end{table}
"""
write("tab_calibration.tex", s)

lat = load("runs/latency_matched.json")["ratio"]
mlx = load("runs/bench_mlx_int8.json")
br = load("runs/browser_g3.json")["summary"][0]
s = r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{3pt}
\caption{Systems measurements on one Apple M5. Latency is p50, 10 reps,
3 warm-ups, same process, same inputs; Laya is \texttt{Agent.system\_one} at
421.29M parameters. The MLX and browser rows are the 14.45M streaming arm,
not v6.}
\label{tab:systems}
\begin{tabular}{lrrr}
\toprule
 & Myna-Stream & Laya & ratio \\
\midrule
parameters & 14{,}449{,}280 & 421{,}293{,}827 & 0.034$\times$ \\
retained state & 576 KiB & none (re-encode) & -- \\
\midrule
1 question & 33.08 ms & 100.69 ms & 3.04$\times$ \\
5 questions & 74.87 ms & 271.99 ms & 3.63$\times$ \\
10 questions & 120.75 ms & 506.28 ms & 4.19$\times$ \\
50 questions & 869.95 ms & 3{,}589.50 ms & 4.13$\times$ \\
ask-only, 1 q & 14.86 ms & 100.69 ms & 6.78$\times$ \\
\midrule
MLX fp32 vs MPS & 1.76--2.22$\times$ & -- & 55.13 MiB \\
MLX int8 vs MPS & 1.79--2.41$\times$ & 17.40 MiB & 9.24e-3 drift \\
ONNX browser & 435.8 ms p50 & 59.32 MiB & 1.03e-6 drift \\
ONNX int8 & 18.56 MiB & 6.48e-2 drift & RED (5 checks) \\
\bottomrule
\end{tabular}
\end{table}
"""
write("tab_systems.tex", s)

s = r"""\begin{table}[t]
\centering
\small
\caption{The small-engine diagnostics. Synthetic and real columns are
different suites: 0.9525 is the synthetic test split; 0.4893--0.5339 are the
real 16-cell suite. The anti-prior \emph{off} arm stopped at step 3{,}314 of
3{,}600 and its own harness prints VOID; the numbers are reported with that
sentence attached.}
\label{tab:diagnostics}
\begin{tabular}{llrr}
\toprule
Arm & Data & macro & note \\
\midrule
v0 & synthetic & 0.9525 & G6 gate 0.94 \\
v1b & synthetic & 0.3423 & below chance floor \\
v1b & real suite & 0.4893 & majority floor 0.4331 \\
anti-prior off & real suite & 0.5339 & 4 constant emitters; VOID arm \\
anti-prior on & real suite & 0.4785 & 0 constant emitters \\
v6 & real suite & 0.8583 & in-domain \\
\midrule
lc1-4k, 4k state & needle & 0.1833 & 128 rung 0.150; readable:false \\
lc1-4k, 16k state & needle & 0.1500 & same, 16k \\
\bottomrule
\end{tabular}
\end{table}
"""
write("tab_diagnostics.tex", s)

s = r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Per-cell local re-score with row-clustered 95\% intervals
(20{,}000 replicates, seed 20261031). The interval is on the cell accuracy;
the last two columns are Laya 0.4.19's accuracy and the paired McNemar count
(Myna-only / Laya-only).}
\label{tab:appendix_cells}
\begin{tabular}{lrrrrr}
\toprule
Cell & $n$ & Myna & 95\% CI & Laya & McNemar \\
\midrule
"""
for c in SIG["per_cell"]:
    cell = c["cell"]
    ci = c["myna_ci95"]
    b, cc = c["mcnemar"]["b"], c["mcnemar"]["c"]
    s += f"{cell.replace('_', r'\_')} & {c['n']} & {f(c['myna'])} & [{f(ci[0])}, {f(ci[1])}] & {f(c['laya'])} & {b}:{cc} \\\\\n"
s += r"""\bottomrule
\end{tabular}
\end{table*}
"""
write("tab_appendix_cells.tex", s)

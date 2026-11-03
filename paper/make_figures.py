#!/usr/bin/env python
"""Generate every figure in the Myna paper from committed JSON artifacts.

Run:  uv run --with matplotlib python paper/make_figures.py
Outputs: paper/figures/fig{2..7}_*.pdf and .png, plus paper/data/figure_values.json
"""
from __future__ import annotations
import json, math, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
C = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "grey": "#8C8C8C",
    "lightgrey": "#D9D9D9", "black": "#222222",
}

def load(p):
    with open(ROOT / p) as f:
        return json.load(f)

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)

def short(cell):
    return {
        "agnews/is_business": "agnews/bus", "agnews/is_scitech": "agnews/sci",
        "agnews/is_sports": "agnews/spo", "agnews/is_world": "agnews/wor",
        "agnews/topic": "agnews/topic", "amazon/stars": "amazon",
        "banking77/intent": "banking77", "boolq/answer": "boolq",
        "contrastive/decision": "contrastive", "dbpedia14/category": "dbpedia14",
        "imdb/positive": "imdb", "mnli/relation": "mnli",
        "sst5/sentiment": "sst5", "trec/answer_type": "trec",
        "yelp/rating": "yelp/rating", "yelp/recommend": "yelp/rec",
    }[cell]

CELLS = [
    "agnews/is_business", "agnews/is_scitech", "agnews/is_sports", "agnews/is_world",
    "agnews/topic", "amazon/stars", "banking77/intent", "boolq/answer",
    "contrastive/decision", "dbpedia14/category", "imdb/positive", "mnli/relation",
    "sst5/sentiment", "trec/answer_type", "yelp/rating", "yelp/recommend",
]
QNAME = {
    "agnews/is_business": "is_business", "agnews/is_scitech": "is_scitech",
    "agnews/is_sports": "is_sports", "agnews/is_world": "is_world",
    "agnews/topic": "topic", "amazon/stars": "stars", "banking77/intent": "intent",
    "boolq/answer": "answer", "contrastive/decision": "decision",
    "dbpedia14/category": "category", "imdb/positive": "positive",
    "mnli/relation": "relation", "sst5/sentiment": "sentiment",
    "trec/answer_type": "answer_type", "yelp/rating": "rating",
    "yelp/recommend": "recommend",
}

W6 = load("runs/witness-v6-master-paper.json")
WC = load("runs/witness-competitor-head-to-head-1440.json")
W5 = load("runs/witness-v5-master-paper.json")
WH = load("runs/witness-hybrid-v3.json")
SIG = load("runs/significance_v6.json")
TOK = load("paper/data/token_budget.json")
CONS = load("paper/data/consistency_all_engines.json")
VALS = {}

def fig2():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.2),
                                 gridspec_kw={"width_ratios": [1.05, 1.25], "wspace": 0.52})
    ks, tot, overflow = [], [], []
    for c in CELLS:
        t = TOK[QNAME[c]]
        k = t["k_max"] if c != "banking77/intent" else 77
        ks.append(k)
        tot.append(t["state_tokens_median"] + t["option_span_tokens_mean"])
        overflow.append(t["overflow_rows"])
    ax.scatter(ks, tot, s=28, c=C["blue"], edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(512, color=C["red"], lw=1.0, ls="--", zorder=1)
    ax.text(2.1, 512, " 512-token window", color=C["red"], va="bottom", fontsize=7)
    ax.annotate("77 options: 490 option tokens\n+ 25 context; 50/116 rows overflow",
                xy=(77, 515), xytext=(3.2, 300),
                arrowprops=dict(arrowstyle="->", color=C["black"], lw=0.6),
                fontsize=6.8, ha="left", va="center")
    ax.set_xscale("log")
    ax.set_xlim(1.8, 110)
    ax.set_xticks([2, 4, 8, 16, 32, 77])
    ax.get_xaxis().set_major_formatter(FixedFormatter(["2", "4", "8", "16", "32", "77"]))
    ax.set_xlabel("options per question, K")
    ax.set_ylabel("median state + option-span tokens")
    ax.set_ylim(0, 620)
    ax.grid(axis="y", color=C["lightgrey"], lw=0.5, zorder=0)
    ax.set_title("(a) the packed window is a budget", loc="left")

    order = sorted(CELLS, key=lambda c: -(TOK[QNAME[c]]["k_max"] if c != "banking77/intent" else 77))
    ys = np.arange(len(order))
    packed = [W5["test_16cell_audit"]["per_cell"][c] for c in order]
    unified = [WH["test"]["per_cell"][c] for c in order]
    for y, p, u in zip(ys, packed, unified):
        color = C["red"] if abs(u - p) > 1e-9 else C["lightgrey"]
        lw = 1.6 if abs(u - p) > 1e-9 else 0.8
        bx.plot([p, u], [y, y], color=color, lw=lw, zorder=2)
    bx.scatter(packed, ys, marker="o", s=16, facecolor="white", edgecolor=C["grey"], lw=0.8, zorder=3, label="v5 Packed")
    bx.scatter(unified, ys, marker="o", s=18, facecolor=C["blue"], edgecolor="white", lw=0.5, zorder=4, label="v5 Unified (routed)")
    bx.set_yticks(ys)
    bx.set_yticklabels([short(c) for c in order])
    bx.invert_yaxis()
    bx.set_xlim(0, 1.03)
    bx.set_xlabel("accuracy")
    bx.grid(axis="x", color=C["lightgrey"], lw=0.5, zorder=0)
    bx.legend(loc="lower right", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    bx.set_title("(b) routing moves exactly one cell", loc="left")
    bx.annotate("+0.5259 here\n= +0.0329 macro", xy=(0.54, 0.25), xytext=(0.58, 4.0),
                fontsize=6.8, color=C["red"],
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=0.6))
    bx.text(0.02, 15.4, "other 15 cells: identical", fontsize=6.4, color=C["grey"])
    save(fig, "fig2_cardinality")
    VALS["fig2"] = {"K": dict(zip(CELLS, ks)), "total_tokens": dict(zip(CELLS, tot)),
                    "overflow_rows": dict(zip(CELLS, overflow)),
                    "packed": dict(zip(order, packed)), "unified": dict(zip(order, unified))}

def fig3():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.0),
                                 gridspec_kw={"width_ratios": [1.0, 1.55], "wspace": 0.6})
    names = ["v5 Packed", "v5 Unified", "v6 test", "v6 local"]
    vals = [W5["test_16cell_audit"]["macro_acc"], WH["test"]["macro_acc"],
            W6["test_audit"]["macro_acc"], SIG["point"]["myna_macro"]]
    colors = [C["grey"], C["blue"], C["green"], C["orange"]]
    bars = ax.bar(range(4), vals, color=colors, width=0.62, edgecolor="white", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylim(0.79, 0.88)
    ax.set_ylabel("16-cell macro accuracy")
    ax.grid(axis="y", color=C["lightgrey"], lw=0.5, zorder=0)
    ax.set_title("(a) two steps, both one cell", loc="left")
    ax.text(0.5, 0.859, "+0.0329", fontsize=7, color=C["blue"], ha="center")
    ax.text(1.55, 0.869, "+0.0081", fontsize=7, color=C["green"], ha="center")

    order = sorted(CELLS, key=lambda c: -(TOK[QNAME[c]]["k_max"] if c != "banking77/intent" else 77))
    ys = np.arange(len(order))
    d1 = [WH["test"]["per_cell"][c] - W5["test_16cell_audit"]["per_cell"][c] for c in order]
    d2 = [W6["test_audit"]["per_cell"][c] - WH["test"]["per_cell"][c] for c in order]
    bx.barh(ys - 0.19, d1, height=0.34, color=C["blue"], label="routing (v5 Packed to Unified)")
    bx.barh(ys + 0.19, d2, height=0.34, color=C["green"], label="set-aware head (Unified to v6)")
    bx.axvline(0, color=C["black"], lw=0.6)
    bx.set_yticks(ys)
    bx.set_yticklabels([short(c) for c in order])
    bx.invert_yaxis()
    bx.set_xlim(-0.02, 0.56)
    bx.set_xlabel("accuracy change on the cell")
    bx.grid(axis="x", color=C["lightgrey"], lw=0.5, zorder=0)
    bx.legend(loc="center right", frameon=False, handletextpad=0.3, borderaxespad=0.2, fontsize=6.5)
    bx.set_title("(b) the gain is not broad", loc="left")
    save(fig, "fig3_ablation")
    VALS["fig3"] = {"macro": dict(zip(names, vals)), "delta_routing": dict(zip(order, d1)),
                    "delta_v6": dict(zip(order, d2))}

def fig4():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.9),
                                 gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.35})
    names = ["Laya\ndefault", "Laya\ntournament", "Laya\ntyped", "VegaML", "v5\nPacked", "v5\nUnified", "v6"]
    vals = [WC["Laya (English Default)"]["bundle_exact_acc"],
            WC["Laya (Raised + Tournament)"]["bundle_exact_acc"],
            WC["Laya (Typed-Decisions Best)"]["bundle_exact_acc"],
            WC["VegaML (vega-U 0.8B)"]["bundle_exact_acc"],
            WC["Myna v5 Pristine (Packed)"]["bundle_exact_acc"],
            WC["Myna v5 Unified (Card-Routed)"]["bundle_exact_acc"],
            W6["test_audit"]["bundle_exact_acc"]]
    colors = [C["grey"]] * 4 + [C["lightgrey"], C["blue"], C["green"]]
    vals = [v * 100 for v in vals]
    bars = ax.bar(range(7), vals, color=colors, width=0.66, edgecolor="white", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(7)); ax.set_xticklabels(names, rotation=18, ha="right", fontsize=6.5)
    ax.set_ylim(0, 95); ax.set_ylabel("joint bundle exact-match (%)")
    ax.grid(axis="y", color=C["lightgrey"], lw=0.5, zorder=0)
    ax.set_title("(a) all questions in a bundle correct", loc="left")

    labs = ["Myna v6", "Laya 0.4.19", "VegaML 0.15.0"]
    cnt = [CONS[labs[0]]["total"], CONS[labs[1]]["total"], CONS[labs[2]]["total"]]
    bars = bx.bar(range(3), cnt, color=[C["green"], C["grey"], C["grey"]], width=0.58,
                  edgecolor="white", linewidth=0.5)
    for b, v in zip(bars, cnt):
        bx.text(b.get_x() + b.get_width() / 2, v + 0.8, str(v), ha="center", va="bottom", fontsize=7)
    bx.set_xticks(range(3)); bx.set_xticklabels(["Myna v6", "Laya", "VegaML"], rotation=12, ha="right")
    bx.set_ylim(0, 36); bx.set_ylabel("contradictions / 172 bundles")
    bx.grid(axis="y", color=C["lightgrey"], lw=0.5, zorder=0)
    bx.set_title("(b) same rows, explicit predicate", loc="left")
    save(fig, "fig4_consistency")
    VALS["fig4"] = {"bundle_exact": dict(zip(names, vals)), "contradictions": dict(zip(labs, cnt))}

def reliability():
    d = load("runs/paper_myna_v6_test.rowoutcomes.json")
    qs = []
    for r in d["rows"]:
        for q in r["questions"].values():
            qs.append((float(q["conf"]), int(q["ok"])))
    qs.sort(key=lambda x: -x[0])
    bins = [[] for _ in range(10)]
    for c, ok in qs:
        bins[min(9, int(c * 10))].append((c, ok))
    xs, ys, ns = [], [], []
    ece = 0.0
    for b in bins:
        if not b:
            continue
        conf = sum(x[0] for x in b) / len(b)
        acc = sum(x[1] for x in b) / len(b)
        xs.append(conf); ys.append(acc); ns.append(len(b))
        ece += len(b) / len(qs) * abs(acc - conf)
    return xs, ys, ns, ece, len(qs)

def fig5():
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(7.4, 2.7),
                                     gridspec_kw={"width_ratios": [1.0, 1.55, 0.8], "wspace": 0.55})
    xs, ys, ns, ece, n = reliability()
    ax.plot([0, 1], [0, 1], color=C["grey"], lw=0.7, ls="--", zorder=1)
    ax.plot(xs, ys, color=C["blue"], lw=1.0, marker="o", ms=3.5, zorder=3)
    for x, y, k in zip(xs, ys, ns):
        ax.scatter([x], [y], s=8 + 40 * k / max(ns), facecolor=C["blue"], edgecolor="white", lw=0.4, zorder=4)
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.grid(color=C["lightgrey"], lw=0.5, zorder=0)
    ax.set_title("(a) reliability, 10 bins", loc="left")
    ax.text(0.04, 0.92, f"ECE = {ece:.4f}\nn = {n}", fontsize=7, va="top")

    curves = [
        ("calibration, 11 sources (568)", "runs/risk_coverage_v6.json", C["blue"]),
        ("test, G5 trio (348)", "runs/risk_coverage_v6_g5trio_test.json", C["green"]),
        ("dev, G5 trio (348)", "runs/risk_coverage_v6_g5trio_dev.json", C["orange"]),
        ("calibration, G5 trio (120)", "runs/risk_coverage_v6_g5trio.json", C["purple"]),
    ]
    for lab, path, col in curves:
        d = load(path)
        pts = sorted([(c["coverage"], c["accuracy"]) for c in d["curve"]])
        bx.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=1.0, marker="o", ms=2.5, label=lab)
        r = min(d["curve"], key=lambda c: abs(c["coverage"] - 0.6))
        bx.scatter([r["coverage"]], [r["accuracy"]], s=22, facecolor="white", edgecolor=col, lw=1.0, zorder=5)
    bx.axvline(0.6, color=C["red"], lw=0.7, ls=":")
    bx.axhline(0.95, color=C["red"], lw=0.7, ls=":")
    bx.set_xlim(0.08, 1.01); bx.set_ylim(0.82, 1.005)
    bx.set_xlabel("coverage"); bx.set_ylabel("accuracy")
    bx.grid(color=C["lightgrey"], lw=0.5, zorder=0)
    bx.legend(loc="lower left", frameon=False, handletextpad=0.3, borderaxespad=0.2, fontsize=5.9)
    bx.set_title("(b) risk-coverage", loc="left")

    nom = [90, 95]
    emp = [W6["calibration_test"]["conformal_empirical_90"] * 100,
           W6["calibration_test"]["conformal_empirical_95"] * 100]
    cx.plot([80, 100], [80, 100], color=C["grey"], lw=0.7, ls="--")
    cx.scatter(nom, emp, s=26, color=C["red"], zorder=3)
    for x, y in zip(nom, emp):
        cx.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(3, -9), fontsize=7, color=C["red"])
    cx.set_xlim(80, 100); cx.set_ylim(80, 100)
    cx.set_xlabel("nominal coverage (%)"); cx.set_ylabel("empirical (%)", labelpad=1)
    cx.grid(color=C["lightgrey"], lw=0.5, zorder=0)
    cx.set_title("(c) conformal", loc="left")
    save(fig, "fig5_calibration")
    VALS["fig5"] = {"ece10": ece, "reliability": {"conf": xs, "acc": ys, "n": ns},
                    "conformal": {"nominal": nom, "empirical": emp}}

def fig6():
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(7.4, 2.65),
                                     gridspec_kw={"width_ratios": [1.2, 1.0, 1.2], "wspace": 0.55})
    lat = load("runs/latency_matched.json")
    qs = [r["questions"] for r in lat["ratio"]]
    laya = [r["laya_system_one_p50_ms"] for r in lat["ratio"]]
    e2e = [r["myna_predict_p50_ms"] for r in lat["ratio"]]
    ask = [r["myna_ask_only_p50_ms"] for r in lat["ratio"]]
    ax.plot(qs, laya, color=C["grey"], marker="o", ms=3, lw=1.0, label="Laya system_one")
    ax.plot(qs, e2e, color=C["blue"], marker="o", ms=3, lw=1.0, label="Myna end-to-end")
    ax.plot(qs, ask, color=C["green"], marker="o", ms=3, lw=1.0, label="Myna, state cached")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(qs); ax.get_xaxis().set_major_formatter(FixedFormatter([str(q) for q in qs]))
    ax.set_xlabel("questions per request"); ax.set_ylabel("p50 latency (ms)")
    ax.grid(color=C["lightgrey"], lw=0.5, zorder=0, which="both")
    ax.legend(frameon=False, handletextpad=0.3, borderaxespad=0.2, loc="upper left")
    ax.set_title("(a) same box, same state", loc="left")

    e2e_s = [r["speedup_e2e"] for r in lat["ratio"]]
    ask_s = [r["speedup_stream"] for r in lat["ratio"]]
    x = np.arange(len(qs))
    bx.bar(x - 0.18, e2e_s, width=0.34, color=C["blue"], label="end-to-end")
    bx.bar(x + 0.18, ask_s, width=0.34, color=C["green"], label="cached state")
    for i, (a, b) in enumerate(zip(e2e_s, ask_s)):
        bx.text(i - 0.18, a + 0.08, f"{a:.1f}", ha="center", fontsize=6.2)
        bx.text(i + 0.18, b + 0.08, f"{b:.1f}", ha="center", fontsize=6.2)
    bx.set_xticks(x); bx.set_xticklabels([str(q) for q in qs])
    bx.set_xlabel("questions per request"); bx.set_ylabel("speedup (x)")
    bx.set_ylim(0, 7.6)
    bx.grid(axis="y", color=C["lightgrey"], lw=0.5, zorder=0)
    bx.legend(frameon=False, handletextpad=0.3, borderaxespad=0.2, loc="upper right")
    bx.set_title("(b) ratio", loc="left")

    mlx = load("runs/bench_mlx_int8.json")
    L = [r["state_tokens"] for r in mlx["latency"]]
    cx.plot(L, [r["torch_ms"] for r in mlx["latency"]], color=C["grey"], marker="o", ms=2.5, lw=1.0, label="PyTorch MPS")
    cx.plot(L, [r["mlx_fp32"] for r in mlx["latency"]], color=C["blue"], marker="o", ms=2.5, lw=1.0, label="MLX fp32")
    cx.plot(L, [r["mlx_int8"] for r in mlx["latency"]], color=C["orange"], marker="o", ms=2.5, lw=1.0, label="MLX int8")
    cx.set_xscale("log", base=2); cx.set_yscale("log")
    cx.set_xticks([128, 512, 2048, 8192, 16384])
    cx.get_xaxis().set_major_formatter(FixedFormatter(["128", "512", "2k", "8k", "16k"]))
    cx.set_xlabel("state tokens"); cx.set_ylabel("latency (ms)", labelpad=1)
    cx.grid(color=C["lightgrey"], lw=0.5, zorder=0, which="both")
    cx.legend(frameon=False, handletextpad=0.3, borderaxespad=0.2, loc="upper left")
    cx.set_title("(c) 1.76-2.22x on MLX", loc="left")
    save(fig, "fig6_systems")
    VALS["fig6"] = {"latency": {"questions": qs, "laya": laya, "e2e": e2e, "ask": ask,
                                "speedup_e2e": e2e_s, "speedup_ask": ask_s},
                    "mlx": {"state_tokens": L,
                            "torch": [r["torch_ms"] for r in mlx["latency"]],
                            "fp32": [r["mlx_fp32"] for r in mlx["latency"]],
                            "int8": [r["mlx_int8"] for r in mlx["latency"]]}}

def fig7():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.9),
                                 gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.35})
    labels = ["v0\nsynth", "v1b\nsynth", "v1b\nreal", "AP\noff", "AP\non", "v6\nreal"]
    vals = [load("runs/g6_v0_synthetic_s0.json")["macro_test"],
            load("runs/g6_v1b_synthetic_s0.json")["macro_test"],
            load("runs/v1b_kaggle_3600b.report.json")["macro"]["acc"],
            load("runs/antiprior_off_s0.report.json")["macro"]["acc"],
            load("runs/antiprior_on_s0c.report.json")["macro"]["acc"],
            W6["test_audit"]["macro_acc"]]
    cols = [C["grey"], C["grey"], C["blue"], C["orange"], C["red"], C["green"]]
    bars = ax.bar(range(6), vals, color=cols, width=0.66, edgecolor="white", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=6.4)
    ax.axhline(0.94, color=C["black"], lw=0.7, ls="--", label="G6 gate 0.94 (synthetic)")
    ax.axhline(0.4331, color=C["red"], lw=0.7, ls=":", label="majority floor 0.433")
    ax.set_xticks(range(6)); ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylim(0, 1.08); ax.set_ylabel("macro accuracy")
    ax.legend(loc="upper center", frameon=False, fontsize=6.2, handlelength=1.4, borderaxespad=0.2)
    ax.grid(axis="y", color=C["lightgrey"], lw=0.5, zorder=0)
    ax.set_title("(a) synthetic success, real-suite plateau", loc="left")
    ax.annotate("4 to 0 emitters, -0.055 macro", xy=(4, 0.50), xytext=(1.55, 0.73),
                fontsize=6.2, color=C["red"], ha="left",
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=0.5))

    d = load("runs/needle_lc1_stageC_final.json")
    L = [r["tokens"] for r in d["rows"]]
    A = [r["acc"] for r in d["rows"]]
    bx.plot(L, A, color=C["purple"], marker="o", ms=3, lw=1.0, label="lc1-4k stageC (1200 updates)")
    bx.axhline(0.167, color=C["grey"], lw=0.7, ls=":")
    bx.text(140, 0.11, "uniform floor 0.167", fontsize=6.2, color=C["grey"])
    bx.axhline(0.90, color=C["red"], lw=0.7, ls="--")
    bx.axhline(0.85, color=C["red"], lw=0.7, ls="--")
    bx.text(130, 0.905, "G4: 0.90 at 4k", fontsize=6.2, color=C["red"])
    bx.text(130, 0.855, "G4: 0.85 at 16k", fontsize=6.2, color=C["red"])
    bx.set_xscale("log", base=2)
    bx.set_xticks([128, 1024, 4096, 16384])
    bx.get_xaxis().set_major_formatter(FixedFormatter(["128", "1k", "4k", "16k"]))
    bx.set_ylim(0, 1.02); bx.set_xlabel("state tokens"); bx.set_ylabel("needle accuracy")
    bx.grid(color=C["lightgrey"], lw=0.5, zorder=0)
    bx.legend(frameon=False, loc="upper right", handletextpad=0.3, borderaxespad=0.2, fontsize=6.4)
    bx.set_title("(b) flat at chance, not decay", loc="left")
    bx.text(150, 0.03, "harness prints readable: false", fontsize=6.3, color=C["grey"])
    save(fig, "fig7_diagnostics")
    VALS["fig7"] = {"bars": dict(zip(labels, vals)),
                    "needle": {"tokens": L, "acc": A, "floor": 0.167}}

def main():
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("fig2", fig2), ("fig3", fig3), ("fig4", fig4),
                     ("fig5", fig5), ("fig6", fig6), ("fig7", fig7)]:
        if only in ("all", name):
            fn()
            print("wrote", name)
    with open(ROOT / "paper/data/figure_values.json", "w") as f:
        json.dump(VALS, f, indent=1)

if __name__ == "__main__":
    main()

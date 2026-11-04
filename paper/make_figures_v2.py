#!/usr/bin/env python
"""Figures for the corrected manuscript, generated only from committed artifacts.

    .venv/bin/python paper/make_figures_v2.py

Outputs paper/figures/fig2_cardinality.{pdf,png}, fig3_external.{pdf,png}, and
fig4_calibration.{pdf,png}. Figures 1, 5, 6, and 7 are unchanged campaign artifacts
(architecture schematic, systems, diagnostics, decision sequence) and are not regenerated
here.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedFormatter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "paper" / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "grey": "#8C8C8C", "lightgrey": "#D9D9D9", "black": "#222222"}


def load(p):
    return json.loads((REPO / p).read_text())


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


def fig2():
    budget = load("paper/data/token_budget.json")
    nums = load("runs/paper_numbers.json")["arms"]
    ks, total, labels = [], [], []
    for qname, rec in budget.items():
        k = rec["k_stats"][-1] if "k_stats" in rec else int(rec["k_max"])
        ks.append(k)
        total.append(rec["state_tokens_median"] + rec["option_span_tokens_mean"])
        labels.append(qname)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    ax.axhline(512, color=C["red"], lw=0.9, ls="--", label="512-token window")
    ax.scatter(ks, total, s=26, color=C["blue"], zorder=3)
    for k, t, lab in zip(ks, total, labels):
        if k > 40:
            ax.annotate("banking77\n(77-way)", (k, t), textcoords="offset points",
                        xytext=(-10, -26), ha="right", fontsize=6.5)
    ax.set_xlabel("options per question ($K$, max)")
    ax.set_ylabel("median state + mean option spans (tokens)")
    ax.set_title("(a) The packed window is a budget")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, 84)
    ax.set_ylim(0, 580)

    ax = axes[1]
    vals = [nums["myna_v6_packed_forced"]["cells"]["banking77/intent"]["acc"],
            nums["myna_v6"]["cells"]["banking77/intent"]["acc"]]
    bars = ax.bar([0, 1], vals, color=[C["grey"], C["blue"]], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.4f}", ha="center", fontsize=7)
    ax.set_xticks([0, 1], ["packed\neverywhere", "routed\n($K>16$ decoupled)"])
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("banking77 accuracy ($n=116$)")
    ax.set_title("(b) One cell moves; same weights")
    fig.tight_layout()
    save(fig, "fig2_cardinality")


def fig3():
    arms = load("runs/paper_numbers.json")["arms"]
    order = [("myna_v6", "Myna v6, routed", C["blue"]),
             ("myna_v6_packed_forced", "Myna v6, packed", C["lightgrey"]),
             ("tfidf", "TF-IDF + logreg", C["green"]),
             ("laya_best", "Laya (typed best)", C["orange"]),
             ("laya_raised", "Laya (raised)", C["orange"]),
             ("laya_english", "Laya (English)", C["orange"]),
             ("vegaml_0150", "VegaML 0.15.0", C["purple"]),
             ("majority", "Majority prior", C["grey"]),
             ("nli", "Zero-shot NLI", C["grey"])]
    comp = load("runs/witness-competitor-head-to-head-1440.json")
    extra = {
        "laya_best": (comp["Laya (Typed-Decisions Best)"]["macro_acc"],
                      comp["Laya (Typed-Decisions Best)"]["bundle_exact_acc"]),
        "laya_raised": (comp["Laya (Raised + Tournament)"]["macro_acc"],
                        comp["Laya (Raised + Tournament)"]["bundle_exact_acc"]),
        "laya_english": (comp["Laya (English Default)"]["macro_acc"],
                         comp["Laya (English Default)"]["bundle_exact_acc"]),
    }
    names, macro, bundle, colors = [], [], [], []
    for key, label, color in order:
        if key in extra:
            m, b = extra[key]
        else:
            m, b = arms[key]["macro"], arms[key]["bundle_exact"]
        names.append(label); macro.append(m); bundle.append(b); colors.append(color)
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.barh(y, macro, color=colors, height=0.62, label="macro")
    ax.scatter(bundle, y, marker="D", s=14, color=C["black"], zorder=3,
               label="bundle exact")
    for yy, m in zip(y, macro):
        ax.text(m + 0.008, yy, f"{m:.3f}", va="center", fontsize=6.5)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("score on the same 1,176 test bundles")
    ax.set_title("All arms, one frozen split: macro and joint bundle exact")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig3_external")


def reliability():
    nums = load("runs/paper_numbers.json")["arms"]["myna_v6"]["calibration"]
    return nums["bins"]


def risk_coverage():
    rows = load("runs/paper_myna_v6_test.rowoutcomes.json")["rows"]
    rec = [(q["conf"], q["ok"]) for r in rows for q in r["questions"].values()]
    rec.sort(key=lambda t: -t[0])
    ok = np.array([o for _, o in rec], dtype=float)
    cov = np.arange(1, len(ok) + 1) / len(ok)
    acc = np.cumsum(ok) / np.arange(1, len(ok) + 1)
    return cov, acc


def fig4():
    bins = reliability()
    cov, acc = risk_coverage()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=C["grey"], lw=0.8, ls="--")
    xs = [b["conf"] for b in bins]
    ys = [b["acc"] for b in bins]
    ns = np.array([b["n"] for b in bins], dtype=float)
    ax.plot(xs, ys, color=C["blue"], lw=1.0, marker="o", ms=3)
    ax.scatter(xs, ys, s=6 + 60 * ns / ns.max(), color=C["blue"], alpha=0.5, zorder=2)
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("(a) Reliability, 10 bins")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = axes[1]
    ax.plot(cov, acc, color=C["blue"], lw=1.0)
    ax.axhline(0.95, color=C["grey"], lw=0.8, ls=":")
    ax.axvline(0.60, color=C["grey"], lw=0.8, ls=":")
    ax.scatter([0.60], [acc[int(0.60 * len(acc)) - 1]], s=14, color=C["red"], zorder=3)
    ax.set_xlabel("coverage")
    ax.set_ylabel("accuracy on covered questions")
    ax.set_title("(b) Risk--coverage")
    ax.set_xlim(0, 1); ax.set_ylim(0.5, 1.0)

    ax = axes[2]
    nominal = [90, 95]
    empirical = [0.8368 * 100, 0.9333 * 100]
    ax.plot([82, 97], [82, 97], color=C["grey"], lw=0.8, ls="--")
    ax.scatter(nominal, empirical, color=C["red"], zorder=3)
    for x, yv in zip(nominal, empirical):
        ax.annotate(f"{yv:.1f}%", (x, yv), textcoords="offset points", xytext=(6, -10),
                    fontsize=6.5, color=C["red"])
    ax.set_xlabel("nominal coverage (%)")
    ax.set_ylabel("empirical coverage (%)")
    ax.set_title("(c) Conformal: target missed")
    ax.set_xlim(82, 97); ax.set_ylim(82, 97)
    fig.tight_layout()
    save(fig, "fig4_calibration")


def fig6_diagnostics():
    """The small-engine diagnostics, with the corrected v6 bar (0.7777, not 0.8583)."""
    labels = ["v0\nsynth", "v1b\nsynth", "v1b\nreal", "AP\noff", "AP\non", "v6\nreal"]
    vals = [load("runs/g6_v0_synthetic_s0.json")["macro_test"],
            load("runs/g6_v1b_synthetic_s0.json")["macro_test"],
            load("runs/v1b_kaggle_3600b.report.json")["macro"]["acc"],
            load("runs/antiprior_off_s0.report.json")["macro"]["acc"],
            load("runs/antiprior_on_s0c.report.json")["macro"]["acc"],
            load("runs/paper_numbers.json")["arms"]["myna_v6"]["macro"]]
    cols = [C["grey"], C["grey"], C["blue"], C["orange"], C["red"], C["green"]]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.9),
                                 gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.35})
    bars = ax.bar(range(6), vals, color=cols, width=0.66, edgecolor="white", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center",
                va="bottom", fontsize=6.4)
    ax.axhline(0.94, color=C["black"], lw=0.7, ls="--", label="G6 gate 0.94 (synthetic)")
    ax.axhline(0.4331, color=C["red"], lw=0.7, ls=":", label="majority floor 0.433")
    ax.set_xticks(range(6)); ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylim(0, 1.08); ax.set_ylabel("macro accuracy")
    ax.legend(loc="upper center", frameon=False, fontsize=6.2, handlelength=1.4,
              borderaxespad=0.2)
    ax.grid(axis="y", color=C["lightgrey"], lw=0.5, zorder=0)
    ax.set_title("(a) synthetic success, real-suite plateau", loc="left")
    ax.annotate("4 to 0 emitters, -0.055 macro", xy=(4, 0.50), xytext=(1.55, 0.73),
                fontsize=6.2, color=C["red"], ha="left",
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=0.5))

    d = load("runs/needle_lc1_stageC_final.json")
    L = [r["tokens"] for r in d["rows"]]
    A = [r["acc"] for r in d["rows"]]
    bx.plot(L, A, color=C["purple"], marker="o", ms=3, lw=1.0,
            label="lc1-4k stageC (1200 updates)")
    bx.axhline(0.167, color=C["grey"], lw=0.7, ls=":")
    bx.text(140, 0.11, "uniform floor 0.167", fontsize=6.2, color=C["grey"])
    bx.set_xscale("log", base=2)
    bx.set_xticks([128, 1024, 4096, 16384])
    bx.get_xaxis().set_major_formatter(FixedFormatter(["128", "1k", "4k", "16k"]))
    bx.set_ylim(0, 1.02); bx.set_xlabel("state tokens"); bx.set_ylabel("needle accuracy")
    bx.grid(color=C["lightgrey"], lw=0.5, zorder=0)
    bx.legend(frameon=False, loc="upper right", handletextpad=0.3, borderaxespad=0.2,
              fontsize=6.4)
    bx.set_title("(b) flat at chance, not decay", loc="left")
    bx.text(150, 0.03, "harness prints readable: false", fontsize=6.3, color=C["grey"])
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig6_diagnostics_journal.{ext}")
    plt.close(fig)


def main():
    fig2(); fig3(); fig4(); fig6_diagnostics()
    print("wrote fig2_cardinality, fig3_external, fig4_calibration, fig6_diagnostics_journal")


if __name__ == "__main__":
    main()

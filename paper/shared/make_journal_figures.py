#!/usr/bin/env python
"""Figures for the single-system IEEE/Elsevier paper (no external baselines)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
})
C = {"blue":"#0072B2","orange":"#E69F00","green":"#009E73","red":"#D55E00",
     "purple":"#CC79A7","grey":"#8C8C8C","lightgrey":"#D9D9D9","black":"#222222"}
def load(p): return json.load(open(ROOT/p))
def save(fig,name):
    for ext in ("pdf","png"): fig.savefig(FIG/f"{name}.{ext}")
    plt.close(fig)

W5=load("runs/witness-v5-master-paper.json")
WH=load("runs/witness-hybrid-v3.json")
W6=load("runs/witness-v6-master-paper.json")
SIG=load("runs/significance_v6.json")
CONS=load("paper/data/consistency_all_engines.json")

# Fig 3: our arms only
fig,(ax,bx)=plt.subplots(1,2,figsize=(7.0,2.7),gridspec_kw={"width_ratios":[1.35,1.0],"wspace":0.35})
labels=["v5 Packed","v5 Unified","v6 test","v6 local"]
vals=[100*W5["test_16cell_audit"]["bundle_exact_acc"],100*WH["test"]["bundle_exact_acc"],
      100*W6["test_audit"]["bundle_exact_acc"],100*(987/1176)]
cols=[C["grey"],C["blue"],C["green"],C["orange"]]
b=ax.bar(range(4),vals,color=cols,width=0.62,edgecolor="white",linewidth=0.5)
for i,v in enumerate(vals): ax.text(i,v+0.6,f"{v:.2f}",ha="center",va="bottom",fontsize=7)
ax.set_xticks(range(4)); ax.set_xticklabels(labels,rotation=18,ha="right")
ax.set_ylim(0,92); ax.set_ylabel("joint bundle exact-match (%)")
ax.grid(axis="y",color=C["lightgrey"],lw=0.5,zorder=0)
ax.set_title("(a) all questions in a bundle correct",loc="left")
# contradictions by source (v6 recomputed)
srcs=["agnews","yelp","other 9 sources"]
cnt=[CONS["Myna v6"]["agnews_contradictions"],CONS["Myna v6"]["yelp_contradictions"],0]
bx.bar(range(3),cnt,color=[C["red"],C["orange"],C["lightgrey"]],width=0.58,edgecolor="white",linewidth=0.5)
for i,v in enumerate(cnt): bx.text(i,v+0.06,str(v),ha="center",va="bottom",fontsize=7)
bx.set_xticks(range(3)); bx.set_xticklabels(srcs,rotation=12,ha="right")
bx.set_ylim(0,3.2); bx.set_ylabel("contradictions / 172 bundles")
bx.grid(axis="y",color=C["lightgrey"],lw=0.5,zorder=0)
bx.set_title("(b) v6, explicit predicate",loc="left")
bx.text(0.02,2.75,"3/172 total; checkpoint records 0/172",fontsize=6.3,color=C["grey"])
save(fig,"fig3_consistency_ours")

# Fig 5: systems, ours only
lat=load("runs/latency_matched.json")["ratio"]
mlx=load("runs/bench_mlx_int8.json")
fig,(ax,bx,cx)=plt.subplots(1,3,figsize=(7.2,2.6),
                            gridspec_kw={"width_ratios":[1.2,1.2,0.8],"wspace":0.55})
qs=[r["questions"] for r in lat]
ax.plot(qs,[r["myna_predict_p50_ms"] for r in lat],color=C["blue"],marker="o",ms=3,lw=1.0,label="end-to-end")
ax.plot(qs,[r["myna_ask_only_p50_ms"] for r in lat],color=C["green"],marker="o",ms=3,lw=1.0,label="state cached")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(qs); ax.get_xaxis().set_major_formatter(FixedFormatter([str(q) for q in qs]))
ax.set_xlabel("questions per request"); ax.set_ylabel("p50 latency (ms)")
ax.grid(color=C["lightgrey"],lw=0.5,zorder=0,which="both")
ax.legend(frameon=False,loc="upper left",handletextpad=0.3,borderaxespad=0.2)
ax.set_title("(a) one M5, one process",loc="left")

L=[r["state_tokens"] for r in mlx["latency"]]
bx.plot(L,[r["torch_ms"] for r in mlx["latency"]],color=C["grey"],marker="o",ms=2.5,lw=1.0,label="PyTorch MPS")
bx.plot(L,[r["mlx_fp32"] for r in mlx["latency"]],color=C["blue"],marker="o",ms=2.5,lw=1.0,label="MLX fp32")
bx.plot(L,[r["mlx_int8"] for r in mlx["latency"]],color=C["orange"],marker="o",ms=2.5,lw=1.0,label="MLX int8")
bx.set_xscale("log",base=2); bx.set_yscale("log")
bx.set_xticks([128,512,2048,8192,16384]); bx.get_xaxis().set_major_formatter(FixedFormatter(["128","512","2k","8k","16k"]))
bx.set_xlabel("state tokens"); bx.set_ylabel("latency (ms)")
bx.grid(color=C["lightgrey"],lw=0.5,zorder=0,which="both")
bx.legend(frameon=False,loc="upper left",handletextpad=0.3,borderaxespad=0.2,fontsize=6.2)
bx.set_title("(b) runtime port",loc="left")

sizes=mlx["sizes"]; fp=sizes["mlx_fp32"]["file_bytes"]/2**20; i8=sizes["mlx_int8"]["file_bytes"]/2**20
cx.bar([0,1],[fp,i8],color=[C["grey"],C["orange"]],width=0.55)
for i,v in enumerate([fp,i8]): cx.text(i,v+1.0,f"{v:.2f}",ha="center",va="bottom",fontsize=7)
cx.set_xticks([0,1]); cx.set_xticklabels(["fp32","int8"],rotation=0)
cx.set_ylim(0,65); cx.set_ylabel("artifact size (MiB)")
cx.grid(axis="y",color=C["lightgrey"],lw=0.5,zorder=0)
cx.set_title("(c) int8 is smaller",loc="left")
cx.text(0.55,42,"9.24e-3 drift\n1 argmax flip",fontsize=6.2,color=C["red"])
save(fig,"fig5_systems_ours")

# aliases for the journal body
import shutil
for src,dst in [("fig5_calibration.pdf","fig4_calibration.pdf"),("fig7_diagnostics.pdf","fig6_diagnostics.pdf"),
                ("figA1_sequence.png","fig7_sequence.png")]:
    shutil.copyfile(FIG/src, FIG/dst)
print("wrote journal figures")

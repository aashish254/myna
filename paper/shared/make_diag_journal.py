import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter
import numpy as np
ROOT=Path.cwd()
FIG=ROOT/"paper/figures"
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":8,
"axes.titlesize":9,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,
"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":300,"savefig.dpi":300,
"savefig.bbox":"tight","savefig.pad_inches":0.02,"pdf.fonttype":42,"ps.fonttype":42})
C={"blue":"#0072B2","orange":"#E69F00","green":"#009E73","red":"#D55E00","purple":"#CC79A7","grey":"#8C8C8C","lightgrey":"#D9D9D9","black":"#222222"}
def load(p): return json.load(open(ROOT/p))
fig,(ax,bx)=plt.subplots(1,2,figsize=(7.2,2.7),gridspec_kw={"width_ratios":[1.4,1.0],"wspace":0.35})
labels=["v0\nsynth","v1b\nsynth","v1b\nreal","AP\noff","AP\non","v6\nreal"]
vals=[load("runs/g6_v0_synthetic_s0.json")["macro_test"],load("runs/g6_v1b_synthetic_s0.json")["macro_test"],
      load("runs/v1b_kaggle_3600b.report.json")["macro"]["acc"],load("runs/antiprior_off_s0.report.json")["macro"]["acc"],
      load("runs/antiprior_on_s0c.report.json")["macro"]["acc"],load("runs/witness-v6-master-paper.json")["test_audit"]["macro_acc"]]
cols=[C["grey"],C["grey"],C["blue"],C["orange"],C["red"],C["green"]]
b=ax.bar(range(6),vals,color=cols,width=0.66,edgecolor="white",linewidth=0.5)
for i,v in enumerate(vals): ax.text(i,v+0.012,f"{v:.3f}",ha="center",va="bottom",fontsize=6.6)
ax.axhline(0.94,color=C["black"],lw=0.8,ls="--")
ax.axhline(0.4331,color=C["red"],lw=0.8,ls=":")
ax.set_xticks(range(6)); ax.set_xticklabels(labels,fontsize=6.8)
ax.set_ylim(0,1.08); ax.set_ylabel("macro accuracy")
ax.grid(axis="y",color=C["lightgrey"],lw=0.5,zorder=0)
ax.set_title("(a) synthetic success, real-suite plateau",loc="left")

d=load("runs/needle_lc1_stageC_final.json")
L=[r["tokens"] for r in d["rows"]]; A=[r["acc"] for r in d["rows"]]
bx.plot(L,A,color=C["purple"],marker="o",ms=3,lw=1.0)
bx.axhline(0.167,color=C["grey"],lw=0.8,ls=":")
bx.text(140,0.12,"uniform floor 0.167",fontsize=6.3,color=C["grey"])
bx.set_xscale("log",base=2); bx.set_xticks([128,1024,4096,16384])
bx.get_xaxis().set_major_formatter(FixedFormatter(["128","1k","4k","16k"]))
bx.set_ylim(0,0.32); bx.set_xlabel("state tokens"); bx.set_ylabel("needle accuracy")
bx.grid(color=C["lightgrey"],lw=0.5,zorder=0)
bx.set_title("(b) flat at chance, not decay",loc="left")
for ext in ("pdf","png"): fig.savefig(FIG/f"fig6_diagnostics_journal.{ext}")
print("wrote fig6_diagnostics_journal")

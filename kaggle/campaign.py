#!/usr/bin/env python3
"""Generate the Kaggle cells for the GPU lane, from the run that actually completed.

Why this file exists: the launch docs used to carry hand-typed cells, and every one
of them was wrong in the same two ways. They named `--corpus
/kaggle/input/decision-v2-pilot`, while datasets mount at
`/kaggle/input/datasets/<owner>/<slug>`; and they ran `python kaggle/run.py` from
`/kaggle/working`, which starts EMPTY — nothing unpacks the code dataset there. A
cell that cannot find its entrypoint does not fail loudly. It fails after a human
has picked a GPU, pressed Shift+Enter and waited.

So the cells come from here, and the shape is the one recorded in
`runs/v1b_kaggle_3600b.train.log`: a setup cell that witnesses the mount and stages
a writable copy of the repo, then one cell per experiment that runs `kaggle/run.py`
with `--free-gib` pinned. Line 122 of that log is the command the completed run
actually executed; the flags below are it.

The cost is measured too, and it is not what the first draft of these docs promised.
That 3600-step run printed `step 3599 ... 20225s` of wall clock on a T4: **5 h 37 m**,
5.62 s per update with evaluations included. The "1.5 hours" figure was a guess.
Priced at the measured rate, the four-cell CE-vs-EMD² ablation is ~22.4 GPU-hours and
one 10k-step run ~15.6 — against 30 h/week of free quota. That arithmetic is why
`campaign.py`'s default output carries the open lane and the ablation only on
request: `--score-loss` was measured at its local dose and did not resolve
(`runs/ordinal_ab.json`), so those four cells would spend a week of quota re-asking a
question this loop already answered with "not shown".

Usage:
    python kaggle/campaign.py --list
    python kaggle/campaign.py                    # setup cell + the open experiments
    python kaggle/campaign.py --include-dead     # print the ablation cells too
    python kaggle/campaign.py --experiment extended_ce_s1
    python kaggle/campaign.py --owner <account>  # to name --corpus in the mount path
    python kaggle/campaign.py --nb /tmp/kernel --owner <account>   # a pushable kernel

Nothing here talks to Kaggle, in either direction. It prints cells or writes a
directory `kaggle kernels push` would take; a human runs the push.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# 20225 s over 3600 updates, both read from runs/v1b_kaggle_3600b.train.log.
SECONDS_PER_STEP = 20225 / 3600
# 14806 MiB free of 14911 MiB, from the same log. The plan has to be told the number,
# because it measures free bytes before the model is resident and over-batches if it may.
FREE_GIB = 9.0
STAGED = "/kaggle/working/myna"

EXPERIMENTS = {
    "extended_ce_s0": {
        "steps": 10_000, "loss": "ce", "seed": 0, "open": True,
        "target": "G1: dev was flat from step 2250 of the 3600-step run, so what is on "
                  "the table is dose, not loss",
    },
    "extended_ce_s1": {
        "steps": 10_000, "loss": "ce", "seed": 1, "open": True,
        "target": "G1: the replicate. One 10k run is an anecdote about a seed",
    },
    "ablation_ce_s0": {
        "steps": 3600, "loss": "ce", "seed": 0, "open": False,
        "target": "P9 9a's arm, measured at local dose and not resolved "
                  "(runs/ordinal_ab.json). Read the docstring before spending",
    },
    "ablation_ce_s1": {"steps": 3600, "loss": "ce", "seed": 1, "open": False,
                      "target": "as above, control replicate"},
    "ablation_emd_s0": {"steps": 3600, "loss": "emd", "seed": 0, "open": False,
                        "target": "as above, treatment arm"},
    "ablation_emd_s1": {"steps": 3600, "loss": "emd", "seed": 1, "open": False,
                        "target": "as above, treatment replicate"},
}

# @STAGED@ and @INPUT@ are replaced rather than .format()ed: the cell is python
# source full of f-string braces of its own, and a templating pass over it would
# either crash or quietly eat one. @INPUT@ exists so a test can point the same code
# at a fake symlinked mount; on the box it is `/kaggle/input`.
MOUNT_CELL = '''# SETUP 1/2, once per kernel: witness the mount, then stage a writable copy of the repo.
import os, shutil

entries = []
for root, dirs, files in os.walk("@INPUT@", followlinks=True):
    for name in sorted(dirs) + sorted(files):
        entries.append(os.path.join(root, name))
print(f"@INPUT@: {len(entries)} entries", flush=True)
print("(followlinks=True on purpose: Kaggle mounts datasets as symlinks, and "
      "Path.glob('**') does not descend into a symlinked directory)", flush=True)

hits = sorted(l for l in entries if l.endswith("/kaggle/run.py"))
if not hits:
    raise SystemExit("no kaggle/run.py under the mount - see the entries above")
src_repo = os.path.dirname(os.path.dirname(hits[0]))
print("staged from:", src_repo, flush=True)
shutil.rmtree("@STAGED@", ignore_errors=True)   # never train onto a stale copy
shutil.copytree(src_repo, "@STAGED@",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))

corpora = sorted({os.path.dirname(l) for l in entries
                  if l.endswith("calibration.jsonl")})
print("corpora found:", corpora, flush=True)
if not corpora:
    raise SystemExit("no decision-v2-pilot corpus mounted: add it to the kernel inputs")
'''

ENV_CELL = '''# SETUP 2/2: the environment, then the runner's own pre-flight over the staged copy.
# Needs SETUP 1/2 in the same kernel: it leaves `corpora` in the notebook namespace.
import subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "-r", "@STAGED@/kaggle/requirements.txt"], check=True)
import torch
assert torch.cuda.is_available(), "no CUDA - this cell trains on a GPU, not a laptop"
print("device:", torch.cuda.get_device_name(0), flush=True)
free, total = torch.cuda.mem_get_info()
print(f"free {free >> 20} MiB of {total >> 20} MiB", flush=True)
# --check: interpreter, imports, device, corpus - all four before an hour of training.
subprocess.run([sys.executable, "@STAGED@/kaggle/run.py", "--name", "setup-check",
                "--corpus", corpora[0], "--check"], check=True)
'''


def hours(steps: int) -> float:
    return steps * SECONDS_PER_STEP / 3600


def experiment_cell(name: str, cfg: dict, corpus: str | None, owner: str | None,
                   free_gib: float) -> str:
    mount = corpus or (f"/kaggle/input/datasets/{owner}/decision-v2-pilot"
                       if owner else None)
    lines = [f"# {name} - {cfg['target']}",
             f"# cost at the measured {SECONDS_PER_STEP:.2f} s/update: "
             f"~{hours(cfg['steps']):.1f} GPU-hours on a T4",
             f"!python {STAGED}/kaggle/run.py \\",
             f"    --name {name}"]
    if mount:
        lines[-1] += " \\"
        lines.append(f"    --corpus {mount}")
    lines[-1] += " \\"
    lines.append(f"    --steps {cfg['steps']} \\\n"
                 f"    --score-loss {cfg['loss']} \\\n"
                 f"    --seed {cfg['seed']} \\\n"
                 f"    --stop-factor 3.0 \\\n"
                 f"    --save-every 250 \\\n"
                 f"    --free-gib {free_gib:g}")
    return "\n".join(lines)


def blocks(include_dead: bool, corpus: str | None, owner: str | None,
           free_gib: float, only: str | None = None) -> list[str]:
    """The lane as ordered notebook cells: guidance, the two setup cells, the cost
    notes, then one cell per experiment. `render` joins these with blank lines;
    `notebook` writes them as `nbformat` cells. One structure, two outputs, so a
    pasted cell and a pushed cell cannot disagree."""
    if only:
        if only not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {only!r}: "
                             f"{', '.join(sorted(EXPERIMENTS))}")
        picked = [(only, EXPERIMENTS[only])]
    else:
        picked = [(k, v) for k, v in EXPERIMENTS.items() if v["open"] or include_dead]
    total = sum(hours(cfg["steps"]) for _, cfg in picked)
    out = ["# Paste these two cells first, once, in a kernel with GPU T4 x2 and "
           "internet ON (pip needs it). In order: the second uses what the first "
           "leaves in the namespace.",
           MOUNT_CELL.replace("@STAGED@", STAGED).replace("@INPUT@", "/kaggle/input"),
           ENV_CELL.replace("@STAGED@", STAGED),
           f"# {len(picked)} experiment cell(s): {total:.1f} GPU-hours at the measured "
           "rate.\n# Free quota is 30 h/week. If the number above is closer to that "
           "than to\n# zero, use one kernel per cell: a quota stop kills the kernel "
           "outright, and a\n# sequential notebook loses everything after the cell that "
           "was cut.\n# --save-every 250 bounds that loss to 250 updates."]
    if not only:
        skipped = [k for k, v in EXPERIMENTS.items() if not v["open"]]
        if skipped and not include_dead:
            out.append("# Not here (measured, not resolved): " + ", ".join(skipped)
                       + ".\n# campaign.py --include-dead prints them.")
    if not owner and not corpus:
        out.append("# --corpus is absent, so run.py discovers the mounted corpus. That "
                   "works when\n# exactly one dataset with a calibration.jsonl is "
                   "mounted; name --owner for\n# the explicit path.")
    out += [experiment_cell(n, c, corpus, owner, free_gib) for n, c in picked]
    return out


def render(include_dead: bool, corpus: str | None, owner: str | None,
           free_gib: float, only: str | None = None) -> str:
    return "\n\n".join(blocks(include_dead, corpus, owner, free_gib, only)) + "\n"


def notebook(owner: str, include_dead: bool, free_gib: float, slug: str,
             corpus: str | None) -> tuple[dict, dict]:
    """The cells as a kernel, because pasting is how the wrong paths got in.

    `kaggle kernels push` takes a directory holding `kernel-metadata.json` and the
    code file, and the metadata names the two datasets to mount. Nothing here
    submits: writing these two files is local and free, and pushing is the user's
    call. `title` has to slugify to the id's second half or the CLI rejects the
    push, so both come from `slug` rather than from prose.
    """
    cells = blocks(include_dead, corpus, owner, free_gib)
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                     "outputs": [], "source": c + "\n"} for c in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 4}
    meta = {"id": f"{owner}/{slug}", "title": slug.replace("-", " "),
            "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True,
            # the setup cell pip-installs from requirements.txt, so a kernel with
            # internet OFF cannot get past cell 2. The old doc said OFF.
            "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [f"{owner}/myna-code", f"{owner}/decision-v2-pilot"],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    return nb, meta


def write_notebook(dest: Path, owner: str, **kw) -> int:
    nb, meta = notebook(owner, **kw)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / meta["code_file"]).write_text(json.dumps(nb, indent=1) + "\n")
    (dest / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return len(nb["cells"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="the design matrix and its status")
    ap.add_argument("--experiment", default=None, help="print one cell, not the lane")
    ap.add_argument("--include-dead", action="store_true",
                    help="also print the measured-dead ablation cells")
    ap.add_argument("--corpus", default=None, help="the mounted corpus directory; beats "
                    "--owner, and leaving both off lets run.py discover it")
    ap.add_argument("--owner", default=os.environ.get("KAGGLE_USERNAME"),
                    help="the Kaggle account the datasets live under, for the mount "
                         "path. Never guessed here (see kaggle/package_dataset.py)")
    ap.add_argument("--nb", default=None, metavar="DIR", help="write a pushable kernel "
                    "(`.ipynb` + `kernel-metadata.json`) into DIR instead of printing "
                    "cells; needs --owner, and submits nothing")
    ap.add_argument("--slug", default="myna-campaign", help="the kernel slug for --nb")
    ap.add_argument("--free-gib", type=float, default=FREE_GIB,
                    help=f"bytes to pin the memory plan at (default {FREE_GIB:g}: the "
                         "value the completed T4 run used)")
    args = ap.parse_args(argv)

    if args.list:
        for key, cfg in EXPERIMENTS.items():
            state = "open" if cfg["open"] else "measured, not resolved"
            print(f"{key:18s} {cfg['loss']:3s} seed {cfg['seed']} {cfg['steps']:6d} "
                  f"steps  ~{hours(cfg['steps']):5.1f} h  {state}")
            print(f"{'':18s} {cfg['target']}")
        return 0
    if args.nb:
        if not args.owner:
            raise SystemExit("--nb needs an owner: pass --owner <account> or set "
                             "KAGGLE_USERNAME. A kernel cannot name the datasets to "
                             "mount without one, and this file does not guess accounts.")
        if args.experiment:
            raise SystemExit("--nb writes the lane, not one cell: drop --experiment")
        n = write_notebook(Path(args.nb), args.owner, include_dead=args.include_dead,
                           free_gib=args.free_gib, slug=args.slug, corpus=args.corpus)
        print(f"{args.nb}/{args.slug}.ipynb: {n} cells, "
              f"mounts {args.owner}/myna-code + {args.owner}/decision-v2-pilot, private, "
              "GPU T4, internet on.\nNothing pushed: "
              f"`kaggle kernels push -p {args.nb}` is the user's call.")
        return 0
    sys.stdout.write(render(args.include_dead, args.corpus, args.owner, args.free_gib,
                            args.experiment))
    return 0


if __name__ == "__main__":
    sys.exit(main())

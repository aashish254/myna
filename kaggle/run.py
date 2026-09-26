"""One-command Kaggle entrypoint (SPEC §5 P3: 3b).

The MacBook does not train (user directive, 2026-09-26; SPEC §6): this file is
the whole of what runs on the GPU box, and it is deliberately a *thin*, locally
testable composition of `myna.train`'s own flags rather than a second trainer. A
second trainer is how a cloud run ends up measuring a different model than the
one every gate here was checked against.

What it pins, and why each is a rule rather than a taste:

* `EXPERIMENT_NAME` (env or `--name`) — no name, no run. Two jobs writing the same
  directory overwrite each other's checkpoint, and a checkpoint whose experiment
  is not named is not evidence.
* output under `/kaggle/working`, which is the only path Kaggle persists.
* `--seed` — recorded in the command line and in `run.json` next to the metrics.
* `--resume` when a snapshot already exists at that path, so re-running the
  notebook cell continues instead of restarting.
* the device is `auto`, which on a CUDA box picks CUDA and *fails loud* if CUDA is
  present but unreadable — the silent CPU fallback is §6's other lesson.

It then execs `python -m myna.train` as a subprocess (so the stop rule's non-zero
exit reaches the notebook) and tees the log to the run directory.

Usage on Kaggle (one cell, after uploading this repo as a kernel input):

    !pip install -r /kaggle/working/myna/kaggle/requirements.txt
    !python /kaggle/working/myna/kaggle/run.py --corpus /kaggle/input/decision-v2-pilot

Locally, to see the command without running it:

    python kaggle/run.py --corpus data/decision-v2-pilot --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKING = Path("/kaggle/working")
if str(REPO / "src") not in sys.path:  # so a test can import this file and call main()
    sys.path.insert(0, str(REPO / "src"))

SPLITS = ("train.jsonl", "development.jsonl", "test.jsonl", "calibration.jsonl")

# The configuration this loop has earned: the per-row batch (boolq/mnli need it),
# gradient accumulation across question sets (interference), pool weighting
# (measured: uniform starves the data-rich sets), the paraphrase table with the
# suite wording held out, and a memory plan + stop rule so a 10k-update job that
# goes wrong stops itself instead of being killed.
DEFAULTS = {
    "steps": 10_000,
    "batch": 32,  # an upper bound: the memory plan clamps it at startup
    "accum_groups": 8,
    "max_q_cells": 2048,
    "group_sample": "pool",
    "eval_every": 250,
    "save_every": 25,
    "mem_safety": 0.6,
    "stop_factor": 1.5,
    "seed": 0,
    "vocab": 8192,
}


def find_corpus(explicit: str | None) -> Path:
    """The pilot slice ships as a Kaggle dataset of the same four jsonl files the
    adapter reads here. Fail loud with what *is* mounted: a silently wrong corpus
    trains a model that scores itself on the wrong test set."""
    if explicit:
        p = Path(explicit)
        missing = [s for s in SPLITS if not (p / s).exists()]
        if missing:
            raise SystemExit(f"--corpus {p} is missing {', '.join(missing)}")
        return p
    roots = [WORKING]
    if Path("/kaggle/input").exists():
        roots += sorted(d for d in Path("/kaggle/input").iterdir() if d.is_dir())
    for root in roots:
        for cand in sorted(root.glob("**/train.jsonl"))[:50]:
            if all((cand.parent / s).exists() for s in SPLITS):
                return cand.parent
    raise SystemExit("no decision-v2-pilot corpus found: pass --corpus, or mount the "
                     f"dataset. Searched: {', '.join(str(r) for r in roots)}")


def run_dir(name: str, out_root: str | None) -> Path:
    """`out_root` is the directory that *holds* runs; the name is the one directory
    a run owns. Flattening the two is how concurrent jobs overwrite each other's
    checkpoint, and on Kaggle anything outside /kaggle/working is not persisted."""
    return (Path(out_root) if out_root else WORKING / "runs") / name


def build_command(corpus: Path, out: Path, args) -> list[str]:
    d = DEFAULTS
    cmd = [sys.executable, "-m", "myna.train",
           "--suite", str(corpus), "--out", str(out),
           "--device", args.device, "--steps", str(args.steps or d["steps"]),
           "--batch", str(args.batch or d["batch"]),
           "--accum-groups", str(d["accum_groups"]),
           "--max-q-cells", str(d["max_q_cells"]),
           "--group-sample", d["group_sample"],
           "--eval-every", str(d["eval_every"]),
           "--save-every", str(args.save_every if args.save_every is not None
                               else d["save_every"]),
           "--mem-safety", str(d["mem_safety"]),
           "--stop-factor", str(args.stop_factor if args.stop_factor is not None
                                else d["stop_factor"]),
           "--seed", str(args.seed if args.seed is not None else d["seed"]),
           "--vocab", str(args.vocab if args.vocab is not None else d["vocab"]),
           "--paraphrase", args.paraphrase, "--row-batch"]
    if args.min_train_pool:
        cmd += ["--min-train-pool", str(args.min_train_pool)]
    if (out / "model_last.pt").exists():
        cmd += ["--resume"]
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=None, help="dir holding train/development/test/"
                    "calibration.jsonl (the packaged decision-v2-pilot dataset)")
    ap.add_argument("--name", default=os.environ.get("EXPERIMENT_NAME"),
                    help="experiment name; $EXPERIMENT_NAME is read, and it is required")
    ap.add_argument("--out-root", default=None, help="the directory that holds runs "
                    "(default /kaggle/working/runs, the only path Kaggle persists)")
    ap.add_argument("--device", default="auto", help="auto picks CUDA on a Kaggle GPU box")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--vocab", type=int, default=None, help="tokenizer size; the box trains "
                    "8192, and a local smoke of this entrypoint needs a smaller one")
    ap.add_argument("--stop-factor", type=float, default=None, help="override the drift "
                    "tolerance the loop stops itself on; the default is what runs on Kaggle")
    ap.add_argument("--min-train-pool", type=int, default=0)
    ap.add_argument("--paraphrase", choices=["off", "on"], default="on")
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    ap.add_argument("--check", action="store_true", help="verify the interpreter, the "
                    "imports, the device and the corpus, then exit")
    args = ap.parse_args(argv)

    if not args.name:
        raise SystemExit("EXPERIMENT_NAME is not set: a run without a name has nowhere "
                         "to write that anyone could find again")
    corpus = find_corpus(args.corpus)
    out = run_dir(args.name, args.out_root)

    if args.check:
        import torch

        import myna.train  # the real import, not a syntax check (3c)

        print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__,
                          "cuda": torch.cuda.is_available(), "corpus": str(corpus),
                          "out": str(out), "myna": str(Path(myna.__file__).parent)},
                         indent=2))
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda on an interpreter without CUDA")
        return 0

    cmd = build_command(corpus, out, args)
    print(" ".join(shlex.quote(c) for c in cmd), flush=True)
    if args.dry_run:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(
        {"command": cmd, "corpus": str(corpus), "name": args.name, "defaults": DEFAULTS,
         "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "python": sys.version.split()[0]},
        indent=2) + "\n")
    log = open(out / "train.log", "a", buffering=1)
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    print(f"run: {args.name} -> {out} (log: {out / 'train.log'})", flush=True)
    # Popen, not run(capture_output=...): a 10k-update job has to stream, or the
    # notebook shows nothing for hours and gets stopped by a bored human
    with log:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = proc.wait()
    if code:
        # the trainer's own stop rule exits non-zero on purpose; say what it was
        raise SystemExit(f"training ended with code {code}; see {out / 'train.log'}")
    print(f"done: {out / 'model.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    sys.exit(main())

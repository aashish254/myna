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
  It now fixes the weights as well as the data order: `myna.train` seeds torch from
  it, which is what makes a two-cell ablation differ in one thing.
* `--score-loss` — `ce` unless stated, so a plain run reproduces every published
  figure. `emd` is one flag and one output directory away, which is what makes the
  P9 ablation a pair of runs rather than a re-run whose drift reads as an effect.
* `--resume` when a snapshot already exists at that path, so re-running the
  notebook cell continues instead of restarting — and a refusal when the snapshot
  has already reached the requested step, because resuming a spent cosine schedule
  trains at learning rate ~0. `--warm-start` is the flag for that case.
* `--free-gib` is passed through to the trainer's memory plan rather than measured
  by it. On a T4 the plan reads the free bytes *before* the model is resident, so an
  unpinned plan over-batches; the pinned value is a property of the box, which is
  why it stays opt-in here and unset by default.
* the device is `auto`, which on a CUDA box picks CUDA and *fails loud* if CUDA is
  present but unreadable — the silent CPU fallback is §6's other lesson.

It then execs `python -m myna.train` as a subprocess (so the stop rule's non-zero
exit reaches the notebook) and tees the log to the run directory.

Usage on Kaggle. Two facts the first version of this header got wrong: datasets mount
at `/kaggle/input/datasets/<owner>/<slug>`, and `/kaggle/working` starts EMPTY, so
`/kaggle/working/myna` does not exist until a cell creates it. Run off the mount:

    !pip install -r /kaggle/input/datasets/<owner>/myna-code/kaggle/requirements.txt
    !python /kaggle/input/datasets/<owner>/myna-code/kaggle/run.py \
        --corpus /kaggle/input/datasets/<owner>/decision-v2-pilot --name my-run

`--corpus` may be left off (see `_mount_glob`), but name it anyway: the two accounts
this repo has uploaded under make an unqualified guess a real risk. To keep a writable
copy of the repo, stage it first with `shutil.copytree(..., ignore=symlinks)` into
`/kaggle/working/myna` and use those paths instead.

Locally, to see the command without running it:

    python kaggle/run.py --corpus data/decision-v2-pilot --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKING = Path("/kaggle/working")
# A module constant, not a literal in find_corpus: the discovery branch is the one
# thing about this file that only ever runs on the box, and a test has to be able
# to hand it a mount.
INPUT = Path("/kaggle/input")
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
    # ce, not emd: every figure this repo has published was trained with cross-
    # entropy on `score` cells, so the runner's default has to reproduce them. The
    # flag exists so one notebook cell can be flipped to `emd` and the other left
    # alone — a paired GPU run, not a lone re-run whose drift reads as an effect.
    "score_loss": "ce",
    # Same shape as `score_loss`, for the other one-flag pair the repo can run:
    # `--anti-prior on` re-weights the mini-batch draw by the inverse label prior
    # (SPEC §5 P10). Off is what every published figure — 0.4893 included — was
    # drawn with, and `bench/anti_prior_audit.py` measures what on changes: the six
    # skewed cells' drawn marginals flatten, the task mix does not move.
    "anti_prior": "off",
}


def _mount_glob(root: Path) -> list[Path]:
    """Every `train.jsonl` under `root`, descending into symlinked directories.

    `Path.glob("**")` does not follow a symlinked dir — measured on 3.9, 3.11 and
    3.13, where `glob.glob(recursive=True)` returned 2 hits across the same fixture
    and pathlib returned 0. Kaggle mounts each dataset as a symlink under
    `/kaggle/input`, so the pathlib form of this search found the corpus locally and
    nothing on the box, which is why every run to date had to pass `--corpus` by hand.
    """
    return sorted(Path(p) for p in glob.glob(str(root / "**" / "train.jsonl"),
                                             recursive=True))


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
    if INPUT.exists():
        roots += sorted(d for d in INPUT.iterdir() if d.is_dir())
    for root in roots:
        for cand in _mount_glob(root)[:50]:
            if all((cand.parent / s).exists() for s in SPLITS):
                return cand.parent
    raise SystemExit("no decision-v2-pilot corpus found: pass --corpus, or mount the "
                     f"dataset. Searched: {', '.join(str(r) for r in roots)}")


def run_dir(name: str, out_root: str | None) -> Path:
    """`out_root` is the directory that *holds* runs; the name is the one directory
    a run owns. Flattening the two is how concurrent jobs overwrite each other's
    checkpoint, and on Kaggle anything outside /kaggle/working is not persisted."""
    return (Path(out_root) if out_root else WORKING / "runs") / name


def continuation(out: Path, steps: int) -> list[str]:
    """`--resume` only when there is a schedule left to continue.

    The snapshot is the same file either way, so the entrypoint cannot tell a
    killed job from a finished one by looking for it. It can read `metrics.json`,
    which the trainer writes last: a run that got to its requested step spent its
    cosine decay, and `--resume` restores that spent scheduler (its `T_max` and
    AdamW's `lr`), so the cell trains at learning rate ~0 and prints numbers that
    read as drift. SPEC §5 P9 9e measured exactly that. A finished directory is
    therefore a refusal that names the two things that are not a resume: a longer
    `--steps`, or `--warm-start` for weights only."""
    if not (out / "model_last.pt").exists():
        return []
    metrics = out / "metrics.json"
    if metrics.exists():
        try:
            reached = json.loads(metrics.read_text()).get("last_step")
        except (ValueError, OSError):
            reached = None  # unreadable witness: fall through to the resume rule
        if isinstance(reached, int) and reached >= steps:
            raise SystemExit(
                f"{out} already trained to step {reached} of {steps}: --resume would "
                "continue a spent schedule at lr ~0. Use --warm-start for weights "
                f"only, or raise --steps past {reached} to genuinely extend it.")
    return ["--resume"]


def build_command(corpus: Path, out: Path, args) -> list[str]:
    d = DEFAULTS
    steps = args.steps or d["steps"]
    cmd = [sys.executable, "-m", "myna.train",
           "--suite", str(corpus), "--out", str(out),
           "--device", args.device, "--steps", str(steps),
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
           "--score-loss", args.score_loss or d["score_loss"],
           "--anti-prior", args.anti_prior or d["anti_prior"],
           "--paraphrase", args.paraphrase, "--row-batch"]
    if args.min_train_pool:
        cmd += ["--min-train-pool", str(args.min_train_pool)]
    if args.free_gib is not None:
        cmd += ["--free-gib", str(args.free_gib)]
    if args.warm_start:
        # the trainer refuses the pair, and the pair is what a stale snapshot plus a
        # fresh flag would silently produce, so say it here with the paths in hand.
        if (out / "model_last.pt").exists():
            raise SystemExit(f"--warm-start with a snapshot at {out}: the runner would "
                             "have added --resume too. Name a new --name, or drop "
                             "--warm-start to continue that run.")
        cmd += ["--warm-start", str(args.warm_start)]
    else:
        cmd += continuation(out, steps)
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
    ap.add_argument("--free-gib", type=float, default=None, help="pin the bytes the "
                    "trainer's memory plan reads instead of measuring them; on a T4 the "
                    "plan reads free bytes before the model is resident, so an unpinned "
                    "plan over-batches (SPEC §6). Unset by default: the right number is a "
                    "property of the box, not of this entrypoint")
    ap.add_argument("--warm-start", default=None, metavar="MODEL_PT", help="train from an "
                    "existing checkpoint's WEIGHTS only, with a fresh optimizer and schedule "
                    "— what a paired ablation needs. `--resume` continues a killed run and "
                    "restores its spent scheduler instead; the two are not interchangeable")
    ap.add_argument("--paraphrase", choices=["off", "on"], default="on")
    ap.add_argument("--score-loss", choices=["ce", "emd"], default=None,
                    help="how a `score` cell is priced (SPEC §5 P9). Default from the runner "
                         "is ce, the loss every published figure used; emd is the ablation, "
                         "and run.json records which one ran")
    ap.add_argument("--anti-prior", choices=["off", "on"], default=None,
                    help="draw each mini-batch by the inverse label prior of the "
                         "(source, question) cells its rows answer (SPEC §5 P10). Default "
                         "off — the draw every published figure came from. The entrypoint "
                         "always passes --row-batch, which is what decision-v2 needs: a "
                         "shared-set drawer can only re-weight rows *inside* one set, and "
                         "the median set there holds a single row")
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

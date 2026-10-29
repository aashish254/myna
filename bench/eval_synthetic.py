"""G6's missing measurement: any checkpoint scored on the synthetic test split.

G6 reads "no regression on what already worked" and sets its level at **synthetic test
≥ 0.94**, but the only figure ever quoted for it came from `runs/train-v0.log`, which
prints no command, no seed and no row counts (SPEC §9.30) — so the gate could be read
against v0 forever and never against a v1 checkpoint. This is the half that was
missing: it loads a checkpoint, rebuilds the synthetic splits with the trainer's own
recipe and a documented seed, and scores the test split through
`myna.train.evaluate()` — the function that printed v0's nine `=== test ===` rows, so
the number below is on the same basis as the number in the artifact.

Read the two arms together, never alone. `runs/myna-v0` is the baseline the gate was
written against and a v1 arm is the subject; both are scored on one seed, so a
difference between them is a difference in weights and nothing else. What the v1
arms' numbers do NOT measure is decay of a trained capability: `v1b-kaggle-3600b` and
the anti-prior pair trained on the real `decision-v2` suite and never saw one synthetic
row, so their score here is zero-shot retention of the task family G6 names. That
distinction is the finding, and it is why the gate's threshold is reported against
both arms rather than against the unreproducible 0.9523.

    uv run python bench/eval_synthetic.py --ckpt runs/v1b-kaggle-3600b --device cpu \
        --seed 0 --n-train 4000 --n-eval 1200 --out runs/g6_v1b_synthetic.md

The split is a deterministic function of `--seed`, `--n-train` and `--n-eval` alone
(`myna.data.generate` consumes one `random.Random` in `WORKFLOWS` order, train then dev
then test), and `n_train` matters because it sets where in that stream `test` begins.
The harness hashes the test rows it is about to score and writes the digest into both
witnesses, so "these two arms saw identical rows" is checkable rather than asserted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from myna.data import WORKFLOWS, generate  # noqa: E402
from myna.model import MynaConfig, MynaModel  # noqa: E402
from myna.report import cell_stats  # noqa: E402
from myna.train import evaluate, macro_acc, split_report  # noqa: E402

#: G6's level, restated here so the witness carries the verdict it was read against.
#: `bench/gates.py`'s G6 claim is the authority; this is the copy the artifact shows.
G6_THRESHOLD = 0.94


def split_floors(test: dict) -> dict:
    """The two floors of the rows about to be scored, read off those rows.

    0.34 on this split means nothing without them: the nine cells are 2-, 3-, 4- and
    5-option questions, so `myna.train`'s own macro has a chance level of ~1/3 and a
    modal-label level above it. Same `cell_stats` the real-corpus report uses, so the
    floor here is the same quantity the G1 row quotes — no second definition of
    "chance" that could quietly disagree with the first.
    """
    cells = cell_stats(test)
    out = {}
    for (source, qname), c in cells.items():
        out[f"{source}/{qname}"] = {"n": c["n"], "majority": c["majority"],
                                    "uniform": c["uniform"]}
    keys = sorted(out)
    out_macro = {
        "majority": sum(out[k]["majority"] for k in keys) / len(keys),
        "uniform": sum(out[k]["uniform"] for k in keys) / len(keys),
    }
    return {"cells": out, "macro": out_macro, "n_cells": len(keys)}


def synthetic_splits(seed: int, n_train: int, n_eval: int) -> dict:
    """The trainer's own recipe (train.py:995-1006), with the ordering it depends on.

    `torch.manual_seed` first because the trainer seeds the global RNG before touching
    the corpus; `rng` then walks `WORKFLOWS` in literal order for train, dev and test,
    so the split a reader reproduces is this function's three arguments and nothing
    else — steps, batch and device never reach it.
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    data = {
        "train": {wf: generate(n_train, wf, rng, "train") for wf in WORKFLOWS},
        "dev": {wf: generate(n_eval, wf, rng, "dev") for wf in WORKFLOWS},
        "test": {wf: generate(n_eval, wf, rng, "test") for wf in WORKFLOWS},
    }
    return {split: {wf: (WORKFLOWS[wf][0], exs) for wf, exs in groups.items()}
            for split, groups in data.items()}


def split_fingerprint(test: dict) -> str:
    """sha256 over the exact rows being scored: state text, then the gold vector.

    A paired comparison across two checkpoints is only paired if the rows are the
    rows, and `--n-train` makes that non-obvious. Sorted cell keys, so the digest does
    not depend on dict order.
    """
    h = hashlib.sha256()
    for wf in sorted(test):
        questions, exs = test[wf]
        names = ",".join(q.name for q in questions)
        h.update(f"{wf}|{names}\n".encode())
        for ex in exs:
            h.update(ex.state.encode())
            h.update(b"\x00")
            h.update(",".join(str(int(g)) for g in ex.gold).encode())
            h.update(b"\n")
    return h.hexdigest()


def load_checkpoint(ckpt: Path, device: str):
    """Weights, config and tokenizer as a set — the checkpoint scores with its OWN vocab.

    Scoring a suite-trained checkpoint with a freshly trained synthetic vocab would
    change what is measured: the embedding rows mean nothing to a tokenizer they were
    never trained against. The guard below is what refuses that, and it is the same
    rule `bench/eval_scratch.py:82-85` applies to the control.
    """
    if not (ckpt / "model.pt").exists():
        raise SystemExit(f"--ckpt {ckpt} has no model.pt: a `model_last.pt` resume "
                         f"snapshot is a different artifact, carries the optimizer, and "
                         f"is loaded by different code")
    tok_path = ckpt / "tokenizer.json"
    if not tok_path.exists():
        raise SystemExit(f"--ckpt {ckpt} has no tokenizer.json beside the weights")
    ck = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=False)
    model = MynaModel(MynaConfig(**ck["cfg"]))
    model.load_state_dict(ck["state_dict"])
    tok = Tokenizer.from_file(str(tok_path))
    if model.cfg.vocab != tok.get_vocab_size():
        raise SystemExit(f"{ckpt} has {model.cfg.vocab} embeddings but {tok_path} is a "
                         f"{tok.get_vocab_size()}-entry vocab: a checkpoint must be "
                         f"scored with the tokenizer it was trained against")
    model.to(device)
    model.eval()
    return model, tok, ck


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="checkpoint dir: model.pt + tokenizer.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=4000,
                    help="rows per workflow drawn before dev/test; it sets where in the "
                         "rng stream the test split begins, so it is part of the identity "
                         "of the measurement")
    ap.add_argument("--n-eval", type=int, default=1200,
                    help="rows per workflow in the dev and test splits")
    ap.add_argument("--out", default="runs/g6_synthetic.md")
    argv = list(argv if argv is not None else sys.argv[1:])
    args = ap.parse_args(argv)
    cmd = shlex.join(["python", "bench/eval_synthetic.py", *argv])

    ckpt = Path(args.ckpt)
    out_json = Path(args.out).with_suffix(".json")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model, tok, ck = load_checkpoint(ckpt, args.device)
    data = synthetic_splits(args.seed, args.n_train, args.n_eval)
    fp = split_fingerprint(data["test"])
    floors = split_floors(data["test"])

    for line in ("$ " + cmd,
                 f"what is being scored: checkpoint {ckpt} (cfg.vocab {model.cfg.vocab}, "
                 f"tokenizer {ckpt / 'tokenizer.json'}), device `{args.device}`",
                 f"split recipe: seed {args.seed}, n_train {args.n_train}, "
                 f"n_eval {args.n_eval} per workflow over {len(WORKFLOWS)} workflows "
                 f"({', '.join(WORKFLOWS)})"):
        print(line, flush=True)
    for split in ("train", "dev", "test"):
        print(split_report(split, data[split]), flush=True)
    print(f"test split sha256: {fp}", flush=True)
    print(f"load average {'/'.join(f'{x:.2f}' for x in os.getloadavg())}", flush=True)

    dev_m = evaluate(model, tok, data["dev"], args.device, temperature=1.0)
    test_m = evaluate(model, tok, data["test"], args.device, temperature=1.0)
    dev_acc = {k: v for k, v in dev_m.items() if not k.endswith((":brier", ":ece"))}
    cells = {k: v for k, v in test_m.items() if not k.endswith((":brier", ":ece"))}
    macro_test, macro_dev = macro_acc(test_m), macro_acc(dev_m)
    met = macro_test >= G6_THRESHOLD

    print()
    print("=== test === (maj = modal-label floor, uni = chance, both on these same rows)")
    for key in sorted(cells):
        fl = floors["cells"][key]
        print(f"{key:34} {cells[key]:.4f}   maj {fl['majority']:.4f}  "
              f"uni {fl['uniform']:.4f}")
    print(f"\nMACRO over the {len(cells)} synthetic test cells: {macro_test:.4f} "
          f"(dev on the same recipe: {macro_dev:.4f})")
    print(f"MACRO floors on the same rows: majority "
          f"{floors['macro']['majority']:.4f} · uniform {floors['macro']['uniform']:.4f}")
    print(f"G6 (synthetic test ≥ {G6_THRESHOLD:.2f}): "
          f"{'MET' if met else 'NOT MET'} at {macro_test:.4f}", flush=True)

    body = [f"`{cmd}`\n",
            f"load average {'/'.join(f'{x:.2f}' for x in os.getloadavg())}, device "
            f"`{args.device}`, checkpoint `{ckpt}`, vocab {model.cfg.vocab}, "
            f"params {sum(p.numel() for p in model.parameters()):,}.",
            "",
            f"Split recipe: seed `{args.seed}`, `--n-train {args.n_train}`, "
            f"`--n-eval {args.n_eval}` per workflow, `myna.train`'s own train→dev→test "
            f"draw order over `WORKFLOWS` = {list(WORKFLOWS)}. "
            f"test split sha256 `{fp}`.",
            "",
            "```",
            "=== test === (maj = modal-label floor, uni = chance, both on these same rows)"]
    body += [f"{key:34} {cells[key]:.4f}   maj {floors['cells'][key]['majority']:.4f}  "
             f"uni {floors['cells'][key]['uniform']:.4f}" for key in sorted(cells)]
    body += ["```", "",
             f"MACRO over the {len(cells)} synthetic test cells: **{macro_test:.4f}** "
             f"(dev on the same recipe: {macro_dev:.4f}).",
             "",
             f"MACRO floors over the same {len(cells)} cells, read off these rows: "
             f"majority **{floors['macro']['majority']:.4f}**, uniform (chance) "
             f"**{floors['macro']['uniform']:.4f}**.",
             "",
             f"G6 asks for **≥ {G6_THRESHOLD:.2f}** on this split, so on these weights "
             f"the gate is **{'met' if met else 'NOT met'}** at "
             f"**{macro_test:.4f}**."]
    Path(args.out).write_text("\n".join(body) + "\n")
    out_json.write_text(json.dumps({
        "cmd": cmd, "load_avg": list(os.getloadavg()), "device": args.device,
        "ckpt": str(ckpt), "vocab": model.cfg.vocab, "cfg": ck["cfg"],
        "params": sum(p.numel() for p in model.parameters()),
        "seed": args.seed, "n_train": args.n_train, "n_eval": args.n_eval,
        "workflows": list(WORKFLOWS), "temperature": 1.0,
        "test_split_sha256": fp,
        "floors": floors,
        "cells": {k: cells[k] for k in sorted(cells)},
        "dev_cells": {k: dev_acc[k] for k in sorted(dev_acc)},
        "macro_test": macro_test, "macro_dev": macro_dev,
        "g6": {"threshold": G6_THRESHOLD, "measured": macro_test, "met": met},
    }, indent=2) + "\n")
    print(f"wrote {args.out} and {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

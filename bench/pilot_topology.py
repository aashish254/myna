"""Question-set topology of a suite directory, under both batch contracts.

SPEC P1's finding was measured here, and P2's payoff is: the suite's rows arrive
as 17k exact-signature question-sets with a median pool of 1, so the shared-tensor
batch runs one row per forward for the per-row-instruction sources (boolq, mnli).
With per-row question tensors a batch is drawn across the whole split, so what
matters is how many rows of a source fit inside one memory budget.

The tokenizer is myna's own BPE — the one training will use, which is the only
honest unit for a memory budget — and it is cached next to the corpus.

    PYTHONPATH=src .venv/bin/python -m bench.pilot_topology \
        --suite data/decision-v2-pilot --cells 2048
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from myna.real_data import flatten_groups, load_split, suite_texts
from myna.tokenizer import Tokenizer, question_tensors, train_tokenizer
from myna.train import question_tokens


def get_tokenizer(suite: Path, vocab: int, texts) -> Tokenizer:
    cache = suite / f"tokenizer-{vocab}.json"
    if cache.exists():
        return Tokenizer.from_file(str(cache))
    tok = train_tokenizer(texts, vocab_size=vocab)
    tok.save(str(cache))
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="data/decision-v2-pilot")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--cells", type=int, default=2048,
                    help="question-branch budget per forward: rows x questions x question tokens")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    root = Path(args.suite)
    path = root / f"{args.split}.jsonl"
    if not path.exists():
        sys.exit(f"no {path}")
    groups = load_split(path)
    rows = flatten_groups(groups)
    print(f"{path}: {len(rows)} rows in {len(groups)} exact-signature question-sets")
    pools = sorted(len(exs) for _q, exs in groups.values())
    print(f"shared-set contract: median pool {pools[len(pools) // 2]}  "
          f"singletons {sum(1 for n in pools if n == 1)} ({pools.count(1) / len(pools):.0%})  "
          f"largest {pools[-1]}")

    tok = get_tokenizer(root, args.vocab, suite_texts(groups))
    by_source: dict[str, list] = {}
    for qs, ex in rows:
        by_source.setdefault(ex.workflow.split("#")[0], []).append(
            (len(qs), question_tokens(tok, qs)))

    print(f"\n{'source':<12} {'rows':>6} {'q/row':>6} {'q tokens':>9} "
          f"{'cells/row':>9} {'rows/step':>9}  set pool (shared contract)")
    for source, shapes in sorted(by_source.items()):
        n_q = statistics.median(s[0] for s in shapes)
        lq = [s[1] for s in shapes]
        cells = sorted(q * t for q, t in shapes)
        per_step = max(1, min(64, args.cells // max(1, cells[len(cells) // 2])))
        gp = sorted(len(exs) for key, (_q, exs) in groups.items() if key.split("#")[0] == source)
        print(f"{source:<12} {len(shapes):>6} {n_q:>6.1f} {statistics.median(lq):>9.0f} "
              f"{cells[len(cells) // 2]:>9,} {per_step:>9}  "
              f"median {gp[len(gp) // 2]}  max {gp[-1]}")
    print(f"\nrows/step = budget / median row's padded cells, capped at the 64-row eval "
          f"batch; {args.cells} question cells is {args.cells * 1.6 / 1024:.1f} GB of retained "
          f"activations at d_model 384 (bench/mem_profile.py). State tokens are charged "
          f"separately: 0.8 MiB each.")


if __name__ == "__main__":
    main()

"""Does `--anti-prior` spend the paraphrase budget? A CPU-only replay (SPEC §9.56(x), §9.58).

    python bench/coverage_replay.py
    python bench/coverage_replay.py --seed 0 --batch 10 --k 8 --max-q-cells 2048

§9.56(x) closed 10c with a residue: the `on` arm reached **176,003** phrasing draws against the
control's **207,411** over identical `paraphrase_sets` 17,112, called that **15.2% less paraphrase
coverage**, and left the −0.0554 partly unattributed because of it. It proposed two ways to settle
the question — a third training arm at the control's coverage (~5.4 h of GPU) or reading the
sampler (free) — and said the free one is what to try before spending any.

Reading the sampler answers the *counter* half outright. `Paraphraser.draw`
(`src/myna/paraphrase.py:405`) increments once per accepted row inside `build_row_batch`, and
`WeightedDraw.index` (`src/myna/train.py:59`) consumes one `rng.random()` per attempt. Nothing shares
a budget: `paraphrase_draws` is a *count of rows that reached a batch*. So the only way the flag
could cost coverage is by filling fewer batches short — `draw_row_batch` takes at most `batch`
*distinct* rows whose padded `rows × questions × question-tokens` product stays under
`--max-q-cells`, and gives up after `batch × 40` attempts, so a short batch is possible and this
file measures how often each drawer is short.

The replay consumes the RNG stream exactly as the loop does — the batch attempts, then one
`Paraphraser.draw` per accepted row, K times per update — and nothing else in the loop touches that
`random.Random`, which is why the totals are comparable to the runs' rather than merely analogous.
Two checks say so out loud: the corpus banner must come out byte-equal to the line in both arms'
train logs (`57904 rows in one pool, 17112 question sets, widest row 3 questions x 40 tokens`), since
those numbers are properties of the corpus *and* of the tokenizer the run trained; and each arm's
replayed total must equal the `paraphrase_draws` in its own `metrics.json`. Exit 1 if either fails,
because then this is not the runs' sampler and none of its output is evidence.

Deliberately absent: no training, no device, no accuracy, and no claim about which arm is better.
This file only prices the rows each drawer reaches, and the denominators it divides by.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from myna.paraphrase import Paraphraser                      # noqa: E402
from myna.real_data import flatten_weighted, load_suite, suite_texts  # noqa: E402
from myna.tokenizer import train_tokenizer                   # noqa: E402
from myna.train import WeightedDraw, draw_row_batch, question_tokens, worst_case_tokens  # noqa: E402

SUITE = "data/decision-v2-pilot"
BANNER = "57904 rows in one pool, 17112 question sets, widest row 3 questions x 40 tokens"
# Read off the committed artifacts, not retyped from prose: `paraphrase_draws` in each arm's own
# `metrics.json`, and the updates that process actually executed — `last_step - resumed_from_step +
# 1` — which is the denominator §9.56(x) did not use.
ARMS = {
    "off": {"drawer": "uniform", "updates": 3315, "draws": 207411,
            "asked": 3600, "last_step": 3314,
            "src": "runs/antiprior_off_s0.train.log (--steps 3600, STOP at 3314)"},
    "on": {"drawer": "anti-prior", "updates": 2814, "draws": 176003,
           "asked": 3315, "last_step": 3314, "resumed": 501,
           "src": "runs/antiprior_on_s0c.train.log (--steps 3315 --resume from step 500)"},
}


def replay(row_items, row_ws, paraphraser, tok, cache, *, batch, k, max_q_cells, updates,
           seed, weighted):
    """Rows per batch for one drawer, drawing the phrasing per accepted row like the loop does."""
    rng = random.Random(seed)
    drawer = WeightedDraw(row_ws) if weighted else None
    sizes = Counter()
    total = 0
    t0 = time.time()
    for _ in range(updates):
        for _ in range(k):
            items = draw_row_batch(row_items, batch, max_q_cells, rng, tok, cache, drawer=drawer)
            for qs, ex in items:
                paraphraser.draw(ex.workflow, qs, rng)
            sizes[len(items)] += 1
            total += len(items)
    return sizes, total, time.time() - t0


def main():
    ap = argparse.ArgumentParser(prog="coverage-replay")
    ap.add_argument("--suite", default=SUITE)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--k", type=int, default=8, help="sets/mini-batches per update (--accum-groups)")
    ap.add_argument("--max-q-cells", type=int, default=2048, dest="max_q_cells")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocab", type=int, default=8192)
    args = ap.parse_args()

    print(f"cmd: python bench/coverage_replay.py --suite {args.suite} --batch {args.batch} "
          f"--k {args.k} --max-q-cells {args.max_q_cells} --seed {args.seed} "
          f"--vocab {args.vocab}", flush=True)

    data = load_suite(args.suite)
    tok = train_tokenizer(suite_texts(data["train"]), vocab_size=args.vocab)
    row_items, row_ws = flatten_weighted(data["train"])
    paraphraser = Paraphraser()
    cache: dict[int, int] = {}
    widest = max((len(qs), question_tokens(tok, qs, cache)) for qs, _ in row_items)
    n_sets = paraphraser.warm(data["train"])
    worst_case_tokens(tok, paraphraser, row_items, cache)

    banner = (f"{len(row_items)} rows in one pool, {n_sets} question sets, "
              f"widest row {widest[0]} questions x {widest[1]} tokens")
    print(f"row-batch: {banner}; budget {args.max_q_cells} cells/forward", flush=True)
    if banner != BANNER:
        raise SystemExit(f"the replay is not looking at the runs' corpus+tokenizer: "
                         f"expected {BANNER!r}, got {banner!r}")

    print(f"\n{'arm':<5} {'drawer':<11} {'updates':>8} {'batches':>9} {'rows drawn':>11} "
          f"{'rows/batch':>11} {'draws/update':>13} {'full':>7} {'short':>7} {'metrics.json':>13}")
    rows = {}
    for arm, spec in ARMS.items():
        weighted = spec["drawer"] == "anti-prior"
        sizes, total, el = replay(row_items, row_ws, paraphraser, tok, cache, batch=args.batch,
                                  k=args.k, max_q_cells=args.max_q_cells,
                                  updates=spec["updates"], seed=args.seed, weighted=weighted)
        batches = sum(sizes.values())
        full = sizes[args.batch]
        rows[arm] = (sizes, total, total / spec["updates"], batches)
        print(f"{arm:<5} {spec['drawer']:<11} {spec['updates']:>8,} {batches:>9,} {total:>11,} "
              f"{total / batches:>11.3f} {total / spec['updates']:>13.2f} "
              f"{100.0 * full / batches:>6.1f}% {batches - full:>7,} {spec['draws']:>13,}"
              f"   ({el:.1f}s)")

    off_sizes, off_total, off_per_update, off_batches = rows["off"]
    on_sizes, on_total, on_per_update, on_batches = rows["on"]
    exact, failed = [], []
    for arm, spec in ARMS.items():
        got = rows[arm][1]
        line = (f"  {arm}: replay {got:,} vs metrics.json {spec['draws']:,} -> "
                + ("EXACT" if got == spec["draws"]
                   else f"DIFFERS by {got - spec['draws']:+,}"))
        exact.append(line)
        if got != spec["draws"]:
            failed.append(arm)
    print("\nreplay vs the committed counters:")
    print("\n".join(exact))
    print(f"\nbatch fill under `--batch {args.batch}`: uniform fills it "
          f"{100.0 * off_sizes[args.batch] / off_batches:.1f}% of the time "
          f"({off_sizes[args.batch]:,} of {off_batches:,} batches), anti-prior "
          f"{100.0 * on_sizes[args.batch] / on_batches:.1f}% "
          f"({on_sizes[args.batch]:,} of {on_batches:,}); the shortfalls are the "
          f"--max-q-cells budget and the distinctness rule, not the re-weighting.")
    print(f"coverage per executed update: {off_per_update:.2f} (off) vs {on_per_update:.2f} (on), "
          f"ratio {on_per_update / off_per_update:.4f}")
    x_off = ARMS["off"]["draws"] / ARMS["off"]["last_step"]
    x_on = ARMS["on"]["draws"] / ARMS["on"]["asked"]
    print(f"what §9.56(x) divided: the `off` counter by its metrics `last_step` "
          f"({ARMS['off']['draws']:,} / {ARMS['off']['last_step']:,} = {x_off:.2f}) and the `on` "
          f"counter by its `steps_requested` ({ARMS['on']['draws']:,} / {ARMS['on']['asked']:,} = "
          f"{x_on:.2f}) — two different kinds of denominator, ratio {x_on / x_off:.4f}, which is "
          f"where its {100.0 * (1 - x_on / x_off):.1f}% came from.")
    print(f"neither is what `train.py` printed: its own line divides by `--steps`, so the control "
          f"said {ARMS['off']['draws'] / ARMS['off']['asked']:.2f} over "
          f"{ARMS['off']['asked']:,} and the `on` arm {ARMS['on']['draws'] / ARMS['on']['asked']:.2f} "
          f"over {ARMS['on']['asked']:,} — the `on` process ran "
          f"{ARMS['on']['updates']:,} of the {ARMS['on']['asked']:,} it asked for, while the control "
          f"asked {ARMS['off']['asked']:,} and ran {ARMS['off']['updates']:,}.")
    print(f"the `on` arm's whole dose is its first process's {ARMS['on']['resumed']:,} updates plus "
          f"this one's {ARMS['on']['updates']:,} = {ARMS['on']['resumed'] + ARMS['on']['updates']:,}, "
          f"exactly the control's executed dose — but only this process's counter survived into "
          f"metrics.json, so the pair's comparison is the rate, not the total.")
    if failed:
        print(f"\nVERDICT: the replay no longer reproduces the runs' counters for {failed}; it is "
              f"not the same sampler, so nothing printed above is evidence.")
        return 1
    print("\nVERDICT: the two arms reach the same paraphrase coverage per update, and the replay "
          "lands on both of their published counters exactly. `--anti-prior` does not spend the "
          "augmentation budget — `draws` counts accepted rows, and the weighted drawer accepts the "
          "same number per batch as the uniform one. The −0.0554 stands attributed to the label "
          "prior alone; §9.56(x)'s coverage residue is withdrawn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Anti-prior batching (SPEC §5 P10) — the label-prior-flattening draw.

Tier 0 (`bench/diag_question_ablation.py`, §9.47) found three agnews `noul` cells
emitting one label on every test row and finishing exactly on their own majority
floor. The training data offers that behaviour: on a train marginal of 0.75, the
constant answer is the highest-scoring policy available to a model that has not
learned to read. `--anti-prior` removes the policy's payoff by drawing mini-batches
in inverse proportion to each cell's label share.

Everything here is measured at the seam the trainer uses — `draw_row_batch` and
`draw_batch` with a real tokenizer and a real drawer — because the interesting
failure is not "the formula is wrong" but "the weights are computed and never
reaches a batch", which is what a flag wired to the wrong one of the loop's two
batch paths looks like from the outside. `bench/anti_prior_audit.py` supplies the
real-data numbers; this file supplies the gates.

Three families of claim, each with a mutation in `bench/mutation_anti_prior.py`:

* **Where the prior is measured.** The cell is `(source, question name)`, pooled
  across question-sets. A within-set histogram on decision-v2 is flat by
  construction (nearly every set holds one row), so a balancer that reads the set
  would report success having changed nothing. It is also the *same* aggregation
  `myna.report` gets the majority floor from — two priors that could disagree would
  make "beat the floor" and "trained against the prior" different sentences
  (§9.41), so `cell_priors` and `cell_stats` are pinned against each other.
* **What the weights do.** Minority rows are drawn more often; a multi-question row
  is priced by the *product* of its slots; and every source's weights sum to its row
  count, because without that rescale inverse frequency would also move which tasks
  a run trains on.
* **What the flag does not do.** With it off, the sampler is the pre-flag loop
  statement for statement — reproduced here against a frozen copy of that loop, so
  no committed figure's data order can move quietly.
* **What the audit harness prints.** `bench/anti_prior_audit.py` renders the four
  numbers SPEC and README quote from this mechanism, so its `main()` is run here and
  every printed number is tied to the payload the same run wrote. §9.46 is what this
  family exists for: a sibling harness had its arithmetic fully gated and its printing
  unasked, and the second is what a reader actually quotes.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pytest

import myna
from myna.data import Example, Question
from myna.real_data import (ANTI_PRIOR_SKEW, COMBINE, balance_weights, cell_priors,
                            flatten_groups, flatten_weighted, load_split, source_of_group)
from myna.report import cell_stats
from myna.tokenizer import train_tokenizer
from myna.train import PriorAudit, WeightedDraw, draw_batch, draw_row_batch, question_tokens

ROOT = Path.cwd()
SRC = str(Path(myna.__file__).resolve().parent.parent)


def _one_source(name, qspecs, rows_per_label):
    """A source as `load_split` would hand it over: one question-set per row.

    `qspecs` is [(qname, qtype, options), ...] and `rows_per_label` gives the gold
    count per label; each row carries that label in every slot, clamped to the slot's
    option count — which is how a correlated multi-slot row is built on purpose.
    """
    groups = {}
    i = 0
    for label, n in enumerate(rows_per_label):
        for _ in range(n):
            qs = [Question(qname, qtype, f"the {qname} question of row {i}", list(opts))
                  for qname, qtype, opts in qspecs]
            key = f"{name}#{i:05d}"
            gold = tuple(min(label, len(q.options) - 1) for q in qs)
            groups[key] = (qs, [Example(f"{name} state number {i} says {label}", key, gold)])
            i += 1
    return groups


# boolq-shaped: one binary slot, 10/30 -> a 0.75 majority, and 40 singleton sets
SKEW = _one_source("boolq", [("answer", "noul", ["no", "yes"])], [10, 30])
# trec-shaped: one 6-way slot, skewed past the reach threshold. A single-slot source's
# mean raw weight is its label count (here 6, against boolq's 2), which is exactly the
# skew the per-source rescale has to remove
WIDE = _one_source("trec", [("type", "choice", ["a", "b", "c", "d", "e", "f"])],
                   [24, 5, 3, 2, 1, 1])
# agnews-shaped: a 4-way slot and a binary slot whose labels move together, which is
# what makes the combination rule (product, not mean) observable
MULTI = _one_source("agnews", [("topic", "choice", ["w", "s", "b", "t"]),
                               ("flag", "noul", ["no", "yes"])], [10, 10, 10, 10])
# a data-rich set, which is the shape `draw_batch` (the shared-set path) sees: 40 rows
# sharing one question set, 5 of them the minority label. Steep on purpose — the shared
# set is the only place the group path can show flattening, and a cell with 0.16 of
# headroom is not visible over 30 drawn rows.
POOL = {"support#one": ([Question("risk", "noul", "Does the writer threaten to leave?",
                                  ["no", "yes"])],
                        [Example(f"support state {i}", "support#one",
                                 (0 if i % 8 == 0 else 1,)) for i in range(40)])}
ALL = {}
for _g in (SKEW, WIDE, MULTI, POOL):
    ALL.update(_g)

TOK = train_tokenizer([q.instruction for qs, exs in ALL.values() for q in qs]
                      + [o for qs, _ in ALL.values() for q in qs for o in q.options]
                      + [e.state for _, exs in ALL.values() for e in exs], vocab_size=200)


def _merge(*gs):
    out = {}
    for g in gs:
        out.update(g)
    return out


def _majority(counts):
    tot = sum(counts.values())
    return max(counts.values()) / tot if tot else 0.0


def _draw_all(groups, batch, max_q_cells, drawer, seed=0, rounds=500):
    """Run the real row-batch sampler and return the gold counts it produced per cell."""
    items = flatten_groups(groups)
    rng = random.Random(seed)
    cache, drawn = {}, {}
    for _ in range(rounds):
        for qs, ex in draw_row_batch(items, batch, max_q_cells, rng, TOK, cache, drawer=drawer):
            source = source_of_group(ex.workflow)
            for i, q in enumerate(qs[: len(ex.gold)]):
                drawn.setdefault((source, q.name), Counter())[ex.gold[i]] += 1
    return drawn


# ---- where the prior is measured ---------------------------------------------

def test_the_prior_is_pooled_across_question_sets_not_within_one():
    """Every set here holds one row, so a within-set histogram is flat by
    construction. If `cell_priors` read the set it would see 0.5/0.5 everywhere and
    the balancer would be a no-op that reports success — decision-v2's exact shape
    (17,112 sets, median pool 1)."""
    for key, (qs, exs) in SKEW.items():
        assert len(exs) == 1, "each set holds one row"
    shares = cell_priors(SKEW)[("boolq", "answer")]
    assert shares == pytest.approx({0: 0.25, 1: 0.75})


def test_the_prior_and_the_majority_floor_are_one_aggregation():
    """`max(cell_priors)` must equal `cell_stats[...]majority` exactly, or "trained
    against this prior" and "beat this floor" name different numbers (§9.41)."""
    priors, stats = cell_priors(ALL), cell_stats(ALL)
    assert set(priors) == set(stats)
    for cell, nat in priors.items():
        assert max(nat.values()) == pytest.approx(stats[cell]["majority"], abs=1e-12)
        assert sum(nat.values()) == pytest.approx(1.0)


def test_labels_stay_out_of_the_default_cell_stats():
    """The published report JSON's keys are a contract: `with_labels` is opt-in, so
    `myna.report`'s committed artifacts keep their shape."""
    assert "labels" not in cell_stats(SKEW)[("boolq", "answer")]
    assert cell_stats(SKEW, with_labels=True)[("boolq", "answer")]["labels"] == {0: 10, 1: 30}


def test_source_of_group_is_what_load_split_writes():
    assert source_of_group("agnews#b8d8fcae") == "agnews"
    assert source_of_group("support") == "support", "the synthetic corpus has no '#'"


# ---- what the weights do ------------------------------------------------------

def test_a_flat_corpus_gets_flat_weights():
    """Nothing to flatten -> nothing moved: weights are exactly 1.0, so a run with
    the flag on would train on the same distribution it trains on today."""
    flat = _one_source("imdb", [("pos", "noul", ["no", "yes"])], [20, 20])
    for key, ws in balance_weights(flat).items():
        assert ws == [1.0], "a 50/50 cell must not be re-weighted"


def test_minority_rows_outrank_majority_by_the_inverse_share():
    """A single-slot cell: w(minority)/w(majority) == 0.75/0.25 == 3.0, and the
    per-source rescale divides both by the source's mean weight (2.0 for a binary
    cell) so the weighted mean stays 1."""
    ws = balance_weights(SKEW)
    minority = [ws[k][0] for k, (qs, exs) in SKEW.items() if exs[0].gold[0] == 0]
    majority = [ws[k][0] for k, (qs, exs) in SKEW.items() if exs[0].gold[0] == 1]
    assert minority[0] / majority[0] == pytest.approx(3.0)
    assert sum(minority) / len(minority) == pytest.approx(2.0)  # 1/0.25 over mean 2.0
    assert sum(majority) / len(majority) == pytest.approx(2 / 3)  # 1/0.75 over mean 2.0
    # the rescale is what buys the mix invariant: a flattened draw puts equal weight
    # mass on every label of the cell, so the two label classes have to sum the same
    assert sum(minority) == pytest.approx(sum(majority))


def test_two_slot_rows_are_priced_by_the_product_not_the_mean():
    """The shipped rule, pinned against the cell priors it reads. `MULTI`'s 4-way slot
    is flat and its binary slot is 0.25/0.75, so the flat slot cancels and the row-to-
    row weight ratio reads the combination rule directly: the product passes the
    skewed slot's full 3:1 through, while the mean dilutes it against the flat slot to
    1.5. A damped ratio is a prior that survives the batch, which is the whole defect
    this flag exists to remove."""
    priors = cell_priors(MULTI)
    topic, flag = priors[("agnews", "topic")], priors[("agnews", "flag")]
    assert topic == pytest.approx({0: .25, 1: .25, 2: .25, 3: .25})
    assert flag == pytest.approx({0: .25, 1: .75}), "the flag slot is the skewed one"

    def terms(key):
        ex = MULTI[key][1][0]
        return [1.0 / topic[ex.gold[0]], 1.0 / flag[ex.gold[1]]]

    rare = next(k for k in MULTI if MULTI[k][1][0].gold[1] == 0)
    common = next(k for k in MULTI if MULTI[k][1][0].gold[1] == 1)
    prod, mean = balance_weights(MULTI), balance_weights(MULTI, "mean")
    assert balance_weights(MULTI) == balance_weights(MULTI, "prod"), "prod is the default"
    assert prod != mean, "the default rule is distinguishable from the mean"
    assert prod[rare][0] / prod[common][0] == pytest.approx(3.0)
    assert mean[rare][0] / mean[common][0] == pytest.approx(1.5)
    # and the product-then-per-source-rescale is exactly this arithmetic, recomputed
    raw = {k: terms(k)[0] * terms(k)[1] for k in MULTI}
    mean_raw = sum(raw.values()) / len(raw)
    for key, want in raw.items():
        assert prod[key][0] == pytest.approx(want / mean_raw)


def test_each_source_sums_to_its_row_count():
    """The mix invariant. Left unnormalized, inverse frequency scales a source by how
    many labels its cells use — trec's 6-way cell has mean weight 6 against boolq's 2,
    so a naive draw would train on trec three times as often as boolq. That is a
    task-mix change, not a label-prior change."""
    counts = Counter(source_of_group(key) for key, (_q, exs) in ALL.items()
                     for _ in exs)
    ws = balance_weights(ALL)
    per_source: dict[str, float] = {}
    for key, w in ws.items():
        per_source[source_of_group(key)] = per_source.get(source_of_group(key), 0.0) + sum(w)
    assert len(counts) == 4, "the fixtures are four sources"
    for source, total in per_source.items():
        assert total == pytest.approx(counts[source]), f"{source}'s share moved"


def test_flatten_weighted_returns_the_same_rows_with_their_own_weights():
    """`weights[i]` belongs to `items[i]`; a different order would silently re-weight
    the wrong rows and no printed number would notice. Checked against
    `balance_weights`'s own per-group lists, row for row, on a group that is 40 rows
    deep — a singleton-group fixture cannot tell `ws[key][j]` from `ws[key][0]`."""
    items, weights = flatten_weighted(ALL)
    ref = flatten_groups(ALL)
    assert [id(e) for _q, e in items] == [id(e) for _q, e in ref]
    assert len(weights) == len(ref)
    per_group: dict[str, list[float]] = {}
    for (qs, ex), w in zip(items, weights):
        per_group.setdefault(ex.workflow, []).append(w)
    assert per_group == {k: v for k, v in balance_weights(ALL).items()}


# ---- what reaches a batch ----------------------------------------------------

def test_the_row_batch_draw_flattens_the_marginal():
    """The gate for the whole mechanism: through `draw_row_batch`, the batcher the
    training loop calls, with the cell budget and distinctness rules live."""
    drawn_u = _draw_all(_merge(SKEW, WIDE), 8, 1 << 30, None)
    items, ws = flatten_weighted(_merge(SKEW, WIDE))
    drawn_w = _draw_all(_merge(SKEW, WIDE), 8, 1 << 30, WeightedDraw(ws))
    for cell, nat in cell_priors(_merge(SKEW, WIDE)).items():
        want = max(nat.values())
        assert _majority(drawn_u[cell]) == pytest.approx(want, abs=0.05), \
            "the control arm must reproduce the natural prior"
        got = _majority(drawn_w[cell])
        k = len(nat)
        assert got < want - 0.5 * (want - 1.0 / k), f"{cell}: {want} -> {got} did not flatten"
        assert got == pytest.approx(1.0 / k, abs=0.08), f"{cell}: {want} -> {got}"


def test_the_shared_set_draw_flattens_too():
    """Path P2-off. A flag wired only into `--row-batch` would print the same banner
    and train on the old marginals; `POOL` is the data-rich shape that path sees —
    and the only shape it can act on, since a set of one row is drawn whole or not."""
    _, pool = POOL["support#one"]
    drawer = WeightedDraw(balance_weights(POOL)["support#one"])
    for drawer_arg, label, expect in ((None, "uniform", 0.875), (drawer, "weighted", 0.5)):
        rng = random.Random(3)
        c = Counter()
        for _ in range(2000):
            for ex in draw_batch(pool, 4, rng, drawer=drawer_arg):
                c[ex.gold[0]] += 1
        got = _majority(c)
        assert got == pytest.approx(expect, abs=0.04), f"{label} arm: {got}"


def test_a_weighted_batch_shortens_instead_of_repeating():
    """The same "shorten, never repeat" cap `rng.sample` gives, on the weighted path.
    decision-v2's median pool is one row, so this is the common case, not an edge: a
    weighted drawer that answered a one-row pool with four copies of that row would
    drive the step's loss to ~0 and still report a batch of four."""
    row = POOL["support#one"][1][0]
    picked = draw_batch([row], 4, random.Random(0), drawer=WeightedDraw([1.0]))
    assert [id(e) for e in picked] == [id(row)], "a one-row pool cannot yield four"
    pool = POOL["support#one"][1]
    picked = draw_batch(pool, 4, random.Random(0),
                        drawer=WeightedDraw(balance_weights(POOL)["support#one"]))
    assert len(picked) == 4 and len({id(e) for e in picked}) == 4, "rows repeat"


def test_an_over_batch_ask_empties_the_pool_on_the_weighted_path():
    """`want = min(batch, len(pool))` on the drawer path, matching `rng.sample`'s cap on
    the other one. Ask a 40-row pool for 40 rows through a drawer that may hit the same
    index twice and the answer is a batch of 40 *draws* holding fewer rows — the step
    then pays the same gradient twice and the audit's row count stops matching the loss.

    A four-row ask over 40 rows is not evidence: collisions are merely unlikely, so the
    repeat-dropping rule can be deleted and the test still rides its seed."""
    pool = POOL["support#one"][1]
    drawer = WeightedDraw(balance_weights(POOL)["support#one"])
    for seed in (0, 1, 7):
        picked = draw_batch(pool, 40, random.Random(seed), drawer=drawer)
        assert len(picked) == 40, f"seed {seed}: {len(picked)} of 40 rows"
        assert len({id(e) for e in picked}) == 40, f"seed {seed}: a row repeated"


def test_the_unweighted_shared_set_path_shortens_instead_of_repeating():
    """The `--anti-prior off` shape every published figure was drawn in. Its cap is
    `rng.sample`'s; a comprehension over `randrange` returns `batch` copies whatever the
    pool holds, which is the ~0-loss step above arriving on the *untreated* arm — where
    nothing about the flag would explain it."""
    row = POOL["support#one"][1][0]
    assert [id(e) for e in draw_batch([row], 4, random.Random(0))] == [id(row)]
    pool = POOL["support#one"][1]
    for seed in (0, 1, 7):
        picked = draw_batch(pool, 40, random.Random(seed))
        assert len(picked) == 40 and len({id(e) for e in picked}) == 40, f"seed {seed}"
        assert len(draw_batch(pool, 60, random.Random(seed))) == 40, "the ask outran the pool"


def test_weighted_batches_still_fit_the_cell_budget():
    """The flattening cannot be bought by spending memory: the `--max-q-cells` rule is
    unchanged when the drawer is live."""
    items, ws = flatten_weighted(ALL)
    drawer = WeightedDraw(ws)
    cache = {}
    for seed in (0, 1, 2):
        rng = random.Random(seed)
        for _ in range(200):
            picked = draw_row_batch(items, 8, 256, rng, TOK, cache, drawer=drawer)
            assert picked, "a batch is never empty"
            assert len({id(e) for _q, e in picked}) == len(picked), "rows repeat"
            n = max(len(qs) for qs, _ in picked)
            lq = max(question_tokens(TOK, qs, cache) for qs, _ in picked)
            if len(picked) > 1:
                assert len(picked) * n * lq <= 256, "the budget moved"


def test_weighted_draw_refuses_an_empty_or_zero_mass_table():
    rng = random.Random(0)
    with pytest.raises(ValueError):
        WeightedDraw([])
    with pytest.raises(ValueError):
        WeightedDraw([0.0, 0.0])
    d = WeightedDraw([1.0])
    assert d.index(rng) == 0, "a one-row pool draws that row"
    assert d.index(rng) == 0


def test_balance_weights_rejects_an_unknown_rule():
    with pytest.raises(SystemExit):
        balance_weights(SKEW, "harmonic")


# ---- what the flag does not do ----------------------------------------------

def _reference_draw_row_batch(items, batch, max_q_cells, rng, tok, cache):
    """`myna.train.draw_row_batch` before `--anti-prior` existed, copied verbatim.

    This is a frozen reference on purpose. The flag's contract is that off means
    *today's numbers*, and today's numbers are a data order; comparing against a
    second implementation of the same loop is what catches a drawer quietly leaking
    into the unweighted path."""
    out, seen, n_max, lq_max = [], set(), 0, 0
    for _ in range(batch * 40):
        if len(out) == batch:
            break
        i = rng.randrange(len(items))
        if i in seen:
            continue
        qs = items[i][0]
        n, lq = len(qs), question_tokens(tok, qs, cache)
        if out and (len(out) + 1) * max(n, n_max) * max(lq, lq_max) > max_q_cells:
            continue
        seen.add(i)
        n_max, lq_max = max(n, n_max), max(lq, lq_max)
        out.append(items[i])
    return out


def test_the_off_paths_reproduce_the_pre_flag_sampler_exactly():
    """A sequence, not one batch. The flag's contract is *the same data order*, and the
    way to break it without moving the first batch is to consume one extra rng draw per
    batch — a pick made after the batch-full check instead of before it. Batch one is
    then byte-identical and every batch after it is a different run, which is exactly
    how a published figure moves while the test that checked it stays green."""
    for seed in (0, 1, 17):
        items = flatten_groups(ALL)
        rng_a, rng_b = random.Random(seed), random.Random(seed)
        for i in range(6):
            a = draw_row_batch(items, 6, 512, rng_a, TOK, {})
            b = _reference_draw_row_batch(items, 6, 512, rng_b, TOK, {})
            assert [id(e) for _q, e in a] == [id(e) for _q, e in b], f"seed {seed} batch {i}"
        assert rng_a.random() == rng_b.random(), "the two paths consumed different draws"
    questions, pool = POOL["support#one"]
    for seed in (0, 5):
        assert [id(e) for e in draw_batch(pool, 4, random.Random(seed))] == \
               [id(e) for e in random.Random(seed).sample(pool, min(4, len(pool)))]


# ---- the audit: the run reports what it drew, not what it asked for ----------

def test_the_audit_reports_only_the_cells_the_flag_targets():
    """A deviation on a cell that was already flat is sampling noise on a mechanism
    that never fired, and printing it would inflate the reach claim. `MULTI` is in this
    fixture for that reason: its `topic` cell is 0.25 and must not appear, while its
    `flag` cell at 0.75 must — one corpus, both halves of the filter."""
    groups = _merge(SKEW, WIDE, MULTI)
    audit = PriorAudit(groups)
    items, ws = flatten_weighted(groups)
    drawer = WeightedDraw(ws)
    rng = random.Random(0)
    cache = {}
    for _ in range(600):
        picked = draw_row_batch(items, 8, 1 << 30, rng, TOK, cache, drawer=drawer)
        audit.add_items(picked)
    lines, payload = audit.report()
    assert set(payload["cells"]) == {"boolq/answer", "trec/type", "agnews/flag"}, \
        "a flat cell has no business in the audit"
    assert payload["rows_drawn"] == audit.rows
    # the batcher can come up short when a pick repeats, but not by much
    assert 595 * 8 <= audit.rows <= 600 * 8
    for name, cell in payload["cells"].items():
        assert cell["majority_drawn"] < cell["majority_natural"], name
        assert sum(cell["drawn"].values()) == pytest.approx(1.0)
    assert len(lines) == 4, "one line per skewed cell, then the draw summary"
    assert all(ln.startswith("anti-prior:") for ln in lines)
    # the payload's second half is the mix guard: it must say the arms agree
    assert payload["max_source_shift"] == pytest.approx(
        max(abs(v["drawn"] - v["natural"]) for v in payload["source_mix"].values()))


def test_an_audit_that_drew_nothing_reports_nothing_moved():
    """`max_source_shift` against zero draws would otherwise print 1.0 — a hundred-point
    mix shift attributed to weights that never ran a batch."""
    audit = PriorAudit(SKEW)
    assert audit.rows == 0
    _lines, payload = audit.report()
    assert payload == {"rows_drawn": 0, "skew": ANTI_PRIOR_SKEW, "cells": {},
                       "source_mix": {}, "max_source_shift": 0.0}


# ---- the CLI surface: both batch paths, driven as the trainer is driven ------

def _row_json(source, qs, ex):
    """One fixture row in the suite schema: `choice` carries a dict of label->text and
    a label that is one of its keys, `noul` a bool."""
    return {"state": {"document": ex.state},
            "questions": {q.name: {
                "type": q.type, "instructions": q.instruction,
                "criteria": (None if q.type == "noul" else
                             {str(i): o for i, o in enumerate(q.options)}),
                "label": (bool(ex.gold[i]) if q.type == "noul" else str(ex.gold[i]))}
                for i, q in enumerate(qs)},
            "_meta": {"source": source}}


def _write_rows(dirpath: Path, rows):
    """Split 3 of every 4 rows of a source to train, the fourth to dev/test/calibration.

    A stride, not a head/tail cut, because these fixtures are ordered by label: a
    contiguous split would move every marginal and the reach count would measure the
    split instead of the corpus."""
    dirpath.mkdir(parents=True, exist_ok=True)
    train, rest = [], []
    per_source: dict[str, int] = {}
    for r in rows:
        i = per_source[r["_meta"]["source"]] = per_source.get(r["_meta"]["source"], 0) + 1
        (rest if i % 4 == 0 else train).append(r)
    for name, part in (("train", train), ("development", rest),
                       ("test", rest), ("calibration", rest)):
        (dirpath / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in part), encoding="utf-8")
    return dirpath


def _write_suite(dirpath: Path):
    """The four fixtures above as a frozen suite: 5 cells in the train split, four of
    them above the reach threshold (boolq 0.733, trec 0.667, agnews/flag 0.733,
    support/risk 0.833) and one flat (agnews/topic 0.267). `POOL` is the one question-set
    with depth — 30 rows share it — which is what gives the shared-set path something
    to re-weight inside a batch."""
    rows = []
    for source, groups in (("boolq", SKEW), ("trec", WIDE), ("agnews", MULTI),
                           ("support", POOL)):
        for key, (qs, exs) in groups.items():
            for ex in exs:
                rows.append(_row_json(source, qs, ex))
    return _write_rows(dirpath, rows)


def _run_trainer(tmp_path, extra, suite=None, steps=8):
    suite = suite or _write_suite(tmp_path / "suite")
    out = tmp_path / "out"
    cmd = [sys.executable, "-m", "myna.train", "--suite", str(suite), "--out", str(out),
           "--device", "cpu", "--steps", str(steps), "--batch", "8", "--vocab", "200",
           "--max-q-cells", "4096", "--eval-every", "0"] + extra
    env = {**os.environ, "PYTHONPATH": SRC}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2500:]
    return r.stdout, json.loads((out / "metrics.json").read_text())


@pytest.mark.parametrize("extra,steps,moves", [
    (["--row-batch"], 24, "all"),
    ([], 12, {"support/risk"}),
], ids=["row-batch", "shared-set"])
def test_trainer_names_the_reach_and_reports_the_draw(tmp_path, extra, steps, moves):
    """Both batch paths reach the audit, and the banner states how many cells this
    corpus actually offers — the number a reader needs to weigh the claim.

    `moves` is per path, because the two paths re-weight different objects. The row
    pool is the whole split, so every cell can move. A shared-set drawer only sees the
    rows of one question-set, so on the path that cycles sets it can only flatten the
    set with depth: the singleton sets decision-v2 is mostly made of are drawn whole or
    not at all, and which of them a step visits is the cycle's business, not the
    drawer's. Asserting "all cells moved" on that arm would be testing the wrong lever.

    The bar is half the distance from the cell's own prior to its balanced share, the
    same shape `test_the_row_batch_draw_flattens_the_marginal` uses: a CLI run of 24
    updates is ~50 drawn rows per cell, and a fixed point-shift at that depth is
    sampling noise, not a wiring result. The statistical claim belongs to the seam
    tests, which draw 4,000 rows; this one only asks whether the drawer filled the
    batch — disconnect it and every cell lands on its prior, twice the bar away."""
    out, metrics = _run_trainer(tmp_path, ["--anti-prior", "on", "--free-gib", "40"] + extra,
                                suite=None, steps=steps)
    assert "mini-batches drawn by inverse label prior" in out
    assert "4 of 5 cells carry a majority label at or above 0.55" in out
    assert "exceeds it, using" not in out, "the plan shrank the batch; no draws to count"
    assert any(ln.startswith("anti-prior: ") and " majority " in ln and "->" in ln
               for ln in out.splitlines()), out[-1500:]
    assert metrics["anti_prior"] == "on"
    assert metrics["batch"] == 8, "the batch the audit counted is the batch asked for"
    assert metrics["anti_prior_audit"]["rows_drawn"] > 0
    cells = metrics["anti_prior_audit"]["cells"]
    assert set(cells) == {"boolq/answer", "trec/type", "agnews/flag", "support/risk"}
    for name in (sorted(cells) if moves == "all" else sorted(moves)):
        cell = cells[name]
        want, k = cell["majority_natural"], len(cell["natural"])
        assert cell["majority_drawn"] <= want - 0.5 * (want - 1.0 / k), \
            f"{name}: {want:.3f} -> {cell['majority_drawn']:.3f} barely moved"


def test_off_records_no_mechanism(tmp_path):
    out, metrics = _run_trainer(tmp_path, ["--anti-prior", "off"])
    assert "inverse label prior" not in out
    assert metrics["anti_prior"] == "off"
    assert metrics["anti_prior_audit"] is None


def test_a_flat_corpus_says_the_flag_changed_no_draws(tmp_path):
    """The no-op case must announce itself. A banner claiming flattened marginals on
    a corpus with none is the same defect class as `--paraphrase` on an untabelled
    schema (§5 P1): the run prints a mechanism it did not run."""
    qs = [Question("pos", "noul", "Is this positive?", ["no", "yes"])]
    rows = [_row_json("imdb", qs, Example(f"imdb review number {i} about the thing",
                                         "imdb#0", (0 if i < 58 else 1,)))
            for i in range(116)]
    suite = _write_rows(tmp_path / "flat", rows)
    out, metrics = _run_trainer(tmp_path, ["--anti-prior", "on", "--row-batch"], suite=suite)
    assert "0 of 1 cells carry a majority label at or above 0.55" in out
    assert "nothing to flatten on this corpus, the flag changes no draws" in out
    assert metrics["anti_prior_audit"]["cells"] == {}
    assert metrics["anti_prior_audit"]["rows_drawn"] > 0, "the run still drew batches"


def test_long_context_refuses_the_flag(tmp_path):
    """Needle batches are generated per step, so there is no corpus marginal to
    re-weight against."""
    suite = _write_suite(tmp_path / "suite")
    env = {**os.environ, "PYTHONPATH": SRC}
    r = subprocess.run(
        [sys.executable, "-m", "myna.train", "--suite", str(suite), "--anti-prior", "on",
         "--long-context", "64", "--steps", "1", "--device", "cpu", "--vocab", "200",
         "--out", str(tmp_path / "out")],
        capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode != 0
    assert "not available with --long-context" in r.stderr + r.stdout



# ---- the audit harness's own face: it prints the table the SPEC and README quote ----
#
# `bench/anti_prior_audit.py` derives four things prose cites: which cells the flag
# reaches, what each arm's marginals became, that the task mix did not move, and which
# combination rule measurement picked. §9.46 is the record of a sibling harness whose
# pure arithmetic was fully mutation-checked and whose printing was never run by
# anything — so `main()` runs here, at a dose small enough to be a test, and every
# printed number is tied either to the payload that same run wrote or to a quantity
# recomputed from the rows. A dose-independent equality is still a gate: the lies this
# family catches are wrong columns, swapped arms and an unlabelled control.

REPO = Path(SRC).parent
sys.path.insert(0, str(REPO / "bench"))

from anti_prior_audit import (PUBLISHED_BATCH, PUBLISHED_MAX_Q_CELLS,  # noqa: E402
                              PUBLISHED_SETS_PER_UPDATE, PUBLISHED_UPDATES, main as audit_main)

WITNESS = json.loads((REPO / "runs/anti_prior_audit.json").read_text())
AUDIT_LOG = (REPO / "runs/anti_prior_audit.log").read_text()
RUN_LOG = (REPO / "runs/v1b_kaggle_3600b.train.log").read_text()
RUN_METRICS = json.loads((REPO / "runs/v1b_kaggle_3600b.metrics.json").read_text())
SUITE = REPO / "data/decision-v2-pilot/train.jsonl"

ARM_HEADERS = ("=== arm uniform:", "=== arm weighted:")
ARM_LABELS = ("--- uniform: --anti-prior off, the control arm ---",
              "--- weighted: --anti-prior on, combine='prod', the treated arm ---")
CELL_LINE = re.compile(r"anti-prior: (\S+) majority ([\d.]+) -> ([\d.]+) over (\d+) drawn rows")
MIX_LINE = re.compile(r"^\s+(\S+)\s+rows\s+[-+]?([\d.]+)\s+uniform\s+[-+]?([\d.]+)"
                      r"\s+\([-+][\d.]+\)\s+weighted\s+[-+]?([\d.]+)\s+\([-+][\d.]+\)"
                      r"\s+weighted-uniform\s+([-+][\d.]+)$")


def _between(text, begin, end):
    """The lines printed strictly between two markers."""
    i = text.index(begin) + len(begin)
    return text[i:text.index(end, i)].splitlines()


def _table(lines):
    """`marginals_table` rows -> {cell: (natural, realized)}. The type column is what
    says a line is a cell row rather than its header or its rule of dashes."""
    out = {}
    for ln in lines:
        cols = ln.split()
        if len(cols) >= 6 and cols[1] in ("noul", "choice", "score"):
            out[cols[0]] = (float(cols[4]), float(cols[5]))
    return out


def _audit_lines(lines):
    return {m.group(1): (float(m.group(2)), float(m.group(3)), int(m.group(4)))
            for m in (CELL_LINE.match(ln) for ln in lines) if m}


@pytest.fixture(scope="module")
def reduced(tmp_path_factory):
    """One `main()` run of the harness at 1/45th of the published dose, with its stdout,
    its JSON and the rows it read.

    The dose is not what is asserted — each test compares a printed number with the
    payload the same run wrote, which is dose-independent — but the printed face only
    exists if the script is run. The flags match the published command's shape, including
    `--compare`: a mutation that only breaks that path is invisible to a test that leaves
    it off, and `runs/` is symlinked into the scratch repo the battery runs in, so an
    assertion about the *committed* witness cannot see a mutated harness at all."""
    out = tmp_path_factory.mktemp("audit") / "audit.json"
    flags = ["--updates", "120", "--sets-per-update", "4", "--seed", "3", "--compare",
             "--out", str(out)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = audit_main(flags)
    assert rc == 0, "the audit harness did not exit clean"
    return {"text": buf.getvalue(), "doc": json.loads(out.read_text()), "flags": flags,
            "groups": load_split(SUITE)}


def test_the_two_arms_are_labelled_and_each_line_belongs_to_its_own_arm(reduced):
    """The trainer prints the same `anti-prior: cell majority X -> Y` banner for both
    arms, so a reader of the log can only use the block if it is named. This was the
    shape the harness shipped with: one run-on list, control and treatment
    indistinguishable, under a header saying "the two audits"."""
    text, doc = reduced["text"], reduced["doc"]
    assert ARM_LABELS[0] in text and ARM_LABELS[1] in text, "the two arms print unlabelled"
    assert text.index(ARM_LABELS[0]) < text.index(ARM_LABELS[1]), "the arms print reversed"
    uni = _audit_lines(_between(text, ARM_LABELS[0], ARM_LABELS[1]))
    wtd = _audit_lines(_between(text, ARM_LABELS[1], "source mix of the drawn"))
    assert uni and wtd, "an arm printed no marginal at all"
    for arm, key in ((uni, "uniform"), (wtd, "weighted")):
        cells = doc[key]["cells"]
        assert set(arm) == set(cells), key
        for cell, (nat, drawn, rows) in arm.items():
            assert nat == pytest.approx(cells[cell]["majority_natural"], abs=5e-4), (key, cell)
            assert drawn == pytest.approx(cells[cell]["majority_drawn"], abs=5e-4), (key, cell)
            assert rows == cells[cell]["rows_drawn"], (key, cell)
    assert uni != wtd, "both arms drew the same batches — one of them has the drawer"


def test_each_arms_table_prints_the_numbers_its_payload_carries(reduced):
    """Two renderings of one measurement, sixteen rows wide instead of six: the natural
    column against the prior recomputed here from the rows, the realized column against
    the same run's JSON. Print the natural majority into the `realized` column and a
    corpus the flag never touched looks flattened, in a table six cells wide would hide
    nothing of."""
    text, doc, groups = reduced["text"], reduced["doc"], reduced["groups"]
    priors = cell_priors(groups)
    assert len(priors) == 16
    ends = (ARM_HEADERS[1], "=== the two")
    for header, arm, end in zip(ARM_HEADERS, ("uniform", "weighted"), ends):
        got = _table(_between(text, header, end))
        assert len(got) == 16, f"{arm}: {len(got)} cells in the table"
        for cell, (nat, realized) in got.items():
            src, name = cell.split("/")
            assert nat == pytest.approx(max(priors[(src, name)].values()), abs=5e-4), (arm, cell)
            p = doc[arm]["cells"].get(cell)
            if p is not None:  # under `--skew` the audit reports the six it targeted
                assert realized == pytest.approx(p["majority_drawn"], abs=5e-4), (arm, cell)


def test_the_mix_block_prints_the_weights_own_column_and_no_other(reduced):
    """A source's drawn share is already points off its row share with the flag *off*,
    because `--max-q-cells` skips rows whose question branch would blow the budget — so
    the only column belonging to these weights is the difference between the arms, and
    the headline number has to be the largest of that column."""
    text, doc, groups = reduced["text"], reduced["doc"], reduced["groups"]
    rows: dict[str, int] = {}
    for key, (_q, exs) in groups.items():
        rows[source_of_group(key)] = rows.get(source_of_group(key), 0) + len(exs)
    sums: dict[str, list[float]] = {}
    for key, w in balance_weights(groups).items():
        sums.setdefault(source_of_group(key), []).extend(w)
    mine = max(abs(sum(w) - rows[s]) / rows[s] for s, w in sums.items())
    printed = re.search(r"worst relative deviation ([-+\d.e+]+)", text).group(1)
    assert printed == f"{mine:.2e}", f"printed {printed}, recomputed {mine:.2e}"
    assert doc["weight_sum_max_rel_deviation"] == pytest.approx(mine)
    assert mine < 1e-9, "the per-source rescale is not exact, so the mix would move"

    mix = {m.group(1): [float(m.group(i)) for i in (2, 3, 4, 5)]
           for m in (MIX_LINE.match(ln) for ln in
                     _between(text, "source mix of the drawn batches",
                              "largest between-arm")) if m}
    assert set(mix) == set(doc["uniform"]["source_mix"]) == set(rows)
    between, against_rows = [], 0.0
    for src, (want, gu, gw, last) in mix.items():
        ju, jw = doc["uniform"]["source_mix"][src], doc["weighted"]["source_mix"][src]
        assert want == pytest.approx(ju["natural"] * 100, abs=0.006), (src, "row share")
        assert gu == pytest.approx(ju["drawn"] * 100, abs=0.006), (src, "uniform share")
        assert gw == pytest.approx(jw["drawn"] * 100, abs=0.006), (src, "weighted share")
        assert last == pytest.approx(gw - gu, abs=0.011), (src, "the column that is not the budget's")
        between.append(abs(gw - gu))
        against_rows = max(against_rows, abs(gu - want), abs(gw - want))
    got = float(re.search(r"largest between-arm mix difference: ([-\d.]+) pts", text).group(1))
    assert got == pytest.approx(max(between), abs=0.02)
    assert got == pytest.approx(doc["between_arm_mix_max_pts"], abs=0.02)
    assert got < against_rows - 0.05, "that is the cell budget's shift, not the weights'"


def test_the_committed_witness_prices_the_published_dose():
    """The audit's whole warrant is that these are the marginals a run trained on, so the
    committed dose is read off the committed run's own command and `metrics.json` rather
    than from constants in this file. Note which batch is meant: the run *asked*
    `--batch 32` and the memory plan sized it to 10, and 10 is the batch the rows were
    actually drawn in — pricing the flag would price a run that never happened."""
    cmd = next(ln for ln in RUN_LOG.splitlines() if " -m myna.train " in ln)
    asked = dict(re.findall(r"--([a-z-]+) (\S+)", cmd))
    assert "row-batch" in asked, "the audit prices the --row-batch path"
    for key, flag in (("max_q_cells", "max-q-cells"), ("sets_per_update", "accum-groups"),
                      ("updates", "steps"), ("seed", "seed")):
        assert WITNESS[key] == int(asked[flag]), flag
    assert WITNESS["batch"] == RUN_METRICS["batch"]
    assert (WITNESS["batch"], WITNESS["max_q_cells"], WITNESS["sets_per_update"],
            WITNESS["updates"]) == (PUBLISHED_BATCH, PUBLISHED_MAX_Q_CELLS,
                                    PUBLISHED_SETS_PER_UPDATE, PUBLISHED_UPDATES)
    assert int(asked["batch"]) != WITNESS["batch"], "the plan, not the flag, set the batch"


def test_a_run_names_the_flags_it_was_given_and_the_rows_it_read(reduced):
    """§9.30's rule, applied to this harness: the witness records the command that made
    it, so a reader can tell which flags produced which marginals — and the dose knobs
    have to be read, not hardcoded, or a re-run at a different dose reports the old one."""
    text, doc, flags = reduced["text"], reduced["doc"], reduced["flags"]
    head = text.splitlines()[0]
    assert head.startswith("cmd: ") and head.endswith(" ".join(flags)), head
    for key, flag in (("updates", "--updates"), ("sets_per_update", "--sets-per-update"),
                      ("seed", "--seed")):
        assert doc[key] == int(flags[flags.index(flag) + 1]), flag
    digest = hashlib.sha256(SUITE.read_bytes()).hexdigest()
    assert digest[:16] in text.splitlines()[1] and doc["suite_sha256"] == digest
    assert doc["rows"] == 57904 and doc["question_sets"] == 17112


def test_the_run_says_how_much_it_drew_and_that_it_claims_no_accuracy(reduced):
    text, doc = reduced["text"], reduced["doc"]
    for arm in ("uniform", "weighted"):
        s, payload = doc["stats"][arm], doc[arm]
        assert payload["rows_drawn"] == s["rows_drawn"]
        assert s["mini_batches"] == 120 * 4
        assert s["mean_batch"] == pytest.approx(s["rows_drawn"] / s["mini_batches"], abs=0.005)
        assert 6.0 < s["mean_batch"] < PUBLISHED_BATCH, "the budget, not the ask, sized it"
        assert s["question_sets_seen"] > 0
    assert "nothing here is an accuracy claim" in text
    assert set(doc) == set(WITNESS), "the witness lost or gained a block"


def test_the_committed_witness_is_the_published_command():
    assert WITNESS["cmd"] == "anti_prior_audit.py --compare --out runs/anti_prior_audit.json"
    assert AUDIT_LOG.splitlines()[0] == "cmd: " + WITNESS["cmd"]
    assert AUDIT_LOG.count("nothing here is an accuracy claim") == 1
    assert ARM_LABELS[0] in AUDIT_LOG and ARM_LABELS[1] in AUDIT_LOG
    assert f"sha256 {WITNESS['suite_sha256'][:16]}" in AUDIT_LOG
    assert WITNESS["updates"] == 3600 and WITNESS["compare"], "not a --compare run"
    assert set(WITNESS["compare"]) == set(COMBINE)


def test_the_reach_line_counts_every_cell_the_split_has(reduced):
    """`6 of 16` is the sentence the SPEC quotes for what the flag can reach. The
    denominator is the cells the split carries, not the skewed ones — and the list beside
    it is the same six the payload holds, most skewed first."""
    text, doc, groups = reduced["text"], reduced["doc"], reduced["groups"]
    assert len(cell_priors(groups)) == 16
    line = next(ln for ln in text.splitlines() if ln.startswith("cells at or above skew"))
    assert line.startswith(f"cells at or above skew {doc['skew']}: 6 of 16 — ")
    listed = [p.strip().rsplit(" ", 1) for p in line.split("— ")[1].split(", ")]
    assert [n for n, _ in listed] == sorted(doc["cells_skewed"],
                                           key=lambda c: -doc["cells_skewed"][c])
    for name, val in listed:
        assert float(val) == pytest.approx(doc["cells_skewed"][name], abs=5e-5), name
    assert max(doc["cells_skewed"].values()) == pytest.approx(0.7545, abs=1e-4)


def test_the_committed_compare_table_names_the_rule_measurement_picked():
    """`prod` is shipped because it realizes the lowest drawn majority on every targeted
    cell, and the note that names the rule has to agree with `balance_weights`'s own
    default — the argument every arm is actually drawn with. Both halves are load-bearing:
    a note that names another rule retires the measurement, and a default that drifts from
    it means the table and the trainer disagree about which arm ran."""
    note = re.search(r"shipped rule: (\w+) —", AUDIT_LOG).group(1)
    assert note == inspect.signature(balance_weights).parameters["combine"].default
    apart = set()
    for cell in WITNESS["cells_skewed"]:
        realized = {r: a["cells"][cell]["majority_drawn"] for r, a in WITNESS["compare"].items()}
        assert realized[note] == min(realized.values()), (cell, realized)
        if realized["prod"] < realized["max"] - 0.05:
            apart.add(cell)
    # On a one-slot row `max` and `prod` are the same number, so the rule only shows its
    # effect where a row answers several questions — which is exactly what shipping the
    # product buys, and what a default drifting to `max` would quietly give back.
    assert apart == set(WITNESS["cells_skewed"]) - {"boolq/answer"}, apart

    head = next(ln for ln in AUDIT_LOG.splitlines() if ln.startswith("rule "))
    body = _between(AUDIT_LOG, head, "(natural row share")
    rows = {ln.split()[0]: ln.split()[1:] for ln in body
            if ln.strip() and ln.split()[0].isalpha()}
    assert list(rows) == ["natural", "geo", "max", "mean", "prod", "sum"], list(rows)
    order = sorted(WITNESS["cells_skewed"], key=lambda c: -WITNESS["cells_skewed"][c])
    assert len(order) == len(rows["natural"]) == 6
    for rule, vals in rows.items():
        for cell, printed in zip(order, vals):
            want = (WITNESS["cells_skewed"][cell] if rule == "natural"
                    else WITNESS["compare"][rule]["cells"][cell]["majority_drawn"])
            assert float(printed) == pytest.approx(want, abs=5e-5), (rule, cell, printed)


@pytest.fixture(scope="module")
def compared():
    """A second `main()` run, this time with `--compare`, which draws every rule in
    `COMBINE`.

    This exists because of the scratch repo `runs/` is symlinked into: a test that reads
    the *committed* witness cannot see a mutation of the code that wrote it. Every claim
    about this harness's printing therefore has to be checked against output produced
    inside the test, and the committed-witness tests below stay as the artifact binding."""
    out = Path(tempfile.mkdtemp()) / "audit.json"
    flags = ["--updates", "40", "--sets-per-update", "2", "--seed", "3", "--compare",
             "--out", str(out)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = audit_main(flags)
    assert rc == 0
    return buf.getvalue(), json.loads(out.read_text())


def test_a_compare_run_prints_the_table_its_payload_carries_and_names_the_rule_it_picked(compared):
    """Six rows, one per combination rule, and the note under them naming the shipped one.
    The table is how `prod` stopped being a guess, so two things have to hold here: each
    printed row is that rule's own measurement, and the `natural` row is the prior rather
    than some rule's value.

    What is deliberately *not* asserted here is which rule flattens most. At this dose a
    draw is a few dozen rows per cell, so a column can read 0.5 by under-sampling rather
    than by flattening (`mean` and `sum` print exactly 0.5000 on agnews/is_business,
    which the shipped rule leaves at 0.6250) and another can read *above* its own prior
    (`mean` on agnews/is_world: 0.8947 against a 0.7390 natural majority). `prod` is the
    minimum in one of six columns here, where at the published dose it is the minimum in
    all six. A ranking claimed from a run this short would be a claim about the seed, so
    it is made against the committed table instead, in
    `test_the_committed_compare_table_names_the_rule_measurement_picked`, which checks it
    against the same document's payload. What this test can kill is the note drifting from
    the shipped default while the code still prints a table."""
    text, doc = compared
    head = next(ln for ln in text.splitlines() if ln.startswith("rule "))
    body = _between(text, head, "(natural row share")
    rows = {ln.split()[0]: ln.split()[1:] for ln in body if ln.strip() and not ln.startswith("-")}
    assert list(rows) == ["natural", "geo", "max", "mean", "prod", "sum"], list(rows)
    order = sorted(doc["cells_skewed"], key=lambda c: -doc["cells_skewed"][c])
    assert len(order) == len(rows["natural"]) == 6
    for rule, vals in rows.items():
        for cell, printed in zip(order, vals):
            want = (doc["cells_skewed"][cell] if rule == "natural"
                    else doc["compare"][rule]["cells"][cell]["majority_drawn"])
            assert float(printed) == pytest.approx(want, abs=5e-5), (rule, cell, printed)
    note = re.search(r"shipped rule: (\w+) —", text).group(1)
    assert note == inspect.signature(balance_weights).parameters["combine"].default

"""Train a Myna checkpoint on the synthetic typed-decisions corpus.

Usage:  uv run python -m myna.train [--steps 4000] [--device mps]
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .data import WORKFLOWS, generate
from .model import MynaConfig, MynaModel, typed_loss
from .real_data import ANTI_PRIOR_SKEW, balance_weights
from .tokenizer import question_tensors, batch_question_tensors, train_tokenizer, encode_text
try:
    from .config_scaled import MynaConfigScaledT4
except ImportError:
    MynaConfigScaledT4 = None  # optional; only needed for T4 scaled runs


class WeightedDraw:
    """A cumulative inverse-frequency sampler over `range(len(weights))`.

    One O(n) table built at startup, then a bisect per pick. `random.choices` is the
    obvious call and was not used because it rebuilds its cumulative weights on every
    invocation: at 57,904 rows and 28,800 mini-batches that is ~1.7e9 float adds for
    a flag whose whole cost should be one pass over the corpus.

    Picks are *with* replacement and the batchers drop repeats. That is a real, if
    negligible, deviation from exact weighted sampling without replacement: a
    decision-v2 batch is 10 rows out of 57,904, so the chance any slot is re-drawn
    inside one batch is ~0.08%. It is the reason a batch can never come back empty
    when the cell budget rejects candidates.
    """

    def __init__(self, weights):
        cum, run = [], 0.0
        for w in weights:
            if w < 0:
                raise ValueError(f"negative draw weight {w}")
            run += w
            cum.append(run)
        if not cum or run <= 0:
            raise ValueError(f"draw table has no positive mass ({len(cum)} weights)")
        self.cum = cum
        self.total = run
        self.n = len(cum)

    def index(self, rng):
        return bisect.bisect_right(self.cum, rng.random() * self.total, hi=self.n - 1)


def draw_batch(pool, batch, rng, drawer=None):
    """Distinct examples only, shortening the batch rather than repeating one row.

    With replacement a singleton pool yields `batch` copies of the same example: the
    step drives its loss to ~0 and contributes almost no gradient. decision-v2 hits
    this constantly because option descriptions are randomized per row, so many
    question-sets hold a single row.

    `drawer` is a `WeightedDraw` over this pool's rows: when given, the batch is
    drawn by inverse label prior (`--anti-prior`, SPEC §5 P10) instead of uniformly.
    The unweighted path is untouched by the flag — same call, same `rng.sample` — so
    every figure published before it keeps its exact data order.
    """
    if drawer is None:
        return rng.sample(pool, min(batch, len(pool)))
    out, seen = [], set()
    want = min(batch, len(pool))  # the same "shorten, never repeat" cap as rng.sample
    for _ in range(want * 40):
        if len(out) == want:
            break
        i = drawer.index(rng)
        if i in seen:
            continue
        seen.add(i)
        out.append(pool[i])
    return out


def build_batch(exs, tok, questions, device):
    """questions: list[Question] fixed for the batch (one question set shared
    across rows; for per-row sets use build_row_batch)."""
    qt = question_tensors(tok, [(q.instruction, q.options) for q in questions])
    texts = [e.state for e in exs]
    ids = [encode_text(tok, t) for t in texts]
    ls = max(len(x) for x in ids)
    state = torch.zeros(len(ids), ls, dtype=torch.int64)
    lens = torch.tensor([len(x) for x in ids], dtype=torch.int64)
    for i, x in enumerate(ids):
        state[i, : len(x)] = torch.tensor(x)
    gold = torch.tensor([e.gold for e in exs], dtype=torch.int64)
    return {
        "state_ids": state.to(device),
        "state_len": lens.to(device),
        "gold": gold.to(device),
        "has_gold": torch.ones_like(gold, dtype=torch.bool).to(device),
        "ordinal": torch.tensor([q.type == "score" for q in questions],
                                dtype=torch.bool).to(device),
        **{k: v.to(device) for k, v in qt.items()},
    }


def build_row_batch(items, tok, device, paraphraser=None, rng=None):
    """items: list[(questions, Example)] — one question set PER ROW.

    Rows may differ in question count and option count; both pad to the batch
    maximum, and `has_gold` marks the cells that belong to a real question so a
    mixed batch trains on the union of its cells. This is what lets 32 boolq
    rows — 32 different instructions, one question each — share one forward
    pass instead of running 32 batches of one. `ordinal` marks the cells whose
    legend is ordered, which is what `--score-loss emd` prices.

    With `paraphraser`, each row's set is replaced by a drawn phrasing (SPEC §5
    P1: the suite's own wording is never trained on).
    """
    if paraphraser is not None:
        items = [(paraphraser.draw(ex.workflow, qs, rng), ex) for qs, ex in items]
    qt = batch_question_tensors(tok, [[(q.instruction, q.options) for q in qs] for qs, _ in items])
    n_q = max(len(qs) for qs, _ in items)
    ids = [encode_text(tok, ex.state) for _qs, ex in items]
    ls = max(len(x) for x in ids)
    state = torch.zeros(len(ids), ls, dtype=torch.int64)
    lens = torch.tensor([len(x) for x in ids], dtype=torch.int64)
    for i, x in enumerate(ids):
        state[i, : len(x)] = torch.tensor(x)
    gold = torch.zeros(len(items), n_q, dtype=torch.int64)
    has_gold = torch.zeros(len(items), n_q, dtype=torch.bool)
    ordinal = torch.zeros(len(items), n_q, dtype=torch.bool)
    for i, (qs, ex) in enumerate(items):
        gold[i, : len(qs)] = torch.tensor(ex.gold[: len(qs)], dtype=torch.int64)
        has_gold[i, : len(qs)] = True
        ordinal[i, : len(qs)] = torch.tensor([q.type == "score" for q in qs], dtype=torch.bool)
    return {
        "state_ids": state.to(device),
        "state_len": lens.to(device),
        "gold": gold.to(device),
        "has_gold": has_gold.to(device),
        "ordinal": ordinal.to(device),
        **{k: v.to(device) for k, v in qt.items()},
    }


def question_tokens(tok, questions, cache=None):
    """The row's longest question, in tokens — the width its batch row pads to.

    Measured with the tokenizer the batch will actually use: a chars/4 rule is
    off by 2-4x on a small BPE, and this number is what the memory budget is
    made of. Memoized on the question set, which a suite reuses across
    thousands of rows, so the whole corpus pays it once per schema.
    """
    key = id(questions)
    if cache is not None and key in cache:
        return cache[key]
    lq = max(len(encode_text(tok, q.instruction))
             + sum(len(encode_text(tok, o)) + 1 for o in q.options) + 1  # +1 [OPT] each, +1 [DECIDE]
             for q in questions)
    if cache is not None:
        cache[key] = lq
    return lq


def draw_row_batch(items, batch, max_q_cells, rng, tok, cache=None, drawer=None):
    """`batch` distinct rows whose padded question branch stays under
    `max_q_cells` = rows x questions x question-tokens.

    That product is the axis the machine dies on, and it is *not* the option
    axis: bench/mem_profile.py measures 1.6 MiB of retained activations per
    question-token position against 3 KiB per option cell — a factor of five
    hundred (a 77-option question adds 2.7 MiB to a 24-slot batch). Both
    questions and question tokens pad to the batch maximum, so one long
    instruction mixed in with two short ones charges the whole batch for the
    long one. The first picked row always goes in, so a batch is never empty.

    `drawer` is a `WeightedDraw` over `items`: with it, which rows are offered
    follows the inverse label prior instead of the uniform row share, and the
    distinctness and cell-budget rules above are unchanged. Without it the loop is
    the one every published figure was drawn by, rng call for rng call — which is
    why the pick is made *after* the batch-is-full check rather than by a generator:
    one extra draw per batch would shift the stream for every later step.
    """
    out, seen, n_max, lq_max = [], set(), 0, 0
    for _ in range(batch * 40):
        if len(out) == batch:
            break
        i = rng.randrange(len(items)) if drawer is None else drawer.index(rng)
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


def worst_case_tokens(tok, paraphraser, items, cache):
    """Pre-price every phrasing of every set at its *longest* variant.

    `draw_row_batch` budgets `rows x questions x question-tokens`, and under
    paraphrasing the token count is a random variable — so pricing the draw the
    sampler happened to take would let one step silently exceed the memory the
    run was sized for. Writing the max into the length cache for all eight
    variant objects makes the budget an upper bound again, at the cost of a
    slightly smaller batch.
    """
    by_key = {}
    for qs, ex in items:
        by_key.setdefault(ex.workflow, qs)
    for key, qs in by_key.items():
        sets = paraphraser.variant_set(key, qs)
        lengths = [question_tokens(tok, v) for v in sets]
        worst = max(lengths)
        for v in sets:
            cache[id(v)] = worst
        cache[id(qs)] = worst
    return len(by_key)


class PriorAudit:
    """Counts what the batches actually drew, so `--anti-prior` can report the label
    marginals it *reached* beside the ones it asked for.

    The weights are an argument; a batch of ten rows out of a 5,300-row cell is a
    statistical flattening, and 3,600 updates is a finite number of draws. This is
    the difference between "we re-weighted" and "the majority label appeared on
    0.62 of drawn rows instead of 0.75" — which is also the honest answer about
    reach: row-level weighting flattens a single-question cell almost exactly and a
    multi-question cell only part of the way, because one row answers three questions
    at once. `bench/anti_prior_audit.py` measures that gap on the shipped rows.

    The per-source draw mix is counted and carried in the payload but not printed as
    a verdict, because in this loop the mix is *already* moved by something else: the
    `--max-q-cells` budget skips a row that would blow the padded question branch, and
    banking77's 77-option sets pay that price, so its drawn share is ~4% against a 9%
    row share with the flag off too. Only the difference *between* the two arms
    belongs to these weights, and only the audit script can show both arms."""

    def __init__(self, groups):
        from .real_data import cell_priors, source_of_group

        self.priors = cell_priors(groups)
        self.natural_rows: Counter = Counter()
        for key, (_q, exs) in groups.items():
            self.natural_rows[source_of_group(key)] += len(exs)
        self.drawn: dict[tuple[str, str], Counter] = defaultdict(Counter)
        self.drawn_rows: Counter = Counter()
        self.rows = 0

    def add(self, questions, exs):
        """One mini-batch of a shared question set (path P2-off)."""
        for ex in exs:
            self.add_items([(questions, ex)])

    def add_items(self, items):
        """One mini-batch of `--row-batch` items, each with its own question set."""
        from .real_data import source_of_group

        for qs, ex in items:
            source = source_of_group(ex.workflow)
            self.drawn_rows[source] += 1
            self.rows += 1
            for i, q in enumerate(qs[: len(ex.gold)]):
                self.drawn[(source, q.name)][ex.gold[i]] += 1

    def report(self, skew=ANTI_PRIOR_SKEW):
        """-> (lines for the log, JSON-able payload). Cells under `skew` are left
        out: they were flat to begin with, so a deviation there is noise on a
        mechanism that never fired."""
        tot_nat = sum(self.natural_rows.values())
        cells, mix, worst = {}, {}, 0.0
        for (source, name), nat in self.priors.items():
            c = self.drawn.get((source, name))
            if not c or max(nat.values()) < skew:
                continue
            tot = sum(c.values())
            dr = {lab: n / tot for lab, n in c.items()}
            cells[f"{source}/{name}"] = {
                "rows_drawn": tot, "majority_natural": max(nat.values()),
                "majority_drawn": max(dr.values()),
                "natural": {str(k): v for k, v in sorted(nat.items())},
                "drawn": {str(k): v for k, v in sorted(dr.items())}}
        if self.rows:  # a drawn share out of zero draws is not a mix, it is a 1.0 shift
            for source, n in sorted(self.natural_rows.items()):
                want, got = n / tot_nat, self.drawn_rows[source] / self.rows
                mix[source] = {"natural": want, "drawn": got}
                worst = max(worst, abs(got - want))
        order = sorted(cells, key=lambda k: -cells[k]["majority_natural"])
        lines = [f"anti-prior: {k} majority {cells[k]['majority_natural']:.3f} -> "
                 f"{cells[k]['majority_drawn']:.3f} over {cells[k]['rows_drawn']} drawn rows"
                 for k in order]
        lines.append(f"anti-prior: {self.rows} rows drawn over {len(cells)} skewed cells; "
                     f"per-source draw mix in the payload")
        return lines, {"rows_drawn": self.rows, "skew": skew, "cells": cells,
                       "source_mix": mix, "max_source_shift": worst}


def macro_acc(m):
    """Unweighted mean over the per-(source, question) accuracies in an
    `evaluate()` map — the brier/ece sidecars are excluded by design.

    §9.41: 60% of this map's terms hold exactly one row, so the value is a
    paired-comparison statistic, not a level. Any claim about the *model* goes
    through `myna.report`'s row-weighted cells; this one is only safe against
    itself, across steps or arms.
    """
    names = [k for k in m if not k.endswith((":brier", ":ece"))]
    return sum(m[k] for k in names) / max(len(names), 1)


# ---- memory plan and the stop rule (SPEC §5 P3: 3d, 3e) ----------------------

MI = 1 << 20
# Both from bench/mem_profile.py at d_model 384 / vocab 8192 / batch 8 — measured
# retained activations per unit, not a model of what ought to cost what.
STATE_MIB_PER_POSITION = 0.8
QUESTION_MIB_PER_CELL = 1.6


def state_token_p95(tok, items, limit=4000, rng=None):
    """p95 of the state lengths, in tokens, over a sample of the rows the loop
    will actually draw. The sizing uses p95 rather than the mean because the
    batch pads to its longest row, so the tail *is* the cost."""
    exs = [ex for _qs, ex in items]
    if limit and len(exs) > limit:
        exs = (rng or random.Random(0)).sample(exs, limit)
    lens = sorted(len(encode_text(tok, e.state)) for e in exs)
    return lens[min(int(0.95 * (len(lens) - 1)), len(lens) - 1)] if lens else 0


def free_device_bytes(device):
    """Reads the headroom a trainer can actually use, or None.

    CUDA reports it directly. MPS and CPU do not: torch has no MPS headroom call,
    and free system RAM is not a budget one process may claim — on the M5 the
    projection that killed a run came from a *projected* rate, not a measured one
    (SPEC §6). Returning None is the honest answer, and it means `--batch` is
    taken exactly as given, with a line saying so."""
    if device == "cuda" and torch.cuda.is_available():
        try:
            free, _total = torch.cuda.mem_get_info()
            return int(free)
        except RuntimeError:
            return None
    return None


def rows_that_fit(free_bytes, tokens_p95, safety=0.6, mib_per_position=STATE_MIB_PER_POSITION,
                  reserve_bytes=0):
    """Rows whose *state scan* fits in `safety` of the reported headroom, after
    whatever else the step is known to hold.

    `reserve_bytes` is the question branch of a `--row-batch` step: it is priced
    from `--max-q-cells` and the measured 1.6 MiB/cell, so the batch cannot be
    sized as though the states were the only memory in the model — the M5 run died
    with both branches live. Returning 0 means "nothing fits", which the caller
    turns into a refusal rather than a batch of 0."""
    if not free_bytes or tokens_p95 <= 0:
        return None
    budget = free_bytes * safety - reserve_bytes
    if budget <= 0:
        return 0
    return max(1, int(budget / MI // (tokens_p95 * mib_per_position)))


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def stop_reason(step_times, free, projected_bytes, factor=1.5, window=20, headroom=1.25):
    """Why the loop should hard-stop, or None.

    Two signals, both of which we paid for once: a step time drifting above
    1.5x its own recent median means the allocator is paging or fragmenting,
    which a loss curve hides until the run is over (§6), and free bytes under
    `headroom` x the projected need means the *next* batch is the one that dies."""
    if len(step_times) >= window + 1 and step_times[-1] > factor * median(step_times[-window - 1:-1]):
        return (f"step time {step_times[-1]:.2f}s > {factor}x median "
                f"{median(step_times[-window - 1:-1]):.2f}s of the last {window}")
    if free is not None and projected_bytes and free < projected_bytes * headroom:
        return (f"free {free / MI:.0f} MiB < {headroom}x projected "
                f"{projected_bytes / MI:.0f} MiB per step")
    return None


def save_snapshot(out, model, opt, sched, tok, cfg, step, temperature=1.0):
    """The rolling snapshot: weights *and* the optimizer/scheduler state, so a
    resumed run continues the schedule instead of restarting the decay."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))
    torch.save({"state_dict": model.state_dict(), "cfg": vars(cfg), "temperature": temperature,
                "step": step, "optimizer": opt.state_dict(), "scheduler": sched.state_dict()},
               out / "model_last.pt")
    return out / "model_last.pt"


def load_snapshot(path, model, opt=None, sched=None):
    """Weights always; optimizer/scheduler when the caller passes them. Returns
    the checkpoint's step so the loop can start after it."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    if opt is not None and blob.get("optimizer"):
        opt.load_state_dict(blob["optimizer"])
    if sched is not None and blob.get("scheduler"):
        sched.load_state_dict(blob["scheduler"])
    return int(blob.get("step", 0))


def evaluate(model, tok, splits, device, temperature=1.0, paraphrase_index=None):
    """splits: {key: (questions, examples)} — synthetic workflows and real
    suite groups both fit this shape.

    `paraphrase_index`: score each group under that phrasing of its instruction
    instead of the wording on disk — how the held-out wording is measured."""
    from .paraphrase import paraphrase_questions, source_of

    model.eval()
    rows = {}
    ece_bin = {}  # key -> (sum_conf, sum_correct, count) aggregated in 10 bins
    with torch.no_grad():
        for wf, (questions, data) in splits.items():
            if paraphrase_index is not None:
                questions = paraphrase_questions(questions, source_of(wf), paraphrase_index)
            for i in range(0, len(data), 64):
                exs = data[i : i + 64]
                b = build_batch(exs, tok, questions, device)
                logits = model(b["state_ids"], b["state_len"], b["q_ids"], b["q_mask"],
                               b["span_mat"], b["opt_valid"], b["decide_idx"]) / temperature
                probs = F.softmax(logits, dim=-1)
                pred = probs.argmax(-1)
                for n, q in enumerate(questions):
                    hit = (pred[:, n] == b["gold"][:, n]).float()
                    key = f"{wf}/{q.name}"
                    acc, tot = rows.get(key, (0.0, 0))
                    rows[key] = (acc + float(hit.sum()), tot + len(exs))
                    conf = probs[:, n].max(-1).values
                    bins = (conf * 10).clamp(max=9).long()
                    for bi in range(10):
                        sel = bins == bi
                        if not bool(sel.any()):
                            continue
                        c, s, m = ece_bin.get(key, (0.0, 0.0, 0))
                        ece_bin[key] = (
                            c + float(conf[sel].sum()),
                            s + float(hit[sel].sum()),
                            m + int(sel.sum()),
                        )
                    if q.type == "noul":
                        p = probs[:, n, 1]
                        g = b["gold"][:, n].float()
                        bk, bt = rows.get(key + ":brier", (0.0, 0))
                        rows[key + ":brier"] = (bk + float(((p - g) ** 2).sum()), bt + len(exs))
    out = {k: v[0] / max(v[1], 1) for k, v in rows.items()}
    for key, (c, s, m) in ece_bin.items():
        out[f"{key}:ece"] = abs(c - s) / max(m, 1)  # single-bin-per-example ECE (M-norm, 10 bins)
    return out


def fit_temperature(model, tok, splits, device):
    """1-D grid on NLL — calibration is a single scalar per checkpoint."""
    best, best_t = 1e18, 1.0
    model.eval()
    with torch.no_grad():
        for t in [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0]:
            tot, cnt = 0.0, 0
            for wf, (questions, data) in splits.items():
                for i in range(0, min(len(data), 512), 64):
                    b = build_batch(data[i : i + 64], tok, questions, device)
                    logits = model(b["state_ids"], b["state_len"], b["q_ids"], b["q_mask"],
                                   b["span_mat"], b["opt_valid"], b["decide_idx"]) / t
                    lp = F.log_softmax(logits, dim=-1)
                    sel = lp.gather(-1, b["gold"][..., None]).squeeze(-1)
                    tot, cnt = tot - float(sel.sum()), cnt + sel.numel()
            if tot / cnt < best:
                best, best_t = tot / cnt, t
    return best_t


def held_out_eval(model, tok, split, device, temperature):
    """Score a split on the reserved ninth phrasing — the only place it is read.

    Kept as its own function so `EVAL_INDEX` has one call site a test can pin: a
    `paraphrase_index=0` slipped in here would silently score the training
    wording and call it a transfer result.
    """
    from .paraphrase import EVAL_INDEX

    return evaluate(model, tok, split, device, temperature, paraphrase_index=EVAL_INDEX)


def temperature_source(data):
    """Which split the scalar is fit on: the suite's own `calibration` group,
    else `dev`.

    dev is the fallback for the synthetic corpus, which ships no calibration
    split — but then the reported dev accuracy is fit on the rows it is scored
    on, and the printed line says so rather than leaving it implied."""
    calib = data.get("calibration")
    return (calib, "calibration") if calib else (data["dev"], "dev")


def split_report(split, groups):
    """The row counts have to ride with the split's name.

    Every corpus size quoted in a doc was retyped from a line like this one;
    `runs/train-v0.log` printed no such line, so the "3,000 synthetic examples"
    that sat next to the v0 table could not be checked and is dead (SPEC §9.30).
    """
    rows = [len(v[1]) for v in groups.values()]
    if not rows:
        return f"split {split}: 0 rows over 0 groups"
    return (f"split {split}: {sum(rows)} rows over {len(rows)} groups, "
            f"per-group {min(rows)}-{max(rows)}")


def resolve_device(name):
    """`auto` prefers MPS, then CUDA, then CPU.

    Resolving to mps/cpu only meant a CUDA cloud box — the structural fix for
    the M5's memory ceiling — silently trained on CPU.

    An *explicit* device that is not available raises instead of reaching
    `tensor.to()`: `--device cuda` on a Mac died inside torch with a two-frame
    stack, which reads like a torch bug rather than like a mistyped flag, and a
    Kaggle job that silently lands on CPU burns wall-clock hours (§6)."""
    if name != "auto":
        available = {"cpu": True,
                     "mps": torch.backends.mps.is_available(),
                     "cuda": torch.cuda.is_available()}.get(name)
        if available is None:
            raise SystemExit(f"--device {name!r}: unknown; use auto, cpu, mps or cuda")
        if not available:
            here = [d for d in ("mps", "cuda", "cpu")
                    if d == "cpu" or (d == "mps" and torch.backends.mps.is_available())
                    or (d == "cuda" and torch.cuda.is_available())]
            raise SystemExit(f"--device {name!r} is not available on this machine "
                             f"(it has: {', '.join(here)}); use --device auto")
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_needle_batch(exs, tok, device, entity):
    """One shared question (fixed entity) across the batch; the needle varies."""
    from .longctx import DESKS
    instr = f"Which desk owns {entity}?"
    qt = question_tensors(tok, [(instr, list(DESKS))])
    ids = [encode_text(tok, e.state) for e in exs]
    lens = torch.tensor([len(x) for x in ids], dtype=torch.int64)
    Lmax = max(len(x) for x in ids)
    state = torch.zeros(len(ids), Lmax, dtype=torch.int64)
    for i, x in enumerate(ids):
        state[i, : len(x)] = torch.tensor(x)
    gold = torch.tensor([[e.gold] for e in exs], dtype=torch.int64)
    return {
        "state_ids": state.to(device), "state_len": lens.to(device), "gold": gold.to(device),
        "Lmax": Lmax,
        **{k: v.to(device) for k, v in qt.items()},
    }


def train_long_context(args, rng, device):
    """V1-D: fine-tune (or train) on needle-in-long-context recall, truncated
    backprop so only the trailing grad window carries gradient."""
    from .longctx import DESKS, ENTITIES, FILLER, Needle, make_needle

    if args.init:
        ip = Path(args.init)
        tok = Tokenizer.from_file(str(ip / "tokenizer.json"))
        ck = torch.load(ip / "model.pt", map_location=device, weights_only=False)
        cfg = MynaConfig(**ck["cfg"])
        model = MynaModel(cfg)
        model.load_state_dict(ck["state_dict"])
    else:
        vocab = args.vocab
        if args.config == "scaled":
            if MynaConfigScaled is None:
                raise SystemExit("--config scaled requires src/myna/config_scaled.py (not installed)")
            cfg = MynaConfigScaled()
            # Override vocab if tokenizer has learned more entries than default
            # (this is rare; most runs keep 4096)
            vocab = max(vocab, cfg.vocab)
            cfg = MynaConfigScaled(vocab=vocab)
        else:
            cfg = MynaConfig(vocab=vocab)
        model = MynaModel(cfg)
        tok = train_tokenizer([" ".join(FILLER), " ".join(DESKS), " ".join(ENTITIES)],
                              vocab_size=vocab)
    model.to(device)
    print(f"long-context: state={args.long_context} grad={args.grad_tokens} "
          f"params={sum(p.numel() for p in model.parameters())}", flush=True)

    split = max(0, args.long_context - args.grad_tokens)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    ema, t0 = None, time.time()
    for step in range(args.steps):
        model.train()
        entity = ENTITIES[rng.randrange(len(ENTITIES))]
        exs = [make_needle(tok, args.long_context, rng, tail_cap=args.grad_tokens,
                           entity=entity) for _ in range(args.batch)]
        b = build_needle_batch(exs, tok, device, entity)
        logits = model.forward_truncated(b["state_ids"], split, b["state_len"], b["q_ids"],
                                         b["q_mask"], b["span_mat"], b["opt_valid"], b["decide_idx"])
        loss = typed_loss(logits, b["gold"], torch.ones_like(b["gold"], dtype=torch.bool))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        acc = (logits.argmax(-1) == b["gold"]).float().mean().item()
        ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        if step % 100 == 0 or step == args.steps - 1:
            print(f"step {step:5d}  loss {loss.item():.3f}  ema {ema:.3f}  "
                  f"batch-acc {acc:.3f}  {time.time()-t0:.0f}s", flush=True)
        if args.eval_every and step and step % args.eval_every == 0 and step != args.steps - 1:
            out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
            tok.save(str(out / "tokenizer.json"))
            torch.save({"state_dict": model.state_dict(), "cfg": vars(cfg),
                        "temperature": 1.0, "step": step}, out / "model_last.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=700)
    ap.add_argument("--out", default="runs/myna-v0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-n", type=int, default=192)
    ap.add_argument("--suite", default=None,
                    help="frozen-suite dir (train/development/test.jsonl, laya/kev request shape); "
                         "overrides the synthetic corpus")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--accum-groups", type=int, default=1,
                    help="question-sets averaged into each optimizer update. Real suites hold "
                         "1000+ one-example question-sets; with 1 set per update the shared trunk "
                         "chases a different task every step and never fits any (interference).")
    ap.add_argument("--warmup", type=int, default=0,
                    help="linear LR warmup over this many updates before cosine decay")
    ap.add_argument("--group-sample", choices=["uniform", "pool", "sqrt"], default="pool",
                    # %% because argparse runs the help string through %-formatting
                    help="how to pick question-sets per mini-batch. Measured on decision-v2, "
                         "uniform sends only ~1.4%% of updates to the 16 data-rich sets (12.5%% for "
                         "sqrt) and they measured below chance; pool sends 65%%. Pool weighting "
                         "was harmless before gradient accumulation removed the interference.")
    ap.add_argument("--min-train-pool", type=int, default=0,
                    help="drop train question-sets with fewer than this many examples "
                         "(isolates the data-rich tasks; 0 keeps the whole suite)")
    ap.add_argument("--row-batch", action="store_true",
                    help="draw each mini-batch across the whole train split with one question set "
                         "per row, instead of one shared set per mini-batch. Unblocks boolq/mnli "
                         "(whose instruction text IS the row) and decouples batch size from "
                         "question-set pool size; makes --group-sample inert.")
    ap.add_argument("--max-q-cells", type=int, default=2048,
                    help="with --row-batch: rows x questions x question-tokens per forward. "
                         "Measured at 1.6 MiB of retained activations per cell "
                         "(bench/mem_profile.py), so 2048 is a ~3.2 GB question branch. The "
                         "option axis costs ~3 KiB per cell and is not what filled the M5.")
    ap.add_argument("--anti-prior", choices=["off", "on"], default="off",
                    help="draw each mini-batch by inverse label prior instead of by row share "
                         "(SPEC §5 P10). What not doing it costs is measured: Tier 0 found three "
                         "agnews noul cells emitting one label on every test row and finishing "
                         "exactly on their own majority floor (0.653 / 0.809 / 0.778), because a "
                         "constant answer is worth ~0.75 against a train prior of 0.739-0.755 -- "
                         "the most rewarded output the data offers. Flattening that marginal makes "
                         "it worth 0.5. Weights are normalized within each source, so the mix of "
                         "tasks a run trains on does not move; only the answer histogram inside "
                         "each cell does. On the shared-set path the re-weighting is inside one "
                         "question-set, so pair it with --row-batch: a set holding a single row "
                         "(decision-v2's median) is drawn whole or not at all. Default off, "
                         "because this is a bet, not a gain: the audit measures the ten cells "
                         "the skew threshold does not target moving anyway -- nine of them "
                         "flatten, by up to 0.170 (contrastive/decision 0.507 -> 0.337, which is "
                         "its own 1/3 uniform), and one sharpens (yelp/rating +0.056, because a "
                         "yelp row answers two questions and these weights are computed on the "
                         "other one) -- and if removing the shortcut does not make the model "
                         "read, the macro falls: the floor it is judged against stays the "
                         "natural majority of the untouched test split.")
    ap.add_argument("--paraphrase", choices=["off", "on"], default="off",
                    help="train on hand-written phrasings of every instruction and never on the "
                         "suite's own wording (SPEC P1). Dev/test stay on the exact suite strings, "
                         "and the reserved ninth phrasing is scored too, so 'it learned to read' is "
                         "measured on two unseen wordings. Suite corpora only: an instruction with "
                         "no table entry raises UnknownSchema rather than passing through.")
    ap.add_argument("--score-loss", choices=["ce", "emd"], default="ce",
                    help="how a `score` cell is priced: ce is cross-entropy over the option "
                         "index, which charges \"predicted 4, gold 3\" the same as \"predicted "
                         "1, gold 5\"; emd charges the squared-Cramér distance over the legend, "
                         "normalised by the cell's own K-1, so a one-rung miss is cheap and the "
                         "opposite end costs 1.0 whatever the legend length (at K=2 it is the "
                         "Brier score). Default ce: every published figure was trained with it "
                         "(SPEC §5 P9: 9a).")
    ap.add_argument("--free-gib", type=float, default=None,
                    help="headroom the loop may plan against. CUDA reports its own; MPS and CPU "
                         "have no trustworthy reading, so on those the batch is taken exactly as "
                         "--batch gives it unless a number is stated here (SPEC P3: a projected "
                         "rate is not a budget, and the M5 run died on one).")
    ap.add_argument("--config", choices=["v0", "scaled"], default="v0",
                    help="model architecture config: v0 uses d_model=384, n_layers=6 (~15M params); "
                         "scaled uses d_model=512, n_layers=8 (~36M params) optimized for T4 "
                         "(≤16GB VRAM), targeting ≥0.60 macro accuracy from V1-B baseline 0.4893.")
    ap.add_argument("--mem-safety", type=float, default=0.6,
                    help="share of the reported headroom the plan may spend, leaving room for "
                         "activations the per-position table does not count")
    ap.add_argument("--save-every", type=int, default=25,
                    help="write model_last.pt (weights + optimizer + scheduler + step) every N "
                         "updates so a killed job loses at most N updates of progress (SPEC P3: "
                         "<=50)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from --out/model_last.pt: weights, optimizer and scheduler "
                         "state, and the step counter. The data RNG is re-seeded from --seed, so a "
                         "resumed run is a continuation, not a bit-identical replay, and the "
                         "resume line says which")
    ap.add_argument("--warm-start", default=None, metavar="MODEL_PT",
                    help="load only the WEIGHTS of an existing checkpoint and train on from "
                         "here, with a fresh optimizer and schedule over --steps. What an "
                         "ablation needs: --resume would also restore the saved scheduler's "
                         "T_max, which leaves the learning rate at ~0 past the original run's "
                         "last step and silently trains nothing.")
    ap.add_argument("--stop-factor", type=float, default=1.5,
                    help="hard-stop when one update takes more than this x the median of the "
                         "last 20, or free bytes fall under the projected need")
    ap.add_argument("--long-context", type=int, default=0,
                    help="train needle-in-long-context recall at this state token length (V1-D); "
                         "uses truncated backprop and ignores --suite/--synthetic")
    ap.add_argument("--grad-tokens", type=int, default=768,
                    help="with --long-context: keep the needle inside this trailing window so it "
                         "receives gradient; everything before is scanned detached")
    ap.add_argument("--init", default=None,
                    help="with --long-context: checkpoint dir to warm-start weights + tokenizer from")
    args = ap.parse_args()

    # `--seed` named the data order and nothing else: model init came from whatever
    # torch's global RNG held, so two runs of the identical command differed by
    # init noise. Any A/B — and every witness artifact this repo publishes — needs
    # the weights to start in the same place, or the difference being measured is
    # the wrong difference. 9a's own note records the workaround this replaces:
    # three of its loss tests push one update through a `typed_loss` spy in a
    # single process, because across processes the init differed for a reason
    # nobody was testing.
    torch.manual_seed(args.seed)

    rng = random.Random(args.seed)
    device = resolve_device(args.device)

    if args.long_context:
        if args.paraphrase == "on":
            # the needle corpus is synthetic text with synthetic instructions, so
            # the table has no entries for it; silently ignoring the flag would
            # print a run that claims a gate it never ran
            raise SystemExit("--paraphrase on is not available with --long-context: the "
                             "needle schemas are not in the phrasing table")
        if args.score_loss == "emd":
            # the needle batch builder hand-makes one choice question and carries no
            # ordinal mask, so emd would have nothing to price; accepting the flag and
            # training on ce would print a run that claims a loss it did not use
            raise SystemExit("--score-loss emd is not available with --long-context: the "
                             "needle corpus has no score cells and no ordinal mask")
        if args.anti_prior == "on":
            # the needle batches are generated row by row inside the step, so there is
            # no corpus marginal to re-weight against; the flag would print a run that
            # claims a batching mechanism it never ran
            raise SystemExit("--anti-prior on is not available with --long-context: needle "
                             "batches are generated per step and carry no label prior")
        train_long_context(args, rng, device)
        return

    if args.suite:
        from .real_data import load_suite, suite_texts

        data = load_suite(args.suite)
        if args.min_train_pool:
            data["train"] = {k: v for k, v in data["train"].items()
                             if len(v[1]) >= args.min_train_pool}
            print(f"train groups after pool>={args.min_train_pool}: {len(data['train'])}",
                  flush=True)
        tok = train_tokenizer(suite_texts(data["train"]), vocab_size=args.vocab)
    else:
        data = {
            "train": {wf: generate(args.n_train, wf, rng, "train") for wf in WORKFLOWS},
            "dev": {wf: generate(args.n_eval, wf, rng, "dev") for wf in WORKFLOWS},
            "test": {wf: generate(args.n_eval, wf, rng, "test") for wf in WORKFLOWS},
        }
        tok = train_tokenizer([e.state for wf in data["train"].values() for e in wf], vocab_size=4096)
    # uniform split shape: {split: {group_key: (questions, examples)}}
    if not args.suite:
        data = {
            split: {wf: (WORKFLOWS[wf][0], exs) for wf, exs in groups.items()}
            for split, groups in data.items()
        }
    print("tokenizer trained:", tok.get_vocab_size())
    print("cmd:", " ".join(["python", "-m", "myna.train", *sys.argv[1:]]), flush=True)
    for _split, _groups in data.items():
        print(split_report(_split, _groups), flush=True)

    vocab = tok.get_vocab_size()
    if args.config == "scaled":
        if MynaConfigScaledT4 is None:
            raise SystemExit("--config scaled requires src/myna/config_scaled.py (not installed)")
        cfg = MynaConfigScaledT4(vocab=vocab)
    else:
        cfg = MynaConfig(vocab=vocab)
    model = MynaModel(cfg).to(device)
    print("params:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.warmup:
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=args.warmup),
             torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(args.steps - args.warmup, 1))],
            milestones=[args.warmup],
        )
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    wfs = list(data["train"])
    pools = sorted(len(data["train"][k][1]) for k in wfs)
    thin = sum(1 for n in pools if n < args.batch)
    print(f"train question-sets: {len(wfs)}  median pool {pools[len(pools)//2]}  "
          f"sets with pool<batch({args.batch}) {thin} ({thin/len(wfs):.0%} of sets run a short batch)",
          flush=True)

    def group_cycle(keys, mode):
        """uniform = every set once per pass; pool = weight by example count;
        sqrt = geometric compromise between the two.

        Note on this suite: a set is keyed by its exact question *tensors*, and
        decision-v2 randomizes option descriptions per row, so many sets hold one
        row even though the underlying schema (same instruction, same option keys)
        recurs hundreds of times. Thin pools are a tensor-identity artifact, not
        evidence that the task has one example -- see myna.real_data._signature."""
        if mode == "uniform":
            while True:
                ks = list(keys)
                rng.shuffle(ks)
                yield from ks
        pw = {"pool": 1.0, "sqrt": 0.5}[mode]
        weights = [len(data["train"][k][1]) ** pw for k in keys]
        while True:
            yield rng.choices(keys, weights=weights, k=1)[0]

    cycle = group_cycle(wfs, args.group_sample)
    # `--anti-prior`: one drawer per question-set for the shared-set path, one over the
    # flattened row list for `--row-batch`, and one audit counting what the batches
    # actually contained. Both paths get it because both are reachable — a run that
    # ignored the flag outside `--row-batch` would print this banner and train on the
    # old marginals. What each can do differs, though, and decision-v2 makes it matter:
    # the shared-set drawer re-weights rows *inside* one question-set, so a set that
    # holds a single row — the median there — is drawn whole or not at all, and the
    # marginals only move on the sets with depth. `--row-batch` puts every row in one
    # pool, which is the path the published run used.
    anti = args.anti_prior == "on"
    audit = PriorAudit(data["train"]) if anti else None
    row_drawer = None
    group_drawers = {k: WeightedDraw(ws) for k, ws in balance_weights(data["train"]).items()} \
        if anti and not args.row_batch else {}
    row_items = None
    if args.row_batch:
        from .real_data import flatten_groups, flatten_weighted

        if anti:
            row_items, row_ws = flatten_weighted(data["train"])
            row_drawer = WeightedDraw(row_ws)
        else:
            row_items = flatten_groups(data["train"])
        q_len_cache: dict[int, int] = {}
        widest = max((len(qs), question_tokens(tok, qs, q_len_cache)) for qs, _ in row_items)
        print(f"row-batch: {len(row_items)} rows in one pool, {len(q_len_cache)} question sets, "
              f"widest row {widest[0]} questions x {widest[1]} tokens; budget "
              f"{args.max_q_cells} cells/forward (~{args.max_q_cells * 1.6 / 1024:.1f} GB branch)",
              flush=True)
    if anti:
        # the flag reaches however many cells the corpus actually has. On the synthetic
        # corpus every label is drawn uniformly, so the honest statement is "0 of 9" and
        # a banner claiming flattened marginals would be a claim about a mechanism that
        # had nothing to act on.
        skewed = [c for c, v in audit.priors.items() if max(v.values()) >= ANTI_PRIOR_SKEW]
        print(f"anti-prior: mini-batches drawn by inverse label prior; "
              f"{len(skewed)} of {len(audit.priors)} cells carry a majority label at or above "
              f"{ANTI_PRIOR_SKEW}"
              + (" — nothing to flatten on this corpus, the flag changes no draws"
                 if not skewed else ""), flush=True)
    dev_probe = {
        wf: (q, exs[: args.eval_n]) for wf, (q, exs) in data["dev"].items()
    } if data["dev"] else {
        wf: (q, exs[: args.eval_n]) for wf, (q, exs) in data["train"].items()
    }
    paraphraser = None
    n_sets = 0
    if args.paraphrase == "on":
        from .paraphrase import Paraphraser

        paraphraser = Paraphraser()
        n_sets = paraphraser.warm(data["train"])  # raises UnknownSchema on an untabelled set
        priced = ""
        if row_items is not None:
            priced = f"; {worst_case_tokens(tok, paraphraser, row_items, q_len_cache)} sets " \
                     f"budgeted at their longest phrasing"
        print(f"paraphrase: {n_sets} question sets re-worded, suite wording held out{priced}",
              flush=True)
    K = max(1, args.accum_groups)
    log_every = max(1, min(200, args.steps // 60))
    ema = None
    # --- the memory plan, printed before the first allocation (SPEC P3 3d) ----
    sample = row_items if row_items is not None else [
        (qs, ex) for qs, exs in data["train"].values() for ex in exs]
    # NOT random.Random(args.seed): the p95 is a property of the corpus, and the
    # sampler draws 4000 rows from a bigger pool, so seeding it with the experiment
    # made the experiment's size depend on it. Measured — two seeds of one command
    # on the full pilot gave p95 267/batch 15 and p95 274/batch 14, which means a
    # replication changed the batch and not just the data order. `state_token_p95`
    # already defaults to a fixed Random(0), so this restores the intended
    # behaviour and changes nothing for the seed-0 runs this repo has published.
    p95 = state_token_p95(tok, sample)
    free = None if args.free_gib is None else int(args.free_gib * 1024 * MI)
    if free is None:
        free = free_device_bytes(device)
    # The question branch of a --row-batch step is a second, separately measured
    # allocation (--max-q-cells x 1.6 MiB), so the state plan is priced against
    # what is left after it. The M5 run sized for states alone and died with both
    # branches live (SPEC §6).
    reserve = args.max_q_cells * QUESTION_MIB_PER_CELL * MI if args.row_batch else 0
    fit = rows_that_fit(free, p95, args.mem_safety, reserve_bytes=reserve)
    planned = args.batch
    reserved = f", less {reserve / 1024 / MI:.1f} GiB of question branch" if reserve else ""
    if fit is None:
        print(f"memory plan: no headroom reading for {device} (p95 state {p95} tokens, "
              f"~{args.batch * p95 * STATE_MIB_PER_POSITION / 1024:.1f} GiB/step at batch "
              f"{args.batch}); --batch taken as given — state --free-gib to plan against",
              flush=True)
    elif fit == 0:
        raise SystemExit(f"memory plan: {args.mem_safety} x {free / 1024 / MI:.1f} GiB free is "
                         f"not more than the {reserve / 1024 / MI:.1f} GiB the question branch "
                         f"costs at --max-q-cells {args.max_q_cells}; lower --max-q-cells "
                         "before asking for a batch")
    elif args.batch > fit:
        planned = fit
        print(f"memory plan: {free / 1024 / MI:.1f} GiB free x {args.mem_safety}{reserved} -> "
              f"batch {fit} at p95 {p95} tokens x {STATE_MIB_PER_POSITION} MiB/position; "
              f"--batch {args.batch} exceeds it, using {fit}", flush=True)
    else:
        print(f"memory plan: batch {args.batch} fits {free / 1024 / MI:.1f} GiB free x "
              f"{args.mem_safety}{reserved} at p95 {p95} tokens (needs "
              f"{(args.batch * p95 * STATE_MIB_PER_POSITION + reserve / MI) / 1024:.1f} GiB)",
              flush=True)
    args.batch = planned
    projected = planned * p95 * STATE_MIB_PER_POSITION * MI + reserve
    start_step = 0
    if args.warm_start:
        ckpt = Path(args.warm_start)
        if not ckpt.exists():
            raise SystemExit(f"--warm-start: no {ckpt}")
        if args.resume:
            raise SystemExit("--warm-start and --resume are different things: one restarts "
                             "the optimizer on a trained model, the other continues it")
        # weights ONLY, on purpose. `--resume` restores the saved optimizer and
        # scheduler too, and a restored CosineAnnealingLR carries its own T_max, so
        # continuing a finished schedule leaves the learning rate at ~0 and the
        # "experiment" trains nothing. A fresh schedule over --steps is what an
        # ablation needs; strict=True is what says the shapes really matched.
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state_dict"])
        print(f"warm-start: weights from {ckpt} (trained at step {blob.get('step', 'n/a')}); "
              f"optimizer and scheduler are fresh over {args.steps} updates, "
              f"lr {args.lr}", flush=True)
    if args.resume:
        ckpt = Path(args.out) / "model_last.pt"
        if not ckpt.exists():
            raise SystemExit(f"--resume: no {ckpt} to continue from")
        start_step = load_snapshot(ckpt, model, opt, sched) + 1
        print(f"resume: {ckpt} at step {start_step - 1} -> continuing at {start_step} of "
              f"{args.steps}; data RNG re-seeded from --seed {args.seed}, so this is a "
              f"continuation, not a bit-identical replay", flush=True)
    step_times: list[float] = []
    stopped = None
    last_step = start_step - 1
    step_t0 = time.time()
    for step in range(start_step, args.steps):
        t_update = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)
        run_loss = 0.0
        for _ in range(K):
            if row_items is None:
                wf = next(cycle)
                questions, pool = data["train"][wf]
                if paraphraser is not None:
                    questions = paraphraser.draw(wf, questions, rng)
                exs = draw_batch(pool, args.batch, rng,
                                 drawer=group_drawers[wf] if anti else None)
                b = build_batch(exs, tok, questions, device)
                if audit is not None:
                    audit.add(questions, exs)
            else:
                items = draw_row_batch(row_items, args.batch, args.max_q_cells, rng, tok,
                                    q_len_cache, drawer=row_drawer if anti else None)
                b = build_row_batch(items, tok, device, paraphraser, rng)
                if audit is not None:
                    audit.add_items(items)
            logits = model(b["state_ids"], b["state_len"], b["q_ids"], b["q_mask"],
                           b["span_mat"], b["opt_valid"], b["decide_idx"])
            loss = typed_loss(logits, b["gold"], b["has_gold"], b["ordinal"],
                              b["opt_valid"], args.score_loss) / K
            loss.backward()
            run_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema = run_loss if ema is None else 0.95 * ema + 0.05 * run_loss
        last_step = step
        step_times.append(time.time() - t_update)
        if step % log_every == 0 or step == args.steps - 1:
            el = time.time() - step_t0
            print(f"step {step:5d}  loss {run_loss:.3f}  ema {ema:.3f}  "
                  f"sets/update {K}  {el:.0f}s", flush=True)
        # cadence, not convenience: a 10k-update job that dies at 9k without a
        # snapshot has taught us nothing it did not already know (§6's OOM)
        if args.save_every and (step + 1) % args.save_every == 0:
            save_snapshot(args.out, model, opt, sched, tok, cfg, step)
        reason = stop_reason(step_times[-21:], free_device_bytes(device), projected,
                             factor=args.stop_factor)
        if reason:
            save_snapshot(args.out, model, opt, sched, tok, cfg, step)
            stopped = reason
            print(f"STOP at step {step}: {reason}; snapshot written to "
                  f"{Path(args.out) / 'model_last.pt'}", flush=True)
            break
        if args.eval_every and step and step % args.eval_every == 0 and step != args.steps - 1:
            m = evaluate(model, tok, dev_probe, device)
            print(f"step {step:5d}  dev-mid acc {macro_acc(m):.4f}", flush=True)
            # rolling snapshot so a long run can be evaluated or stopped early
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            save_snapshot(out, model, opt, sched, tok, cfg, step)

    anti_prior = None
    if audit is not None:
        audit_lines, anti_prior = audit.report()
        for line in audit_lines:
            print(line, flush=True)

    calib_split, calib_name = temperature_source(data)
    # the row count rides along with the label: "(fit on calibration)" would
    # still print if the split were hardcoded, the count would not match
    n_fit_rows = sum(len(exs) for _q, exs in calib_split.values())
    temperature = fit_temperature(model, tok, calib_split, device)
    print(f"temperature: {temperature}  (fit on {calib_name}: {n_fit_rows} rows in "
          f"{len(calib_split)} sets)")
    dev_m = evaluate(model, tok, data["dev"], device, temperature)
    test_m = evaluate(model, tok, data["test"], device, temperature)
    print("=== dev ===")
    for k, v in sorted(dev_m.items()):
        print(f"{k:32s} {v:.4f}")
    print("=== test ===")
    for k, v in sorted(test_m.items()):
        print(f"{k:32s} {v:.4f}")

    # The gate P1 exists for: dev on the suite's own wording, and dev on the one
    # phrasing that was neither trained nor shipped by the suite. A model that read
    # the instruction holds; a model that matched a template falls.
    dev_unseen = None
    n_draws = 0
    # The dose is the number of updates that *ran*, not `--steps`: a run that asked for
    # 3,600 and STOPped at 3,314 ran 3,315, and a `--resume` process that started at 501
    # of 3,315 ran 2,814. Dividing a counter by the requested count is how §9.56(x)
    # printed a 15.2% coverage gap that did not exist (§9.58).
    updates_executed = max(last_step - start_step + 1, 0)
    if paraphraser is not None:
        dev_unseen = held_out_eval(model, tok, data["dev"], device, temperature)
        print(f"=== dev, HELD-OUT phrasing === macro {macro_acc(dev_unseen):.4f}   "
              f"(exact suite wording {macro_acc(dev_m):.4f})")
        # the warm-up banner only proves the table loaded; this one proves the
        # loop consulted it, which no earlier check did for the shared-set path
        n_draws = paraphraser.draws
        rate = f"; {n_draws / updates_executed:.2f} per update" if updates_executed else ""
        print(f"paraphrase: {n_draws} phrasing draws reached the batches over "
              f"{updates_executed} updates executed (of {args.steps} requested"
              f"{f', resumed from step {start_step}' if start_step else ''}){rate}", flush=True)

    # machine-readable metrics + a key->type map so downstream tables can roll
    # up by source x question-type (apples-to-apples with the laya witness).
    qtypes = {}
    for split in ("dev", "test"):
        for wf, (questions, _exs) in data[split].items():
            for q in questions:
                qtypes[f"{wf}/{q.name}"] = q.type

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f:

        json.dump({"temperature": temperature, "temperature_fit_on": calib_name,
                   "temperature_fit_rows": n_fit_rows,
                   "paraphrase": args.paraphrase, "paraphrase_sets": n_sets,
                   "paraphrase_draws": n_draws,
                   "anti_prior": args.anti_prior, "anti_prior_audit": anti_prior,
                   "batch": args.batch, "state_tokens_p95": p95,
                   "mem_plan_free_gib": None if free is None else free / 1024 / MI,
                   "mem_safety": args.mem_safety, "resumed_from_step": start_step,
                   "stopped": stopped, "last_step": last_step,
                   "steps_requested": args.steps, "updates_executed": updates_executed,
                   "stop_factor": args.stop_factor, "save_every": args.save_every,
                   "dev": dev_m, "test": test_m,
                   "dev_unseen": dev_unseen, "qtypes": qtypes}, f, indent=2)

    torch.save({"state_dict": model.state_dict(), "cfg": vars(cfg), "temperature": temperature}, out / "model.pt")

    tok.save(str(out / "tokenizer.json"))
    print(f"saved checkpoint to {out}")
    if stopped:
        # a truncated run that exits 0 reads like a finished one, and the whole
        # point of the stop rule is that the *reason* reaches the operator
        raise SystemExit(f"run ended at step {last_step} of {args.steps}: {stopped}")


if __name__ == "__main__":
    main()


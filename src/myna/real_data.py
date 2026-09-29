"""Adapter: real labelled decision suites (JSONL) -> myna training shapes.

One line per decision request, in the laya/kev API shape the frozen suites
ship:

    {"state": "...", "questions": {qid: {"type": "choice"|"noul"|"score",
      "instructions": "...", "criteria": {opt: desc|null} | [legend] | null,
      "label": key | bool | int}}, "_meta": {...}}

Rows that share a question set are grouped by question-set signature, so a
group behaves like a synthetic workflow and one shared tensor set serves its
whole batch. Rows whose instructions differ from each other (boolq, mnli) each
form a group of one; `flatten_groups` hands those rows to the per-row batch
contract instead. Labels become option indices: choice -> position of the
label key among criteria keys, noul -> 0/1 on ["no", "yes"], score -> the
level index. Options with no description fall back to the key humanized.

`balance_weights` / `flatten_weighted` sit beside the loaders because the thing
they measure — the gold-label prior of a (source, question) cell — is a property
of these rows, not of the sampler that reads them.
"""

from __future__ import annotations

import hashlib
import json
from math import exp, log
from pathlib import Path

from .data import Example, Question

#: A cell whose most common label carries at least this share of its rows has a
#: prior worth flattening; below it, re-weighting is a rounding error. Measured on
#: decision-v2/train: the six cells above it are agnews' four `noul` questions
#: (0.739-0.755) plus boolq (0.624) and yelp/recommend (0.605).
ANTI_PRIOR_SKEW = 0.55


def source_of_group(key):
    """The group-key contract. `load_split` writes `f"{source}#{sig}"`, so the
    source is everything before the first `#` — and `myna.report` parses the same
    key to name its cells, which is why the rule lives here rather than twice."""
    return key.split("#", 1)[0]


def _option_text(key, desc):
    return desc if isinstance(desc, str) and desc.strip() else str(key).replace("_", " ").strip()


def _state_text(state):
    """Suites ship states as str, {"document": ...} dicts, or segment lists;
    myna scans plain text, so flatten recursively in insertion order."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return " ".join(_state_text(v) for v in state.values())
    if isinstance(state, (list, tuple)):
        return " ".join(_state_text(v) for v in state)
    raise ValueError(f"unsupported state type {type(state).__name__}")


def parse_questions(row):
    """-> (list[Question], gold tuple). Skips rows with labels outside criteria."""
    qs, golds = [], []
    for qid, q in row["questions"].items():
        t, lab, instr = q["type"], q["label"], _state_text(q["instructions"])
        if t == "noul":
            qs.append(Question(qid, "noul", instr, ["no", "yes"]))
            golds.append(1 if lab else 0)
        elif t == "score":
            opts = [_state_text(o) for o in q["criteria"]]
            if not (0 <= int(lab) < len(opts)):
                raise ValueError(f"{qid}: score label {lab} outside legend of {len(opts)}")
            qs.append(Question(qid, "score", instr, opts))
            golds.append(int(lab))
        elif t == "choice":
            crit = q["criteria"]
            if lab not in crit:
                raise ValueError(f"{qid}: choice label {lab!r} not in criteria")
            qs.append(Question(qid, "choice", instr, [_option_text(k, v) for k, v in crit.items()]))
            golds.append(list(crit).index(lab))
        else:
            raise ValueError(f"unknown question type {t!r}")
    return qs, tuple(golds)


def _signature(questions_json):
    blob = json.dumps(questions_json, sort_keys=False, ensure_ascii=False)
    return hashlib.md5(blob.encode()).hexdigest()[:8]


def load_split(path):
    """JSONL file -> {group_key: (questions, examples)}, source in the key."""
    groups = {}
    n_skipped = 0
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                qs, gold = parse_questions(row)
            except ValueError:
                n_skipped += 1
                continue
            sig = _signature({qid: {k: v for k, v in q.items() if k != "label"}
                              for qid, q in row["questions"].items()})
            source = (row.get("_meta") or {}).get("source", "suite")
            key = f"{source}#{sig}"
            questions, examples = groups.setdefault(key, (qs, []))
            examples.append(Example(_state_text(row["state"]), key, gold))
    if n_skipped:
        print(f"note: skipped {n_skipped}/{n_skipped + sum(len(e) for _, e in groups.values())} rows in {path}")
    return groups


def load_suite(root, train="train.jsonl", dev="development.jsonl", test="test.jsonl",
               calibration="calibration.jsonl"):
    """A frozen-suite directory -> {"train"|"dev"|"test"|"calibration": groups}.
    Splits missing on disk map to {} (e.g. eval-only suites).

    `calibration` is the suite's own labelled split for fitting the temperature
    scalar. Fitting it on dev instead — which we did through v0 — uses the same
    rows to pick the scalar and to report calibration on, so it quietly
    flatters the dev number."""
    root = Path(root)
    out = {}
    for name, fname in (("train", train), ("dev", dev), ("test", test), ("calibration", calibration)):
        p = root / fname
        out[name] = load_split(p) if p.exists() else {}
    return out


def flatten_groups(groups):
    """{group_key: (questions, examples)} -> [(questions, Example), ...] one entry
    per row.

    Lossless: a group is keyed on the *exact* question JSON, so every row in it
    genuinely shares its question set. Flattening therefore loses nothing while
    giving the per-row batch contract (P2) a source-blind row list to draw from
    -- which is what boolq and mnli need, their 300 singleton groups each being
    one row deep."""
    return [(q, ex) for q, exs in groups.values() for ex in exs]


def _prod(ts):
    w = 1.0
    for t in ts:
        w *= t
    return w


#: How a row's per-slot inverse frequencies become one weight. Every one of these
#: was drawn against the shipped pilot by `bench/anti_prior_audit.py --compare`,
#: which is what picks `prod` (see `balance_weights`); the rest are kept so that
#: comparison stays runnable through the same function the trainer calls, rather
#: than through a copy in a scratch file that could drift from it.
COMBINE = {
    "prod": _prod,
    "mean": lambda ts: sum(ts) / len(ts),
    "sum": lambda ts: float(sum(ts)),
    "max": max,
    "geo": lambda ts: exp(sum(log(t) for t in ts) / len(ts)),
}


def balance_weights(groups, combine="prod"):
    """{group_key: (questions, examples)} -> {group_key: [one weight per example]}.

    `combine` is how a row's per-slot inverse frequencies collapse to one number; it
    defaults to and ships as "prod".

    The label prior is the thing a model learns when the state does not decide the
    answer: it answers the cell's most frequent label and collects that share as
    accuracy. Tier 0 measured it on the trained checkpoint — three agnews `noul`
    cells emit one label on *every* row and finish exactly at their own majority
    floor (0.653 / 0.809 / 0.778), and those floors come from a train prior of
    0.739-0.755. So the shortcut is the most rewarded output the training data
    offers. Flattening that marginal makes the constant answer worth 0.5 rather
    than ~0.75, which is the only way the training signal can say "read instead".

    A row's raw weight is the **product** over its questions of `1 / share` of its
    gold label in that cell — equivalently `exp(-logP(gold labels | cell priors))`,
    the reciprocal of how predictable the row's whole answer set is from the priors
    alone. Rows that defy the prior are the ones drawn more often.

    The product was chosen by measurement, not by taste
    (`bench/anti_prior_audit.py`, 3,600 updates x 8 sets x batch 10 on the shipped
    pilot train split, same seed per arm). The realized majority of agnews/is_business
    after the draws: natural 0.7545, mean-of-inverses 0.7112, max-of-inverses 0.7483,
    geometric mean 0.6997, **product 0.6184**. Mean and sum agree to the digit because
    a source's rows all carry the same slot count, so their ratio is a constant the
    per-source rescale below removes anyway. What separates them is that a *marginal*
    is a joint object: an agnews row answers three questions at once, and only the
    product prices all three of them together.

    Two things about this shape are load-bearing:

    * The cell is `(source, question name)`, pooled across every question-set of
      that source — not the tensor group. decision-v2 randomizes option descriptions
      per row, so nearly all of its 17,112 groups hold one row and a within-group
      histogram is flat by construction: the prior only exists at the cell level (see
      `myna.train.group_cycle`'s note on `_signature`).
    * Weights are then rescaled so each *source* has mean weight 1. Inverse frequency
      alone also scales by how many labels a cell uses — a 77-intent cell has mean
      weight ~77 and a binary one ~2 — so a naive draw would send tens of times more
      gradient to banking77 than to imdb. That is a source-mix change wearing a
      label-balancing name; this rescale keeps each source the expectation a uniform
      draw gives it, so the only thing that moves is the label histogram inside it.
      Checked on the shipped rows by `bench/anti_prior_audit.py`: `sum(w) == n_rows`
      per source to a relative deviation of 0.00e+00.
    """
    priors = cell_priors(groups)
    if combine not in COMBINE:
        raise SystemExit(f"balance_weights: unknown combine {combine!r}; "
                         f"choose among {', '.join(sorted(COMBINE))}")
    f = COMBINE[combine]
    raw: dict[str, list[float]] = {}
    per_source: dict[str, list[float]] = {}
    for key, (questions, examples) in groups.items():
        source = source_of_group(key)
        cells = [priors.get((source, q.name), {}) for q in questions]
        ws = []
        for ex in examples:
            # a row can carry fewer golds than its set has questions: those slots are
            # unlabelled for this row and stay out of the terms rather than entering as
            # a factor of 1.0. For the shipped product that is a no-op — 1.0 is its own
            # identity — but for the mean-style rules it would price a short row as if
            # it answered every question it was shown. A row with no labelled slot at
            # all gets weight 1.0: it is not rarer, it is just unpriceable.
            terms = [1.0 / cells[i][ex.gold[i]]
                     for i in range(min(len(cells), len(ex.gold))) if cells[i].get(ex.gold[i])]
            ws.append(f(terms) if terms else 1.0)
        raw[key] = ws
        per_source.setdefault(source, []).extend(ws)
    means = {s: (sum(ws) / len(ws) if ws else 1.0) for s, ws in per_source.items()}
    return {key: [w / means[source_of_group(key)] for w in ws] for key, ws in raw.items()}


def flatten_weighted(groups, combine="prod"):
    """-> (items, weights) aligned row-for-row with `flatten_groups(groups)`.

    One call site produces both, because the alignment is the contract: `weights[i]`
    must belong to `items[i]` or the sampler re-weights the wrong rows. Returning
    them together is what makes passing one without the other impossible."""
    ws = balance_weights(groups, combine)
    items, weights = [], []
    for key, (questions, examples) in groups.items():
        for j, ex in enumerate(examples):
            items.append((questions, ex))
            weights.append(ws[key][j])
    return items, weights


def cell_priors(groups):
    """{(source, qname): {label_index: share}} — the natural label marginal of
    every cell in the split, read off the rows.

    One cell per (source, question), which is the unit G1 is scored in, so the
    prior that anti-prior batching flattens and the majority floor that decides
    whether a cell beat chance are two readings of the same aggregation.
    `balance_weights` uses it for the same reason."""
    from .report import cell_stats  # lazy: report imports this module at top level

    return {(src, name): {lab: n / c["n"] for lab, n in c["labels"].items()}
            for (src, name), c in cell_stats(groups, with_labels=True).items()
            if c["n"]}


def suite_texts(groups):
    """Corpus for tokenizer training: states plus every instruction/option
    string, so option spans never fragment at eval time."""
    texts = []
    for questions, examples in groups.values():
        texts += [e.state for e in examples]
        for q in questions:
            texts.append(q.instruction)
            texts += q.options
    return texts

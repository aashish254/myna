"""Bundle consistency: the released contradiction predicate.

The paper reports cross-question consistency as an *observed* property of each engine,
never as an enforced constraint. This module is the definition that makes the
observation reproducible. It is deliberately small: a predicate over two questions'
answers, applied only to the bundles that ask both.

A bundle is *inconsistent* iff one of the following holds on the same state:

* ``agnews`` — the predicted topic is class ``c`` and the bundle also asks ``is_c``,
  and the engine answers that boolean with "no".  A text cannot be about sports and
  simultaneously not be about sports.
* ``yelp`` — the predicted star rating is 4--5 while ``recommend`` is "no", or the
  rating is 1--2 while ``recommend`` is "yes".  A reviewer who recommends the business
  does not give it one star.

The predicate reads answers in a normalized form: booleans become "yes"/"no", a score
prediction becomes its 1-based level (an ``int``), and a choice prediction stays in its
own label space.  `normalize_answer` performs that conversion per question type, and
`answers_from_row` accepts the two record shapes the repository's row-outcome witnesses
use: 1-based / 0-based indices (Myna) and named labels (Laya, VegaML).

Why the predicate is shipped as code rather than described in prose: the campaign
record for the same checkpoint says 0/172, this predicate says 3/172, and a reader can
only adjudicate that difference by running the definition. See
``bench/consistency_audit.py``.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

#: The four AG News classes, in the order the suite's criteria dict lists them.
AGNEWS_CLASSES = ("world", "sports", "business", "scitech")

#: Question name -> class name, for the four boolean side-questions.
AGNEWS_BOOL = {f"is_{c}": c for c in AGNEWS_CLASSES}


def normalize_answer(qtype: str, pred: Any) -> str:
    """Map one engine's prediction onto the predicate's label space.

    Parameters
    ----------
    qtype:
        ``"choice"``, ``"score"``, or ``"noul"``.
    pred:
        The engine's prediction as it appears in a row-outcome record: a label, an
        integer index, or a boolean.

    Returns
    -------
    str
        ``"yes"``/``"no"`` for ``noul``; the level as a decimal string for ``score``;
        the label itself for ``choice``.
    """
    if qtype == "noul":
        if isinstance(pred, str):
            return pred.strip().lower()
        return "yes" if pred else "no"
    if qtype == "score":
        return str(int(pred))
    return str(pred)


def answers_from_row(row: Mapping[str, Any], spec: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Normalize every answer of one bundle.

    ``spec`` (the bundle's question spec from ``data/decision-v2-pilot``) is required only
    when an engine records a ``choice`` as an index into the option list, as Myna's
    row-outcome witness does. Named-label records (Laya, VegaML) need no spec.
    """
    out: dict[str, str] = {}
    for qname, q in row["questions"].items():
        pred = q["pred"]
        if q["type"] == "choice" and isinstance(pred, int) and spec is not None:
            crit = spec[qname].get("criteria", {})
            labels = list(crit) if isinstance(crit, dict) else [str(c) for c in crit]
            pred = labels[pred]
        out[qname] = normalize_answer(q["type"], pred)
    return out


def is_contradiction(source: str, answers: Mapping[str, str]) -> bool:
    """Apply the predicate to one normalized bundle. Returns ``True`` on a contradiction."""
    if source == "agnews":
        topic = answers.get("topic")
        for qname, cls in AGNEWS_BOOL.items():
            if topic == cls and answers.get(qname) == "no":
                return True
        return False
    if source == "yelp":
        if "rating" not in answers or "recommend" not in answers:
            return False
        level = int(answers["rating"])
        rec = answers["recommend"]
        return (level >= 4 and rec == "no") or (level <= 2 and rec == "yes")
    return False


def contradictions(rows: Iterable[Mapping[str, Any]],
                   specs: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Count contradictions over the multi-question bundles of a split.

    Parameters
    ----------
    rows:
        Row-outcome records (each with ``meta``, ``questions``, and ``group``).
    specs:
        Optional ``group id -> question spec`` map, needed when choice answers are
        indices. Bundles whose source is neither ``agnews`` nor ``yelp`` are counted as
        ineligible, not as consistent.
    """
    specs = specs or {}
    n = 0
    bad: list[dict[str, str]] = []
    by_source: dict[str, dict[str, int]] = {}
    for row in rows:
        source = row["meta"]["source"]
        if source not in ("agnews", "yelp"):
            continue
        if len(row["questions"]) < 2:
            continue
        # Keyed by row id, not group id: two rows of the same group may order their
        # criteria dict differently, and a `choice` answer is an index into that order.
        rid = row["meta"].get("id")
        answers = answers_from_row(row, specs.get(rid))
        n += 1
        hit = is_contradiction(source, answers)
        rec = by_source.setdefault(source, {"bundles": 0, "contradictions": 0})
        rec["bundles"] += 1
        if hit:
            rec["contradictions"] += 1
            bad.append({"id": row["meta"]["id"], "source": source,
                        "answers": "; ".join(f"{k}={v}" for k, v in sorted(answers.items()))})
    return {"bundles": n, "contradictions": len(bad), "rate": len(bad) / n if n else 0.0,
            "by_source": by_source, "cases": bad}


# --------------------------------------------------------------------------------------
# The campaign predicate, recovered from the released notebook
# --------------------------------------------------------------------------------------
# `release/notebooks/01-v5-campaign-full.ipynb` (cell "GRAPH ABLATION") counts a
# contradiction only when a bundle carries both a `choice` and a `noul` question, the
# predicted option text for the choice appears inside the boolean question's instruction,
# and the boolean is answered "no". It therefore cannot see a yelp rating/recommendation
# contradiction (rating is a `score`, not a `choice`), and it misses any contradiction
# whose option text is not a substring of the question. The published 0/172 record came
# from this rule; the released predicate above is broader. Both are reported.

def _choice_options(spec: dict) -> list[str]:
    from .real_data import _option_text, _state_text

    crit = spec["criteria"]
    if isinstance(crit, dict):
        return [_option_text(k, v) for k, v in crit.items()]
    return [_state_text(c) for c in crit]


def campaign_contradiction(row: Mapping[str, Any], spec: Mapping[str, Any]) -> bool:
    """The 2026 campaign's contradiction rule, as recovered from the notebook.

    ``row`` is a row-outcome record and ``spec`` the bundle's question spec. Choice
    predictions are accepted as either an index or a label, so the Myna, Laya, and VegaML
    record shapes all work.
    """
    choice = None
    noul = None
    for qname, rec in row["questions"].items():
        s = spec[qname]
        if s["type"] == "choice":
            opts = _choice_options(s)
            pred = rec["pred"]
            idx = int(pred) if isinstance(pred, int) else (
                opts.index(pred) if str(pred) in opts else -1)
            if 0 <= idx < len(opts):
                choice = (opts[idx], str(s.get("instructions", "")))
        elif s["type"] == "noul":
            pred = rec["pred"]
            answered_no = (pred is False) or (pred == 0) or (str(pred).lower() == "no")
            noul = (answered_no, str(s.get("instructions", "")))
    if not choice or not noul:
        return False
    option_text, _ = choice
    answered_no, instr = noul
    return bool(answered_no and option_text.lower() in instr.lower())


def campaign_contradictions(rows, specs) -> dict[str, Any]:
    """Count contradictions under the campaign's rule, on the same eligible bundles."""
    n = 0
    bad = 0
    for row in rows:
        if row["meta"]["source"] not in ("agnews", "yelp"):
            continue
        if len(row["questions"]) < 2:
            continue
        spec = specs.get(row["meta"]["id"])
        if spec is None:
            continue
        n += 1
        bad += int(campaign_contradiction(row, spec))
    return {"bundles": n, "contradictions": bad,
            "rate": bad / n if n else 0.0}

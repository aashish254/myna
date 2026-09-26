"""Adapter: real labelled decision suites (JSONL) -> myna training shapes.

One line per decision request, in the laya/kev API shape the frozen suites
ship:

    {"state": "...", "questions": {qid: {"type": "choice"|"noul"|"score",
      "instructions": "...", "criteria": {opt: desc|null} | [legend] | null,
      "label": key | bool | int}}, "_meta": {...}}

myna's batch contract is one shared question tensor set per batch (v0), so
rows are grouped by question-set signature and each group behaves like a
synthetic workflow. Labels become option indices: choice -> position of the
label key among criteria keys, noul -> 0/1 on ["no", "yes"], score -> the
level index. Options with no description fall back to the key humanized.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .data import Example, Question


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


def load_suite(root, train="train.jsonl", dev="development.jsonl", test="test.jsonl"):
    """A frozen-suite directory -> {"train"|"dev"|"test": groups}. Splits
    missing on disk map to {} (e.g. eval-only suites)."""
    root = Path(root)
    out = {}
    for name, fname in (("train", train), ("dev", dev), ("test", test)):
        p = root / fname
        out[name] = load_split(p) if p.exists() else {}
    return out


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

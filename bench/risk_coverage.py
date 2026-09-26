"""Risk/coverage curve over a labelled split — the artifact gate G5 is read from.

    .venv/bin/python -m bench.risk_coverage --ckpt runs/myna-v0 \
        --suite data/decision-v2-pilot --split calibration \
        --out runs/risk_coverage.md

Abstention is only a contribution if the confidence that drives it is *worth
something*, so this measures the trade directly: sort every answered question by
its committed-side probability, keep the top `coverage` fraction, and report the
accuracy on exactly those rows. A curve that stays flat as coverage falls means
the model knows which answers it does not know; a flat line at the model's overall
accuracy means the confidence is decoration.

Three rules the harness enforces on itself, because each is a way this table could
become marketing:

* **Confidence is the probability of the side committed to**, not of "yes", and
  not the engine's own `confidence` field taken on trust: it is recomputed here
  from the printed distribution (or from `noul`, whose other side is `1-p`). The
  curve and the engine must therefore disagree loudly if either reads a different
  quantity.
* **The selected threshold is implementable.** After picking the floor the script
  re-runs the engine *at that floor* and requires its abstention count to equal
  the count this curve withheld. A threshold that cannot be set on the engine is
  not a product feature, it is a plot.
* **Accuracy is measured against the gold indices the split carries**, in the same
  (source, question) cells `myna.report` uses, so no cell appears here that the
  accuracy tables would refuse.

Labelled at the top of the output: on a checkpoint that never trained on this
corpus the curve is a *harness witness* — it proves the measurement works — and
not a G5 pass. The gate needs the real-trained checkpoint (`KAGGLE`-gated).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from myna.engine import Myna
from myna.real_data import load_split

#: G5's pass condition (§2.2), as numbers rather than as prose.
G5_TARGET_ACC = 0.95
G5_MIN_COVERAGE = 0.60
G5_SOURCES = ("banking77", "dbpedia14", "trec")

#: coverage rungs the table prints, high to low.
COVERAGE_RUNGS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


def engine_questions(questions) -> dict:
    """myna's typed spec for one question-set, without the golds.

    `criteria` is the option-text list rather than the suite's key->description
    dict, so the answer labels and the gold indices line up position-for-position:
    the curve's `correct` is an index comparison, never a string match that could
    drift between the adapter and the engine."""
    return {q.name: {"type": q.type, "instructions": q.instruction, "criteria": q.options}
            for q in questions}


def committed_probability(ans: dict) -> float:
    """The probability of the side this answer commits to, recomputed from the
    distribution the answer prints.

    noul carries only `noul` = p(Yes); its committed side is whichever of
    `p` and `1-p` is larger. Reading `noul` as the confidence would put the
    model's surest "no"s at the bottom of the ranking — the exact rows a
    decision system should be answering first.

    The distribution is preferred over the engine's own `confidence` field on
    purpose, and it is checked against it in `collect()`: a curve that ranks on the
    number the harness derives and a gate that fires on the number the engine
    prints are two different products.
    """
    if "probabilities" in ans and ans["probabilities"]:
        return max(float(p) for p in ans["probabilities"].values())
    if ans.get("noul") is not None:
        p = float(ans["noul"])
        return max(p, 1.0 - p)
    if ans.get("confidence") is not None:
        return float(ans["confidence"])
    raise ValueError(f"answer carries no probability to rank on: {sorted(ans)}")


def is_correct(ans: dict, qtype: str, gold: int) -> bool:
    """Score one answer against its gold option index."""
    if qtype == "noul":
        return int(float(ans["noul"]) >= 0.5) == int(gold)
    probs = ans.get("probabilities")
    if not probs:
        raise ValueError(f"{qtype} answer carries no distribution to argmax")
    return max(range(len(probs)), key=lambda i: list(probs.values())[i]) == int(gold)


def collect(myna: Myna, groups: dict) -> list[dict]:
    """One row per (example, question) with its confidence, gold and correctness."""
    points: list[dict] = []
    for key, (questions, examples) in sorted(groups.items()):
        source = key.split("#", 1)[0]
        specs = engine_questions(questions)
        for ex in examples:
            answers = myna.observe(ex.state).ask(specs)["answers"]
            missing = set(specs) - set(answers)
            if missing:
                # a question the engine skipped is not an abstention, it is a hole
                # in the sample: the curve would silently rank the survivors.
                raise AssertionError(f"{key}: engine skipped {sorted(missing)}")
            for i, q in enumerate(questions):
                a = answers[q.name]
                conf = committed_probability(a)
                ec = a.get("confidence")
                if ec is not None and abs(float(ec) - conf) > 1e-6:
                    # The curve ranks on `conf` and the gate abstains on `ec`. They
                    # are read from different fields of the same answer, so a
                    # divergence here means the published floor is not the rule the
                    # engine runs — which is the whole G5 claim.
                    raise AssertionError(f"{key}/{q.name}: engine confidence {ec} != "
                                         f"top of its own distribution {conf}")
                points.append({"source": source, "question": q.name, "type": q.type,
                               "confidence": conf,
                               "correct": int(is_correct(a, q.type, ex.gold[i]))})
    return points


def curve(points: list[dict], rungs=COVERAGE_RUNGS) -> list[dict]:
    """Risk/coverage: keep the top-`c` fraction by confidence, score exactly those.

    `rung` is the requested coverage and `coverage` the realized one: a sample of
    568 questions cannot be cut at exactly 30%, and reporting the realized figure
    keeps the table honest about how many rows a cell is an average of. The
    `threshold` is the confidence of the last kept row — the floor that realizes
    that coverage, which is the number the engine is then set to.
    """
    ranked = sorted(points, key=lambda p: -p["confidence"])
    n = len(ranked)
    out = []
    for c in rungs:
        k = min(n, max(1, int(round(c * n)))) if n else 0
        kept = ranked[:k]
        acc = sum(p["correct"] for p in kept) / len(kept) if kept else None
        out.append({"rung": c, "coverage": len(kept) / n if n else 0.0, "n": len(kept),
                    "accuracy": acc, "risk": None if acc is None else 1.0 - acc,
                    "threshold": kept[-1]["confidence"] if kept else None})
    return out


def rung_at(rows: list[dict], c: float):
    return next((r for r in rows if r["rung"] == c), None)


def at_or_above(points: list[dict], floor: float) -> list[dict]:
    """The rows a floor keeps. Written as the negation of the engine's rule
    (`abstain` when `confidence < floor`) rather than as `>= floor`, so the curve,
    the per-source cut and the engine can never disagree about a row whose
    confidence *is* the floor — which is exactly the row the floor was taken from.
    """
    return [p for p in points if not p["confidence"] < floor]


def verify_floor(points: list[dict], floor: float, engine_abstained: int) -> dict:
    """Does the engine, set to this floor, abstain on exactly the rows the curve
    puts below it?

    The only witness that the published threshold is a *setting* and not a plot
    annotation: `n_below` is computed from the curve's own numbers, `engine_abstained`
    is counted from the per-answer flags of a second pass. They come from different
    code paths over the same vectors, so they disagree loudly if the two rules are
    not the same rule."""
    n_below = len(points) - len(at_or_above(points, floor))
    return {"floor": floor, "engine_abstained": engine_abstained, "curve_below_floor": n_below,
            "realized_coverage": (len(points) - n_below) / len(points) if points else None,
            "agree": engine_abstained == n_below}


def g5_verdict(rows: list[dict], target=G5_TARGET_ACC, min_cov=G5_MIN_COVERAGE) -> dict:
    """Does any realized coverage reach `target` at or above `min_cov`?

    Reported with the coverage actually achieved, because a curve that crosses
    0.95 only at 12% coverage is a model that abstains on almost everything —
    a different result from the gate, and it has to read that way here."""
    ok = [r for r in rows if r["accuracy"] is not None and r["accuracy"] >= target]
    best = max(ok, key=lambda r: r["coverage"]) if ok else None
    return {"target_accuracy": target, "min_coverage": min_cov,
            "pass": bool(best and best["coverage"] >= min_cov),
            "best_at_target": best,
            "max_accuracy": max((r["accuracy"] for r in rows if r["accuracy"] is not None),
                                default=None)}


def table(rows, per_source, title, g5_rung) -> str:
    """The curve, plus the same curve cut per source.

    The per-source block is the one G5 is actually written against (banking77 +
    dbpedia14 + trec), and a pooled curve can hide a source whose confidence
    carries no information at all: the pool averages, the cell does not."""
    lines = [title, ""]
    lines.append(f"{'rung':>6}{'realized':>10}{'n':>7}{'accuracy':>11}{'risk':>8}{'floor':>8}")
    for r in rows:
        lines.append(f"{r['rung']:>6.2f}{r['coverage']:>10.3f}{r['n']:>7}"
                     + ("      n/a" if r["accuracy"] is None else
                        f"{r['accuracy']:>11.3f}{r['risk']:>8.3f}{r['threshold']:>8.3f}"))
    lines.append("")
    floor = 0.0 if g5_rung is None else g5_rung["threshold"]
    lines.append("per source, whole split and then only the rows at or above the "
                 f"{floor:.3f} floor:")
    lines.append(f"{'source':<16}{'questions':>10}{'acc@1.0':>10}{'cov@floor':>11}"
                 f"{'acc@floor':>11}")
    for source in sorted(per_source):
        pts = per_source[source]
        full = sum(p["correct"] for p in pts) / len(pts)
        kept = at_or_above(pts, floor)
        cov = len(kept) / len(pts)
        acc = (sum(p["correct"] for p in kept) / len(kept)) if kept else None
        lines.append(f"{source:<16}{len(pts):>10}{full:>10.3f}{cov:>11.3f}"
                     + ("        n/a" if acc is None else f"{acc:>11.3f}"))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="risk/coverage curve for one checkpoint")
    ap.add_argument("--ckpt", default="runs/myna-v0")
    ap.add_argument("--suite", default="data/decision-v2-pilot")
    ap.add_argument("--split", default="calibration",
                    choices=["calibration", "development", "test"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None, help="cap rows per source (smoke runs)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the engine re-run that proves the floor is settable")
    ap.add_argument("--out", default="runs/risk_coverage.md")
    args = ap.parse_args(argv)

    fname = {"calibration": "calibration.jsonl", "development": "development.jsonl",
             "test": "test.jsonl"}[args.split]
    path = Path(args.suite) / fname
    if not path.exists():
        raise SystemExit(f"--suite {args.suite} has no {fname}")
    groups = load_split(path)
    if args.limit:
        groups = {k: (q, exs[: args.limit]) for k, (q, exs) in groups.items()}
    n_rows = sum(len(exs) for _q, exs in groups.values())
    n_q = sum(len(q) * len(exs) for q, exs in groups.values())
    report: list[str] = []

    def emit(line: str = "") -> None:
        """Print and keep. The markdown is the committed witness (G7), so every
        verdict line a reader sees on the terminal has to be in the file too — a
        table without its verdict is how a NOT MET gets quoted as a number."""
        print(line)
        report.append(line)

    emit(f"{args.split}: {n_rows} rows / {n_q} questions over {len(groups)} question-sets")

    myna = Myna(args.ckpt, device=args.device)
    emit(f"checkpoint {args.ckpt}: {myna.n_params/1e6:.2f}M params, temperature "
         f"{myna.temperature}, abstain_below={myna.abstain_below} (curve applies its own floor)")
    points = collect(myna, groups)
    rows = curve(points)
    per_source = defaultdict(list)
    for p in points:
        per_source[p["source"]].append(p)
    per_source_rows = {s: curve(pts) for s, pts in per_source.items()}

    v = g5_verdict(rows)
    per_source_g5 = {s: g5_verdict(c) for s, c in per_source_rows.items()}
    floor_row = v["best_at_target"] or rung_at(rows, G5_MIN_COVERAGE)
    emit()
    emit(table(rows, per_source,
               f"risk / coverage — {args.ckpt} on {args.split} ({n_q} questions)", floor_row))
    best = v["best_at_target"]
    emit(f"\nG5 (§2.2): accuracy >= {v['target_accuracy']} at coverage >= "
         f"{v['min_coverage']:.2f}: {'PASS' if v['pass'] else 'NOT MET'}"
         f" — best coverage at target: {None if not best else round(best['coverage'], 3)}"
         f" · max accuracy on this curve: "
         f"{None if v['max_accuracy'] is None else round(v['max_accuracy'], 3)}")
    if v["max_accuracy"] is not None and v["max_accuracy"] < v["target_accuracy"]:
        emit(f"  the curve never reaches {v['target_accuracy']} accuracy, so the floor "
             f"below is the highest-coverage rung that exists on this checkpoint — a harness "
             f"witness, not a G5 pass")
    # G5 names three sources. The table above carries their per-source cut; what it
    # cannot carry is a source that is not in this split at all, which would
    # otherwise read as "absent from the table" rather than "never measured".
    missing = [s for s in G5_SOURCES if s not in per_source_g5]
    if missing:
        emit(f"  the gate's own sources missing from this split, so not judged here: "
             f"{', '.join(missing)}")
    verify = None
    if floor_row is not None and not args.no_verify:
        floor = floor_row["threshold"]
        probe = Myna(args.ckpt, device=args.device, abstain_below=floor)
        n_abstain = 0
        for key, (questions, examples) in sorted(groups.items()):
            for ex in examples:
                out = probe.observe(ex.state).ask(engine_questions(questions))
                n_abstain += len(out["policy"]["abstained"])
        verify = verify_floor(points, floor, n_abstain)
        emit(f"\nfloor {floor:.3f} re-run on the engine: it abstained on "
             f"{verify['engine_abstained']} of {n_q} questions, the curve puts "
             f"{verify['curve_below_floor']} below that floor "
             f"(coverage {verify['realized_coverage']:.3f})")
        if not verify["agree"]:
            emit("MISMATCH: the curve's threshold is not the engine's abstain rule — this "
                 "table cannot be read as G5")
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text("\n".join(report) + "\n")
            return 1
    out_json = Path(str(args.out).rsplit(".", 1)[0] + ".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"ckpt": str(args.ckpt), "suite": str(args.suite),
                                    "split": args.split, "file": fname,
                                    "temperature": myna.temperature,
                                    "n_rows": n_rows, "n_questions": n_q,
                                    "curve": rows, "per_source": per_source_rows,
                                    "g5": v, "g5_per_source": per_source_g5,
                                    "floor_rerun": verify}, indent=2) + "\n")
    md = Path(args.out)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(report) + "\n")
    emit(f"wrote {md} and {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

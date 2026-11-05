"""Per-cell TF-IDF + logistic regression baseline, with per-question records.

Rebuilds the classical reference of Section V so its predictions can be paired with Myna's row by
row. Each cell gets its own vectoriser and classifier, trained on that cell's train questions and
scored on the test split; the text is the state plus the question instructions, exactly the input
the other classical arms receive.

    python -m bench.tfidf_baseline --out runs/tfidf_test.rowoutcomes.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "release"))

from colab_baselines import state_text  # noqa: E402


def gold_index(spec: dict) -> int:
    if spec["type"] == "noul":
        return 1 if spec["label"] else 0
    if spec["type"] == "score":
        return int(spec["label"])
    crit = spec["criteria"]
    labels = list(crit) if isinstance(crit, dict) else [str(c) for c in crit]
    return labels.index(spec["label"])


def rows(split: str) -> list[dict]:
    return [json.loads(l) for l in (REPO / f"data/decision-v2-pilot/{split}.jsonl").read_text().splitlines()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/tfidf_rebuild.rowoutcomes.json")
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    train, test = rows("train"), rows("test")
    texts: dict[str, tuple[list[str], list[int]]] = {}
    for r in train:
        base = state_text(r["state"])
        for q, spec in r["questions"].items():
            cell = f"{r['_meta']['source']}/{q}"
            text = base + "\n" + str(spec.get("instructions", ""))
            texts.setdefault(cell, ([], []))
            texts[cell][0].append(text)
            texts[cell][1].append(gold_index(spec))

    models = {}
    for cell, (xs, ys) in texts.items():
        if len(set(ys)) < 2:
            continue
        model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                              LogisticRegression(max_iter=2000, C=4.0))
        model.fit(xs, ys)
        models[cell] = model

    out = {"cmd": "python -m bench.tfidf_baseline", "kind": "tfidf-rebuild",
           "note": "reconstruction from the published description; 0.663 macro against the 0.7111 the original harness reported, so this is not the published arm",
           "vectorizer": "tfidf word 1-2gram, sublinear, min_df=1",
           "classifier": "logistic regression C=4.0, max_iter=2000", "rows": []}
    for r in test:
        base = state_text(r["state"])
        qs = {}
        for q, spec in r["questions"].items():
            cell = f"{r['_meta']['source']}/{q}"
            text = base + "\n" + str(spec.get("instructions", ""))
            gold = gold_index(spec)
            model = models.get(cell)
            pred = int(model.predict([text])[0]) if model is not None else -1
            qs[q] = {"gold": gold, "pred": pred, "ok": int(pred == gold)}
        out["rows"].append({"meta": {"id": r["_meta"]["id"], "source": r["_meta"]["source"]},
                            "questions": qs})
    (REPO / args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out, "| cells:", len(models))


if __name__ == "__main__":
    main()

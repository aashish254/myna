#!/usr/bin/env python
"""Vegaml per-question outcomes on a whole frozen split — the control arm beside laya's.

    "<vegaml venv python>" bench/eval_vegaml_row_outcomes.py \
        --out runs/vegaml_0150_test.rowoutcomes.json

`vegaml` ships an evaluation module — `vegaml/evals.py:60 evaluate(model, rows, questions)` —
and it writes aggregates only: per-question `{n, accuracy, abstain_rate, ece, coverage,
mean_set_size}` and an `overall` block, printed to stdout. Nothing in that package puts a
per-item outcome on disk, so the library cannot be paired with myna out of the box. This file
scores it one row at a time on the same 1,176 test rows `bench/row_outcomes.py` dumped, keeps
every question's outcome, and is therefore the second engine `bench/significance.py`'s
instrument can be run against.

The engine is `vegaml` 0.15.0 from PyPI in its own venv, not the git checkout: this
repository must not edit a competitor's tree to measure it, and a published wheel is
re-fetchable by a reader while a local checkout is not. What answered is a witness field —
version, package digest, torch, transformers, the checkpoint dir and the sha256 of every
weight file that was read — not an assumption.

**The schema shim is one rename and two drops.** `noul` is this suite's spelling of what
vegaml calls `boolean` (`vegaml/_common.py:41 QTYPES = {"choice": 0, "score": 1, "boolean": 2}`);
`label` and `src` are gold and provenance and must not reach the engine. `instructions`
stays as it is even when it is a dict — three of the fifteen `score` cells store a
`{question, focus}` mapping and `_api.py:194` serialises a non-string instruction itself.

**One scoring rule genuinely differs from the competitor column already published.** For a
`score` question, laya answers a distribution keyed by the *level text* and
`eval_laya_real.score_pred` argmaxes over those keys, while vegaml's labels are the
positional indices `list(range(k))` (`_api.py:275`) — so here the gold is compared as an
index. That is only safe because every `score` cell in this suite stores its gold as a
0-based index into an ordered `criteria` list: `amazon/stars`, `sst5/sentiment` and
`yelp/rating` each have five criteria and labels in 0..4, which
`test_score_labels_are_positional` asserts rather than assumes. For `choice` and `noul` the
rule is the same two lines that decided the laya column — `pred == label`, and `p >= 0.5`.

**Abstention is recorded, not scored.** vegaml is the only engine here that can decline: an
answer carries `abstain`, `prediction_set` and `unbound` from the checkpoint's own thermal
calibration (`_api.py:281-298`). Forcing a choice keeps the three arms the same measurement,
so `ok` comes from the argmax regardless of the flag, and the abstain rate, set sizes and the
accuracy abstentions *would* have cost are reported beside it as facts about the control.

Three guards, all fatal, inherited from the laya pass and not weakened:

* **Geometry.** This file refuses to write unless the (source, question) cells have the same
  `n` as the myna dump's, cell for cell, and every `(meta.id, question)` pair myna has is
  answered or named in `errors`.
* **The witness holds the rows.** After writing, the file is read back and its `rows` count
  compared with this pass's before the crash-resume sidecar is deleted — the check the first
  run of `eval_laya_row_outcomes.py` failed to have.
* **Self-consistency.** vegaml's `choice`/`score` field and an argmax over its own `probs`
  must name the same option; a disagreement is counted, printed, and put in the witness
  rather than averaged away.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "bench"))

from row_outcomes import META_FIELDS  # noqa: E402

#: This suite's type names to vegaml's. `noul` is the only one that is not already a vegaml type.
TYPE_MAP = {"choice": "choice", "score": "score", "noul": "boolean"}

#: Which weight files the pass read, named so the digest block is a list of artifacts and not
#: a glob a reader has to re-guess. `port.safetensors` is the 0.8B reader's port at the
#: repository root (`vegaml/hub.py:22 VARIANTS = {"0.8b": "", ...}`).
CHECKPOINT_FILES = ("engine.safetensors", "port.safetensors", "vega_config.json", "config.json")


def vegaml_questions(qs):
    """The suite's question table as vegaml's `decide` wants it: types renamed, gold removed."""
    out = {}
    for qid, q in qs.items():
        t = q["type"]
        if t not in TYPE_MAP:
            raise SystemExit(f"{qid}: question type {t!r} has no vegaml counterpart in "
                             f"{sorted(TYPE_MAP)} — the control would be silently dropping it")
        d = {"type": TYPE_MAP[t], "instructions": q["instructions"]}
        if q.get("criteria") is not None:
            d["criteria"] = q["criteria"]
        out[qid] = d
    return out


def score_pred(ans):
    """Discrete level for a score answer: argmax of the level distribution, as an index."""
    return int(max(ans["probs"], key=lambda k: ans["probs"][k]))


def outcome(q, a):
    """(ok, pred, conf, disagree) for one vegaml answer, under the two rules named above.

    `disagree` is whether vegaml's own headline field and an argmax over its `probs` named
    different options. It is not a scoring question — `ok` follows the same field a reader of
    `vegaml.decide` would follow — but a control whose answer fields disagree with its own
    probability table is a fact the witness has to carry.
    """
    t = q["type"]
    if t == "choice":
        pred, arg = a["choice"], max(a["probs"], key=a["probs"].get)
        ok, disagree = int(pred == q["label"]), int(str(pred) != str(arg))
    elif t == "score":
        pred, arg = int(a["score"]), score_pred(a)
        ok, disagree = int(pred == int(q["label"])), int(pred != arg)
    else:
        p = float(a["boolean"])
        pred = bool(p >= 0.5)
        ok, disagree = int(pred == bool(q["label"])), int(boolean_disagree(a, pred))
    conf = a.get("confidence")
    return ok, pred, (None if conf is None else float(conf)), disagree


def boolean_disagree(a, pred):
    """Whether a boolean answer contradicts its own probability table.

    vegaml builds `probs` for a boolean from `labels = [False, True]` and sets `boolean` to
    `p[1]` with `confidence` to `p.max()` (`_api.py:275-290`), so two things must hold at once:
    the side the `boolean` field favours is the argmax of `probs`, and `confidence` equals the
    larger of the two. Checking only the field against a half-line would pass a table that says
    the opposite, and comparing `confidence` against `p.max()` alone would pass an answer whose
    `boolean` leans the other way.
    """
    probs = a["probs"]
    best = max(probs, key=probs.get)
    wrong_side = str(best) != str(pred)
    conf = a.get("confidence")
    wrong_conf = conf is not None and abs(float(conf) - max(probs.values())) > 1e-9
    return wrong_side or wrong_conf


def identify(path):
    """[(_meta, questions, state)] for every row of the split, in file order.

    Same shape as `eval_laya_row_outcomes.identify`: the rows go to the engine raw, so the
    split is read directly and `_meta` is kept rather than going through `myna.real_data`.
    """
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row.get("_meta") or {}
            out.append(({k: meta.get(k) for k in META_FIELDS}, row["questions"], row["state"]))
    return out


def _sha256(path, limit=None):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def ckpt_digests(d):
    """sha256 (first 16 hex) of each weight file this pass could have read, plus its bytes."""
    out = {}
    for name in CHECKPOINT_FILES:
        p = Path(d) / name
        if p.exists():
            out[name] = {"sha256": _sha256(p), "bytes": p.stat().st_size}
    return out


def version_pin(repo_id, device, threads, mode, views, alpha):
    """Which vegaml answered: the package, the checkpoint and the flags.

    The reader is resolved from the checkpoint's own `backbone_name`, which is a bare name
    ("Qwen3.5-0.8B") and not a Hub id — `vegaml/hub.py:34 backbone_hub_id` is what adds the
    namespace, and `try_to_load_from_cache` then proves the weights that answer are already on
    this box rather than downloading a second copy mid-measurement. A reader that is not
    cached fails here, loudly, with the command that would fetch it.
    """
    import numpy
    import torch
    import transformers
    import vegaml
    from vegaml.hub import backbone_hub_id, fetch

    root = Path(vegaml.__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # a wheel has no git dir; say so rather than pretending
        commit = f"unresolved ({type(e).__name__})"
    ckpt = Path(fetch(repo_id))
    cfg = json.loads((ckpt / "vega_config.json").read_text())
    reader = cfg.get("backbone_name") or "unknown"
    hub_id = backbone_hub_id(reader) if reader != "unknown" else None
    pin = {"version": getattr(vegaml, "__version__", "unknown"), "path": str(root),
           "commit": commit or "no git commit reported (installed from a wheel)",
           "init_sha256": _sha256(root / "__init__.py"),
           "python": sys.version.split()[0], "torch": torch.__version__,
           "transformers": transformers.__version__, "numpy": numpy.__version__,
           "repo": repo_id, "checkpoint_dir": str(ckpt), "checkpoint_files": ckpt_digests(ckpt),
           "reader": reader, "reader_hub_id": hub_id,
           "device": device, "threads": threads, "mode": mode, "views": views, "alpha": alpha}
    if hub_id and "/" in str(hub_id):
        from huggingface_hub import try_to_load_from_cache

        def cached(name):
            p = try_to_load_from_cache(hub_id, name)
            return Path(p) if p and str(p).startswith("/") and Path(p).exists() else None

        # The shard list comes from the repository's own index, so what is hashed below is the
        # set of files the reader is defined by rather than whatever a glob happens to find.
        single = cached("model.safetensors")
        if single is not None:
            shards, root = [single], single.parent
        else:
            ix = cached("model.safetensors.index.json")
            if ix is None:
                raise SystemExit(
                    f"reader {hub_id} is not on this box (`model.safetensors` and its index "
                    f"are both absent from the HF cache). This pass refuses to download a "
                    f"model during a measurement: fetch it first with "
                    f"`huggingface_hub.snapshot_download({hub_id!r}, "
                    f"allow_patterns=['*.safetensors', '*.json', '*.jinja'])` and re-run, so "
                    f"the bytes read are ones somebody chose to put here")
            names = sorted(set(json.loads(ix.read_text())["weight_map"].values()))
            shards = [cached(n) for n in names]
            if any(s is None for s in shards):
                raise SystemExit(f"{hub_id}: the index names {names} but "
                                 f"{[n for n, s in zip(names, shards) if s is None]} is not "
                                 f"cached — a partial reader download would score the control "
                                 f"on weights nobody can re-fetch")
            root = ix.parent
        pin["reader_dir"] = str(root)
        pin["reader_weights"] = [{"file": p.name, "sha256": _sha256(p),
                                  "bytes": p.stat().st_size} for p in shards]
        pin["reader_weight_bytes"] = sum(w["bytes"] for w in pin["reader_weights"])
    else:
        pin["reader_dir"] = None
        pin["reader_weights"] = []
        pin["reader_weight_bytes"] = 0
    return pin


def build_vega(pin):
    """`vegaml.load(...)` with whatever it warns about kept, per constructor and per call.

    No `backbone=` is passed: the engine resolves the reader from its own config
    (`_api.py:73-94`), and `v.engine.backbone` is then a fact about what loaded rather than a
    restatement of what this file asked for. The parameter counts below are read off that
    object for the same reason.

    The same two-stage capture the laya pass needed: a reader can warn on the first `decide`
    even when construction was quiet, and `bench/significance.py` cites the warning list for
    the sentence "only correctness is comparable here" — an empty list has to mean nothing was
    emitted, not that nobody looked.
    """
    import vegaml

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        v = vegaml.load(pin["repo"], device=pin["device"], mode=pin["mode"],
                        threads=pin["threads"])
    return v, [f"{w.category.__name__}: {w.message}" for w in caught]


def record_loaded(v, pin):
    """What actually loaded, measured off the engine object: the reader id and its size.

    The control's cost only means something next to myna's, so the parameter count is counted
    from the modules in memory rather than quoted from a model card.
    """
    e = v.engine
    pin["backbone_used"] = str(e.backbone)
    pin["reader_params"] = int(sum(p.numel() for p in e.perc.model.parameters()))
    # `_core()` unwraps the vega-U port wrapper (`_api.py:144`), so the engine count is the
    # engine's and not the port's; the wrapper total is printed beside it rather than merged.
    pin["engine_params"] = int(sum(p.numel() for p in e._core().parameters()
                                   if p.requires_grad))
    pin["engine_params_total"] = int(sum(p.numel() for p in e._core().parameters()))
    pin["engine_params_with_port"] = int(sum(p.numel() for p in e.engine.parameters()))
    pin["perc_dtype"] = str(next(e.perc.model.parameters()).dtype)
    pin["n_layers"] = int(e.perc.n_layers)
    pin["feature_layers"] = list(e.perc.feature_layers)
    pin["option_layers"] = list(e.perc.option_layers)
    pin["engine_class"] = type(e.engine).__name__
    return pin


def cell_table(rows):
    """(source, question) cells, merged across types exactly as `myna.report.laya_cells` does."""
    cells = defaultdict(lambda: {"n": 0, "hits": 0, "parts": defaultdict(lambda: [0, 0])})
    for r in rows:
        for name, q in r["questions"].items():
            c = cells[(r["meta"]["source"], name)]
            c["n"] += 1
            c["hits"] += q["ok"]
            p = c["parts"][q["type"]]
            p[0] += q["ok"]
            p[1] += 1
    return [{"source": s, "question": q, "n": c["n"], "hits": c["hits"], "acc": c["hits"] / c["n"],
             "types": {t: {"acc": h / n, "n": n} for t, (h, n) in sorted(c["parts"].items())}}
            for (s, q), c in sorted(cells.items())]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--suite", default="data/decision-v2-pilot")
    ap.add_argument("--split", default="test", choices=("test", "dev", "calibration"))
    ap.add_argument("--repo", default="nandakishorm/vega-U",
                    help="checkpoint repo id; the 0.8B variant is its root (vegaml/hub.py:22)")
    ap.add_argument("--device", default="cpu", help="cpu/mps/cuda; cpu so this can share a box")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch CPU threads — vegaml/hub.py documents 1 as the right value for "
                         "the shipped Qwen3.5 readers; the flag is recorded in the witness")
    ap.add_argument("--mode", default="engine", choices=("auto", "engine", "ttt", "both"),
                    help="engine is the readout vegaml's own evals and bench/ score at")
    ap.add_argument("--views", type=int, default=1, choices=(1, 2, 3),
                    help="1 = one reading. 2 averages a reversed-option view and adds "
                         "`view_agreement`; a cost, not a free accuracy")
    ap.add_argument("--alpha", default="0.1", help="conformal level for `prediction_set`")
    ap.add_argument("--myna-dump", default="runs/myna_v1b_test.rowoutcomes.json",
                    help="the myna row dump this pass must have the same geometry as")
    ap.add_argument("--out", default="runs/vegaml_0150_test.rowoutcomes.json")
    ap.add_argument("--raw", default="runs/vegaml_0150_test.rowoutcomes.raw.json",
                    help="crash-resume sidecar for the per-row outcomes; written before any "
                         "aggregation and removed when the witness is written")
    ap.add_argument("--from-raw", action="store_true",
                    help="re-aggregate from --raw instead of running inference again")
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N rows — the throughput probe, which prices a "
                         "full pass in ms/row before 1,176 rows are paid for")
    ap.add_argument("--probe-out", default=None,
                    help="with --limit, write the probe's per-row outcomes here. A `--views 2` "
                         "probe is a measurement of option-order robustness, and a number "
                         "printed once to a terminal is not a witness anyone can re-read")
    ap.add_argument("--every", type=int, default=100, help="progress print interval")
    args = ap.parse_args(argv)
    given = list(sys.argv[1:]) if argv is None else list(argv)
    flags = " ".join(json.dumps(a) if " " in a else a for a in given)
    print("$ " + f"'{sys.executable}' bench/eval_vegaml_row_outcomes.py {flags}", flush=True)

    fname = {"test": "test.jsonl", "dev": "development.jsonl",
             "calibration": "calibration.jsonl"}[args.split]
    path = Path(args.suite) / fname
    if not path.exists():
        print(f"--suite {args.suite} has no {fname}", file=sys.stderr)
        return 2
    dump_path = Path(args.myna_dump)
    if not dump_path.exists():
        print(f"--myna-dump {dump_path} is not on disk: this pass is the control half of a "
              f"paired bootstrap and needs the other half's geometry to check itself against",
              file=sys.stderr)
        return 2
    mine = json.loads(dump_path.read_text())
    want = {(r["meta"]["id"], name) for r in mine["rows"] for name in r["questions"]}
    if args.limit:
        want = {k for k in want
                if k[0] in {r["meta"]["id"] for r in mine["rows"][:args.limit]}}

    raw_path = Path(args.raw)
    if args.from_raw:
        if not raw_path.exists():
            print(f"--from-raw but {raw_path} is not on disk", file=sys.stderr)
            return 2
        raw = json.loads(raw_path.read_text())
        rows, errors, pin, warned = raw["rows"], raw["errors"], raw["vegaml"], raw["warnings"]
        seen_n, seconds, n_tokens = raw["n_rows_in_file"], raw["seconds"], raw["n_tokens"]
        disagree = raw["argmax_disagreements"]
        inference_s = seconds
        cmd = raw["cmd"]
        print(f"re-aggregating {len(rows)} rows from {raw_path} (vegaml {pin['version']}, "
              f"{seconds}s of inference already paid)", flush=True)
    else:
        pin = version_pin(args.repo, args.device, args.threads, args.mode, args.views,
                          args.alpha)
        print(f"vegaml {pin['version']} at {pin['path']} · reader {pin['reader']} "
              f"({pin['reader_weight_bytes'] / 1e9:.2f} GB) · ckpt "
              f"{pin['checkpoint_files'].get('engine.safetensors', {}).get('sha256', '?')} "
              f"· python {pin['python']} torch {pin['torch']} transformers {pin['transformers']} "
              f"· {args.device} threads={args.threads} mode={args.mode} views={args.views}",
              flush=True)
        cmd = f"'{sys.executable}' bench/eval_vegaml_row_outcomes.py {flags}"
        v, warned = build_vega(pin)
        record_loaded(v, pin)
        print(f"loaded: reader {pin['backbone_used']} at {pin['reader_params']:,} params "
              f"({pin['perc_dtype']}, {pin['n_layers']} layers) · engine "
              f"{pin['engine_params']:,} trainable · warned {len(warned)}", flush=True)
        for w in warned:
            print(f"  vegaml warned: {w}", flush=True)
        rows, errors = [], []
        seen = identify(path)
        if args.limit:
            seen = seen[:args.limit]
        started, n_tokens, disagree = time.time(), 0, 0
        live_seen = 0
        with warnings.catch_warnings(record=True) as live:
            warnings.simplefilter("once")
            for i, (meta, questions, state) in enumerate(seen, 1):
                try:
                    out = v.decide(state, vegaml_questions(questions), mode=args.mode,
                                   alpha=args.alpha, views=args.views)
                except Exception as e:
                    errors.append({"id": meta.get("id"), "source": meta.get("source"),
                                   "error": f"{type(e).__name__}: {e}"})
                    continue
                n_tokens += int(out.get("n_tokens") or 0)
                ans = out["answers"]
                rec = {}
                for qid, q in questions.items():
                    a = ans.get(qid)
                    if a is None:
                        errors.append({"id": meta.get("id"), "source": meta.get("source"),
                                       "error": f"no answer for {qid}"})
                        continue
                    ok, pred, conf, bad = outcome(q, a)
                    disagree += bad
                    rec[qid] = {"type": q["type"], "ok": ok, "gold": q["label"],
                                "pred": pred if isinstance(pred, (str, bool, int)) else str(pred),
                                "conf": None if conf is None else round(conf, 6),
                                "abstain": bool(a.get("abstain")),
                                "set_n": None if a.get("prediction_set") is None
                                         else len(a["prediction_set"]),
                                "agree": a.get("view_agreement"),
                                "unbound": bool(a.get("unbound"))}
                    if bad:
                        errors.append({"id": meta.get("id"), "source": meta.get("source"),
                                       "error": f"probs argmax disagrees with {q['type']} "
                                                f"field for {qid}"})
                rows.append({"meta": meta, "questions": rec})
                for entry in live[live_seen:]:
                    warned.append(f"{entry.category.__name__}: {entry.message}")
                    print(f"  vegaml warned: {entry.category.__name__}: {entry.message}",
                          flush=True)
                live_seen = len(live)
                if i % args.every == 0 or i == len(seen):
                    el = time.time() - started
                    print(f"  {i} rows in {el:.0f}s ({el / i * 1000:.0f} ms/row, "
                          f"{n_tokens / i:.0f} tokens/row, {len(errors)} unanswered)", flush=True)
            t0 = time.time() - started

        n_q = sum(len(r["questions"]) for r in rows)
        print(f"{len(rows)} rows answered / {n_q} question outcomes in {t0:.0f}s on "
              f"{args.device} · {len(errors)} problems · {n_tokens} reader tokens · "
              f"{disagree} argmax disagreements", flush=True)
        if args.limit:
            per_row = t0 / max(len(seen), 1)
            agree = [q["agree"] for r in rows for q in r["questions"].values()
                     if q.get("agree") is not None]
            print(f"PROBE: {len(seen)} rows at {per_row:.2f}s/row puts the "
                  f"full 1,176-row pass at {per_row * 1176 / 60:.0f} min on "
                  f"this box with device={args.device} threads={args.threads}; nothing was "
                  f"written to {args.out}", flush=True)
            if args.probe_out:
                payload = {"cmd": cmd, "engine": "vegaml", "vegaml": pin, "warnings": warned,
                           "probe": True, "n_rows": len(rows),
                           "n_questions": sum(len(r["questions"]) for r in rows),
                           "seconds": round(t0, 1),
                           "ms_per_row": round(per_row * 1000, 1),
                           "tokens_per_row": round(n_tokens / max(len(seen), 1), 1),
                           "view_agreement": {
                               "n": len(agree),
                               "mean": (sum(agree) / len(agree)) if agree else None,
                               "flipped": sum(1 for x in agree if x < 1.0),
                               "rule": "views=1 emits no agreement field; this is None on a "
                                       "single-reading pass, not zero"},
                           "cells": cell_table(rows), "rows": rows}
                po = Path(args.probe_out)
                po.parent.mkdir(parents=True, exist_ok=True)
                po.write_text(json.dumps(payload, indent=1) + "\n")
                back = json.loads(po.read_text())
                if len(back.get("rows", [])) != len(rows):
                    print(f"GUARD FAIL: {po} carries {len(back.get('rows', []))} rows and the "
                          f"probe answered {len(rows)}", file=sys.stderr)
                    return 3
                print(f"wrote the probe to {po} — {payload['n_questions']} outcomes, "
                      f"view agreement {payload['view_agreement']}", flush=True)
            return 0
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps({"cmd": cmd, "engine": "vegaml", "vegaml": pin,
                                       "warnings": warned,
                                       "suite": str(args.suite), "split": args.split,
                                       "file": fname, "n_rows": len(rows),
                                       "n_rows_in_file": len(seen), "n_questions": n_q,
                                       "n_tokens": n_tokens, "errors": errors,
                                       "argmax_disagreements": disagree,
                                       "seconds": round(t0, 1), "rows": rows}, indent=1) + "\n")
        print(f"wrote the raw outcomes to {raw_path} before aggregating", flush=True)
        inference_s, seen_n = round(t0, 1), len(seen)

    got = {(r["meta"]["id"], name) for r in rows for name in r["questions"]}
    missing = sorted(want - got)
    if sorted(got - want):
        print(f"GUARD FAIL: {len(got - want)} (row, question) pairs are in the vegaml pass and "
              f"not in the myna dump (first {sorted(got - want)[:3]})", file=sys.stderr)
        return 3
    if missing and not args.from_raw:
        print(f"GUARD FAIL: {len(missing)} (row, question) pairs myna answers are missing here "
              f"(first {missing[:3]}) — they are named in `errors`, but a control that cannot "
              f"answer a row is reported as a drop, not written silently", file=sys.stderr)
        for e in errors[:10]:
            print(f"  {e}", file=sys.stderr)
        return 3

    cell_rows = cell_table(rows)
    published = {(c["source"], c["question"]): c for c in mine["cells"]}
    geom = []
    for c in cell_rows:
        p = published.get((c["source"], c["question"]))
        if p is None:
            geom.append(f"vegaml scored a cell myna does not have: {c['source']}/{c['question']}")
        elif c["n"] != p["n"]:
            geom.append(f"{c['source']}/{c['question']}: vegaml has {c['n']} question "
                        f"instances, the myna dump has {p['n']}")
    for key in published:
        if not any((c["source"], c["question"]) == key for c in cell_rows):
            geom.append(f"{key[0]}/{key[1]} is in the myna dump and absent from vegaml")
    if geom:
        for line in geom:
            print(f"GUARD FAIL: {line}", file=sys.stderr)
        print("nothing was written", file=sys.stderr)
        return 3

    n_q = sum(len(r["questions"]) for r in rows)
    abstained = [(q.get("abstain"), q.get("ok")) for r in rows for q in r["questions"].values()]
    n_abstain = sum(1 for a, _ in abstained if a)
    sets = [q["set_n"] for r in rows for q in r["questions"].values() if q.get("set_n")]
    payload = {"cmd": cmd, "engine": "vegaml", "vegaml": pin, "warnings": warned,
               "suite": str(args.suite), "split": args.split, "file": fname,
               "n_rows": len(rows), "n_questions": n_q, "n_rows_in_file": seen_n,
               "errors": errors, "cluster_key": "meta.id",
               "myna_dump": str(dump_path), "myna_n_rows": mine["n_rows"],
               "paired_cells": len(cell_rows), "cells": cell_rows,
               "reader_tokens": n_tokens, "argmax_disagreements": disagree,
               "abstain": {"n": n_abstain, "rate": n_abstain / max(n_q, 1),
                           "accuracy_of_abstained": (
                               sum(ok for a, ok in abstained if a) / n_abstain
                               if n_abstain else None),
                           "mean_set_size": (sum(sets) / len(sets)) if sets else None,
                           "rule": "scored forced-choice; the flag is recorded, not applied"},
               "rows": rows, "seconds": inference_s}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    back = json.loads(out.read_text())
    if len(back.get("rows", [])) != len(rows):
        print(f"GUARD FAIL: {out} carries {len(back.get('rows', []))} rows and this pass has "
              f"{len(rows)}, so the per-question outcomes are not in the witness. "
              f"{raw_path.name} is being kept — significance.py cannot read cell means",
              file=sys.stderr)
        return 3
    if raw_path.exists():
        raw_path.unlink()
    macro = sum(c["acc"] for c in cell_rows) / len(cell_rows)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB) holding {len(back['rows'])} rows "
          f"/ {sum(len(r['questions']) for r in back['rows'])} question outcomes, removed the "
          f"{raw_path.name} sidecar — the witness carries the rows")
    print(f"cell-macro over {len(cell_rows)} cells = {macro:.16f} · abstained "
          f"{n_abstain}/{n_q} = {payload['abstain']['rate']:.4f}")
    for c in cell_rows:
        print(f"  {c['source']}/{c['question']:<14s} n={c['n']:>4d} acc={c['acc']:.4f}  "
              + " ".join(f"{t}:{v['acc']:.3f}({v['n']})" for t, v in c["types"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

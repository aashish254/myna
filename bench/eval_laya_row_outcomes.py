#!/usr/bin/env python
"""Laya Router per-question outcomes on a whole frozen split — the paired half of a bootstrap.

    PYTHONPATH="<laya checkout at 0.4.19>" "<that checkout>/.venv/bin/python" \
        bench/eval_laya_row_outcomes.py --out runs/laya_0419_test.rowoutcomes.json

`bench/eval_laya_real.py` produced the committed competitor witness and it does two things
this one must not: it subsamples (`--n-per-source 40`, so 440 of 1,176 rows) and it keeps
only counts (`acc`, `n`, `brier` per (source, qid, type)). Both are right for what that
witness is — a per-cell competitor column — and both are fatal for a paired test. A row
bootstrap compares myna and laya on the *same rows*, so the competitor has to be scored on
all of them, at one pass, with every question's outcome kept. `bench/significance.py` reads
this file beside `runs/myna_v1b_test.rowoutcomes.json` and the two join on
`(meta.id, question)`.

The interpreter is laya's own venv, not this repository's: `Router.load` needs `safetensors`,
which myna's lockfile does not carry, and adding a competitor's dependency to this project
to score a competitor would be the wrong kind of coupling. What the pass ran on is therefore
a witness field — python, torch, laya version, checkout and git commit — not an assumption.

What is reused rather than restated: the schema shim and the scoring rules are imported
from `eval_laya_real` (`laya_questions`, `score_pred`), so the correctness of an outcome is
decided by the same three lines that decided the published competitor column — argmax over
`probabilities` for `score`, `choice == label` for `choice`, `p >= 0.5` for `noul`.

Three guards, all fatal:

* **The witness holds the rows.** Before the raw sidecar is deleted, the written file is read
  back and its row count compared with this pass's. `bench/significance.py` joins on
  `(meta.id, question)`, so a witness of cell means is useless to it and the sidecar that
  held the rows must survive — which is the check the first run of this file lacked.

* **Geometry.** The (source, question) cells this file prices must have the same `n` as the
  myna dump's, cell for cell, apart from rows laya could not answer — and those rows are
  named, counted and dropped from both sides by `significance.py` rather than silently
  missing. 1,440 question instances over 1,176 rows is the published geometry (SPEC §9.32:
  `contrastive/decision` is one merged cell of 80 noul + 116 choice questions).
* **Version.** `--compare` re-derives the committed witness's own 40-per-source row set from
  `_meta.id` and prints laya 0.4.19's accuracy on exactly those rows against the committed
  0.3.20-era numbers, per cell. A difference there is not this run being wrong; it is the
  competitor moving under the citation, which is the defect SPEC §4.2 carries and this turn
  makes visible with numbers instead of adjectives. It costs no extra inference: a per-row
  `predict` is independent of the other rows, so restricting this pass to those row ids
  *is* the subset run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "bench"))

from eval_laya_real import laya_questions, score_pred  # noqa: E402
from row_outcomes import META_FIELDS  # noqa: E402


def outcome(q, a):
    """(ok, pred, conf) for one laya answer, scored by eval_laya_real's own rules."""
    t = q["type"]
    if t == "choice":
        return (int(a["choice"] == q["label"]), a["choice"],
                max(a["probabilities"].values()) if a.get("probabilities") else None)
    if t == "score":
        level = score_pred(a)
        probs = a["probabilities"]
        return int(level == str(q["label"])), level, max(probs.values())
    p = a["noul"]
    return int((p >= 0.5) == bool(q["label"])), bool(p >= 0.5), max(p, 1.0 - p)


def identify(path):
    """[(_meta, questions, state)] for every row of the split, in file order.

    `myna.real_data.load_split` is the loader on myna's side; on laya's side the rows go to
    `Router.predict` raw, so the split is read directly and `_meta` is kept. The row count
    is checked against myna's dump downstream — a row that parses for one engine and not the
    other is a geometry difference, not a scoring one, and it must show up as a count.
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


def subset_ids(path, n_per_source, seed):
    """The row ids `eval_laya_real.py` would have drawn — same rng calls, same order.

    Reproduced rather than re-run because the committed witness is a comparison over a
    sample someone else's harness picked: to price what changed in laya between that witness
    and this pass, the rows have to be the same ones. `by_src` insertion order is the file's
    order, `rng.shuffle(rows)` mutates each source's list in that order, and the final
    `rng.shuffle(records)` does not affect a per-row predict — so the id set is the whole of
    what the sample means.
    """
    by_src = defaultdict(list)
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by_src[(r.get("_meta") or {}).get("source", "?")].append(
                (r.get("_meta") or {}).get("id"))
    rng = random.Random(seed)
    ids = []
    for src, rows in by_src.items():
        rng.shuffle(rows)
        ids.extend(rows[:n_per_source])
    rng.shuffle(ids)
    return set(ids)


def version_pin():
    """Which laya is answering: version, checkout, git commit, and the wheel's own digest.

    The competitor is not a number in this repository — it is code in a directory named by
    `PYTHONPATH`, and that directory moves when its owner pushes. A witness that does not pin
    it cannot be re-scored, and §9.37 binds a witness to its inputs.
    """
    import laya
    import torch
    root = Path(laya.__file__).resolve().parent.parent
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # a source tarball has no git dir; say so rather than pretending
        commit = f"unresolved ({type(e).__name__})"
    return {"version": getattr(laya, "__version__", "unknown"), "path": str(root),
            "commit": commit or "no git commit reported",
            "init_sha256": hashlib.sha256(Path(laya.__file__).read_bytes()).hexdigest()[:16],
            "python": sys.version.split()[0], "torch": torch.__version__}


def build_router(device):
    """`Router(device=...)`, with whatever it warns about kept.

    The committed witness was produced under a `RuntimeWarning` from laya — this checkpoint
    ships a temperature outside its own [0.5, 5] clamp, and laya says the confidence of the
    affected entries is uncalibrated. That is a fact about the competitor's artifact and it
    belongs in the witness beside the accuracies, not in a terminal buffer nobody kept.

    This only ever sees warnings the *constructor* emits. `Router` builds its `Agent` on the
    first `predict`, so the warning about the checkpoint's temperatures does not appear here —
    `main()` wraps the scoring loop for that reason, and the two lists are concatenated.
    """
    import warnings

    from laya import Router
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        router = Router(device=device)
    return router, [f"{w.category.__name__}: {w.message}" for w in caught]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--suite", default="data/decision-v2-pilot")
    ap.add_argument("--split", default="test", choices=("test", "dev", "calibration"))
    ap.add_argument("--device", default="cpu", help="cpu/mps/cuda; cpu so this can share a box")
    ap.add_argument("--myna-dump", default="runs/myna_v1b_test.rowoutcomes.json",
                    help="the myna row dump this pass must have the same geometry as")
    ap.add_argument("--compare", default="runs/laya_decision_v2_test.json",
                    help="the committed subset witness, to price the version drift")
    ap.add_argument("--out", default="runs/laya_0419_test.rowoutcomes.json")
    ap.add_argument("--raw", default="runs/laya_0419_test.rowoutcomes.raw.json",
                    help="crash-resume sidecar for the per-row outcomes; written before any "
                         "aggregation and removed when the witness is written")
    ap.add_argument("--from-raw", action="store_true",
                    help="re-aggregate from --raw instead of running inference again")
    ap.add_argument("--every", type=int, default=100, help="progress print interval")
    args = ap.parse_args(argv)
    given = list(sys.argv[1:]) if argv is None else list(argv)
    flags = " ".join(json.dumps(a) if " " in a else a for a in given)
    print("$ " + f"PYTHONPATH='<laya checkout>' <laya venv python> "
          f"bench/eval_laya_row_outcomes.py {flags}", flush=True)

    fname = {"test": "test.jsonl", "dev": "development.jsonl",
             "calibration": "calibration.jsonl"}[args.split]
    path = Path(args.suite) / fname
    if not path.exists():
        print(f"--suite {args.suite} has no {fname}", file=sys.stderr)
        return 2
    dump_path = Path(args.myna_dump)
    if not dump_path.exists():
        print(f"--myna-dump {dump_path} is not on disk: this pass is the paired half of a "
              f"bootstrap and needs the other half's geometry to check itself against",
              file=sys.stderr)
        return 2
    mine = json.loads(dump_path.read_text())
    want = {(r["meta"]["id"], name) for r in mine["rows"] for name in r["questions"]}

    raw_path = Path(args.raw)
    if args.from_raw:
        if not raw_path.exists():
            print(f"--from-raw but {raw_path} is not on disk", file=sys.stderr)
            return 2
        raw = json.loads(raw_path.read_text())
        rows, errors, pin, warned = raw["rows"], raw["errors"], raw["laya"], raw["laya_warnings"]
        seen_n, seconds = raw["n_rows_in_file"], raw["seconds"]
        cmd = raw["cmd"]
        print(f"re-aggregating {len(rows)} rows from {raw_path} "
              f"(laya {pin['version']} commit {pin['commit'][:12]}, {seconds}s of inference "
              f"already paid)", flush=True)
    else:
        pin = version_pin()
        # The recorded command names the interpreter that can actually run it. myna's `.venv`
        # has no `safetensors`, so a copy-paste of the printed line into this project's shell
        # dies on an import — the receipt has to say which environment produced it.
        cmd = (f"PYTHONPATH='{pin['path']}' '{sys.executable}' "
               f"bench/eval_laya_row_outcomes.py {flags}")
        print(f"laya {pin['version']} at {pin['path']} commit {pin['commit'][:12]} "
              f"· python {pin['python']} torch {pin['torch']}", flush=True)
        router, warned = build_router(args.device)
        for w in warned:
            print(f"  laya warned: {w}", flush=True)

    if not args.from_raw:
        rows, errors = [], []
        seen = identify(path)
        t0 = time.time()
        # The whole loop is inside the capture, not just the constructor: `Router` builds its
        # `Agent` lazily, so the warning about this checkpoint's out-of-clamp temperatures
        # arrives on the first `predict` — after `build_router` has already left its own
        # `catch_warnings`. Measured the hard way: the first pass recorded an empty
        # `laya_warnings` while the warning printed to stderr, and `bench/significance.py`
        # cites that field for the sentence "only correctness is comparable here". "once" so a
        # per-row warning does not fill the list 1,176 times, and each new one is printed the
        # moment it appears so the log still reads in order.
        import warnings as _w
        with _w.catch_warnings(record=True) as live:
            _w.simplefilter("once")
            n_reported = 0
            for i, (meta, questions, state) in enumerate(seen, 1):
                try:
                    ans = router.predict(state, laya_questions(questions))["answers"]
                except Exception as e:  # truncated state or unsupported schema, as in eval_laya_real
                    errors.append({"id": meta.get("id"), "source": meta.get("source"),
                                   "error": f"{type(e).__name__}: {e}"})
                    continue
                rec = {}
                for qid, q in questions.items():
                    a = ans.get(qid)
                    if a is None:
                        errors.append({"id": meta.get("id"), "source": meta.get("source"),
                                       "error": f"no answer for {qid}"})
                        continue
                    ok, pred, conf = outcome(q, a)
                    rec[qid] = {"type": q["type"], "ok": ok, "gold": q["label"],
                                "pred": pred if isinstance(pred, (str, bool, int)) else str(pred),
                                "conf": None if conf is None else round(float(conf), 6)}
                rows.append({"meta": meta, "questions": rec})
                for entry in live[n_reported:]:
                    warned.append(f"{entry.category.__name__}: {entry.message}")
                    print(f"  laya warned: {entry.category.__name__}: {entry.message}", flush=True)
                n_reported = len(live)
                if i % args.every == 0:
                    print(f"  {i} rows in {time.time() - t0:.0f}s "
                          f"({(time.time() - t0) / i * 1000:.0f} ms/row, {len(errors)} "
                          f"unanswered)", flush=True)

        n_q = sum(len(r["questions"]) for r in rows)
        print(f"{len(rows)} rows answered / {n_q} question outcomes in "
              f"{time.time() - t0:.0f}s on {args.device} · {len(errors)} rows/questions failed",
              flush=True)

        # The outcomes are written before anything is aggregated. A pass over 1,176 rows is
        # minutes of someone else's model, and a bug in a summary line must never be a reason
        # to pay it twice: the witness deletes this sidecar and `--from-raw` reads it back.
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps({"cmd": cmd, "engine": "laya", "laya": pin,
                                       "device": args.device, "laya_warnings": warned,
                                       "suite": str(args.suite), "split": args.split,
                                       "file": fname, "n_rows": len(rows),
                                       "n_rows_in_file": len(seen), "n_questions": n_q,
                                       "errors": errors,
                                       "seconds": round(time.time() - t0, 1),
                                       "rows": rows}, indent=1) + "\n")
        print(f"wrote the raw outcomes to {raw_path} before aggregating", flush=True)
        device_used, inference_s = args.device, round(time.time() - t0, 1)
    else:
        n_q = sum(len(r["questions"]) for r in rows)
        seen = identify(path)
        device_used, inference_s = raw["device"], raw["seconds"]

    got = {(r["meta"]["id"], name) for r in rows for name in r["questions"]}
    missing = sorted(want - got)
    if sorted(got - want):
        print(f"GUARD FAIL: {len(got - want)} (row, question) pairs are in the laya pass and "
              f"not in the myna dump (first {sorted(got - want)[:3]})", file=sys.stderr)
        return 3

    # Cells, merged across types exactly the way `myna.report.laya_cells` merges the
    # published column: a cell is (source, question), and a question asked two ways inside
    # one source is one cell weighted by rows — not whichever row came last (§9.32).
    cells = defaultdict(lambda: {"n": 0, "hits": 0, "parts": defaultdict(lambda: [0, 0])})
    for r in rows:
        for name, q in r["questions"].items():
            c = cells[(r["meta"]["source"], name)]
            c["n"] += 1
            c["hits"] += q["ok"]
            p = c["parts"][q["type"]]
            p[0] += q["ok"]
            p[1] += 1
    cell_rows = [{"source": s, "question": q, "n": c["n"], "hits": c["hits"],
                  "acc": c["hits"] / c["n"],
                  "types": {t: {"acc": h / n, "n": n} for t, (h, n) in sorted(c["parts"].items())}}
                 for (s, q), c in sorted(cells.items())]
    published = {(c["source"], c["question"]): c for c in mine["cells"]}
    geom = []
    for c in cell_rows:
        p = published.get((c["source"], c["question"]))
        if p is None:
            geom.append(f"laya scored a cell myna does not have: {c['source']}/{c['question']}")
        elif c["n"] != p["n"]:
            geom.append(f"{c['source']}/{c['question']}: laya has {c['n']} question instances, "
                        f"the myna dump has {p['n']}")
    for key in published:
        if not any((c["source"], c["question"]) == key for c in cell_rows):
            geom.append(f"{key[0]}/{key[1]} is in the myna dump and absent from laya")

    drift = None
    if args.compare:
        cmp_path = Path(args.compare)
        if cmp_path.exists():
            ref = json.loads(cmp_path.read_text())
            ids = subset_ids(path, ref["n_per_source"], ref["seed"])
            acc = defaultdict(lambda: [0, 0])
            for r in rows:
                if r["meta"]["id"] not in ids:
                    continue
                for name, q in r["questions"].items():
                    a = acc[(r["meta"]["source"], name, q["type"])]
                    a[0] += q["ok"]
                    a[1] += 1
            ref_rows = {(x["source"], x["qid"], x["type"]): x for x in ref["rows"]}
            per = []
            for key, (h, n) in sorted(acc.items()):
                r0 = ref_rows.get(key)
                now = h / n
                per.append({"cell": "/".join(key), "n": n, "acc_now": now,
                            "acc_committed": None if r0 is None else r0["acc"],
                            "delta": None if r0 is None else now - r0["acc"]})
            drift = {"committed": str(cmp_path), "committed_split": ref["split"],
                     "n_per_source": ref["n_per_source"], "seed": ref["seed"],
                     "rows_in_subset": len(ids), "cells": per,
                     "max_abs_delta": max((abs(x["delta"]) for x in per
                                           if x["delta"] is not None), default=None)}
            print(f"subset check vs {cmp_path.name}: {len(per)} (source, qid, type) rows on "
                  f"{len(ids)} rows, largest per-cell |delta| "
                  f"{drift['max_abs_delta']}", flush=True)

    if geom:
        for line in geom:
            print(f"GUARD FAIL: {line}", file=sys.stderr)
        print("nothing was written", file=sys.stderr)
        return 3

    payload = {"cmd": cmd, "engine": "laya", "laya": pin, "device": device_used,
               "laya_warnings": warned,
               "suite": str(args.suite), "split": args.split, "file": fname,
               "n_rows": len(rows), "n_questions": n_q,
               "n_rows_in_file": len(seen),
               "errors": errors, "cluster_key": "meta.id",
               "myna_dump": str(dump_path), "myna_n_rows": mine["n_rows"],
               "paired_cells": len(cell_rows),
               "cells": cell_rows, "subset_check": drift, "rows": rows,
               "seconds": inference_s}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    # The sidecar is the expensive object here, so it is deleted only once the witness has
    # been read back and shown to carry the rows. `bench/significance.py` joins on
    # `(meta.id, question)`: a witness of cell means alone cannot be paired, and the first
    # run of this file wrote exactly that — no `rows` key — printed "the witness carries the
    # rows now", and unlinked the sidecar that held them. The claim below is now taken from
    # the bytes on disk rather than from the payload built in memory.
    back = json.loads(out.read_text())
    if len(back.get("rows", [])) != len(rows):
        print(f"GUARD FAIL: {out} carries {len(back.get('rows', []))} rows and this pass has "
              f"{len(rows)}, so the per-question outcomes are not in the witness. "
              f"{raw_path.name} is being kept — significance.py cannot read cell means",
              file=sys.stderr)
        return 3
    raw_path.unlink()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB) holding {len(back['rows'])} rows "
          f"/ {sum(len(r['questions']) for r in back['rows'])} question outcomes, removed the "
          f"{raw_path.name} sidecar — the witness carries the rows")
    for c in cell_rows:
        print(f"  {c['source']}/{c['question']:<14s} n={c['n']:>4d} acc={c['acc']:.4f}  "
              + " ".join(f"{t}:{v['acc']:.3f}({v['n']})" for t, v in c["types"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

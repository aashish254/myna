"""V2-A: an upstream training slice for decision-v2, deconflicted against the frozen eval.

decision-v2 froze ~500 rows per source as a *benchmark*. Its manifest lists all 11
sources as trainable, has `holdout_sources: []`, and explicitly disclaims fuzzy
decontamination -- so the upstream corpora behind those sources are in scope for
training, and the 500-row freeze is a sampling budget, not a licence limit.

This pulls a larger slice from the **same pinned revisions**, through **kev's own
converters**, so the question definitions, option-description phrasing pools,
provenance fields and the tokenizer admission policy are identical to the frozen
suite rather than an approximation of it. Then it drops any state whose normalized
text hash already appears in development/test/calibration and **prints the
collision counts** -- there is no leak claim without that number.

Output is a suite-shaped directory that `myna.train --suite` consumes unchanged:

    data/decision-v2-pilot/train.jsonl              upstream slice + the frozen train rows (disjoint)
    data/decision-v2-pilot/development.jsonl        byte copy of the frozen split
    data/decision-v2-pilot/test.jsonl               byte copy
    data/decision-v2-pilot/calibration.jsonl        byte copy (temperature fit; never trained on)
    data/decision-v2-pilot/upstream_manifest.json   this pull's provenance

`datasets` is a data-prep dependency, not a myna runtime one, so run this under an
env that has it (kev's own venv) with kev importable:

  cd "/Users/aashish/next ko ne next gen model" && \
  KEV_ROOT="/Users/aashish/github contribution /GIT/kev" \
    "$KEV_ROOT/.venv/bin/python" -m bench.pull_upstream --per-source 5000

  --per-source 5            smoke run: same code path, tiny slice
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import sys
from collections import Counter
from pathlib import Path

DEFAULT_SUITE = "/Users/aashish/github contribution /GIT/kev/evals/decision-v2"


def import_kev(kev_root):
    """kev is a data-prep dependency here, not a myna one: use it as installed if it is
    already importable, else insert the checkout named by --kev-root / $KEV_ROOT."""
    if not kev_root:
        try:
            import kev  # noqa: F401
        except ImportError:
            sys.exit("kev is not importable. Run under kev's venv, or pass --kev-root "
                     "pointing at a kev checkout that has `datasets` installed.")
    else:
        sys.path.insert(0, str(Path(kev_root).resolve()))
    from kev import contrastive, data, model, suite
    return data, model, suite, contrastive


def state_text_of(record):
    """The text a state's dedup hash is computed over: the raw string, or the canonical
    JSON of a structured state (the convention kev's own load_records uses)."""
    s = record["state"]
    return s if isinstance(s, str) else json.dumps(s, sort_keys=True, ensure_ascii=False)


def state_digest(record):
    """sha256 of the whitespace-collapsed, casefolded *state exactly as the model reads it*.

    The suite's own `_meta.text_sha256` hashes an upstream *field* instead, and the two
    keys disagree in both directions: for yelp/imdb/amazon the hashed field is the review
    *before* the converter cuts it to 220 words (two different reviews sharing a prefix are
    two hashes for one state), while for contrastive it covers the case sentences but not
    the policy (one hash for two states that differ). The rendered state is the primary key
    here because it is what an eval row can be memorized from."""
    return hashlib.sha256(" ".join(state_text_of(record).casefold().split()).encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def partition_keys(suite_dir, names):
    """(rendered-state digests, upstream-text digests) for every record in the named
    partitions -- both keys, because neither alone is a leak claim.

    The state digest catches "the model would see this exact input". The text digest
    catches the same content re-expressed: `_wrap_state` renders a given review as a bare
    string, a `{"document": ...}` dict, a ticket or a chat turn, and those are different
    states with the same underlying text. Blocking on the union costs a few hundred rows
    out of a 15k pool and makes the printed collision count mean both things."""
    states, texts = set(), set()
    for name in names:
        p = Path(suite_dir) / f"{name}.jsonl"
        if p.exists():
            for r in read_jsonl(p):
                states.add(state_digest(r))
                texts.add(r["_meta"]["text_sha256"])
    return states, texts


def record_keys(record):
    return state_digest(record), record["_meta"]["text_sha256"]


def filter_pool(pool, blocked_eval, blocked_frozen, admitted_states, col):
    """Drop pool rows that collide with the eval partitions or the frozen train on *either*
    key, plus rows whose rendered state this pull has already claimed. `col` gets the counts
    -- the collision numbers are the publishable part of this pipeline, not a side effect."""
    keep = []
    for r in pool:
        state, text = record_keys(r)
        if state in blocked_eval[0] or text in blocked_eval[1]:
            col["eval_collision"] += 1
            continue
        if state in blocked_frozen[0] or text in blocked_frozen[1]:
            col["in_frozen_train"] += 1
            continue
        if state in admitted_states:
            col["duplicate_state"] += 1
            continue
        admitted_states.add(state)
        keep.append(r)
    return keep


def admit(records, count, seen, tokenizers, report, max_branch, data, model):
    """Local mirror of kev.suite.select_unique -- same selection order, same dedup key,
    same admission predicate -- that returns a shortfall instead of raising, so one thin
    source cannot abort a ten-source pull. tests/test_upstream.py pins the two against
    each other on identical inputs."""
    selected = []
    for record in sorted(records, key=lambda r: r["_meta"]["row_sha256"]):
        report["considered"] += 1
        key = record["_meta"]["text_sha256"]
        if key in seen:
            report["duplicate_state"] += 1
            continue
        if not model.fits(data.materialize(record), *tokenizers, max_branch=max_branch):
            report["context_rejected"] += 1
            continue
        record["_meta"].update(group_id=record["_meta"]["id"], variant="clean")
        seen.add(key)
        selected.append(record)
        report["accepted"] += 1
        if len(selected) == count:
            return selected
    report["shortfall"] = count - len(selected)
    return selected


def pull_contrastive(pairs_per_family, seed, families, admitted_states, blocked_eval,
                     blocked_train, tokenizers, max_branch, data, model, contrastive):
    """Generated, not downloaded. Siblings are kept or dropped together: a pair missing
    one side is still a trainable row, but the per-family report would then lie about it."""
    records, report = contrastive.generate(pairs_per_family, seed=f"{seed}-train", families=families)
    collisions = Counter()
    kept = []
    for pair in zip(records[::2], records[1::2]):
        keys = [record_keys(r) for r in pair]
        if any(state in blocked_eval[0] or text in blocked_eval[1] for state, text in keys):
            collisions["eval_collision"] += 1
            continue
        if any(state in blocked_train[0] or text in blocked_train[1] for state, text in keys):
            collisions["in_frozen_train"] += 1
            continue
        if any(state in admitted_states for state, _ in keys):
            collisions["duplicate_state"] += 1
            continue
        if not all(model.fits(data.materialize(r), *tokenizers, max_branch=max_branch) for r in pair):
            collisions["context_rejected"] += 1
            continue
        for r in pair:
            r["_meta"].update(group_id=r["_meta"]["family_id"], variant="clean")
        admitted_states.update(state for state, _ in keys)
        kept += list(pair)
    summary = {"pairs_generated": sum(v["pairs"] for v in report.values()),
               "pairs_rejected_by_checks": sum(v["rejected"] for v in report.values()),
               "rows_admitted": len(kept)}
    return kept, summary, dict(collisions)


def token_stats(records, tokenizer):
    """State lengths in *the suite's* units, so the pilot's length distribution can be
    compared with what the frozen eval actually contains."""
    lens = sorted(len(tokenizer(state_text_of(r), add_special_tokens=False)["input_ids"]) for r in records)
    if not lens:
        return {}
    return {"n": len(lens), "median": lens[len(lens) // 2],
            "p95": lens[int(0.95 * (len(lens) - 1))], "max": lens[-1], "sum": sum(lens)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=os.environ.get("DECISION_V2_DIR", DEFAULT_SUITE),
                    help="frozen decision-v2 directory (manifest + partitions)")
    ap.add_argument("--out", default="data/decision-v2-pilot")
    ap.add_argument("--kev-root", default=os.environ.get("KEV_ROOT"))
    ap.add_argument("--per-source", type=int, default=5000,
                    help="upstream rows to admit per HF source (the freeze itself took 340)")
    ap.add_argument("--contrastive-pairs", type=int, default=600,
                    help="contrastive pairs per trainable family (the freeze used 60); each pair is 2 rows")
    ap.add_argument("--without-suite-train", action="store_true",
                    help="do not copy the frozen train partition into the output")
    ap.add_argument("--only", help="comma-separated source subset (each upstream repo downloads in full; "
                                   "use this to exercise one converter before pulling the whole slice)")
    args = ap.parse_args(argv)
    # the manifest is this pull's provenance (SPEC §4.1 reads it rather than a log),
    # and --per-source/--contrastive-pairs decide how many rows exist
    cmd = shlex.join(["python", "bench/pull_upstream.py",
                     *(argv if argv is not None else sys.argv[1:])])

    data, model, suite, contrastive = import_kev(args.kev_root)
    suite_dir = Path(args.suite).resolve()
    man = json.loads((suite_dir / "manifest.json").read_text(encoding="utf-8"))
    seed = man["seed"]
    available = [s for s in man["trainable_sources"] if s != "contrastive"]
    if args.only:
        wanted = [x for x in args.only.split(",") if x]
        if unknown := [x for x in wanted if x not in available]:
            sys.exit(f"--only names must be trainable HF sources of this suite: {unknown} "
                     f"(available: {available})")
        available = wanted
    names = available
    if missing := [n for n in names if n not in data.ALL_SOURCES]:
        sys.exit(f"kev has no converter for: {missing}")
    sources = {k: data.ALL_SOURCES[k] for k in names}
    repos = {k: data.ALL_REPOS[k] for k in names}
    revisions = man["dataset_revisions"]
    if unbound := [repos[n] for n in names if repos[n] not in revisions]:
        sys.exit(f"the manifest pins no revision for: {unbound}")
    if man["context"]["max_state"] != model.MAX_STATE:
        sys.exit(f"this pull assumes the default training context ({model.MAX_STATE}); "
                 f"the suite admitted at {man['context']['max_state']}")
    # the same branch headroom the freeze used, so every pilot row is one the suite itself
    # would have admitted (the variants it builds add an option and must still encode).
    max_branch = suite.MAX_BRANCH - suite.ADMISSION_BRANCH_HEADROOM
    tokenizers = [model.load_tokenizer(base, revision=rev) for base, rev in man["base_revisions"].items()]

    eval_states, eval_texts = partition_keys(suite_dir, ("development", "test", "calibration"))
    frozen_states, frozen_texts = partition_keys(suite_dir, ("train",))

    admitted_states = set()  # state digests this pull has already taken (cross-source)
    seen_text = set()        # the suite's own key, handed to admit() for select_unique parity
    rows, reports = [], {}

    def counts(source, pool, considered, admitted, col, rep):
        """One normalized row per source, so the printed table cannot hide a column."""
        return {"source": source, "pool": pool, "considered": considered,
                "eval_collision": col.get("eval_collision", 0),
                "in_frozen_train": col.get("in_frozen_train", 0),
                "duplicate_state": col.get("duplicate_state", 0),
                "context_rejected": col.get("context_rejected", rep.get("context_rejected", 0)),
                "admitted": admitted, "shortfall": rep.get("shortfall", 0),
                "admission_report": dict(rep), "state_counts": dict(col)}

    for name in names:
        pool = data.build(max(3 * args.per_source, 800), "train", seed, only=[name],
                          revisions=revisions, sources=sources, repos=repos)
        rep, col = Counter(), Counter()
        # a state claimed here stays claimed even if admission rejects the row later: the pull
        # must never hand the model the same input twice under a different hash.
        pre = filter_pool(pool, (eval_states, eval_texts), (frozen_states, frozen_texts),
                          admitted_states, col)
        chosen = admit(pre, args.per_source, seen_text, tokenizers, rep, max_branch, data, model)
        rows += chosen
        reports[name] = counts(name, len(pool), len(pre), len(chosen), col, rep)

    if args.only:  # a converter check; the generated source needs no download and no check
        reports["contrastive"] = counts("contrastive", 0, 0, 0, Counter(), Counter({"skipped": True}))
        crows = []
    else:
        fams = man["contrastive"]["trainable_families"]
        crows, csum, ccol = pull_contrastive(
            args.contrastive_pairs, seed, fams, admitted_states, (eval_states, eval_texts),
            (frozen_states, frozen_texts), tokenizers, max_branch, data, model, contrastive)
        reports["contrastive"] = counts("contrastive", csum["pairs_generated"] * 2,
                                        csum["pairs_generated"] * 2, csum["rows_admitted"],
                                        Counter(ccol), Counter(csum))
        reports["contrastive"]["pairs_rejected_by_checks"] = csum["pairs_rejected_by_checks"]
    rows += crows

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    upstream = sorted(rows, key=lambda r: (r["_meta"]["source"], r["_meta"]["row_sha256"]))
    for r in upstream:
        r["_meta"]["pilot"] = True  # marks this pull's rows; the frozen copies stay byte-verbatim
    frozen = [] if args.without_suite_train else read_jsonl(suite_dir / "train.jsonl")
    train = upstream + frozen
    train_path = out / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for split in ("development", "test", "calibration"):
        shutil.copyfile(suite_dir / f"{split}.jsonl", out / f"{split}.jsonl")

    # the gate: re-read the file from disk and prove what it claims on BOTH keys. Nothing
    # downstream may assert "no eval overlap" until this prints zeros.
    written = read_jsonl(train_path)
    pulled = [r for r in written if r["_meta"].get("pilot")]
    copied = [r for r in written if not r["_meta"].get("pilot")]
    pulled_keys = [record_keys(r) for r in pulled]
    copied_keys = [record_keys(r) for r in copied]
    leak_eval = (len({s for s, _ in pulled_keys} & eval_states)
                 + len({t for _, t in pulled_keys} & eval_texts))
    leak_train = (len({s for s, _ in pulled_keys} & {s for s, _ in copied_keys})
                  + len({t for _, t in pulled_keys} & {t for _, t in copied_keys}))
    dup_within = len(pulled_keys) - len({s for s, _ in pulled_keys})
    round_trip = len(written) == len(train)
    if leak_eval or leak_train or dup_within or not round_trip:
        sys.exit(f"GATE FAILED: {leak_eval} eval-state/text matches, {leak_train} frozen-train "
                 f"matches, {-dup_within} repeated states inside the pull, "
                 f"read-back {len(written)} rows vs {len(train)} written")

    toks = token_stats(train, tokenizers[0])
    (out / "upstream_manifest.json").write_text(json.dumps({
        "cmd": cmd,
        "derived_from": str(suite_dir),
        "seed": seed,
        "converters": "kev.data.build + kev.contrastive.generate at the pinned suite above",
        "dataset_revisions": {repos[n]: revisions[repos[n]] for n in names},
        "base_revisions": man["base_revisions"],
        "admission": {"max_branch": max_branch, "context": man["context"]},
        "per_source_target": args.per_source,
        "contrastive_pairs_per_family": args.contrastive_pairs,
        "includes_frozen_suite_train": not args.without_suite_train,
        "dedup_key": "blocked on BOTH: sha256 of the casefolded, whitespace-collapsed rendered state "
                     "(state_digest) and the suite's own _meta.text_sha256 upstream-field hash",
        "records": {"train": len(train), "upstream": len(upstream), "frozen_suite_train": len(frozen),
                    "questions": sum(len(r["questions"]) for r in train)},
        "state_tokens_under_first_base": toks,
        "dedup": {"excluded_against": ["development", "test", "calibration"],
                  "eval_states": len(eval_states), "eval_texts": len(eval_texts),
                  "frozen_train_states": len(frozen_states), "frozen_train_texts": len(frozen_texts)},
        "verification": {"rows_re_read": len(written), "eval_matches": leak_eval,
                         "frozen_train_matches": leak_train,
                         "repeated_states_within_pull": dup_within,
                         "round_trip_exact": round_trip},
        "reports": reports,
        "sha256_train": hashlib.sha256(train_path.read_bytes()).hexdigest(),
    }, indent=2) + "\n", encoding="utf-8")

    print(f"{'source':<13}{'pool':>8}{'eval':>6}{'in-tr':>7}{'dup':>6}{'ctx':>6}{'admit':>7}{'short':>7}", flush=True)
    for name in names + ["contrastive"]:
        r = reports[name]
        print(f"{name:<13}{r['pool']:>8}{r['eval_collision']:>6}{r['in_frozen_train']:>7}"
              f"{r['duplicate_state']:>6}{r['context_rejected']:>6}{r['admitted']:>7}{r['shortfall']:>7}", flush=True)
    print(f"\ntrain rows {len(train)} = upstream {len(upstream)} + frozen {len(frozen)}"
          f"   questions {sum(len(r['questions']) for r in train)}", flush=True)
    print(f"state tokens under {list(man['base_revisions'])[0]}: median {toks.get('median')} "
          f"p95 {toks.get('p95')} max {toks.get('max')}  SUM {toks.get('sum', 0):,}", flush=True)
    print(f"VERIFIED: {leak_eval} state/text matches against development/test/calibration, "
          f"{leak_train} against the frozen train, {dup_within} repeated states inside the pull "
          f"(all three must be 0)  [{len(pulled)} pull + {len(copied)} frozen rows read back]",
          flush=True)


if __name__ == "__main__":
    main()

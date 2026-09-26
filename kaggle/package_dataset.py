"""Package the pilot corpus as the Kaggle dataset the training box mounts (P3 3b).

Kaggle cannot see this laptop, and there is no git remote yet (SPEC §5 P0: the
largest unmanaged risk), so the *data* half of the job travels as a dataset and
the *code* half as the kernel's uploaded source. This script builds the data half
and proves the copy is the same bytes:

    python kaggle/package_dataset.py                 # stage into .kaggle-dataset/
    python kaggle/package_dataset.py --dest /tmp/pilot
    python kaggle/package_dataset.py --verify-only   # re-hash an existing staging dir

It does **not** upload. `kaggle datasets create` needs credentials, publishes to a
shared account, and is exactly the kind of side effect that stays a human
decision; the script prints the command instead.

The hash matters more than it looks: the whole acceptance argument is "the same
data, the same splits". If the Kaggle copy were a different slice, G1 would be
measured on a corpus this loop never sized against, and the memory plan's p95
would be describing someone else's rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "data" / "decision-v2-pilot"
FILES = ("train.jsonl", "development.jsonl", "test.jsonl", "calibration.jsonl")
EXTRA = ("tokenizer-8192.json", "upstream_manifest.json")
DATASET_ID = "aashish254/decision-v2-pilot"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(root: Path) -> dict:
    out = {}
    for name in sorted(p.name for p in root.iterdir() if p.is_file()):
        p = root / name
        with open(p, "rb") as f:
            rows = sum(1 for line in f if line.strip()) if p.suffix == ".jsonl" else None
        out[name] = {"bytes": p.stat().st_size, "sha256": sha256(p), **({"rows": rows} if rows else {})}
    return out


def build(dest: Path, source: Path) -> dict:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in FILES + EXTRA:
        if (source / name).exists():
            shutil.copy2(source / name, dest / name)
    manifest = {"dataset": source.name, "files": inventory(source)}
    (dest / "SOURCE_SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (dest / "dataset-metadata.json").write_text(json.dumps(
        {"ownerSlug": DATASET_ID.split("/")[0], "datasetSlug": DATASET_ID.split("/")[1],
         "title": "myna decision-v2 pilot slice (kev frozen suite, re-expressed)",
         "isPrivate": True, "resources": [
            {"path": name, "description": f"decision-v2 pilot: {name}", "fileType": "jsonl"
             if name.endswith(".jsonl") else "json"} for name in FILES + EXTRA]}, indent=2) + "\n")
    return manifest


def check(dest: Path, source: Path) -> list[str]:
    """Return problems; empty means the staged copy *is* the on-disk corpus."""
    problems = []
    for name in FILES:
        a, b = source / name, dest / name
        if not b.exists():
            problems.append(f"{name}: missing from {dest}")
            continue
        if sha256(a) != sha256(b):
            problems.append(f"{name}: sha256 differs from {source}")
        elif a.stat().st_size != b.stat().st_size:
            problems.append(f"{name}: size differs")
    meta = dest / "dataset-metadata.json"
    if not meta.exists():
        problems.append("dataset-metadata.json: absent, `kaggle datasets create` would fail")
    else:
        d = json.loads(meta.read_text())
        if d.get("datasetSlug") != DATASET_ID.split("/")[1]:
            problems.append(f"dataset-metadata.json: slug {d.get('datasetSlug')!r}")
        listed = {r["path"] for r in d.get("resources", [])}
        if not set(FILES) <= listed:
            problems.append(f"dataset-metadata.json: resources {sorted(listed)} omit a split")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="stage the pilot corpus for Kaggle (no upload)")
    ap.add_argument("--dest", default=".kaggle-dataset/decision-v2-pilot")
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the inventory as json")
    args = ap.parse_args(argv)
    dest, source = Path(args.dest), Path(args.source)
    if not (source / "train.jsonl").exists():
        raise SystemExit(f"{source} has no train.jsonl — run bench/pull_upstream.py first")
    if args.verify_only:
        problems = check(dest, source)
        print(f"{'OK' if not problems else 'PROBLEMS'}: {dest}")
        for p in problems:
            print(f"  {p}")
        return 1 if problems else 0
    manifest = build(dest, source)
    problems = check(dest, source)
    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        rows = {n: v.get("rows") for n, v in manifest["files"].items() if n in FILES}
        total = sum(v["bytes"] for v in manifest["files"].values())
        print(f"staged {len(manifest['files'])} files / {total / 1e6:.1f} MB into {dest}")
        print(f"rows: {rows}")
    if problems:
        for p in problems:
            print(f"  PROBLEM {p}")
        return 1
    print("verified: every staged file hashes equal to the on-disk corpus")
    print(f"upload when you decide to:  kaggle datasets create -p {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

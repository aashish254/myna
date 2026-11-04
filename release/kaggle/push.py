"""Push the baseline kernel to Kaggle, poll it, and download the outputs.

    KAGGLE_USERNAME=... KAGGLE_KEY=... python release/kaggle/push.py --wait

Requires `pip install kaggle`. Credentials come from the environment and are never written
to disk by this script.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNEL_DIR = ROOT / "release" / "kaggle"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="<username>/<kernel-slug>")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--timeout", type=int, default=8 * 3600)
    ap.add_argument("--out", default="runs/kaggle")
    args = ap.parse_args()

    meta = json.loads((KERNEL_DIR / "kernel-metadata.json").read_text())
    meta["id"] = args.slug
    meta["code_file"] = "run.py"
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))

    subprocess.run(["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)], check=True)
    if not args.wait:
        print("pushed; poll with: kaggle kernels status", args.slug)
        return

    t0 = time.time()
    while time.time() - t0 < args.timeout:
        status = subprocess.run(["kaggle", "kernels", "status", args.slug],
                                capture_output=True, text=True).stdout.strip()
        print(f"[{int(time.time()-t0)}s] {status}", flush=True)
        if "complete" in status.lower() or "error" in status.lower():
            break
        time.sleep(args.poll)

    dest = ROOT / args.out
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", args.slug, "-p", str(dest)], check=False)
    print("downloaded to", dest)


if __name__ == "__main__":
    main()

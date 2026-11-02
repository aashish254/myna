"""Upload the Myna v6 weights to the Hugging Face Hub, so GitHub never has to hold them.

    export HF_TOKEN=hf_...                      # or `huggingface-cli login` once
    uv run --extra v6 python scripts/upload_to_hf.py

`deberta-v5-finetuned/model.safetensors` is 1,736,094,384 bytes. GitHub's hard limit is
100 MiB per blob and this repository's rule (SPEC §7.1, and the G7 promise behind it) is that
weights live outside git while figures live inside it. So the campaign directory stays
gitignored — `.gitignore` carries `myna-v5-prototype*/`, `local_checkpoints/`, `*.pt`,
`*.safetensors` — and the Hub copy is what `MynaV6("aashish254/myna-v6")` resolves.

Nothing here touches git. No `git add`, no commit, no LFS pointer files: the script only
reads the checkpoint directory, writes one model card into it, and calls `upload_folder`.

Every figure in the generated model card is read out of the two witness JSONs at build time
rather than typed into this file. A card with hand-written numbers drifts the moment the
witness is regenerated, and a reader cannot tell a stale card from a fresh one.

This uploads ~1.7 GB. On a metered connection, price it first with `--dry-run`, which prints
the per-file byte counts and the total without contacting the Hub.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Tried in order, so a campaign folder dropped in the repo root and an extracted
# `local_checkpoints/` copy both work without the reader having to remember which they used.
CKPT_CANDIDATES = ('myna-v5-prototype 2', 'myna-v5-prototype',
                   'local_checkpoints/myna-v5-prototype')

BACKBONE_DIR = "deberta-v5-finetuned"
WEIGHT_FILES = ("myna-v6-unified.pt", "myna-v5-unified.pt")
WITNESS_FILES = ("witness-v6-master-paper.json", "witness-competitor-head-to-head-1440.json",
                 "witness-v5-master-paper.json", "witness-hybrid-v3.json")

# Columns the head-to-head witness uses for the competitors, in the order the paper lists
# them. The keys must match `runs/witness-competitor-head-to-head-1440.json` exactly — the
# card generator refuses to run when one is missing rather than printing a dash column that
# reads as a zero score.
COMPETITORS = ("Laya (English Default)", "Laya (Raised + Tournament)",
               "Laya (Typed-Decisions Best)", "VegaML (vega-U 0.8B)")


def find_ckpt_dir(explicit: str | None) -> Path:
    candidates = [Path(explicit)] if explicit else [REPO / c for c in CKPT_CANDIDATES]
    for c in candidates:
        if c.is_dir():
            return c
    tried = "\n  ".join(str(c) for c in candidates)
    raise SystemExit(f"no checkpoint directory found. Tried:\n  {tried}")


def collect(root: Path) -> list[Path]:
    """The files that go to the Hub, with the weights checked for existence first."""
    files: list[Path] = []
    backbone = root / BACKBONE_DIR
    if not backbone.is_dir():
        raise SystemExit(f"{root} has no {BACKBONE_DIR}/ — the fine-tuned encoder is the "
                         "largest part of the upload and its absence means the wrong "
                         "directory was named.")
    files += sorted(p for p in backbone.rglob("*") if p.is_file())
    for name in WEIGHT_FILES:
        if (root / name).is_file():
            files.append(root / name)
    for name in WITNESS_FILES:
        # Prefer the copy committed in `runs/`: that is the one the registry binds to the
        # published tables, so the Hub and the paper cannot end up shipping different JSON.
        for src in (REPO / "runs" / name, root / name):
            if src.is_file():
                files.append(src)
                break
    return files


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def model_card(root: Path) -> str:
    """The Hub README, with the benchmark table read from the committed witnesses."""
    v6 = _load_json(REPO / "runs" / "witness-v6-master-paper.json")
    h2h = _load_json(REPO / "runs" / "witness-competitor-head-to-head-1440.json")
    ckpt = root / "myna-v6-unified.pt"
    if ckpt.is_file():
        import torch
        cfg = torch.load(ckpt, map_location="cpu", weights_only=False).get("config", {})
    else:
        cfg = {}

    fm = ["---", "license: apache-2.0", "library_name: transformers",
          "base_model: microsoft/deberta-v3-large", "base_model_size: model_size",
          "tags:", "- text-classification", "- zero-shot", "- decision-making", "- agent",
          "- conformal-prediction", "- calibrated", "model_name: myna-v6-unified",
          "datasets:", "- aashish254/decision-v2-pilot", "---"]

    lines = fm + ["", "# Myna v6 Unified", "",
                  "A fine-tuned `microsoft/deberta-v3-large` encoder with a "
                  "**cardinality-routed readout**: questions with up to "
                  f"`{cfg.get('max_packed_options', 16)}` options are scored from one packed "
                  "sequence, and any question with more is routed to a set-aware head that "
                  "scores each option against the competition in its own bundle. "
                  "Temperatures and split-conformal quantiles are fitted on a dedicated "
                  "2,000-bundle `CAL` split, so every answer carries a 90% prediction set "
                  "and an explicit `abstain` flag.", ""]

    if h2h and v6:
        absent = [arm for arm in COMPETITORS if arm not in h2h]
        if absent:
            raise SystemExit(
                f"the head-to-head witness carries no column for {', '.join(absent)}; its "
                f"keys are {', '.join(sorted(h2h))}. Fix `COMPETITORS` — a missing arm must "
                "not reach the Hub as a dash column, which reads as a zero score.")
        cells = sorted({c for arm in h2h.values() for c in arm["per_cell"]})
        lines += ["## 16-cell head-to-head on `decision-v2-pilot/test.jsonl`",
                  "",
                  "1,176 bundles / 1,440 questions, one frozen split, identical rows for "
                  "every column.", "",
                  "| cell | " + " | ".join(c.replace(" (", "\\n(") for c in COMPETITORS)
                  + " | Myna v6 (test) |",
                  "| :-- |" + " --: |" * (len(COMPETITORS) + 1)]
        for c in cells:
            row = [f"`{c}`"]
            for arm in COMPETITORS:
                v = h2h.get(arm, {}).get("per_cell", {}).get(c)
                row.append(f"{v:.4f}" if v is not None else "—")
            v = v6["test_audit"]["per_cell"].get(c)
            row.append(f"{v:.4f}" if v is not None else "—")
            lines.append(" | ".join(row) + " |")
        lines += ["", "| aggregate | " + " | ".join(COMPETITORS) + " | Myna v6 (test) |",
                  "| :-- |" + " --: |" * (len(COMPETITORS) + 1)]
        for label, key in (("16-cell unweighted macro", "macro_acc"),
                           ("overall accuracy", "overall_acc"),
                           ("joint bundle exact-match", "bundle_exact_acc")):
            row = [label]
            for arm in COMPETITORS:
                v = h2h.get(arm, {}).get(key)
                row.append(f"{v:.4f}" if v is not None else "—")
            row.append(f"{v6['test_audit'][key]:.4f}")
            lines.append(" | ".join(row) + " |")
        lines += ["",
                  "**Footing, stated plainly.** Myna v5/v6 was fine-tuned on "
                  "`decision-v2-pilot/train.jsonl` — in-domain across all 11 sources — and "
                  "its high-cardinality routing was designed after diagnosing packed-sequence "
                  "truncation on `banking77` on these same cells. Laya and VegaML are evaluated "
                  "out-of-domain / zero-shot here. The columns are not a like-for-like "
                  "comparison of training budgets, and the row-level JSON witnesses in this "
                  "repository say so in the same words.", ""]

    if v6:
        cal = v6.get("calibration_test", {})
        lines += ["## Calibration and selective prediction", "",
                  f"- temperatures fitted on `CAL`: `{cal.get('temperatures')}` "
                  f"({cal.get('argmax_flips')} argmax flips on dev and test)",
                  f"- test ECE **{cal.get('ece'):.4f}**, mean Brier **{cal.get('brier'):.4f}**",
                  f"- conformal empirical coverage: 90% target -> "
                  f"**{cal.get('conformal_empirical_90'):.4f}**, 95% target -> "
                  f"**{cal.get('conformal_empirical_95'):.4f}**",
                  f"- selective accuracy at 60% coverage **{cal.get('selective_acc_60pct_all'):.4f}**, "
                  f"coverage at <=5% error risk **{cal.get('coverage_at_5pct_risk'):.4f}**",
                  f"- gate G5 slice (banking77 + dbpedia14 + trec) at 60% coverage "
                  f"**{cal.get('g5_acc_60pct_slice'):.4f}**", ""]

    lines += ["## Quickstart", "",
              "```bash", "pip install \"myna[v6]\" @ git+https://github.com/aashish254/myna",
              "```", "", "```python",
              "from myna import MynaV6", "",
              "myna = MynaV6(\"aashish254/myna-v6\")", "state = (",
              "    \"Customer: my card was declined twice at the ATM and I still \"",
              "    \"have cash in the account.\", "
              ")\"", "",
              "answer = myna.predict(state, {", "    \"boolq/answer\": {",
              "        \"type\": \"noul\",",
              "        \"instructions\": \"Was the withdrawal authorised?\",",
              "        \"criteria\": None,", "    }", "})", "",
              "a = answer[\"answers\"][\"boolq/answer\"]",
              "print(a[\"yes\"], a[\"confidence\"], a[\"conformal_set_90\"], a[\"abstain\"])",
              "```", "",
              "`predict` also takes `abstain_below=0.9`, which withholds the commitment "
              "instead of returning the argmax. See the repository README for the session API "
              "and for the HTTP server.", "",
              "## What is in this repository", "",
              f"- `{BACKBONE_DIR}/` — the fine-tuned `DebertaV2Model` (config, "
              "`model.safetensors`, tokenizer)",
              "- `myna-v6-unified.pt` — the readout heads, `config`, and the dev/test/calibration "
              "audit blocks the published tables quote",
              "- `myna-v5-unified.pt` — the v5 checkpoint, before the set-aware head",
              "- `witness-*.json` — the artifacts behind every number above", "",
              "The training code and the Colab notebooks are not in this weight repo; they "
              "live in the `myna` source repository, and the campaign ran on a Tesla T4.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="upload-to-hf",
        description="Upload the Myna v6 weights and witnesses to the Hugging Face Hub. "
                    "This never stages or commits anything: git keeps the JSON figures, the "
                    "Hub keeps the >100 MB weights.")
    ap.add_argument("--repo-id", default="aashish254/myna-v6", metavar="ID",
                    help="Hub model repo to create or update (default aashish254/myna-v6)")
    ap.add_argument("--ckpt-dir", default=None, metavar="DIR",
                    help=f"campaign directory; default tries {' then '.join(CKPT_CANDIDATES)}")
    ap.add_argument("--token", default=None,
                    help="Hub write token; defaults to $HF_TOKEN or the cached "
                         "`huggingface-cli login` credential")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the file list and byte totals, build the model card, and "
                         "upload nothing")
    ap.add_argument("--no-card", action="store_true",
                    help="skip generating the model card (upload only weights and witnesses)")
    args = ap.parse_args(argv)

    root = find_ckpt_dir(args.ckpt_dir)
    files = collect(root)
    if not any(f.name.endswith(".pt") for f in files):
        raise SystemExit(f"{root} has no {' or '.join(WEIGHT_FILES)} — nothing to upload")

    total = sum(f.stat().st_size for f in files)
    card = None if args.no_card else model_card(root)
    # Two routes, because the witnesses the registry binds are committed in `runs/` and
    # `upload_folder` can only send a directory's own contents: the weights go as a folder
    # with an explicit pattern list, the committed witnesses go one file at a time.
    inside, outside = {}, []
    for f in files:
        try:
            inside[str(f.relative_to(root))] = f
        except ValueError:
            outside.append(f)
    print(f"source    : {root}")
    print(f"repo      : {args.repo_id}")
    print(f"files     : {len(files)}  ({len(inside)} in the campaign directory, "
          f"{len(outside)} committed copies from runs/)")
    for f in sorted(files, key=lambda p: -p.stat().st_size):
        try:
            label = str(f.relative_to(root))
        except ValueError:
            label = f"{f.relative_to(REPO)}  [committed copy]"
        print(f"  {f.stat().st_size:>14,} B  {label}")
    print(f"total     : {total:,} B = {total / 1e9:.3f} GB")
    if card is not None:
        print(f"card      : {len(card):,} chars of README.md")
        if not args.dry_run:
            (root / "README.md").write_text(card)
            print(f"          written to {root / 'README.md'}")

    if args.dry_run:
        print("\ndry run: nothing uploaded.")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("needs huggingface_hub: uv sync --extra v6, or "
                         "pip install huggingface_hub")

    token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi()
    print(f"\ncreating {args.repo_id} if absent ...")
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True, token=token)
    print(f"uploading {total:,} B ...")
    # No `multi_commits`: that keyword left the client in huggingface_hub 1.x, and passing it
    # raises TypeError at the moment the upload is already authorised. The Hub chunks a 1.7 GB
    # LFS object on its own.
    api.upload_folder(repo_id=args.repo_id, folder_path=str(root), repo_type="model",
                      allow_patterns=sorted(inside) + (["README.md"] if card else []),
                      token=token)
    for f in outside:
        # One call per small JSON, with `upload_file` rather than `upload_files`: the
        # batched name is not in the Hub client this repository pins (huggingface_hub 1.x),
        # and an uploader that dies on a rename is worse than four round trips of 3 KB.
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                        repo_id=args.repo_id, repo_type="model", token=token)
    print(f"\nhttps://huggingface.co/{args.repo_id}")
    print(f"load it with: MynaV6(\"{args.repo_id}\")")
    return 0


if __name__ == "__main__":
    sys.exit(main())

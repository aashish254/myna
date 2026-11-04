"""Paired, row-clustered comparisons between Myna-v6 and each reference arm.

Resamples bundles (rows) with replacement, recomputes 16-cell macro for both systems on the same
draw, and reports the paired difference with a percentile interval plus a McNemar test on the
question-level discordant pairs.

    uv run python -m bench.paired_bootstrap
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
B = 2000
SEED = 20261104


def load_rows(path: Path) -> list[dict]:
    d = json.loads(path.read_text())
    return d["rows"]


def ok_map(rows: list[dict], cell_key: str = "qname") -> dict[tuple[str, str], int]:
    out = {}
    for r in rows:
        rid = r["meta"]["id"]
        for q, v in r["questions"].items():
            out[(rid, f"{r['meta']['source']}/{q}")] = int(v.get("ok", 0))
    return out


def cell_index(m: dict[tuple[str, str], int]) -> dict[str, list[tuple[str, str]]]:
    idx: dict[str, list[tuple[str, str]]] = {}
    for key in m:
        idx.setdefault(key[1], []).append(key)
    return idx


def macro(sample_rows: list[str], ok: dict[tuple[str, str], int],
          idx: dict[str, list[tuple[str, str]]]) -> float:
    accs = []
    for cell, keys in idx.items():
        hit = tot = 0
        for rid in sample_rows:
            k = (rid, cell)
            if k in ok:
                hit += ok[k]
                tot += 1
        if tot:
            accs.append(hit / tot)
    return sum(accs) / len(accs)


def mcnemar(a: dict, b: dict) -> tuple[int, int, float]:
    keys = set(a) & set(b)
    n01 = sum(1 for k in keys if a[k] == 0 and b[k] == 1)
    n10 = sum(1 for k in keys if a[k] == 1 and b[k] == 0)
    if n01 + n10 == 0:
        return n01, n10, 1.0
    chi = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p = math.erfc(math.sqrt(chi / 2))
    return n01, n10, p


def main() -> None:
    myna_rows = load_rows(REPO / "runs/paper_myna_v6_test.rowoutcomes.json")
    myna = ok_map(myna_rows)
    idx = cell_index(myna)
    rows = sorted({k[0] for k in myna})
    rng = random.Random(SEED)

    arms = {}
    base = json.loads((REPO / "runs/kaggle/results/baseline_seed42.json").read_text())
    arms["deberta_percell_seed42"] = {(r["row_id"], r["cell"]): int(r["ok"])
                                      for r in base["records"]}
    for name, path in (("laya_0419", "runs/laya_0419_test.rowoutcomes.json"),
                       ("vegaml_0150", "runs/vegaml_0150_test.rowoutcomes.json")):
        arms[name] = ok_map(load_rows(REPO / path))

    out = {}
    for name, ok in arms.items():
        if not ok:
            continue
        draws = []
        for _ in range(B):
            sample = [rng.choice(rows) for _ in rows]
            draws.append(macro(sample, myna, idx) - macro(sample, ok, idx))
        draws.sort()
        n01, n10, p = mcnemar(myna, ok)
        out[name] = {
            "macro_myna": round(sum(myna[k] for k in myna) / len(myna), 4),
            "macro_arm": round(sum(ok[k] for k in ok) / len(ok), 4),
            "delta_macro": round(macro(rows, myna, idx) - macro(rows, ok, idx), 4),
            "ci95": [round(draws[int(0.025 * B)], 4), round(draws[int(0.975 * B)], 4)],
            "mcnemar": {"myna_only": n10, "arm_only": n01, "p": f"{p:.3g}"},
        }
    (REPO / "runs/paired_bootstrap.json").write_text(json.dumps(out, indent=1))
    for name, d in out.items():
        print(f"{name:24s} delta={d['delta_macro']:+.4f}  CI {d['ci95']}  "
              f"McNemar p={d['mcnemar']['p']}")


if __name__ == "__main__":
    main()

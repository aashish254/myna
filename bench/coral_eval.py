"""Evaluate the disabled CORAL ordinal head under the corrected gold.

    python -m bench.coral_eval --out runs/coral_eval.json

The v6 checkpoint stores an ordinal head that the shipped config disables. The campaign
gated it on development accuracy against the packed head's 0.6542 — a threshold measured
under the pre-correction score convention — so the gate cannot tell us whether the head is
better under the corrected one. This harness loads the stored head, decodes it with the
notebook's own forward pass (pointwise + cumulative-link + expected utility, mixed by
`ordinal_mix`), and scores the 240 ordinal questions against the corrected gold. The packed
head is scored on the same rows for a paired comparison.

The decoder is the notebook's, not a reconstruction: `04-v6-campaign.ipynb` cell 4 defines
the module and the fold from its four cumulative probabilities to five class logits.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]


class CoralOrdinalScoreHead(nn.Module):
    """The notebook's head, copied so the checkpoint tensors decode as trained."""

    def __init__(self, hidden_size: int = 1024, num_levels: int = 5):
        super().__init__()
        self.num_levels = num_levels
        self.pointwise = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        )
        self.level_pos = nn.Parameter(torch.randn(1, num_levels, hidden_size) * 0.02)
        self.level_attn = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=0.15,
                                                batch_first=True)
        self.level_norm = nn.LayerNorm(hidden_size)
        self.utility_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
        )
        self.b0 = nn.Parameter(torch.tensor(-1.5))
        self.deltas = nn.Parameter(torch.ones(num_levels - 2) * 0.8)
        self.ordinal_mix = nn.Parameter(torch.tensor(0.35))

    def get_thresholds(self) -> torch.Tensor:
        inc = F.softplus(self.deltas)
        b = [self.b0]
        for j in range(inc.numel()):
            b.append(b[-1] + inc[j])
        return torch.stack(b)

    def forward(self, h_levels):
        pw_logits = self.pointwise(h_levels).squeeze(-1)
        x = self.level_norm(h_levels + self.level_pos)
        attn_out, _ = self.level_attn(x, x, x)
        x = x + 0.2 * attn_out
        summary = torch.cat([x.mean(dim=1), x[:, -1, :] - x[:, 0, :]], dim=-1)
        u = self.utility_head(summary)
        thresh = self.get_thresholds().unsqueeze(0)
        cum_probs = torch.sigmoid(u - thresh)
        p0 = 1.0 - cum_probs[:, 0:1]
        p_mid = cum_probs[:, :-1] - cum_probs[:, 1:]
        p4 = cum_probs[:, -1:]
        coral_probs = torch.clamp(torch.cat([p0, p_mid, p4], dim=-1), min=1e-6)
        coral_logits = coral_probs.log()
        mix = torch.sigmoid(self.ordinal_mix)
        final_logits = (1.0 - mix) * pw_logits + mix * coral_logits
        return final_logits, cum_probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="myna-v5-prototype 2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="runs/coral_eval.json")
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(REPO / "src"))
    from myna.v6_engine import MynaV6

    torch.set_num_threads(8)
    engine = MynaV6(args.ckpt, device="cpu")
    ck = torch.load(REPO / args.ckpt / "myna-v6-unified.pt", map_location="cpu",
                    weights_only=False)
    head = CoralOrdinalScoreHead(1024, 5)
    head.load_state_dict(ck["ordinal_score_head"], strict=True)
    head.eval()
    mix = float(torch.sigmoid(head.ordinal_mix))

    rows = [json.loads(line) for line in
            (REPO / "data" / "decision-v2-pilot" / f"{args.split}.jsonl").read_text().splitlines()
            if line.strip()]

    coral_ok = defaultdict(list)
    packed_ok = defaultdict(list)
    per_row = []
    with torch.no_grad():
        for row in rows:
            for qname, spec in row["questions"].items():
                if spec["type"] != "score":
                    continue
                labels = [str(c) for c in spec["criteria"]]
                gold = int(spec["label"])              # 0-based class index, as stored
                prefix = engine._prefix(f"{row['_meta']['source']}/{qname}", spec)
                options = engine._options(spec)
                seq, positions = engine._packed_ids(prefix, row["state"], options)
                hs = engine._packed_hidden(seq, positions)   # [5, 1024]
                if hs.shape[0] != len(labels):
                    continue
                logits, _ = head(hs.unsqueeze(0))
                coral_pred = int(logits[0].argmax())
                packed_logits = engine.packed_scorer(hs).squeeze(-1)
                packed_pred = int(packed_logits.argmax())
                cell = f"{row['_meta']['source']}/{qname}"
                coral_ok[cell].append(int(coral_pred == gold))
                packed_ok[cell].append(int(packed_pred == gold))
                per_row.append({"id": row["_meta"]["id"], "cell": cell, "gold": gold,
                                "coral": coral_pred, "packed": packed_pred})
            if len(per_row) and len(per_row) % 80 == 0:
                print(f"{len(per_row)} ordinal questions", flush=True)

    def summary(d):
        return {c: {"n": len(v), "acc": round(sum(v) / len(v), 4)} for c, v in sorted(d.items())}

    coral = summary(coral_ok)
    packed = summary(packed_ok)
    out = {
        "head": "CoralOrdinalScoreHead (v6 notebook cell 4)",
        "ordinal_mix": round(mix, 4),
        "gate_threshold_recorded": 0.6542,
        "gold": "corrected (1-based level -> 0-based option index)",
        "n": len(per_row),
        "coral": coral,
        "packed": packed,
        "coral_macro": round(sum(v["acc"] for v in coral.values()) / len(coral), 4),
        "packed_macro": round(sum(v["acc"] for v in packed.values()) / len(packed), 4),
        "paired": {"coral_only": sum(1 for r in per_row if r["coral"] == r["gold"] and r["packed"] != r["gold"]),
                   "packed_only": sum(1 for r in per_row if r["packed"] == r["gold"] and r["coral"] != r["gold"]),
                   "both": sum(1 for r in per_row if r["packed"] == r["gold"] and r["coral"] == r["gold"])},
        "rows": per_row,
    }
    (REPO / args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n", "ordinal_mix", "coral_macro", "packed_macro",
                                          "paired", "coral", "packed")}, indent=1))


if __name__ == "__main__":
    main()

"""Both consistency predicates, pinned to the released row outcomes.

The paper reports two numbers for the same 172 bundles: the campaign's recovered rule and the
broader released predicate. This test keeps them from drifting apart, and keeps the campaign
rule honest about what it can see (it never inspects the yelp rating/recommendation pair).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from myna.consistency import campaign_contradictions, contradictions  # noqa: E402

SPECS = {json.loads(line)["_meta"]["id"]: json.loads(line)["questions"]
         for line in (REPO / "data" / "decision-v2-pilot" / "test.jsonl").read_text().splitlines()
         if line.strip()}

CASES = {
    "runs/paper_myna_v6_test.rowoutcomes.json": (3, 0),
    "runs/laya_0419_test.rowoutcomes.json": (10, 1),
    "runs/vegaml_0150_test.rowoutcomes.json": (29, 2),
}


def test_released_and_campaign_counts_match_the_paper():
    for rel, (released, campaign) in CASES.items():
        rows = json.loads((REPO / rel).read_text())["rows"]
        assert contradictions(rows, SPECS)["contradictions"] == released, rel
        assert campaign_contradictions(rows, SPECS)["contradictions"] == campaign, rel


def test_campaign_rule_cannot_see_yelp():
    rows = json.loads((REPO / "runs/paper_myna_v6_test.rowoutcomes.json").read_text())["rows"]
    yelp = [r for r in rows if r["meta"]["source"] == "yelp"]
    assert campaign_contradictions(yelp, SPECS)["contradictions"] == 0
    # The released predicate does see the one yelp case.
    assert contradictions(yelp, SPECS)["contradictions"] == 1

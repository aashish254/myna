"""V2-A invariants for the upstream pull: the collision counts *are* the claim.

Nothing here needs the network or kev: the blocking/dedup/admission helpers take their
converters as arguments, so the gate is testable with stubs. The one test that does need
kev (admit() must not drift from the function it mirrors) skips when it is absent.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.pull_upstream import (admit, filter_pool, partition_keys, pull_contrastive,  # noqa: E402
                                 record_keys, state_digest)


def rec(state, text_sha, row=0):
    """A labelled request in the suite's shape: `state` is what the model reads,
    `text_sha` stands in for the suite's upstream-field hash. The two keys diverge on
    purpose -- that divergence is what these tests pin down."""
    return {"state": state,
            "questions": {"q": {"type": "noul", "instructions": "Is it urgent?", "label": True}},
            "_meta": {"source": "fake", "row": row, "id": f"fake/train/{row}", "split": "train",
                      "row_sha256": f"{row:064d}", "text_sha256": text_sha}}


def keys_of(records):
    return ({state_digest(r) for r in records}, {r["_meta"]["text_sha256"] for r in records})


class FakeData:
    def materialize(self, record):
        return record


class FakeModel:
    def __init__(self, reject=()):
        self.reject = set(reject)

    def fits(self, record, *tokenizers, max_branch=960):
        return record["state"] not in self.reject


class FakeContrastive:
    def __init__(self, records):
        self.records = records

    def generate(self, n_pairs, seed, families=None):
        return self.records, {f: {"pairs": 1, "rejected": 0} for f in (families or ["f"])}


def test_state_digest_normalizes_case_and_whitespace_only():
    assert state_digest(rec("Pay   my BILL", "a")) == state_digest(rec("pay my bill", "b"))
    # a re-wrapped state is a different state -- which is why the text key is checked too
    assert state_digest(rec("text", "a")) != state_digest(rec({"document": "text"}, "a"))


def test_partition_keys_reads_both_keys_from_a_suite_dir(tmp_path):
    (tmp_path / "development.jsonl").write_text(
        json.dumps(rec("alpha", "h1")) + "\n" + json.dumps(rec("beta", "h2")) + "\n", encoding="utf-8")
    states, texts = partition_keys(tmp_path, ("development", "test"))  # a missing file adds nothing
    assert texts == {"h1", "h2"}
    assert states == keys_of([rec("alpha", "h1"), rec("beta", "h2")])[0]


def test_filter_pool_blocks_on_either_key():
    eval_keys = keys_of([rec("eval one", "e1"), rec("eval two", "e2")])
    frozen_keys = keys_of([rec("frozen", "f1")])
    pool = [rec("eval one", "other"),      # same state as an eval row
            rec("re-wrapped eval", "e2"),  # a different rendered state, the same upstream text
            rec("frozen thing", "f1"),     # already in the frozen train
            rec("brand new", "n1")]        # the only keepable row
    col, admitted = Counter(), set()
    kept = filter_pool(pool, eval_keys, frozen_keys, admitted, col)
    assert [r["state"] for r in kept] == ["brand new"]
    assert col["eval_collision"] == 2 and col["in_frozen_train"] == 1
    assert record_keys(kept[0])[0] in admitted


def test_filter_pool_claims_a_state_once_even_across_sources():
    admitted, col = set(), Counter()
    empty = (set(), set())
    kept = filter_pool([rec("same text", "x"), rec("same text", "y")], empty, empty, admitted, col)
    assert len(kept) == 1 and col["duplicate_state"] == 1


def test_admit_orders_by_row_sha_and_stops_at_the_count():
    pool = [rec("c", "3", row=3), rec("a", "1", row=1), rec("b", "2", row=2)]
    rep = Counter()
    chosen = admit(pool, 2, set(), [], rep, 960, FakeData(), FakeModel())
    assert [r["state"] for r in chosen] == ["a", "b"]
    assert rep["accepted"] == 2 and "shortfall" not in rep


def test_admit_reports_a_shortfall_instead_of_raising():
    rep = Counter()
    chosen = admit([rec("a", "1", row=1)], 5, set(), [], rep, 960, FakeData(), FakeModel())
    assert len(chosen) == 1 and rep["shortfall"] == 4


def test_admit_separates_context_rejection_from_collision():
    rep = Counter()
    pool = [rec("ok", "1", row=1), rec("too long", "2", row=2)]
    chosen = admit(pool, 10, set(), [], rep, 960, FakeData(), FakeModel(reject={"too long"}))
    assert [r["state"] for r in chosen] == ["ok"]
    assert rep["context_rejected"] == 1 and rep["accepted"] == 1


def test_contrastive_pairs_are_kept_or_dropped_whole():
    def crec(state, sha, row, family):
        r = rec(state, sha, row)
        r["_meta"].update(family_id=f"{family}/p{row}", family=family, pair_id=f"p{row}")
        return r

    a1, a2 = crec("pair A side a", "ta1", 1, "f"), crec("pair A side b", "ta2", 2, "f")
    b1, b2 = crec("pair B side a", "tb1", 3, "f"), crec("pair B side b", "tb2", 4, "f")
    rows, summary, collisions = pull_contrastive(
        2, 7, ["f"], set(), ({state_digest(b2)}, set()), (set(), set()), [], 960,
        FakeData(), FakeModel(), FakeContrastive([a1, a2, b1, b2]))
    assert [r["state"] for r in rows] == ["pair A side a", "pair A side b"]
    assert all(r["_meta"]["group_id"] == r["_meta"]["family_id"] for r in rows)
    assert collisions["eval_collision"] == 1
    assert summary["rows_admitted"] == 2 and summary["pairs_generated"] == 1  # one family in the report


@pytest.mark.skipif(not os.environ.get("KEV_ROOT"),
                    reason="set KEV_ROOT and run under an env with `datasets` to check admit() "
                           "against the kev function it mirrors")
def test_admit_agrees_with_kev_select_unique():
    import sys as _sys

    _sys.path.insert(0, os.environ["KEV_ROOT"])
    from kev.data import materialize
    from kev.model import fits
    from kev.suite import select_unique

    class KevData:
        materialize = staticmethod(materialize)

    class KevModel:
        fits = staticmethod(fits)

    pool = [rec("the premise states it plainly", "k1", row=1),
            rec("the premise contradicts it", "k2", row=2),
            rec("the premise states it plainly", "k3", row=3)]
    mine = admit([dict(r, _meta=dict(r["_meta"])) for r in pool], 2, set(), [], Counter(),
                 1024, KevData(), KevModel())
    theirs = select_unique([dict(r, _meta=dict(r["_meta"])) for r in pool], 2, set(), [], Counter())
    assert [r["_meta"]["id"] for r in mine] == [r["_meta"]["id"] for r in theirs]

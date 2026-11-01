"""Row-clustered paired bootstrap, gated: the interval must be around the published number.

`bench/significance.py` turns the four headline numbers of this project — myna's 0.4893 macro,
the 0.4331 majority floor, G1's +0.0563 margin and the −0.1775 gap to laya — from point
estimates into intervals, and then adjudicates the sentences the repo prints them in. That is a
lot of surface for a silent wrong answer, so every claim the file makes is checked here against
arithmetic the test computes itself:

* **The resample unit is the row.** `pack()` collapses each row to one vector of per-cell sums
  and `boot()` multiplies an integer multinomial weight matrix against it, so a replicate can
  only ever contain whole rows. Checked by replaying `boot()`'s own `default_rng(seed)` call
  sequence in the test and re-deriving both the first block's digest and the first replicate's
  macro from the drawn weights. A question-level resample gives a different digest, and a
  different number, and the replay says so.
* **The point estimate is the published statistic.** `main()` refuses (exit 3) unless macro,
  both floors and G1's observed margin agree with the committed report to 1e-9. Each of those
  four is tested by moving exactly one input: a doctored floor, an invented `g1.margin_observed`,
  `--min-rows` raised so a cell leaves the macro, a row absent from the competitor dump so the
  paired set shrinks, and a `--allow-subset` that says it out loud and continues. The tolerance
  itself is measured from both sides — 1e-7 refused, 1e-10 accepted — because a boundary that is
  only probed from outside is a value the code chose and no test read.
* **Paired means both sides answered the same `(row, question)`.** `pack()` intersects on
  `_meta.id`, counts what it dropped, and McNemar reads only discordance — so the pooled counts
  are checked against the per-cell table and against a hand-counted truth table, and
  myna-versus-floor is rebuilt from the dump's own gold-index histogram.
* **The floors are recomputed inside every replicate.** If the floor were frozen,
  `margin_over_floor` would be `myna_macro` minus a constant in every replicate, so the two
  intervals would differ by the same amount at both ends. They do not, and that is one assertion.

The competitor side here is **synthetic and labelled so**: a deterministic rule over the myna
dump's own rows — every fourth row of a group is wrong, and wrong for all of its questions at
once. The claims under test are the arithmetic of pairing, resampling and adjudication, not
laya's ability, and the row-level rule is what puts the within-row correlation the cluster
exists to respect into the fixture. The committed witness (`runs/significance.json`) is the real
laya dump read by the same `main()`.

`fx` and the fixture constants come from `tests/test_row_outcomes.py`, which builds the suite,
the checkpoint and the report-guarded dump this file reads as its input.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import bench.significance as SIGN  # noqa: E402
from test_row_outcomes import (CELLS_KEPT, G1_MARGIN, G1_TARGET, MIN_ROWS,  # noqa: E402
                               N_QUESTIONS, N_ROWS, TEMPERATURE, fx)

B = 4000
SEED = 11
MARK_EVERY = 4  # every fourth row of a group is the competitor's weak one


# ----------------------------------------------------------------------- the synthetic laya
#
# `pack()` reads a laya row as {meta.id, questions: {name: {ok, ...}}} and `main()` reads the
# engine block, so those are the fields this builds. Everything else is copied so a reader can
# check the two dumps against each other: the same ids, the same golds, the same `k`, the same
# cell sizes.
#
# The rule is row-level on purpose. `idx` is the row's position inside its question-set, so
# `idx % 4 == 0` marks about a quarter of each cell's rows, and a marked row is the only place
# the two engines may disagree *against* the competitor — the other three copy myna's `ok` and
# are therefore never discordant in myna's favour. That puts the within-row correlation the
# cluster exists to respect into the fixture (a question-level rule would leave the cluster key
# doing nothing) and it fixes the sign: the competitor wins overall, which is the direction the
# committed witness has, while the random-init myna macro sits below its own floor, so the
# adjudication prints both verdicts.


def laya_rows(myna, skip_ids=()):
    out = []
    for r in myna["rows"]:
        if r["meta"]["id"] in skip_ids:
            continue
        strong = r["idx"] % MARK_EVERY != 0
        rec = {name: {"type": q["type"], "gold": q["gold"], "k": q["k"],
                      "ok": 1 if strong else q["ok"], "conf": 0.5}
               for name, q in r["questions"].items()}
        out.append({"meta": r["meta"], "group": r["group"], "idx": r["idx"],
                    "questions": rec})
    return out


def laya_dump(myna, skip_ids=()):
    rows = laya_rows(myna, skip_ids)
    return {"engine": "laya", "version_marker": "synthetic-fixture-competitor",
            "split": myna["split"], "suite": myna["suite"], "file": myna["file"],
            "min_rows": myna["min_rows"], "n_rows": len(rows),
            "n_questions": sum(len(r["questions"]) for r in rows),
            "laya": {"version": "0.0.0-fixture", "commit": "0" * 40,
                     "path": str(REPO / "fixture")},
            "laya_warnings": ["RuntimeWarning: synthetic competitor"],
            "rows": rows}


def write_laya(path, myna, skip_ids=()):
    path.write_text(json.dumps(laya_dump(myna, skip_ids), indent=1) + "\n")
    return path


def packed(myna, report, laya=None, min_rows=MIN_ROWS):
    return SIGN.pack(myna, laya if laya is not None else laya_dump(myna),
                     report["cells"], min_rows)


@pytest.fixture(scope="module")
def sx(fx, tmp_path_factory):
    """The paired run: `main()` in-process over the guarded myna dump and the synthetic laya."""
    root = tmp_path_factory.mktemp("paired")
    laya_path = write_laya(root / "laya.rowoutcomes.json", fx.dump)
    out, md = root / "significance.json", root / "significance.md"
    rc = SIGN.main(["--myna", str(fx.dump_path), "--laya", str(laya_path),
                    "--report", str(fx.report_path), "--B", str(B), "--seed", str(SEED),
                    "--out", str(out), "--md", str(md)])
    assert rc == 0, "the paired run did not reproduce the fixture report"
    return SimpleNamespace(root=root, laya_path=laya_path, out=out, md=md,
                           res=json.loads(out.read_text()), md_text=md.read_text(),
                           myna=fx.dump, report=fx.report, fx=fx)


# ----------------------------------------------------------------------------- McNemar alone


def test_mcnemar_matches_an_exact_binomial_computed_with_fractions():
    """The harness builds the tail from its largest term downward in float; the test builds it
    term by term in exact rational arithmetic. Five cases, including the two where a missing
    doubling or a missing `min(1.0, ·)` shows up."""
    for b, c in ((5, 0), (50, 30), (3, 2), (10, 10), (117, 41)):
        n = b + c
        k = min(b, c)
        tail = sum(Fraction(math.comb(n, i), 2 ** n) for i in range(k + 1))
        want = float(min(Fraction(1), 2 * tail))
        assert SIGN.mcnemar_exact(b, c) == pytest.approx(want, rel=1e-9), (b, c)


def test_mcnemar_names_the_absence_of_evidence_instead_of_answering_zero():
    assert SIGN.mcnemar_exact(0, 0) is None
    assert SIGN.mcnemar_exact(1, 0) == pytest.approx(1.0)
    assert SIGN.mcnemar_exact(0, 7) == pytest.approx(SIGN.mcnemar_exact(7, 0))
    # P(X=0) at n=1400 is 2^-1400, past float64's smallest normal: the log-space lead term is
    # what keeps this a number instead of 0.0
    assert SIGN.mcnemar_exact(1300, 100) > 0.0
    assert SIGN.mcnemar_exact(1300, 100) < SIGN.mcnemar_exact(700, 680) <= 1.0


# ------------------------------------------------------------------------------- the cell set


def test_the_cell_set_is_filtered_by_min_rows_before_anything_is_resampled(sx):
    """A cell below the floor must not reach the arrays at all: a replicate that carries it would
    average a 4-row cell with the same weight as a 116-row one, which is not the published
    macro."""
    small = dict(sx.report["cells"][0], source="trec", question="answer_type", n=2,
                 acc=0.5, majority=0.5, uniform=0.33, type="choice", sets=1, hits=1)
    cells = list(sx.report["cells"]) + [small]
    kept = packed(sx.myna, {"cells": cells})["keys"]
    assert ("trec", "answer_type") not in kept and len(kept) == CELLS_KEPT
    everything = packed(sx.myna, {"cells": cells}, min_rows=1)["keys"]
    assert ("trec", "answer_type") in everything and len(everything) == CELLS_KEPT + 1
    assert sx.res["n_cells"] == CELLS_KEPT and sx.res["min_rows"] == MIN_ROWS
    assert set(kept) == {(c["source"], c["question"]) for c in sx.report["cells"]}


# --------------------------------------------------------------------------- packing the rows


def test_every_packed_sum_is_the_dump_counted_a_different_way(sx):
    """`nm`/`hm`/`um`/`lab` are the four things a replicate re-weights. Each is rebuilt here from
    the JSON rows, because a wrong one flows into every statistic and every interval."""
    p = packed(sx.myna, sx.report)
    assert p["ids"] == sorted(r["meta"]["id"] for r in sx.myna["rows"])
    N, C = len(p["ids"]), len(p["keys"])
    assert p["nm"].shape == p["hm"].shape == p["um"].shape == (N, C)
    assert p["lab"].shape == (N, C, p["n_labels"])
    in_cells = sum(1 for r in sx.myna["rows"] for n in r["questions"]
                   if (r["meta"]["source"], n) in p["cell_ix"])
    assert p["nm"].sum() == in_cells
    assert p["hm"].sum() == sum(c["hits"] for c in sx.myna["cells"])
    for i, rid in enumerate(p["ids"]):
        row = next(r for r in sx.myna["rows"] if r["meta"]["id"] == rid)
        for name, q in row["questions"].items():
            j = p["cell_ix"].get((row["meta"]["source"], name))
            if j is None:
                continue
            assert p["nm"][i, j] == 1
            assert p["hm"][i, j] == q["ok"]
            assert p["um"][i, j] == pytest.approx(1.0 / q["k"])
            assert p["lab"][i, j].sum() == 1, "one gold label per row per cell"
    three = [r for r in sx.myna["rows"] if len(r["questions"]) == 3]
    assert three, "the fixture needs a row that asks three questions for the cluster to mean " \
                  "anything"
    assert p["nm"][p["ids"].index(three[0]["meta"]["id"])].sum() == 3


def test_a_row_the_competitor_cannot_answer_leaves_both_sides(sx):
    """Pairing is an intersection, and a one-sided removal would price myna on rows laya never
    saw. The row's id and the questions that lost their partner are both fields."""
    full = packed(sx.myna, sx.report)
    victim_row = sx.myna["rows"][3]
    victim = victim_row["meta"]["id"]
    lost = sum(1 for n in victim_row["questions"]
               if (victim_row["meta"]["source"], n) in full["cell_ix"])
    assert lost == len(victim_row["questions"]), "the victim must ask only scored questions"
    p = SIGN.pack(sx.myna, laya_dump(sx.myna, skip_ids=(victim,)), sx.report["cells"], MIN_ROWS)
    assert p["dropped_rows"] == [victim]
    assert victim not in p["ids"] and len(p["ids"]) == N_ROWS - 1
    assert len(p["pairs"]) == len(full["pairs"]) - lost
    assert p["nm"].sum() == full["nm"].sum() - lost
    assert p["nl"].sum() == p["nm"].sum()
    # myna's own sums do not know the row left, because `nm` is the myna side of the pair only
    assert (p["ids"] == [i for i in full["ids"] if i != victim])


def test_a_question_the_competitor_did_not_answer_is_counted_not_silently_paired(sx):
    """laya's own dump can carry a row with no answer for one question. That question still
    prices myna's cell — dropping it would move myna's accuracy too — and leaves the pairs."""
    laya = laya_dump(sx.myna)
    target = laya["rows"][0]
    del target["questions"][sorted(target["questions"])[0]]
    p = SIGN.pack(sx.myna, laya, sx.report["cells"], MIN_ROWS)
    assert p["dropped_questions"] == 1 and p["dropped_rows"] == []
    assert p["nl"].sum() == p["nm"].sum() - 1
    assert len(p["pairs"]) == int(p["nl"].sum())
    assert p["pairs"] and all(j >= 0 for j, _m, _l, _g in p["pairs"])


# --------------------------------------------------------------------------- the point estimate


def test_the_point_estimate_is_the_committed_report_to_nine_places(sx):
    for k, want in (("myna_macro", sx.report["macro"]["acc"]),
                    ("majority_floor", sx.report["macro"]["majority"]),
                    ("uniform_floor", sx.report["macro"]["uniform"]),
                    ("margin_over_floor", sx.report["g1"]["margin_observed"])):
        assert sx.res["point"][k] == pytest.approx(want, abs=1e-12), (k, want)
    assert sx.res["report_macro"] == sx.report["macro"]["acc"]
    assert sx.res["point"]["gap_myna_minus_laya"] < 0, "the synthetic competitor is meant to win"


def test_the_floors_are_functions_of_the_labels_and_move_with_them():
    """A frozen floor and a recomputed floor are the same number on the full sample and differ on
    every resample, so the distinction can only be tested off the point estimate: two label
    matrices with identical counts and different distributions. The arrays are (1, C) totals —
    one sample, one cell — which is the shape `main()` hands `statize` for the point estimate."""
    nm = np.array([[4.0]])
    hm = np.array([[2.0]])
    skewed = np.array([[[4.0, 0.0]]])             # every gold is label 0 → floor 1.0
    balanced = np.array([[[2.0, 2.0]]])           # two and two → floor 0.5
    s1 = SIGN.statize(nm, hm, nm, hm, np.array([[2.0]]), skewed)
    s2 = SIGN.statize(nm, hm, nm, hm, np.array([[2.0]]), balanced)
    assert s1["majority_floor"] == pytest.approx(1.0)
    assert s2["majority_floor"] == pytest.approx(0.5)
    assert s1["margin_over_floor"] == pytest.approx(0.5 - 1.0)
    assert s2["margin_over_floor"] == pytest.approx(0.5 - 0.5)
    # the uniform floor is 1/K row-weighted, the way myna.report.cell_stats weights it: 4 rows
    # of a 2-option cell carry Σ1/K = 2.0 over 4 rows
    assert s1["uniform_floor"] == pytest.approx(0.5)


def test_a_cell_a_replicate_empties_is_dropped_from_that_replicate_and_counted():
    """`statize` masks on `nm > 0` rather than dividing, so an empty cell is `nan` and leaves the
    mean instead of becoming a 0.0 accuracy — a floor-level loss priced into a macro that never
    scored that cell."""
    nm = np.array([[1.0, 1.0], [0.0, 1.0]])
    hm = np.array([[1.0, 0.0], [0.0, 1.0]])
    s = SIGN.statize(nm, hm, nm, hm, np.ones((2, 2)), np.full((2, 2, 2), 0.5))
    assert s["myna_macro"][0] == pytest.approx(0.5)
    assert s["myna_macro"][1] == pytest.approx(1.0)
    assert s["_cells_kept"][0] == 2 and s["_cells_kept"][1] == 1
    assert s["gap_myna_minus_laya"][1] == pytest.approx(0.0)


# ------------------------------------------------------------------------------- the resample


def test_the_interval_is_the_two_and_a_half_and_ninety_seven_and_a_half_percentiles():
    """The prose everywhere else in this repository calls these "95% CI", and `ci()` is the only
    place the nominal level is chosen. Pinning the two endpoints against a hand-built array is
    what makes a 90% interval labelled 95% a caught mutation instead of a sentence in a paper."""
    x = np.arange(1000, dtype=np.float64)
    got = SIGN.ci(x)
    assert got == [24.975, 974.025], got
    assert all(isinstance(v, float) for v in got), "the payload must stay JSON-serialisable"
    assert SIGN.ci(np.zeros(50)) == [0.0, 0.0]
    big = np.arange(10000, dtype=np.float64)
    lo, hi = SIGN.ci(big)
    assert hi - lo == pytest.approx(9499.5, abs=1.0), (lo, hi)
    assert 0.949 <= float(np.mean((big >= lo) & (big <= hi))) <= 0.951, "not a 95% interval"


def test_the_replicates_are_drawn_from_the_seed_this_file_records(sx):
    """Replay `boot()`'s call sequence outside `boot()`: the same `default_rng(seed)`, the same
    first block of multinomial draws, and so the same digest and the same first replicate. That
    is what makes `first_replicate_digest` a witness to the resample rather than a fingerprint of
    whatever the process happened to do."""
    p = packed(sx.myna, sx.report)
    N, C = len(p["ids"]), len(p["keys"])
    rng = np.random.default_rng(SEED)
    w = rng.multinomial(N, [1.0 / N] * N, size=min(1000, B)).astype(np.float32)
    assert hashlib.sha256(w.tobytes()).hexdigest()[:16] \
        == sx.res["bootstrap"]["first_replicate_digest"]
    scal, acc_m, acc_l, kept, digest = SIGN.boot(p, B, SEED)
    assert digest == sx.res["bootstrap"]["first_replicate_digest"]
    s = SIGN.statize(w[:1] @ p["nm"].astype(np.float32),
                     w[:1] @ p["hm"].astype(np.float32),
                     w[:1] @ p["nl"].astype(np.float32),
                     w[:1] @ p["hl"].astype(np.float32),
                     w[:1] @ p["um"].astype(np.float32),
                     (w[:1] @ p["lab"].reshape(N, C * p["n_labels"]).astype(np.float32))
                     .reshape(1, C, p["n_labels"]))
    assert float(s["myna_macro"][0]) == pytest.approx(float(scal["myna_macro"][0]), rel=1e-6)
    assert float(s["majority_floor"][0]) == pytest.approx(float(scal["majority_floor"][0]),
                                                          rel=1e-6)
    # the drawn weights are integers summing to N: whole rows, with replacement
    assert w[0].sum() == N and np.all(w[0] == np.round(w[0]))
    assert kept.max() == C and sx.res["bootstrap"]["cells_kept_max"] == C


def test_a_different_seed_draws_a_different_sample_and_the_same_kind_of_interval(sx):
    p = packed(sx.myna, sx.report)
    s1, _a1, _b1, _k1, d1 = SIGN.boot(p, 200, SEED)
    s2, _a2, _b2, _k2, d2 = SIGN.boot(p, 200, SEED + 1)
    s3, _a3, _b3, _k3, d3 = SIGN.boot(p, 200, SEED)
    assert d1 == d3 and d1 != d2
    for k in SIGN.STATS:
        assert np.array_equal(s1[k], s3[k]), f"{k} is not seed-deterministic"
        assert not np.array_equal(s1[k], s2[k]), f"{k} ignores the seed"
    assert abs(float(np.mean(s1["myna_macro"])) - sx.res["point"]["myna_macro"]) < 0.05


def test_the_interval_of_the_margin_is_not_the_interval_of_the_accuracy_shifted(sx):
    m = sx.res["ci95"]["myna_macro"]
    g = sx.res["ci95"]["margin_over_floor"]
    f = sx.res["ci95"]["majority_floor"]
    assert f[0] < f[1], "a frozen floor would print [x, x]"
    assert abs((m[0] - g[0]) - (m[1] - g[1])) > 1e-6, (m, g)


def test_the_gap_is_a_per_replicate_difference_not_two_intervals_subtracted(sx):
    p = packed(sx.myna, sx.report)
    scal, acc_m, acc_l, _kept, _d = SIGN.boot(p, 200, SEED)
    diff = np.nanmean(acc_m - acc_l, axis=-1)
    assert np.allclose(scal["gap_myna_minus_laya"], diff, rtol=0, atol=1e-6)
    myna, laya = sx.res["ci95"]["myna_macro"], sx.res["ci95"]["laya_macro"]
    naive = [myna[0] - laya[1], myna[1] - laya[0]]
    got = sx.res["ci95"]["gap_myna_minus_laya"]
    assert got != naive and got[1] - got[0] < naive[1] - naive[0], (got, naive)


# ------------------------------------------------------------------------------- adjudication


def _cis(floor=(0.06, 0.18), macro=(0.56, 0.90), gap=(-0.40, -0.10)):
    return {"margin_over_floor": list(floor), "myna_macro": list(macro),
            "gap_myna_minus_laya": list(gap), "majority_floor": [0.0, 1.0],
            "uniform_floor": [0.0, 1.0], "laya_macro": [0.0, 1.0]}


POINT = {"margin_over_floor": 0.10, "myna_macro": 0.60, "gap_myna_minus_laya": -0.20,
         "majority_floor": 0.50, "uniform_floor": 0.33, "laya_macro": 0.80}
REPORT = {"split": "test", "g1": {"target": G1_TARGET, "margin": G1_MARGIN}}


def _by_id(cis, point=POINT, report=REPORT):
    """`claims()` gives each adjudicated sentence an id, so a test addresses the rule instead of
    matching its prose — and the four ids are the four sentences the paper carries."""
    out = {c["id"]: c for c in SIGN.claims(point, cis, report)}
    assert set(out) == {"floor", "g1-margin", "g1-target", "competitor"}
    return out


def test_an_interval_that_covers_the_threshold_rewrites_the_sentence():
    stands = _by_id(_cis())
    assert all(c["survives"] for c in stands.values()), \
        [c["claim"] for c in stands.values() if not c["survives"]]
    assert stands["floor"]["consequence"] == "the sentence as written stands"
    assert stands["floor"]["statistic"] == "margin_over_floor"
    assert stands["floor"]["point"] == POINT["margin_over_floor"]
    assert stands["floor"]["threshold"] == "CI excludes 0"
    assert stands["g1-margin"]["threshold"] == f"CI lower bound ≥ {G1_MARGIN}"
    assert stands["g1-target"]["threshold"] == f"CI lower bound ≥ {G1_TARGET}"
    assert stands["competitor"]["threshold"] == "CI excludes 0 on the negative side"
    moved = _by_id(_cis(floor=(-0.01, 0.14), macro=(0.50, 0.62)))
    assert not moved["floor"]["survives"] and not moved["g1-margin"]["survives"]
    assert not moved["g1-target"]["survives"] and moved["competitor"]["survives"]
    assert moved["floor"]["consequence"] == "the sentence must be rewritten in the same PR"


def test_the_competitor_claim_needs_the_whole_interval_below_zero():
    assert _by_id(_cis(gap=(-0.05, -0.001)))["competitor"]["survives"] is True
    assert _by_id(_cis(gap=(-0.05, 0.004)))["competitor"]["survives"] is False


def test_the_g1_rules_are_at_least_not_greater_than():
    """Both halves of G1 are `≥` on the interval's *lower bound*, so a CI that touches the
    threshold exactly still satisfies the rule. Measured at the boundary, because a `<` here
    would fail a gate the point estimate passes."""
    report = {"split": "test", "g1": {"target": 0.60, "margin": 0.10}}
    exact = _by_id(_cis(floor=(0.10, 0.20), macro=(0.60, 0.90)), report=report)
    assert exact["floor"]["survives"] and exact["g1-margin"]["survives"]
    assert exact["g1-target"]["survives"]
    by_a_hair = _by_id(_cis(floor=(0.0999, 0.20), macro=(0.5999, 0.90)), report=report)
    assert not by_a_hair["g1-margin"]["survives"]
    assert not by_a_hair["g1-target"]["survives"]
    assert by_a_hair["floor"]["survives"], "0.0999 > 0 still clears the floor rule"


def test_the_fixture_adjudicates_its_own_intervals(sx):
    by_id = {c["id"]: c for c in sx.res["claims"]}
    assert set(by_id) == {"floor", "g1-margin", "g1-target", "competitor"}
    for c in sx.res["claims"]:
        lo, hi = c["ci95"]
        assert c["point"] == pytest.approx(sx.res["point"][c["statistic"]])
        assert c["statistic"] in SIGN.STATS
        if c["id"] in ("floor", "g1-margin"):
            want = (lo > 0) if c["id"] == "floor" else (lo >= sx.report["g1"]["margin"])
            assert c["survives"] == want, c
        if c["id"] == "g1-target":
            assert c["survives"] == (lo >= sx.report["g1"]["target"]), c
        if c["id"] == "competitor":
            assert c["survives"] == (hi < 0), c
    # the random-init fixture loses to its own floor, so three of the four sentences must be
    # marked for rewriting here — and the competitor, which the rule says is above myna, stands
    assert not by_id["floor"]["survives"] and not by_id["g1-margin"]["survives"]
    assert not by_id["g1-target"]["survives"] and by_id["competitor"]["survives"]


# ------------------------------------------------------------- McNemar against the dump itself


def test_the_pooled_discordance_is_the_per_cell_table_summed(sx):
    p = packed(sx.myna, sx.report)
    b = c = 0
    for _j, ok_m, ok_l, _gold in p["pairs"]:
        b += int(ok_m and not ok_l)
        c += int(ok_l and not ok_m)
    pooled = sx.res["mcnemar"]["vs_laya"]
    assert (pooled["b"], pooled["c"], pooled["n"]) == (b, c, b + c)
    assert sum(x["mcnemar"]["b"] for x in sx.res["per_cell"]) == b
    assert sum(x["mcnemar"]["c"] for x in sx.res["per_cell"]) == c
    assert sum(x["n"] for x in sx.res["per_cell"]) == sx.res["paired_question_instances"]
    assert len(p["pairs"]) == sx.res["paired_question_instances"]


def test_myna_against_the_floor_is_scored_on_the_majority_label_of_its_own_cell(sx):
    """The floor claim without a resample: `ok` against `gold == the cell's majority`, pooled.
    Rebuilt from the JSON here, including the tie-break, because the dump stores golds as option
    *indices* and an off-by-one in that mapping would silently compare a label to a number."""
    hist = {}
    for r in sx.myna["rows"]:
        for name, q in r["questions"].items():
            key = (r["meta"]["source"], name)
            if any(key == (c["source"], c["question"]) for c in sx.report["cells"]):
                hist.setdefault(key, Counter())[q["gold"]] += 1
    maj = {k: min(v, key=lambda lab: (-v[lab], lab)) for k, v in hist.items()}
    b = c = 0
    for r in sx.myna["rows"]:
        for name, q in r["questions"].items():
            key = (r["meta"]["source"], name)
            if key not in maj:
                continue
            floor_ok = int(q["gold"] == maj[key])
            b += int(q["ok"] and not floor_ok)
            c += int(floor_ok and not q["ok"])
    got = sx.res["mcnemar"]["vs_floor"]
    assert (got["b"], got["c"]) == (b, c), (got, b, c)
    assert got["n"] == b + c > 0


def test_the_per_cell_accuracies_are_the_paired_counts_divided_by_the_paired_n(sx):
    for c in sx.res["per_cell"]:
        assert c["n"] > 0 and c["mcnemar"]["n"] <= c["n"]
        assert c["myna"] == pytest.approx(round(c["myna"] * c["n"]) / c["n"], abs=1e-12)
        assert c["gap"] == pytest.approx(c["myna"] - c["laya"], abs=1e-12)
        lo, hi = c["gap_ci95"]
        assert lo <= hi
        assert c["mcnemar"]["p"] is None or 0.0 < c["mcnemar"]["p"] <= 1.0
    assert len(sx.res["per_cell"]) == CELLS_KEPT


# ------------------------------------------------------------------------------ the printed face


def _md_cells(md):
    out = []
    for line in md.splitlines():
        if line.startswith("| ") and line.count("|") == 10 and "n pairs" not in line:
            out.append([x.strip() for x in line.strip("|").split("|")])
    return out


def test_the_per_cell_table_prints_the_numbers_the_json_carries(sx):
    """§9.34: read by column offset, not by substring, so a `Δ` that prints a level instead of a
    difference cannot pass by containing the right digits somewhere."""
    rows = _md_cells(sx.md_text)
    assert len(rows) == CELLS_KEPT
    for cols, j in zip(rows, sx.res["per_cell"]):
        assert cols[0] == j["cell"] and cols[1] == str(j["n"])
        assert float(cols[2]) == pytest.approx(j["myna"], abs=5e-5)
        assert float(cols[3]) == pytest.approx(j["laya"], abs=5e-5)
        assert float(cols[4]) == pytest.approx(j["gap"], abs=5e-5)
        assert float(cols[4]) == pytest.approx(j["myna"] - j["laya"], abs=5e-5)
        assert cols[5] == str(j["mcnemar"]["b"]) and cols[6] == str(j["mcnemar"]["c"])
        assert cols[7] == ("n/a (no discordant pairs)" if j["mcnemar"]["p"] is None
                           else f"{j['mcnemar']['p']:.3g}"), (j["cell"], cols[7])
        assert [float(x) for x in cols[8].strip("[] ").split(", ")] == \
            pytest.approx(j["gap_ci95"], abs=5e-5)


def _md_headline(md):
    out = {}
    for line in md.splitlines():
        if line.startswith("| `") and line.count("|") == 5:
            cols = [x.strip() for x in line.strip("|").split("|")]
            out[cols[0].strip("`")] = (float(cols[1]), cols[2])
    return out


def test_the_headline_table_prints_all_six_statistics_and_no_others(sx):
    got = _md_headline(sx.md_text)
    assert set(got) == set(SIGN.STATS), set(got) ^ set(SIGN.STATS)
    for k, (point, ci_col) in got.items():
        assert point == pytest.approx(sx.res["point"][k], abs=5e-5), k
        lo, hi = sx.res["ci95"][k]
        assert ci_col == f"[{lo:.4f}, {hi:.4f}]", (k, ci_col)
    assert repr(sx.res["point"]["myna_macro"]) in sx.md_text
    assert sx.res["bootstrap"]["first_replicate_digest"] in sx.md_text
    assert str(sx.res["bootstrap"]["seed"]) in sx.md_text


def test_the_adjudication_lines_are_printed_from_the_same_verdicts_the_json_writes(sx):
    assert len(sx.res["claims"]) == 4
    for c in sx.res["claims"]:
        word = "stands" if c["survives"] else "does not stand"
        assert f"**{word}** — {c['claim']}" in sx.md_text, c
        lo, hi = c["ci95"]
        assert f"[{lo:.4f}, {hi:.4f}], rule: {c['threshold']}" in sx.md_text, c


def test_the_run_announces_the_point_the_replicates_and_the_verdicts(sx, tmp_path, capsys):
    """The stdout face is what a reader of `runs/significance.log` sees, and every number on it
    is a rounded view of a JSON field: `:.4f` becoming `:.2f`, the gap losing its `+`, the row
    count becoming the question count, or a verdict word coming from a different claim than the
    line it prints on all pass the tests above. Reconstructing each whole line from the witness
    is what ties the two together."""
    out, md = tmp_path / "ann.json", tmp_path / "ann.md"
    argv = ["--myna", str(sx.fx.dump_path), "--laya", str(sx.laya_path),
            "--report", str(sx.fx.report_path), "--B", "200", "--seed", str(SEED),
            "--out", str(out), "--md", str(md)]
    assert SIGN.main(argv) == 0
    lines = capsys.readouterr().out.splitlines()
    res = json.loads(out.read_text())
    assert lines[0] == "$ uv run python bench/significance.py " + " ".join(argv)
    p = res["point"]
    assert lines[1] == (f"point: myna {p['myna_macro']:.4f} · floor "
                        f"{p['majority_floor']:.4f} · laya {p['laya_macro']:.4f} · gap "
                        f"{p['gap_myna_minus_laya']:+.4f} · macro, both floors and G1's margin "
                        f"match {sx.fx.report_path.name} to 1e-9")
    assert re.match(rf"^200 replicates over {N_ROWS} rows × {CELLS_KEPT} cells in "
                    rf"\d+\.\d+s · cells kept per replicate min "
                    rf"{res['bootstrap']['cells_kept_min']} max "
                    rf"{res['bootstrap']['cells_kept_max']}$", lines[2]), lines[2]
    assert lines[3] == f"wrote {out} and {md}"
    assert lines[4:] == [
        f"  {'STANDS   ' if c['survives'] else 'REWRITTEN'} {c['claim']} · {c['statistic']} "
        f"{c['point']:.4f} [{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}] vs {c['threshold']}"
        for c in res["claims"]]
    # three of the fixture's four sentences are for rewriting, and they are the first three
    assert [l[2:11].strip() for l in lines[4:]] == ["REWRITTEN", "REWRITTEN", "REWRITTEN",
                                                    "STANDS"]


def test_a_row_that_leaves_the_paired_set_is_said_out_loud(sx, tmp_path, capsys):
    """`dropped_rows` is in the witness, but a row that silently vanishes from the pairing is
    the defect this file exists to avoid — the published laya column came from a subsample and
    nothing on its face said so. So the announcement is tested on its own, including the case
    where it must not print."""
    assert SIGN.main(["--myna", str(sx.fx.dump_path), "--laya", str(sx.laya_path),
                      "--report", str(sx.fx.report_path), "--B", "50", "--seed", str(SEED),
                      "--out", str(tmp_path / "a.json"), "--md", str(tmp_path / "a.md")]) == 0
    assert "leave the paired set" not in capsys.readouterr().out

    skip = tuple(r["meta"]["id"] for r in sx.myna["rows"] if r["meta"]["source"] == "trec")
    assert len(skip) == 2
    laya = write_laya(tmp_path / "short3.json", sx.myna, skip_ids=skip)
    assert SIGN.main(["--myna", str(sx.fx.dump_path), "--laya", str(laya),
                      "--report", str(sx.fx.report_path), "--B", "50", "--seed", str(SEED),
                      "--out", str(tmp_path / "b.json"), "--md", str(tmp_path / "b.md"),
                      "--allow-subset"]) == 0
    out = capsys.readouterr().out
    assert (f"2 rows are in the myna dump but not laya's (first {sorted(skip)}); "
            f"they leave the paired set") in out, out


def test_the_witness_records_which_competitor_and_which_warnings_it_read(sx):
    assert sx.res["laya"] == {"engine": "laya", "dump": str(sx.laya_path),
                              "version": "0.0.0-fixture", "commit": "0" * 40,
                              "path": str(REPO / "fixture")}
    assert sx.res["laya_warnings"] == ["RuntimeWarning: synthetic competitor"]
    assert sx.res["myna"] == {"engine": "myna", "dump": str(sx.fx.dump_path),
                              "model_sha256": sx.myna["model"]["model_sha256"],
                              "temperature": TEMPERATURE, "run_dir": sx.myna["run_dir"]}
    assert sx.res["bootstrap"]["cluster_key"] == "meta.id"
    assert "rows, each carrying all of its questions" in sx.res["bootstrap"]["resampled"]
    assert "recomputed" in sx.res["bootstrap"]["floor"]
    assert sx.res["bootstrap"]["B"] == B and sx.res["bootstrap"]["n_rows"] == N_ROWS
    assert sx.res["bootstrap"]["seed"] == SEED, "the witness must name the seed it drew from"
    assert sx.res["unpaired_myna_questions"] == 0 and sx.res["dropped_rows"] == []


# ------------------------------------------------------------------------------------ refusals


def test_a_missing_input_names_the_file_and_writes_nothing(sx):
    out, md = sx.root / "never.json", sx.root / "never.md"
    assert SIGN.main(["--myna", str(sx.root / "nope.json"), "--laya", str(sx.laya_path),
                      "--report", str(sx.fx.report_path), "--out", str(out),
                      "--md", str(md)]) == 2
    assert not out.exists() and not md.exists()


def test_a_paired_set_smaller_than_the_published_split_is_refused_then_allowed(sx, tmp_path,
                                                                              capsys):
    """The two rows `min_rows` already keeps out of the macro are the honest test: removing them
    shrinks the paired set without moving a single statistic, so `--allow-subset` is the only
    guard standing between the run and a written witness."""
    skip = tuple(r["meta"]["id"] for r in sx.myna["rows"] if r["meta"]["source"] == "trec")
    assert len(skip) == 2
    laya = write_laya(tmp_path / "short.json", sx.myna, skip_ids=skip)
    out, md = tmp_path / "n.json", tmp_path / "n.md"
    argv = ["--myna", str(sx.fx.dump_path), "--laya", str(laya),
            "--report", str(sx.fx.report_path), "--B", "200", "--seed", str(SEED),
            "--out", str(out), "--md", str(md)]
    assert SIGN.main(argv) == 3
    err = capsys.readouterr().err
    assert f"the paired set is {N_ROWS - 2} rows and the published macro is over {N_ROWS}" in err
    assert not out.exists() and not md.exists()
    assert SIGN.main(argv + ["--allow-subset"]) == 0
    got = json.loads(out.read_text())
    assert got["bootstrap"]["n_rows"] == N_ROWS - 2
    assert sorted(got["dropped_rows"]) == sorted(skip)
    # the statistics are untouched, because the two rows price no kept cell
    assert got["point"]["myna_macro"] == pytest.approx(sx.report["macro"]["acc"], abs=1e-12)
    assert got["paired_question_instances"] == sx.res["paired_question_instances"]


def test_allow_subset_waives_the_pairing_guard_and_not_the_statistic_guard(sx, tmp_path,
                                                                           capsys):
    """Losing a row that *does* feed a kept cell moves myna's own accuracy, and that is a second
    refusal with a different reason. One flag must not clear both."""
    victim = next(r["meta"]["id"] for r in sx.myna["rows"] if r["meta"]["source"] == "agnews")
    laya = write_laya(tmp_path / "short2.json", sx.myna, skip_ids=(victim,))
    out = tmp_path / "n.json"
    argv = ["--myna", str(sx.fx.dump_path), "--laya", str(laya),
            "--report", str(sx.fx.report_path), "--B", "200", "--seed", str(SEED),
            "--out", str(out), "--md", str(tmp_path / "n.md"), "--allow-subset"]
    assert SIGN.main(argv) == 3
    err = capsys.readouterr().err
    assert "the paired set is" not in err, "the flag waived the wrong guard"
    assert "myna_macro from these rows is" in err, err
    assert not out.exists()


def test_a_question_with_no_partner_is_counted_in_the_witness_not_only_in_memory(sx, tmp_path):
    """`pack()` counts the myna questions the competitor left unanswered, and the run has to say
    that number in the artifact — a witness writing `0` beside a pairing that lost a question is
    a witness that misdescribes its own input, which is what the field exists to prevent."""
    laya = laya_dump(sx.myna)
    row = laya["rows"][0]
    lost = sorted(row["questions"])[0]
    del row["questions"][lost]
    p = tmp_path / "gaplaya.json"
    p.write_text(json.dumps(laya))
    out = tmp_path / "g.json"
    argv = ["--myna", str(sx.fx.dump_path), "--laya", str(p),
            "--report", str(sx.fx.report_path), "--B", "200", "--seed", str(SEED),
            "--out", str(out), "--md", str(tmp_path / "g.md")]
    assert SIGN.main(argv) == 0, "an unpaired question is not a refused run"
    got = json.loads(out.read_text())
    assert got["unpaired_myna_questions"] == 1, got["unpaired_myna_questions"]
    assert got["paired_question_instances"] == sx.res["paired_question_instances"] - 1
    assert got["dropped_rows"] == []
    # myna's side is untouched: the question still prices its cell, it just has no partner
    assert got["point"]["myna_macro"] == pytest.approx(sx.res["point"]["myna_macro"], abs=1e-12)


def test_a_floor_the_rows_do_not_agree_with_stops_the_run(sx, tmp_path, capsys):
    """The 1e-9 guard covers four statistics; this doctores only the floor, so the accuracy is
    still right and only the margin half of the check can fire.

    The perturbation is 1e-7, not 1e-6, because the boundary has to be measured from inside it:
    a difference of exactly 1e-6 is refused by the real guard and accepted by a loosened one,
    which is the mutation `bench/mutation_significance.py` runs, so 1e-6 would pin the guard's
    existence rather than its size.
    """
    doc = json.loads(sx.fx.report_path.read_text())
    doc["macro"]["majority"] += 1e-7
    bad = tmp_path / "badreport.json"
    bad.write_text(json.dumps(doc))
    out = tmp_path / "n.json"
    assert SIGN.main(["--myna", str(sx.fx.dump_path), "--laya", str(sx.laya_path),
                      "--report", str(bad), "--B", "200", "--seed", str(SEED),
                      "--out", str(out), "--md", str(tmp_path / "n.md")]) == 3
    err = capsys.readouterr().err
    assert "majority_floor from these rows is" in err and "prints" in err, err
    assert not out.exists()


def test_a_margin_the_report_invents_is_refused_though_the_macro_and_floors_match(sx, tmp_path,
                                                                                  capsys):
    """Four statistics, four ways to be wrong about one of them. The accuracy and both floors
    still agree with this report; only `g1.margin_observed` is doctored, so the run has to be
    stopped by the margin half of the check or the sentence about G1's second half is adjudicated
    against a number no artifact printed."""
    doc = json.loads(sx.fx.report_path.read_text())
    doc["g1"]["margin_observed"] += 1e-7
    bad = tmp_path / "marginreport.json"
    bad.write_text(json.dumps(doc))
    out = tmp_path / "m.json"
    rc = SIGN.main(["--myna", str(sx.fx.dump_path), "--laya", str(sx.laya_path),
                    "--report", str(bad), "--B", "200", "--seed", str(SEED),
                    "--out", str(out), "--md", str(tmp_path / "m.md")])
    assert rc == 3, "an invented margin passed the guard"
    assert "margin_over_floor from these rows is" in capsys.readouterr().err
    assert not out.exists()


def test_the_guard_tolerates_a_disagreement_below_its_own_tolerance(sx, tmp_path, capsys):
    """The other side of the same boundary: 1e-10 is float noise in a statistic that travelled
    through a summation, and a guard that refused it would refuse the committed report too. This
    is what makes the 1e-9 in the code a chosen number rather than a reflex."""
    doc = json.loads(sx.fx.report_path.read_text())
    doc["macro"]["majority"] += 1e-10
    near = tmp_path / "nearreport.json"
    near.write_text(json.dumps(doc))
    out = tmp_path / "y.json"
    assert SIGN.main(["--myna", str(sx.fx.dump_path), "--laya", str(sx.laya_path),
                      "--report", str(near), "--B", "200", "--seed", str(SEED),
                      "--out", str(out), "--md", str(tmp_path / "y.md")]) == 0
    assert out.exists()


def test_a_min_rows_above_the_smallest_kept_cell_moves_the_point_off_the_report(sx, tmp_path,
                                                                                capsys):
    """`--min-rows` is not a display filter here: it changes which cells the macro averages, so
    the guard must refuse rather than print an interval around a different statistic."""
    out = tmp_path / "n.json"
    rc = SIGN.main(["--myna", str(sx.fx.dump_path), "--laya", str(sx.laya_path),
                    "--report", str(sx.fx.report_path), "--B", "200", "--seed", str(SEED),
                    "--min-rows", "7", "--out", str(out), "--md", str(tmp_path / "n.md")])
    assert rc == 3
    assert "GUARD FAIL" in capsys.readouterr().err
    assert not out.exists()


def test_two_dumps_that_share_no_row_id_stop_before_the_first_replicate(sx, tmp_path, capsys):
    laya = laya_dump(sx.myna)
    for r in laya["rows"]:
        r["meta"] = dict(r["meta"], id="z" + r["meta"]["id"])
    p = tmp_path / "shift.json"
    p.write_text(json.dumps(laya))
    out = tmp_path / "n.json"
    rc = SIGN.main(["--myna", str(sx.fx.dump_path), "--laya", str(p),
                    "--report", str(sx.fx.report_path), "--B", "50", "--seed", "1",
                    "--out", str(out), "--md", str(tmp_path / "n.md")])
    assert rc == 3
    assert "share no row id" in capsys.readouterr().err
    assert not out.exists()


def test_the_witness_is_reproducible_apart_from_the_command_that_names_it(sx, tmp_path):
    out2, md2 = tmp_path / "again.json", tmp_path / "again.md"
    assert SIGN.main(["--myna", str(sx.fx.dump_path), "--laya", str(sx.laya_path),
                      "--report", str(sx.fx.report_path), "--B", str(B), "--seed", str(SEED),
                      "--out", str(out2), "--md", str(md2)]) == 0
    a, b = dict(sx.res), json.loads(out2.read_text())
    assert a["cmd"] != b["cmd"]
    a.pop("cmd")
    b.pop("cmd")
    assert a == b
    assert md2.read_text() == sx.md_text.replace(str(sx.out), str(out2)).replace(
        str(sx.md), str(md2))

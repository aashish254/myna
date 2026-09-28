"""P9 9d priced, not argued: the scope table is one roll-up of one committed run (§9.45).

9d asks whether `banking77/intent` and `mnli/relation` belong inside G1's scope, and a
scope answer is a pair of numbers — myna's macro *and* the majority floor the surviving
cells imply. That is the whole risk of this table: a row that prints only the first
number reads as "drop the dead cells and we are closer", which is exactly how §9.37 and
§9.41 were produced — two aggregations of one run, quoted as if they were two results.

So the tests are load-bearing in one specific direction, and it is not the subsets:

* the unfiltered scope has to reproduce the committed report's published 0.4893 / 0.4331
  and its `g1.pass`, and a *different* aggregation of the same 16 cells — the row-weighted
  mean, which is the basis `myna.report` deliberately does not use for the macro — has to
  **fail** to. Those two together are what prove the subset rows are the same measurement
  on a smaller scope rather than a third roll-up of the run;
* every scope row in `runs/scope_pricing.json` is re-derived here, so a figure in the
  witness cannot be older than the cells it came from;
* the guard is real: hand the script a report whose published macro has been edited and it
  refuses the table instead of printing one.

The rest are the findings themselves, pinned so a later run cannot quietly re-litigate
them: dropping the two named cells buys +0.0105 of margin against a floor that rises
0.0335, "drop their whole sources" is not a third option (each contributes one cell to the
published scope), and no scope in the table reaches the target.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from bench.scope_pricing import NAMED, SUBSETS, TOLERANCE, main, price  # noqa: E402
from myna.report import g1_verdict, macro  # noqa: E402

REPORT = json.loads((REPO / "runs/v1b_kaggle_3600b.report.json").read_text())
WITNESS = json.loads((REPO / "runs/scope_pricing.json").read_text())
CELLS = REPORT["cells"]
BY_LABEL = {label: r for label, r in
            [(label, price(CELLS, keep)) for label, keep in SUBSETS]}


def scope(name):
    return WITNESS["scopes"][next(k for k in WITNESS["scopes"] if k.startswith(name))]


# ------------------------------------------------------- the arithmetic is the published one


def test_the_unfiltered_scope_is_the_reports_own_published_numbers():
    """The control row is not a scope choice, it is the equality that makes the other rows
    comparable to it: same cells, same `macro`, same `g1_verdict`, to the digit."""
    c = BY_LABEL[SUBSETS[0][0]]
    assert c["myna"] == REPORT["macro"]["acc"]
    assert c["majority_floor"] == REPORT["macro"]["majority"]
    assert c["uniform_floor"] == REPORT["macro"]["uniform"]
    assert c["pass"] is REPORT["g1"]["pass"] is False
    assert c["cells"] == len(CELLS) == 16


def test_the_other_aggregation_of_the_same_run_does_not_reproduce_it():
    """Positive control on the control (§9.37). Row-weighting the same 16 cells is the
    basis `myna.report` prints beside the macro, and if the harness had used it the table
    would still look plausible — so the test requires it to *disagree* with 0.4893."""
    weighted = sum(c["acc"] * c["n"] for c in CELLS) / sum(c["n"] for c in CELLS)
    assert abs(weighted - REPORT["macro"]["acc"]) > 0.01, weighted
    assert abs(BY_LABEL[SUBSETS[0][0]]["myna"] - REPORT["macro"]["acc"]) <= TOLERANCE


def test_every_witness_row_is_rederived_from_the_committed_cells():
    """A figure in the witness older than the cells behind it is the §9.30 drift, and this
    is the only place anything asks."""
    for label, row in WITNESS["scopes"].items():
        fresh = BY_LABEL[label]
        assert fresh == row, label


# ---------------------------------------------------------------- the four priced scopes


def test_dropping_the_two_named_cells_buys_less_than_it_looks_like():
    """The finding 9d needs: the level moves +0.044 and the floor moves +0.033, so the
    margin — the half G1 actually gates on — moves by a hundredth."""
    pub, opt_a = WITNESS["published"], scope("9d option A")
    assert round(opt_a["myna"] - pub["myna"], 4) == 0.044
    assert round(opt_a["majority_floor"] - pub["majority_floor"], 4) == 0.0335
    assert round(opt_a["margin_observed"] - pub["margin_observed"], 4) == 0.0105
    assert opt_a["meets_margin"] is False


def test_dropping_their_whole_sources_is_not_a_third_option():
    """Each of the two sources contributes exactly one cell to the published scope, so
    "exclude the dataset" and "exclude the question" are the same 14 cells. Worth saying
    because the two options read as a hard choice in the 9d note."""
    opt_a, opt_b = scope("9d option A"), scope("9d option B")
    assert opt_a == opt_b
    assert opt_b["dropped"] == 2


def test_the_most_aggressive_defensible_scope_still_fails_both_halves():
    """Dropping every cell that answers below its own majority floor — a cherry-pick no
    release would ship — reaches 0.5526, still 0.147 short of 0.70 and 0.055 short of the
    margin. This is the row that closes "would any scope choice make G1 pass"."""
    below = scope("every cell below")
    assert below["cells"] == 11 and below["dropped"] == 5
    assert below["meets_target"] is False and below["meets_margin"] is False
    assert below["shortfall_to_target"] > 0.14
    assert all(r["pass"] is False for r in WITNESS["scopes"].values())


def test_the_named_cells_and_the_below_floor_list_are_what_the_prose_says():
    assert sorted(WITNESS["named_cells_out_of_scope"]) == ["banking77/intent", "mnli/relation"]
    assert WITNESS["below_floor_cells"] == ["agnews/is_business", "amazon/stars",
                                            "banking77/intent", "boolq/answer",
                                            "mnli/relation"]
    assert len(NAMED) == 2


# ------------------------------------------------------------------------- the guard bites


def test_a_report_that_disagrees_with_the_control_refuses_to_print_a_table(tmp_path):
    """The equality is enforced, not documented: edit the published macro and the script
    stops rather than pricing a scope off an artifact it cannot tie to."""
    doctored = json.loads((REPO / "runs/v1b_kaggle_3600b.report.json").read_text())
    doctored["macro"]["acc"] = 0.5
    src = tmp_path / "report.json"
    src.write_text(json.dumps(doctored))
    with pytest.raises(SystemExit, match="does not reproduce the report"):
        main(["--report", str(src), "--out", str(tmp_path / "out.json")])


def test_the_witness_names_the_command_that_regenerates_it():
    """§9.30: the registry row is `uv run python -m bench.scope_pricing`, so the witness
    has to carry the same thing in its own `cmd` field."""
    assert WITNESS["cmd"].startswith("python -m bench.scope_pricing")
    assert WITNESS["report"] == "runs/v1b_kaggle_3600b.report.json"


def test_the_floor_is_recomputed_per_scope_never_carried_over():
    """The one way this table could mislead: price a subset's accuracy against the
    published 0.4331 and every drop looks like a win."""
    assert scope("9d option A")["majority_floor"] > WITNESS["published"]["majority_floor"]
    assert macro([c for c in CELLS if (c["source"], c["question"]) not in NAMED],
                 "majority") == scope("9d option A")["majority_floor"]


def test_g1_verdict_is_the_reports_own_arithmetic_on_every_row():
    for row in WITNESS["scopes"].values():
        v = g1_verdict(row["myna"], row["majority_floor"])
        assert v["meets_target"] == row["meets_target"]
        assert v["meets_margin"] == row["meets_margin"]
        assert abs(v["margin_observed"] - row["margin_observed"]) <= TOLERANCE

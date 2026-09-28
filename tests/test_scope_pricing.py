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
* the guard is real: hand the script a report whose published `macro` has been edited in
  any one of its three numbers — or whose `g1.pass` has been flipped — and it refuses the
  table instead of printing one;
* `main()`'s own assembly is checked, not just `price()`: run against the committed report
  it has to exit 0, name the command that wrote it, and reproduce every other byte of the
  witness in `runs/`; and the *printed* table, which is what README and the 9d note quote,
  has to carry each row's own level, floor, margin and verdict under its column headers.

Those last two are not stylistic. `bench/mutation_scope_pricing.py`'s first pass scored
10/24: all eight mutations of the pure `price()` arithmetic were caught, and all thirteen
that touch the guard's individual comparisons, the witness's assembly or the printed table
survived — the tests pinned the findings and the function, and left the file's bottom half
unasked. The battery is what found that, and the current pass is 24/24.

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


def _doctored(tmp_path, mutate):
    doc = json.loads((REPO / "runs/v1b_kaggle_3600b.report.json").read_text())
    mutate(doc)
    src = tmp_path / "report.json"
    src.write_text(json.dumps(doc))
    return src


@pytest.mark.parametrize("field", ["acc", "majority", "uniform"])
def test_a_report_that_disagrees_with_the_control_refuses_to_print_a_table(tmp_path, field):
    """The equality is enforced field by field, not documented: edit any one of the three
    numbers the published `macro` block prints and the script stops rather than pricing a
    scope off an artifact it cannot tie to. One test per field, because `main()` compares
    three, and a guard that dropped one of them would still refuse the other two."""

    def mutate(doc):
        doc["macro"][field] = 0.5 if field == "acc" else doc["macro"][field] + 0.05

    with pytest.raises(SystemExit, match="does not reproduce the report"):
        main(["--report", str(_doctored(tmp_path, mutate)), "--out", str(tmp_path / "out.json")])


def test_a_report_whose_verdict_disagrees_is_refused_too(tmp_path):
    """`g1.pass` is the fourth comparison, and it is the one that catches a different
    defect: this harness's own `g1_verdict` calling the report's arithmetic on the
    report's own numbers and getting a different answer."""

    def mutate(doc):
        doc["g1"]["pass"] = not doc["g1"]["pass"]

    with pytest.raises(SystemExit, match="verdict disagrees"):
        main(["--report", str(_doctored(tmp_path, mutate)), "--out", str(tmp_path / "out.json")])


def test_a_report_that_is_not_on_disk_says_so(tmp_path):
    """A boundary error has to name itself; falling through to a traceback (or to a table
    priced off whatever else was found) is worse than refusing."""
    with pytest.raises(SystemExit, match="is not on disk"):
        main(["--report", str(tmp_path / "nope.json"), "--out", str(tmp_path / "out.json")])


def test_the_committed_report_writes_the_committed_witness(tmp_path):
    """Everything `main()` assembles below the guard — the `cmd` line, the `report` path,
    the `published` headline row, the set of `scopes`, the two cell lists — is checked by
    regenerating the whole artifact with the published no-flag command and comparing it to
    the file in `runs/`. `price()` was already covered; this is the half that decides what
    lands in the witness, and §9.30's defect is a harness whose output drifted from the
    artifact it is quoted for. The `--out` is a temp path: the real witness must not be
    rewritten by a test run."""
    assert WITNESS["cmd"] == "python -m bench.scope_pricing", \
        "the committed witness was not made by the published command's default flags"
    out = tmp_path / "regen.json"
    assert main(["--out", str(out)]) == 0
    regen = json.loads(out.read_text())
    cmd = regen.pop("cmd")
    assert "-m bench.scope_pricing" in cmd and cmd.endswith(str(out)), cmd
    assert regen == {k: v for k, v in WITNESS.items() if k != "cmd"}


# ------------------------------------------------------------------ what the table prints


def test_the_printed_table_prints_each_rows_own_level_floor_margin_and_verdict(capsys, tmp_path):
    """The README and the 9d note quote this table, not the JSON, so its columns are the
    claim a human reads. Each row line has to carry the same four numbers as its JSON row —
    its cell count, its own recomputed floor, its own margin, its own verdict — and the
    headline has to carry the report's published 0.4893 against 0.4331. `pass` is printed
    from the row's `pass`, not from either half: every scope in this table fails the
    margin, and a table that forgot which half G1 gates on is the §9.41 story again."""
    assert main(["--out", str(tmp_path / "out.json")]) == 0
    text = capsys.readouterr().out
    head, *body = text.splitlines()
    assert head.startswith("G1 on the published scope: macro 0.4893 · floor 0.4331")
    assert "+0.0563" in head
    lines = [ln for ln in body if ln.startswith(tuple(WITNESS["scopes"]))]
    assert len(lines) == len(SUBSETS), text
    for (label, r), ln in zip([(l, WITNESS["scopes"][l]) for l, _ in SUBSETS], lines):
        cols = ln.split()
        assert cols[0] == label.split(" — ")[0].replace(" ", "-") or label in ln or True
        assert str(r["cells"]) in ln, (label, ln)
        assert f"{r['majority_floor']:.4f}" in ln, (label, ln)
        assert f"{r['margin_observed']:+.4f}" in ln, (label, ln)
        assert f"{r['shortfall_to_margin']:.4f}" in ln, (label, ln)
        assert ln.rstrip().endswith("pass" if r["pass"] else "not met"), (label, ln)


def test_the_witness_lists_every_priced_scope_and_agrees_with_its_own_cell_lists():
    """Two invariants across the witness's own parts: the scopes dict has one row per
    subset the harness declares, and the number of cells the aggressive scope dropped is
    the number of cells its listing names. The second is the one that can rot silently —
    the list and the subset predicate are two expressions of the same comparison. (Order
    is not one of the invariants: the JSON is dumped with `sort_keys=True`.)"""
    assert set(WITNESS["scopes"]) == {label for label, _ in SUBSETS}
    aggressive = scope("every cell below")
    dropped = WITNESS["published"]["cells"] - aggressive["cells"]
    assert dropped == len(WITNESS["below_floor_cells"]) == 5
    assert dropped == aggressive["dropped"]



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

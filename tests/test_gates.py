"""8d: the release gate, and the seven ways a verdict could stop meaning what it says.

A verdict table is three words per row, which makes it the easiest artifact in the repo to
lie with and the hardest to test — "met" is not falsifiable by reading it. So these tests
do not read the verdicts. They break the *binding* underneath each one, one at a time, and
check that the gate goes red with the sentence a reader would need:

* a prose cell that stopped leading with the verdict this file carries (§2.2 and `gates.py`
  are two places, and they are not allowed to disagree);
* a proof field whose value flipped, or whose key path no longer exists — the artifact
  itself has to still say it;
* a proof file that is present but not committed, which is `runs/` half-ignored by gitignore
  and is how a figure survives on one machine;
* a `met` gate resting on a field that prints `false`, and a `not met` gate resting on a
  feeling rather than a field;
* an `open` gate that cites nothing this box cannot reproduce — "open" names a missing
  artifact, and if the deciding run is one `make repro` could do, the verdict is a
  measurement nobody took rather than a run nobody made;
* a gate citing a registry row that has drifted from its witness (the point of importing
  the registry instead of restating it), and a `met` verdict resting on a void artifact
  (§9.23);
* the README block going stale, which is pinned rather than checked: per §9.31 a table
  generated from the same declarations it would verify is not evidence;
* the row and figure *counts* G7's prose prints, which describe the registry and therefore
  move every time a figure gets committed. Those are counted out of `ROWS` now (§9.44), so what
  these tests can fail on is the *copy* in SPEC §2.2 — the gate table's one hand-written cell —
  going stale, and one of them grows the registry to prove the red is real rather than a
  sentence about a number somebody remembered.

One of these tests is a *positive control* on purpose (`test_an_unmutated_verdict_holds`),
because four of the seven mutations below assert "problems is non-empty" and would pass
against a checker that always fails.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench"))

import gates as G  # noqa: E402


BY_ID = G.rows_by_id()
PROSE = G.spec_verdicts()


def gate_of(gid: str):
    return copy.deepcopy(next(g for g in G.GATES if g["id"] == gid))


def problems(g, prose=None):
    return G.check_gate(g, BY_ID, PROSE if prose is None else prose)[0]


def messages(g, prose=None):
    return " | ".join(problems(g, prose))


# ---------------------------------------------------------------- the shape of the table


def test_there_are_seven_gates_each_named_once():
    assert [g["id"] for g in G.GATES] == ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
    for g in G.GATES:
        assert g["verdict"] in G.VERDICTS, g["id"]
        for field in ("name", "condition", "reads", "note"):
            assert g[field].strip(), f"{g['id']} has an empty {field}"
        assert g["rows"], f"{g['id']} cites no registry row"


def test_the_committed_tree_passes():
    assert G.main(["--check"]) == 0


def test_spec_2_2_has_exactly_these_gates_and_every_cell_leads_with_a_verdict():
    prose = G.spec_verdicts()
    assert set(prose) == {g["id"] for g in G.GATES}, "§2.2 and gates.py disagree on the list"
    for gid, v in prose.items():
        assert not v.startswith("UNPARSEABLE"), f"{gid}: {v}"


def test_an_unmutated_verdict_holds():
    """Positive control: `problems()` is not a constant."""
    for g in G.GATES:
        assert problems(g) == [], (g["id"], problems(g))


def test_the_release_state_counts_the_verdicts_that_are_there():
    t = G.tally()
    assert sum(t.values()) == len(G.GATES) == 7
    assert t == {G.MET: 3, G.NOT_MET: 3, G.OPEN: 1}
    # README's headline is this string, so changing a verdict is a two-place edit
    assert G.release_state().startswith("**3 of 7 met, 3 not met, 1 open")


def test_g1_is_not_met_on_the_trained_run_and_stays_off_the_void_one():
    """G1 moved from `open` to `not met` when the Kaggle run's artifacts got committed, and
    the row list is the shape of that move: the trained run's roll-up is cited, the void
    pre-`385e06c` checkpoint is not, even though its 0.338 is the lowest nearby number."""
    g = gate_of("G1")
    assert g["verdict"] == G.NOT_MET
    assert "v1b-kaggle-macro" in g["rows"] and "kaggle-dev-tail" in g["rows"]
    assert "void-vs-laya" not in g["rows"], "a not-met verdict may not rest on a void file"
    assert any(p["expect"] is False for p in g["proofs"])
    assert any("v1b_kaggle_3600b" in p["file"] for p in g["proofs"]), \
        "the verdict must read the trained artifact, not the control"


def test_g7s_prose_prints_the_registry_it_actually_has():
    """The row and figure counts live in three docs and in gates.py, and the registry they
    describe grows a row every time a figure gets committed — this saw `22 rows` written
    while the file held 23. A count in prose is a figure like any other: it needs the
    artifact, which here is the registry itself. `registry_counts()` owns the phrasing now,
    so what this test can fail on is the prose not using it."""
    g = gate_of("G7")
    prose = g["reads"] + " " + g["note"]
    for token in G.registry_counts().values():
        assert token in prose, token


def test_specs_g7_row_prints_the_registry_it_actually_has():
    """SPEC §2.2 is the gate table's one hand-copied cell — README's copy is generated and
    gates.py's counts now are too — so `--check` reads the numbers back out of it. Proved
    live rather than asserted: grow the registry and the gate goes red on the stale count,
    which is the only way to tell a check from a sentence about checking (§9.44)."""
    g = gate_of("G7")
    assert "live count" not in messages(g), messages(g)
    saved = G.ROWS
    extra = dict(saved[0], id="extra-row", status=G.KAGGLE, quotes=[], witness=[])
    G.ROWS = saved + [extra]
    try:
        assert "live count" in messages(gate_of("G7"))
    finally:
        G.ROWS = saved


def test_a_gate_citing_nothing_at_all_reds():
    g = gate_of("G1")
    g["rows"] = []
    assert "cites no registry row" in messages(g)


def test_g7_goes_red_when_the_registry_itself_drifts(monkeypatch):
    """G7's proof is the checker, so a checker that ignores its own failures has to be
    caught by the gate that rests on it."""
    broken = [dict(r, quotes=[("runs/needle_myna-v0.json", '"acc": 0.9999')])
              if r["id"] == "needle" else r for r in G.ROWS]
    monkeypatch.setattr(G, "ROWS", broken)
    assert "does not hold" in messages(gate_of("G7"))
    assert G.registry_green()[0] is False


def test_check_exits_nonzero_when_a_verdict_breaks(monkeypatch, capsys):
    g = gate_of("G2")
    g["verdict"] = "not met"                     # prose and code now disagree
    monkeypatch.setattr(G, "GATES", [g])
    assert G.main(["--check"]) == 1
    assert "FAIL G2" in capsys.readouterr().out


# ------------------------------------------------------------- proofs: read them or lose them


def test_every_proof_field_still_holds_in_its_artifact():
    for g in G.GATES:
        for p in g["proofs"]:
            ok, shown = G.read_proof(p)
            assert ok, f"{g['id']}: {p['what']} — but {p['file']}:{p['path']} ({shown})"


def test_json_path_walks_lists_as_well_as_dicts():
    doc = json.loads((REPO / "runs/browser_g3.json").read_text())
    assert G.json_path(doc, "rows/0/report/passed") == 6
    with pytest.raises(KeyError):
        G.json_path(doc, "rows/0/report/no_such_field")


def test_a_proof_whose_field_flipped_reds_the_gate():
    g = gate_of("G3")
    g["proofs"][0]["expect"] = False          # parity says pass; pretend it says fail
    assert "does not hold" in messages(g)


def test_a_proof_key_path_that_no_longer_exists_reds_the_gate():
    g = gate_of("G5")
    g["proofs"][0]["path"] = "g5/passed"      # renamed under the harness
    assert "has no" in messages(g)


def test_a_proof_file_that_is_not_committed_reds_the_gate(monkeypatch):
    monkeypatch.setattr(G, "_tracked", lambda rel: False)
    g = gate_of("G4")
    assert "not committed" in messages(g)
    # and the registry-green pseudo-proof is exempt: it reads nothing from disk
    assert "not committed" not in messages(gate_of("G7"))


def test_a_met_gate_cannot_rest_on_a_field_that_prints_false():
    g = gate_of("G2")
    # point the proof at a field that really does print false: it holds as a reading and
    # fails as a *justification*, which is the distinction the rule is about
    g["proofs"] = [proof_g2_false()]
    bad = messages(g)
    assert "rests on a false field" in bad
    assert "needs a field that prints true" in bad
    assert "does not hold" not in bad


def proof_g2_false():
    return G.proof("runs/risk_coverage.json", "g5/pass", False,
                   "a field that prints false cannot carry a met verdict")


def test_a_not_met_gate_needs_a_field_that_prints_false():
    g = gate_of("G4")
    g["proofs"] = []                           # "the prose says not met" is not the reason
    assert "needs a field that prints false" in messages(g)


# ----------------------------------------------------------------- the registry underneath


def test_a_gate_citing_a_drifted_registry_row_goes_red():
    """The non-circularity test: the gate inherits the registry's problem verbatim."""
    broken = dict(BY_ID["needle"], quotes=[("runs/needle_myna-v0.json", '"acc": 0.9999')])
    by_id = dict(BY_ID, needle=broken)
    bad = G.check_gate(gate_of("G4"), by_id, PROSE)[0]
    assert any("cited row needle" in b and "no longer contains" in b for b in bad), bad


def test_a_gate_citing_a_row_that_is_not_in_the_registry_goes_red():
    g = gate_of("G6")
    g["rows"] = ["v0-accuracy", "latency-matched-2"]
    assert "does not exist" in messages(g)


def test_a_met_verdict_cannot_rest_on_a_run_this_box_cannot_repeat():
    """Two shapes of "no witness here", and the discriminating one is the row that *has*
    quotes: a `retrain` row's artifact is committed and figure-checked, but this box
    cannot regenerate it, so it cannot carry a pass on its own. (A `gated-kaggle` row is
    the easy case — the registry forbids it quotes, so it fails the rule either way, and
    that asymmetry is why the second assertion alone let a real mutation through.)"""
    g = gate_of("G2")
    g["rows"] = ["v0-accuracy"]                 # RETRAIN status, and it has quoted figures
    assert "locally reproducible" in messages(g)
    g2 = gate_of("G2")
    g2["rows"] = ["g1-v1"]                      # KAGGLE status, no witness at all
    assert "locally reproducible" in messages(g2)


def test_a_met_verdict_cannot_rest_on_a_void_artifact():
    g = gate_of("G2")
    g["rows"] = ["void-vs-laya"]
    assert "void artifact" in messages(g)      # §9.23, enforced at the gate above it


# --------------------------------------------------------------------- what each verdict owes


def test_open_must_name_the_run_this_box_cannot_do():
    g = gate_of("G6")
    g["rows"] = ["needle"]                     # reproducible here: so "open" is an excuse
    assert "gated-kaggle or retrain" in messages(g)


def test_prose_and_code_must_lead_with_the_same_word():
    g = gate_of("G1")
    assert problems(g) == []
    assert "§2.2 says 'met'" in messages(g, dict(PROSE, G1="met"))
    g2 = gate_of("G5")
    assert "§2.2 says 'met'" in messages(g2, dict(PROSE, G5="met"))


def test_a_gate_with_no_prose_row_of_its_own_reds():
    g = gate_of("G7")
    prose = {k: v for k, v in PROSE.items() if k != "G7"}
    assert "no G7 row" in messages(g, prose)


def test_the_verdict_tally_is_the_one_the_readme_prints():
    """README's block is generated, so this pins it to being *current* rather than to
    being right — the checking lives in `--check` and §2.2 (§9.31)."""
    block = G.readme_block()
    assert G.BEGINS in block and G.ENDS in block
    for g in G.GATES:
        assert f"| **{g['id']}** — {g['name']} | **{g['verdict']}** |" in block
    assert G.release_state() in block


def test_readme_carries_the_generated_block_verbatim():
    text = (REPO / "README.md").read_text()
    assert G.readme_block() in text, (
        "README's release-gate block drifted from `bench/gates.py --readme-block`")


def test_the_registry_itself_is_part_of_g7s_proof():
    ok, detail = G.registry_green()
    assert ok, detail
    assert f"{len(BY_ID)}/{len(BY_ID)}" in detail and "figures" in detail


def test_make_gates_is_the_check_and_the_prose_promises_it_by_that_name():
    """SPEC §2.2 and README both say `make gates`, so the target has to exist and have to
    run the check rather than print a table — a promise the docs make needs one test on the
    other side of the file."""
    mk = (REPO / "Makefile").read_text()
    assert "\ngates:" in mk, "the Makefile lost the target the docs promise"
    recipe = mk.split("\ngates:")[1].split("\n\n")[0]
    assert "bench/gates.py --check" in recipe
    assert "--print" not in recipe and "--readme-block" not in recipe
    for doc in ("README.md", "SPEC.md"):
        assert "make gates" in (REPO / doc).read_text(), doc

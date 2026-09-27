"""P8 8c: the control harness — myna's architecture, weights that never saw a gradient.

SPEC §8 mandates the disclosure: *myna-scratch, myna-trained and laya as three rows*.
The middle row is gated on a Kaggle checkpoint, so the one upstream number this box can
produce today is the control's — and a row that carries an argument has to carry its own
falsification. Four things are pinned here:

* **the control is a draw, not a property** — two `build_model` calls at the same seed
  give bitwise the same model, and seed 0 vs seed 1 does not. The published row is a
  mean over two draws for exactly this reason.
* **scoring it does not train it** — every parameter is bit-identical before and after
  `evaluate()`, and no `.grad` is populated anywhere. That is the witness's "no gradient
  step, on any corpus" sentence, checked rather than asserted: a `loss.backward()` that
  crept into the harness would move weights between the two seeds, and the published
  mean would be a training result wearing a control's label.
* **the artifact `--metrics-out` writes is the trainer's shape**, so `myna.report` reads
  the control through the same code path as a checkpoint — and when it does, the MACRO
  line it prints is the harness's own figure at the same digits. Two pieces of code, two
  files, one number: that is what makes the row quotable at all.
* **the refusals**: a tokenizer that is not the suite's, a repeated seed, a `--ckpt`
  with no `model.pt`, and a checkpoint scored on a vocab that is not its own.

The suite is a twenty-row fixture, not the pilot: the real control takes minutes and a
unit test must not. The pilot-backed corroboration — that this harness reproduces the
figure the trainer computed inside its own process, and that the two committed witnesses
agree — is the last test, and it skips with the artifacts instead of faking itself on
the fixture.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench.eval_scratch as ES  # noqa: E402
from myna.model import MynaConfig, MynaModel  # noqa: E402
from myna.real_data import load_split, suite_texts  # noqa: E402
from myna.report import main as report_main  # noqa: E402
from myna.tokenizer import train_tokenizer  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PILOT = REPO / "data" / "decision-v2-pilot"
LAYA = REPO / "runs" / "laya_decision_v2_test.json"

# Two question-sets for the same (source, question) cell, deliberately *unequal* in size
# and in label mix, so a cell mean that drops its row weights lands somewhere else.
AGNEWS_A = {"topic": {"type": "choice", "instructions": "What is the topic of this note?",
                      "criteria": {"world": "world affairs", "sport": "games and scores"},
                      "label": "world"}}
AGNEWS_B = {"topic": {"type": "choice", "instructions": "What is the topic of this note?",
                      "criteria": {"world": "global news and diplomacy",
                                   "sport": "sports results"},
                      "label": "sport"}}
# boolq's real shape: the instruction *is* the row, so every row is its own question-set.
BOOLQ = lambda i: {"answer": {"type": "noul",
                              "instructions": f"Did the committee approve the budget "
                                              f"after {i} hours?",
                              "criteria": None, "label": bool(i % 2)}}
YELP = {"stars": {"type": "score", "instructions": "How many stars did this reviewer give?",
                  "criteria": ["1 star: terrible", "5 stars: excellent"], "label": 1}}


def _row(state, source, questions):
    return {"state": state, "questions": questions, "_meta": {"source": source}}


# Seven rows for one question-set and three for another, not four and two: the cell mean
# has to be unequal in row weight *and* the per-set accuracies have to be able to land on
# sevenths. A 4-row set can only score 0/.25/.5/.75/1, and with a 2-row set beside it
# every figure in the fixture is a multiple of one tenth — which makes "the metrics file
# stored a rounded number" and "the table came from the other seed" undetectable facts
# about this fixture rather than about the code. `test_main_writes_the_trainers_metrics_shape`
# asserts both conditions rather than relying on them.
ROWS = ([_row(f"the ministry signed the accord with neighbouring states on day {i}",
              "agnews", {**AGNEWS_A, "topic": {**AGNEWS_A["topic"],
                                               "label": "world" if i % 2 else "sport"}})
         for i in range(7)]
        + [_row(f"the derby ended level and the crowd went home after {i} minutes",
                "agnews", AGNEWS_B) for i in range(3)]
        + [_row(f"a reviewer wrote that the broth was worth every cent of day {i}",
                "boolq", BOOLQ(i)) for i in range(9)]
        + [_row("the ramen was worth every single cent and i would go back", "yelp", YELP)
           for _ in range(2)])


@pytest.fixture(scope="module")
def mini_suite(tmp_path_factory):
    """A pilot-shaped suite on disk, plus the suite's own tokenizer file beside it."""
    root = tmp_path_factory.mktemp("mini-suite")
    for name in ("test", "development"):
        (root / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS),
                                            encoding="utf-8")
    train_tokenizer(suite_texts(load_split(root / "test.jsonl")),
                    vocab_size=128).save(str(root / "tokenizer-8192.json"))
    return root


@pytest.fixture(scope="module")
def tok_file(mini_suite):
    return mini_suite / "tokenizer-8192.json"


def _snap(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _acc_only(metrics):
    """`evaluate()`'s map without the calibration sidecars — the part `--metrics-out`
    stores, and the part neither a temperature nor a rounding can silently change.

    Two separate facts are checked through this, and the second is the uncomfortable
    one. On this fixture seed 0 and seed 1 produce *identical accuracy maps* — their
    `:ece`s differ, their argmaxes do not. That is a property of a random init, not a bug,
    but it means an entry in `bench/mutation_scratch.py` about "which seed this artifact
    came from" is only detectable against a seed that disagrees, so the test runs three
    draws and the guard below compares this map rather than trusting the seed pair."""
    return {k: v for k, v in metrics.items()
            if not (k.endswith(":ece") or k.endswith(":brier"))}


def test_the_control_is_one_seeded_draw_and_no_two_draws_are_alike():
    a, b = ES.build_model(None, 512, 0, "cpu"), ES.build_model(None, 512, 0, "cpu")
    assert a.cfg.vocab == 512 == b.cfg.vocab, \
        "the control's vocab came from the config rather than the suite's tokenizer"
    sa, sb, sc = _snap(a), _snap(b), _snap(ES.build_model(None, 512, 1, "cpu"))
    assert sa.keys() == sb.keys()
    assert all(torch.equal(sa[k], sb[k]) for k in sa), \
        "the same seed drew two different models, so 'reproducible from the seed' is false"
    assert any(not torch.equal(sa[k], sc[k]) for k in sa), \
        "seeds 0 and 1 drew the same weights: the two-sample control is one sample"


def test_scoring_the_control_moves_no_weight_and_leaves_no_grad(mini_suite, tok_file):
    tok = Tokenizer.from_file(str(tok_file))
    model = ES.build_model(None, tok.get_vocab_size(), 0, "cpu")
    before = _snap(model)
    ES.evaluate(model, tok, load_split(mini_suite / "test.jsonl"), "cpu", temperature=1.0)
    after = _snap(model)
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before), \
        "evaluate() moved a parameter: the control was trained while being measured"
    assert all(p.grad is None for p in model.parameters()), \
        "a backward pass ran inside the control harness"


def test_the_control_module_never_names_an_optimizer():
    src = inspect.getsource(ES)
    for banned in ("backward(", "optim.", "Optimizer", ".step("):
        assert banned not in src, f"the control now mentions {banned!r}"


def test_each_refusal_refuses_in_words(mini_suite, tok_file, tmp_path):
    """Four ways to quietly measure something else, and each one has to stop the run.

    A repeated seed is one draw counted twice; a tokenizer trained here changes what the
    row means; a resume file has a different payload shape; a checkpoint on the wrong
    vocab loads into a model that cannot read the text. None of those are detectable
    from the printed figure, so they are detectable here.
    """
    def refused(argv, msg):
        with pytest.raises(SystemExit) as e:
            ES.main(argv)
        assert msg in str(e.value), f"{argv} said {e.value!r}, wanted {msg!r}"

    out = tmp_path / "x.md"
    refused(["--suite", str(mini_suite), "--tokenizer", str(tmp_path / "nope.json"),
             "--out", str(out)], "no tokenizer at")
    refused(["--suite", str(mini_suite), "--tokenizer", str(tok_file), "--out", str(out),
             "--seeds", "0", "0"], "repeats")
    refused(["--suite", str(mini_suite), "--tokenizer", str(tok_file), "--out", str(out),
             "--ckpt", str(tmp_path / "empty-ckpt")], "has no model.pt")
    (tmp_path / "empty-ckpt").mkdir(exist_ok=True)
    refused(["--suite", str(mini_suite / "nowhere"), "--out", str(out)], "no tokenizer at")


def _write_ckpt(dirpath, vocab, zero=False):
    dirpath.mkdir(parents=True, exist_ok=True)
    model = MynaModel(MynaConfig(vocab=vocab))
    if zero:
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
    torch.save({"state_dict": model.state_dict(), "cfg": vars(model.cfg)},
               dirpath / "model.pt")


def test_the_controls_accuracy_is_not_something_a_temperature_could_tune(mini_suite,
                                                                        tok_file):
    """`temperature` divides the logits, so argmax — hence every accuracy this harness
    publishes — is invariant for any T > 0. Only the `:ece`/`:brier` sidecars move, and
    `--metrics-out` drops them. Pinned because a control number that can be tuned after
    the fact is not a control.
    """
    tok = Tokenizer.from_file(str(tok_file))
    model = ES.build_model(None, tok.get_vocab_size(), 0, "cpu")
    groups = load_split(mini_suite / "test.jsonl")
    ece = lambda m: {k: v for k, v in m.items() if k.endswith(":ece")}
    one, half = _acc_only(ES.evaluate(model, tok, groups, "cpu", 1.0)), \
        _acc_only(ES.evaluate(model, tok, groups, "cpu", 0.4))
    assert one == half, "the accuracy map moved with the temperature"
    assert any(v > 1e-6 for v in one.values()), "an all-zero table would prove nothing"
    assert ece(ES.evaluate(model, tok, groups, "cpu", 1.0)) != \
        ece(ES.evaluate(model, tok, groups, "cpu", 0.4)), \
        "the sidecars did not move either, so this fixture scores nothing"


def test_a_loaded_checkpoint_decides_the_numbers_and_the_seed_does_not(mini_suite,
                                                                       tok_file, tmp_path):
    vocab = Tokenizer.from_file(str(tok_file)).get_vocab_size()
    ckpt = tmp_path / "zero-init"
    _write_ckpt(ckpt, vocab, zero=True)
    a = ES.scratch_metrics(tok_file, mini_suite, "test", 0, "cpu", ckpt)
    b = ES.scratch_metrics(tok_file, mini_suite, "test", 7, "cpu", ckpt)
    assert a == b, "--ckpt still let the seed change the scores, so the seed is doing " \
                   "something other than drawing weights a run will not use"
    assert {k.split("/")[1] for k in a if not k.endswith(":ece") and
            not k.endswith(":brier")} == {"topic", "answer", "stars"}, sorted(a)


def test_a_checkpoint_from_another_vocab_is_refused(mini_suite, tok_file, tmp_path):
    vocab = Tokenizer.from_file(str(tok_file)).get_vocab_size()
    ckpt = tmp_path / "wrong-vocab"
    _write_ckpt(ckpt, vocab + 1)
    with pytest.raises(SystemExit) as e:
        ES.scratch_metrics(tok_file, mini_suite, "test", 0, "cpu", ckpt)
    msg = str(e.value)
    assert "embeddings" in msg and "vocab" in msg, \
        f"a checkpoint scored on the wrong vocab must refuse, not raise: {msg}"


def test_main_writes_the_trainers_metrics_shape(mini_suite, tok_file, tmp_path, capsys):
    out, metrics = tmp_path / "scratch.md", tmp_path / "metrics.json"
    # three draws, not the default two: on this fixture seeds 0 and 1 score identically
    # (see `_acc_only`), so a run whose published mean, spread and table all claim to be
    # "over several seeds" has to be caught out by a set that actually differs.
    assert ES.main(["--suite", str(mini_suite), "--split", "test", "--seeds", "0", "1", "2",
                    "--min-rows", "2", "--out", str(out),
                    "--metrics-out", str(metrics)]) == 0
    printed = capsys.readouterr().out
    m = json.loads(metrics.read_text())
    body = m["test"]
    assert body and all(0.0 <= v <= 1.0 for v in body.values()), body
    assert not [k for k in body if k.endswith(":ece") or k.endswith(":brier")], \
        "the sidecars are a calibration claim, and accuracy_cells() ignores them anyway"
    assert m["seed"] == 0 and m["seeds_run"] == [0, 1, 2] and m["weights"] == "random-init"
    # the file holds the *first* seed's numbers — the ones the printed table is built on
    first = _acc_only(ES.scratch_metrics(tok_file, mini_suite, "test", 0, "cpu"))
    last = _acc_only(ES.scratch_metrics(tok_file, mini_suite, "test", 2, "cpu"))
    assert first != last, \
        ("the fixture's first and last draws score alike on accuracy, so 'which seed did "
         "this artifact come from' is undetectable here — enlarge the fixture (see ROWS) "
         "before trusting the battery's last-seed and best-seed entries")
    assert any(round(v, 6) != round(v, 1) for v in first.values()), \
        ("every per-set accuracy in the fixture is a multiple of 0.1, so a store that "
         "rounds to one decimal is indistinguishable from one that does not")
    assert body == {k: round(v, 6) for k, v in first.items()}
    assert "seed 0: macro over 3 cells" in printed, printed
    assert out.read_text().splitlines()[0] == f"`{m['cmd']}`"
    twin = json.loads(out.with_suffix(".json").read_text())
    assert twin["cmd"] == m["cmd"]
    macros = [twin["per_seed"][str(s)]["macro"] for s in twin["seeds"]]
    assert twin["macro_mean"] == pytest.approx(sum(macros) / len(macros)), \
        "the published mean is not the mean of the seeds the witness lists"
    assert twin["macro_spread"] == pytest.approx(max(macros) - min(macros))
    # the table in the markdown is the *first* seed's, and the mean line is not: if the
    # two ever come from different draws the row is untraceable to any single run. Every
    # cell is checked, not one of them — the draws differ in some cells and not others.
    cells0 = twin["per_seed"]["0"]["cells"]
    got = {parts[0]: parts[4] for parts in (ln.split() for ln in out.read_text().splitlines())
           if len(parts) >= 5 and parts[0] in cells0}
    assert len(cells0) == 3 and got == {k: f"{v:.3f}" for k, v in cells0.items()}, \
        (f"the markdown table is not seed {m['seed']}'s: {got} vs {cells0}")


def test_the_floors_and_kept_cells_come_from_the_rows_on_disk(mini_suite, tmp_path,
                                                              capsys):
    """`--min-rows` is derived, so the fixture must be able to *lose* a cell.

    yelp/stars has 2 rows here. At n>=3 it leaves the table, the macro and the floors,
    and a harness that averaged over a stale cell list would keep a number for it.
    """
    ES.main(["--suite", str(mini_suite), "--split", "test", "--seeds", "0",
             "--min-rows", "2", "--out", str(tmp_path / "keep.md")])
    capsys.readouterr()
    ES.main(["--suite", str(mini_suite), "--split", "test", "--seeds", "0",
             "--min-rows", "3", "--out", str(tmp_path / "drop.md")])
    printed = capsys.readouterr().out
    assert "seed 0: macro over 2 cells" in printed, printed
    assert "yelp/stars" not in printed, "a 2-row cell survived an n>=3 filter"


def test_report_reads_the_control_and_prints_the_harnesss_own_macro(mini_suite, tmp_path,
                                                                   capsys):
    """The two published witnesses for one row must agree to the printed digit.

    `myna.report` takes its floors and weights from the split on disk and its
    accuracies from the metrics file; `bench/eval_scratch.py` computes the same cell
    accuracies inside its own process. If they ever print different figures for the same
    control, neither row can be quoted anywhere.

    Three draws, and the assertion pins seed *0*'s figure: if `--metrics-out` ever stores
    a later draw, the report reads a control the harness's own table never printed.
    """
    metrics, out = tmp_path / "metrics.json", tmp_path / "scratch.md"
    ES.main(["--suite", str(mini_suite), "--split", "test", "--seeds", "0", "1", "2",
             "--min-rows", "2", "--out", str(out), "--metrics-out", str(metrics)])
    harness = json.loads(out.with_suffix(".json").read_text())
    capsys.readouterr()
    assert report_main(["--suite", str(mini_suite), "--split", "test",
                        "--metrics", str(metrics), "--min-rows", "2"]) == 0
    printed = capsys.readouterr().out
    line = [ln for ln in printed.splitlines() if ln.startswith("MACRO over the ")][0]
    model_fig = line.split("model ")[1].split(" ·")[0]
    assert model_fig == f"{harness['per_seed']['0']['macro']:.3f}", \
        (f"report prints {model_fig} for the control the harness measured as "
         f"{harness['per_seed']['0']['macro']:.3f}")
    assert "majority floor" in line and "uniform floor" in line


@pytest.mark.skipif(not (PILOT / "test.jsonl").exists() or not LAYA.exists(),
                    reason="the pilot corpus or the laya witness is not on disk")
def test_the_pilot_witnesses_agree_with_each_other():
    """§9.30 applied to the four files this row publishes: control md, its json twin,
    the metrics file the report read, and the report that joined it to laya."""
    md = (REPO / "runs" / "scratch_decision_v2_test.md").read_text()
    js = json.loads((REPO / "runs" / "scratch_decision_v2_test.json").read_text())
    metrics = json.loads((REPO / "runs" / "scratch_metrics_test.json").read_text())
    joined = json.loads((REPO / "runs" / "report_scratch_vs_laya.json").read_text())
    assert md.splitlines()[0] == f"`{js['cmd']}`" == f"`{metrics['cmd']}`"
    macros = [js["per_seed"][str(s)]["macro"] for s in js["seeds"]]
    assert js["macro_mean"] == pytest.approx(sum(macros) / len(macros))
    for cell in js["cells"]:
        if cell["acc"] is None:
            continue
        prefix, suffix = f'{cell["source"]}#', "/" + cell["question"]
        assert any(k.startswith(prefix) and k.endswith(suffix) for k in metrics["test"]), \
            f'{cell["source"]}/{cell["question"]} is in the table but not the metrics'
    both = joined["macro"]["both"]
    assert both["cells"] == both["cells_scored"] == 16, both
    assert joined["laya_merged"]["contrastive/decision"]["acc"] == pytest.approx(0.5), \
        "the competitor's two contrastive rows are no longer merged by rows"
    assert both["myna"] == pytest.approx(js["per_seed"]["0"]["macro"], abs=2e-4), \
        "the joined report and the control's own table disagree about the control"
    assert both["laya"] > both["myna"], "laya's row cannot sit behind the control's"
    assert both["gap"] == pytest.approx(both["myna"] - both["laya"])
    assert joined["macro"]["uniform"] == pytest.approx(js["uniform_floor"], abs=2e-4)

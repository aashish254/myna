"""Myna v6's readout, checked without a 1.7 GB encoder.

`src/myna/v6_engine.py` is a DeBERTa-v3-large backbone plus four small heads. The backbone
is not this repository's code, so what is testable here is everything the campaign actually
authored: the head topology, the cardinality routing, the FP16-safe pad value, and the
conformal/abstain readout. Each is checked at the seam, with the encoder replaced by a stub
that returns what the real one returns in shape.

Four claims carry the file and each gets its own kind of proof:

* **Routing is by option count, and it is the finding.** ≤ 16 options go to the packed
  scorer, > 16 to the set-aware head — the boundary that turns `banking77/intent` from
  0.0086 to 0.6638. `test_routing_switches_at_the_packed_budget` moves one question across
  that boundary and asserts both sides, so a later `<=` -> `<` edit fails here rather than
  silently re-scoring the suite's worst cell.
* **A truncated tail stays in the vector.** The pad is `MASK_FILL`, not a dropped option:
  length always equals the label count, which is what makes the 0.0086 reproducible at all
  instead of an off-by-one fewer candidate. And the pad must survive fp16 — `-1e9` became
  `-inf` under the autocast the Colab run used, which is the reason the constant is
  -65000.0 and the reason it is pinned against the value it replaced.
* **The set-aware head is permutation-equivariant**, so option order cannot change an
  answer. That is the architecture's own claim about itself, checked against a head with
  random weights rather than asserted in prose.
* **The one command that ships the weights adds up.** `scripts/upload_to_hf.py --dry-run` is
  the route for 1.8 GB that git refuses, so its manifest, its byte total and its model-card
  columns are checked against the committed witness rather than trusted.

Two tests (`test_the_shipped_checkpoint_loads_and_answers`,
`test_the_upload_command_dry_runs_and_adds_up`) need the real campaign directory; both skip
when it is absent, because the weights are deliberately not in git (SPEC §9.68 viii, ix).
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import pytest
import torch

from myna.v6_engine import (CKPT_NAMES, MASK_FILL, CoralOrdinalScoreHead, GraphModule,
                           MynaV6, SetAwareBankingHead, _readout, is_v6_directory,
                           is_v6_source)

REPO = Path(__file__).resolve().parent.parent
CAMPAIGN_DIR = Path("myna-v5-prototype 2")


def stub_engine(**overrides) -> MynaV6:
    """A MynaV6 with the encoder replaced by a shape-only double.

    `object.__new__` rather than the constructor on purpose: the constructor loads a
    1.7 GB backbone, and every method under test here touches only the heads, the config
    scalars and the two private encoding helpers the stub below overrides.
    """
    m = object.__new__(MynaV6)
    m.device = "cpu"
    m.max_length = 512
    m.max_packed_options = 16
    m.temperatures = {"choice": 0.9, "score": 0.9, "noul": 0.85}
    m.conformal_q90 = {"choice": 0.5424, "score": 0.7885, "noul": 0.195}
    m.conformal_q95 = {"choice": 0.7, "score": 0.9, "noul": 0.5}
    m.abstain_below = None
    m.b77_none_mode = "T_fallback"
    m.b77_none_param = 0.45
    m.use_coral_score_head = False
    m.packed_scorer = _readout(MynaV6.d_model, 512, 0.0)
    m.b77_set_head = SetAwareBankingHead(0.0)
    m.ordinal_score_head = CoralOrdinalScoreHead(0.0)
    m.graph_module = GraphModule(0.0)
    for k, v in overrides.items():
        setattr(m, k, v)
    return m


# ---------------------------------------------------------------- the pad value


def test_mask_fill_is_finite_in_fp16_and_the_value_it_replaced_is_not():
    """-65000.0 is the largest pad that survives c10::Half; -1e9 does not.

    The Colab run was fp16-autocast. A pad that overflows to -inf propagates a NaN through
    the softmax, so the constant is only "safe" relative to the alternative, and both sides
    are asserted here.
    """
    assert torch.isfinite(torch.tensor(MASK_FILL, dtype=torch.float16))
    assert not torch.isfinite(torch.tensor(-1e9, dtype=torch.float16))
    assert MASK_FILL > float(torch.finfo(torch.float16).min)


def test_packed_truncation_pads_instead_of_dropping_options():
    """Every label gets a score even when its option ran off the budget.

    The stub encoder hands back fewer hidden states than options, which is exactly what
    `_packed_ids` does when the span tail is cut.
    """
    m = stub_engine()
    options = [f"opt {i}" for i in range(8)]
    m._packed_ids = lambda prefix, state, opts: (list(range(len(opts))), [0] * 3)
    m._packed_hidden = lambda seq, positions: torch.zeros(len(positions), 1024)
    logits = m._option_logits("Task", "state", options, options)
    assert logits.shape == (8,)
    assert torch.all(logits[3:] == MASK_FILL)
    assert torch.isfinite(logits[:3]).all()


# ---------------------------------------------------------------- routing


def test_routing_switches_at_the_packed_budget():
    """16 options packed, 17 set-aware — the boundary the +0.0329 sits on."""
    seen = {"packed": 0, "decoupled": 0}

    m = stub_engine()
    m._packed_ids = lambda prefix, state, opts: (seen.__setitem__("packed", seen["packed"] + 1)
                                                 or (list(range(len(opts))), list(range(len(opts)))))
    m._packed_hidden = lambda seq, positions: torch.zeros(len(positions), 1024)

    def decoupled(prefix, state, opts):
        seen["decoupled"] += 1
        return torch.zeros(len(opts), 1024)

    m._decoupled_hidden = decoupled

    for n in (15, 16):
        m._option_logits("Task", "state", [f"o{i}" for i in range(n)], [f"o{i}" for i in range(n)])
    for n in (17, 77):
        m._option_logits("Task", "state", [f"o{i}" for i in range(n)], [f"o{i}" for i in range(n)])
    assert seen == {"packed": 2, "decoupled": 2}

    m.max_packed_options = 77
    seen.update(packed=0, decoupled=0)
    m._option_logits("Task", "state", [f"o{i}" for i in range(77)], [f"o{i}" for i in range(77)])
    assert seen == {"packed": 1, "decoupled": 0}, "the budget is a config scalar, not a constant"


def test_routing_label_is_printed_per_question():
    """`policy`/`routing` is the seam a caller reads to know which head answered."""
    m = stub_engine()
    m._encode_state = lambda text: [1, 2, 3]
    m._option_logits = lambda prefix, state, options, labels: torch.zeros(len(labels))
    out = m._ask("a state", {
        "small": {"type": "choice", "instructions": "i", "criteria": ["a", "b"]},
        "big": {"type": "choice", "instructions": "i",
                "criteria": [str(i) for i in range(20)]}})
    assert out["routing"]["small"]["mode"] == "packed"
    assert out["routing"]["big"]["mode"] == "set-aware"
    assert out["routing"]["big"]["n_options"] == 20
    assert out["policy"]["b77_none_mode"] == "T_fallback"
    assert out["policy"]["use_coral_score_head"] is False


# ---------------------------------------------------------------- heads


def test_set_aware_head_is_permutation_equivariant():
    """Re-ordering the option set re-orders the logits and changes nothing else.

    `base_scorer` sees one option; `comp_mlp` sees that option's deviation from the bundle
    mean. Both are order-free, and this is the property the head exists to provide, so it is
    tested with random weights rather than asserted in a docstring.
    """
    torch.manual_seed(0)
    head = SetAwareBankingHead(0.0)
    hs = torch.randn(23, 1024)
    ref = head.raw_option_logits(hs)
    perm = torch.randperm(hs.shape[0])
    assert torch.allclose(head.raw_option_logits(hs[perm]), ref[perm], atol=1e-5)


def test_set_aware_head_starts_at_base_only():
    """`comp_scale` inits at 0, so the competition term is off until trained."""
    head = SetAwareBankingHead(0.0)
    hs = torch.randn(9, 1024)
    with torch.no_grad():
        base = head.base_scorer(head.norm(hs)).squeeze(-1)
    assert torch.allclose(head.raw_option_logits(hs), base, atol=1e-6)


def test_set_aware_scale_parameter_is_zero_dimensional():
    """`torch.zeros(1)` would be shape (1,) and `strict=True` would reject the weights."""
    head = SetAwareBankingHead()
    assert head.comp_scale.shape == ()
    fresh = SetAwareBankingHead()
    fresh.load_state_dict(head.state_dict(), strict=True)


def test_gate_features_are_six_floats_and_the_shipped_mode_does_not_use_them():
    """The first `none_gate` Linear takes 6 inputs; the checkpoint's feature vector for
    those 6 slots is in the notebook and not here, so this pins the shape and the mode
    string that says the path is dead."""
    head = SetAwareBankingHead(0.0)
    feats = head.extract_gate_features(torch.randn(12, 1024))
    assert feats.shape == (6,) and torch.isfinite(feats).all()
    assert head.none_gate(feats).shape == (1,)
    assert head.none_of_these_by_threshold(torch.tensor([0.4, 0.3, 0.2]), 0.45) is True
    assert head.none_of_these_by_threshold(torch.tensor([0.5, 0.3, 0.2]), 0.45) is False


def test_coral_cuts_are_monotone_and_underdetermined_by_their_count():
    """`b0` + 3 softplus-cumsum offsets = 3 thresholds; 5 levels need 4.

    The fold is not determined by the artifact, so the head returns the parts and this test
    pins the part that the prose would otherwise round: three cuts, not four.
    """
    head = CoralOrdinalScoreHead(0.0)
    with torch.no_grad():
        head.deltas.fill_(0.5)
        cuts = head.cut_offsets()
    assert cuts.shape == (3,)
    assert bool((cuts.diff() > 0).all()), "a negative delta would un-order the cumulative link"
    out = head(torch.randn(5, 1024))
    assert set(out) == {"pointwise", "utility", "cuts", "mix"}
    assert out["pointwise"].shape == (5,) and out["utility"].shape == (5,)


def test_graph_head_starts_small_and_forward_preserves_shape():
    """The gate inits at sigmoid(-2.0)=0.1192; the shipped one measures -2.3757 (0.0850).

    Either way it is a small mixing coefficient, which is what the 5-arm ablation reported:
    five arms, same accuracy. This asserts the shape and the init, not the ablation.
    """
    g = GraphModule(0.0)
    assert math.isclose(float(g.gate().detach()), 0.11920292202211757, rel_tol=1e-6)
    hq = torch.randn(3, 1024)
    assert g(hq).shape == hq.shape
    assert g.scorer.state_dict().keys() == _readout(1024, 512, 0.0).state_dict().keys()


def test_readout_shapes_are_per_primitive():
    """`choice` commits a label, `score` a positional expectation plus a level, `noul` a
    probability. The conformal key is `conformal_set_90` — README quotes it verbatim."""
    m = stub_engine()
    labels = ["a", "b", "c"]
    logits = torch.tensor([2.0, 0.0, -1.0])
    choice = m._readout({"type": "choice"}, labels, logits, "choice", None)
    assert set(choice) >= {"type", "choice", "confidence", "abstain", "reason", "margin",
                           "conformal_set_90", "probabilities"}
    assert choice["choice"] == "a" and not choice["abstain"]
    assert math.isclose(choice["margin"], choice["probabilities"]["a"]
                        - choice["probabilities"]["b"], rel_tol=1e-9)

    score = m._readout({"type": "score"}, labels, logits, "score", None)
    assert score["level"] == "a" and score["legend"] == {"0": "a", "1": "b", "2": "c"}
    assert 0.0 <= score["score"] <= 2.0

    noul = m._readout({"type": "noul"}, ["no", "yes"], torch.tensor([3.0, -3.0]), "noul", None)
    assert not noul["abstain"]
    assert noul["noul"] == noul["probabilities"]["yes"] and noul["yes"] is False


def test_readout_abstains_on_an_empty_conformal_set_and_on_a_low_floor():
    """Two independent withhold paths, both with the measured reason printed.

    The first is architectural: a 90% conformal set that excludes the argmax licenses no
    answer, so a flat distribution abstains even with no caller-set floor. The second is
    `abstain_below`, which can only withhold — it never un-abstains a set that came back
    empty.
    """
    m = stub_engine()
    labels = ["w", "x", "y", "z"]
    flat = torch.zeros(4)
    out = m._readout({"type": "choice"}, labels, flat, "choice", None)
    assert out["abstain"] is True and out["choice"] is None
    assert "conformal floor" in out["reason"]

    peaked = torch.tensor([2.0, 0.0, 0.0, 0.0])          # top p = 0.755, set = ["w"]
    assert m._readout({"type": "choice"}, labels, peaked, "choice", None)["abstain"] is False
    floored = m._readout({"type": "choice"}, labels, peaked, "choice", 0.8)
    assert floored["abstain"] is True and floored["choice"] is None
    assert "falls under" in floored["reason"]

    # The engine floor is applied in `_ask`, and a per-call floor overrides it — the seam
    # that lets a session with one loaded encoder ask at two thresholds.
    assert m._readout({"type": "choice"}, labels, flat, "choice", 0.0)["abstain"] is True
    m.abstain_below = 0.8
    m._encode_state = lambda text: [1, 2, 3]
    m._option_logits = lambda prefix, state, options, labels: peaked
    assert m._ask("s", {"q": {"type": "choice", "instructions": "i",
                              "criteria": ["w", "x", "y", "z"]}}
                  )["answers"]["q"]["abstain"] is True
    assert m._ask("s", {"q": {"type": "choice", "instructions": "i", "criteria":
                             ["w", "x", "y", "z"]}}, abstain_below=0.0
                  )["answers"]["q"]["abstain"] is False
    assert m._ask("s", {"q": {"type": "choice", "instructions": "i", "criteria":
                             ["w", "x", "y", "z"]}})["policy"]["abstain_below"] == 0.8


def test_options_use_the_suites_own_renderer_for_dict_criteria():
    """A `choice` criteria dict maps label -> description, and most banking77 descriptions
    are null. `f"{label}: {desc}"` would put the literal "None" into 61 of 77 prompts."""
    m = stub_engine()
    opts = m._options({"type": "choice",
                       "criteria": {"pay_bill": None, "online_dispute": "raise a dispute"}})
    assert opts == ["pay bill", "raise a dispute"]
    assert m._labels({"type": "choice", "criteria": {"pay_bill": None}}) == ["pay_bill"]
    assert m._options({"type": "noul", "criteria": {}}) == ["no", "yes"]


# ---------------------------------------------------------------- dispatch


def test_v6_dispatch_is_by_contents_and_never_by_repo_id(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert is_v6_directory(empty) is False
    for name in CKPT_NAMES:
        (tmp_path / name).write_bytes(b"x")
        assert is_v6_directory(tmp_path) is True
        (tmp_path / name).unlink()
    (tmp_path / "model.pt").write_bytes(b"x")
    assert is_v6_directory(tmp_path) is False, "a legacy checkpoint is not a v6"
    (tmp_path / "deberta-v5-finetuned").mkdir()
    assert is_v6_directory(tmp_path) is True

    # A `user/repo` id is not detected: it is indistinguishable from `runs/myna-v0`.
    assert is_v6_source("aashish254/myna-v6") is False
    assert is_v6_source(tmp_path) is True
    assert is_v6_source(tmp_path / "nope") is False


def test_myna_dispatches_a_campaign_directory_to_the_v6_engine(tmp_path):
    """`Myna(<campaign dir>)` must hand the directory to `MynaV6`, not open it as an SSM
    checkpoint. The load itself then fails on the placeholder bytes, and the frame it
    fails in is the proof of which engine answered — no 1.7 GB weights needed.
    """
    from myna import Myna

    def last_myna_frame(err):
        frames = [f for f in traceback.extract_tb(err.__traceback__)
                  if Path(f.filename).parent.name == "myna"]
        return Path(frames[-1].filename).name

    (tmp_path / CKPT_NAMES[0]).write_bytes(b"not a checkpoint")
    with pytest.raises(Exception) as exc:
        Myna(tmp_path)
    assert last_myna_frame(exc.value) == "v6_engine.py", "Myna() opened the campaign as an SSM ckpt"

    # The legacy path is untouched: a `model.pt` directory still builds the SSM engine.
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "model.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(Exception) as exc2:
        Myna(legacy)
    assert last_myna_frame(exc2.value) == "engine.py"


def test_the_upload_command_dry_runs_and_adds_up(tmp_path):
    """`scripts/upload_to_hf.py --dry-run` is the one command the README publishes for moving
    1.8 GB of weights, so its own arithmetic is checked here rather than trusted: the byte
    total must equal the sum of the lines above it, every file it names must exist, and the
    competitor columns must be keys of the committed witness — a renamed arm would otherwise
    reach the Hub as an empty column that reads as a zero score.
    """
    if not CAMPAIGN_DIR.is_dir():
        pytest.skip(f"{CAMPAIGN_DIR} holds the weights the uploader lists; it is not in git")
    import re
    import subprocess

    r = subprocess.run([sys.executable, "scripts/upload_to_hf.py", "--repo-id",
                        "aashish254/myna-v6", "--dry-run"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr[-600:]
    assert "dry run: nothing uploaded." in r.stdout, r.stdout[-400:]
    sizes = [int(m.group(1).replace(",", "")) for m in
             re.finditer(r"^\s+([\d,]+) B\s+\S", r.stdout, re.M)]
    total = int(re.search(r"total\s+: ([\d,]+) B", r.stdout).group(1).replace(",", ""))
    assert sum(sizes) == total, (len(sizes), sum(sizes), total)
    assert len(sizes) == int(re.search(r"files\s+: (\d+)", r.stdout).group(1))

    sys.path.insert(0, str(REPO / "scripts"))
    import upload_to_hf
    card = upload_to_hf.model_card(CAMPAIGN_DIR)
    assert "| — |" not in card, "an arm lost its witness key and printed an empty column"
    assert "| Myna v6 (test) |" in card
    h2h = json.loads((REPO / "runs" / "witness-competitor-head-to-head-1440.json").read_text())
    missing = [arm for arm in upload_to_hf.COMPETITORS if arm not in h2h]
    assert not missing, f"the card would print an empty column for {missing}"


# ---------------------------------------------------------------- the real thing


@pytest.mark.skipif(not CAMPAIGN_DIR.is_dir(),
                    reason=f"{CAMPAIGN_DIR} is the campaign directory; it is not in git (SPEC §9.68 ix)")
def test_the_shipped_checkpoint_loads_and_answers():
    pytest.importorskip("transformers")
    m = MynaV6(CAMPAIGN_DIR, device="cpu")
    assert m.head_params == sum(p.numel() for mod in
                                (m.packed_scorer, m.b77_set_head, m.ordinal_score_head,
                                 m.graph_module) for p in mod.parameters())

    out = m.predict("My order never arrived and I want a refund.", {
        "department": {"type": "choice", "instructions": "Which team?",
                       "criteria": {"billing": None, "technical": None, "shipping": None,
                                    "returns": None, "other": None}},
        "severity": {"type": "score", "instructions": "How severe?",
                     "criteria": ["low", "medium", "high"]},
        "refund": {"type": "noul", "instructions": "Grant a refund?"}})

    assert out["routing"]["department"]["mode"] == "packed"
    for key, spec in out["answers"].items():
        assert set(spec) >= {"type", "confidence", "margin", "conformal_set_90",
                             "abstain", "probabilities"}, key
    assert len(out["answers"]["department"]["probabilities"]) == 5
    assert out["answers"]["refund"]["noul"] == out["answers"]["refund"]["probabilities"]["yes"]
    assert out["usage"]["state_tokens"] > 0
    assert out["latency_ms"] > 0
    assert set(m.temperatures) == {"choice", "score", "noul"}
    assert m.audit("test")["macro_acc"] == pytest.approx(0.8582510184910809)


def test_the_engine_re_encodes_the_state_on_every_ask():
    """v6 is a fine-tuned encoder, not the delta engine: `Observation.append` returns text
    and the next ask re-reads all of it. This is disclosed in README and SPEC §9.68 vii, so
    it is pinned here rather than left to prose that could quietly become wrong."""
    m = stub_engine()
    m._encode_state = lambda text: list(range(len(text)))
    m._option_logits = lambda prefix, state, options, labels: torch.zeros(len(labels))
    obs = m.observe("abcd")
    grown = obs.append("efgh")
    assert grown.text == "abcd\nefgh"
    assert grown.ids == list(range(len("abcd\nefgh"))), "the state is not a retained activation"

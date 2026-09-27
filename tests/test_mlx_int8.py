"""P6 6c: the MLX int8 serving path has to be the same model in fewer bytes.

Three things can go wrong with a quantised weight path, and a test is aimed at
each: the engine silently serving fp32 while reporting int8 bytes; the artifact
on disk declaring the wrong group size or bit depth, so it unpacks differently
than it was packed; and quantisation mutating the fp32 tree beside it, so one
deployment corrupts the other. All of it runs on MLX's CPU stream, so the
figures do not move with the GPU.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import mlx.core as mx  # noqa: E402

from myna.model import MynaConfig, MynaModel  # noqa: E402
from myna.mlx_model import MynaMLX  # noqa: E402

from test_mlx_parity import make_batch  # noqa: E402

# d_model, d_ff and heads*d_v are all multiples of 32 so that the smallest legal
# MLX group size divides every Linear's input dimension.
TINY_Q = dict(vocab=64, d_model=64, n_layers=1, n_heads=2, d_k=32, d_v=32,
              d_ff=96, d_ptr=64)

CKPT = Path("runs/myna-v0/model.pt")
SUITE = Path("data/decision-v2-pilot/calibration.jsonl")


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)
    yield


def _tiny_checkpoint(tmp: Path) -> Path:
    cfg = MynaConfig(**TINY_Q)
    torch.manual_seed(0)
    m = MynaModel(cfg)
    p = tmp / "tiny.pt"
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in m.state_dict().items()},
                "cfg": vars(cfg), "temperature": 1.0}, p)
    return p


def _n_linear(cfg):
    """Every 2-D `.weight` except the token embedding — the set `quantized_` claims to cover."""
    return cfg["n_layers"] * 2 * 3 + cfg["n_layers"] * 2 + 2  # qkv,gate,out_proj per branch; ff.0,ff.2; wq,wk


def test_quantized_engine_actually_holds_quantized_weights():
    """The byte figure is only worth anything if the engine really is int8: a filter
    that matches nothing leaves a working fp32 engine wearing an int8 label."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ck = _tiny_checkpoint(tmp)
        eng = MynaMLX.from_checkpoint(ck)
        before = {k for k, v in eng.p.items() if k.endswith(".weight") and v.ndim == 2}
        n = eng.quantized_(group_size=32, bits=8)
        after_w = {k for k, v in eng.p.items() if k.endswith(".weight") and v.ndim == 2}
        assert n == _n_linear(eng.cfg), f"quantised {n}, expected {_n_linear(eng.cfg)}"
        assert after_w == {"trunk.tok.weight"}, f"left fp32: {sorted(after_w)}"
        assert len(after_w) + n == len(before), "a weight vanished instead of being replaced"
        qd = [v.dtype for k, v in eng.p.items() if k.endswith("weight_q")]
        assert qd and all(t == mx.uint32 for t in qd), f"packed weights are {set(map(str, qd))}"
        assert eng.quant == {"group_size": 32, "bits": 8,
                             "keep": ["trunk.tok.weight"], "n_quantized": n}


def test_quantizing_one_engine_leaves_the_fp32_one_alone():
    """Two engines built from one checkpoint share nothing. Quantising `a` must move
    `a`'s answers (the int8 path is really taken) and must not move `b`'s by one bit
    (no shared tree is being mutated in place)."""
    with tempfile.TemporaryDirectory() as d:
        ck = _tiny_checkpoint(Path(d))
        batch = make_batch(vocab=TINY_Q["vocab"])
        a = MynaMLX.from_checkpoint(ck)
        b = MynaMLX.from_checkpoint(ck)
        ref_a, ref_b = np.array(a.forward(*batch)), np.array(b.forward(*batch))
        a.quantized_(group_size=32, bits=8)
        assert not np.array_equal(np.array(a.forward(*batch)), ref_a), \
            "quantizing changed nothing: the int8 path is dead code"
        np.testing.assert_array_equal(np.array(b.forward(*batch)), ref_b)


def test_artifact_round_trip_answers_bit_identically():
    """The saved file is the deployment. Reloading it through meta.json's declared
    group size and bits has to reproduce the in-memory engine exactly — the check
    that fails if either number is a label rather than a fact."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ck = _tiny_checkpoint(tmp)
        batch = make_batch(vocab=TINY_Q["vocab"])
        eng = MynaMLX.from_checkpoint(ck, quantize=True, group_size=32, bits=8)
        live = np.array(eng.forward(*batch))
        out = eng.save(tmp / "artifact")
        assert (out / "params.safetensors").exists() and (out / "meta.json").exists()
        again = MynaMLX.from_mlx_dir(out)
        np.testing.assert_array_equal(np.array(again.forward(*batch)), live)
        assert again.quant == eng.quant
        # the file's own size, not the sum of tensor nbytes, is what ships
        file_n = (out / "params.safetensors").stat().st_size
        assert file_n < ck.stat().st_size
        meta = json.loads((out / "meta.json").read_text())
        assert meta["param_bytes_file"] == file_n, \
            "the byte figure the README would quote is not the file's size"
        assert meta["param_bytes_tensors"] != meta["param_bytes_file"], \
            "the two byte figures are the same number, so one of them is not measured"


def test_meta_lying_about_bits_is_refused():
    """`from_mlx_dir` trusts meta.json for the unpacking shape, so a declared bit
    depth MLX cannot implement must stop the load rather than serve garbage."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ck = _tiny_checkpoint(tmp)
        out = MynaMLX.from_checkpoint(ck, quantize=True, group_size=32).save(tmp / "artifact")
        meta = json.loads((out / "meta.json").read_text())
        meta["quantization"]["bits"] = 6
        (out / "meta.json").write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="bits=6"):
            MynaMLX.from_mlx_dir(out)


def test_group_size_that_does_not_divide_is_an_error_not_silent_fp32():
    """ff.2 has d_ff=96 inputs; a group size of 64 does not divide it. The engine
    must refuse instead of leaving that one weight fp32 behind an int8 label."""
    with tempfile.TemporaryDirectory() as d:
        ck = _tiny_checkpoint(Path(d))
        eng = MynaMLX.from_checkpoint(ck)
        with pytest.raises(ValueError, match="not a multiple of group_size 64"):
            eng.quantized_(group_size=64, bits=8)


def test_keep_substrings_leave_the_named_weights_in_fp32():
    with tempfile.TemporaryDirectory() as d:
        ck = _tiny_checkpoint(Path(d))
        all_in = MynaMLX.from_checkpoint(ck, quantize=True, group_size=32)
        kept = MynaMLX.from_checkpoint(ck, quantize=True, group_size=32,
                                       keep=["trunk.tok.weight", ".gate.weight"])
        assert kept.quant["n_quantized"] < all_in.quant["n_quantized"]
        assert "trunk.layers.0.fwd.gate.weight" in kept.p
        assert "trunk.layers.0.fwd.gate.weight_q" not in kept.p
        assert "trunk.layers.0.fwd.gate.weight" not in all_in.p
        assert "trunk.layers.0.fwd.gate.weight_q" in all_in.p


def test_int8_logits_stay_close_to_fp32_on_random_weights():
    """A quantised trunk still answers the same way: the option probabilities move,
    but by far less than the spread between the leading and the runner-up option —
    i.e. the error stays smaller than the decision it sits on. The 2e-2 bound is the
    measurement with an order of magnitude of room, so an artifact unpacked with the
    wrong group size, or a weight routed through the wrong op, fails it."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ck = _tiny_checkpoint(tmp)
        batch = make_batch(vocab=TINY_Q["vocab"])
        fp32 = np.array(MynaMLX.from_checkpoint(ck).forward(*batch))
        int8 = np.array(MynaMLX.from_checkpoint(ck, quantize=True, group_size=32).forward(*batch))
        valid = batch[5]
        p32, p8 = _softmax(fp32, valid), _softmax(int8, valid)
        spread = p32.max(-1) - np.sort(p32, axis=-1)[..., -2]
        d_prob = np.abs(p8 - p32)[:, valid]
        assert d_prob.max() < 2e-2, f"max |dp| {d_prob.max():.2e}"
        assert d_prob.max() < float(spread.min()), \
            f"error {d_prob.max():.2e} exceeds the narrowest top-two gap {spread.min():.2e}"
        assert (p8[:, valid].argmax(-1) == p32[:, valid].argmax(-1)).all()


def _softmax(logits, valid):
    x = np.where(valid, logits, -1e9).astype(np.float64)
    x -= x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


@pytest.mark.skipif(not (CKPT.exists() and SUITE.exists()),
                    reason="needs the trained checkpoint and the pilot suite on disk")
def test_real_checkpoint_int8_drift_is_measured_and_bounded():
    """The shipped claim, on the real model: int8 answers the suite's own questions
    within a bounded probability error of the fp32 engine, on the checkpoint the
    byte figure belongs to."""
    from myna.engine import Myna
    from myna.real_data import load_split
    from myna.tokenizer import Tokenizer, batch_question_tensors, encode_text

    myna = Myna(str(CKPT.parent), device="cpu")
    tok = myna.tok
    groups = load_split(SUITE)
    keys = sorted(groups)
    picks = keys[:: max(1, len(keys) // 8)][:8]
    fp32 = MynaMLX.from_checkpoint(CKPT)
    int8 = MynaMLX.from_checkpoint(CKPT, quantize=True)
    worst = 0.0
    n_q = 0
    for key in picks:
        qs, exs = groups[key]
        row_q = [(q.instruction, myna._options({"type": q.type, "criteria": q.options}))
                 for q in qs]
        qt = batch_question_tensors(tok, [row_q])
        valid = qt["opt_valid"][0].numpy().astype(bool)
        ids = encode_text(tok, exs[0].state)
        args = (np.array([ids], np.int64), np.array([len(ids)], np.int64),
                qt["q_ids"][0].numpy(), qt["q_mask"][0].numpy(), qt["span_mat"][0].numpy(),
                qt["opt_valid"][0].numpy(), qt["decide_idx"][0].numpy())
        p32 = _softmax(np.array(fp32.forward(*args))[0], valid)
        p8 = _softmax(np.array(int8.forward(*args))[0], valid)
        worst = max(worst, float(np.abs(p8 - p32)[valid].max()))
        n_q += len(qs)
    assert n_q >= 8
    # 2e-2 is the bound this run measured against; the drift the int8 path costs is
    # an order of magnitude inside it, and the number published is the measurement,
    # not the bound.
    assert worst < 2e-2, f"max |dp| int8 vs fp32 = {worst:.2e} over {n_q} questions"

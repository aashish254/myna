"""The ONNX parity gate G3 rests on, measured rather than assumed (SPEC §5 P6 6a).

`myna.onnx_export` writes two graphs and compares them to torch. The tests here are
about the *comparison*, because a parity harness is the kind of code that can report
1e-9 while checking nothing: an ignored mask, a padded row counted as real, an error
averaged over zero questions, or a bound applied to a quantity whose scale is 500 all
produce a confident PASS.

Two facts about the export shape the assertions:

* The state's entries reach ~4e2, so its error is gated *relatively*. The absolute
  number is still reported — this suite asserts both directions of that choice, so
  neither half of the gate can be dropped in favour of the other.
* Both graphs are exported at fixed widths (tokens per chunk, questions per request),
  and a request that does not fit must fail loudly. `test_a_question_too_wide_for_the
  _graph_fails_instead_of_passing_on_nothing` is the one that keeps a vacuous 0.0 error
  from being read as perfect parity.

Requires the optional `browser` extra (`uv sync --extra browser`); it skips with that
name said out loud rather than passing quietly.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import random
from pathlib import Path

import pytest
import numpy as np
import torch

from myna.data import generate
from myna.engine import Myna
from myna.model import MynaConfig, MynaModel
from myna.onnx_export import (WEIGHTS, chain_state, export, parity, pointer_probs,
                             read_head, state_shapes)
from myna.tokenizer import build_question, train_tokenizer

HAS_ORT = importlib.util.find_spec("onnxruntime") is not None

pytestmark = pytest.mark.skipif(not HAS_ORT, reason="needs the optional `browser` extra")


@pytest.fixture(scope="session", autouse=True)
def _order_the_teardown():
    """Close the ONNX sessions before the interpreter starts dismantling itself.

    Without this, a run of this module aborts in `recursive_mutex lock failed` *after*
    every assertion has passed — onnxruntime's static environment and torch's dynamo
    export state are torn down in the wrong order, and a green suite gets exit 134. The
    tests still pass; the process does not survive to say so, which a mutation battery
    reads as a broken baseline.
    """
    yield
    import gc

    import myna.onnx_export as ox
    ox._SESSIONS.clear()
    gc.collect()


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    """A 3-layer checkpoint: same graph shapes, seconds to export."""
    exs = generate(48, "support", random.Random(2))
    tok = train_tokenizer([e.state for e in exs], vocab_size=600)
    cfg = MynaConfig(vocab=tok.get_vocab_size(), d_model=64, n_layers=3, n_heads=4,
                     d_k=16, d_v=16, d_ff=128, d_ptr=32)
    torch.manual_seed(1)
    d = tmp_path_factory.mktemp("onnx-ckpt")
    torch.save({"state_dict": MynaModel(cfg).state_dict(), "cfg": vars(cfg),
                "temperature": 1.0}, d / "model.pt")
    tok.save(str(d / "tokenizer.json"))
    return d


@pytest.fixture(scope="module")
def bundle(small, tmp_path_factory):
    out = tmp_path_factory.mktemp("onnx")
    meta = export(small, out, chunk=32, n_questions=4, q_len=96)
    return small, out, meta


ITEMSIZE = {1: 4, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 8}   # onnx TensorProto dtypes


def _inits(path, min_bytes=0):
    """name -> (bytes, external?) for every initializer a graph declares.

    Read from the graph with `load_external_data=False`, so the sizes are what the
    artifact says about itself, and the sharing contract is checked where it is
    declared: an initializer at or above `min_bytes` must be external, because an inline
    trunk weight is a copy the browser downloads twice; an external one must point into
    `weights.bin` and carry no bytes of its own.
    """
    import onnx

    g = onnx.load(path, load_external_data=False)
    out = {}
    for i in g.graph.initializer:
        nbytes = int(np.prod(i.dims)) * ITEMSIZE.get(i.data_type, 0)
        external = i.data_location == onnx.TensorProto.EXTERNAL
        if external:
            ed = {e.key: e.value for e in i.external_data}
            assert ed.get("location") == "weights.bin", (path.name, i.name, ed)
            assert not i.HasField("raw_data") and not len(i.float_data), i.name
        else:
            assert i.HasField("raw_data"), (path.name, i.name)
            assert nbytes < min_bytes, (
                f"{path.name}:{i.name} keeps {nbytes} bytes inline, above the "
                f"{min_bytes}-byte sharing threshold — the trunk is duplicated again")
        out[i.name] = (nbytes, external)
    return out


def _extents(path):
    """(offset, length) per external initializer name, as the graph declares them."""
    import onnx

    g = onnx.load(path, load_external_data=False)
    extents = {}
    for i in g.graph.initializer:
        if i.data_location != onnx.TensorProto.EXTERNAL:
            continue
        ed = {e.key: e.value for e in i.external_data}
        assert ed.get("location") == "weights.bin", (path.name, i.name)
        assert not i.HasField("raw_data") and not len(i.float_data), i.name
        extents[i.name] = (int(ed["offset"]), int(ed["length"]))
    return extents


def _content(t):
    """SHA-256 of a parameter's bytes, in either layout it could have been written in.

    `dynamo` exports `F.linear` with the weight transposed relative to the state dict,
    so matching only the training layout reports every weight as missing.
    """
    a = np.ascontiguousarray(t.detach().cpu().numpy())
    return {hashlib.sha256(a.tobytes()).hexdigest(),
            hashlib.sha256(np.ascontiguousarray(a.T).tobytes()).hexdigest()}


def _hashes(blob, extents):
    """Content per name, read back out of the one file both graphs point into."""
    return {n: hashlib.sha256(blob[o:o + ln]).hexdigest() for n, (o, ln) in extents.items()}


def test_both_graphs_load_and_report_their_weights(bundle, small):
    """The bounds the byte budget rests on, read off the artifact rather than assumed.

    The first two passes of this test used a size heuristic and still let a byte-budget
    lie through (§9.26), and the shape of the lie kept moving: a graph skeleton is a few
    hundred KiB whatever it carries, a file *list* can leave the weights out, and the
    weight names dynamo writes (`val_N`) are per-graph counters, so no name-based check
    can say whether a weight shipped. So here: every trunk weight at or above the
    sharing threshold must be physically present in `weights.bin` — byte-identical, in
    either layout — the extents both graphs declare must sit inside that file, and the
    reported transfer must cover the furthest byte either graph reaches plus the head
    and the tokenizer, which no listing of the directory can quietly omit.
    """
    _, out, meta = bundle
    mb = meta["weight_sharing"]["min_bytes"]
    step, question = _inits(out / "state_step.onnx", mb), _inits(out / "question.onnx", mb)
    assert step and question
    blob = (out / "weights.bin").read_bytes()
    extents = _extents(out / "state_step.onnx") | _extents(out / "question.onnx")
    reach = max(o + ln for o, ln in extents.values())
    assert reach <= len(blob), (reach, len(blob))
    align = meta["weight_sharing"]["align"]
    assert all(o % align == 0 for o, _ in extents.values()), "a region the page reads misaligned"
    present = {h for h in _hashes(blob, extents).values()}
    trunk = {n: p for n, p in Myna(small).trunk.state_dict().items()
             if p.numel() * p.element_size() >= mb}
    missing = sorted(n for n, p in trunk.items() if not _content(p) & present)
    assert not missing, f"weights absent from {WEIGHTS}: {missing}"
    # The accounting in meta has to describe this file, not a number the writer liked:
    # `per_graph_bytes` is a sum over *references*, so it exceeds the file exactly where
    # a graph aliases a region twice (the zero-initialised gates) and nowhere else.
    sh = meta["weight_sharing"]
    assert sh["weights_bytes"] == len(blob)
    for g, inits in (("state_step.onnx", step), ("question.onnx", question)):
        declared = sum(b for b, ext in inits.values() if b >= mb and ext)
        assert sh["per_graph_bytes"][g] == declared, (g, sh["per_graph_bytes"][g], declared)
    assert sh["per_reference_bytes"] == sum(sh["per_graph_bytes"].values())
    assert meta["transfer_mib"] * 2**20 >= (reach + meta["head"]["head_bytes"]
                                            + meta["head"]["tokenizer_bytes"])
    assert meta["state_bytes_total"] == 4 * 16 * 16 * 4 * 3
    assert json.loads((out / "meta.json").read_text())["exporter"] == "dynamo"


def test_the_two_graphs_reference_one_copy_of_the_trunk(bundle):
    """6a shipped 93.7 MiB fp32 for a 14.45M-parameter model: the same trunk, twice.

    Sharing has to be structural, or the byte claim is a sentence — and it has to be
    checked by *content*, because dynamo names each graph's initializers `val_N` from a
    per-graph counter. Measured on this artifact the two graphs share every region the
    state graph reads, while their name sets overlap by two, so a name intersection
    reports the counter rather than the sharing. So: the state graph's weight regions are
    a subset of the question graph's (by bytes, not labels), the question graph has
    regions the state graph lacks (the backward scan and the final norm), nothing
    survives as a per-graph `*.onnx.data`, and the file on disk is smaller than both
    graphs' claims.
    """
    _, out, meta = bundle
    mb = meta["weight_sharing"]["min_bytes"]
    sh = meta["weight_sharing"]
    blob = (out / "weights.bin").read_bytes()
    step_ext = _extents(out / "state_step.onnx")
    ques_ext = _extents(out / "question.onnx")
    step = _hashes(blob, step_ext)
    question = _hashes(blob, ques_ext)
    # the state graph is a strict subset of the question graph's content
    st, qu = set(step.values()), set(question.values())
    assert st and qu
    assert st - qu == set(), f"weights only state_step reads: {sorted(st - qu)}"
    assert qu - st, ("the question graph carries no weight the state graph lacks — "
                     "the backward scan is missing from the artifact")
    assert not list(out.glob("*.onnx.data")), "a per-graph copy survived the rewrite"
    assert meta["shared_weights"] is True
    assert sh["shared_regions"] == len(st), (sh, len(st))
    # deduplicated by region, not by name: several of state_step's initializers alias one
    # extent (every zero-initialised gate), and that is a fraction of the file's bytes.
    assert sh["shared_bytes"] == sum(ln for _, ln in set(step_ext.values())), sh
    twice = sum(b for b, ext in _inits(out / "state_step.onnx", mb).values()
                if b >= mb and ext) + sum(
        b for b, ext in _inits(out / "question.onnx", mb).values() if b >= mb and ext)
    assert len(blob) < twice, "one file holding both graphs' bytes counted twice over"
    assert meta["transfer_mib"] < meta["unshared_transfer_mib"], meta


def test_no_share_reproduces_the_duplicated_baseline(bundle, tmp_path, small):
    """The flag that measures what sharing bought has to be honest about its own size."""
    _, _, shared_meta = bundle
    out = export(small, tmp_path / "dup", chunk=32, n_questions=4, q_len=96, share=False)
    assert out["shared_weights"] is False
    assert out["transfer_mib"] > shared_meta["transfer_mib"], (
        out["transfer_mib"], shared_meta["transfer_mib"])
    assert sorted(f.name for f in (tmp_path / "dup").glob("*.onnx.data")) == \
        ["question.onnx.data", "state_step.onnx.data"]


def test_the_head_blob_answers_the_same_question(bundle, small):
    """The page does the pointer head in JavaScript, so the shipped blob has to be
    enough to reproduce the engine's probabilities without torch's modules."""
    _, out, meta = bundle
    fields, meta = read_head(out)
    for name, lay in meta["head"]["head_layout"].items():
        want = [meta["d_ptr"], meta["d_model"]] if "weight" in name else [meta["d_ptr"]]
        assert lay["shape"] == want, (name, lay)
    assert meta["head"]["head_bytes"] == 2 * (meta["d_ptr"] * meta["d_model"] + meta["d_ptr"]) * 4

    m = Myna(small)
    ex = generate(1, "support", random.Random(0))[0]
    spec = {"type": "choice", "instructions": "x",
            "criteria": {"billing": "b", "tech": "t", "sales": "s"}}
    opts = m._options(spec)
    qrow, spans, dec = build_question(m.tok, "Which team?", opts)
    obs = m.observe(ex.state)
    with torch.no_grad():
        hq = m.trunk.encode_questions([torch.tensor([qrow])], obs.S_cache,
                                      len(obs.ids))[0][0]
    want = m._readout(spec, opts, hq, dec, spans)["probabilities"]
    got = pointer_probs(fields, m.cfg.d_ptr, m.temperature, hq.numpy(), spans, dec)
    assert max(abs(w - g) for w, g in zip(list(want.values()), got)) < 1e-6, (want, got)


def test_a_smaller_scan_tile_is_the_same_answer_at_less_memory(tmp_path, small):
    """`--scan-chunk` is the tab's memory knob, not an accuracy knob.

    The parallel scan allocates a `[rows, heads, chunk, chunk, d_k]` tile per layer, so
    the exported width is what a request costs: at the engine's inference chunk of 256
    and 8 questions the tile is hundreds of MB, which is a crashed tab rather than a
    slow one. The arithmetic is identical at any chunk size
    (`tests/test_trunk_numerics.py`), so what this test holds is that the *exported*
    graph still passes parity at the small tile, and that the tile the metadata reports
    is the one that was asked for.
    """
    wide = export(small, tmp_path / "wide", chunk=32, n_questions=4, q_len=96, scan_chunk=32)
    tight = export(small, tmp_path / "tight", chunk=32, n_questions=4, q_len=96, scan_chunk=4)
    assert wide["peak_tile_bytes"] > tight["peak_tile_bytes"], (wide, tight)
    # tiles scale with chunk^2: 32 -> 4 is eight steps down, sixty-four times smaller.
    # Exact, not approximate -- the key is reported in bytes for this reason.
    assert wide["peak_tile_bytes"] == tight["peak_tile_bytes"] * 64, (wide, tight)
    # and it is the tile that was actually asked for, recomputed from the geometry in
    # the same file rather than trusted from the writer.
    for m, s in ((wide, 32), (tight, 4)):
        assert m["peak_tile_bytes"] == max(
            r * m["n_heads"] * s * s * m["d_k"] * 4 for r in (1, m["n_questions"])), m
    assert wide["scan_chunk"] == 32 and tight["scan_chunk"] == 4
    for d, m in ((tmp_path / "wide", wide), (tmp_path / "tight", tight)):
        res = parity(small, d, n_chunks=2)
        assert res["pass"] is True, (d, res)
        assert res["decision_max_abs_error"] < 1e-4


def test_the_scan_tile_must_fit_inside_the_graph_width(tmp_path, small):
    """A tile wider than the input is a shape the graph was never traced at."""
    with pytest.raises(SystemExit, match="scan-chunk"):
        export(small, tmp_path / "bad", chunk=32, n_questions=4, q_len=96, scan_chunk=64)


def test_the_tokenizer_travels_with_the_artifact(bundle, small):
    """A page cannot answer what it cannot tokenize, and the byte budget must say so."""
    _, out, meta = bundle
    assert (out / "tokenizer.json").read_bytes() == (small / "tokenizer.json").read_bytes()
    assert meta["files_mib"]["tokenizer.json"] > 0
    assert meta["head"]["tokenizer_bytes"] == (out / "tokenizer.json").stat().st_size


def test_parity_passes_within_the_bound(bundle):
    small, out, _ = bundle
    res = parity(small, out, max_abs_error=1e-4, max_rel_error=1e-4, n_chunks=3)
    assert res["pass"] is True, res
    # the relative figure has to be a ratio, not the absolute error wearing its name:
    # a state of scale ~2e2 puts the two a factor of 1e4 apart.
    assert res["chain_max_rel_error"] < 1e-4, res
    assert res["chain_max_abs_error"] > res["chain_max_rel_error"], res
    assert res["decision_questions_compared"] >= 1, res


def test_the_decision_number_is_a_probability_not_a_hidden_state(bundle):
    small, out, _ = bundle
    res = parity(small, out, n_chunks=2)
    # probabilities are bounded by 1, so the absolute figure is meaningful here; if
    # the check were comparing unit-scale hidden states instead, this bound would be
    # the same number by coincidence and mean nothing.
    assert 0.0 <= res["decision_max_abs_error"] <= 1.0
    assert res["decision_max_abs_error"] < 1e-4


def test_masked_padding_leaves_the_real_row_exactly_alone(bundle):
    small, out, _ = bundle
    res = parity(small, out, n_chunks=2)
    assert res["isolation_max_abs_error"] == 0.0, res


def test_a_question_too_wide_for_the_graph_fails_instead_of_passing_on_nothing(tmp_path,
                                                                              small):
    """The vacuous-pass shape: zero questions compared, so the error is 0.0 and 0.0
    beats every bound. `--q-len` is a contract, and breaking it has to be an error."""
    out = export(small, tmp_path / "narrow", chunk=32, n_questions=4, q_len=8)
    assert out["q_len"] == 8
    with pytest.raises(AssertionError, match="compared no questions"):
        parity(small, tmp_path / "narrow", n_chunks=2)


def test_each_half_of_the_gate_can_be_the_one_that_fails(tmp_path, small):
    """Two bounds, two failure modes. If only one were enforced, tightening the other
    would look like a passing run."""
    out = tmp_path / "wide"
    export(small, out, chunk=32, n_questions=4, q_len=96)
    tight_abs = parity(small, out, max_abs_error=1e-12, max_rel_error=1.0, n_chunks=2)
    assert tight_abs["pass"] is False, tight_abs
    tight_rel = parity(small, out, max_abs_error=1.0, max_rel_error=1e-12, n_chunks=2)
    assert tight_rel["pass"] is False, tight_rel
    assert tight_abs["chain_max_rel_error"] > 0, "the relative half has nothing to measure"


def test_the_chain_consumes_the_document_once_and_no_more(tmp_path, small):
    """`chain_state` is the browser's loop, so its accounting is the length claim:
    padded to the chunk width, advanced by the real token count each call."""
    out = tmp_path / "chained"
    export(small, out, chunk=32, n_questions=4, q_len=64)
    from myna.onnx_export import _session
    step = _session(out / "state_step.onnx")
    meta = json.loads((out / "meta.json").read_text())
    L, shapes = meta["n_layers"], state_shapes(Myna(small).cfg)
    ids = list(range(1, 30))                      # 29 tokens: one chunk and a stub tail
    carried, fed = chain_state(step, ids, 32, L, shapes)
    assert fed == len(ids), (fed, len(ids))
    assert len(carried) == L and all(c.shape[0] == 1 for c in carried)


def test_the_engine_and_the_export_share_the_state_geometry(small, bundle):
    """576 KiB is a claim about the shipped state, and the exported graph is what a
    browser would run: if the two disagreed, the memory row in §2.1 would describe a
    different artifact than the one that ships."""
    m = Myna(small)
    _, _, meta = bundle
    per_layer = meta["n_heads"] * meta["d_k"] * meta["d_v"] * 4
    obs = m.observe(generate(1, "support", random.Random(0))[0].state)
    assert per_layer * meta["n_layers"] == meta["state_bytes_total"]
    assert sum(t.numel() * t.element_size() for t in obs.S_cache) == meta["state_bytes_total"]

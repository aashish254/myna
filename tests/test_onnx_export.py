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
import json
import random
from pathlib import Path

import pytest
import numpy as np
import torch

from myna.data import generate
from myna.engine import Myna
from myna.model import MynaConfig, MynaModel
from myna.onnx_export import chain_state, export, parity, state_shapes
from myna.tokenizer import train_tokenizer

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


def test_both_graphs_load_and_report_their_weights(bundle):
    _, out, meta = bundle
    for graph in ("state_step.onnx", "question.onnx"):
        assert (out / graph).exists(), graph
        # The exporter keeps initializers in a sibling `*.onnx.data`, so a listing of
        # the graphs alone would call a 36 MB model a few hundred KiB — and a loose
        # threshold would not notice, which is exactly how the first pass of
        # `bench/mutation_onnx.py` scored 11/12. The bound has to be the size the
        # weights *are*: an artifact cannot be smaller than what it carries.
        # The property is "the reported size covers the weights the graph references",
        # so read the graph and add up what it says it needs. A size heuristic cannot
        # catch this — a graph skeleton is a few hundred KiB either way, which is why
        # the first two passes of this test still let the mutation through.
        import onnx
        g = onnx.load(out / graph, load_external_data=False)
        # mixed dtypes: the weights are float32 (1) and the graph constants are int64
        # (7), so the itemsize comes from the tensor, not from an assumption.
        itemsize = {1: 4, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 8}
        init_bytes = sum(int(np.prod(i.dims)) * itemsize[i.data_type]
                         for i in g.graph.initializer if i.data_type in itemsize)
        assert meta["files_mib"][graph] * 2**20 >= init_bytes * 0.99, (
            graph, meta["files_mib"], init_bytes)
    assert meta["state_bytes_total"] == 4 * 16 * 16 * 4 * 3
    assert json.loads((out / "meta.json").read_text())["exporter"] == "dynamo"


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

"""Engine (recurrent + cached streaming) must produce the same answers as the
batched training path (chunked) on the same weights."""

import random

import pytest
import torch
import torch.nn.functional as F

from myna.data import WORKFLOWS, generate
from myna.engine import Myna
from myna.model import MynaConfig, MynaModel
from myna.tokenizer import question_tensors, train_tokenizer, encode_text


@pytest.fixture(scope="module")
def myna(tmp_path_factory):
    rng = random.Random(3)
    exs = generate(64, "support", rng)
    tok = train_tokenizer([e.state for e in exs], vocab_size=600)
    cfg = MynaConfig(vocab=tok.get_vocab_size(), d_model=64, n_layers=2, n_heads=4, d_k=16, d_v=16, d_ff=128, d_ptr=32)
    torch.manual_seed(0)
    model = MynaModel(cfg)
    d = tmp_path_factory.mktemp("ckpt")
    torch.save({"state_dict": model.state_dict(), "cfg": vars(cfg), "temperature": 1.0}, d / "model.pt")
    tok.save(str(d / "tokenizer.json"))
    return Myna(d)


def test_engine_matches_batch_path(myna):
    rng = random.Random(4)
    exs = generate(6, "support", rng)
    questions = WORKFLOWS["support"][0]
    # mirror the engine's exact option texts so spans line up token-for-token
    qt_noul = question_tensors(myna.tok, [(q.instruction, (["No", "Yes"] if q.type == "noul" else q.options)) for q in questions])

    myna.model.eval()
    for e in exs:
        obs = myna.observe(e.state)
        ids = encode_text(myna.tok, e.state)
        state = torch.tensor([ids])
        lens = torch.tensor([len(ids)])
        logits = myna.model(state, lens, qt_noul["q_ids"], qt_noul["q_mask"], qt_noul["span_mat"],
                            qt_noul["opt_valid"], qt_noul["decide_idx"])
        probs = F.softmax(logits, -1)[0]
        ans = obs.ask({
            q.name: {"type": q.type, "instructions": q.instruction, "criteria": q.options}
            for q in questions
        })["answers"]
        p_choice = [ans["department"]["probabilities"][l] for l in questions[0].options]
        assert max(abs(a - b) for a, b in zip(p_choice, probs[0])) < 2e-3
        p_score = [ans["urgency"]["probabilities"][l] for l in questions[1].options]
        assert max(abs(a - b) for a, b in zip(p_score, probs[1])) < 2e-3


def test_append_is_exact_through_heads(myna):
    """observe(a)+append(b) must answer identically to observe(a+b)."""
    rng = random.Random(5)
    e = generate(4, "support", rng)[0]
    sentences = e.state.split(". ")
    half = len(sentences) // 2
    a = ". ".join(sentences[:half])
    b = ". " + ". ".join(sentences[half:])
    questions = {
        q.name: {"type": q.type, "instructions": q.instruction, "criteria": q.options}
        for q in WORKFLOWS["support"][0]
    }
    full = myna.observe(a + b).ask(questions)["answers"]
    streamed = myna.observe(a).append(b).ask(questions)["answers"]
    for name in full:
        for key in ("probabilities", "noul"):
            if key in full[name]:
                if isinstance(full[name][key], dict):
                    assert max(abs(full[name][key][l] - streamed[name][key][l]) for l in full[name][key]) < 2e-3
                else:
                    assert abs(full[name][key] - streamed[name][key]) < 2e-3


def test_save_restore_state_round_trip(myna, tmp_path):
    """A snapshot must answer exactly like the live observation it came from,
    and stay fixed-size no matter how long the observation grew."""
    rng = random.Random(6)
    e = generate(3, "support", rng)[0]
    questions = {
        q.name: {"type": q.type, "instructions": q.instruction, "criteria": q.options}
        for q in WORKFLOWS["support"][0]
    }
    obs = myna.observe(e.state)
    p = tmp_path / "obs.pt"
    obs.save_state(p)
    assert p.stat().st_size / 1024 < 64  # small model: sub-64KB state
    restored = myna.restore(p)
    assert len(restored.ids) == len(obs.ids)
    a = obs.ask(questions)["answers"]
    b = restored.ask(questions)["answers"]
    for name in a:
        if "probabilities" in a[name]:
            d = max(abs(a[name]["probabilities"][k] - b[name]["probabilities"][k]) for k in a[name]["probabilities"])
            assert d < 1e-6, (name, d)
        else:
            assert abs(a[name]["noul"] - b[name]["noul"]) < 1e-6
    # a 4x longer observation may only grow the snapshot by its token list —
    # never by a per-layer hidden-state stack (the fixed-size claim, E2E)
    long_obs = myna.observe(" ".join([e.state] * 4))
    pl = tmp_path / "obs_long.pt"
    long_obs.save_state(pl)
    growth_kb = (pl.stat().st_size - p.stat().st_size) / 1024
    hidden_stack_growth_kb = (3 * 4 * len(long_obs.ids) * 64 * 4) / 1024  # what an h-cache would add
    assert growth_kb < hidden_stack_growth_kb / 2, (growth_kb, hidden_stack_growth_kb)

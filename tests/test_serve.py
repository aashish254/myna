"""Session API round trip: predict, open/append/ask/close, and the streaming
guarantee that a session asks the same questions as a fresh observe."""

import random

import pytest
import torch
from fastapi.testclient import TestClient

from myna.data import WORKFLOWS, generate
from myna.engine import Myna
from myna.model import MynaConfig, MynaModel
from myna.serve import create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    rng = random.Random(3)
    exs = generate(64, "support", rng)
    from myna.tokenizer import train_tokenizer

    tok = train_tokenizer([e.state for e in exs], vocab_size=600)
    cfg = MynaConfig(vocab=tok.get_vocab_size(), d_model=64, n_layers=2, n_heads=4, d_k=16, d_v=16, d_ff=128, d_ptr=32)
    torch.manual_seed(0)
    model = MynaModel(cfg)
    d = tmp_path_factory.mktemp("ckpt")
    torch.save({"state_dict": model.state_dict(), "cfg": vars(cfg), "temperature": 1.0}, d / "model.pt")
    tok.save(str(d / "tokenizer.json"))
    return TestClient(create_app(Myna(d)))


QUESTIONS = {
    q.name: {"type": q.type, "instructions": q.instruction, "criteria": q.options}
    for q in WORKFLOWS["support"][0]
}


def test_predict_endpoint(client):
    rng = random.Random(9)
    e = generate(1, "support", rng)[0]
    r = client.post("/v1/predict", json={"state": e.state, "questions": QUESTIONS})
    assert r.status_code == 200
    assert set(r.json()["answers"]) == set(QUESTIONS)


def test_session_stream_matches_direct(client):
    rng = random.Random(11)
    e = generate(1, "support", rng)[0]
    sentences = e.state.split(". ")
    half = len(sentences) // 2
    sid = client.post("/v1/sessions", json={"state": ". ".join(sentences[:half])}).json()["session_id"]
    appended = client.post(f"/v1/sessions/{sid}/append", json={"text": ". ".join(sentences[half:])})
    assert appended.status_code == 200
    asked = client.post(f"/v1/sessions/{sid}/ask", json={"questions": QUESTIONS}).json()
    direct = client.post("/v1/predict", json={"state": e.state, "questions": QUESTIONS}).json()
    for name in QUESTIONS:
        a, b = asked["answers"][name], direct["answers"][name]
        if "probabilities" in a:
            worst = max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"])
            assert worst < 1e-2, (name, worst)
        elif "noul" in a:
            assert abs(a["noul"] - b["noul"]) < 1e-2, name
    assert client.delete(f"/v1/sessions/{sid}").status_code == 200
    assert client.post(f"/v1/sessions/{sid}/ask", json={"questions": QUESTIONS}).status_code == 404

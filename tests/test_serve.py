"""Session API round trip: predict, open/append/ask/close, and the streaming
guarantee that a session asks the same questions as a fresh observe.

The abstention tests at the bottom are here rather than only in
`tests/test_abstain.py` because the HTTP surface is where a fast path could go
silent on the way out: the engine can refuse a decision correctly and the server
can still drop the `abstain` flag, the `reason`, or the policy echo while returning
a 200."""

import random
import sys
import types

import pytest
import torch
from fastapi.testclient import TestClient

from myna.data import WORKFLOWS, generate
from myna.engine import Myna
from myna.model import MynaConfig, MynaModel
from myna.serve import create_app, main


def _build_ckpt(maker):
    rng = random.Random(3)
    exs = generate(64, "support", rng)
    from myna.tokenizer import train_tokenizer

    tok = train_tokenizer([e.state for e in exs], vocab_size=600)
    cfg = MynaConfig(vocab=tok.get_vocab_size(), d_model=64, n_layers=2, n_heads=4, d_k=16, d_v=16, d_ff=128, d_ptr=32)
    torch.manual_seed(0)
    model = MynaModel(cfg)
    d = maker.mktemp("ckpt")
    torch.save({"state_dict": model.state_dict(), "cfg": vars(cfg), "temperature": 1.0}, d / "model.pt")
    tok.save(str(d / "tokenizer.json"))
    return d


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    return _build_ckpt(tmp_path_factory)


@pytest.fixture(scope="module")
def client(ckpt):
    return TestClient(create_app(Myna(ckpt)))


@pytest.fixture(scope="module")
def refusing_client(ckpt):
    """The same weights behind a floor high enough to refuse everything, so the
    assertions below are about what the transport carries, not about the model."""
    return TestClient(create_app(Myna(ckpt, abstain_below=0.99)))


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


def test_health_names_the_floor_in_force(client, refusing_client):
    """A deployment that abstains and one that never will must not answer to the
    same health check."""
    assert client.get("/v1/health").json()["abstain_below"] is None
    assert refusing_client.get("/v1/health").json()["abstain_below"] == 0.99


def test_the_transport_does_not_drop_the_refusal(client, refusing_client):
    """JSON is where an abstention can go missing: `choice: None` is valid JSON and
    a client that only reads `choice` cannot tell it from a bug, so the flag and
    the reason have to cross the wire too."""
    rng = random.Random(9)
    state = generate(1, "support", rng)[0].state
    body = refusing_client.post("/v1/predict",
                                json={"state": state, "questions": QUESTIONS}).json()
    assert body["policy"]["abstain_below"] == 0.99
    assert sorted(body["policy"]["abstained"]) == sorted(QUESTIONS)
    for name, a in body["answers"].items():
        assert a["abstain"] is True, name
        withheld = {"choice": "choice", "score": "level", "noul": "yes"}[a["type"]]
        assert a[withheld] is None, (name, a)
        assert a["reason"], name
    # the same weights, no floor: nothing is refused and nothing is withheld
    clean = client.post("/v1/predict", json={"state": state, "questions": QUESTIONS}).json()
    assert clean["policy"]["abstained"] == []
    assert all(a["abstain"] is False for a in clean["answers"].values())


def test_the_cli_floor_reaches_the_engine_that_serves(monkeypatch, ckpt):
    """`--abstain-below` that stops in argparse is a flag in the help text and a
    fast path in production.

    uvicorn is an optional extra and is not installed here, so the import inside
    `main()` is faked: what is under test is the wiring from the parsed arguments
    to the engine, not the web server. It also keeps the existing guarantee that
    `--help` works without the extra — the import stays inside `main()`."""
    from myna.serve import main

    started = {}
    fake = types.ModuleType("uvicorn")
    fake.run = lambda app, **kw: started.__setitem__("app", app)
    monkeypatch.setitem(sys.modules, "uvicorn", fake)
    monkeypatch.setattr(sys, "argv", ["serve", "--ckpt", str(ckpt), "--abstain-below", "0.99"])
    main()
    assert TestClient(started["app"]).get("/v1/health").json()["abstain_below"] == 0.99


def test_a_session_ask_carries_the_same_refusal_as_predict(refusing_client):
    """The streaming path is a second entry point to the same heads; if it dropped
    the policy block, a session could quietly answer what /v1/predict refuses."""
    rng = random.Random(13)
    state = generate(1, "support", rng)[0].state
    sid = refusing_client.post("/v1/sessions", json={"state": state}).json()["session_id"]
    asked = refusing_client.post(f"/v1/sessions/{sid}/ask",
                                 json={"questions": QUESTIONS}).json()
    assert asked["policy"]["abstain_below"] == 0.99
    assert sorted(asked["policy"]["abstained"]) == sorted(QUESTIONS)


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
        # this checks plumbing (same questions reach the same heads), not
        # numeric exactness: append vs single-pass are different float32
        # summation orders on random-init weights, and exactness is pinned
        # in float64 by tests/test_trunk_numerics.py
        tol = 5e-2
        if "probabilities" in a:
            worst = max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"])
            assert worst < tol, (name, worst)
        elif "noul" in a:
            assert abs(a["noul"] - b["noul"]) < tol, name
    assert client.delete(f"/v1/sessions/{sid}").status_code == 200
    assert client.post(f"/v1/sessions/{sid}/ask", json={"questions": QUESTIONS}).status_code == 404


def test_a_missing_ckpt_names_the_fetch_that_fixes_it(tmp_path):
    """`--ckpt` defaults to `runs/myna-v0`, which exists only inside this checkout.
    A reader following the README from a pip install gets a torch `FileNotFoundError`
    for a directory with no weights in it, which names neither the directory nor the
    way to fill it."""
    empty = tmp_path / "no-weights"
    with pytest.raises(SystemExit) as e:
        main(["--ckpt", str(empty), "--port", "8099"])
    msg = str(e.value)
    assert "model.pt" in msg and "tokenizer.json" in msg
    assert "myna-weights" in msg and str(empty) in msg


def test_the_serve_extra_missing_is_a_sentence_not_a_traceback(ckpt, monkeypatch):
    """Same class as the `--help` defects `tests/test_cli_help.py` pins: an optional
    dependency may be absent, but the reader has to be told which install adds it."""
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    with pytest.raises(SystemExit) as e:
        main(["--ckpt", str(ckpt), "--port", "8099"])
    assert "myna[serve]" in str(e.value)
    assert "uv sync --extra serve" in str(e.value)

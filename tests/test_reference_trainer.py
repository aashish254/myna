"""The reference trainer must render exactly what the engine renders.

A reference training implementation that tokenizes its inputs differently from the serving
engine trains a different task, and the difference would be invisible in a loss curve. This
test pins the two renderers together on real rows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "release"))
sys.path.insert(0, str(REPO / "src"))

pytest.importorskip("transformers")
pytest.importorskip("torch")


def _rows(n=3):
    path = REPO / "data" / "decision-v2-pilot" / "test.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[:n]


def test_renderer_matches_engine():
    from myna.v6_engine import MynaV6

    from train_v6 import Renderer

    ckpt = REPO / "myna-v5-prototype 2"
    if not (ckpt / "myna-v6-unified.pt").exists():
        pytest.skip("checkpoint not present")

    engine = MynaV6(str(ckpt), device="cpu")
    renderer = Renderer(engine.tok, max_length=engine.max_length)

    from myna.real_data import _state_text

    for row in _rows(3):
        state = _state_text(row["state"])
        for qname, spec in row["questions"].items():
            prefix = renderer._prefix(f"{row['_meta']['source']}/{qname}", spec)
            options = renderer._options(spec)
            if len(options) > 16:
                continue
            ref = engine._packed_ids(prefix, state, options)
            got = renderer._packed_ids(prefix, state, options)
            assert got == ref, f"renderer drifted on {row['_meta']['id']}/{qname}"


def test_reference_checkpoint_reloads_strict(tmp_path):
    """A two-step run writes a checkpoint whose heads load with strict=True."""
    import subprocess

    out = tmp_path / "ckpt"
    cmd = [sys.executable, str(REPO / "release" / "train_v6.py"), "--steps", "1",
           "--limit", "2", "--out", str(out), "--no-encoder"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    assert res.returncode == 0, res.stderr[-2000:]
    assert (out / "myna-v6-unified.pt").exists()
    assert "strict=True" in res.stdout

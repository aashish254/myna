"""P0 hygiene: the temperature scalar's split, and what `--device auto` picks.

Both are measurement-integrity items, so both are gated here:
- fitting the single calibration scalar on `dev` scores it on the rows used to
  pick it, which flatters dev; the suite ships `calibration.jsonl` for this.
- `auto` resolving to mps/cpu only made a CUDA box train on CPU silently.
"""

import pytest
import torch

from myna.real_data import load_suite
from myna.train import resolve_device, temperature_source
from test_real_data import CHOICE_Q, _row, write_suite  # noqa: E402  (tests/ is on sys.path under pytest)


def _labelled_row(state, label):
    return _row(state, {"intent": {**CHOICE_Q["intent"], "label": label}})


def test_calibration_split_loads_into_its_own_group(tmp_path):
    write_suite(tmp_path, {"train": [_labelled_row("pay my bill", "pay_bill")],
                           "calibration": [_labelled_row("lost my card", "card_loss")] * 3})
    suite = load_suite(tmp_path)
    assert len(suite["train"]) == 1 and len(suite["calibration"]) == 1
    assert [len(e) for e in (v[1] for v in suite["train"].values())] == [1], \
        "calibration rows must not be appended to train"


def test_absent_calibration_is_empty_not_missing(tmp_path):
    write_suite(tmp_path, {"train": [_labelled_row("x", "top_up")]})
    assert load_suite(tmp_path)["calibration"] == {}


def test_temperature_prefers_calibration_and_says_so():
    calib = {"g": (["q"], [1, 2])}
    splits, name = temperature_source({"dev": {"d": 1}, "calibration": calib})
    assert (splits, name) == (calib, "calibration")


def test_temperature_falls_back_to_dev_when_no_calibration_split():
    splits, name = temperature_source({"dev": {"d": 1}, "calibration": {}})
    assert (splits, name) == ({"d": 1}, "dev")


def test_synthetic_data_with_no_calibration_key_reports_dev():
    """The synthetic corpus builds its split dict without the key at all. A
    `.get("calibration", dev)` default would fit on dev and *print* "calibration",
    which is the mislabel this line exists to prevent."""
    splits, name = temperature_source({"dev": {"d": 1}})
    assert (splits, name) == ({"d": 1}, "dev")


def test_explicit_device_never_rewritten(monkeypatch):
    """The policy under test is "an explicit name comes back unchanged", and reading
    MPS off the machine made that a claim about the laptop: CI's runner is CPU-only, so
    `resolve_device("mps")` raised there and the test went red on a box that cannot have
    MPS (§9.52). Availability is monkeypatched for both accelerators — the mirror of the
    fail-loud test below, which already refuses to assume either way."""
    assert resolve_device("cpu") == "cpu"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("mps") == "mps"
    assert resolve_device("cuda") == "cuda"


def test_explicit_unavailable_device_fails_loud():
    """P3 3a: a Kaggle job that silently lands on CPU burns hours, and `--device
    cuda` on a Mac used to die two frames inside torch. This box has MPS but no
    CUDA, so `cuda` is the naturally-unavailable case and `mps` is made
    unavailable by monkeypatch rather than assumed either way."""
    with pytest.raises(SystemExit) as e:
        resolve_device("cuda")
    assert "not available on this machine" in str(e.value) and "--device auto" in str(e.value)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(torch.backends.mps, "is_available", lambda: False)
    try:
        with pytest.raises(SystemExit) as e2:
            resolve_device("mps")
        assert "it has: cpu" in str(e2.value), "the message must name what the box does have"
    finally:
        monkey.undo()
    with pytest.raises(SystemExit) as e3:
        resolve_device("tpu")
    assert "unknown" in str(e3.value)


def test_auto_picks_cuda_when_mps_absent(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_auto_picks_mps_first_when_both_present(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"

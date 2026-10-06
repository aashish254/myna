"""The GitHub Pages site: generated, witnessed, and falsifiable in one pytest pass.

`docs/index.html` is the repo's only Pages candidate, and it is the only artifact in this repo
whose whole claim is that its numbers are somebody else's — every figure is read out of a
committed run artifact by `bench/gen_site.py`, and `bench/site_witness.py` re-derives them from
those files a second time. A green witness is worth nothing unless it can go red, so the third
test here moves two arm macros inside the page's own text and asks whether the witness notices.

`bench/mutation_site.py` is the full battery — thirty mutations across both halves of that
relationship, including the ones where the page stays honest and an artifact moves under it.
It runs subprocesses and is not a unit test, so it stays outside this file; what lives here is
the part CI can afford every push.

The two checks the site exists to hold: `make repro`-style currency (a stale page is a failed
page, not a cosmetic one) and the witness's own coverage floor, which is what catches a
presentation change that quietly stops verifying something.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "bench" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def page():
    return (REPO / "docs" / "index.html").read_text()


def test_the_committed_page_is_what_the_generator_writes():
    """A hand-edited page is the §9.30 failure with a stylesheet: the copy moved, not the data."""
    gen = _load("gen_site")
    assert gen.main(["--check"]) == 0, "docs/index.html is stale; run `uv run python bench/gen_site.py`"


def test_the_witness_passes_the_committed_page(page):
    sw = _load("site_witness")
    w = sw.run(page)
    assert not w.fails, "\n".join(w.fails)
    assert w.checks >= 180, f"the witness only made {w.checks} checks; its floor is 180"


def test_every_figure_on_the_page_names_a_file(page):
    """A number with no artifact beside it is a number nobody can re-measure."""
    reports = {"off": "runs/antiprior_off_s0.report.json",
               "on": "runs/antiprior_on_s0c.report.json",
               "v1b": "runs/v1b_kaggle_3600b.report.json"}
    arms = [ln for ln in page.splitlines() if 'class="arm"' in ln]
    assert len(arms) == 3, arms
    for tag, rel in reports.items():
        block = page.split(f'data-arm="{tag}">', 1)[1].split("</li>", 2)[0]
        assert "source <span translate=" in block, f"{tag} prints no source"
        exact = repr(json.loads((REPO / rel).read_text())["macro"]["acc"])
        assert exact in block, f"{tag} prints no exact float from {rel}"


def test_the_witness_refuses_a_page_whose_arms_swapped_places(page):
    """The battery's cheapest member, run in-process: swap the two anti-prior macros and the
    witness must name the arm it caught, not merely go red."""
    sw = _load("site_witness")
    report = json.loads((REPO / "runs/antiprior_off_s0.report.json").read_text())
    other = json.loads((REPO / "runs/antiprior_on_s0c.report.json").read_text())
    off, on = repr(report["macro"]["acc"]), repr(other["macro"]["acc"])
    assert page.count(off) and page.count(on), "the page stopped printing one of the pair"
    swapped = page.replace(off, "@@").replace(on, off).replace("@@", on)
    assert swapped != page
    w = sw.run(swapped)
    assert w.fails, "the witness accepted two arms whose macros traded places"
    assert any("off" in f and "macro.acc" in f for f in w.fails), w.fails

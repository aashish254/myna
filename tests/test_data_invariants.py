"""Corpus invariants: every label must be derivable from the state, and the
splits must be disjoint at the surface-form level. These guard the exact
things that made the first training runs uninterpretable."""

import random

from myna.data import (
    GENERIC_NOUNS,
    SPLIT_NOUNS,
    WORKFLOWS,
    generate,
)


def test_splits_have_disjoint_nouns():
    a, b, c = (set(SPLIT_NOUNS[s]) for s in ("train", "dev", "test"))
    assert not (a & b) and not (a & c) and not (b & c)


def test_every_label_is_cued():
    """For each question at least one cue template of the gold option must
    appear in the state (by its static prefix — the slots are randomized)."""
    rng = random.Random(7)
    _, cues = WORKFLOWS["support"]
    for e in generate(40, "support", rng, "train"):
        questions, _ = WORKFLOWS["support"]
        norm = e.state.lower()
        for q, lab in zip(questions, e.gold):
            frags = cues[f"{q.name}#{lab}"]
            statics = [f.split("{")[0][:20].lower() for f in frags]
            assert any(s in norm for s in statics), (e.state[:120], q.name, lab)


def test_noul_options_are_binary_ordered():
    for wf, (questions, _) in WORKFLOWS.items():
        for q in questions:
            if q.type == "noul":
                assert q.options == ["no", "yes"], (wf, q.name)


def test_generic_noun_dropout_present_and_bounded():
    import re

    rng = random.Random(8)
    exs = generate(400, "support", rng, "train")
    pat = re.compile(r"\b(" + "|".join(map(re.escape, GENERIC_NOUNS)) + r")\b")
    hits = sum(1 for e in exs if pat.search(e.state))
    # dropout fires with p=0.15 per example; allow generous statistical slack
    assert 20 < hits < 120, hits


def test_score_options_are_ordered_levels():
    for wf, (questions, _) in WORKFLOWS.items():
        for q in questions:
            if q.type == "score":
                assert len(q.options) >= 3
                # cue keys must exist for every level so index=level holds
                _, cues = WORKFLOWS[wf]
                for i in range(len(q.options)):
                    assert f"{q.name}#{i}" in cues


def test_draw_batch_never_repeats_a_row():
    from myna.train import draw_batch

    rng = random.Random(3)
    pool = list(range(400))
    for _ in range(50):
        b = draw_batch(pool, 32, rng)
        assert len(b) == 32 and len(set(b)) == 32


def test_draw_batch_shortens_rather_than_duplicating():
    """A one-row question-set must give one row, not `batch` copies of it.

    The real suite produces many such sets: option descriptions are randomized per
    row, so rows of one task land in separate tensor groups. Sampling with
    replacement there drives a step's loss to ~0 while it sees a single example."""
    from myna.train import draw_batch

    rng = random.Random(3)
    for batch in (4, 16, 32):
        assert draw_batch([7], batch, rng) == [7]
    assert draw_batch([1, 2, 3], 8, rng) != [1, 1, 1, 1, 1, 1, 1, 1]

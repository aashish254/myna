"""P1's last box: instruction paraphrase augmentation, with the suite wording held out.

The gate is not "the strings differ" — it is that **no training phrasing is ever the
wording the eval rows show**, and that a phrasing can never move a gold label. Both
are checked here against the table *and*, when the pilot corpus is on disk, against
all 73,804 of its question slots.
"""

import random
from pathlib import Path

import pytest

from myna.data import Question
from myna.paraphrase import (
    CORES,
    EVAL_INDEX,
    EXTRA_SUFFIXES,
    N_VARIANTS,
    Paraphraser,
    SUITE_SUFFIXES,
    TRAIN_INDEXES,
    UnknownSchema,
    draw_index,
    mnli_parts,
    paraphrase_questions,
    source_of,
    variants,
)

PILOT = Path("data/decision-v2-pilot")

ALL_CORES = [(src, core) for src, table in CORES.items() for core in table]


def test_every_schema_has_enough_phrasings():
    assert len(ALL_CORES) == 17, "the pilot ships exactly 17 templated cores"
    for src, core in ALL_CORES:
        vs = variants(core, src)
        assert len(vs) == N_VARIANTS >= 9, f"{src}: {len(vs)} phrasings"
        assert len(set(vs)) == N_VARIANTS, f"{src}: duplicate phrasing for {core[:40]!r}"


def test_phrasings_vary_the_tail_as_well_as_the_question():
    """Nine phrasings that differ only in their first words would still pass a
    'strings differ' check while teaching one surface form."""
    for src, core in ALL_CORES:
        vs = variants(core, src)
        ends = {s for s in EXTRA_SUFFIXES if s and any(v.endswith(s) for v in vs)}
        assert len(ends) >= 3, f"{src}: every phrasing ends the same way: {vs[0][:50]!r}"


def test_the_held_out_index_is_outside_the_training_pool():
    assert EVAL_INDEX not in set(TRAIN_INDEXES)
    assert EVAL_INDEX == N_VARIANTS - 1 and len(TRAIN_INDEXES) == N_VARIANTS - 1


def test_paraphrase_questions_returns_exactly_the_requested_index():
    """Pins the index through `variants`, so an implementation that always takes
    phrasing 0 cannot pass the 'nine phrasings exist' tests."""
    core = "Is this movie review positive?"
    vs = variants(core, "imdb")
    qs = [Question("pos", "noul", core, ["no", "yes"])]
    for i in range(N_VARIANTS):
        assert paraphrase_questions(qs, "imdb", i)[0].instruction == vs[i]


def test_a_phrasing_equal_to_the_suite_wording_is_an_error_not_a_pass():
    """The guard inside `variants` needs its own witness: mutate it away and
    nothing else notices, because the table happens not to contain that string."""
    from myna.paraphrase import CORES as TABLE

    core = "Is this movie review positive?"
    original = list(TABLE["imdb"][core])
    try:
        TABLE["imdb"][core] = [core] + original[1:]
        with pytest.raises(AssertionError):
            variants(core, "imdb")
        # a suffixed suite wording counts too: the core is what it reduces to
        TABLE["imdb"][core] = [core + SUITE_SUFFIXES[0]] + original[1:]
        with pytest.raises(AssertionError):
            variants(core + SUITE_SUFFIXES[0], "imdb")
    finally:
        TABLE["imdb"][core] = original


def test_no_phrasing_is_the_suites_own_wording():
    for src, core in ALL_CORES:
        assert core not in variants(core, src)
        for suffix in SUITE_SUFFIXES:
            assert core + suffix not in variants(core + suffix, src)


def test_unknown_instruction_fails_loud_instead_of_passing_through():
    with pytest.raises(UnknownSchema):
        variants("What is the sentiment of this tweet?", "sst5")
    with pytest.raises(UnknownSchema):
        variants("What is the topic of this article?", "never-seen-source")


def test_boolq_frame_varies_and_question_text_is_verbatim():
    q = "is the book of james in the new testament?"
    vs = variants(q, "boolq")
    assert len(vs) == N_VARIANTS and q not in vs
    for v in vs:
        assert q in v, f"the row's question text must not be edited: {v!r}"


def test_boolq_keeps_a_suite_suffix_that_happens_to_end_its_question():
    """The suite appends its three suffixes at random, so a boolq question can end
    with one. `strip_suite_suffix` runs before the source branch, and dropping that
    tail would edit the row's own question — so the whole string must survive."""
    q = "did the two sides agree to a ceasefire? " + SUITE_SUFFIXES[1].strip()
    assert q.endswith(SUITE_SUFFIXES[1].strip())
    for v in variants(q, "boolq"):
        assert q in v, f"a suite-looking tail was stripped from the row's question: {v!r}"


def test_content_frames_vary_their_tail_too():
    """boolq/mnli vary by frame, and a frame-only axis would leave all nine
    sentences ending the same way — the same single surface form the gate exists
    to avoid, just further into the string."""
    cases = {"boolq": "is the bridge closed to traffic?",
             "mnli": 'Hypothesis: "The bridge is closed." How does it relate to the premise?'}
    for src, instr in cases.items():
        vs = variants(instr, src)
        ends = {s for s in EXTRA_SUFFIXES if s and any(v.endswith(s) for v in vs)}
        assert len(ends) >= 3, f"{src}: the nine frames all end the same way"


def test_mnli_frame_varies_and_hypothesis_is_verbatim():
    instr = 'Hypothesis: "The man shared his wealth with the country." How does it relate to the premise?'
    hyp = '"The man shared his wealth with the country."'
    vs = variants(instr, "mnli")
    assert mnli_parts(instr) == hyp
    assert len(vs) == N_VARIANTS and instr not in vs
    for v in vs:
        assert hyp in v, f"the row's hypothesis must not be edited: {v!r}"


def test_mnli_parts_rejects_a_foreign_shape():
    with pytest.raises(UnknownSchema):
        mnli_parts("Is this movie review positive?")


def test_paraphrase_touches_only_the_instruction():
    qs = [Question("stars", "score", "How many stars did this reviewer give?",
                   ["1 star: terrible experience", "2 stars: poor"]),
          Question("recommend", "noul", "Would this reviewer recommend the business?",
                   ["no", "yes"])]
    for i in range(N_VARIANTS):
        got = paraphrase_questions(qs, "yelp", i)
        assert [q.name for q in got] == [q.name for q in qs]
        assert [q.type for q in got] == [q.type for q in qs]
        # the gold is an option index, so option order is what must not move
        assert [q.options for q in got] == [q.options for q in qs]
        assert all(g.instruction != o.instruction for g, o in zip(got, qs))
    assert got[0].options is not qs[0].options, "options must be a copy, not a shared list"


def test_out_of_range_index_is_an_error():
    with pytest.raises(ValueError):
        paraphrase_questions([Question("x", "noul", "Is this movie review positive?", ["no", "yes"])],
                             "imdb", N_VARIANTS)


def test_training_never_draws_the_held_out_phrasing():
    rng = random.Random(0)
    drawn = {draw_index(rng) for _ in range(4000)}
    assert drawn == set(TRAIN_INDEXES) and EVAL_INDEX not in drawn


def test_paraphraser_is_memoized_and_stays_inside_the_training_pool():
    rng = random.Random(7)
    qs = [Question("topic", "choice", "What is the topic of this article?", ["w", "s", "b", "t"])]
    p = Paraphraser()
    sets = p.variant_set("agnews#abc", qs)
    assert len(sets) == len(TRAIN_INDEXES), "the pool holds eight phrasings, not nine"
    assert p.variant_set("agnews#abc", qs) is sets
    base = qs[0].instruction
    held_out = variants(base, "agnews")[EVAL_INDEX]
    seen = set()
    for _ in range(500):
        v = p.draw("agnews#abc", qs, rng)
        assert v is not qs
        assert v[0].instruction != base, "a drawn phrasing must never be the suite wording"
        assert v[0].instruction != held_out, "nor the reserved eval phrasing"
        seen.add(v[0].instruction)
    assert seen == {s[0].instruction for s in sets}
    assert p.warm({"agnews#abc": (qs, [])}) == 0, "already built"
    assert p.draws == 500, "the trainer's witness counts draws, so they must be counted"


def test_paraphraser_warm_reaches_every_schema_in_the_split():
    qs = [Question("tone", "score", "What is the sentiment of this review sentence?", ["a", "b"])]
    p = Paraphraser()
    assert p.warm({"sst5#1": (qs, []), "yelp#2": ([Question(
        "stars", "score", "How many stars did this reviewer give?", ["1", "2"])], [])}) == 2
    with pytest.raises(UnknownSchema):
        p.warm({"trec#9": ([Question("q", "choice", "not a known trec core", ["a"])], [])})


@pytest.mark.skipif(not (PILOT / "train.jsonl").exists(), reason="pilot corpus not on disk")
def test_pilot_coverage_every_slot_paraphrases_and_no_train_wording_is_an_eval_wording():
    from myna.real_data import load_suite

    d = load_suite(PILOT)
    eval_wordings = {q.instruction for _key, (qs, _e) in d["dev"].items() for q in qs}
    eval_wordings |= {q.instruction for _key, (qs, _e) in d["test"].items() for q in qs}
    p = Paraphraser()
    built = p.warm(d["train"])
    shapes = sum(len(qs) for _key, (qs, _e) in d["train"].items())
    rows = sum(len(exs) for _key, (_qs, exs) in d["train"].items())
    labels = sum(len(qs) * len(exs) for _key, (qs, exs) in d["train"].items())
    train_wordings = {q.instruction for sets in p._sets.values() for v in sets for q in v}
    leaked = train_wordings & eval_wordings
    assert not leaked, f"{len(leaked)} training phrasings are suite eval wordings: {list(leaked)[:3]}"
    # the suite's own strings must be absent from the training pool too
    suite_strings = {q.instruction for _key, (qs, _e) in d["train"].items() for q in qs}
    assert not (train_wordings & suite_strings)
    assert (built, shapes, rows, labels) == (17112, 19596, 57904, 73804), \
        "the warm pass must reach every set in the split, not most of them"
    print(f"\npilot: {built} sets / {shapes} question shapes / {rows} rows / {labels} labelled "
          f"slots paraphrased, {len(train_wordings)} distinct training wordings, 0 leakage")

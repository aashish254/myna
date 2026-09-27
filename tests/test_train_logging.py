"""A run must record what it trained on (SPEC §9.30).

`runs/train-v0.log` printed the vocab and the parameter count but no row counts,
so every corpus size in the docs was unverifiable — the "3,000 synthetic
examples" beside v0's table was one of them. `split_report` is the line that
fixes it, and it is only a fix if the line cannot silently stop carrying the
numbers, which is what these tests are for.
"""

from myna.train import split_report


def _groups(sizes):
    return {name: (["q"], list(range(n))) for name, n in sizes.items()}


def test_report_names_rows_groups_and_the_uneven_ends():
    line = split_report("dev", _groups({"a": 8, "b": 6, "c": 10}))
    assert line == "split dev: 24 rows over 3 groups, per-group 6-10"
    # the sums are not the same number, so a report that printed either twice
    # would still look plausible — pin all four figures
    assert "24" in line and "3 groups" in line and "6-10" in line


def test_uneven_split_cannot_hide_behind_its_own_total():
    """600 rows over 3 groups reads the same whether it is 200/200/200 or
    598/1/1 — and only the second one means the macro is weighted by luck."""
    even = split_report("train", _groups({"a": 200, "b": 200, "c": 200}))
    lopsided = split_report("train", _groups({"a": 598, "b": 1, "c": 1}))
    assert even.split(":")[0] == lopsided.split(":")[0] == "split train"
    assert "600 rows over 3 groups" in even and "600 rows over 3 groups" in lopsided
    assert even.endswith("per-group 200-200") and lopsided.endswith("per-group 1-598")


def test_single_group_row_is_still_a_row_count():
    assert split_report("test", _groups({"only": 3})) == \
        "split test: 3 rows over 1 groups, per-group 3-3"


def test_empty_split_prints_instead_of_raising():
    """A suite with an empty group set must still log, or the run dies at the
    instrumentation rather than at the training it was about to do."""
    assert split_report("dev", {}) == "split dev: 0 rows over 0 groups"

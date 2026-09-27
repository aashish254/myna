"""Mutation battery for the ordinal loss (SPEC §5 P9: 9a, §7.1, §9.18).

9a changes what a training run *prices*, so the code is the artifact and the verdict
is the next Kaggle run's. That makes this battery the only local witness available:
`tests/test_ordinal_loss.py` has to fail for a named reason when any of the decisions
in `typed_loss`'s emd branch is broken — the divisor, the step function, which cells
get repriced, the tail mask, the sum's own axis, and the `ce` default that keeps every
published figure where it is.

Same discipline as the other batteries: `src`, `tests` and `bench` are copied into a
scratch repo (pyproject pins `pythonpath = ["src"]` against the rootdir, so a mutated
file reached through `PYTHONPATH` silently loses to the real one — §9.12), `data`,
`runs` and `.git` are symlinked and every `*.md/*.toml/*.cfg` sits beside them, one
textual mutation is applied, and `tests/test_ordinal_loss.py` runs there. Three of the
tests call `train.main()` for a single CPU update through a `typed_loss` spy, because
the trainer does not seed torch: two processes never share an init, so "the printed
loss changed" is not evidence that a flag did anything, and the call site has to be
watched instead of its output.

Three entries are worth naming before the run.

* `** 2 * opt_valid.to(cdf.dtype)` → `** 2 * 1.0` survives *every* hand-built masked
  input and is caught by exactly one test, `test_the_builder_masks_feed_the_loss_in_
  the_order_the_loop_passes_them`, which feeds the builders' real `opt_valid` beside
  unmasked random logits. Masked padding makes the tail terms `(1-1)²` and the
  mutation invisible; that is why the sum is masked rather than trusted.
* `argparse default "ce"` → `"emd"` is the mutation that would move every published
  number while changing no code path anyone tests explicitly. It is caught by the
  spy test running the trainer with **no** `--score-loss` at all.
* `q.type == "score"` → `q.type != "score"` in `build_batch` marks the choice cells
  and unmarks the ordinal ones: the run still trains, still converges, and reports an
  ordinal objective it never applied. Only the builder test's literal can see it.

One entry was *tried*, survived, and removed, and the removal is the finding: an
explicit `ordinal.expand(gold.shape[0], -1)` for the shared `[N]` form could not be
caught by any input, because torch's own broadcasting turns `[N]` against `[B,N]` into
exactly the same tensor — a shared mask, by definition, says the same thing about every
row. `if ordinal.dim() == 1:` was therefore dead code wearing a contract, and it is
gone; `test_the_shared_and_per_row_forms_agree` stays as the witness that the shared
form still prices identically to the per-row one.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
TESTS = ["tests/test_ordinal_loss.py"]
MO = "src/myna/model.py"
TR = "src/myna/train.py"

MUTATIONS = [
    # --- the ce default: every published figure rests on it -------------------
    ("typed_loss's own default stops being the published objective", MO,
     '    score_loss: str = "ce",', '    score_loss: str = "emd",'),
    ("the ce branch stops returning early, so ce needs the emd masks", MO,
     '    if score_loss == "ce":', "    if False:"),
    ("the trainer's flag default flips to emd", TR,
     'choices=["ce", "emd"], default="ce"', 'choices=["ce", "emd"], default="emd"'),
    # --- the divisor ----------------------------------------------------------
    ("the price is normalised by the batch's padded O, not the cell's K", MO,
     "emd / (opt_valid.sum(-1) - 1).clamp(min=1)", "emd / (logits.shape[-1] - 1)"),
    ("the legend is normalised by K instead of K-1, so the worst price is under 1", MO,
     "(opt_valid.sum(-1) - 1).clamp(min=1)", "opt_valid.sum(-1).clamp(min=1)"),
    # --- the step function and the CDF ----------------------------------------
    ("the gold rung is exclusive, so a perfect prediction still costs", MO,
     ">= gold[..., None]", "> gold[..., None]"),
    ("the CDF is accumulated from the wrong end of the legend", MO,
     "lp.exp().cumsum(-1)", "lp.exp().flip(-1).cumsum(-1).flip(-1)"),
    ("the distance is absolute rather than squared", MO,
     "** 2 * opt_valid.to(cdf.dtype)", "** 1 * opt_valid.to(cdf.dtype)"),
    ("the sum stops being bounded by the cell's own options", MO,
     "** 2 * opt_valid.to(cdf.dtype)", "** 2 * 1.0"),
    # --- which cells get repriced ---------------------------------------------
    ("every cell with a gold becomes ordinal", MO,
     "torch.where(ordinal, emd /", "torch.where(has_gold, emd /"),
    ("no cell is ordinal, so emd is ce wearing a new name", MO,
     "torch.where(ordinal, emd /", "torch.where(torch.zeros_like(ordinal), emd /"),
    # --- the loop's call site --------------------------------------------------
    ("the loop forgets to pass the flag", TR,
     'b["opt_valid"], args.score_loss) / K', 'b["opt_valid"]) / K'),
    ("the loop passes has_gold where the ordinal mask belongs", TR,
     'loss = typed_loss(logits, b["gold"], b["has_gold"], b["ordinal"],',
     'loss = typed_loss(logits, b["gold"], b["has_gold"], b["has_gold"],'),
    ("the loop withholds the masks the objective asked for", TR,
     'loss = typed_loss(logits, b["gold"], b["has_gold"], b["ordinal"],\n'
     '                              b["opt_valid"], args.score_loss) / K',
     'loss = typed_loss(logits, b["gold"], b["has_gold"], None,\n'
     '                              None, args.score_loss) / K'),
    # --- the builders' mark ----------------------------------------------------
    ("build_batch marks the non-score questions ordinal", TR,
     '[q.type == "score" for q in questions]', '[q.type != "score" for q in questions]'),
    ("build_row_batch marks every live cell ordinal", TR,
     "ordinal[i, : len(qs)] = torch.tensor([q.type == \"score\" for q in qs], dtype=torch.bool)",
     "ordinal[i, : len(qs)] = True"),
    # --- the refusal that keeps a flag from lying ------------------------------
    ("--long-context accepts a loss it cannot price", TR,
     '        if args.score_loss == "emd":', "        if False:"),
    # --- the mask stops being built at all -------------------------------------
    ("build_batch stops emitting the ordinal mask", TR,
     '        "ordinal": torch.tensor([q.type == "score" for q in questions],\n'
     '                                dtype=torch.bool).to(device),\n', ""),
    ("build_row_batch stops emitting the ordinal mask", TR,
     '        ordinal[i, : len(qs)] = torch.tensor([q.type == "score" for q in qs],'
     ' dtype=torch.bool)\n', ""),
]


def make_scratch(tmp):
    """Copy the code under test, link everything the tests read out of the repo.

    The `.git` symlink and the docs are not needed by `tests/test_ordinal_loss.py`
    itself; they are here so the scratch tree is the same shape as the other
    batteries', and so a test that later grows a dependency on the registry or the
    prose does not turn a mutation run into a spurious red baseline.
    """
    repo = Path(tmp) / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir()
    for name in ("src", "tests", "bench"):
        shutil.copytree(ROOT / name, repo / name,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for name in ("data", "runs", ".git"):
        if (ROOT / name).exists():
            (repo / name).symlink_to(ROOT / name)
    for name in sorted(p.name for p in ROOT.iterdir() if p.suffix in (".md", ".toml", ".cfg")):
        (repo / name).symlink_to(ROOT / name)
    return repo


def pytest_in(repo):
    env = {k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-q", "--no-header",
                           "-p", "no:cacheprovider"],
                          capture_output=True, text=True, cwd=repo, env=env, timeout=1800)


def run_one(rel, old, new, tmp):
    repo = make_scratch(tmp)
    path = repo / rel
    text = path.read_text()
    n = text.count(old)
    if n != 1:
        return f"BAD-PATTERN ({n} matches)", ""
    path.write_text(text.replace(old, new))
    r = pytest_in(repo)
    if r.returncode == 0:
        return "SURVIVED", ""
    fails = [ln for ln in r.stdout.splitlines() if ln.startswith("FAILED")
             or ln.startswith("ERROR")]
    detail = fails[0] if fails else (r.stdout.strip().splitlines() or [""])[-1]
    return "caught", detail[:110]


def main():
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__)
        return 0
    survivors, bad = [], []
    with tempfile.TemporaryDirectory() as tmp:
        base = make_scratch(tmp)
        r0 = pytest_in(base)
        if r0.returncode != 0:
            print("baseline (unmutated) copy is RED -- the battery proves nothing")
            print(r0.stdout[-4000:])
            return 2
        print(f"baseline copy: green ({len(TESTS)} test file)\n")
        for i, (label, rel, old, new) in enumerate(MUTATIONS, 1):
            verdict, detail = run_one(rel, old, new, tmp)
            print(f"[{i:2d}/{len(MUTATIONS)}] {verdict:11s} {label}"
                  + (f"\n            {detail}" if detail and verdict == "caught" else ""),
                  flush=True)
            if verdict == "SURVIVED":
                survivors.append(label)
            elif verdict.startswith("BAD-PATTERN"):
                bad.append((label, verdict))
        print(f"\n{len(MUTATIONS) - len(survivors) - len(bad)}/{len(MUTATIONS)} mutations caught")
        for s in survivors:
            print(f"  SURVIVED: {s}")
        for b in bad:
            print(f"  {b}")
        return 1 if survivors or bad else 0


if __name__ == "__main__":
    sys.exit(main())

"""Mutation battery for the paraphrase gate (SPEC §5 P1 task 15g, §7.1 discipline).

    python bench/mutation_paraphrase.py

Each entry rewrites one line of `paraphrase.py` / `train.py` in a scratch copy of
the repo and re-runs the two paraphrase test files there. A mutation that survives
is a hole in the gate: the tests pass while the code is wrong, which is exactly
the failure mode an accuracy number cannot show. Prints the survivors and exits
non-zero, so this is a check, not a report.

Why a *copy of the repo* and not a patched `src` on `PYTHONPATH`: `pyproject.toml`
pins `pythonpath = ["src"]` relative to the pytest rootdir, which outranks
`PYTHONPATH`. The first version of this battery did the latter and printed
"0/29 mutations caught" while testing the unmutated tree — a false green of the
worst kind (it looks like the tests are weak, when the harness is dead). Hence the
two guards: the unmutated baseline must be green, and if the *first* mutation
survives the run aborts as broken rather than reporting a weak suite.

32/32 caught as of this writing. Its second pass found two real holes, both now
tested: an editing of boolq's own question text when a row's question happens to
end with one of the suite's randomized suffixes, and content frames whose nine
variants all ended the same way. Its third pass added the three §9.58 mutations
that pin the draw-rate denominator — the rate over executed updates, the executed
dose reading `--resume`, and `updates_executed` in `metrics.json` — because the
claim being retracted there was made by a print, not by a test. One candidate
mutation was dropped as equivalent
rather than weak: `strip_suite_suffix` ends with `.rstrip()`, and every
`SUITE_SUFFIXES` entry starts with a space, so deleting that call cannot change
any output on this data. It stayed in the code as insurance against a future
suffix that does not.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ["tests/test_paraphrase.py", "tests/test_paraphrase_wiring.py"]

PP = "src/myna/paraphrase.py"
TR = "src/myna/train.py"

MUTATIONS = [
    ("draw can pick the held-out phrasing", PP,
     "return rng.randrange(N_VARIANTS - 1)", "return rng.randrange(N_VARIANTS)"),
    ("EVAL_INDEX points inside the training pool", PP,
     "EVAL_INDEX = N_VARIANTS - 1", "EVAL_INDEX = 0"),
    ("training pool includes the held-out phrasing", PP,
     "TRAIN_INDEXES = range(N_VARIANTS - 1)", "TRAIN_INDEXES = range(N_VARIANTS)"),
    ("draw always returns the same phrasing", PP,
     "return sets[rng.randrange(len(sets))]", "return sets[0]"),
    ("suite suffix never stripped (lookup fails, or trains on eval wording)", PP,
     "core, _had_suffix = strip_suite_suffix(instruction)", "core, _had_suffix = instruction, False"),
    ("boolq content taken from the stripped core, editing the row's question", PP,
     'content = instruction if source == "boolq" else mnli_parts(instruction)',
     'content = core if source == "boolq" else mnli_parts(instruction)'),
    ("mnli hypothesis keeps the frame tail", PP,
     "return instruction[len(MNLI_HEAD): instruction.index(MNLI_TAIL)].strip()",
     "return instruction[len(MNLI_HEAD):].strip()"),
    ("mnli_parts passes a foreign shape through", PP,
     'raise UnknownSchema(f"mnli instruction outside the shipped frame: {instruction[:70]!r}")',
     "return instruction"),
    ("unknown schema passes the suite wording through", PP,
     'raise UnknownSchema(f"no phrasing table for {source}: {core[:70]!r}")',
     "return [instruction] * N_VARIANTS"),
    ("equality guard against the suite wording removed", PP,
     "if any(v.strip() == instruction.strip() for v in out):", "if False:"),
    ("variant index range check removed", PP,
     "if not 0 <= index < N_VARIANTS:", "if False:"),
    ("options list shared instead of copied", PP,
     "out.append(Question(q.name, q.type, v, list(q.options)))",
     "out.append(Question(q.name, q.type, v, q.options))"),
    ("paraphrase_questions leaves the instruction alone", PP,
     "v = variants(q.instruction, source)[index]", "v = q.instruction"),
    ("source_of returns the signature, not the source", PP,
     'return group_key.split("#", 1)[0]', 'return group_key.split("#", 1)[-1]'),
    ("tail axis dropped: nine phrasings, one ending", PP,
     "return [s.strip() + EXTRA_SUFFIXES[i % len(EXTRA_SUFFIXES)] for i, s in enumerate(out)]",
     "return [s.strip() for i, s in enumerate(out)]"),
    ("content frames lose their suffix axis", PP,
     "return [s + EXTRA_SUFFIXES[i % len(EXTRA_SUFFIXES)] for i, s in enumerate(out)]",
     "return list(out)"),
    ("warm reports zero sets built", PP, "built += 1", "built += 0"),
    ("draws stop being counted", PP,
     "self.draws += 1  # counted so the trainer can prove the loop consulted it", "pass"),
    ("row batch ignores the paraphraser", TR,
     "items = [(paraphraser.draw(ex.workflow, qs, rng), ex) for qs, ex in items]",
     "items = [tuple(it) for it in items]"),
    ("row batch draws under the first row's source", TR,
     "items = [(paraphraser.draw(ex.workflow, qs, rng), ex) for qs, ex in items]",
     "items = [(paraphraser.draw(items[0][1].workflow, qs, rng), ex) for qs, ex in items]"),
    ("tensors built from the pre-paraphrase items", TR,
     "    if paraphraser is not None:\n        items = [(paraphraser.draw(ex.workflow, qs, rng), ex) for qs, ex in items]\n    qt = batch_question_tensors(tok, [[(q.instruction, q.options) for q in qs] for qs, _ in items])",
     "    pre = list(items)\n    if paraphraser is not None:\n        items = [(paraphraser.draw(ex.workflow, qs, rng), ex) for qs, ex in items]\n    qt = batch_question_tensors(tok, [[(q.instruction, q.options) for q in qs] for qs, _ in pre])"),
    ("budget priced at the first phrasing", TR, "worst = max(lengths)", "worst = lengths[0]"),
    ("budget priced at the shortest phrasing", TR, "worst = max(lengths)", "worst = min(lengths)"),
    ("base question list left at its true length", TR,
     "        cache[id(qs)] = worst", "        pass"),
    ("evaluate ignores the requested phrasing", TR,
     "if paraphrase_index is not None:", "if False:"),
    ("held-out eval reads a training index", TR,
     "return evaluate(model, tok, split, device, temperature, paraphrase_index=EVAL_INDEX)",
     "return evaluate(model, tok, split, device, temperature, paraphrase_index=0)"),
    ("shared-set path stops drawing", TR,
     "                    questions = paraphraser.draw(wf, questions, rng)", "                    pass"),
    ("--paraphrase off still paraphrases", TR,
     'if args.paraphrase == "on":\n        from .paraphrase import Paraphraser',
     'if True:\n        from .paraphrase import Paraphraser'),
    ("long-context + paraphrase runs instead of refusing", TR,
     'if args.paraphrase == "on":\n            # the needle corpus',
     'if False:\n            # the needle corpus'),
    # the denominator the counter is divided by (§9.58): a rate over `--steps` is a rate over
    # updates that never happened, which is how a 15.2% coverage gap was printed out of a pair
    # that was coverage-matched to 0.04%.
    ("the draw rate is reported over the steps asked for", TR,
     'f"{updates_executed} updates executed (of {args.steps} requested"',
     'f"{args.steps} updates executed (of {args.steps} requested"'),
    ("the executed dose ignores --resume", TR,
     "    updates_executed = max(last_step - start_step + 1, 0)",
     "    updates_executed = max(last_step, 0)"),
    ("metrics records the requested dose as the executed one", TR,
     '"steps_requested": args.steps, "updates_executed": updates_executed,',
     '"steps_requested": args.steps, "updates_executed": args.steps,'),
]


def make_scratch(tmp: str) -> Path:
    repo = Path(tmp) / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir()
    for name in ("src", "tests"):
        shutil.copytree(ROOT / name, repo / name,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    shutil.copy(ROOT / "pyproject.toml", repo / "pyproject.toml")
    (repo / "data").symlink_to(ROOT / "data")  # the pilot-coverage test reads it
    return repo


def pytest_in(repo: Path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run(
        [sys.executable, "-m", "pytest", *TESTS, "-q", "-x", "--no-header",
         "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=repo, env=env, timeout=1800)


def run_one(label, rel, old, new, tmp) -> str:
    repo = make_scratch(tmp)
    path = repo / rel
    text = path.read_text()
    if text.count(old) != 1:
        return f"BAD-PATTERN({text.count(old)})"
    path.write_text(text.replace(old, new))
    r = pytest_in(repo)
    if r.returncode == 0:
        return "SURVIVED"
    failed = [ln for ln in r.stdout.splitlines() if ln.startswith("FAILED")]
    return "caught  " + (failed[0] if failed else r.stdout.strip().splitlines()[-1])[:88]


def main() -> int:
    survivors, bad = [], []
    with tempfile.TemporaryDirectory() as tmp:
        base = pytest_in(make_scratch(tmp))
        if base.returncode != 0:
            print("baseline (unmutated) copy is RED -- the battery proves nothing")
            print(base.stdout[-3000:])
            return 2
        print(f"baseline copy: green ({len(TESTS)} test files)\n", flush=True)
        for i, (label, rel, old, new) in enumerate(MUTATIONS, 1):
            verdict = run_one(label, rel, old, new, tmp)
            print(f"[{i:2d}/{len(MUTATIONS)}] {verdict:12s} {label}", flush=True)
            if verdict == "SURVIVED":
                survivors.append(label)
                if i == 1:
                    print("  the first mutation survived: the harness is not testing the "
                          "mutated code. Aborting rather than reporting a weak suite.",
                          flush=True)
                    return 2
            elif verdict.startswith("BAD-PATTERN"):
                bad.append((label, verdict))
    caught = len(MUTATIONS) - len(survivors) - len(bad)
    print(f"\n{caught}/{len(MUTATIONS)} mutations caught")
    for s in survivors:
        print(f"  SURVIVED: {s}")
    for b in bad:
        print(f"  stale pattern: {b[0]} ({b[1]}) — the code moved; re-derive the mutation")
    return 1 if survivors or bad else 0


if __name__ == "__main__":
    if {"-h", "--help"} & set(sys.argv[1:]):  # not argparse: the battery takes no options
        print(__doc__)
        sys.exit(0)
    sys.exit(main())

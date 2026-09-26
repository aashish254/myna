"""Mutation battery for the G5 abstention gates (SPEC §5 P5, §7.1 discipline).

    python bench/mutation_p5.py

Three modules publish a claim that no external truth can check:

* `myna.engine` says *this answer is committed* or *this answer is refused*;
* `myna.fallback` says *this engine committed to it* on every answer;
* `bench.risk_coverage` says *at this floor the model is 95% right on 60% of the
  questions* — a curve, whose numbers are the artifact.

So the witness is the same one used for G2: mutate each lie that is available, and
require a test to fail. The lies cluster in three families, and each family is how
the feature would actually rot.

**A fast path going silent.** `"choice": None if gate["abstain"] else labels[i]`
becoming `"choice": labels[i]` leaves every test about probabilities green and ships
a model that answers when it should not — the exact failure G5 exists to prevent.
Same family: the engine ignoring its own threshold, `still_abstained` emptied, the
policy echo reporting `None`.

**A quantity read off the wrong side.** noul's confidence is `max(p, 1-p)`, so an
implementation that reads `p(Yes)` abstains on the model's surest "no"s; the
fallback's `model` label becoming the primary's makes a routed answer claim to be
something it is not.

**A boundary flipping quietly.** The engine abstains under the floor and the curve
keeps at or above it, and the floor is always read from a row that sits *exactly*
on the line — so `<` versus `<=` moves real rows in every published table, which is
why `test_the_floor_belongs_to_the_kept_set_and_to_nobody_else` crosses from the
curve into `abstain_check`.

Same scratch-repo mechanics as `mutation_latency_matched.py`: copy `src` + `bench` +
`tests` (pyproject pins `pythonpath = ["src"]` relative to the rootdir, so a mutated
file reached through PYTHONPATH silently loses to the unmutated one, §9.12), symlink
`data` and `runs`, require a green baseline, abort if the *first* mutation survives,
and report any pattern that did not match exactly once. Four files, 49 mutations, all
caught as of this writing: `engine.py` (16), `fallback.py` (15, four of them the laya
adapter), `serve.py` (2) and `risk_coverage.py` (16).

Deliberately absent: nothing mutates `INFER_CHUNK`, the pointer-head arithmetic, or
the rounding in the printed tables. Those change numbers without changing a claim,
and counting them would inflate the coverage figure.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
TESTS = ["tests/test_abstain.py", "tests/test_fallback.py", "tests/test_risk_coverage.py",
         "tests/test_serve.py"]

EN = "src/myna/engine.py"
FB = "src/myna/fallback.py"
RC = "bench/risk_coverage.py"
SV = "src/myna/serve.py"

MUTATIONS = [
    # --- the engine's own commitment ------------------------------------------
    ("the default engine abstains on everything", EN,
     '        return {"abstain": False, "confidence": None, "margin": None, "reason": None}',
     '        return {"abstain": True, "confidence": None, "margin": None, "reason": None}'),
    ("the floor becomes exclusive: a row exactly on it is refused", EN,
     '    return {"abstain": conf < threshold,',
     '    return {"abstain": conf <= threshold,'),
    ("noul confidence read off p(Yes) instead of the committed side", EN,
     "    i = max(range(len(probs)), key=lambda k: probs[k])",
     "    i = len(probs) - 1"),
    ("margin becomes the confidence minus nothing", EN,
     "    margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)",
     "    margin = ordered[0]"),
    ("a zero floor is accepted, so the guard never fires", EN,
     "    if not 0.0 < threshold <= 1.0:",
     "    if not 0.0 <= threshold <= 1.0:"),
    ("labels and probabilities no longer have to line up", EN,
     "    if len(probs) != len(labels) or not probs:",
     "    if False:"),
    ("the reason names whichever option came first", EN,
     '            "reason": f"{labels[i]} at p={conf:.3f}, {margin:.3f} ahead of the runner-up, "',
     '            "reason": f"{labels[0]} at p={conf:.3f}, {margin:.3f} ahead of the runner-up, "'),
    ("the engine ignores the threshold it was built with", EN,
     "        gate = abstain_check(pl, self.abstain_below, labels)",
     "        gate = abstain_check(pl, None, labels)"),
    ("an abstained choice keeps its label: the silent fast path", EN,
     '            return {"type": "choice", "choice": None if gate["abstain"] else labels[i],',
     '            return {"type": "choice", "choice": labels[i],'),
    ("an abstained score keeps its level", EN,
     '                    "level": None if gate["abstain"] else labels[i],',
     '                    "level": labels[i],'),
    ("an abstained score keeps its expectation", EN,
     '                    "score": None if gate["abstain"] else sum(k * p for k, p in enumerate(pl)),',
     '                    "score": sum(k * p for k, p in enumerate(pl)),'),
    ("an abstained noul still says which side it chose", EN,
     '        return {"type": "noul", "noul": pl[1], "yes": None if gate["abstain"] else pl[1] >= 0.5,',
     '        return {"type": "noul", "noul": pl[1], "yes": pl[1] >= 0.5,'),
    ("a withheld noul loses the probability the Brier score needs", EN,
     '        return {"type": "noul", "noul": pl[1], "yes": None if gate["abstain"] else pl[1] >= 0.5,',
     '        return {"type": "noul", "noul": None if gate["abstain"] else pl[1], "yes": None if gate["abstain"] else pl[1] >= 0.5,'),
    ("the response stops echoing the policy", EN,
     '            "policy": {"abstain_below": self.m.abstain_below, "abstained": abstained},',
     '            "policy": {"abstain_below": None, "abstained": abstained},'),
    ("the abstention count is always zero", EN,
     '        abstained = sorted(k for k, a in answers.items() if a.get("abstain"))',
     "        abstained = []"),
    ("a construction-time floor skips validation", EN,
     "        if abstain_below is not None and not 0.0 < float(abstain_below) <= 1.0:",
     "        if abstain_below is not None and not 0.0 <= float(abstain_below) <= 1.0:"),
    # --- the seam: who answered, and who was asked ------------------------------
    ("the router never routes: nothing is ever pending", FB,
     '        pending = sorted(k for k, a in answers.items() if a.get("abstain"))',
     "        pending = []"),
    ("the router re-asks everything, hiding what abstention costs", FB,
     '        pending = sorted(k for k, a in answers.items() if a.get("abstain"))',
     "        pending = sorted(questions)"),
    ("a routed answer is labelled with the fast engine", FB,
     '                answers[k] = {**a, "model": self.secondary.name}',
     '                answers[k] = {**a, "model": self.primary.name}'),
    ("the primary label overwrites the ones already routed", FB,
     '            answers[k].setdefault("model", self.primary.name)',
     '            answers[k]["model"] = self.primary.name'),
    ("a fallback that also refuses is reported as resolved", FB,
     '                        "still_abstained": pending, "rounds": rounds},',
     '                        "still_abstained": [], "rounds": rounds},'),
    ("the routed count is hardcoded to zero", FB,
     '    return sum(1 for a in answers.values() if a.get("model") == name)',
     "    return 0"),
    ("an engine can be its own fallback", FB,
     "        if primary is secondary:",
     "        if False:"),
    ("a zero round budget is accepted", FB,
     "        if max_rounds < 1:",
     "        if max_rounds < 0:"),
    ("the registry loses laya, leaving one load-bearing engine", FB,
     'DECIDERS: dict[str, type[Decider]] = {"myna": MynaDecider, "laya": LayaDecider}',
     'DECIDERS: dict[str, type[Decider]] = {"myna": MynaDecider}'),
    ("an unknown decider raises something no CLI reports", FB,
     '        raise SystemExit(f"unknown decider {name!r}; registry has: {\', \'.join(sorted(DECIDERS))}")',
     "        raise KeyError(name)"),
    ("the secondary gets a second round nobody asked for", FB,
     "        while pending and rounds < self.max_rounds:",
     "        while pending and rounds <= self.max_rounds:"),
    ("the laya shim keeps noul's criteria, so the fallback raises mid-answer", FB,
     '        if q.get("type") == "noul":',
     "        if False:"),
    ("the laya shim hands noul myna's own shape", FB,
     '            out[qid] = {"type": "noul", "instructions": q["instructions"]}',
     "            out[qid] = {**q}"),
    ("a gold label crosses the seam into the fallback", FB,
     '            out[qid] = {k: v for k, v in q.items() if k != "label"}',
     "            out[qid] = {k: v for k, v in q.items()}"),
    ("the laya side reaches the Router, which picks its own model", FB,
     "        return self.agent.system_one(state, laya_spec(questions))",
     "        return self.agent.router(state, laya_spec(questions))"),
    # --- the transport and the CLI: a refusal that never leaves the process -----
    ("health stops reporting the floor in force", SV,
     '                "abstain_below": myna.abstain_below}',
     '                "abstain_below": None}'),
    ("the --abstain-below flag never reaches the engine", SV,
     "    myna = Myna(args.ckpt, device=args.device, abstain_below=args.abstain_below)",
     "    myna = Myna(args.ckpt, device=args.device)"),
    # --- the curve: what the table claims --------------------------------------
    ("the curve ranks on the least likely option", RC,
     '        return max(float(p) for p in ans["probabilities"].values())',
     '        return min(float(p) for p in ans["probabilities"].values())'),
    ("an empty distribution ranks as zero confidence", RC,
     '    if "probabilities" in ans and ans["probabilities"]:',
     '    if "probabilities" in ans:'),
    ("noul ranked on p(Yes), so sure noes sink to the bottom", RC,
     "        return max(p, 1.0 - p)",
     "        return p"),
    ("noul's coin flip commits to no", RC,
     '        return int(float(ans["noul"]) >= 0.5) == int(gold)',
     '        return int(float(ans["noul"]) > 0.5) == int(gold)'),
    ("the curve ranks ascending: risk improves as coverage falls", RC,
     '    ranked = sorted(points, key=lambda p: -p["confidence"])',
     '    ranked = sorted(points, key=lambda p: p["confidence"])'),
    ("a rung below one row keeps none, and the cell averages nothing", RC,
     "        k = min(n, max(1, int(round(c * n)))) if n else 0",
     "        k = min(n, int(round(c * n))) if n else 0"),
    ("accuracy computed over the whole sample, not the kept prefix", RC,
     "        acc = sum(p['correct'] for p in kept) / len(kept) if kept else None".replace("'", '"'),
     "        acc = sum(p['correct'] for p in ranked) / n if kept else None".replace("'", '"')),
    ("the printed coverage is the rung that was asked for", RC,
     '        out.append({"rung": c, "coverage": len(kept) / n if n else 0.0, "n": len(kept),',
     '        out.append({"rung": c, "coverage": c if n else 0.0, "n": len(kept),'),
    ("the target becomes exclusive at exactly 0.95", RC,
     '    ok = [r for r in rows if r["accuracy"] is not None and r["accuracy"] >= target]',
     '    ok = [r for r in rows if r["accuracy"] is not None and r["accuracy"] > target]'),
    ("the coverage half of the gate is dropped", RC,
     '            "pass": bool(best and best["coverage"] >= min_cov),',
     '            "pass": bool(best),'),
    ("the floor's own row is excluded from the kept set", RC,
     "    return [p for p in points if not p[\"confidence\"] < floor]",
     '    return [p for p in points if p["confidence"] > floor]'),
    ("below-floor counts the kept rows instead", RC,
     "    n_below = len(points) - len(at_or_above(points, floor))",
     "    n_below = len(at_or_above(points, floor))"),
    ("the engine/curve agreement inverts", RC,
     '            "agree": engine_abstained == n_below}',
     '            "agree": engine_abstained != n_below}'),
    ("a question the engine skipped stops being an error", RC,
     "            if missing:",
     "            if False and missing:"),
    ("the curve stops checking the engine's own confidence field", RC,
     "                if ec is not None and abs(float(ec) - conf) > 1e-6:",
     "                if False:"),
    ("the gate's own coverage rung is not measured", RC,
     "COVERAGE_RUNGS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)",
     "COVERAGE_RUNGS = (1.0, 0.9, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05)"),
]


def make_scratch(tmp):
    repo = Path(tmp) / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir()
    for name in ("src", "bench", "tests"):
        shutil.copytree(ROOT / name, repo / name,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    shutil.copy(ROOT / "pyproject.toml", repo / "pyproject.toml")
    for name in ("data", "runs"):
        if (ROOT / name).exists():
            (repo / name).symlink_to(ROOT / name)
    return repo


def pytest_in(repo):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
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
    fails = [ln for ln in r.stdout.splitlines()
             if ln.startswith("FAILED") or ln.startswith("ERROR")]
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
        print(f"baseline copy: green ({len(TESTS)} test files)\n")
        for i, (label, rel, old, new) in enumerate(MUTATIONS, 1):
            verdict, detail = run_one(rel, old, new, tmp)
            print(f"[{i:2d}/{len(MUTATIONS)}] {verdict:11s} {label}"
                  + (f"\n            {detail}" if detail and verdict == "caught" else ""),
                  flush=True)
            if verdict == "SURVIVED":
                survivors.append(label)
                if i == 1:
                    print("  first mutation survived: the battery is not testing the "
                          "mutated code; aborting", flush=True)
                    return 2
            elif verdict.startswith("BAD-PATTERN"):
                bad.append((label, verdict))
        print(f"\n{len(MUTATIONS) - len(survivors) - len(bad)}/{len(MUTATIONS)} mutations caught")
        for s in survivors:
            print(f"  SURVIVED: {s}")
        for b in bad:
            print(f"  {b}")
        return 1 if survivors or bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

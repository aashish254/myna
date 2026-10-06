"""Mutation battery for `bench/site_witness.py` — the check on the check for the Pages site.

$ uv run python bench/mutation_site.py

The page makes one claim above all others: every printed figure was read out of a committed
artifact, and an independent script recomputes it. That claim is only worth what its failure
mode is worth, so this file breaks the page twenty times over and asks whether the witness
refuses it. Two families, because the witness has two things to be right about:

* **page mutations** — `docs/index.html` is copied, one lie is typed into the copy, and the
  real witness is run against it with `--html`. It must refuse. This is the class the page
  exists to catch: a swapped pair of arm macros, a floor that moved in the last digit, a
  verdict turned green, a rounded margin sold as the stored one.
* **source mutations** — the *artifacts* the witness reads are copied into a scratch tree and
  one of them is changed there, while the page stays honest. The witness must still refuse,
  because it noticed the file moved under it. A witness that only pattern-matches the page
  passes these, and that is the difference between a recomputation and a grep.

The scratch tree's unmutated baseline has to be green before any of this means anything
(§9.12: a mutated copy reached through the wrong path tests the code you did not change),
and the first survivor aborts the run for the same reason.

Nothing in this file is a published figure: the strings it mutates are read out of the same
artifacts the page reads them from, so a mutation targets whatever the current number is.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "docs" / "index.html"

# every file the witness opens, so a scratch tree can be a real second repo
NEEDED = ["pyproject.toml", "SPEC.md", "docs/index.html", "bench/site_witness.py",
          "bench/gates.py", "bench/reproduce.py", "src/myna/serve.py", "src/myna/engine.py",
          "runs/tier0_antiprior_off_s0.json", "runs/quickstart_from_gh_download.log",
          "runs/quickstart_from_git_install.log", "runs/quickstart_from_public_install.log"]
for _tag in ("antiprior_off_s0", "antiprior_on_s0c", "v1b_kaggle_3600b"):
    NEEDED += [f"runs/{_tag}.report.json", f"runs/{_tag}.metrics.json", f"runs/{_tag}.train.log"]


def lit(rel):
    return json.loads((REPO / rel).read_text())


def literals():
    """The page's own values, read from the artifacts rather than remembered by this file."""
    off = lit("runs/antiprior_off_s0.report.json")
    on = lit("runs/antiprior_on_s0c.report.json")
    v1b = lit("runs/v1b_kaggle_3600b.report.json")
    t0 = lit("runs/tier0_antiprior_off_s0.json")
    metrics = lit("runs/antiprior_off_s0.metrics.json")
    py = (REPO / "pyproject.toml").read_text()
    engine = (REPO / "src/myna/engine.py").read_text()
    none_dict = re.search(r'return \{"abstain": False, "confidence": None, '
                          r'"margin": None, "reason": None\}', engine).group(0)
    line = engine[:engine.index(none_dict)].count("\n") + 1
    sys.path.insert(0, str(REPO / "bench"))
    import reproduce  # the registry's own command lines, not a paraphrase of them
    cmd = next(r["cmd"] for r in reproduce.ROWS if r["id"] == "antiprior-off-macro")
    # the footer's date is the newest commit among the figure sources; this asks git the same
    # question the witness asks, so the mutation targets the current stamp whatever it is
    stamp = subprocess.run(
        ["git", "-C", str(REPO), "log", "-1", "--format=%cd", "--date=format:%Y-%m-%d", "--",
         "runs/antiprior_off_s0.report.json", "runs/antiprior_on_s0c.report.json",
         "runs/v1b_kaggle_3600b.report.json", "runs/tier0_antiprior_off_s0.json",
         "runs/quickstart_from_gh_download.log", "runs/quickstart_from_git_install.log",
         "runs/quickstart_from_public_install.log"],
        capture_output=True, text=True).stdout.strip()
    if not stamp:
        raise SystemExit("git printed no stamp for the figure sources; the battery has nothing to mutate")
    return {
        "off": repr(off["macro"]["acc"]), "on": repr(on["macro"]["acc"]),
        "v1b": repr(v1b["macro"]["acc"]),
        "floor": repr(off["macro"]["majority"]), "floor_short": f"{off['macro']['majority']:.4f}",
        "margin": repr(off["g1"]["margin_observed"]),
        "margin_short": f"{off['g1']['margin_observed']:.4f}",
        "sha": t0["model"]["model_sha256"],
        "params": f"{t0['model']['params']:,}",
        # the page prints updates executed, i.e. one past the metrics' `last_step` index
        "executed": f"{int(metrics['last_step']) + 1:,}",
        "version": re.search(r'^version = "(.*)"', py, re.M).group(1),
        "dist": re.search(r'^name = "(.*)"', py, re.M).group(1),
        "engine_ref": f"src/myna/engine.py:{line}",
        "cmd": cmd,
        "confidence": re.search(r"^confidence (\S+) weakest",
                                (REPO / "runs/quickstart_from_public_install.log")
                                .read_text(), re.M).group(1),
        "stamp": stamp,
        "stamp_line": f"Figures as of {stamp}",
    }


L = literals()

# ------------------------------------------------------------------ page mutations
# (label, [(old, new), ...]) applied to a copy of docs/index.html
PAGE_MUTATIONS = [
    ("the two anti-prior arms' macros swap places",
     [(L["off"], "@@SWAP@@"), (L["on"], L["off"]), ("@@SWAP@@", L["on"])]),
    ("the Kaggle arm prints the ON arm's macro", [(L["v1b"], L["on"])]),
    ("the majority floor shifts in the last printed digit",
     [(f"floor {L['floor_short']}", "floor 0.4330")]),
    ("the G1 target tile trades its honesty label for `measured`",
     [('G1 target <span class="tag notmet">gated — not met</span>',
       'G1 target <span class="tag measured">measured</span>')]),
    ("the G1 verdict flips to met on the page",
     [('<td>Accuracy beats the witness</td>\n        <td class="verdict">'
       '<span class="tag notmet">not met</span></td>',
       '<td>Accuracy beats the witness</td>\n        <td class="verdict">'
       '<span class="tag measured">met</span></td>')]),
    ("the truncated arm stops being called VOID",
     [('<span class="arm-state void">VOID — truncated dose, no gate judged from it (§7.1)</span>',
       '<span class="arm-state done">run completed</span>')]),
    ("the complete Kaggle arm is labelled VOID",
     [('<span class="arm-state done">run completed</span> ·\n            3,600 updates executed',
       '<span class="arm-state void">VOID — truncated dose, no gate judged from it (§7.1)</span> ·\n'
       '            3,600 updates executed')]),
    ("a dose digit moves by one",
     [(f"{L['executed']} updates executed", "3,316 updates executed")]),
    ("the dose-matched pair stops being dose-matched",
     [("3,315 updates executed, 3,315 requested", "3,315 updates executed, 3,600 requested")]),
    ("a parameter count digit changes", [(L["params"], "16,926,849")]),
    ("an invented figure is injected into the copy",
     [('<section id="gates">', '<p class="note">0.62 on the held-out set.</p>\n'
       '<section id="gates">')]),
    ("the off arm's source line is deleted",
     [('            source <span translate="no">runs/antiprior_off_s0.report.json · macro.acc'
       '</span></div>', '</div>')]),
    ("a stored margin is replaced by its four-decimal rounding",
     [(f'+{L["margin"]}', f'+{L["margin_short"]}')]),
    ("the checkpoint's sha256 is retyped", [(L["sha"], "0" * len(L["sha"]))]),
    ("a session route drops off the list",
     [('<li class="mono">DELETE /v1/sessions/{id}</li>', "")]),
    ("the abstention citation points a line away",
     [(L["engine_ref"], L["engine_ref"] + "9")]),
    ("the pip line publishes a version that does not exist",
     [(f"pip install {L['dist']}=={L['version']}", f"pip install {L['dist']}==0.9.0")]),
    ("a registry command loses a flag", [(L["cmd"], L["cmd"].replace("--split test ", ""))]),
    ("the quickstart card's confidence is rounded",
     [(L["confidence"], "0.79")]),
    ("the footer's date is moved to a day no figure source was committed on",
     [(L["stamp_line"], "Figures as of 2027-01-01")]),
]

# ------------------------------------------------------------------ source mutations
# (label, repo-relative file, old, new): the page is honest, the artifact moved under it
SRC_MUTATIONS = [
    ("gates.py stops saying G1 is not met", "bench/gates.py",
     """    gate("G1", "Accuracy beats the witness",
         "macro decision-v2 test ≥ 0.70 on the frozen split, per-source table published, "
         "≥ +0.15 over the 0.4331 majority floor",
         NOT_MET,""",
     """    gate("G1", "Accuracy beats the witness",
         "macro decision-v2 test ≥ 0.70 on the frozen split, per-source table published, "
         "≥ +0.15 over the 0.4331 majority floor",
         MET,"""),
    ("the OFF report's macro gains a digit", "runs/antiprior_off_s0.report.json",
     f'"acc": {L["off"]}', '"acc": 0.5438735348381732'),
    ("the OFF metrics stop being truncated", "runs/antiprior_off_s0.metrics.json",
     '"last_step": 3314', '"last_step": 3315'),
    ("pyproject's version moves ahead of the published wheel", "pyproject.toml",
     f'version = "{L["version"]}"', 'version = "0.9.0"'),
    ("engine.py returns a confidence where the page quotes a None",
     "src/myna/engine.py", '"confidence": None, "margin": None, "reason": None}',
     '"confidence": 0.5, "margin": None, "reason": None}'),
    ("serve.py renames a route the page lists", "src/myna/serve.py",
     "/v1/sessions/{id}/append", "/v1/sessions/{id}/extend"),
    ("the checkpoint's parameter count changes in the tier0 artifact",
     "runs/tier0_antiprior_off_s0.json", f'"params": {L["params"].replace(",", "")}',
     '"params": 16926849'),
    ("the registry's command for the OFF arm is rewritten", "bench/reproduce.py",
     '"--metrics runs/antiprior_off_s0.metrics.json "',
     '"--metrics runs/antiprior_off_s0.metricsX.json "'),
    ("the quickstart log's confidence is retyped", "runs/quickstart_from_public_install.log",
     f"confidence {L['confidence']}", "confidence 0.79"),
    ("SPEC stops carrying the section the page's VOID label cites", "SPEC.md",
     "§7.1", "§7.2"),
]


def run_witness(html, script=None):
    argv = [sys.executable, str(script or (REPO / "bench" / "site_witness.py"))]
    if html is not None:
        argv += ["--html", str(html)]
    env = os.environ.copy()
    # the scratch tree has no git of its own; point it at this repo's so the footer's
    # committed-date check asks the same question it asks in a real checkout
    env["GIT_DIR"] = str(REPO / ".git")
    r = subprocess.run(argv, capture_output=True, text=True, cwd=REPO, env=env, timeout=300)
    return r.returncode, (r.stdout + r.stderr).strip().splitlines()


def make_scratch(tmp):
    root = Path(tmp) / "repo"
    if root.exists():
        shutil.rmtree(root)
    for rel in NEEDED:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, dst)
    return root


def apply(path, pairs):
    text = path.read_text()
    for old, new in pairs if isinstance(pairs, list) else [pairs]:
        if text.count(old) < 1:
            return f"BAD-PATTERN ({old[:48]!r} is not in the file)"
        text = text.replace(old, new)
    path.write_text(text)
    return None


def run_page_mutation(label, pairs, tmp):
    copy = Path(tmp) / "page.html"
    copy.write_text(PAGE.read_text())
    if (bad := apply(copy, pairs)):
        return bad, ""
    rc, out = run_witness(copy)
    if rc == 0:
        return "SURVIVED", ""
    return "caught", next((ln.strip() for ln in out if "FAIL" in ln), "")[:110]


def run_src_mutation(label, rel, old, new, tmp):
    root = make_scratch(tmp)
    if (bad := apply(root / rel, (old, new))):
        return bad, ""
    rc, out = run_witness(None, script=root / "bench" / "site_witness.py")
    if rc == 0:
        return "SURVIVED", ""
    return "caught", next((ln.strip() for ln in out if "FAIL" in ln), "")[:110]


def main():
    ap = argparse.ArgumentParser(
        description="Break the Pages site, once per mutation, and check the witness refuses it.",
        epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=("page", "src"), help="run one family only")
    args = ap.parse_args()

    survivors, bad = [], []
    with tempfile.TemporaryDirectory() as tmp:
        # control 1: the real page passes the real witness, or nothing below means anything
        rc, out = run_witness(None)
        if rc != 0:
            print("the committed page does not pass the witness: " + " | ".join(out[-3:]))
            return 2
        print("control: docs/index.html green against bench/site_witness.py")

        # control 2 (§9.12): the scratch tree, unmutated, must also be green
        root = make_scratch(tmp)
        rc, out = run_witness(None, script=root / "bench" / "site_witness.py")
        if rc != 0:
            print("scratch copy is RED before any mutation -- the battery would prove nothing: "
                  + " | ".join(out[-3:]))
            return 2
        print(f"control: {len(NEEDED)}-file scratch tree green, so a source mutation is "
              "caught by the file it changed\n")

        cases = []
        if args.family != "src":
            cases += [("page", m) for m in PAGE_MUTATIONS]
        if args.family != "page":
            cases += [("src", m) for m in SRC_MUTATIONS]
        total = len(cases)
        for i, (family, case) in enumerate(cases, 1):
            if family == "page":
                label, pairs = case
                verdict, detail = run_page_mutation(label, pairs, tmp)
            else:
                label, rel, old, new = case
                verdict, detail = run_src_mutation(label, rel, old, new, tmp)
            print(f"[{i:2d}/{total}] {verdict:11s} {family}: {label}"
                  + (f"\n            {detail}" if detail else ""), flush=True)
            if verdict == "SURVIVED":
                survivors.append(f"{family}: {label}")
                if i == 1:
                    print("  first mutation survived: the witness is not reading what it "
                          "claims to; aborting", flush=True)
                    return 2
            elif verdict.startswith("BAD-PATTERN"):
                bad.append((label, verdict))

        print(f"\n{total - len(survivors) - len(bad)}/{total} mutations caught")
        for s in survivors:
            print(f"  SURVIVED: {s}")
        for b in bad:
            print(f"  {b}")
        return 1 if survivors or bad else 0


if __name__ == "__main__":
    sys.exit(main())

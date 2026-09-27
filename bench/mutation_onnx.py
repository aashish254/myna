"""Mutation battery for the ONNX-parity gate G3 rests on (SPEC §5 P6 6a, §7.1).

    python bench/mutation_onnx.py

`myna.onnx_export` is the piece of this project whose whole job is to *not be
fooled by itself*: it writes graphs and then prints a number saying the graphs agree
with torch. A parity harness that always agrees is worse than no parity harness,
because it converts an unverified artifact into a certified one.

The mutations cluster in five places where a lie fits:

* **the mask.** Both graphs are fixed-width, so everything is padded, and a padded
  token that contributes to the memory is a wrong answer that only appears on
  documents which do not divide evenly — i.e. almost all of them.
* **the accounting.** The chained loop must consume the document once, carry the
  state forward, and count the weights it actually wrote (`dynamo` stores initializers
  in a sibling `*.onnx.data`, so a graph listing alone reports a few hundred KiB for a
  36 MB model — and once the two graphs share one file, the listing can leave *that*
  file out, which is the same lie in a new shape).
* **the sharing.** 6b's whole claim is that the trunk crosses the network once. A
  dedup keyed on the wrong thing (names, which `dynamo` numbers per graph), a rewrite
  that leaves the per-graph copies behind on disk, or an initializer that keeps its
  inline bytes next to an external pointer all leave an artifact that loads, answers,
  and is twice the size it reports.
* **the tab's memory.** The scan tile is a separate knob from the graph width, and its
  reported peak is the only number that says whether a request survives in a browser.
* **the two bounds.** The state's entries reach ~4e2, so its error is gated relatively
  while probabilities are gated absolutely. Either half can be dropped, and dropping
  one is exactly how a gate becomes decorative.

Also absent, and named rather than padded over: the widest-real-request block needs the
`decision-v2` pilot corpus, which is gitignored by design (only its manifest is committed), so
no committed test reaches that branch and a mutation there would be caught by nothing. It is
covered by the CLI run in `runs/onnx_parity.json`, which is a witness for one machine rather
than a gate.

Deliberately absent: nothing mutates `isolation_max_abs_error`. What that line measures
is that a masked neighbour cannot move a real row's hidden states, and rows are separate
along the batch axis by construction — so it cannot be broken by any single-token edit,
and a mutation that fakes its value would be caught only by an assertion written to catch
that mutation and nothing else. It stays in the report as what it is: a mask-inertness
check, cheap to run, and not the branch-isolation property (which
`tests/test_engine_parity.py` owns).

Same scratch-repo mechanics as the other batteries: copy `src` + `bench` + `tests`
(pyproject's `pythonpath = ["src"]` outranks PYTHONPATH below the rootdir, §9.12),
green baseline required, abort on a first-mutation survivor, and any pattern that does
not match exactly once is reported instead of skipped.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
TESTS = ["tests/test_onnx_export.py"]
OX = "src/myna/onnx_export.py"

MUTATIONS = [
    # --- the mask: padding must be inert in both graphs -------------------------
    ("the state graph ignores its mask, so padded tokens decay the memory", OX,
     "        m = mask[:, None, :, None]\n        h = self.trunk.tok(ids)",
     "        h = self.trunk.tok(ids)"),
    ("the chain marks every padded position as a real token", OX,
     "        m = np.zeros((1, chunk), dtype=\"float32\")\n        m[0, :len(take)] = 1.0",
     "        m = np.ones((1, chunk), dtype=\"float32\")"),
    ("the padded question ids are left as unmasked garbage", OX,
     "        mk = np.zeros((N, q_len), dtype=\"float32\")\n        mk[0, :Lq] = 1.0",
     "        mk = np.ones((N, q_len), dtype=\"float32\")"),
    # --- the accounting: the loop and the bytes ---------------------------------
    ("the chain over-reads the document, advancing by the padded width", OX,
     "        carried = [y[i] for i in range(L)]\n        fed += len(take)",
     "        carried = [y[i] for i in range(L)]\n        fed += chunk"),
    ("the chain throws away the state between chunks", OX,
     "        carried = [y[i] for i in range(L)]\n        fed += len(take)",
     "        carried = [np.zeros_like(y[i]) for i in range(L)]\n        fed += len(take)"),
    ("the byte budget totals only the graph skeletons, not the weight files", OX,
     '            "artifact_bytes_total": sum(sizes.values())}',
     '            "artifact_bytes_total": sum(v for k, v in sizes.items() if k.endswith(".onnx"))}'),
    ("the state geometry claims one layer's worth for the whole stack", OX,
     '"state_bytes_per_layer": per_layer, "state_bytes_total": per_layer * L,',
     '"state_bytes_per_layer": per_layer, "state_bytes_total": per_layer,'),
    ("the exporter flag is a label instead of the thing that ran", OX,
     '    kw = {"dynamo": True, "opset_version": opset}',
     '    kw = {"dynamo": False, "opset_version": opset}'),
    # --- the sharing: one copy of the trunk, and the bytes to prove it -------------
    ("the dedup key is the graph's own name, so both graphs write their own copy", OX,
     "            key = hashlib.sha256(raw).digest()",
     "            key = t.name.encode()"),
    ("the dedup never fires: every reference gets a fresh region", OX,
     "            got = placed.get(key)",
     "            got = None"),
    ("every initializer is pushed to the external file, shape-inference constants too", OX,
     "            if len(raw) < min_bytes:",
     "            if False:"),
    ("the external pointer keeps its inline bytes, so `onnx.save` rewrites the shared file", OX,
     '            nt.ClearField("raw_data")\n            new_inits.append(nt)',
     "            new_inits.append(nt)"),
    ("the per-graph `.onnx.data` copies survive the rewrite", OX,
     '    for f in out.glob("*.onnx.data"):\n        f.unlink()',
     '    for f in out.glob("*.onnx.data"):\n        continue'),
    # --- the head: the artifact has to answer, not just load -----------------------
    ("the head layout records where the blob started instead of where each field is", OX,
     '        layout[name] = {"shape": list(arr.shape), "offset": len(blob),',
     '        layout[name] = {"shape": list(arr.shape), "offset": 0,'),
    ("the shipped head's option pooling stops being a mean", OX,
     "    pooled = np.stack([h[s:e].mean(0) for s, e in spans]).astype(np.float32)",
     "    pooled = np.stack([h[s:e].max(0) for s, e in spans]).astype(np.float32)"),
    # --- the tile: the knob the tab's memory is decided by -------------------------
    ("the tile knob is read for the metadata but not for the graph", OX,
     "    scan = scan_chunk or min(chunk, q_len)",
     "    scan = 32"),
    ("a tile wider than the graph it feeds is accepted", OX,
     "    if not 1 <= scan <= min(chunk, q_len):",
     "    if not 1 <= scan:"),
    ("the reported peak tile counts one row whatever the request carries", OX,
     "                n_questions * cfg.n_heads * scan * scan * cfg.d_k * 4),",
     "                cfg.n_heads * scan * scan * cfg.d_k * 4),"),
    # --- the bounds -------------------------------------------------------------
    ("the relative half of the gate is dropped", OX,
     '           "pass": (worst_abs <= max_abs_error\n'
     '                    and max(chain_rel, q_rel) <= max_rel_error),',
     '           "pass": worst_abs <= max_abs_error,'),
    ("the absolute half of the gate is dropped", OX,
     '           "pass": (worst_abs <= max_abs_error\n'
     '                    and max(chain_rel, q_rel) <= max_rel_error),',
     '           "pass": max(chain_rel, q_rel) <= max_rel_error,'),
    ("relative error is reported as the absolute error", OX,
     "    return abs_err / max(scale, 1e-12)",
     "    return abs_err"),
    ("a request too wide for the graph passes on zero comparisons", OX,
     "    if n_compared == 0:",
     "    if False:"),
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


TEARDOWN_CRASH = "recursive_mutex lock failed"


def pytest_in(repo):
    """Run the gate, and separate a failing test from a crashing interpreter.

    onnxruntime's session pool and torch's export state are torn down after Python has
    started exiting, and that race aborts the process *after* the summary line has been
    printed — intermittently, so a run of eight passing tests can exit 134. Reading only
    the exit code makes the battery refuse to start on its own baseline, and reading only
    "passed" would forgive a real failure, so the two are cross-checked: the summary must
    say every test passed and named none failed or errored, and the crash line must be
    the known teardown one. Anything else is red, and the note is printed rather than
    swallowed.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-q", "--no-header",
                        "-p", "no:cacheprovider"],
                       capture_output=True, text=True, cwd=repo, env=env, timeout=1800)
    if r.returncode == 0:
        return r
    out = r.stdout
    summary_ok = ("passed" in out and " failed" not in out and " error" not in out
                  and "no tests ran" not in out)
    if summary_ok and (TEARDOWN_CRASH in r.stderr or TEARDOWN_CRASH in out):
        print(f"  note: interpreter aborted after a green summary ({TEARDOWN_CRASH}); "
              "treated as green, see the docstring")
        r.returncode = 0
    return r


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
        print(f"baseline copy: green ({len(TESTS)} test file)\n")
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

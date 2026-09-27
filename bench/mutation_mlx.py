"""Mutation battery for the MLX int8 gate (SPEC §5 P6 6c, §7.1).

    python bench/mutation_mlx.py

`myna.mlx_model`'s int8 path has one job and one way to fail it: it has to be the
same model in fewer bytes, and every interesting failure mode looks like a
working engine. A filter that matches nothing still forwards; a group size that is
a label rather than a fact still loads; a `meta.json` that misreports the file it
sits beside still answers. The tests catch those, and this battery checks that the
tests do.

Clusters:

* **the filter.** Which weights become int8 is a claim about the artifact — the
  embedding is gathered rather than matmulled, and the count has to say what
  actually happened.
* **the dispatch.** `mx.quantized_matmul` has to be reached, with the orientation,
  group size and bit depth the weights were packed with. Each of those four can be
  wrong while the engine loads, runs and prints an int8 label.
* **the artifact.** `meta.json` is the only place group size, bits and the byte
  figure live; if the load trusts any of them wrongly, the deployment answers
  differently from the engine that saved it.
* **the two byte figures.** One is the file, one is the sum of tensors. Quoting
  whichever is smaller, or the wrong one, is how a size claim becomes fiction.

Also absent, and named rather than padded over: nothing here mutates
`bench/bench_mlx.py`. Its drift table is a witness printed on one machine — the
comparisons it makes are not asserted by any committed test — so a mutation there
would be caught only by rerunning the command and reading the number, which is the
same limitation `bench/mutation_onnx.py` records for its widest-request block.

Same scratch-repo mechanics as the other batteries: copy `src` + `bench` + `tests`,
symlink `data` and `runs`, green baseline required, abort on a first-mutation
survivor, and any pattern that does not match exactly once is reported instead of
skipped. MLX is pinned to the CPU stream by the test module's own fixture, so the
battery does not touch the GPU.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
TESTS = ["tests/test_mlx_int8.py", "tests/test_mlx_parity.py"]
MM = "src/myna/mlx_model.py"

MUTATIONS = [
    # --- the filter: which weights are int8 is a claim about the artifact --------
    ("nothing is quantised, and the engine still forwards in fp32 behind an int8 label", MM,
     "                           if k.endswith(\".weight\") and v.ndim == 2\n"
     "                           and not any(s in k for s in keep)):",
     "                           if False):"),
    ("the keep list is ignored, so the gathered token embedding is packed too", MM,
     "                           and not any(s in k for s in keep)):",
     "                           and True):"),
    ("the count recorded in the metadata is the number of groups, not of weights", MM,
     "                      \"keep\": sorted(keep), \"n_quantized\": n}",
     "                      \"keep\": sorted(keep), \"n_quantized\": 0}"),
    # --- the dispatch: the quantised op has to be reached, and correctly ----------
    ("`_linear` never looks for quantised weights, so the int8 path is dead code", MM,
     '        q = self.p.get(f"{scope}.weight_q")',
     "        q = None"),
    ("the quantised matmul reads the weight in the wrong orientation", MM,
     "                                    self.p[f\"{scope}.weight_biases\"], transpose=True,",
     "                                    self.p[f\"{scope}.weight_biases\"], transpose=False,"),
    ("the group size is a constant instead of what the weights were packed with", MM,
     "                                    group_size=self.quant[\"group_size\"], bits=self.quant[\"bits\"])",
     "                                    group_size=64, bits=self.quant[\"bits\"])"),
    ("the bit depth is a constant instead of what the weights were packed with", MM,
     "                                    group_size=self.quant[\"group_size\"], bits=self.quant[\"bits\"])",
     "                                    group_size=self.quant[\"group_size\"], bits=4)"),
    ("the fp32 branch transposes the wrong factor", MM,
     "            y = x @ self.p[f\"{scope}.weight\"].T",
     "            y = x @ self.p[f\"{scope}.weight\"]"),
    # --- the guard: a weight that cannot be packed must stop the run --------------
    ("a group size that does not divide the input dimension is accepted anyway", MM,
     "            if w.shape[-1] % group_size:",
     "            if False:"),
    # --- the artifact: meta.json is what the loader believes ----------------------
    ("the metadata stops recording the quantisation, so the load unpacks as fp32", MM,
     '        meta = {"cfg": dict(self.cfg), "quantization": self.quant,',
     '        meta = {"cfg": dict(self.cfg), "quantization": None,'),
    ("the loader assumes a group size and bit depth instead of reading the metadata", MM,
     "        return cls(meta[\"cfg\"], p, quant=q)",
     '        return cls(meta["cfg"], p, quant={"group_size": 64, "bits": 8})'),
    ("a declared bit depth MLX cannot implement is loaded instead of refused", MM,
     "        if q is not None and q.get(\"bits\") not in (4, 8):",
     "        if False:"),
    # --- the two byte figures -----------------------------------------------------
    ("the metadata quotes the tensor sum as the file size", MM,
     '                "param_bytes_file": (d / "params.safetensors").stat().st_size,',
     '                "param_bytes_file": self.param_bytes(),'),
    ("the metadata's tensor figure is the file size, so both columns are one number", MM,
     '                "param_bytes_tensors": self.param_bytes()}',
     '                "param_bytes_tensors": (d / "params.safetensors").stat().st_size}'),
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

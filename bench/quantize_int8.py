"""G3's other half of the byte budget: quantise the exported artifact and report what is left.

    python bench/quantize_int8.py [--src runs/onnx] [--out runs/onnx_int8]
                                  [--weight-type quint8] [--no-gate]

6a closed with "93.7 MiB fp32 is the trunk twice, so ≤ 20 MB needs shared weights, not
quantisation" (§9.26). 6b then asks for the download bytes **fp32 and int8**, and sharing
alone lands at 59.32 MiB — under 6a's number, still three times G3's target. So this takes
the artifact the exporter wrote (not a fresh torch model, so the graphs measured by
`bench/mutation_onnx.py` are the thing being shrunk) and runs onnxruntime's dynamic
quantisation over it, then re-shares the result through the same `share_weights` the fp32
export uses, so the int8 row is comparable to the fp32 row file by file.

The output is deliberately not just a size. A quantised graph the browser cannot execute is
not a smaller artifact, it is a different failure, so the last thing this does is run the
real gate — `node browser/selftest.mjs --artifact <out>` — and record what it printed. That
gate compares against `browser/expected.json`, which was written by the torch engine at the
fp32 tolerances, so a red parity line here is *information about the loss*, not a broken
build: the FAIL lines carry the measured worst |Δp|, which is the number a reader needs to
judge whether int8 would ever be shippable. If onnxruntime-web refuses the ops outright, the
refusal is captured verbatim instead of being described from documentation.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from myna.onnx_export import GRAPHS, WEIGHTS, share_weights  # noqa: E402

WEIGHT_TYPES = {"quint8": "QUInt8", "qint8": "QInt8", "qint4": "QInt4"}


def op_histogram(path: Path) -> dict[str, int]:
    m = onnx.load(path, load_external_data=False)
    return dict(collections.Counter(n.op_type for n in m.graph.node))


def quantize(src: Path, out: Path, weight_type: str) -> tuple[dict, dict]:
    """Dynamic-quantise both graphs into `out`, then share their weights once."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out.mkdir(parents=True, exist_ok=True)
    qt = getattr(QuantType, WEIGHT_TYPES[weight_type])
    before = {}
    with tempfile.TemporaryDirectory() as tmp:
        for g in GRAPHS:
            # The exporter writes initializers as external data; the quantizer wants a
            # self-contained model, so load the bytes in, quantise, and let the sharing
            # step below put them back out — in one file, for both graphs, again.
            model = onnx.load(src / g)
            before[g] = op_histogram(src / g)
            step = Path(tmp) / g
            quantize_dynamic(model, step, weight_type=qt)
            shutil.copy(step, out / g)
        shared = share_weights(out, GRAPHS)
    for f in ("head.bin", "tokenizer.json"):
        shutil.copy(src / f, out / f)
    return before, shared


def write_meta(src: Path, out: Path, weight_type: str, shared: dict,
               before: dict, gate: dict) -> dict:
    """Write the quantised artifact's meta.json.

    Called twice when the gate runs: once before it (the gate loads this file, so it has
    to exist to be gated at all) and once after, to record the verdict it produced.
    """
    base = json.loads((src / "meta.json").read_text())
    sizes = {f.name: f.stat().st_size for f in out.iterdir()
             if f.is_file() and f.name != "meta.json"}
    raw = len(json.dumps(base, indent=2).encode())  # close enough for a 2 KiB file
    meta = {**base,
            "quantization": {
                "scheme": f"onnxruntime dynamic ({weight_type} weights)",
                "weight_type": weight_type,
                "tool": "onnxruntime.quantization.quantize_dynamic",
                "ops_before": before,
                "ops_after": {g: op_histogram(out / g) for g in GRAPHS},
                "fp32_weights_bytes": json.loads((src / "meta.json").read_text())[
                    "weight_sharing"]["weights_bytes"],
                "quant_weights_bytes": shared["weights_bytes"],
                "gate": gate,
            },
            "files_mib": {f: round(b / 2**20, 2) for f, b in sorted(sizes.items())},
            "weight_sharing": shared,
            "artifact_bytes_total": sum(sizes.values()),
            "transfer_mib": round((sum(sizes.values()) + raw) / 2**20, 2)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def run_gate(out: Path) -> dict:
    """The real node gate against the quantised artifact, captured rather than described."""
    node = shutil.which("node")
    if not node:
        return {"ran": False, "reason": "node not on PATH"}
    r = subprocess.run([node, "browser/selftest.mjs", "--artifact", str(out),
                        "--repeats", "2"], capture_output=True, text=True,
                       cwd=str(Path(__file__).resolve().parent.parent), timeout=600)
    lines = (r.stdout + r.stderr).splitlines()
    return {"ran": True, "returncode": r.returncode,
            "checks_passed": sum(1 for ln in lines if ln.startswith("ok")),
            "fails": [ln.strip() for ln in lines if ln.startswith("FAIL")],
            "last_lines": lines[-6:]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="runs/onnx")
    ap.add_argument("--out", default="runs/onnx_int8")
    ap.add_argument("--weight-type", default="quint8", choices=sorted(WEIGHT_TYPES))
    ap.add_argument("--no-gate", action="store_true",
                    help="quantise and size only; do not run the node gate against it")
    args = ap.parse_args(argv)

    src, out = Path(args.src), Path(args.out)
    if not (src / "meta.json").exists():
        raise SystemExit(f"{src} is not an exported artifact (no meta.json)")
    before, shared = quantize(src, out, args.weight_type)
    write_meta(src, out, args.weight_type, shared, before, {"ran": False, "reason": "pending"})
    gate = {"ran": False, "reason": "--no-gate"} if args.no_gate else run_gate(out)
    meta = write_meta(src, out, args.weight_type, shared, before, gate)

    fp32 = json.loads((src / "meta.json").read_text())
    mib = lambda b: b / 2**20  # noqa: E731
    print(f"{out}: dynamic int8 over the exported graphs")
    for f, b in sorted(meta["files_mib"].items()):
        old = fp32["files_mib"].get(f)
        print(f"  {f:16s} {b:7.2f} MiB" + (f"   (fp32 {old:.2f})" if old else ""))
    print(f"  {'artifact':16s} {mib(meta['artifact_bytes_total']):7.2f} MiB"
          f"   (fp32 {mib(fp32['artifact_bytes_total']):.2f})")
    print(f"weights: {mib(meta['quantization']['fp32_weights_bytes']):.2f} MiB fp32 -> "
          f"{mib(meta['quantization']['quant_weights_bytes']):.2f} MiB with "
          f"{args.weight_type} ({shared['unique_regions']} regions, "
          f"{shared['shared_regions']} read by both graphs)")
    added = sorted(set(meta["quantization"]["ops_after"]["state_step.onnx"])
                   - set(before["state_step.onnx"]))
    print(f"new ops the browser's runtime must implement: {', '.join(added) or 'none'}")
    if gate.get("ran"):
        verdict = "green" if gate["returncode"] == 0 else "RED"
        print(f"node browser/selftest.mjs --artifact {out}: {verdict} "
              f"({gate['checks_passed']} checks ok)")
        for ln in gate["fails"][:4]:
            print(f"  {ln}")
    else:
        print(f"gate: not run ({gate['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""V1-C latency benchmark: MLX (Metal) vs PyTorch (MPS) full-predict path,
and the int8 MLX serving path beside them (P6 6c).

The delta-streaming claim is already carried by bench_stream (torch). This one
measures the *serving* payoff of the MLX port: the same checkpoint, the same
questions, a state of matched token length, timed end-to-end on the inference
stacks. Lower is better. The int8 column answers the other half of 6c — what it
costs in bytes, in milliseconds, and in agreement with the fp32 engine — and the
drift it prints is measured against the real decision-v2 questions, because a
quantisation that only ever met itself on a toy question proves nothing.

Usage: uv run python -m bench.bench_mlx --ckpt runs/myna-v0 [--reps 20]
       uv run python -m bench.bench_mlx --ckpt runs/myna-v0 --reps 20 \
           --quantize int8 --out runs/bench_mlx_int8.md \
           --out-json runs/bench_mlx_int8.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from myna.engine import Myna
from myna.mlx_model import MynaMLX
from myna.tokenizer import Tokenizer, batch_question_tensors, encode_text

QUESTIONS = [
    ("Which team should this route to?", ["billing", "technical support", "sales", "abuse"]),
    ("Is this urgent?", ["No", "Yes"]),
]
LENS = [128, 512, 1024, 2048, 4096, 8192, 16384]


def build_state(tok, target):
    seed = encode_text(tok, "customer reports a recurring payment failure on the pro plan since monday")
    ids = []
    while len(ids) < target:
        ids += seed
    return ids[:target]


def probs_of(logits, opt_valid, temperature):
    """Softmax the way the engine does: invalid options pinned to -1e9, then the
    refit temperature. logits [N, O], opt_valid [N, O]."""
    x = np.where(opt_valid, logits, -1e9).astype(np.float64) / temperature
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def scalar_of(p, valid, qtype):
    """The number each question type actually publishes, straight off the option
    distribution: the engine's `confidence` for choice, its `score` (an
    index-weighted mean over the ordinal legend) for score, its `noul` = p(Yes)."""
    row = p[valid]
    if row.size == 0:
        return None
    if qtype == "noul":
        return float(row[1]) if row.size > 1 else float(row[0])
    if qtype == "score":
        return float(sum(k * float(v) for k, v in enumerate(row)))
    return float(row.max())


def drift_vs_reference(myna, engines, groups, tok, dev, n_rows):
    """Answer the suite's own questions with every engine and compare.

    One batch per group, B=1, at the group's real state and real question width —
    no padding to a fixed shape, which is what makes this a different witness from
    the latency ladder above. Returns (rows, summary).
    """
    keys = sorted(groups)
    stride = max(1, len(keys) // max(1, n_rows))
    # A strided sample of sorted keys is a sample of whatever sorts first, and the
    # calibration groups are short documents. The two rows that stress the arithmetic
    # differently are therefore added by name: the longest state (most scan steps for
    # a weight error to ride through) and the widest question (a softmax over 77
    # options, not 4) — the same pair the ONNX parity harness had to add explicitly.
    widths = {k: max(len(myna._options({"type": q.type, "criteria": q.options}))
                     for q in groups[k][0]) for k in keys}
    lengths = {k: len(encode_text(tok, groups[k][1][0].state)) for k in keys}
    by_len = max(keys, key=lambda k: lengths[k])
    by_opt = max(keys, key=lambda k: widths[k])
    picked = sorted(set(keys[::stride][:n_rows]) | {by_len, by_opt})
    out, flips = [], {}
    for key in picked:
        qs, exs = groups[key]
        source = key.split("#")[0]
        reason = [r for r, k in (("longest-state", by_len), ("widest-question", by_opt)) if k == key]
        row_q = [(q.instruction, myna._options({"type": q.type, "criteria": q.options}))
                 for q in qs]
        qt = batch_question_tensors(tok, [row_q])
        ids = encode_text(tok, exs[0].state)
        valid = qt["opt_valid"][0].numpy().astype(bool)
        tensors = [qt[k].to(dev) for k in ("q_ids", "q_mask", "span_mat",
                                           "opt_valid", "decide_idx")]
        with torch.no_grad():
            ref = myna.model(torch.tensor([ids], dtype=torch.int64, device=dev),
                              torch.tensor([len(ids)], dtype=torch.int64, device=dev),
                              *tensors).float().cpu().numpy()[0]
        ref_p = probs_of(ref, valid, myna.temperature)
        margs = (np.array([ids], dtype=np.int64), np.array([len(ids)], dtype=np.int64),
                 qt["q_ids"][0].numpy(), qt["q_mask"][0].numpy(),
                 qt["span_mat"][0].numpy(), qt["opt_valid"][0].numpy(),
                 qt["decide_idx"][0].numpy())
        rec = {"source": source, "state_tokens": len(ids),
               "questions": len(qs), "max_options": int(valid.sum(-1).max()),
               "why": reason or ["strided"]}
        probs = {name: probs_of(np.array(eng.forward(*margs))[0], valid, myna.temperature)
                 for name, eng in engines.items()}
        # Two yardsticks, because they answer different questions. Against the torch
        # engine is what a caller feels (port numerics and quantisation together);
        # against the fp32 MLX engine on the same device isolates the quantisation,
        # which is the only number that says what int8 itself cost.
        bases = {f"vs_torch_{dev}": ref_p}
        if len(engines) > 1:
            bases["vs_mlx_fp32"] = probs["mlx_fp32"]
        for cmp_name, base in bases.items():
            for name, p in probs.items():
                if cmp_name == "vs_mlx_fp32" and name == "mlx_fp32":
                    continue
                mets = _metrics(p, base, valid, qs)
                rec.setdefault(name, {})[cmp_name] = mets
                key = f"{name} {cmp_name}"
                flips[key] = flips.get(key, 0) + mets["choice_flips"]
        out.append(rec)
    pairs = [(name, cmp_name) for name in engines
             for cmp_name in (["vs_torch_" + dev] + ([] if len(engines) < 2 else ["vs_mlx_fp32"]))
             if not (cmp_name == "vs_mlx_fp32" and name == "mlx_fp32")]
    summary = {"n_rows": len(out), "n_questions": sum(r["questions"] for r in out),
               "state_tokens": [r["state_tokens"] for r in out],
               "choice_flips": flips,
               "comparisons": [
                   {"engine": name, "against": cmp_name,
                    "prob_max_abs": max(r[name][cmp_name]["prob_max_abs"] for r in out),
                    "scalar_max_abs": max(r[name][cmp_name]["scalar_max_abs"] for r in out),
                    "choice_flips": flips[f"{name} {cmp_name}"],
                    "flip_margins": sorted({round(m, 6) for r in out
                                            for m in r[name][cmp_name]["flip_margins"]})}
                   for name, cmp_name in pairs]}
    return out, summary


def _metrics(p, base, valid, qs):
    d_prob = float(np.abs(p - base)[valid].max())
    d_scalar, flip, margins = 0.0, 0, []
    for i in range(len(qs)):
        v = valid[i]
        d_scalar = max(d_scalar, abs(scalar_of(p[i], v, qs[i].type)
                                     - scalar_of(base[i], v, qs[i].type)))
        if p[i][v].argmax() != base[i][v].argmax():
            flip += 1
            # The reference's own top-two gap at the flipped question: a decision that
            # moves at a 0.004 margin is noise agreeing with noise, and the reader
            # cannot tell that from a broken engine without the number.
            margins.append(float(np.sort(base[i][v])[-1] - np.sort(base[i][v])[-2]))
    return {"prob_max_abs": d_prob, "scalar_max_abs": d_scalar,
            "choice_flips": flip, "flip_margins": margins}


def size_of(eng, directory, label):
    d = eng.save(directory)
    nbytes = (d / "params.safetensors").stat().st_size
    return {"label": label, "dir": str(d), "file_bytes": nbytes,
            "tensors_bytes": eng.param_bytes(),
            "meta_bytes": (d / "meta.json").stat().st_size,
            "quantization": eng.quant}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/myna-v0")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--lengths", type=int, nargs="+", default=LENS,
                    help="the published ladder is the command that ran it")
    ap.add_argument("--quantize", choices=["none", "int8"], default="int8")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--keep", nargs="+", default=["trunk.tok.weight"],
                    help="param-name substrings left in fp32 (the embedding by default; "
                         "add .gate.weight or .wq./.wk to test what carries the drift)")
    ap.add_argument("--fp32-out", default=None,
                    help="default runs/<ckpt name>-mlx-fp32, so a grown checkpoint writes a "
                         "different artifact instead of overwriting v0's")
    ap.add_argument("--int8-out", default=None)
    ap.add_argument("--suite", default="data/decision-v2-pilot")
    ap.add_argument("--drift-rows", type=int, default=12)
    ap.add_argument("--out", default="runs/bench_mlx.md")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt)
    args.fp32_out = args.fp32_out or f"runs/{ckpt_dir.name}-mlx-fp32"
    args.int8_out = args.int8_out or f"runs/{ckpt_dir.name}-mlx-int8"
    tok = Tokenizer.from_file(str(ckpt_dir / "tokenizer.json"))

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    myna = Myna(ckpt_dir, device=dev)
    tmodel = myna.model
    n_params = myna.n_params

    engines = {"mlx_fp32": MynaMLX.from_checkpoint(ckpt_dir / "model.pt")}
    if args.quantize == "int8":
        engines["mlx_int8"] = MynaMLX.from_checkpoint(
            ckpt_dir / "model.pt", quantize=True, bits=args.bits,
            group_size=args.group_size, keep=args.keep)

    sizes = {"mlx_fp32": size_of(engines["mlx_fp32"], args.fp32_out, "MLX fp32")}
    if "mlx_int8" in engines:
        sizes["mlx_int8"] = size_of(engines["mlx_int8"], args.int8_out, "MLX int8")

    qt = batch_question_tensors(tok, [QUESTIONS])
    q_keys = ("q_ids", "q_mask", "span_mat", "opt_valid", "decide_idx")
    q_np = {k: qt[k][0].numpy() for k in q_keys}
    qt_dev = [qt[k].to(dev) for k in q_keys]

    def t_forward(state_ids):
        Ls = len(state_ids)
        s = torch.tensor([state_ids], dtype=torch.int64, device=dev)
        ln = torch.tensor([Ls], dtype=torch.int64, device=dev)
        with torch.no_grad():
            tmodel(s, ln, *qt_dev)
        if dev == "mps":
            torch.mps.synchronize()

    def m_forward(eng, state_ids):
        Ls = len(state_ids)
        eng.forward(np.array([state_ids], dtype=np.int64), np.array([Ls], dtype=np.int64),
                    q_np["q_ids"], q_np["q_mask"], q_np["span_mat"],
                    q_np["opt_valid"], q_np["decide_idx"])

    print(f"ckpt {ckpt_dir}  params {n_params/1e6:.1f}M  device torch={dev} mlx=metal")
    if args.quantize == "int8":
        q = engines["mlx_int8"].quant
        n_fp32 = sum(1 for k in engines["mlx_fp32"].p if k.endswith(".weight"))
        print(f"quantization bits={q['bits']} group_size={q['group_size']} on "
              f"{q['n_quantized']}/{n_fp32} Linear weights, kept fp32: {q['keep']} "
              f"(the token embedding is gathered, not matmulled)")
    for s in sizes.values():
        print(f"  {s['label']:>10}: {s['file_bytes']/2**20:7.2f} MiB on disk "
              f"({s['tensors_bytes']/2**20:.2f} MiB of tensors)")
    if "mlx_int8" in sizes:
        r = sizes["mlx_fp32"]["file_bytes"] / sizes["mlx_int8"]["file_bytes"]
        print(f"  int8/fp32 file ratio: {r:.2f}x")

    # The artifact on disk has to answer like the engine that wrote it, or the byte
    # figure above belongs to a file nobody can serve from.
    if "mlx_int8" in engines:
        reload_eng = MynaMLX.from_mlx_dir(args.int8_out)
        ids = build_state(tok, 128)
        a = np.array(engines["mlx_int8"].forward(
            np.array([ids], np.int64), np.array([128], np.int64), q_np["q_ids"], q_np["q_mask"],
            q_np["span_mat"], q_np["opt_valid"], q_np["decide_idx"]))
        b = np.array(reload_eng.forward(
            np.array([ids], np.int64), np.array([128], np.int64), q_np["q_ids"], q_np["q_mask"],
            q_np["span_mat"], q_np["opt_valid"], q_np["decide_idx"]))
        print(f"  reload of {args.int8_out}: logits identical = "
              f"{bool((a == b).all())} (max |d| {np.abs(a-b).max():.1e})")

    print(f"\n{'state tokens':>13} | {'torch ms':>9} | " + " | ".join(f"{m:>12}" for m in engines)
          + " | best MLX speedup")
    rows = []
    for L in args.lengths:
        ids = build_state(tok, L)
        t_forward(ids)
        for eng in engines.values():
            m_forward(eng, ids)
        tt = []
        for _ in range(args.reps):
            t0 = time.perf_counter(); t_forward(ids); tt.append((time.perf_counter() - t0) * 1000)
        tm = statistics.median(tt)
        med = {}
        for name, eng in engines.items():
            mt = []
            for _ in range(args.reps):
                t0 = time.perf_counter(); m_forward(eng, ids)
                mt.append((time.perf_counter() - t0) * 1000)
            med[name] = statistics.median(mt)
        best = min(med.values())
        print(f"{L:>13} | {tm:>9.2f} | " + " | ".join(f"{med[m]:>12.2f}" for m in engines)
              + f" | {tm/best:>6.2f}x")
        rows.append({"state_tokens": L, "torch_ms": tm, **med})

    drift = {"rows": [], "summary": {"skipped": "no --suite"}}
    suite = Path(args.suite)
    if args.drift_rows > 0 and (suite / "calibration.jsonl").exists():
        from myna.real_data import load_split
        groups = load_split(suite / "calibration.jsonl")
        drift_rows, drift_sum = drift_vs_reference(myna, engines, groups, tok, dev, args.drift_rows)
        drift = {"rows": drift_rows, "summary": drift_sum}
        print(f"\ndrift against the {dev} engine on {drift_sum['n_rows']} calibration rows "
              f"({drift_sum['n_questions']} questions, real state and question widths):")
        print(f"{'engine':>10} | {'against':>14} | {'max |dp|':>10} | "
              f"{'max |dscalar|':>13} | flips (reference top-two margin)")
        for c in drift_sum["comparisons"]:
            mg = ", ".join(f"{m:.4f}" for m in c["flip_margins"])
            print(f"{c['engine']:>10} | {c['against']:>14} | {c['prob_max_abs']:>10.2e} | "
                  f"{c['scalar_max_abs']:>13.2e} | {c['choice_flips']}/{drift_sum['n_questions']}"
                  f"{f' [{mg}]' if mg else ''}")
    elif args.drift_rows > 0:
        drift = {"rows": [], "summary": {"skipped": f"no {suite/'calibration.jsonl'}"}}
        print(f"\ndrift skipped: {suite/'calibration.jsonl'} not found")

    out_json = Path(args.out_json or (args.out[:-3] + ".json" if args.out.endswith(".md")
                                      else args.out + ".json"))
    payload = {"ckpt": str(ckpt_dir), "n_params": n_params, "device": {"torch": dev, "mlx": "metal"},
               "reps": args.reps, "questions": len(QUESTIONS), "argv": list(sys.argv),
               "quantization": {"bits": args.bits, "group_size": args.group_size,
                                "keep": list(args.keep),
                                "enabled": args.quantize == "int8",
                                "detail": engines.get("mlx_int8", engines["mlx_fp32"]).quant},
               "sizes": sizes, "latency": rows, "drift": drift,
               "load_avg": list(os.getloadavg())}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

    with open(args.out, "w") as f:
        f.write(f"# MLX vs PyTorch(MPS) full-predict latency — {ckpt_dir}\n\n")
        f.write(f"Model {n_params/1e6:.1f}M params, {len(QUESTIONS)} questions/request, "
                f"median of {args.reps} reps, same M5 box. "
                f"int8 = {args.bits} bits, group size {args.group_size}, "
                f"kept fp32: {', '.join('`' + k + '`' for k in args.keep)}.\n\n")
        f.write(f"| state tokens | torch (MPS) ms | " +
                " | ".join(f"{m} ms" for m in engines) + " | best MLX speedup |\n")
        f.write("|---|---|" + "---|" * len(engines) + "---|\n")
        for r in rows:
            f.write(f"| {r['state_tokens']} | {r['torch_ms']:.2f} | " +
                    " | ".join(f"{r[m]:.2f}" for m in engines) +
                    f" | {r['torch_ms']/min(r[m] for m in engines):.2f}x |\n")
        f.write("\n## Artifact size, measured as bytes on disk\n\n")
        f.write("| engine | params.safetensors | tensors | quantization |\n|---|---|---|---|\n")
        for name, s in sizes.items():
            q = "fp32" if s["quantization"] is None else \
                f"int{8 if s['quantization']['bits']==8 else 4} group {s['quantization']['group_size']}"
            f.write(f"| {s['label']} | {s['file_bytes']/2**20:.2f} MiB | "
                    f"{s['tensors_bytes']/2**20:.2f} MiB | {q} |\n")
        if "mlx_int8" in sizes:
            f.write(f"\nint8 is {sizes['mlx_fp32']['file_bytes']/sizes['mlx_int8']['file_bytes']:.2f}x "
                    f"smaller on disk than fp32, same container.\n")
        f.write("\n## Drift on real questions\n\n")
        if drift["rows"]:
            ds = drift["summary"]
            comps = ds["comparisons"]
            f.write(f"{ds['n_rows']} rows of `calibration.jsonl`, each answered at its own state "
                    f"length and question width (no padding to a fixed shape), "
                    f"{ds['n_questions']} questions, option probabilities read through the "
                    f"engine's own temperature ({myna.temperature}). States: "
                    f"{min(ds['state_tokens'])}–{max(ds['state_tokens'])} tokens. "
                    f"Load average at print: "
                    f"{', '.join(f'{x:.2f}' for x in os.getloadavg())}.\n\n")
            f.write("| engine | against | max abs dp on an option prob | max abs d on the "
                    "published scalar | argmax flips | reference top-two margin at the "
                    "flipped questions |\n|---|---|---|---|---|---|\n")
            for c in comps:
                mg = ", ".join(f"{m:.4f}" for m in c["flip_margins"]) or "—"
                f.write(f"| {c['engine']} | {c['against']} | {c['prob_max_abs']:.2e} | "
                        f"{c['scalar_max_abs']:.2e} | {c['choice_flips']}/{ds['n_questions']} "
                        f"| {mg} |\n")
            f.write("\nEvery row, so each maximum has a name beside it (`why` marks the two rows "
                    "added on purpose rather than by stride). Cells are max |dp| with argmax "
                    "flips in parentheses.\n\n")
            f.write("| source | why | state tokens | questions | options | " +
                    " | ".join(f"{c['engine']} vs " + c["against"].replace("vs_", "", 1)
                               for c in comps) + " |\n")
            f.write("|---|---|---|---|---|" + "---|" * len(comps) + "\n")
            for r in drift["rows"]:
                f.write(f"| {r['source']} | {', '.join(r['why'])} | {r['state_tokens']} | "
                        f"{r['questions']} | {r['max_options']} |")
                for c in comps:
                    m = r[c["engine"]][c["against"]]
                    f.write(f" {m['prob_max_abs']:.1e}"
                            + (f" ({m['choice_flips']})" if m["choice_flips"] else "") + " |")
                f.write("\n")
            f.write(f"\nCommand: `uv run python -m bench.bench_mlx "
                    f"{' '.join(a for a in sys.argv[1:])}`\n")
        else:
            f.write(f"Not measured: {drift['summary'].get('skipped', 'no rows')}.\n")
    print(f"\nwrote {args.out} and {out_json}")


if __name__ == "__main__":
    main()

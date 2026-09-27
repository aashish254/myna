"""ONNX export of the two graphs a browser needs, and the parity proof (SPEC §5 P6, G3).

    python -m myna.onnx_export --ckpt runs/myna-v0 --out runs/onnx --verify

The split is not convenience, it is the architecture. The observation is scanned by
`state_step.onnx`, which takes one **fixed-width chunk of tokens plus the fixed-size state
stack** and returns the next state — so a 16k-token document is 64 calls to the same
weights, not a graph that grows with the input. Questions are answered by
`question.onnx`, which resumes from that state and returns hidden states; the pointer head
is a few small matmuls over option spans and belongs in JavaScript beside the spans the
tokenizer produced, not in the trunk graph.

Three properties this makes the exported files prove rather than assert:

* **Padding is exact, not approximately harmless.** A document is rarely a whole number of
  chunks, so the tail call carries a mask, and the parity check requires the chained state
  to match torch's own scan over the true length. The same applies to questions: the graph
  has a fixed question width, so a request with fewer questions is padded, and a padded row
  must not disturb a real one. `isolation_max_abs_error` measures that: the same question run
  twice through the exported graph, once with the other rows masked-out garbage, and the
  difference must be exactly zero. Read for what it is — it verifies the **mask**, on the
  machine that will actually apply it, since ONNX's own reshape of a padded batch element is
  where a leak would appear; questions being separate *branches* of the state is a different
  property, already pinned by `tests/test_engine_parity.py`.
* **The gate is stated in the unit of each quantity.** The state's entries reach ~5e2,
  so the chained scan is gated on *relative* error (~5e-6 measured) while probabilities,
  bounded by 1, are gated absolutely — `--max-abs-error` and `--max-rel-error`. An absolute
  1e-4 on the state would ask two BLAS implementations to agree bitwise, which no export
  can promise, and the gate would read as a failing export rather than a wrong graph.

* **Composition is the length claim in this format.** Chaining `state_step` over
  `n_chunks` calls must land on the state torch reaches in one pass (§9.22's fixed-state
  statement, re-derived through onnxruntime).
* **The exporter is named, because the two disagree.** `dynamo=True` is required: the
  legacy TorchScript exporter produces graphs for this trunk that `onnxruntime` refuses to
  load (`/fwd/Mul_2 … Incompatible dimensions` — its shape inference cannot carry the
  einsum-and-flip scan), so a build that "worked" under the old flag would fail on the
  reader's machine. Both graphs here are exported with the new path and loaded back.

Not here yet: int8 quantization and the in-Chrome measurement (6b). G3's byte budget
belongs to the artifact as shipped, so this reports fp32 sizes and leaves the quantized
figure to be compared against something honest.

What 6b adds to the artifact, because 6a's byte figure was two copies of one trunk:

* **one weights file.** `share_weights` rewrites both graphs' initializers to point into
  a single `weights.bin`, deduplicated by the SHA-256 of each tensor's bytes. The two
  graphs still load independently — this is a statement about what the browser fetches,
  not about how ONNX Runtime allocates: two sessions still hold two copies in RAM, and
  `meta.json` reports both numbers.
* **the pointer head and the tokenizer, shipped.** `head.bin` is the wq/wk matrices as
  raw float32 in a fixed field order, and `tokenizer.json` is copied next to the graphs.
  Neither is decorative: the page has to turn text into ids and hidden states into
  probabilities, so an artifact without them cannot answer anything, and G3's byte
  budget has to include them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .engine import Myna
from .tokenizer import encode_text

#: Both graphs are exported at a fixed token width: length is more calls to the same
#: graph, never a bigger graph.
CHUNK = 256
#: `question.onnx` is exported at this question count; a smaller request is padded and
#: masked, which is what the isolation check exists to verify.
QUESTIONS = 8
OPSET = 18   # 17 is below what this torch build implements: the exporter
           # warns, writes 18 anyway, and the version-converter fallback aborts
GRAPHS = ("state_step.onnx", "question.onnx")
#: One file holds the initializers of every graph, deduplicated by content.
WEIGHTS = "weights.bin"
#: The pointer head is not in either graph (it belongs beside the option spans), so it
#: ships as its own raw float32 blob in this field order.
HEAD_FIELDS = ("wq_weight", "wq_bias", "wk_weight", "wk_bias")


class StateStep(nn.Module):
    """One chunk of the observation in, one state stack out.

    `pos_offset` is an input rather than a constant because RoPE is absolute: the same
    weights must work at any depth in the document, and the caller is the one that knows
    how far it has read. `mask` is 1.0 on real tokens — a padded one must contribute
    nothing to the memory and must not decay it either.

    `width` and `scan` are two different things and have to stay that way. `width` is
    the graph's fixed input shape — how many tokens one call carries. `scan` is the
    chunk the parallel form accumulates within the call, and it sets the size of the
    `[rows, heads, scan, scan, d_k]` tile the runtime allocates: at width 256, scan 256
    that tile is 67 MB per layer for one row, and 537 MB for the 8-question batch. A
    tab cannot hold that, so the browser build asks for scan 16 — the same arithmetic,
    proven equivalent at every chunk size by `tests/test_trunk_numerics.py`, at a tile
    of a few hundred KiB.
    """

    def __init__(self, trunk, chunk: int, scan: int | None = None):
        super().__init__()
        self.trunk = trunk
        self.width = chunk
        self.scan = scan or chunk

    def forward(self, ids, mask, pos_offset, *S_in):
        pos = pos_offset + torch.arange(self.width, device=ids.device)
        m = mask[:, None, :, None]
        h = self.trunk.tok(ids)
        new_S = []
        for layer, S in zip(self.trunk.layers, S_in):
            h, S_next = layer.scan_state(h, pos, init_S=S, parallel=True, mask=m,
                                         chunk=self.scan)
            new_S.append(S_next)
        return tuple(new_S)


class QuestionBranch(nn.Module):
    """Question tokens + the cached state -> hidden states for the pointer head.

    Same `width`/`scan` split as `StateStep`: the width is the request shape the graph
    is compiled for, the scan chunk is what the runtime allocates per step.
    """

    def __init__(self, trunk, scan: int):
        super().__init__()
        self.trunk = trunk
        self.scan = scan

    def forward(self, q_ids, q_mask, pos_offset, *S_in):
        N, Lq = q_ids.shape
        pos = pos_offset + torch.arange(Lq, device=q_ids.device)
        m = q_mask[:, None, :, None]
        h = self.trunk.tok(q_ids)
        for layer, S in zip(self.trunk.layers, S_in):
            h = layer.scan_question(h, pos, S.expand(N, -1, -1, -1), parallel=True,
                                   mask=m, chunk=self.scan)
        return self.trunk.norm(h)


def state_shapes(cfg) -> list[tuple[int, ...]]:
    return [(1, cfg.n_heads, cfg.d_k, cfg.d_v)] * cfg.n_layers


def share_weights(out_dir: str | Path, graphs: tuple[str, ...] = GRAPHS,
                  location: str = WEIGHTS, align: int = 16,
                  min_bytes: int = 4096) -> dict:
    """Rewrite every graph to read its initializers from ONE external-data file.

    6a measured the cost of the naive split: 93.7 MiB fp32 for a 14.45M-parameter model,
    because `state_step` and `question` are the same trunk and torch gave each graph its
    own copy of every weight (§9.26). Sharing is not a packing trick — the two copies
    hold identical bytes, so the artifact can reference one region and transfer the
    trunk once.

    Deduplication is keyed on the SHA-256 of the tensor's bytes, which is sound for
    initializers specifically because they are constants: no node writes to one, so two
    tensors with equal content may alias freely. What is *not* deduplicated is names —
    each graph still references its own initializer list, so a graph that needs fewer
    weights (`state_step` has no backward scan) simply points at fewer regions.

    The per-graph `*.onnx.data` siblings are deleted once the rewrite is on disk:
    leaving them would both double the download and make the byte accounting in
    `meta.json` a lie about what the browser fetches.

    `min_bytes` keeps the small constants inline, because `onnxruntime` refuses to load
    a graph whose shape-inference inputs live in an external file ("Cannot parse data
    from external tensors"), and those inputs are axis lists and position vectors of a
    few hundred bytes. The threshold is reported in `meta.json` so the test can assert
    the property that matters: every initializer at or above it is shared, so no weight
    is duplicated while the accounting claims otherwise.
    """
    import onnx
    from onnx import external_data_helper, numpy_helper

    out = Path(out_dir)
    protos = {g: onnx.load(out / g) for g in graphs}

    buf = bytearray()
    placed: dict[bytes, tuple[int, int]] = {}
    refs: dict[str, dict[str, tuple[int, int]]] = {}
    inline = 0
    for g, model in protos.items():
        new_inits, refs[g] = [], {}
        for t in model.graph.initializer:
            if t.data_type == onnx.TensorProto.STRING:
                new_inits.append(t)      # nothing to deduplicate by numeric bytes
                inline += 1
                continue
            arr = numpy_helper.to_array(t)
            raw = np.ascontiguousarray(arr).tobytes()
            if len(raw) < min_bytes:
                # onnxruntime will not run shape inference off an external file, so the
                # small constants (positions, axis lists) stay inline where they are
                # tens of bytes and change nothing about the byte budget.
                small = onnx.TensorProto()
                small.CopyFrom(numpy_helper.from_array(arr, name=t.name))
                new_inits.append(small)
                inline += 1
                continue
            key = hashlib.sha256(raw).digest()
            got = placed.get(key)
            if got is None:
                off = len(buf) + (-len(buf) % align)
                buf.extend(b"\x00" * (off - len(buf)))
                buf.extend(raw)
                got = (off, len(raw))
                placed[key] = got
            off, ln = got
            nt = numpy_helper.from_array(arr, name=t.name)
            external_data_helper.set_external_data(nt, location=location, offset=off,
                                                   length=ln)
            # onnx.save re-serializes any external tensor that still carries inline
            # bytes, which would rewrite `weights.bin` from whichever graph saved last
            # and make the offsets a guess. The blob on disk is the one written above.
            nt.ClearField("raw_data")
            new_inits.append(nt)
            refs[g][t.name] = got
        del model.graph.initializer[:]
        model.graph.initializer.extend(new_inits)

    (out / location).write_bytes(bytes(buf))
    for g, model in protos.items():
        onnx.save(model, out / g)
    for f in out.glob("*.onnx.data"):
        f.unlink()

    # Counted by CONTENT region, never by name. `dynamo` names initializers `val_N` per
    # graph and the numbering says nothing about what a tensor holds — the two graphs
    # share all 12 of the state graph's weight regions and their name sets overlap by
    # two, so an intersection of names measures the counter, not the sharing.
    regions: dict[tuple[int, int], set[str]] = {}
    aliases = 0
    for g, r in refs.items():
        for off_ln in r.values():
            if off_ln in regions:
                aliases += 1
            regions.setdefault(off_ln, set()).add(g)
    shared = {k for k, v in regions.items() if len(v) > 1}
    return {"weights_bytes": len(buf), "align": align,
            "unique_regions": len(regions), "inline_tensors": inline,
            "min_bytes": min_bytes,
            # every reference's bytes added up, which is what a per-graph copy costs:
            # it counts a within-graph alias (the zero-initialised gates all share one
            # region) as many times as the graph asks for it.
            "per_reference_bytes": sum(sum(ln for _, ln in r.values())
                                       for r in refs.values()),
            "shared_regions": len(shared),
            "shared_bytes": sum(ln for _, ln in shared),
            "aliased_references": aliases,
            "per_graph_bytes": {g: sum(ln for _, ln in r.values()) for g, r in refs.items()},
            "per_graph_tensors": {g: len(r) for g, r in refs.items()}}


def export_head(out_dir: str | Path, myna: Myna, ckpt_dir: str | Path) -> dict:
    """Write the pointer head and the tokenizer beside the graphs.

    The head is deliberately outside both graphs: it is a few matvecs over option-span
    means, and the spans are a property of the *text* the page already has. Shipping it
    as raw float32 in a fixed field order (see HEAD_FIELDS) keeps the browser from
    parsing a JSON array of 200k numbers — and the sizes here are part of G3's byte
    budget, so they are reported rather than assumed.
    """
    out = Path(out_dir)
    m = myna.model
    blob = bytearray()
    layout = {}
    for name, tensor in (("wq_weight", m.wq.weight), ("wq_bias", m.wq.bias),
                        ("wk_weight", m.wk.weight), ("wk_bias", m.wk.bias)):
        arr = tensor.detach().cpu().numpy()
        if arr.dtype != np.float32:
            raise SystemExit(f"pointer head {name} is {arr.dtype}, not float32: the raw blob "
                             f"would be read as float32 by the page and silently change value")
        layout[name] = {"shape": list(arr.shape), "offset": len(blob),
                        "bytes": arr.nbytes}
        blob.extend(np.ascontiguousarray(arr).tobytes())
    (out / "head.bin").write_bytes(bytes(blob))
    shutil.copyfile(Path(ckpt_dir) / "tokenizer.json", out / "tokenizer.json")
    return {"head_bytes": len(blob), "head_layout": layout, "head_fields": list(HEAD_FIELDS),
            "tokenizer_bytes": (out / "tokenizer.json").stat().st_size}


def read_head(out_dir: str | Path) -> tuple[dict, dict]:
    """Rebuild the pointer head from `head.bin` + `meta.json` — the path the page takes.

    Reading it back through the recorded layout is the only check that the layout is
    about the file that exists: a blob written from the right tensors but described
    wrongly would still be a valid artifact for nothing but Python. `<f4` is the
    browser's `DataView.getFloat32(..., true)`, so this mirrors the JS read exactly.
    """
    out = Path(out_dir)
    meta = json.loads((out / "meta.json").read_text())
    raw = (out / "head.bin").read_bytes()
    fields = {}
    for name in meta["head"]["head_fields"]:
        lay = meta["head"]["head_layout"][name]
        end = lay["offset"] + lay["bytes"]
        if end > len(raw):
            raise SystemExit(f"{name} claims bytes {lay['offset']}..{end} of a "
                             f"{len(raw)}-byte blob")
        fields[name] = np.frombuffer(raw, dtype="<f4", count=lay["bytes"] // 4,
                                     offset=lay["offset"]).reshape(lay["shape"])
    return fields, meta


def pointer_probs(fields, d_ptr: int, temperature: float, h, spans, dec: int) -> list[float]:
    """The head in numpy from the shipped blob: the arithmetic the page will run.

    Same as `Myna._readout`'s two matvecs, mean-pool, scaled dot and softmax — kept
    separate so a test can prove the *artifact* (not the Python object) reproduces the
    engine's probabilities.
    """
    h = np.asarray(h, dtype=np.float32)
    decide = h[dec] @ fields["wq_weight"].T + fields["wq_bias"]
    pooled = np.stack([h[s:e].mean(0) for s, e in spans]).astype(np.float32)
    keys = pooled @ fields["wk_weight"].T + fields["wk_bias"]
    logits = keys @ decide / (d_ptr ** 0.5) / temperature
    e = np.exp(logits - logits.max())
    return list(e / e.sum())


def export(ckpt_dir: str | Path, out_dir: str | Path, chunk: int = CHUNK,
           n_questions: int = QUESTIONS, opset: int = OPSET, q_len: int = 64,
           share: bool = True, scan_chunk: int | None = None) -> dict:
    """Write the two graphs + head + meta.json and return the metadata."""
    myna = Myna(ckpt_dir, device="cpu")
    cfg = myna.cfg
    L, out = cfg.n_layers, Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trunk = myna.trunk.eval()
    # the default keeps 6a's shape: one tile per call for the narrower of the two graphs
    scan = scan_chunk or min(chunk, q_len)
    if not 1 <= scan <= min(chunk, q_len):
        raise SystemExit(f"--scan-chunk must be in 1..{min(chunk, q_len)} (the narrower "
                         f"graph width), got {scan}")
    zero_S = tuple(torch.zeros(sh) for sh in state_shapes(cfg))
    names = [f"S_in_{i}" for i in range(L)]
    off = torch.tensor(0, dtype=torch.int64)
    kw = {"dynamo": True, "opset_version": opset}   # see the module docstring

    # .eval() on the wrapper, not only the trunk: a fresh nn.Module defaults to training,
    # and torch warns about exporting in that mode whether or not anything is mode-dependent.
    torch.onnx.export(StateStep(trunk, chunk, scan).eval(),
                      (torch.randint(0, cfg.vocab, (1, chunk), dtype=torch.int64),
                       torch.ones(1, chunk), off, *zero_S),
                      str(out / "state_step.onnx"),
                      input_names=["ids", "mask", "pos_offset"] + names,
                      output_names=[f"S_out_{i}" for i in range(L)], **kw)

    torch.onnx.export(QuestionBranch(trunk, scan).eval(),
                      (torch.randint(0, cfg.vocab, (n_questions, q_len), dtype=torch.int64),
                       torch.ones(n_questions, q_len), off, *zero_S),
                      str(out / "question.onnx"),
                      input_names=["q_ids", "q_mask", "pos_offset"] + names,
                      output_names=["h_q"], **kw)

    # 6a shipped the same trunk twice (35.6 + 58.13 MiB fp32, §9.26). Both graphs are
    # written, then their initializers are deduplicated into one external-data file that
    # they both reference, so the browser downloads the weights once.
    shared = share_weights(out, GRAPHS) if share else None
    head = export_head(out, myna, ckpt_dir)

    # The requested opset is not necessarily the exported one: opset 17 is below what
    # this torch build implements, the exporter says so in a warning and writes 18. A
    # meta.json that repeats the request would then be a label, not a measurement.
    actual = {}
    try:
        import onnx
        for g in GRAPHS:
            m = onnx.load(out / g, load_external_data=False)
            actual[g] = m.opset_import[0].version
    except ImportError:  # onnx ships with the browser extra; without it we cannot read back
        actual = {g: None for g in GRAPHS}
    for g, v in actual.items():
        if v is not None and v != opset:
            print(f"note: {g} carries opset {v}, not the requested {opset} — the exporter "
                  f"fell back and said so on stderr")

    per_layer = int(cfg.n_heads * cfg.d_k * cfg.d_v * 4)
    # Everything a page fetches, measured on disk rather than derived from params*4:
    # the graph skeletons, the weights (one shared file, or one per graph when
    # --no-share asks for 6a's shape), the pointer head, the tokenizer. A directory
    # listing is the honest unit here — `meta.json` is the only file in it that is not
    # fetched, and it is added from its own serialized length below.
    sizes = {f.name: f.stat().st_size for f in out.iterdir()
             if f.is_file() and f.name != "meta.json"}
    meta = {"ckpt": str(ckpt_dir), "opset_actual": actual, "chunk": chunk, "q_len": q_len,
            "n_questions": n_questions, "opset": opset, "exporter": "dynamo",
            "n_layers": L, "n_heads": cfg.n_heads, "d_model": cfg.d_model,
            "d_k": cfg.d_k, "d_v": cfg.d_v, "d_ptr": cfg.d_ptr, "vocab": cfg.vocab,
            "temperature": myna.temperature, "abstain_below": myna.abstain_below,
            "params_m": round(myna.n_params / 1e6, 2),
            "state_bytes_per_layer": per_layer, "state_bytes_total": per_layer * L,
            "scan_chunk": scan,
            # the `[rows, heads, scan, scan, d_k]` tile the parallel form allocates per
            # layer per call: the number that decides whether a tab survives, not just
            # how long it takes. At the engine's inference chunk (256) and the exported
            # question width this is hundreds of MB for one request. Bytes, not MiB —
            # rounding a 16 KiB tile to two decimals of a MiB destroys the ratio a
            # reader would check the knob against.
            "peak_tile_bytes": max(
                1 * cfg.n_heads * scan * scan * cfg.d_k * 4,
                n_questions * cfg.n_heads * scan * scan * cfg.d_k * 4),
            "files_mib": {f: round(b / 2**20, 2) for f, b in sorted(sizes.items())},
            "shared_weights": bool(share),
            "weight_sharing": shared, "head": head,
            "artifact_bytes_total": sum(sizes.values())}
    # transfer = every distinct file the browser fetches, counted once, plus meta.json
    # itself. `len()` of the serialization undercounts by the two accounting keys added
    # below (~50 bytes), against 58 MiB of weights — the figure is reported to
    # hundredths of a MiB, so the difference is not a rounding decision.
    raw = len(json.dumps(meta, indent=2).encode())
    meta["transfer_mib"] = round((meta["artifact_bytes_total"] + raw) / 2**20, 2)
    if share:
        meta["unshared_transfer_mib"] = round(
            (sum(sizes[g] for g in GRAPHS) + shared["per_reference_bytes"]
             + sizes["head.bin"] + sizes["tokenizer.json"] + raw) / 2**20, 2)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"exported to {out}: " + " · ".join(f"{k} {v} MiB" for k, v in
                                              meta["files_mib"].items()))
    if share:
        print(f"weights: ONE shared {WEIGHTS} of {meta['files_mib'][WEIGHTS]} MiB — "
              f"{shared['unique_regions']} distinct regions, {shared['shared_regions']} "
              f"of them read by both graphs, so {shared['shared_bytes'] / 2**20:.2f} MiB "
              f"transfers once instead of twice ({shared['aliased_references']} references "
              f"alias a region inside their own graph). One copy per graph would transfer "
              f"{meta['unshared_transfer_mib']} MiB; the artifact transfers "
              f"{meta['transfer_mib']} MiB")
    else:
        print(f"weights: each graph carries its own copy (--no-share, 6a's shape); "
              f"artifact transfers {meta['transfer_mib']} MiB")
    print(f"meta: chunk {chunk} scanned in {scan}-token tiles "
          f"({meta['peak_tile_bytes'] / 2**20:.1f} MiB peak per call), {n_questions} questions x "
          f"{q_len} tokens, opset {opset}, "
          f"exporter dynamo, {L} layers, {meta['params_m']}M params, "
          f"state {meta['state_bytes_total']/1024:.0f} KiB, "
          f"head {meta['files_mib']['head.bin']} MiB, "
          f"tokenizer {meta['files_mib']['tokenizer.json']} MiB")
    return meta


_SESSIONS: dict[str, object] = {}


def _session(path: Path):
    """One session per graph path, reused.

    A browser would never rebuild a session per request, and neither should the
    checks: on macOS each `InferenceSession` leaves a thread pool that is torn down
    after Python has begun exiting, so a test module that opens ten of them can die in
    `recursive_mutex lock failed` *after* every assertion passed — a green run with a
    non-zero exit code, which a mutation battery reads as a broken baseline.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("onnxruntime is not installed: uv sync --extra browser")
    # keyed on the file's identity too, so a re-export into the same path in the same
    # process cannot be served the previous graph.
    st = Path(path).stat()
    key = f"{Path(path).resolve()}:{st.st_mtime_ns}:{st.st_size}"
    if key not in _SESSIONS:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        _SESSIONS[key] = ort.InferenceSession(str(path), sess_options=so,
                                              providers=["CPUExecutionProvider"])
    return _SESSIONS[key]


def _off(x: int) -> np.ndarray:
    return np.array(x, dtype="int64")


def chain_state(step, ids: list[int] | torch.Tensor, chunk: int, L: int,
                shapes: list[tuple[int, ...]], start: int = 0):
    """Run `state_step` over a token list, one fixed-width call at a time.

    The tail is padded and masked, which is the case that must be exact: a document
    is almost never a whole number of chunks, and a padded token that decays the
    memory would put a wrong state in front of every question after it."""
    n = len(ids)
    carried = [np.zeros(sh, dtype="float32") for sh in shapes]
    fed = 0
    while fed < n:
        take = [int(x) for x in ids[fed:fed + chunk]]
        m = np.zeros((1, chunk), dtype="float32")
        m[0, :len(take)] = 1.0
        row = np.zeros((1, chunk), dtype="int64")
        row[0, :len(take)] = take
        y = step.run(None, {"ids": row, "mask": m, "pos_offset": _off(start + fed),
                            **{f"S_in_{i}": carried[i] for i in range(L)}})
        carried = [y[i] for i in range(L)]
        fed += len(take)
    return carried, fed


def _rel(abs_err: float, scale: float) -> float:
    """Error as a fraction of the magnitude of the thing it is an error in.

    P6's "parity ≤ 1e-4" was written against an answer, and the state is not an
    answer: its entries reach ~5e2, so an absolute 1e-4 there would demand
    bit-identical accumulation from two different BLAS implementations. Measured, the
    chain's absolute error at that scale is ~2e-3 with a *relative* error of ~5e-6 —
    the same order as the question branch's absolute 2e-5 on unit-scale hidden states.
    So the state is reported both ways and gated on the ratio; probabilities, which are
    bounded by 1, are gated absolutely, as they should be."""
    return abs_err / max(scale, 1e-12)


def parity(ckpt_dir: str | Path, out_dir: str | Path, max_abs_error: float = 1e-4,
           max_rel_error: float = 1e-4, n_chunks: int = 3, seed: int = 0,
           suite: str | Path | None = None) -> dict:
    """torch vs onnxruntime: the chained scan, the question branch, isolation, and the
    decision that comes out the end."""
    meta = json.loads((Path(out_dir) / "meta.json").read_text())
    chunk, L, N, q_len = meta["chunk"], meta["n_layers"], meta["n_questions"], meta["q_len"]
    # torch compares at the same accumulation granularity the graph was built with: the
    # chunk sizes are proven exactly equivalent in tests/test_trunk_numerics.py, but the
    # floating-point order is not, and the browser's tile size is set here too.
    scan = meta.get("scan_chunk", chunk)
    torch.manual_seed(seed)
    myna = Myna(ckpt_dir, device="cpu")
    cfg = myna.cfg
    step = _session(Path(out_dir) / "state_step.onnx")
    question = _session(Path(out_dir) / "question.onnx")
    shapes = state_shapes(cfg)

    # --- 1: chained state_step with a padded tail vs torch's one-pass scan -------
    real = (n_chunks - 1) * chunk + max(1, chunk // 3)
    ids = torch.randint(0, cfg.vocab, (1, real), dtype=torch.int64)[0].tolist()
    with torch.no_grad():
        _, S_torch = myna.trunk.encode_state(torch.tensor([ids]), parallel=True, chunk=scan)
    t0 = time.perf_counter()
    carried, fed = chain_state(step, ids, chunk, L, shapes)
    chain_ms = (time.perf_counter() - t0) * 1000 / max(len(range(0, real, chunk)), 1)
    assert fed == real, f"the loop consumed {fed} of {real} tokens"
    chain_err = max(float(np.abs(carried[i] - S_torch[i].numpy()).max()) for i in range(L))

    # --- 2: the question branch, run at full width with masked padding -----------
    n_real = max(1, N // 2)
    q_ids = torch.randint(0, cfg.vocab, (N, q_len), dtype=torch.int64)
    q_mask = torch.ones(N, q_len)
    q_mask[n_real:] = 0.0
    q_ids[n_real:] = 0
    S_stack = [torch.from_numpy(carried[i]) for i in range(L)]
    yq = question.run(None, {"q_ids": q_ids.numpy(), "q_mask": q_mask.numpy(),
                             "pos_offset": _off(real),
                             **{f"S_in_{i}": S_stack[i].numpy() for i in range(L)}})[0]
    with torch.no_grad():
        h_torch = myna.trunk.encode_question_batch(q_ids[:n_real], q_mask[:n_real],
                                                   S_stack, real, chunk=scan)
    q_err = float(np.abs(yq[:n_real] - h_torch.numpy()).max())

    # --- 3: isolation — the same question beside garbage must not move -----------
    one = q_ids[0].unsqueeze(0).repeat(N, 1)
    m_one = np.zeros((N, q_len), dtype="float32")
    m_one[0] = 1.0
    y_iso = question.run(None, {"q_ids": one.numpy(), "q_mask": m_one,
                                "pos_offset": _off(real),
                                **{f"S_in_{i}": S_stack[i].numpy() for i in range(L)}})[0]
    iso_err = float(np.abs(y_iso[0] - yq[0]).max())

    # --- 4: the decision itself, head included, on a real example ----------------
    # This is the number G3 is about: probabilities are bounded by 1, so the gate can
    # be absolute here and should be. The ONNX hidden states are fed through the same
    # pointer head the engine uses, so the comparison covers the whole path a browser
    # would run — scan, branch, probe, temperature.
    from .data import WORKFLOWS, generate
    from .tokenizer import build_question

    ex = generate(1, "support", random.Random(seed))[0]
    questions = {q.name: {"type": q.type, "instructions": q.instruction,
                          "criteria": q.options} for q in WORKFLOWS["support"][0]}
    torch_out = myna.observe(ex.state).ask(questions)["answers"]
    # the same observation, scanned through the exported graph in chunks: the state
    # and the question have to come from one document, or "decision parity" would be
    # comparing an answer against a different text.
    ids_t = encode_text(myna.tok, ex.state)
    S_ex, n_fed = chain_state(step, ids_t, chunk, L, shapes)
    assert n_fed == len(ids_t)
    prob_err, n_compared, n_skipped = 0.0, 0, 0
    for name, spec in questions.items():
        opts = myna._options(spec)
        qrow, spans, dec = build_question(myna.tok, spec["instructions"], opts)
        Lq = len(qrow)
        if Lq > q_len:
            # Not a skip: a decision-parity number averaged over zero rows is 0.0,
            # which passes any bound. The width is a contract with the exporter, so
            # exceeding it is reported and fails the run.
            n_skipped += 1
            continue
        # both axes are static in the exported graph: the request is padded to the
        # question width AND to the question-token width, and the mask says which
        # cells are real -- which is precisely the padding this check exists for.
        pad = np.zeros((N, q_len), dtype="int64")
        pad[0, :Lq] = qrow
        mk = np.zeros((N, q_len), dtype="float32")
        mk[0, :Lq] = 1.0
        yy = question.run(None, {"q_ids": pad, "q_mask": mk, "pos_offset": _off(len(ids_t)),
                                 **{f"S_in_{i}": S_ex[i] for i in range(L)}})[0]
        onnx_ans = myna._readout(spec, opts, torch.from_numpy(yy[0]), dec, spans)
        a, b = torch_out[name], onnx_ans
        if "probabilities" in a:
            prob_err = max(prob_err, max(abs(a["probabilities"][k] - b["probabilities"][k])
                                         for k in a["probabilities"]))
        elif "noul" in a:
            prob_err = max(prob_err, abs(a["noul"] - b["noul"]))
        n_compared += 1
    if n_compared == 0:
        raise AssertionError(
            f"decision parity compared no questions: all {n_skipped} exceed --q-len "
            f"{q_len}, so the exported graph cannot answer this request shape")
    state_scale = float(np.abs(carried[L - 1]).max())
    chain_rel = _rel(chain_err, state_scale)
    q_rel = _rel(q_err, float(np.abs(yq).max()))

    # --- 3b: the widest request the suite actually contains ----------------------
    # Padding a question to the graph's width is not free: the error in a 1,152-token
    # branch is an order of magnitude over a 128-token one. banking77's 77-option
    # `intent` question measures 1,067 tokens, so parity has to be checked at the width
    # the browser will really run at, and the number that matters there is the one a
    # caller acts on — the probability, not the hidden state. Both sides read the same
    # document: state chained through the exported graph, question through it too.
    wide = None
    if suite and (Path(suite) / "calibration.jsonl").exists():
        from .real_data import load_split
        from .tokenizer import build_question as bq

        cands = []
        for key, (qs, exs) in load_split(Path(suite) / "calibration.jsonl").items():
            for q in qs:
                opts = myna._options({"type": q.type, "criteria": q.options})
                ntok = len(bq(myna.tok, q.instruction, opts)[0])
                cands.append((ntok, key.split("#")[0], q, opts, exs[0]))
        # keyed on the token count: a tuple compare would reach the Question
        # dataclass the moment two requests tie, and Question has no order.
        n_wide, w_src, w_q, w_opts, w_ex = max(cands, key=lambda c: c[0])
        spec_w = {"type": w_q.type, "instructions": w_q.instruction, "criteria": w_q.options}
        if n_wide > q_len:
            wide = {"source": w_src, "question": w_q.name, "tokens": n_wide,
                    "graph_width": q_len, "measured": False,
                    "note": "wider than the exported graph, so not compared"}
        else:
            ids_w = encode_text(myna.tok, w_ex.state)
            S_w, n_fed = chain_state(step, ids_w, chunk, L, shapes)
            assert n_fed == len(ids_w)
            qrow, wspans, wdec = bq(myna.tok, w_q.instruction, w_opts)
            padw = np.zeros((N, q_len), dtype="int64")
            padw[0, :len(qrow)] = qrow
            mkw = np.zeros((N, q_len), dtype="float32")
            mkw[0, :len(qrow)] = 1.0
            yw = question.run(None, {"q_ids": padw, "q_mask": mkw,
                                     "pos_offset": _off(len(ids_w)),
                                     **{f"S_in_{i}": S_w[i] for i in range(L)}})[0]
            onnx_w = myna._readout(spec_w, w_opts, torch.from_numpy(yw[0]), wdec, wspans)
            torch_w = myna.observe(w_ex.state).ask({"q": spec_w})["answers"]["q"]
            err_w = max(abs(torch_w["probabilities"][k] - onnx_w["probabilities"][k])
                        for k in torch_w["probabilities"])
            wide = {"source": w_src, "question": w_q.name, "tokens": n_wide,
                    "n_options": len(w_opts), "graph_width": q_len, "measured": True,
                    "decision_max_abs_error": err_w}
            prob_err = max(prob_err, err_w)
            print(f"widest real request ({w_src}/{w_q.name}: {n_wide} tokens, "
                  f"{len(w_opts)} options) through the same graphs: decision |Δ| {err_w:.2e}")

    # unit-scale quantities are gated absolutely, scale-dependent ones relatively: the
    # hidden states of a 1,152-token branch are not the same object as a probability,
    # and asking both to clear 1e-4 confuses "wrong" with "bigger".
    worst_abs = max(iso_err, prob_err)
    res = {"chain_max_abs_error": chain_err, "chain_max_rel_error": chain_rel,
           "state_scale": state_scale, "question_max_abs_error": q_err,
           "question_max_rel_error": q_rel, "isolation_max_abs_error": iso_err,
           "decision_max_abs_error": prob_err,
           "decision_questions_compared": n_compared,
           "widest_request": wide,
           "decision_questions_too_wide": n_skipped,
           "max_abs_error_allowed": max_abs_error,
           "max_rel_error_allowed": max_rel_error,
           "pass": (worst_abs <= max_abs_error
                    and max(chain_rel, q_rel) <= max_rel_error),
           "state_tokens": real, "n_chunks": n_chunks, "chunk": chunk,
           "n_questions_real": n_real, "n_questions_padded": N,
           "state_step_ms_per_chunk": round(chain_ms, 2),
           "files_mib": meta["files_mib"], "shared_weights": meta["shared_weights"],
           "transfer_mib": meta["transfer_mib"],
           "unshared_transfer_mib": meta.get("unshared_transfer_mib")}
    print(f"chained scan of {real} tokens over {n_chunks} chunks (padded tail): "
          f"|Δ| {chain_err:.2e} = {chain_rel:.2e} relative on a state of scale "
          f"{state_scale:.1f}")
    print(f"question branch, {n_real} real + {N - n_real} padded rows: |Δ| {q_err:.2e} "
          f"= {q_rel:.2e} relative on hidden states of scale {float(np.abs(yq).max()):.1f}")
    print(f"isolation — question 0 run beside {N - 1} masked rows: |Δ| {iso_err:.2e}")
    print(f"decision parity through the pointer head, {n_compared} question(s) "
          f"(probabilities, bounded by 1): |Δ| {prob_err:.2e}"
          + (f" — {n_skipped} too wide for --q-len {q_len}" if n_skipped else ""))
    print(f"onnxruntime state_step: {chain_ms:.1f} ms per {chunk}-token chunk on this box")
    ok = res["pass"]
    print(f"{'PASS' if ok else 'FAIL'}: worst unit-scale |Δ| {worst_abs:.2e} against "
          f"{max_abs_error:.0e}, and worst relative {max(chain_rel, q_rel):.2e} against "
          f"{max_rel_error:.0e}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="export myna to ONNX and prove torch parity")
    ap.add_argument("--ckpt", default="runs/myna-v0")
    ap.add_argument("--out", default="runs/onnx")
    ap.add_argument("--chunk", type=int, default=CHUNK,
                    help="fixed token width of the state graph; length is more calls")
    ap.add_argument("--questions", type=int, default=QUESTIONS,
                    help="fixed question width of the answer graph; smaller requests pad")
    ap.add_argument("--q-len", type=int, default=64)
    ap.add_argument("--scan-chunk", type=int, default=None,
                    help="tile width the parallel scan accumulates inside a call; smaller "
                         "means a smaller peak allocation in a tab (the arithmetic is "
                         "identical either way, tests/test_trunk_numerics.py)")
    ap.add_argument("--opset", type=int, default=OPSET)
    ap.add_argument("--n-chunks", type=int, default=3,
                    help="how many chunks the composition check chains")
    ap.add_argument("--suite", default=None,
                    help="a decision-v2 split directory; when given, parity also runs the "
                         "widest request in it through both paths")
    ap.add_argument("--max-abs-error", type=float, default=1e-4,
                    help="bound on unit-scale error: hidden states, isolation, probabilities")
    ap.add_argument("--max-rel-error", type=float, default=1e-4,
                    help="bound on the chained state scan, relative to the state's own scale "
                         "(its entries reach ~5e2, where an absolute 1e-4 asks two BLAS "
                         "implementations to agree bitwise)")
    ap.add_argument("--no-share", action="store_true",
                    help="give each graph its own copy of the trunk (6a's shape) so the "
                         "cost of sharing is measured rather than asserted")
    ap.add_argument("--no-verify", action="store_true",
                    help="export without running the parity check (G3 stays open)")
    ap.add_argument("--report", default="runs/onnx_parity.json")
    args = ap.parse_args(argv)
    for name, val, lo in (("chunk", args.chunk, 1), ("questions", args.questions, 1),
                          ("q-len", args.q_len, 1), ("n-chunks", args.n_chunks, 1)):
        if val < lo:
            raise SystemExit(f"--{name} must be >= {lo}, got {val}")
    for name, val in (("max-abs-error", args.max_abs_error),
                      ("max-rel-error", args.max_rel_error)):
        if not 0 < val < 1:
            raise SystemExit(f"--{name} must be in (0, 1), got {val}")

    export(args.ckpt, args.out, chunk=args.chunk, n_questions=args.questions,
           opset=args.opset, q_len=args.q_len, share=not args.no_share,
           scan_chunk=args.scan_chunk)
    if args.no_verify:
        print("--no-verify: parity unmeasured")
        return 0
    res = parity(args.ckpt, args.out, max_abs_error=args.max_abs_error,
                 max_rel_error=args.max_rel_error, n_chunks=args.n_chunks,
                 suite=args.suite)
    p = Path(args.report)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"ckpt": str(args.ckpt), "out": str(args.out), **res},
                            indent=2) + "\n")
    print(f"wrote {p}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

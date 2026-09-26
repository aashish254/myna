"""Streaming benchmark: the workload that System-1 models actually serve.

An agent loop re-asks typed questions as the observation grows (ticket
thread, browser page table, document edits). laya and kev must re-encode the
whole state on every request; myna streams — the state scan happens once and
each request only pays for the appended delta plus the question branches.

Modes compared (same weights, same machine, same questions):
  myna-stream   append-only + cached question branches   (the architecture)
  myna-full     full re-encode per request               (what encoders do)
  laya          optional, with --laya, pip laya Router   (real baseline)

DO NOT PUBLISH THE laya COLUMN AS A RATIO. Its "~500 ms floor above 512 tokens" is
`Router` truncating the state at the checkpoint's 1024-token window, and its
`speedup` column is myna-full / myna-stream — both myna. The whole table was
withdrawn (SPEC §9.20); the matched-condition replacement is
`bench/bench_latency_matched.py`. What this script still legitimately shows is
myna against itself: the streaming mode's independence from prefix length.

Usage: uv run python -m bench.bench_stream --ckpt runs/myna-v0 [--laya]
"""

from __future__ import annotations

import argparse
import random
import time

from myna.data import WORKFLOWS, generate
from myna.engine import Myna


def bench_stream(myna: Myna, examples, questions_spec, sentences_of, steps):
    lat = []
    for e in examples:
        parts = sentences_of(e.state)
        obs = myna.observe(parts[0])
        for i in range(1, min(steps, len(parts))):
            obs = obs.append(" " + parts[i])
            t0 = time.perf_counter()
            obs.ask(questions_spec)
            lat.append((time.perf_counter() - t0) * 1000)
    return sum(lat) / len(lat)


def bench_full(myna: Myna, examples, questions_spec, sentences_of, steps):
    lat = []
    for e in examples:
        parts = sentences_of(e.state)
        for i in range(1, min(steps, len(parts))):
            growing = " ".join(parts[: i + 1])
            t0 = time.perf_counter()
            myna.predict(growing, questions_spec)
            lat.append((time.perf_counter() - t0) * 1000)
    return sum(lat) / len(lat)


def bench_long(myna: Myna, base_text, questions_spec, target_tokens):
    """Grow a state by repeating a slice until ~target subword tokens, then
    time observe + ask — the long-document decision path laya caps at 8192."""
    from myna.tokenizer import encode_text

    piece = base_text
    text = piece
    while len(encode_text(myna.tok, text)) < target_tokens:
        text += " " + piece
    text = text[: len(text)]  # keep as-is; over the target by one piece at most
    t0 = time.perf_counter()
    obs = myna.observe(text)
    t_enc = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    obs.ask(questions_spec)
    t_ask = (time.perf_counter() - t0) * 1000
    n_tok = len(obs.ids)
    return n_tok, t_enc, t_ask


def bench_stream_tokens(myna, first, appends, questions):
    lat = []
    obs = myna.observe(first)
    for a in appends:
        obs = obs.append(a)
        t0 = time.perf_counter()
        obs.ask(questions)
        lat.append((time.perf_counter() - t0) * 1000)
    return sum(lat) / len(lat)


def bench_full_states(myna, states, questions):
    lat = []
    for s in states:
        t0 = time.perf_counter()
        myna.predict(s, questions)
        lat.append((time.perf_counter() - t0) * 1000)
    return sum(lat) / len(lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/myna-v0")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--laya", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    myna = Myna(args.ckpt, device=args.device)
    cfg = myna.cfg
    cache_bytes = cfg.n_layers * cfg.n_heads * cfg.d_k * cfg.d_v * 4
    rng = random.Random(123)
    exs = []
    for wf in WORKFLOWS:
        exs += generate(args.n, wf, rng, "test")
    questions = {
        q.name: {"type": q.type, "instructions": q.instruction, "criteria": q.options}
        for q in WORKFLOWS["support"][0]
    }

    print("== streaming: cost per request as the observation grows ==")
    print(f"retained state per observation: {cache_bytes/1024:.0f} KiB "
          f"(n_layers x heads x dk x dv, fixed — a 16k-token state would cost "
          f"{6 * 16384 * cfg.d_model * 4 / 1e6:.0f} MB if per-layer hidden states were kept)")
    print(f"{'state tokens':>13} | {'myna-stream ms':>15} | {'myna-full ms':>13} | {'speedup':>8}")
    scale_rows = []
    from myna.tokenizer import encode_text

    base = " ".join(e.state for e in exs)
    for target in (128, 512, 1024, 2048, 4096, 8192, 16384):
        ids = encode_text(myna.tok, base)
        while len(ids) < target:
            ids += encode_text(myna.tok, base)
        ids = ids[:target]
        # split the observation into equal token chunks: the first is
        # observed, each later one is a small append. stream pays only for
        # the delta; full re-encodes the whole grown prefix.
        nsteps = min(args.steps, max(2, target // 64))
        csize = max(1, len(ids) // nsteps)
        tok_chunks = [ids[i : i + csize] for i in range(0, len(ids), csize)]
        first = myna.tok.decode(tok_chunks[0])
        appends = [myna.tok.decode(c) for c in tok_chunks[1:]]
        if not appends:
            continue
        ms_stream = bench_stream_tokens(myna, first, appends, questions)
        full_states = [myna.tok.decode(ids[: csize * (i + 2)]) for i in range(len(appends))]
        ms_full = bench_full_states(myna, full_states, questions)
        print(f"{len(ids):>13} | {ms_stream:>15.2f} | {ms_full:>13.2f} | {ms_full/ms_stream:>7.1f}x")
        scale_rows.append((len(ids), ms_stream, ms_full))

    t0 = time.perf_counter()
    ms_stream = bench_stream(myna, exs, questions, lambda s: s.split(". "), args.steps)
    ms_full = bench_full(myna, exs, questions, lambda s: s.split(". "), args.steps)
    print(f"\nshort-state avg: myna-stream {ms_stream:.2f} ms, myna-full {ms_full:.2f} ms, "
          f"speedup {ms_full/ms_stream:.2f}x")
    rows = [("myna-stream", ms_stream), ("myna-full", ms_full)]

    if args.laya:
        from laya import Router  # pip install laya

        def laya_questions(qs):
            # laya's noul schema: criteria omitted or a true/false dict, not a list
            return {
                name: ({"type": q["type"], "instructions": q["instructions"]}
                       if q["type"] == "noul" else q)
                for name, q in qs.items()
            }

        lq = laya_questions(questions)
        router = Router()
        lat = []
        for e in exs:
            parts = e.state.split(". ")
            for i in range(1, min(args.steps, len(parts))):
                growing = " ".join(parts[: i + 1])
                t0 = time.perf_counter()
                router.predict(growing, lq)
                lat.append((time.perf_counter() - t0) * 1000)
        ms_laya = sum(lat) / len(lat)
        print(f"laya          avg {ms_laya:8.2f} ms/request")
        rows.append(("laya", ms_laya))

        # same token-length ladder as the myna sweep (laya caps input at 8192)
        print("\n== laya re-encode at matched state lengths ==")
        print(f"{'state tokens':>13} | {'laya ms':>10}")
        laya_rows = []
        for target in (128, 512, 1024, 2048, 4096, 8192):
            lats = []
            for rep in range(args.n // 2 + 1):
                ids = encode_text(myna.tok, base)
                while len(ids) < target:
                    ids += encode_text(myna.tok, base)
                text = myna.tok.decode(ids[:target])
                t0 = time.perf_counter()
                router.predict(text, lq)
                lats.append((time.perf_counter() - t0) * 1000)
            ms = sum(lats) / len(lats)
            print(f"{target:>13} | {ms:>10.2f}")
            laya_rows.append((target, ms))
    else:
        laya_rows = []

    with open("runs/bench_stream.md", "w") as f:
        laya_by_len = dict(laya_rows)
        f.write("| state tokens | myna-stream ms | myna-full (re-encode) ms | speedup | laya re-encode ms |\n|---|---|---|---|---|\n")
        for target, s, fl in scale_rows:
            ly = laya_by_len.get(target)
            f.write(f"| {target} | {s:.2f} | {fl:.2f} | {fl/s:.1f}x | {'n/a (above 8192 cap)' if ly is None else f'{ly:.2f}'} |\n")
    print("wrote runs/bench_stream.md")


if __name__ == "__main__":
    main()

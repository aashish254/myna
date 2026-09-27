"""Diagnostic: does the int8 weight path win at the *matmul*, before the engine?

SPEC §5 P6 6c measured the full predictor fp32 and int8 on the same box and got
3.17x fewer bytes and no milliseconds. This asks the narrower question the
engine's number cannot answer on its own: at the shapes this model actually
feeds `mx.quantized_matmul`, is the kernel faster per call at all?

Two timings per shape, because they mean different things:

* **queued** — enqueue `--iters` matmuls, synchronise once. This is the regime
  the chunked scan lives in (16 rows per call, ~1k sequential steps per state,
  one sync at the end), so it is the one that can explain the engine.
* **synced** — one matmul per call, `mx.eval` between them. Every call pays a
  host/GPU round trip, so this is a floor with a name rather than a kernel time.

Also printed per shape: the bytes each weight actually costs to read. int8 group
64 stores the packed matrix plus a scale and a bias per 64-wide group, which is
where the ~3.6x comes from; the arithmetic (MACs per byte of weight) is what
decides whether fewer bytes can become fewer milliseconds at all. And the
effective GB/s behind each queued number, which is the probe checking itself — a
figure over ~1e3 means the loop measured dispatch rather than a kernel, and the
microseconds beside it are not evidence of anything.

Read `fp32/int8` as "how many times faster int8 is"; below 1.0 int8 is slower.
"""

import argparse
import os
import statistics
import shlex
import sys
import time

import mlx.core as mx

# (K, N, rows) triples: the four linears a state token passes through, at the
# row counts the engine actually uses (chunk = 16 in the scan, 128 for the
# 2-question branch at the widths the latency harness runs).
SHAPES = [(384, 1152, 16), (384, 1152, 128), (384, 1024, 16), (1024, 384, 16),
          (1024, 384, 128), (384, 384, 16)]

ap = argparse.ArgumentParser(description="int8 vs fp32 matmul cost at myna's own shapes")
ap.add_argument("--iters", type=int, default=200, help="matmuls per sync in the queued probe")
ap.add_argument("--trials", type=int, default=9, help="repeats per probe; the median is printed")
ap.add_argument("--group-size", type=int, default=64)
ap.add_argument("--bits", type=int, default=8)
ap.add_argument("--out", help="write the table here as markdown")
args = ap.parse_args()


def timed(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def one(x, w, q, s, b, kind):
    if kind == "fp32":
        return x @ w.T
    return mx.quantized_matmul(x, q, s, b, transpose=True,
                               group_size=args.group_size, bits=args.bits)


def queued(x, w, q, s, b):
    """Enqueue `iters` matmuls of each kind, then sync once.

    Every node is kept in a list and passed to one `mx.eval`: evaluating only the
    last array would leave the other `iters - 1` graphs unreferenced and unrun, and
    the loop would measure dispatch instead of the kernel.
    """
    out = []
    for kind in ("fp32", "int8"):
        for _ in range(3):
            mx.eval(one(x, w, q, s, b, kind))

        def run():
            ys = [one(x, w, q, s, b, kind) for _ in range(args.iters)]
            mx.eval(*ys)

        vals = [timed(run) / args.iters * 1e6 for _ in range(args.trials)]
        out.append(statistics.median(vals))
    return out


def synced(x, w, q, s, b):
    out = []
    for kind in ("fp32", "int8"):
        def run():
            mx.eval(one(x, w, q, s, b, kind))
        vals = [timed(run) * 1e6 for _ in range(args.trials)]
        out.append(statistics.median(vals))
    return out


def main():
    mx.set_default_device(mx.gpu)
    rows = []
    print(f"device=metal  shapes={len(SHAPES)}  iters={args.iters}  trials={args.trials}  "
          f"int8 group {args.group_size} bits {args.bits}  "
          f"load {'/'.join(f'{x:.2f}' for x in os.getloadavg())}\n")
    print(f"{'K':>5} {'N':>5} {'rows':>5} {'fp32 KiB':>9} {'int8 KiB':>9} "
          f"{'queued fp32':>12} {'queued int8':>12} {'q ratio':>8} "
          f"{'fp32 GB/s':>10} {'int8 GB/s':>10} "
          f"{'synced fp32':>12} {'synced int8':>12} {'s ratio':>8}")
    for K, N, M in SHAPES:
        w = mx.random.normal(shape=(N, K)) * 0.02
        q, s, b = mx.quantize(w, group_size=args.group_size, bits=args.bits)
        x = mx.random.normal(shape=(M, K))
        qb, ib = w.nbytes, q.nbytes + s.nbytes + b.nbytes
        qf, qi = queued(x, w, q, s, b)
        sf, si = synced(x, w, q, s, b)
        rows.append((K, N, M, qb, ib, qf, qi, sf, si))
        # effective weight-read bandwidth: if this is ~1e3 GB/s the loop is not
        # running the kernels, so the microsecond column beside it means nothing.
        print(f"{K:5d} {N:5d} {M:5d} {qb/1024:9.0f} {ib/1024:9.0f} "
              f"{qf:9.1f} us {qi:9.1f} us {qf/qi:8.2f} "
              f"{qb/1e9/(qf*1e-6):10.0f} {ib/1e9/(qi*1e-6):10.0f} "
              f"{sf:9.1f} us {si:9.1f} us {sf/si:8.2f}")
    faster = [r for r in rows if r[6] < r[5]]
    print(f"\n{len(rows) - len(faster)}/{len(rows)} shapes are NOT faster in int8 while queued "
          f"(the regime the scan runs in); int8 reads {rows[0][3]/rows[0][4]:.1f}x fewer "
          f"bytes for the first weight.")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# int8 vs fp32 at the matmul — myna's own shapes\n\n")
            f.write("`" + shlex.join(["python", "bench/diag_mlx_int8_gem.py", *sys.argv[1:]]) + "`\n\n")
            f.write(f"MLX on metal, `group_size={args.group_size} bits={args.bits}`, "
                    f"{args.iters} matmuls per sync, median of {args.trials} trials, "
                    f"load average {'/'.join(f'{x:.2f}' for x in os.getloadavg())}.\n\n")
            f.write("| K | N | rows | fp32 weight KiB | int8 weight KiB | queued fp32 us "
                    "| queued int8 us | queued fp32/int8 | fp32 GB/s | int8 GB/s "
                    "| synced fp32 us | synced int8 us | synced fp32/int8 |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            for K, N, M, qb, ib, qf, qi, sf, si in rows:
                f.write(f"| {K} | {N} | {M} | {qb/1024:.0f} | {ib/1024:.0f} | {qf:.1f} "
                        f"| {qi:.1f} | {qf/qi:.2f} | {qb/1e9/(qf*1e-6):.0f} "
                        f"| {ib/1e9/(qi*1e-6):.0f} "
                        f"| {sf:.1f} | {si:.1f} | {sf/si:.2f} |\n")
            f.write("\n")
            f.write(f"{len(rows) - len(faster)}/{len(rows)} shapes are not faster in int8 while "
                    f"queued.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

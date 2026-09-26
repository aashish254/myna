"""P4 4a/4b — myna vs laya latency with the axes actually matched.

SPEC §5 P4 puts gate G2 on one sentence being re-measured rather than re-argued.
Every comparison so far mixed three things at once: a different box (myna on the
M5, laya's published numbers from a datacenter T4), a different call path
(`Router`, which adds language detection and possibly a checkpoint swap), and a
different window. This harness removes all three:

* **one process** — both models resident, and the first timed shape re-run at
  the end so page-cache and thread-pool drift is visible instead of assumed;
  enforced by a `flock` on the output path, because a second copy sharing these
  cores silently invalidates the table (§9.23);
* **direct `Agent.system_one`** — never `Router`;
* **matched inputs** — the same Python `str` object and the same Python `dict`
  of questions go to both engines, so question count and options-per-question
  (3 for `choice`, 2 for `noul`) cannot drift apart;
* **the window reported per side, from each engine's own numbers** — laya packs
  the state and one question into one `max_len` sequence *per question*, and
  `system_one` truncates silently beyond it (its own docstring). A row whose
  per-question sequence stops growing with the state is labelled saturated
  rather than compared.

4b is the same grid read the other way: per-call cost fitted as
`fixed + a*state_tokens + b*questions`, with `observe` and `ask` timed
separately for myna because that split *is* the architecture claim. laya has no
cached-state API, so its only row is end-to-end; the harness prints the
asymmetry rather than inventing a laya "ask-only" number.

Nothing here trains, and no accuracy is claimed. Answers are checked only for
shape and non-degeneracy, because an engine that answered nothing would look
fast.

Usage (the laya checkout's directory name ends in a space — quote it):
  PYTHONPATH="/Users/aashish/github contribution /GIT/laya" \\
    .venv/bin/python -m bench.bench_latency_matched --myna-ckpt runs/myna-v0
  .venv/bin/python -m bench.bench_latency_matched --no-laya   # myna half only
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

from myna.engine import Myna
from myna.tokenizer import encode_text

# Held at module scope so the descriptor stays open (and the lock held) for the
# life of the process.
_LOCK_HANDLE = None


def acquire_lock(out_path: str | Path):
    """Refuse to time while another copy of this harness is running.

    Two processes each asking torch for 8 intra-op threads on a 10-core laptop
    do not produce two slower tables — they produce one table whose absolute
    column is neither machine's, and nothing in the output says so. Found the
    hard way: a relaunch left a duplicate running for ~40 s, its output
    interleaved in the same log. `flock` is per open file description, so this
    conflicts with a second copy in the *same* process too, which is what makes
    it testable.
    """
    global _LOCK_HANDLE
    try:
        import fcntl
    except ImportError:  # a platform without flock: say so, never pretend to guard
        print("no fcntl on this platform: single-instance guard unavailable")
        return None
    lock_path = Path(str(out_path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise SystemExit(
            f"another run holds {lock_path}; timings from two processes sharing these cores "
            "are not a measurement. Wait for it, or point --out at a different file only if "
            "you accept that both tables are contaminated.")
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    _LOCK_HANDLE = fh
    return fh


# The laya checkpoint revision their published CPU latency table used (55cf4c4,
# BENCHMARKS.md "Server CPU"), so the window/head budgets here are theirs.
DEFAULT_LAYA_CKPT = str(
    Path.home() / ".cache/huggingface/hub/models--convaiinnovations--laya/snapshots"
    / "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/typed-decisions")

# The ticket from laya's own research/scripts/bench_latency.py, as one string:
# `serialize_state` passes a str through untouched, and myna's engine takes text.
STATE_BODY = ("Hi, my Stripe payouts have failed for 3 days and I am losing "
              "sales. Please help ASAP. ")
Q_CHOICE = {"type": "choice", "instructions": "Which team should handle this?",
            "criteria": {"billing": "payments", "technical": "bugs and integrations",
                         "sales": "pricing"}}
Q_NOUL = {"type": "noul", "instructions": "Does the text express urgency?"}


def state_text(reps: int) -> str:
    return STATE_BODY * reps


def matched_questions(n: int) -> dict:
    """laya's own ladder: a 3-option `choice` alternating with a `noul`, q0 = choice."""
    return {f"q{i}": (Q_CHOICE if i % 2 == 0 else Q_NOUL) for i in range(n)}


def non_degenerate(answers: dict) -> bool:
    """Some confidence strictly between 0 and 1.

    A dead head (all-zero logits, or temperature 0) answers fast and means
    nothing; this is the cheapest check that tells the two apart."""
    ps = [float(a["confidence"]) for a in answers.values() if "confidence" in a]
    ps += [float(a["noul"]) for a in answers.values() if "noul" in a]
    for a in answers.values():  # score answers carry a distribution, not a confidence
        ps += [float(p) for p in a.get("probabilities", {}).values()]
    return bool(ps) and any(0.0 < p < 1.0 for p in ps)


def require_full_answers(out: dict, questions: dict, who: str) -> None:
    """Every question answered, nothing degenerate — or the run stops.

    A skipped question is a free call: whichever engine errored internally and
    answered fewer questions would win the ratio."""
    ans = out["answers"]
    missing = set(questions) - set(ans)
    if missing:
        raise AssertionError(f"{who} answered {len(ans)} of {len(questions)} questions; "
                             f"missing {sorted(missing)[:4]}")
    if not non_degenerate(ans):
        raise AssertionError(f"{who} returned only 0/1 confidences — not a live head")


def stats(times_ms: list[float]) -> dict:
    t = sorted(times_ms)
    n = len(t)

    def pct(p):
        return t[min(n - 1, max(0, int(round(p / 100 * (n - 1)))))]

    return {"n": n, "p50_ms": round(pct(50), 3), "p95_ms": round(pct(95), 3),
            "mean_ms": round(statistics.fmean(t), 3), "min_ms": round(t[0], 3)}


def timed(fn, warmup: int, reps: int) -> tuple[dict, object]:
    for _ in range(warmup):
        fn()
    ts = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return stats(ts), out


def laya_tokens_per_row(input_tokens: int, n_questions: int) -> float:
    """laya's `usage.input_tokens` sums attention over every question row, and a
    row carries the whole state, so the window it used is total/n."""
    return round(input_tokens / max(1, n_questions), 1)


def saturation_note(row_tokens: list[float], max_len: int) -> str | None:
    """Where laya's per-row sequence stops growing, `system_one` is truncating.

    Compared against the window, not against a guessed question cost: the number
    is the system's own."""
    if len(row_tokens) >= 2 and row_tokens[-1] <= row_tokens[-2]:
        return (f"laya per-question sequence {row_tokens[-2]} -> {row_tokens[-1]} tokens while the "
                f"state grew: at or past max_len {max_len}, its state is truncated, myna's is not")
    if row_tokens and row_tokens[-1] >= max_len:
        return f"laya per-question sequence {row_tokens[-1]} >= max_len {max_len}: state truncated"
    return None


def fit_cost(rows: list[tuple[float, float, float]]) -> dict:
    """Least squares for t = fixed + per_state_token*L + per_question*Q.

    Two guards, both learned from a fit that printed confident nonsense:

    * an axis held constant across the grid is dropped rather than fitted — a
      zero-variance column is collinear with the intercept, and the solver will
      happily split the two and report a slope of 0.0 next to a negative fixed
      cost;
    * fewer than two degrees of freedom is reported as not identifiable, with
      the numbers, instead of a rounded slope.

    R^2 rides along either way: a slope published without the variance it left
    behind is the §9.1 failure mode, not a measurement."""
    Ls = [r[0] for r in rows]
    Qs = [r[1] for r in rows]
    names, cols = ["fixed"], [[1.0] * len(rows)]
    if len(set(Ls)) > 1:
        names.append("L")
        cols.append(Ls)
    if len(set(Qs)) > 1:
        names.append("Q")
        cols.append(Qs)
    out = {"n_rows": len(rows), "n_params": len(names), "terms": names}
    if len(rows) < len(names) + 2:
        return {**out, "identifiable": False,
                "reason": f"{len(rows)} rows cannot pin {len(names)} coefficients with 2 dof"}
    X = torch.tensor(list(map(list, zip(*cols))), dtype=torch.float64)
    y = torch.tensor([r[2] for r in rows], dtype=torch.float64)
    beta = torch.linalg.solve(X.T @ X, X.T @ y)
    pred = X @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    coef = dict(zip(names, [float(b) for b in beta]))
    out.update({"identifiable": True,
                "fixed_ms": round(coef["fixed"], 3),
                "per_state_token_us": round(coef.get("L", 0.0) * 1000, 2),
                "per_question_ms": round(coef.get("Q", 0.0), 3),
                "r2": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else None})
    if coef["fixed"] < 0:
        out["note"] = ("negative intercept: cost rises super-linearly with state length, so the "
                       "linear fit understates short states and should not be extrapolated")
    return out


def fit_text(name: str, v: dict) -> str:
    if not v.get("identifiable"):
        return (f"- `{name}`: not identifiable — {v['reason']} "
                f"({v['n_rows']} rows, terms {v['terms']})")
    line = (f"- `{name}`: fixed **{v['fixed_ms']} ms**, **{v['per_state_token_us']} µs** per "
            f"state token, **{v['per_question_ms']} ms** per question, R² {v['r2']} "
            f"({v['n_rows']} rows, terms {'+'.join(v['terms'])})")
    return line + (f"  \n  _{v['note']}_" if "note" in v else "")



def ratio_rows(laya: dict, myna_e2e: dict, myna_ask: dict) -> list[dict]:
    rows = []
    for n in sorted(laya):
        a, b, c = laya[n]["p50_ms"], myna_e2e[n]["p50_ms"], myna_ask[n]["p50_ms"]
        rows.append({"questions": n, "laya_system_one_p50_ms": a,
                     "myna_predict_p50_ms": b, "speedup_e2e": round(a / b, 2),
                     "myna_ask_only_p50_ms": c, "speedup_stream": round(a / c, 2)})
    return rows


def table(rows: list[dict], title: str) -> str:
    if not rows:
        return f"{title}\n\n_(no rows)_\n"
    cols = list(rows[0])
    lines = [title, "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def markdown(res: dict) -> str:
    m = res["meta"]
    fits = res["fit"]
    lines = [f"# Matched-condition latency — {m['platform']}, device `{m['device']}`",
             "",
             f"- One process, direct calls, {m['ns']} questions, state = the laya bench ticket "
             f"({m['myna_state_tokens']} myna tokens); threads intra {m['intra_threads']}, "
             f"inter {m['inter_threads']}; warmup {m['warmup']}, reps {m['reps']}.",
             f"- myna `{m['myna_ckpt']}`, {m['myna_params']/1e6:.2f}M params, "
             f"weights {m.get('myna_dtype', 'unknown')}.",
             ]
    if m["laya_version"]:
        lines.append(f"- laya {m['laya_version']} `{m['laya_ckpt']}`, "
                     f"{m['laya_params']/1e6:.2f}M params, `Agent.system_one`, "
                     f"max_len {m['laya_max_len']}, head_max_len {m['laya_head_max_len']}, "
                     f"weights {m.get('laya_dtype', 'unknown')}, autocast "
                     f"{'enabled' if m.get('laya_amp_enabled') else 'off'} "
                     f"({m.get('laya_autocast')}).")
    else:
        lines.append("- laya **not measured in this run** — no ratio is reported (G2, §9.1).")
    lines.append("")
    if res["ratio"]:
        lines.append(table(res["ratio"], "## p50, same box, same process, same inputs"))
    if res.get("ladder"):
        lines.append(table(res["ladder"], "## Cost ladder (4b)"))
    lines.append("## Fitted cost, `t = fixed + per_state_token*L + per_question*Q`")
    for k, v in fits.items():
        lines.append(fit_text(k, v))
    lines += ["", "Fitted slopes are least-squares over the ladder above; every sample is in "
              "`latency_matched.json`. Accuracy is not measured here and G1 is not open to "
              "inference from this table.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="matched-condition myna vs laya latency (G2)")
    ap.add_argument("--myna-ckpt", default="runs/myna-v0")
    ap.add_argument("--laya-ckpt", default=DEFAULT_LAYA_CKPT,
                    help="a laya checkpoint dir; typed-decisions at their published revision")
    ap.add_argument("--no-laya", action="store_true", help="run only the myna half")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"],
                    help="one value for both engines — this is a same-box comparison")
    ap.add_argument("--intra-threads", type=int, default=8,
                    help="laya's published laptop-CPU advice: physical cores, not every vCPU")
    ap.add_argument("--inter-threads", type=int, default=1,
                    help="system_one is one forward pass; inter-op parallelism only contends")
    ap.add_argument("--ns", default="1,5,10,50", help="question counts, comma separated")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--ladder", default="64,128,256,512,1024",
                    help="state-token ladder for the 4b fit, in each side's own tokenizer")
    ap.add_argument("--state-reps", type=int, default=3, help="ticket repeats for section 1")
    ap.add_argument("--out", default="runs/latency_matched.json")
    ap.add_argument("--md", default="runs/latency_matched.md")
    args = ap.parse_args(argv)

    # before anything is timed, and before the thread pinning that would make a
    # contaminated run look like a fast one
    acquire_lock(args.out)

    torch.set_num_threads(args.intra_threads)
    try:
        torch.set_num_interop_threads(args.inter_threads)
    except Exception as e:  # a pool may already exist; report, never assume
        print(f"could not set inter-op threads: {type(e).__name__}: {e}")
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    ladder = [int(x) for x in args.ladder.split(",") if x.strip()]

    print("== 0. load: one process, both engines ==")
    myna = Myna(args.myna_ckpt, device=args.device)
    laya_ver = laya_max_len = laya_head_max = laya_params = laya_tok = laya_dtype = None
    if not args.no_laya:
        try:
            import laya
        except ImportError as e:
            print(f"laya not importable ({e}); running the myna half only — the ratio is withdrawn")
            args.no_laya = True
        else:
            t0 = time.perf_counter()
            laya_model = laya.load(args.laya_ckpt, device=args.device)
            laya_ver = getattr(laya, "__version__", "unknown")
            laya_max_len = int(laya_model.cfg.get("max_len", 512))
            laya_head_max = int(laya_model.cfg.get("head_max_len", 192))
            laya_tok = laya_model.tok
            laya_params = sum(p.numel() for p in laya_model.model.parameters())
            laya_dtype = str(next(laya_model.model.parameters()).dtype)
            print(f"laya {laya_ver} from {args.laya_ckpt} in {time.perf_counter()-t0:.1f}s: "
                  f"max_len={laya_max_len} head_max_len={laya_head_max} "
                  f"{laya_params/1e6:.1f}M params, weights {laya_dtype}, "
                  f"amp={laya_model.amp_enabled} autocast_dtype={laya_model.dtype}", flush=True)
    myna_params = sum(p.numel() for p in myna.model.parameters())
    myna_dtype = str(next(myna.model.parameters()).dtype)
    print(f"myna {myna_params/1e6:.2f}M params, weights {myna_dtype}, device {args.device}",
          flush=True)

    text = state_text(args.state_reps)
    myna_state_tokens = len(encode_text(myna.tok, text))
    res = {"meta": {"platform": platform.platform(), "machine": platform.machine(),
                    "python": sys.version.split()[0], "torch": torch.__version__,
                    "cpus": os.cpu_count(), "device": args.device,
                    "intra_threads": torch.get_num_threads(),
                    "inter_threads": torch.get_num_interop_threads(),
                    "warmup": args.warmup, "reps": args.reps, "ns": ns,
                    "ladder": ladder, "state_reps": args.state_reps,
                    "myna_state_tokens": myna_state_tokens,
                    "myna_ckpt": args.myna_ckpt, "myna_params": myna_params,
                    "laya_version": laya_ver, "laya_ckpt": None if args.no_laya else args.laya_ckpt,
                    "laya_params": laya_params, "laya_max_len": laya_max_len,
                    "laya_head_max_len": laya_head_max,
                    "myna_dtype": myna_dtype, "laya_dtype": laya_dtype,
                    "laya_autocast": None if args.no_laya else str(laya_model.dtype),
                    "laya_amp_enabled": None if args.no_laya else laya_model.amp_enabled,
                    "myna_window": "unbounded — the state is scanned once, streaming",
                    "inference_only": True},
           "matched": {}, "ladder": [], "fit": {}, "ratio": None}

    print("\n== 1. matched calls: same str state, same dict of questions ==")
    print(f"{'q':>4} | {'laya p50/p95 ms':>18} | {'myna e2e p50/p95':>19} | "
          f"{'myna ask-only p50':>17} | tokens: laya row x q / myna state+q")
    laya_by_n, e2e_by_n, ask_by_n = {}, {}, {}
    for n in ns:
        qs = matched_questions(n)
        row = {"questions": n}
        if not args.no_laya:
            s, out = timed(lambda: laya_model.system_one(text, qs), args.warmup, args.reps)
            require_full_answers(out, qs, "laya")
            s["input_tokens"] = out["usage"]["input_tokens"]
            s["tokens_per_question_row"] = laya_tokens_per_row(s["input_tokens"], n)
            row["laya"] = s
            laya_by_n[n] = s
        s, out = timed(lambda: myna.predict(text, qs), args.warmup, args.reps)
        require_full_answers(out, qs, "myna")
        s["state_tokens"] = out["usage"]["state_tokens"]
        s["question_tokens"] = out["usage"]["question_tokens"]
        row["myna_e2e"] = s
        e2e_by_n[n] = s
        obs = myna.observe(text)
        s, _ = timed(lambda: obs.ask(qs), args.warmup, args.reps)
        row["myna_ask_only"] = s
        ask_by_n[n] = s
        res["matched"][n] = row
        lt = (f"{row['laya']['p50_ms']:>8.2f}/{row['laya']['p95_ms']:<8.2f}"
              if "laya" in row else " " * 18)
        e = row["myna_e2e"]
        toks = (f"{row['laya']['tokens_per_question_row']}x{n} / {e['state_tokens']}+{e['question_tokens']}"
                if "laya" in row else f"- / {e['state_tokens']}+{e['question_tokens']}")
        print(f"{n:>4} | {lt} | {e['p50_ms']:>9.2f}/{e['p95_ms']:<8.2f} "
              f"| {row['myna_ask_only']['p50_ms']:>17.2f} | {toks}", flush=True)

    print("\n== 2. cost ladder (4b): state length x question count ==")
    print(f"{'state tok':>9} {'q':>4} | {'laya p50':>9} {'tok/row':>8} | "
          f"{'myna obs':>9} {'ask':>7} {'e2e':>7}")
    base_ids = encode_text(myna.tok, state_text(60))
    fits_laya, fits_obs, fits_ask, fits_e2e = [], [], [], []
    seen_row_tokens: list[float] = []
    for target in ladder:
        ids = base_ids[:target]
        text_i = myna.tok.decode(ids)
        st = len(encode_text(myna.tok, text_i))
        # The state scan does not depend on the question set, so it is timed once
        # per length. Timing it inside the question loop both wasted reps and left
        # the fit with a constant Q column collinear with the intercept.
        s_obs, _ = timed(lambda: myna.observe(text_i), args.warmup, args.reps)
        fits_obs.append((st, 0.0, s_obs["p50_ms"]))
        ob = myna.observe(text_i)
        for n in (1, 10):
            qs = matched_questions(n)
            entry = {"state_tokens_myna": st, "questions": n,
                     "myna_observe_p50_ms": s_obs["p50_ms"]}
            if not args.no_laya:
                lids = laya_tok(text_i, add_special_tokens=False)["input_ids"]
                ltext = laya_tok.decode(lids[:target])
                lt = len(laya_tok(ltext, add_special_tokens=False)["input_ids"])
                s, out = timed(lambda: laya_model.system_one(ltext, qs), args.warmup, args.reps)
                require_full_answers(out, qs, "laya")
                per_row = laya_tokens_per_row(out["usage"]["input_tokens"], n)
                entry.update({"laya_state_tokens": lt, "laya_p50_ms": s["p50_ms"],
                              "laya_p95_ms": s["p95_ms"], "laya_tokens_per_row": per_row})
                if n == 1:
                    seen_row_tokens.append(per_row)
                    # "" not None: this column is printed into the artifact table,
                    # and "None" reads like a value the harness measured.
                    entry["laya_truncated"] = saturation_note(seen_row_tokens, laya_max_len) or ""
                fits_laya.append((lt, n, s["p50_ms"]))
                ltxt = f"{s['p50_ms']:>9.2f} {per_row:>8.1f}"
            else:
                ltxt = " " * 18
            s_ask, _ = timed(lambda: ob.ask(qs), args.warmup, args.reps)
            s_e2e, _ = timed(lambda: myna.predict(text_i, qs), args.warmup, args.reps)
            entry.update({"myna_ask_p50_ms": s_ask["p50_ms"],
                          "myna_e2e_p50_ms": s_e2e["p50_ms"]})
            fits_ask.append((st, n, s_ask["p50_ms"]))
            fits_e2e.append((st, n, s_e2e["p50_ms"]))
            res["ladder"].append(entry)
            trunc = entry.get("laya_truncated")
            print(f"{st:>9} {n:>4} | {ltxt} | {s_obs['p50_ms']:>9.2f} {s_ask['p50_ms']:>7.2f} "
                  f"{s_e2e['p50_ms']:>7.2f}  {'' if not trunc else '  << LAYA TRUNCATED'}",
                  flush=True)
    res["fit"] = {"myna_observe": fit_cost(fits_obs), "myna_ask": fit_cost(fits_ask),
                  "myna_e2e": fit_cost(fits_e2e)}
    if not args.no_laya and len(fits_laya) >= 3:
        res["fit"]["laya_e2e"] = fit_cost(fits_laya)
    for k, v in res["fit"].items():
        print(fit_text(k, v), flush=True)

    if laya_by_n:
        res["ratio"] = ratio_rows(laya_by_n, e2e_by_n, ask_by_n)
        print("\n== 3. the ratio, recomputed on equal footing ==")
        print(table(res["ratio"], "p50, same box/process/inputs"), flush=True)
    else:
        print("\n== 3. no laya rows: the ratio is WITHDRAWN, not projected (G2, §9.1) ==")

    print("\n== 4. drift check: section 1's first shape again, nothing unloaded ==")
    qs1 = matched_questions(ns[0])
    s1, _ = timed(lambda: myna.predict(text, qs1), 0, args.reps)
    base = res["matched"][ns[0]]
    print(f"myna q={ns[0]}: first {base['myna_e2e']['p50_ms']} ms, repeat {s1['p50_ms']} ms "
          f"({(s1['p50_ms']/base['myna_e2e']['p50_ms']-1)*100:+.1f}%)")
    res["drift"] = {"myna_repeat": s1, "myna_first": base["myna_e2e"]}
    if not args.no_laya:
        s0, _ = timed(lambda: laya_model.system_one(text, qs1), 0, args.reps)
        print(f"laya q={ns[0]}: first {base['laya']['p50_ms']} ms, repeat {s0['p50_ms']} ms "
              f"({(s0['p50_ms']/base['laya']['p50_ms']-1)*100:+.1f}%)")
        res["drift"]["laya_repeat"] = s0
        res["drift"]["laya_first"] = base["laya"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    Path(args.md).write_text(markdown(res))
    print(f"\nwrote {args.out} and {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

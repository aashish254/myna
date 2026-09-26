"""Train a Myna checkpoint on the synthetic typed-decisions corpus.

Usage:  uv run python -m myna.train [--steps 4000] [--device mps]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .data import WORKFLOWS, generate
from .model import MynaConfig, MynaModel, typed_loss
from .tokenizer import question_tensors, batch_question_tensors, train_tokenizer, encode_text


def draw_batch(pool, batch, rng):
    """Distinct examples only, shortening the batch rather than repeating one row.

    With replacement a singleton pool yields `batch` copies of the same example: the
    step drives its loss to ~0 and contributes almost no gradient. decision-v2 hits
    this constantly because option descriptions are randomized per row, so many
    question-sets hold a single row."""
    return rng.sample(pool, min(batch, len(pool)))


def build_batch(exs, tok, questions, device):
    """questions: list[Question] fixed for the batch (one question set shared
    across rows; for per-row sets use build_row_batch)."""
    qt = question_tensors(tok, [(q.instruction, q.options) for q in questions])
    texts = [e.state for e in exs]
    ids = [encode_text(tok, t) for t in texts]
    ls = max(len(x) for x in ids)
    state = torch.zeros(len(ids), ls, dtype=torch.int64)
    lens = torch.tensor([len(x) for x in ids], dtype=torch.int64)
    for i, x in enumerate(ids):
        state[i, : len(x)] = torch.tensor(x)
    gold = torch.tensor([e.gold for e in exs], dtype=torch.int64)
    return {
        "state_ids": state.to(device),
        "state_len": lens.to(device),
        "gold": gold.to(device),
        "has_gold": torch.ones_like(gold, dtype=torch.bool).to(device),
        **{k: v.to(device) for k, v in qt.items()},
    }


def build_row_batch(items, tok, device, paraphraser=None, rng=None):
    """items: list[(questions, Example)] — one question set PER ROW.

    Rows may differ in question count and option count; both pad to the batch
    maximum, and `has_gold` marks the cells that belong to a real question so a
    mixed batch trains on the union of its cells. This is what lets 32 boolq
    rows — 32 different instructions, one question each — share one forward
    pass instead of running 32 batches of one.

    With `paraphraser`, each row's set is replaced by a drawn phrasing (SPEC §5
    P1: the suite's own wording is never trained on).
    """
    if paraphraser is not None:
        items = [(paraphraser.draw(ex.workflow, qs, rng), ex) for qs, ex in items]
    qt = batch_question_tensors(tok, [[(q.instruction, q.options) for q in qs] for qs, _ in items])
    n_q = max(len(qs) for qs, _ in items)
    ids = [encode_text(tok, ex.state) for _qs, ex in items]
    ls = max(len(x) for x in ids)
    state = torch.zeros(len(ids), ls, dtype=torch.int64)
    lens = torch.tensor([len(x) for x in ids], dtype=torch.int64)
    for i, x in enumerate(ids):
        state[i, : len(x)] = torch.tensor(x)
    gold = torch.zeros(len(items), n_q, dtype=torch.int64)
    has_gold = torch.zeros(len(items), n_q, dtype=torch.bool)
    for i, (qs, ex) in enumerate(items):
        gold[i, : len(qs)] = torch.tensor(ex.gold[: len(qs)], dtype=torch.int64)
        has_gold[i, : len(qs)] = True
    return {
        "state_ids": state.to(device),
        "state_len": lens.to(device),
        "gold": gold.to(device),
        "has_gold": has_gold.to(device),
        **{k: v.to(device) for k, v in qt.items()},
    }


def question_tokens(tok, questions, cache=None):
    """The row's longest question, in tokens — the width its batch row pads to.

    Measured with the tokenizer the batch will actually use: a chars/4 rule is
    off by 2-4x on a small BPE, and this number is what the memory budget is
    made of. Memoized on the question set, which a suite reuses across
    thousands of rows, so the whole corpus pays it once per schema.
    """
    key = id(questions)
    if cache is not None and key in cache:
        return cache[key]
    lq = max(len(encode_text(tok, q.instruction))
             + sum(len(encode_text(tok, o)) + 1 for o in q.options) + 1  # +1 [OPT] each, +1 [DECIDE]
             for q in questions)
    if cache is not None:
        cache[key] = lq
    return lq


def draw_row_batch(items, batch, max_q_cells, rng, tok, cache=None):
    """`batch` distinct rows whose padded question branch stays under
    `max_q_cells` = rows x questions x question-tokens.

    That product is the axis the machine dies on, and it is *not* the option
    axis: bench/mem_profile.py measures 1.6 MiB of retained activations per
    question-token position against 3 KiB per option cell — a factor of five
    hundred (a 77-option question adds 2.7 MiB to a 24-slot batch). Both
    questions and question tokens pad to the batch maximum, so one long
    instruction mixed in with two short ones charges the whole batch for the
    long one. The first picked row always goes in, so a batch is never empty.
    """
    out, seen, n_max, lq_max = [], set(), 0, 0
    for _ in range(batch * 40):
        if len(out) == batch:
            break
        i = rng.randrange(len(items))
        if i in seen:
            continue
        qs = items[i][0]
        n, lq = len(qs), question_tokens(tok, qs, cache)
        if out and (len(out) + 1) * max(n, n_max) * max(lq, lq_max) > max_q_cells:
            continue
        seen.add(i)
        n_max, lq_max = max(n, n_max), max(lq, lq_max)
        out.append(items[i])
    return out


def worst_case_tokens(tok, paraphraser, items, cache):
    """Pre-price every phrasing of every set at its *longest* variant.

    `draw_row_batch` budgets `rows x questions x question-tokens`, and under
    paraphrasing the token count is a random variable — so pricing the draw the
    sampler happened to take would let one step silently exceed the memory the
    run was sized for. Writing the max into the length cache for all eight
    variant objects makes the budget an upper bound again, at the cost of a
    slightly smaller batch.
    """
    by_key = {}
    for qs, ex in items:
        by_key.setdefault(ex.workflow, qs)
    for key, qs in by_key.items():
        sets = paraphraser.variant_set(key, qs)
        lengths = [question_tokens(tok, v) for v in sets]
        worst = max(lengths)
        for v in sets:
            cache[id(v)] = worst
        cache[id(qs)] = worst
    return len(by_key)


def macro_acc(m):
    """Unweighted mean over the per-(source, question) accuracies in an
    `evaluate()` map — the brier/ece sidecars are excluded by design."""
    names = [k for k in m if not k.endswith((":brier", ":ece"))]
    return sum(m[k] for k in names) / max(len(names), 1)


def evaluate(model, tok, splits, device, temperature=1.0, paraphrase_index=None):
    """splits: {key: (questions, examples)} — synthetic workflows and real
    suite groups both fit this shape.

    `paraphrase_index`: score each group under that phrasing of its instruction
    instead of the wording on disk — how the held-out wording is measured."""
    from .paraphrase import paraphrase_questions, source_of

    model.eval()
    rows = {}
    ece_bin = {}  # key -> (sum_conf, sum_correct, count) aggregated in 10 bins
    with torch.no_grad():
        for wf, (questions, data) in splits.items():
            if paraphrase_index is not None:
                questions = paraphrase_questions(questions, source_of(wf), paraphrase_index)
            for i in range(0, len(data), 64):
                exs = data[i : i + 64]
                b = build_batch(exs, tok, questions, device)
                logits = model(b["state_ids"], b["state_len"], b["q_ids"], b["q_mask"],
                               b["span_mat"], b["opt_valid"], b["decide_idx"]) / temperature
                probs = F.softmax(logits, dim=-1)
                pred = probs.argmax(-1)
                for n, q in enumerate(questions):
                    hit = (pred[:, n] == b["gold"][:, n]).float()
                    key = f"{wf}/{q.name}"
                    acc, tot = rows.get(key, (0.0, 0))
                    rows[key] = (acc + float(hit.sum()), tot + len(exs))
                    conf = probs[:, n].max(-1).values
                    bins = (conf * 10).clamp(max=9).long()
                    for bi in range(10):
                        sel = bins == bi
                        if not bool(sel.any()):
                            continue
                        c, s, m = ece_bin.get(key, (0.0, 0.0, 0))
                        ece_bin[key] = (
                            c + float(conf[sel].sum()),
                            s + float(hit[sel].sum()),
                            m + int(sel.sum()),
                        )
                    if q.type == "noul":
                        p = probs[:, n, 1]
                        g = b["gold"][:, n].float()
                        bk, bt = rows.get(key + ":brier", (0.0, 0))
                        rows[key + ":brier"] = (bk + float(((p - g) ** 2).sum()), bt + len(exs))
    out = {k: v[0] / max(v[1], 1) for k, v in rows.items()}
    for key, (c, s, m) in ece_bin.items():
        out[f"{key}:ece"] = abs(c - s) / max(m, 1)  # single-bin-per-example ECE (M-norm, 10 bins)
    return out


def fit_temperature(model, tok, splits, device):
    """1-D grid on NLL — calibration is a single scalar per checkpoint."""
    best, best_t = 1e18, 1.0
    model.eval()
    with torch.no_grad():
        for t in [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0]:
            tot, cnt = 0.0, 0
            for wf, (questions, data) in splits.items():
                for i in range(0, min(len(data), 512), 64):
                    b = build_batch(data[i : i + 64], tok, questions, device)
                    logits = model(b["state_ids"], b["state_len"], b["q_ids"], b["q_mask"],
                                   b["span_mat"], b["opt_valid"], b["decide_idx"]) / t
                    lp = F.log_softmax(logits, dim=-1)
                    sel = lp.gather(-1, b["gold"][..., None]).squeeze(-1)
                    tot, cnt = tot - float(sel.sum()), cnt + sel.numel()
            if tot / cnt < best:
                best, best_t = tot / cnt, t
    return best_t


def held_out_eval(model, tok, split, device, temperature):
    """Score a split on the reserved ninth phrasing — the only place it is read.

    Kept as its own function so `EVAL_INDEX` has one call site a test can pin: a
    `paraphrase_index=0` slipped in here would silently score the training
    wording and call it a transfer result.
    """
    from .paraphrase import EVAL_INDEX

    return evaluate(model, tok, split, device, temperature, paraphrase_index=EVAL_INDEX)


def temperature_source(data):
    """Which split the scalar is fit on: the suite's own `calibration` group,
    else `dev`.

    dev is the fallback for the synthetic corpus, which ships no calibration
    split — but then the reported dev accuracy is fit on the rows it is scored
    on, and the printed line says so rather than leaving it implied."""
    calib = data.get("calibration")
    return (calib, "calibration") if calib else (data["dev"], "dev")


def resolve_device(name):
    """`auto` prefers MPS, then CUDA, then CPU.

    Resolving to mps/cpu only meant a CUDA cloud box — the structural fix for
    the M5's memory ceiling — silently trained on CPU."""
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_needle_batch(exs, tok, device, entity):
    """One shared question (fixed entity) across the batch; the needle varies."""
    from .longctx import DESKS
    instr = f"Which desk owns {entity}?"
    qt = question_tensors(tok, [(instr, list(DESKS))])
    ids = [encode_text(tok, e.state) for e in exs]
    lens = torch.tensor([len(x) for x in ids], dtype=torch.int64)
    Lmax = max(len(x) for x in ids)
    state = torch.zeros(len(ids), Lmax, dtype=torch.int64)
    for i, x in enumerate(ids):
        state[i, : len(x)] = torch.tensor(x)
    gold = torch.tensor([[e.gold] for e in exs], dtype=torch.int64)
    return {
        "state_ids": state.to(device), "state_len": lens.to(device), "gold": gold.to(device),
        "Lmax": Lmax,
        **{k: v.to(device) for k, v in qt.items()},
    }


def train_long_context(args, rng, device):
    """V1-D: fine-tune (or train) on needle-in-long-context recall, truncated
    backprop so only the trailing grad window carries gradient."""
    from .longctx import DESKS, ENTITIES, FILLER, Needle, make_needle

    if args.init:
        ip = Path(args.init)
        tok = Tokenizer.from_file(str(ip / "tokenizer.json"))
        ck = torch.load(ip / "model.pt", map_location=device, weights_only=False)
        cfg = MynaConfig(**ck["cfg"])
        model = MynaModel(cfg)
        model.load_state_dict(ck["state_dict"])
    else:
        cfg = MynaConfig(vocab=args.vocab)
        model = MynaModel(cfg)
        tok = train_tokenizer([" ".join(FILLER), " ".join(DESKS), " ".join(ENTITIES)],
                              vocab_size=args.vocab)
    model.to(device)
    print(f"long-context: state={args.long_context} grad={args.grad_tokens} "
          f"params={sum(p.numel() for p in model.parameters())}", flush=True)

    split = max(0, args.long_context - args.grad_tokens)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    ema, t0 = None, time.time()
    for step in range(args.steps):
        model.train()
        entity = ENTITIES[rng.randrange(len(ENTITIES))]
        exs = [make_needle(tok, args.long_context, rng, tail_cap=args.grad_tokens,
                           entity=entity) for _ in range(args.batch)]
        b = build_needle_batch(exs, tok, device, entity)
        logits = model.forward_truncated(b["state_ids"], split, b["state_len"], b["q_ids"],
                                         b["q_mask"], b["span_mat"], b["opt_valid"], b["decide_idx"])
        loss = typed_loss(logits, b["gold"], torch.ones_like(b["gold"], dtype=torch.bool))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        acc = (logits.argmax(-1) == b["gold"]).float().mean().item()
        ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        if step % 100 == 0 or step == args.steps - 1:
            print(f"step {step:5d}  loss {loss.item():.3f}  ema {ema:.3f}  "
                  f"batch-acc {acc:.3f}  {time.time()-t0:.0f}s", flush=True)
        if args.eval_every and step and step % args.eval_every == 0 and step != args.steps - 1:
            out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
            tok.save(str(out / "tokenizer.json"))
            torch.save({"state_dict": model.state_dict(), "cfg": vars(cfg),
                        "temperature": 1.0, "step": step}, out / "model_last.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=700)
    ap.add_argument("--out", default="runs/myna-v0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-n", type=int, default=192)
    ap.add_argument("--suite", default=None,
                    help="frozen-suite dir (train/development/test.jsonl, laya/kev request shape); "
                         "overrides the synthetic corpus")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--accum-groups", type=int, default=1,
                    help="question-sets averaged into each optimizer update. Real suites hold "
                         "1000+ one-example question-sets; with 1 set per update the shared trunk "
                         "chases a different task every step and never fits any (interference).")
    ap.add_argument("--warmup", type=int, default=0,
                    help="linear LR warmup over this many updates before cosine decay")
    ap.add_argument("--group-sample", choices=["uniform", "pool", "sqrt"], default="pool",
                    # %% because argparse runs the help string through %-formatting
                    help="how to pick question-sets per mini-batch. Measured on decision-v2, "
                         "uniform sends only ~1.4%% of updates to the 16 data-rich sets (12.5%% for "
                         "sqrt) and they measured below chance; pool sends 65%%. Pool weighting "
                         "was harmless before gradient accumulation removed the interference.")
    ap.add_argument("--min-train-pool", type=int, default=0,
                    help="drop train question-sets with fewer than this many examples "
                         "(isolates the data-rich tasks; 0 keeps the whole suite)")
    ap.add_argument("--row-batch", action="store_true",
                    help="draw each mini-batch across the whole train split with one question set "
                         "per row, instead of one shared set per mini-batch. Unblocks boolq/mnli "
                         "(whose instruction text IS the row) and decouples batch size from "
                         "question-set pool size; makes --group-sample inert.")
    ap.add_argument("--max-q-cells", type=int, default=2048,
                    help="with --row-batch: rows x questions x question-tokens per forward. "
                         "Measured at 1.6 MiB of retained activations per cell "
                         "(bench/mem_profile.py), so 2048 is a ~3.2 GB question branch. The "
                         "option axis costs ~3 KiB per cell and is not what filled the M5.")
    ap.add_argument("--paraphrase", choices=["off", "on"], default="off",
                    help="train on hand-written phrasings of every instruction and never on the "
                         "suite's own wording (SPEC P1). Dev/test stay on the exact suite strings, "
                         "and the reserved ninth phrasing is scored too, so 'it learned to read' is "
                         "measured on two unseen wordings. Suite corpora only: an instruction with "
                         "no table entry raises UnknownSchema rather than passing through.")
    ap.add_argument("--long-context", type=int, default=0,
                    help="train needle-in-long-context recall at this state token length (V1-D); "
                         "uses truncated backprop and ignores --suite/--synthetic")
    ap.add_argument("--grad-tokens", type=int, default=768,
                    help="with --long-context: keep the needle inside this trailing window so it "
                         "receives gradient; everything before is scanned detached")
    ap.add_argument("--init", default=None,
                    help="with --long-context: checkpoint dir to warm-start weights + tokenizer from")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = resolve_device(args.device)

    if args.long_context:
        if args.paraphrase == "on":
            # the needle corpus is synthetic text with synthetic instructions, so
            # the table has no entries for it; silently ignoring the flag would
            # print a run that claims a gate it never ran
            raise SystemExit("--paraphrase on is not available with --long-context: the "
                             "needle schemas are not in the phrasing table")
        train_long_context(args, rng, device)
        return

    if args.suite:
        from .real_data import load_suite, suite_texts

        data = load_suite(args.suite)
        if args.min_train_pool:
            data["train"] = {k: v for k, v in data["train"].items()
                             if len(v[1]) >= args.min_train_pool}
            print(f"train groups after pool>={args.min_train_pool}: {len(data['train'])}",
                  flush=True)
        tok = train_tokenizer(suite_texts(data["train"]), vocab_size=args.vocab)
    else:
        data = {
            "train": {wf: generate(args.n_train, wf, rng, "train") for wf in WORKFLOWS},
            "dev": {wf: generate(args.n_eval, wf, rng, "dev") for wf in WORKFLOWS},
            "test": {wf: generate(args.n_eval, wf, rng, "test") for wf in WORKFLOWS},
        }
        tok = train_tokenizer([e.state for wf in data["train"].values() for e in wf], vocab_size=4096)
    # uniform split shape: {split: {group_key: (questions, examples)}}
    if not args.suite:
        data = {
            split: {wf: (WORKFLOWS[wf][0], exs) for wf, exs in groups.items()}
            for split, groups in data.items()
        }
    print("tokenizer trained:", tok.get_vocab_size())

    cfg = MynaConfig(vocab=tok.get_vocab_size())
    model = MynaModel(cfg).to(device)
    print("params:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.warmup:
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=args.warmup),
             torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(args.steps - args.warmup, 1))],
            milestones=[args.warmup],
        )
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    wfs = list(data["train"])
    pools = sorted(len(data["train"][k][1]) for k in wfs)
    thin = sum(1 for n in pools if n < args.batch)
    print(f"train question-sets: {len(wfs)}  median pool {pools[len(pools)//2]}  "
          f"sets with pool<batch({args.batch}) {thin} ({thin/len(wfs):.0%} of sets run a short batch)",
          flush=True)

    def group_cycle(keys, mode):
        """uniform = every set once per pass; pool = weight by example count;
        sqrt = geometric compromise between the two.

        Note on this suite: a set is keyed by its exact question *tensors*, and
        decision-v2 randomizes option descriptions per row, so many sets hold one
        row even though the underlying schema (same instruction, same option keys)
        recurs hundreds of times. Thin pools are a tensor-identity artifact, not
        evidence that the task has one example -- see myna.real_data._signature."""
        if mode == "uniform":
            while True:
                ks = list(keys)
                rng.shuffle(ks)
                yield from ks
        pw = {"pool": 1.0, "sqrt": 0.5}[mode]
        weights = [len(data["train"][k][1]) ** pw for k in keys]
        while True:
            yield rng.choices(keys, weights=weights, k=1)[0]

    cycle = group_cycle(wfs, args.group_sample)
    row_items = None
    if args.row_batch:
        from .real_data import flatten_groups

        row_items = flatten_groups(data["train"])
        q_len_cache: dict[int, int] = {}
        widest = max((len(qs), question_tokens(tok, qs, q_len_cache)) for qs, _ in row_items)
        print(f"row-batch: {len(row_items)} rows in one pool, {len(q_len_cache)} question sets, "
              f"widest row {widest[0]} questions x {widest[1]} tokens; budget "
              f"{args.max_q_cells} cells/forward (~{args.max_q_cells * 1.6 / 1024:.1f} GB branch)",
              flush=True)
    dev_probe = {
        wf: (q, exs[: args.eval_n]) for wf, (q, exs) in data["dev"].items()
    } if data["dev"] else {
        wf: (q, exs[: args.eval_n]) for wf, (q, exs) in data["train"].items()
    }
    paraphraser = None
    n_sets = 0
    if args.paraphrase == "on":
        from .paraphrase import Paraphraser

        paraphraser = Paraphraser()
        n_sets = paraphraser.warm(data["train"])  # raises UnknownSchema on an untabelled set
        priced = ""
        if row_items is not None:
            priced = f"; {worst_case_tokens(tok, paraphraser, row_items, q_len_cache)} sets " \
                     f"budgeted at their longest phrasing"
        print(f"paraphrase: {n_sets} question sets re-worded, suite wording held out{priced}",
              flush=True)
    K = max(1, args.accum_groups)
    log_every = max(1, min(200, args.steps // 60))
    ema = None
    step_t0 = time.time()
    for step in range(args.steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        run_loss = 0.0
        for _ in range(K):
            if row_items is None:
                wf = next(cycle)
                questions, pool = data["train"][wf]
                if paraphraser is not None:
                    questions = paraphraser.draw(wf, questions, rng)
                b = build_batch(draw_batch(pool, args.batch, rng), tok, questions, device)
            else:
                items = draw_row_batch(row_items, args.batch, args.max_q_cells, rng, tok,
                                    q_len_cache)
                b = build_row_batch(items, tok, device, paraphraser, rng)
            logits = model(b["state_ids"], b["state_len"], b["q_ids"], b["q_mask"],
                           b["span_mat"], b["opt_valid"], b["decide_idx"])
            loss = typed_loss(logits, b["gold"], b["has_gold"]) / K
            loss.backward()
            run_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema = run_loss if ema is None else 0.95 * ema + 0.05 * run_loss
        if step % log_every == 0 or step == args.steps - 1:
            el = time.time() - step_t0
            print(f"step {step:5d}  loss {run_loss:.3f}  ema {ema:.3f}  "
                  f"sets/update {K}  {el:.0f}s", flush=True)
        if args.eval_every and step and step % args.eval_every == 0 and step != args.steps - 1:
            m = evaluate(model, tok, dev_probe, device)
            print(f"step {step:5d}  dev-mid acc {macro_acc(m):.4f}", flush=True)
            # rolling snapshot so a long run can be evaluated or stopped early
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            tok.save(str(out / "tokenizer.json"))
            torch.save(
                {"state_dict": model.state_dict(), "cfg": vars(cfg), "temperature": 1.0, "step": step},
                out / "model_last.pt",
            )

    calib_split, calib_name = temperature_source(data)
    # the row count rides along with the label: "(fit on calibration)" would
    # still print if the split were hardcoded, the count would not match
    n_fit_rows = sum(len(exs) for _q, exs in calib_split.values())
    temperature = fit_temperature(model, tok, calib_split, device)
    print(f"temperature: {temperature}  (fit on {calib_name}: {n_fit_rows} rows in "
          f"{len(calib_split)} sets)")
    dev_m = evaluate(model, tok, data["dev"], device, temperature)
    test_m = evaluate(model, tok, data["test"], device, temperature)
    print("=== dev ===")
    for k, v in sorted(dev_m.items()):
        print(f"{k:32s} {v:.4f}")
    print("=== test ===")
    for k, v in sorted(test_m.items()):
        print(f"{k:32s} {v:.4f}")

    # The gate P1 exists for: dev on the suite's own wording, and dev on the one
    # phrasing that was neither trained nor shipped by the suite. A model that read
    # the instruction holds; a model that matched a template falls.
    dev_unseen = None
    n_draws = 0
    if paraphraser is not None:
        dev_unseen = held_out_eval(model, tok, data["dev"], device, temperature)
        print(f"=== dev, HELD-OUT phrasing === macro {macro_acc(dev_unseen):.4f}   "
              f"(exact suite wording {macro_acc(dev_m):.4f})")
        # the warm-up banner only proves the table loaded; this one proves the
        # loop consulted it, which no earlier check did for the shared-set path
        n_draws = paraphraser.draws
        print(f"paraphrase: {n_draws} phrasing draws reached the batches over "
              f"{args.steps} steps", flush=True)

    # machine-readable metrics + a key->type map so downstream tables can roll
    # up by source x question-type (apples-to-apples with the laya witness).
    qtypes = {}
    for split in ("dev", "test"):
        for wf, (questions, _exs) in data[split].items():
            for q in questions:
                qtypes[f"{wf}/{q.name}"] = q.type

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump({"temperature": temperature, "temperature_fit_on": calib_name,
                   "temperature_fit_rows": n_fit_rows,
                   "paraphrase": args.paraphrase, "paraphrase_sets": n_sets,
                   "paraphrase_draws": n_draws,
                   "dev": dev_m, "test": test_m,
                   "dev_unseen": dev_unseen, "qtypes": qtypes}, f, indent=2)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": vars(cfg), "temperature": temperature}, out / "model.pt")
    tok.save(str(out / "tokenizer.json"))
    print(f"saved checkpoint to {out}")


if __name__ == "__main__":
    main()


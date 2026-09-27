"""What the browser has to reproduce, written out by the engine it must agree with.

    uv run python bench/browser_expected.py [--artifact runs/onnx]

`browser/selftest.mjs` is the gate: it loads the exported artifact in
onnxruntime-web, answers the same requests through the JavaScript tokenizer, pointer
head and chained scan, and compares. This script writes the other side of that
comparison — token ids, per-layer state fingerprints, and probabilities — from
`myna.engine.Myna` on the same checkpoint, so the JS is checked against the Python that
ships rather than against a transcription someone typed by hand.

Two boundaries this file has to respect:

* **No upstream text.** The pilot corpus is gitignored because its rows are other
  people's data, and an expectations file is committed. So every string here comes from
  `myna.data.generate` (synthetic) or from literals written in this repo. The widest
  *real* request is measured in Python against the real corpus and lands in
  `runs/onnx_parity_widest.json`, which carries numbers and no text.
* **The artifact's own shape.** The questions are sized to fit the profile's `--q-len`
  and `--questions`, read out of `meta.json`. A profile that cannot carry a request is
  reported, not quietly skipped — that is how 6a found the 256-wide graph in the first
  place (§9.26).

Three things the file is sized so that they can *fail*, because a gate that cannot is
decoration: the document spans several `chunk`-token scan calls, so the chain itself is on
trial rather than only the first call; at least one option span is several tokens wide, so
mean-pooling is distinguished from picking an end; and the answers are written twice, once
with no floor and once under `PROBE_THRESHOLD` where the model's two softest commitments
fall below it, so the abstention branch runs in both directions on real probabilities.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from myna.data import WORKFLOWS, generate
from myna.engine import Myna, abstain_check
from myna.tokenizer import build_question, encode_text

# Adversarial on purpose: each line targets a rule the JavaScript re-implementation can
# get wrong without changing a single ordinary sentence. ByteLevel maps UTF-8 *bytes*,
# WhitespaceSplit deletes separators instead of turning them into "Ġ", added tokens are
# matched before the splitter runs, and `tokenizers` applies no word-length cutoff here
# even though its documentation names a default of 64 — so the long lines below are the
# case a reader who trusts the docs would get wrong.
EDGE_CASES = [
    "",
    " ",
    "\n\n",
    "the quick brown fox",
    "thequick brownfox",
    "Hi, we were billed twice for March.",
    "tabs\tand   spaces",
    "line1\nline2",
    "émoji 🎉 test",
    "a" * 63,
    "a" * 64,
    "a" * 80,
    "ab" * 130,
    "tokenizers-" + "re-" * 60 + "long",
    "supercalifragilisticexpialidocious-and-then-some-more-letters-to-pass-sixty-four",
    "[OPT] sneaky [DECIDE] too",
    "UPPER lower MiXeD",
    "3.14, 1,000, 42nd",
    "refund?? yes!! maybe...",
    "quote \"and\" another 'one'",
    "  leading and trailing  ",
    "½¼§¶†‡ non-ascii punctuation",
    "日本語のテキストと latin",
    "comma,separated,no spaces",
    "hyphen-ated and em\u2014dash",
    "\u00a0non-breaking space\u00a0here",
]


def state_fingerprint(S):
    """Per-layer summary statistics of the chained state.

    A fingerprint rather than the tensor: the browser's numbers are what is on trial, and
    mean/max/L2 over ~25k entries per layer catch a wrong scan, a wrong mask and a wrong
    state order, without committing 576 KiB of float text to git.
    """
    out = []
    for t in S:
        f = t.detach().float().reshape(-1)
        out.append({"n": int(f.numel()), "mean": float(f.mean()), "max": float(f.max()),
                    "min": float(f.min()), "l2": float(f.norm()),
                    "tail": [float(x) for x in f[-8:]]})
    return out


def _probs(answer: dict, labels: list[str]) -> list[float]:
    """The probability vector behind an engine answer, in label order.

    `choice` and `score` carry `probabilities`; `noul` deliberately does not — it reports
    the raw p(Yes) because the Brier score is defined on that one number — so the vector is
    rebuilt from it rather than the readout widened for this file's convenience.
    """
    if "probabilities" in answer:
        return [float(answer["probabilities"][l]) for l in labels]
    return [1.0 - float(answer["noul"]), float(answer["noul"])]


# Three floors, chosen against this checkpoint's own probabilities so that each abstain
# branch of each readout type is exercised. At 0.9 the two softest commitments fall and the
# surest stands; at 0.9995 all three fall, which is the only way the `choice` and `noul`
# abstention branches run at all. A floor that is only ever tested while it stays silent
# proves nothing about the day it speaks.
#
# 0.6 is the mirror image: below every argmax this checkpoint produced (the softest is
# 0.7802), so all three answers commit. That row is what makes the score answer's
# *expectation* a checked number — the two higher floors abstain it, and an abstained score
# is `null` on both sides, so a reduce over `p` instead of `j * p` would survive a gate
# that only ever looked at abstentions.
PROBE_THRESHOLDS = (0.6, 0.9, 0.9995)

# Vectors that separate `confidence` from `margin`. The gate reads the top probability, so
# a gate that read the margin instead is caught by the first case (0.45 sure enough, 0.03
# ahead of the runner-up), and one that inverted the comparison by the second.
GATE_CASES = [
    (0.4, ["billing", "technical", "other"], [0.45, 0.42, 0.13]),
    (0.6, ["billing", "technical", "other"], [0.20, 0.30, 0.50]),
    (0.9, ["No", "Yes"], [0.51, 0.49]),
]


def build(ckpt: Path, artifact: Path) -> dict:
    meta = json.loads((artifact / "meta.json").read_text())
    myna = Myna(ckpt, device="cpu")
    chunk, q_len, N = meta["chunk"], meta["q_len"], meta["n_questions"]

    exs = generate(6, "support", random.Random(7))
    workflow = WORKFLOWS["support"][0]
    questions, skipped = [], []
    for q in workflow:
        labels = myna._options({"type": q.type, "criteria": q.options})
        ntok = len(build_question(myna.tok, q.instruction, labels)[0])
        if ntok > q_len:
            skipped.append({"name": q.name, "tokens": ntok, "q_len": q_len})
            continue
        questions.append({"name": q.name, "type": q.type, "instruction": q.instruction,
                          "labels": labels})
    if len(questions) > N:
        questions = questions[:N]

    strings = [e.state for e in exs] + EDGE_CASES
    token_ids = []
    for s in strings:
        from myna.tokenizer import encode_text
        token_ids.append({"text": s, "ids": encode_text(myna.tok, s)})

    # One document across every example, so the browser's scan really chains: on a
    # single-chunk document `fed += chunk`, a missing `break` and a state thrown away
    # between calls are all invisible, and a gate that cannot fail those is not a gate.
    document = "\n".join(e.state for e in exs)
    doc_ids = encode_text(myna.tok, document)
    calls = -(-len(doc_ids) // chunk)
    if calls < 2:
        raise SystemExit(f"the expectations document is {len(doc_ids)} tokens, which fits one "
                         f"{chunk}-token call: the chained-scan mutations would be unkillable")

    built = []
    for q in questions:
        ids, spans, dec = build_question(myna.tok, q["instruction"], q["labels"])
        built.append({"name": q["name"], "ids": ids,
                      "spans": [[s, e] for s, e in spans], "decide": dec})

    # the engine's own answers, through the torch path the browser is compared to.
    # `_options` renders a choice spec as "label: text"; the labels stored above are
    # already rendered, and a list `criteria` passes through `_options` unchanged, so the
    # two sides ask the same question of the same head.
    obs = myna.observe(document)
    spec = {q["name"]: {"type": q["type"], "instructions": q["instruction"],
                        "criteria": q["labels"]} for q in questions}
    answers = obs.ask(spec)["answers"]

    # The same three requests under floors that fire, from the engine that owns that
    # logic, with the probability vectors attached so the JavaScript readout can be run on
    # the numbers rather than on a second sampling of the graph.
    cases = []
    for thr in PROBE_THRESHOLDS:
        gated = Myna(ckpt, device="cpu", abstain_below=thr)
        gated_answers = gated.observe(document).ask(spec)["answers"]
        cases.extend([{"threshold": thr, "name": q["name"], "type": q["type"],
                       "labels": q["labels"],
                       "probs": _probs(gated_answers[q["name"]], q["labels"]),
                       "want": gated_answers[q["name"]]} for q in questions])

    with torch.no_grad():
        _, S_torch = myna.trunk.encode_state(
            torch.tensor([doc_ids]), parallel=True, chunk=meta.get("scan_chunk") or chunk)
    return {
        "note": "generated by bench/browser_expected.py from the checkpoint named below; "
                "the JavaScript side is browser/selftest.mjs",
        "ckpt": str(ckpt), "artifact": str(artifact),
        "profile": {"chunk": chunk, "q_len": q_len, "n_questions": N,
                    "scan_chunk": meta.get("scan_chunk"), "d_model": meta["d_model"],
                    "d_ptr": meta["d_ptr"], "n_layers": meta["n_layers"],
                    "temperature": meta["temperature"],
                    "abstain_below": meta.get("abstain_below")},
        "tokenizer": {"strings": token_ids, "questions": built},
        "questions": questions,
        "questions_skipped_as_too_wide": skipped,
        "document": document,
        "document_tokens": len(doc_ids),
        "document_calls": calls,
        "engine_answers": answers,
        "abstention": {
            "thresholds": list(PROBE_THRESHOLDS),
            "cases": cases,
            "gate_cases": [{"threshold": t, "labels": lb, "probs": p,
                            "want": abstain_check(p, t, lb)} for t, lb, p in GATE_CASES],
        },
        "state": state_fingerprint(S_torch),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="runs/myna-v0")
    ap.add_argument("--artifact", default="runs/onnx")
    ap.add_argument("--out", default="browser/expected.json")
    args = ap.parse_args()
    data = build(Path(args.ckpt), Path(args.artifact))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n")
    n = len(data["tokenizer"]["strings"])
    print(f"wrote {out}: {n} strings ({sum(len(s['ids']) for s in data['tokenizer']['strings'])} "
          f"tokens), {len(data['questions'])} questions, "
          f"document {data['document_tokens']} tokens in {data['document_calls']} scan calls, "
          f"state {data['state'][0]['n']} entries x {len(data['state'])} layers, "
          f"abstention probed at {' and '.join(str(t) for t in data['abstention']['thresholds'])} "
          f"({sum(1 for c in data['abstention']['cases'] if c['want']['abstain'])} of "
          f"{len(data['abstention']['cases'])} answers abstain), "
          f"profile chunk {data['profile']['chunk']} q_len {data['profile']['q_len']} "
          f"tile {data['profile']['scan_chunk']}")
    if data["questions_skipped_as_too_wide"]:
        print(f"{len(data['questions_skipped_as_too_wide'])} question(s) do not fit "
              f"--q-len {data['profile']['q_len']}: "
              + ", ".join(f"{q['name']} ({q['tokens']} tokens)"
                          for q in data["questions_skipped_as_too_wide"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

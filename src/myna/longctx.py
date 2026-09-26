"""Long-context needle corpus for V1-D: does a trained trunk still READ a
single decisive sentence buried in a long observation?

The state is filler sentences (neutral, option-free, so they can't leak the
answer) with one needle sentence inserted at a random position. The needle
alone determines a choice question. Sweeping the state length measures whether
recall stays flat as the observation grows — the acceptance curve for the long
checkpoint (and a baseline for v0/v1, which were only ever trained on short
states)."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .tokenizer import encode_text

# neutral filler: no option nouns, no decision cues
FILLER = [
    "the log entry was recorded at the end of the shift",
    "a routine sync completed without any warning",
    "the dashboard loaded normally on the first try",
    "an unrelated ticket was closed last week",
    "the nightly job finished and sent no alerts",
    "the meeting notes were filed under the archive",
    "a cache refresh ran in the background",
    "the status page showed all systems green",
    "a copy of the report was stored in the drive",
    "the audit trail matched the expected sequence",
    "an idle connection timed out and reconnected",
    "the backup verified against the checksum",
]

DESKS = ["billing", "logistics", "technical support", "returns", "fraud", "sales"]
ENTITIES = ["the invoice", "the parcel", "the login page", "the headset", "the refund",
            "the subscription", "the warranty claim", "the export tool"]


@dataclass
class Needle:
    state: str
    instruction: str
    options: list[str]
    gold: int
    target_tokens: int


def make_needle(tok, target_tokens: int, rng: random.Random) -> Needle:
    gold = rng.randrange(len(DESKS))
    entity = rng.choice(ENTITIES)
    # needle carries the only evidence: which desk owns the entity
    needle = f"note that {entity} is owned by the {DESKS[gold]} desk".capitalize() + "."
    instruction = f"Which desk owns {entity}?"

    # assemble filler until we hit the target token count, insert needle at a
    # random sentence boundary
    enc_needle = len(encode_text(tok, needle))
    sentences: list[str] = []
    ntok = enc_needle
    while ntok < target_tokens:
        s = rng.choice(FILLER).capitalize() + "."
        sentences.append(s)
        ntok += len(encode_text(tok, s))
    pos = rng.randrange(len(sentences) + 1)
    sentences.insert(pos, needle)
    state = " ".join(sentences)
    return Needle(state, instruction, list(DESKS), gold, target_tokens)


def eval_lengths(myna, lengths, n_per_length=40, seed=0):
    """Accuracy-vs-length on any engine exposing .predict(state, questions)."""
    import time
    rng = random.Random(seed)
    rows = []
    for L in lengths:
        correct = 0
        lat = []
        for _ in range(n_per_length):
            nd = make_needle(myna.tok, L, rng)
            q = {"desk": {"type": "choice", "instructions": nd.instruction,
                          "criteria": {d: d for d in nd.options}}}
            t0 = time.perf_counter()
            out = myna.predict(nd.state, q)
            lat.append((time.perf_counter() - t0) * 1000)
            pred = out["answers"]["desk"]["probabilities"]
            pick = max(pred, key=pred.get)
            correct += int(pick == nd.options[nd.gold])
        acc = correct / n_per_length
        rows.append({"tokens": L, "acc": acc, "ms": sum(lat) / len(lat)})
    return rows

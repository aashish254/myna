"""Streaming inference engine: the whole point of Myna, exposed as an API.

    obs = myna.observe(ticket)          # state scanned once
    obs.ask(questions)                   # branch — state not re-read
    obs2 = obs.append(new_message)       # only the new tokens are scanned
    obs2.ask(other_questions)

Question specs use laya's typed schema: choice / score / noul.

Abstention (SPEC §5 P5, gate G5): construct with `Myna(ckpt, abstain_below=t)` and
any decision whose top probability falls under `t` comes back with the answer field
set to `None` and a `reason` naming the measured confidence and the runner-up
margin. It is `None` rather than the argmax on purpose: a caller that ignores
`abstain` then gets a type error at the seam, not a confident-looking wrong label.
With no threshold the engine never abstains — the latency and parity paths ask a
question that abstention cannot answer.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .model import MynaConfig, MynaModel
from .tokenizer import build_question, encode_text

# Inference scans are no-grad, so a large chunk costs only transient memory
# and slashes Python-loop overhead vs the training chunk (16). All chunk sizes
# are proven exactly equivalent in tests/test_trunk_numerics.py.
INFER_CHUNK = 256


def abstain_check(probs: list[float], threshold: float | None,
                  labels: list[str]) -> dict:
    """Decide whether a probability vector supports a commitment, and say why.

    Pure and threshold-only: it never sees gold labels, so tuning the threshold
    on `calibration.jsonl` cannot leak an answer through this function.

    Confidence is the *top* probability, which for a two-way (noul) question is
    the probability of whichever side is being committed to — reading noul's
    confidence off `p_yes` instead would abstain on the model's surest "no"s.

    `margin` is the distance to the runner-up, which is what actually separates
    "the model has decided" from "the model picked the least bad of two".
    """
    if threshold is None:
        return {"abstain": False, "confidence": None, "margin": None, "reason": None}
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"abstain threshold must be in (0, 1], got {threshold!r}")
    if len(probs) != len(labels) or not probs:
        raise ValueError(f"{len(probs)} probabilities for {len(labels)} labels")
    ordered = sorted(probs, reverse=True)
    margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
    i = max(range(len(probs)), key=lambda k: probs[k])
    conf = probs[i]
    return {"abstain": conf < threshold, "confidence": conf, "margin": margin,
            "reason": f"{labels[i]} at p={conf:.3f}, {margin:.3f} ahead of the runner-up, "
                      f"under the {threshold:.3f} floor"}


class Observation:
    def __init__(self, myna, ids: list[int], S_cache):
        self.m = myna
        self.ids = ids
        # the entire retained state: per-layer [1, H, dk, dv] matrices at the
        # state end. Fixed size — an observation of any length costs the same
        # memory (~0.6 MB at production width).
        self.S_cache = S_cache

    @torch.no_grad()
    def append(self, text: str) -> "Observation":
        delta = encode_text(self.m.tok, text)
        if not delta:
            return self
        ids_t = torch.tensor([delta], device=self.m.device)
        new_S = self.m.trunk.append_state(ids_t, self.S_cache, len(self.ids), chunk=INFER_CHUNK)[1]
        return Observation(self.m, self.ids + delta, new_S)

    def save_state(self, path: str | Path) -> None:
        """Persist the whole observation: it is only the token list plus the
        fixed-size S stack, so a session snapshot is 576 KiB of state and a
        few hundred bytes of ids on top of it (592,533 B on disk for a
        51-token thread, measured)."""
        torch.save(
            {"ids": self.ids, "S": [s.detach().cpu() for s in self.S_cache]},
            str(path),
        )

    def ask(self, questions: dict) -> dict:
        t0 = time.perf_counter()
        specs = [(name, spec) for name, spec in questions.items()]
        labels_opts = [self.m._options(s) for _, s in specs]
        built = [build_question(self.m.tok, spec.get("instructions", name), opts)
                 for (name, spec), opts in zip(specs, labels_opts)]
        lq = max(len(ids) for ids, _, _ in built)
        n = len(built)
        dev = self.m.device
        q_ids = torch.zeros(n, lq, dtype=torch.int64, device=dev)
        q_mask = torch.zeros(n, lq, device=dev)
        for i, (ids, _, _) in enumerate(built):
            q_ids[i, : len(ids)] = torch.tensor(ids, device=dev)
            q_mask[i, : len(ids)] = 1.0
        hn_q = self.m.trunk.encode_question_batch(
            q_ids, q_mask, self.S_cache, len(self.ids), chunk=INFER_CHUNK
        )
        answers = {}
        usage = 0
        for i, ((name, spec), opts) in enumerate(zip(specs, labels_opts)):
            _, spans, dec = built[i]
            answers[name] = self.m._readout(spec, opts, hn_q[i], dec, spans)
            usage += len(built[i][0])
        # The policy and the count ride with every response: a caller that never
        # reads the per-answer `abstain` flag still cannot miss that the engine
        # was running with a floor under it, or how many questions it gave up.
        abstained = sorted(k for k, a in answers.items() if a.get("abstain"))
        return {
            "answers": answers,
            "usage": {"state_tokens": len(self.ids), "question_tokens": usage},
            "policy": {"abstain_below": self.m.abstain_below, "abstained": abstained},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }


class Myna:
    def __init__(self, ckpt_dir: str | Path, device: str = "cpu",
                 abstain_below: float | None = None):
        """`abstain_below=None` (the default) never abstains: the latency and
        parity paths must answer every question, and an engine that can quietly
        stop answering would make those measurements mean nothing.

        Validate at construction, not per answer: a threshold of 0 abstains
        never and one of 1.1 abstains always, and both look like a working model
        until the day the fallback bill arrives."""
        if abstain_below is not None and not 0.0 < float(abstain_below) <= 1.0:
            raise ValueError(f"--abstain-below must be in (0, 1], got {abstain_below!r}")
        ckpt_dir = Path(ckpt_dir)
        ck = torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=False)
        self.cfg = MynaConfig(**ck["cfg"])
        self.model = MynaModel(self.cfg)
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(device).eval()
        self.temperature = ck["temperature"]
        self.abstain_below = None if abstain_below is None else float(abstain_below)
        self.device = device
        self.tok = Tokenizer.from_file(str(ckpt_dir / "tokenizer.json"))

    @property
    def trunk(self):
        return self.model.trunk

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def observe(self, text: str) -> Observation:
        ids = encode_text(self.tok, text)
        if not ids:
            ids = [self.tok.token_to_id("[PAD]")]
        ids_t = torch.tensor([ids], device=self.device)
        # chunked scan: identical results to the recurrent path (proven in
        # tests) but vectorized, so long observations encode fast too.
        _, S_cache = self.trunk.encode_state(ids_t, parallel=True, chunk=INFER_CHUNK)
        return Observation(self, ids, S_cache)

    def predict(self, state: str, questions: dict) -> dict:
        return self.observe(state).ask(questions)

    def restore(self, path: str | Path) -> Observation:
        """Reopen an observation saved by Observation.save_state — the scan
        that produced it is never repeated."""
        blob = torch.load(str(path), map_location=self.device, weights_only=True)
        return Observation(self, list(blob["ids"]), [s.to(self.device) for s in blob["S"]])

    def _options(self, spec: dict) -> list[str]:
        qtype = spec["type"]
        if qtype == "choice":
            crit = spec["criteria"]
            if isinstance(crit, dict):
                return [f"{l}: {crit[l]}" for l in crit]
            return list(crit)
        if qtype == "score":
            return [str(c) for c in spec["criteria"]]
        if qtype == "noul":
            return ["No", "Yes"]
        raise ValueError(f"unknown question type {qtype!r}")

    @torch.no_grad()
    def _readout(self, spec: dict, option_texts: list[str], hq_row, dec: int, spans):
        """hq_row: [Lq, d] hidden states for one question branch."""
        qtype = spec["type"]
        if qtype == "choice":
            crit = spec["criteria"]
            labels = list(crit) if isinstance(crit, dict) else list(crit)
        elif qtype == "score":
            labels = [str(c) for c in spec["criteria"]]
        else:
            labels = ["No", "Yes"]
        decide = hq_row[dec][None]  # [1,d]
        pooled = torch.stack([hq_row[s:e].mean(0) for s, e in spans])  # [O,d]
        logits = (self.model.wq(decide) @ self.model.wk(pooled).mT).squeeze(0) / (self.cfg.d_ptr**0.5)
        probs = F.softmax(logits / self.temperature, dim=-1)
        pl = [float(p) for p in probs]
        gate = abstain_check(pl, self.abstain_below, labels)
        if qtype == "choice":
            i = int(probs.argmax())
            return {"type": "choice", "choice": None if gate["abstain"] else labels[i],
                    "confidence": pl[i], "abstain": gate["abstain"], "reason": gate["reason"],
                    "margin": gate["margin"],
                    "probabilities": {l: p for l, p in zip(labels, pl)}}
        if qtype == "score":
            i = int(probs.argmax())
            return {"type": "score",
                    "score": None if gate["abstain"] else sum(k * p for k, p in enumerate(pl)),
                    "level": None if gate["abstain"] else labels[i],
                    "confidence": gate["confidence"], "abstain": gate["abstain"],
                    "reason": gate["reason"], "margin": gate["margin"],
                    "probabilities": {labels[k]: p for k, p in enumerate(pl)},
                    "legend": {str(k): l for k, l in enumerate(labels)}}
        # noul keeps reporting the raw p(Yes): the Brier score is defined on it,
        # and an abstention must not delete the number the gate is measured with.
        return {"type": "noul", "noul": pl[1], "yes": None if gate["abstain"] else pl[1] >= 0.5,
                "confidence": gate["confidence"], "abstain": gate["abstain"],
                "reason": gate["reason"], "margin": gate["margin"]}

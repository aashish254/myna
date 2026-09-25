"""Streaming inference engine: the whole point of Myna, exposed as an API.

    obs = myna.observe(ticket)          # state scanned once
    obs.ask(questions)                   # branch — state not re-read
    obs2 = obs.append(new_message)       # only the new tokens are scanned
    obs2.ask(other_questions)

Question specs use laya's typed schema: choice / score / noul.
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
        return {
            "answers": answers,
            "usage": {"state_tokens": len(self.ids), "question_tokens": usage},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }


class Myna:
    def __init__(self, ckpt_dir: str | Path, device: str = "cpu"):
        ckpt_dir = Path(ckpt_dir)
        ck = torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=False)
        self.cfg = MynaConfig(**ck["cfg"])
        self.model = MynaModel(self.cfg)
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(device).eval()
        self.temperature = ck["temperature"]
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
        if qtype == "choice":
            i = int(probs.argmax())
            return {"type": "choice", "choice": labels[i], "confidence": float(probs[i]),
                    "probabilities": {l: float(p) for l, p in zip(labels, probs)}}
        if qtype == "score":
            dist = {labels[i]: float(p) for i, p in enumerate(probs)}
            expectation = sum(i * float(p) for i, p in enumerate(probs))
            i = int(probs.argmax())
            return {"type": "score", "score": expectation, "level": labels[i],
                    "probabilities": dist, "legend": {str(i): l for i, l in enumerate(labels)}}
        return {"type": "noul", "noul": float(probs[1])}

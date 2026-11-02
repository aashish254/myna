"""Myna v6 Unified: a fine-tuned DeBERTa-v3-large backbone with a cardinality-routed readout.

    myna = MynaV6("myna-v5-prototype 2")          # a local campaign directory
    myna = MynaV6("aashish254/myna-v6")           # the same files from the Hub
    myna.predict(state, questions)                 # typed answers, as engine.py

The checkpoint (`myna-v6-unified.pt`) holds four readout heads, a `config` dict and the
audit dicts that every published v6 figure is quoted from. The encoder itself is a
Hugging Face `DebertaV2Model` in `deberta-v5-finetuned/`. `MynaV6` rebuilds the heads
from this module's class definitions and loads the tensors into them with
`load_state_dict(..., strict=True)`, so a head whose topology drifted from the
checkpoint raises instead of silently scoring with random weights.

WHAT THIS FILE PROVES, AND WHAT IT DOES NOT. The parameter shapes below are read off
the shipped checkpoint, so the *topology* of all four heads is verified on load. The
scoring *recipe* — the option order, the task prefix, which six values feed
`none_gate`, how `CoralOrdinalScoreHead` folds its cumulative cuts into a utility — was
written in a Colab runtime whose script is not in the archive (the zips contain weights
and JSON witnesses only). Two facts narrow the risk to the paths that were actually
scored: `config["use_coral_score_head"]` is `false`, so the ordinal head plays no part
in the published numbers, and `config["b77_none_mode"]` is `T_fallback`, so the open-set
decision is a threshold rule rather than the learned `none_gate`. `extract_gate_features`
and `CoralOrdinalScoreHead.forward` are therefore reconstructions, and both are named as
such at their definitions. Re-scoring the 1,440 test questions here reproduces the
routing and the calibration arithmetic, not the Colab figure, unless the recipe agrees.

v6 is not a streaming engine. The persistent-state, scan-once property that `engine.py`
and SPEC §1 sell is a feature of the from-scratch SSM trunk: a cross-encoder re-reads
the whole state for every question, and `Observation.append` here only grows the text
that will be re-encoded. Sessions are kept because `myna.serve` is built on them, not
because they save a scan.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# The suite's own renderers, reused rather than re-implemented: v6 was trained on
# `data/decision-v2-pilot`, and an option string that differs from the training loader is
# a different input, not a different engine.
from .real_data import _option_text, _state_text

# Padding value for an absent option in a packed batch. -1e9 overflows c10::Half
# (inf, then NaN through the softmax) under CUDA autocast, which is how the Colab run
# was executed; -65000.0 is the largest magnitude that survives fp16 as a finite score.
MASK_FILL = -65000.0

# The checkpoint's filenames and the encoder subdirectory, in the order a reader is most
# likely to find them. v5's unified checkpoint is accepted too: it carries the same
# `packed_scorer` and `config`, and lacks only the heads v6 added.
CKPT_NAMES = ("myna-v6-unified.pt", "myna-v5-unified.pt")
BACKBONE_DIR = "deberta-v5-finetuned"

HUB_PATTERNS = [BACKBONE_DIR + "/**", "*.pt", "*.json"]


def _readout(d_model: int, hidden: int, dropout: float) -> nn.Sequential:
    """LayerNorm -> Linear -> GELU -> Dropout -> Linear(1).

    The parameter indices in the checkpoint are 0, 1, 4: three modules between the two
    Linears carry no weights, which is what GELU and Dropout are. Every scorer in the
    file is this shape, so every one of them is built here.
    """
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )


def _pairwise_mlp(d_model: int, hidden: int, dropout: float) -> nn.Sequential:
    """Linear(2d -> h) -> LayerNorm(h) -> GELU -> Dropout -> Linear(h -> 1).

    Takes a concatenation of two d_model vectors, which is what the competition feature
    `[norm(h), norm(h) - mean(norm(H))]` is. Its layer-norm at index 1 (weight shape
    `(hidden,)`, not `(d_model,)`) is what distinguishes it from `_readout`.
    """
    return nn.Sequential(
        nn.Linear(2 * d_model, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )


class SetAwareBankingHead(nn.Module):
    """Scores options relative to the other options in the same bundle.

    Packed mode collapses on a 77-option question: the sequence budget is spent before
    the last option, so those rows were truncated to three tokens each and scored 0.0086.
    This head instead encodes each option against the state alone and lets the bundle's
    own competition set the scale — `base_scorer` sees one option, `comp_mlp` sees that
    option's deviation from the bundle mean.
    """

    d_model = 1024
    hidden = 512
    gate_features = 6
    gate_hidden = 32

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(self.d_model)
        self.base_scorer = _readout(self.d_model, self.hidden, dropout)
        self.comp_mlp = _pairwise_mlp(self.d_model, self.hidden, dropout)
        # 0-dimensional, like the tensor in the checkpoint: `torch.zeros(1)` would be a
        # shape-(1,) parameter and `strict=True` would reject the shipped weights.
        self.comp_scale = nn.Parameter(torch.zeros(()))
        self.none_gate = nn.Sequential(
            nn.Linear(self.gate_features, self.gate_hidden),
            nn.GELU(),
            nn.Linear(self.gate_hidden, 1),
        )

    def raw_option_logits(self, hs: torch.Tensor) -> torch.Tensor:
        """hs: [O, d] option hidden states for one question. Returns [O]."""
        h = self.norm(hs)
        base = self.base_scorer(h).squeeze(-1)
        deviation = h - h.mean(0, keepdim=True)
        comp = self.comp_mlp(torch.cat([h, deviation], dim=-1)).squeeze(-1)
        return base + torch.tanh(self.comp_scale) * comp

    def extract_gate_features(self, hs: torch.Tensor) -> torch.Tensor:
        """[O, d] -> [6], the input to `none_gate`.

        RECONSTRUCTED. The checkpoint tells us the first Linear is `Linear(6, 32)` and
        nothing else about which six values were fed to it. This returns a fixed,
        documented summary (top-2 margin, entropy, mean and spread of the competition
        logits, their max, and the count) so the module runs; it is not the Colab
        feature vector. `b77_none_mode == "T_fallback"` means the shipped scoring path
        never calls this — the open-set decision is `none_of_these_by_threshold`.
        """
        logits = self.raw_option_logits(hs)
        probs = F.softmax(logits, dim=-1)
        ordered = torch.sort(probs, descending=True).values
        margin = ordered[0] - (ordered[1] if ordered.numel() > 1 else torch.zeros_like(ordered[0]))
        entropy = -(probs.clamp_min(1e-9).log() * probs).sum()
        mean, std, best = logits.mean(), logits.std(unbiased=False), logits.max()
        return torch.stack([margin, entropy, mean, std, best,
                            torch.log1p(torch.tensor(float(logits.numel()),
                                                     dtype=logits.dtype))])

    def none_of_these_by_threshold(self, probs: torch.Tensor, param: float) -> bool:
        """The mode the checkpoint actually ships: `T_fallback`.

        `none_of_these` is accepted when no option clears `param` — a threshold on the
        calibrated top probability, with `param` selected on dev.
        """
        return float(probs.max()) < float(param)


class CoralOrdinalScoreHead(nn.Module):
    """Rank-consistent cumulative-link (CORAL) ordinal head over five levels.

    Trained against Earth Mover's Distance, which is why it is a set of ordered cuts
    rather than a softmax over five levels: `deltas` (3 of them, between 4 adjacent
    levels' cut pairs of a 5-level scale... see below) all came back positive, so the
    cuts stay ordered. `config["use_coral_score_head"]` is **false** in the shipped
    checkpoint, so this head contributes nothing to any published v6 figure.

    `forward` is RECONSTRUCTED for the same reason `extract_gate_features` is: the cut
    arithmetic and the `utility_head`'s 2,048-wide input are constrained by the shapes
    but were written in the runtime, not here.
    """

    d_model = 1024
    hidden = 512
    levels = 5
    heads = 16

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.pointwise = _readout(self.d_model, self.hidden, dropout)
        self.level_attn = nn.MultiheadAttention(self.d_model, self.heads, batch_first=True)
        self.level_norm = nn.LayerNorm(self.d_model)
        self.level_pos = nn.Parameter(torch.zeros(1, self.levels, self.d_model))
        self.utility_head = _pairwise_mlp(self.d_model, self.hidden, dropout)
        self.b0 = nn.Parameter(torch.zeros(()))
        self.deltas = nn.Parameter(torch.zeros(self.levels - 2))
        self.ordinal_mix = nn.Parameter(torch.zeros(()))

    def cut_offsets(self) -> torch.Tensor:
        """Monotone offsets for the cumulative-link thresholds, one per cut."""
        return torch.cumsum(F.softplus(self.deltas), dim=0)

    def forward(self, hs: torch.Tensor) -> dict:
        """hs: [O, d] option hidden states -> the four quantities this head carries.

        It deliberately stops short of a single scalar. A cumulative-link ordinal head
        over five levels needs four cuts, and the checkpoint stores `b0` plus three
        `deltas`, so the fold from those three offsets to five levels is not determined
        by the artifact. Emitting the parts keeps the head loadable and inspectable
        without claiming a decoding that was reconstructed.
        """
        pointwise = self.pointwise(hs).squeeze(-1)
        query = self.level_norm(self.level_pos + hs.mean(0))
        mixed, _ = self.level_attn(query, query, query)
        mixed = mixed[0]
        deviation = mixed - mixed.mean(0, keepdim=True)
        utility = self.utility_head(torch.cat([mixed, deviation], dim=-1)).squeeze(-1)
        return {"pointwise": pointwise, "utility": utility,
                "cuts": self.b0 + self.cut_offsets(), "mix": torch.tanh(self.ordinal_mix)}


class GraphModule(nn.Module):
    """Cross-question message passing over a bundle's questions.

    Present because the checkpoint carries it. The 5-arm ablation is the reason to read
    it as inert: with the gate initialised away from zero so gradients could flow
    (`sigmoid(-2.0) = 0.1192`), all five arms — scorer-only, graph-full, graph-off,
    shuffled, exact factor-graph decoding — returned the same 0.8899 dev question
    accuracy, so no arm's answer depended on this module. The shipped `gate_logit` is
    -2.3757 (sigmoid 0.0850): the gate moved 0.376 in logit space and stayed small.
    """

    d_model = 1024
    hidden = 512
    heads = 16

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.scorer = _readout(self.d_model, self.hidden, dropout)
        self.q_norm = nn.LayerNorm(self.d_model)
        self.q_attn = nn.MultiheadAttention(self.d_model, self.heads, batch_first=True)

    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(self, hq: torch.Tensor) -> torch.Tensor:
        """hq: [Q, d] question hidden states -> gated messages added back to each."""
        x = self.q_norm(hq)
        msg, _ = self.q_attn(x, x, x)
        return hq + self.gate() * self.scorer(msg).squeeze(-1).unsqueeze(-1) * hq


class Observation:
    """A scanned state. Mirrors `engine.Observation`'s surface for `myna.serve`."""

    def __init__(self, myna: "MynaV6", text: str, ids: list[int]):
        self.m = myna
        self.text = text
        self.ids = ids

    def append(self, text: str) -> "Observation":
        # The state is text here, not a retained activation: the encoder re-reads all of
        # it on the next ask. Kept as a new Observation so /v1/sessions works unchanged.
        combined = f"{self.text}\n{text}".strip() if text else self.text
        return Observation(self.m, combined, self.m._encode_state(combined))

    def ask(self, questions: dict) -> dict:
        return self.m._ask(self.text, questions)


class MynaV6:
    """The v6 engine: DeBERTa-v3-large encoder, cardinality-routed scoring, conformal gate."""

    def __init__(self, source: str | Path = "aashish254/myna-v6", device: str = "cpu",
                 abstain_below: float | None = None, max_length: int = 512):
        if abstain_below is not None and not 0.0 < float(abstain_below) <= 1.0:
            raise ValueError(f"--abstain-below must be in (0, 1], got {abstain_below!r}")
        root = self._resolve(Path(source) if str(source).startswith(("/", "."))
                             or Path(source).is_dir() else str(source))
        self.root = root
        self.device = device
        self.max_length = int(max_length)
        self.abstain_below = None if abstain_below is None else float(abstain_below)

        ckpt_path = next((root / n for n in CKPT_NAMES if (root / n).is_file()), None)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"{root} has no {' or '.join(CKPT_NAMES)}. Name a campaign directory that "
                "holds the checkpoint, or a Hugging Face repo id that carries it.")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        self.model_label = ck.get("model_label", "myna-v6")
        self.config = dict(ck["config"])
        self.max_packed_options = int(self.config.get("max_packed_options", 16))
        self.use_coral_score_head = bool(self.config.get("use_coral_score_head", False))
        self.b77_none_mode = str(self.config.get("b77_none_mode", "T_fallback"))
        self.b77_none_param = float(self.config.get("b77_none_param", 0.5))
        self.temperatures = {k: float(v) for k, v in self.config["temperatures"].items()}
        self.temperature = self.temperatures  # /v1/health prints this; it is per-primitive
        self.conformal_q90 = {k: float(v) for k, v in self.config["conformal_q90"].items()}
        self.conformal_q95 = {k: float(v) for k, v in self.config["conformal_q95"].items()}
        self.audits = {"dev": ck.get("dev_audit"), "test": ck.get("test_audit"),
                       "calibration": ck.get("calibration_test")}

        self.packed_scorer = _readout(self.d_model, 512, 0.1)
        self.b77_set_head = SetAwareBankingHead()
        self.ordinal_score_head = CoralOrdinalScoreHead()
        self.graph_module = GraphModule()
        for name, module in (("packed_scorer", self.packed_scorer),
                             ("b77_set_head", self.b77_set_head),
                             ("ordinal_score_head", self.ordinal_score_head),
                             ("graph_module", self.graph_module)):
            module.load_state_dict(ck[name], strict=True)
            module.to(device).eval()

        self.encoder, self.tok = self._load_backbone(root, device)
        self.head_params = sum(p.numel() for m in (
            self.packed_scorer, self.b77_set_head, self.ordinal_score_head,
            self.graph_module) for p in m.parameters())

    d_model = 1024

    @staticmethod
    def _resolve(source) -> Path:
        """A local campaign directory, or a Hub repo id downloaded into the HF cache."""
        if isinstance(source, Path):
            if not source.is_dir():
                raise FileNotFoundError(f"{source} is not a directory")
            return source
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(source, allow_patterns=HUB_PATTERNS,
                                       cache_dir=None))

    @staticmethod
    def _load_backbone(root: Path, device: str):
        # Imported here, not at module scope: transformers is an optional extra, and an
        # eager import would make `import myna` fail on every box that lacks it (CI runs
        # `uv sync --extra browser` and nothing installs transformers).
        from transformers import AutoModel, AutoTokenizer
        backbone = root / BACKBONE_DIR
        if not backbone.is_dir():
            raise FileNotFoundError(
                f"{root} has no {BACKBONE_DIR}/. A campaign directory needs the fine-tuned "
                "encoder beside the .pt; a Hub repo carries it under the same name.")
        tok = AutoTokenizer.from_pretrained(str(backbone))
        model = AutoModel.from_pretrained(str(backbone)).to(device).eval()
        return model, tok

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.encoder.parameters()) + self.head_params

    @torch.no_grad()
    def _encode_state(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _prefix(self, name: str, spec: dict) -> str:
        source = spec.get("source") or name.split("/")[0]
        qname = spec.get("qname") or name.split("/")[-1]
        return f"Task ({source} / {qname}): {_state_text(spec.get('instructions', qname))}"

    def _options(self, spec: dict) -> list[str]:
        """Option texts, rendered by `real_data._option_text` — the same renderer the
        training and eval loaders use for this suite.

        This is not a stylistic match. A `choice` criteria dict maps label -> description
        and most banking77 descriptions are `null`, so an ad-hoc `f"{label}: {desc}"`
        puts the literal string "None" into the prompt for 61 of its 77 options. The
        suite's own rule is the description when it is a non-empty string, otherwise the
        label with underscores turned into spaces.
        """
        qtype = spec["type"]
        if qtype == "choice":
            crit = spec["criteria"]
            if isinstance(crit, dict):
                return [_option_text(k, v) for k, v in crit.items()]
            return [_state_text(c) for c in crit]
        if qtype == "score":
            return [_state_text(c) for c in spec["criteria"]]
        if qtype == "noul":
            return ["no", "yes"]
        raise ValueError(f"unknown question type {qtype!r}")

    @staticmethod
    def _labels(spec: dict) -> list[str]:
        """Answer keys, which for a dict-criteria `choice` are the criteria keys — gold
        is `list(criteria).index(label)`, so the key is what a caller compares against."""
        crit = spec.get("criteria", {})
        if spec["type"] == "choice":
            return list(crit) if isinstance(crit, dict) else [str(c) for c in crit]
        if spec["type"] == "score":
            return [str(c) for c in crit]
        return ["no", "yes"]

    def _special(self) -> tuple[list[int], list[int], int]:
        head = self.tok("[CLS]", add_special_tokens=False)["input_ids"]
        sep = self.tok("[SEP]", add_special_tokens=False)["input_ids"]
        mask = self.tok("[MASK]", add_special_tokens=False)["input_ids"]
        if len(mask) != 1:
            raise ValueError(f"[MASK] is not one token in this tokenizer: {mask}")
        return head, sep, mask[0]

    def _packed_ids(self, prefix: str, state: str, options: list[str]) -> tuple[list[int], list[int]]:
        """One packed sequence: task, then context, then every option with its own MASK.

        Returns (token ids, MASK positions). The state is truncated before the options
        are, because dropping context degrades one shared span while dropping an option
        deletes a candidate answer. When the option spans themselves overflow the budget
        the tail is cut, and the cut options simply have no MASK position — that is the
        v5 packed failure on `banking77/intent` (0.0086), reproduced rather than hidden.
        """
        head, sep, mask_id = self._special()
        context = self.tok(f" {prefix} | Context: {state}", add_special_tokens=False)["input_ids"]
        spans = [self.tok(" " + opt, add_special_tokens=False)["input_ids"] + [mask_id] + sep
                 for opt in options]
        tail = sum(len(s) for s in spans) + len(sep)
        room = (self.max_length - 1) - len(head) - tail
        if room < 1:
            spans, running, kept = [], 0, []
            for s in spans:
                if running + len(s) + len(sep) > self.max_length - 1 - len(head) - 1:
                    break
                kept.append(s)
                running += len(s)
            spans, context, room = kept, context[:1], 1
        context = context[:room]
        body: list[int] = []
        positions: list[int] = []
        for s in spans:
            # MASK sits at the end of the chunk, before the span's trailing [SEP].
            positions.append(len(head) + len(context) + len(body) + len(s) - 1 - len(sep))
            body += s
        return head + context + body + sep, positions

    @torch.no_grad()
    def _packed_hidden(self, seq: list[int], positions: list[int]) -> torch.Tensor:
        if not positions:
            return torch.zeros(0, self.d_model, device=self.device)
        ids = torch.tensor([seq], dtype=torch.int64, device=self.device)
        out = self.encoder(input_ids=ids, attention_mask=torch.ones_like(ids))
        idx = torch.tensor(positions, device=self.device)
        return out.last_hidden_state[0, idx]

    @torch.no_grad()
    def _decoupled_hidden(self, prefix: str, state: str, options: list[str]) -> torch.Tensor:
        """Each option encoded against the state on its own: [O, d].

        This is the high-cardinality path. The state is re-encoded once per option, so
        the batch is O sequences wide rather than one sequence O options long, which is
        what lets a 77-option question see all 77 options.

        The MASK index comes from `_packed_ids`, never from arithmetic on the sequence
        length: a single-option sequence ends in two `[SEP]` tokens (the span's own, then
        the sequence terminator), so a `len(seq) - 2` guess reads a `[SEP]` hidden state
        instead of the `[MASK]` one. That mistake scored every option off a delimiter and
        returned a near-uniform distribution over 77 labels; reading the real MASK position
        puts `banking77/intent`'s clean rows back at the rate the witness publishes.
        """
        seqs, positions = [], []
        for opt in options:
            seq, positions_i = self._packed_ids(prefix, state, [opt])
            if not positions_i:
                break
            seqs.append(seq)
            positions.append(positions_i[0])
        if not seqs:
            return torch.zeros(0, self.d_model, device=self.device)
        width = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.full((len(seqs), width), pad, dtype=torch.int64, device=self.device)
        omask = torch.zeros(len(seqs), width, device=self.device)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, device=self.device)
            omask[i, : len(s)] = 1.0
        out = self.encoder(input_ids=ids, attention_mask=omask)
        idx = torch.tensor(positions, device=self.device)
        return out.last_hidden_state[torch.arange(len(seqs), device=self.device), idx]

    @torch.no_grad()
    def _option_logits(self, prefix: str, state: str, options: list[str],
                       labels: list[str]) -> torch.Tensor:
        """Scores for every label, with absent options filled at MASK_FILL.

        Length is always `len(labels)`, so the softmax cannot silently drop a candidate:
        an option that ran off the packed budget stays in the vector and is driven to
        zero probability, which is what the 0.0086 cell actually measured.
        """
        full = torch.full((len(labels),), MASK_FILL, device=self.device)
        if len(options) <= self.max_packed_options:
            seq, positions = self._packed_ids(prefix, state, options)
            hs = self._packed_hidden(seq, positions)
            if hs.shape[0]:
                full[: hs.shape[0]] = self.packed_scorer(hs).squeeze(-1)
            return full
        hs = self._decoupled_hidden(prefix, state, options)
        full[: hs.shape[0]] = self.b77_set_head.raw_option_logits(hs)
        return full

    def _primitive(self, spec: dict) -> str:
        return spec["type"]

    def _conformal(self, probs: list[float], primitive: str,
                   labels: list[str]) -> tuple[list[str], bool]:
        """Split-conformal 90% set, and whether it pins anything.

        The nonconformity is `1 - p`; an option is in the set when its nonconformity is
        under the quantile fitted on the 2,000-bundle `CAL` split. A set that excludes
        the argmax is a set that does not license an answer, which is the `abstain` case.
        """
        q = self.conformal_q90[primitive]
        keep = [lab for lab, p in zip(labels, probs) if (1.0 - p) <= q]
        return keep, len(keep) == 0

    def _readout(self, spec: dict, labels: list[str], logits: torch.Tensor,
                 primitive: str, floor: float | None) -> dict:
        temp = self.temperatures[primitive]
        probs = F.softmax(logits / temp, dim=-1)
        pl = [float(p) for p in probs]
        top = max(range(len(pl)), key=lambda k: pl[k])
        ordered = sorted(pl, reverse=True)
        margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
        conformal, empty = self._conformal(pl, primitive, labels)
        abstain = empty or (floor is not None and pl[top] < floor)
        reason = None
        if abstain:
            reason = (f"{labels[top]} at p={pl[top]:.3f} falls under the "
                      f"{1.0 - self.conformal_q90[primitive]:.3f} conformal floor for "
                      f"{primitive}")
        base = {"confidence": pl[top], "abstain": abstain, "reason": reason,
                "margin": margin, "conformal_set_90": conformal,
                "probabilities": {l: p for l, p in zip(labels, pl)}}
        if spec["type"] == "choice":
            return {"type": "choice", "choice": None if abstain else labels[top], **base}
        if spec["type"] == "score":
            return {"type": "score",
                    "score": None if abstain else sum(k * p for k, p in enumerate(pl)),
                    "level": None if abstain else labels[top],
                    "legend": {str(k): l for k, l in enumerate(labels)}, **base}
        return {"type": "noul", "noul": pl[1], "yes": None if abstain else pl[1] >= 0.5,
                **base}

    def _ask(self, state: str | dict | list, questions: dict,
             abstain_below: float | None = None) -> dict:
        # States arrive as str, {"document": ...} dicts, or segment lists; the suite's own
        # `_state_text` flattens all three, and the encoder only takes a string.
        state = _state_text(state)
        floor = self.abstain_below if abstain_below is None else float(abstain_below)
        t0 = time.perf_counter()
        answers, routing, usage = {}, {}, 0
        for name, spec in questions.items():
            labels = self._labels(spec)
            options = self._options(spec)
            prefix = self._prefix(name, spec)
            primitive = self._primitive(spec)
            logits = self._option_logits(prefix, state, options, labels)
            answers[name] = self._readout(spec, labels, logits, primitive, floor)
            routing[name] = {"mode": "packed" if len(options) <= self.max_packed_options
                             else "set-aware", "n_options": len(options),
                             "primitive": primitive,
                             "temperature": self.temperatures[primitive]}
            usage += len(self._encode_state(f"{prefix} | Context: {state}"))
        abstained = sorted(k for k, a in answers.items() if a["abstain"])
        return {"answers": answers,
                "usage": {"state_tokens": len(self._encode_state(state)),
                          "question_tokens": usage},
                "policy": {"abstain_below": floor, "abstained": abstained,
                           "conformal_level": 0.90, "b77_none_mode": self.b77_none_mode,
                           "b77_none_param": self.b77_none_param,
                           "use_coral_score_head": self.use_coral_score_head},
                "routing": routing,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    def observe(self, text: str) -> Observation:
        text = _state_text(text)
        return Observation(self, text, self._encode_state(text))

    def predict(self, state: str | dict | list, questions: dict,
                abstain_below: float | None = None) -> dict:
        """`abstain_below` is per-call as well as per-engine: a session that has already
        paid for the 1.7 GB encoder load must not have to load a second one to ask with
        a different floor."""
        return self._ask(state, questions, abstain_below)

    def audit(self, split: str = "test") -> dict:
        """The audit block stored inside the checkpoint, which the witness JSON mirrors."""
        block = self.audits[split]
        if block is None:
            raise KeyError(f"{split}: this checkpoint carries {sorted(self.audits)}")
        return block


def is_v6_directory(path: str | Path) -> bool:
    """Whether `path` holds a v5/v6 campaign checkpoint rather than a legacy `model.pt`."""
    p = Path(path)
    if not p.is_dir():
        return False
    return any((p / n).is_file() for n in CKPT_NAMES) or (p / BACKBONE_DIR).is_dir()


def is_v6_source(source: str | Path) -> bool:
    """Whether `Myna(...)` should hand `source` to the v6 engine.

    Contents only. A `user/repo` Hugging Face id is deliberately not detected here: it is
    indistinguishable in shape from a relative checkpoint path like `runs/myna-v0`, and a
    heuristic that mistook the latter for a repo id would send a fresh clone off to the Hub
    and report a network error instead of "run `myna-weights`". Pass a repo id to `MynaV6`
    directly — it resolves the Hub — or fetch it locally first.
    """
    return is_v6_directory(source)

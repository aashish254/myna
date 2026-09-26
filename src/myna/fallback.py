"""Fallback seam: an abstention routes, it does not dead-end (SPEC §5 P5, G5).

The rule this file exists for: *a fast path that is uncertain must never be
silently load-bearing.* Myna abstains at the engine level (`myna.engine`), so the
only thing a caller can do with an uncertain decision is hand it to someone else —
and the handoff has to be visible in the answer it produced. So every answer
leaving `Fallback.predict` carries `model`: the name of the engine that
**committed** to it, read from a registry rather than inferred from which call
happened to return.

Two properties the tests pin down, because both are the kind of bug that only
shows up in production:

* one primary and one fallback, both resolved from the registry, so no single
  model is load-bearing in the code path — dropping the fallback raises at
  construction instead of silently becoming "always abstain";
* the fallback is asked **only** the abstained questions. Passing the full set
  would re-ask what the fast engine already answered and make the cost of
  abstention invisible in the per-decision labels.

`LayaDecider` is the adapter for the one fallback wired up today. It imports laya
lazily, because laya is not a dependency of this package — it is a checkout on
`PYTHONPATH`, and a module that died at import time would take the tests with it.
"""

from __future__ import annotations

from .engine import Myna


class Decider:
    """The seam: `predict(state, questions)` -> laya-shaped answers dict.

    Anything with that shape can be a primary or a fallback, which is what lets
    the registry below hold `myna` and `laya` as peers instead of one model
    wrapped in special cases."""

    name = "decider"

    def predict(self, state: str, questions: dict) -> dict:  # pragma: no cover
        raise NotImplementedError


def laya_spec(questions: dict) -> dict:
    """myna's typed spec -> laya's. One real difference, and it is measured rather
    than remembered: laya's `noul` wants `criteria` omitted, so myna's `["no",
    "yes"]` list raises there (`bench/eval_laya_real.py` carries the same shim for
    the same reason). Anything that smells like a gold label is stripped, because
    the fallback is being asked to decide, not to be told.
    """
    out = {}
    for qid, q in questions.items():
        if q.get("type") == "noul":
            out[qid] = {"type": "noul", "instructions": q["instructions"]}
        else:
            out[qid] = {k: v for k, v in q.items() if k != "label"}
    return out


class LayaDecider(Decider):
    """laya's `Agent.system_one` behind the same shape, used as the slow-but-stronger
    side.

    A `Router` is deliberately not used: it adds its own model selection, so a
    per-decision `model` label would name the router rather than the thing that
    answered (SPEC §9.1's equal-footing rule, applied to the fallback path)."""

    name = "laya"

    def __init__(self, ckpt_dir: str, device: str = "cpu"):
        from laya import Agent  # lazy: laya lives on PYTHONPATH, not in pyproject

        self.agent = Agent(ckpt_dir, device=device)
        self.ckpt_dir = str(ckpt_dir)

    def predict(self, state: str, questions: dict) -> dict:
        return self.agent.system_one(state, laya_spec(questions))


class MynaDecider(Decider):
    """Myna in the same seat as the fallback, so the router sees two peers."""

    name = "myna"

    def __init__(self, *args, **kwargs):
        self.myna = Myna(*args, **kwargs)

    def predict(self, state: str, questions: dict) -> dict:
        return self.myna.predict(state, questions)


#: name -> factory. A third engine is a registry entry and a constructor call,
#: not a new branch in `Fallback`.
DECIDERS: dict[str, type[Decider]] = {"myna": MynaDecider, "laya": LayaDecider}


def build(name: str, **kwargs) -> Decider:
    try:
        cls = DECIDERS[name]
    except KeyError:
        raise SystemExit(f"unknown decider {name!r}; registry has: {', '.join(sorted(DECIDERS))}")
    return cls(**kwargs)


class Fallback:
    """Ask the fast engine, re-ask only what it refused, label every answer."""

    def __init__(self, primary: Decider, secondary: Decider, max_rounds: int = 1):
        if primary is secondary:
            raise ValueError("primary and secondary must be different engines")
        if max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1, got {max_rounds}")
        self.primary, self.secondary = primary, secondary
        self.max_rounds = max_rounds

    def predict(self, state: str, questions: dict) -> dict:
        out = self.primary.predict(state, questions)
        answers = dict(out["answers"])
        # read the abstentions from the *answers*, not from a policy echo: an
        # engine that abstains without saying so is the bug this class exists to
        # survive, and the only witness available is the per-answer flag.
        pending = sorted(k for k, a in answers.items() if a.get("abstain"))
        rounds = 0
        while pending and rounds < self.max_rounds:
            rounds += 1
            sub = self.secondary.predict(state, {k: questions[k] for k in pending})
            for k in pending:
                a = sub["answers"].get(k)
                if a is None:
                    continue
                answers[k] = {**a, "model": self.secondary.name}
            pending = sorted(k for k in pending if answers[k].get("abstain"))
        for k in answers:
            answers[k].setdefault("model", self.primary.name)
        return {
            "answers": answers,
            "usage": {**out.get("usage", {}),
                      "routed_to_secondary": _n_routed(answers, self.secondary.name)},
            "routing": {"primary": self.primary.name, "secondary": self.secondary.name,
                        "abstained": sorted(k for k, a in out["answers"].items()
                                            if a.get("abstain")),
                        "still_abstained": pending, "rounds": rounds},
        }


def _n_routed(answers: dict, name: str) -> int:
    return sum(1 for a in answers.values() if a.get("model") == name)

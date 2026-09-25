"""Synthetic typed-decisions corpus.

Three workflows, each asking the three typed question kinds (choice, score,
noul). Text is generated from templates conditioned on the gold labels, so
every label is derivable from the state — the model must read, not guess.
Splits are disjoint seeds and disjoint nouns, so dev/test measure
generalization to new surface forms, not memorization.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Question:
    name: str
    type: str  # choice | score | noul
    instruction: str
    options: list[str]  # ordered; for score this is the ordinal legend


@dataclass
class Example:
    state: str
    workflow: str
    gold: tuple[int, ...]  # option index per question


# cue pools: question#option-index -> predicate fragments with {n} {item} slots

SUPPORT_CUES = {
    "department#0": [
        "was charged {n} times for the {item}",
        "the invoice for the {item} is wrong by {n} dollars",
        "the refund for the {item} never arrived on my card",
        "the payment for the {item} failed {n} times in a row",
    ],
    "department#1": [
        "the {item} crashes every time I open the settings",
        "the dashboard shows error {n} right after login",
        "the {item} does not sync with my other devices anymore",
        "the export button in the {item} is completely broken",
    ],
    "department#2": [
        "the package with the {item} has not moved in {n} days",
        "the tracking number for the {item} shows absolutely nothing",
        "the delivery of the {item} arrived at the wrong address",
        "the courier lost the box containing my {item}",
    ],
    "department#3": [
        "I want to send the {item} back and exchange it for another",
        "the {item} arrived in the wrong size and I need a swap",
        "where exactly do I mail the {item} back to",
        "the {item} was damaged out of the box and I want a replacement",
    ],
    "department#4": [
        "I have a question about the partner program",
        "who maintains the privacy policy page these days",
        "the weekly newsletter goes to the wrong address book",
    ],
    "urgency#0": [
        "whenever you get a chance with the {item}",
        "no rush at all on the {item}",
        "someday maybe look at the {item}",
    ],
    "urgency#1": [
        "please look at the {item} sometime soon",
        "this is starting to eat my week on the {item}",
        "kindly reply about the {item} when you can",
    ],
    "urgency#2": [
        "I need the {item} working today",
        "this blocks my whole team from using the {item}",
        "it is urgent that the {item} works again",
    ],
    "urgency#3": [
        "the {item} is broken RIGHT NOW and we are losing money",
        "this is an emergency, the {item} burned our whole pipeline",
        "stop everything and fix the {item} this minute",
    ],
    "churn_risk#0": [
        "I keep using the {item} no matter what",
        "I have been a loyal customer for {n} years",
        "please keep improving the {item}",
    ],
    "churn_risk#1": [
        "I plan to cancel the subscription today",
        "otherwise I will leave for a competitor",
        "I am ready to end the contract over the {item}",
        "I will not renew unless this changes fast",
    ],
}

INCIDENT_CUES = {
    "severity#0": [
        "a typo on the help page about the {item}",
        "one user on one device sees a warning near the {item}",
        "the {item} log line is noisy but harmless",
    ],
    "severity#1": [
        "a few customers report slow {item} responses",
        "the {item} retries succeed but with errors",
        "degraded {item} latency affects one region a little",
    ],
    "severity#2": [
        "most requests to the {item} are failing",
        "the {item} is broken for every enterprise account",
        "checkout through the {item} fails for many many users",
    ],
    "severity#3": [
        "the entire {item} service is offline",
        "the {item} outage is total and money is bleeding",
        "nothing at all reaches the {item}, a site-wide failure",
    ],
    "team#0": [
        "the {item} timeouts spike on the load balancer",
        "dns records for the {item} are stale",
        "packet loss hits the {item} gateway",
    ],
    "team#1": [
        "the {item} queries deadlock on the primary",
        "the {item} writes fail after the replication lag",
        "the index on the {item} table is corrupt",
    ],
    "team#2": [
        "the {item} button renders blank in the app",
        "the {item} screen white-screens on reload",
        "the {item} form submits to a 404 page",
    ],
    "team#3": [
        "the {item} pods crash-loop on every node",
        "disk space on the {item} hosts is completely full",
        "the {item} deploy is stuck on a bad container image",
    ],
    "page_oncall#0": [
        "this can wait for business hours on the {item}",
        "no one needs to be woken for the {item}",
        "a ticket is enough for the {item} tonight",
    ],
    "page_oncall#1": [
        "wake the on-call engineer for the {item} immediately",
        "pages are authorized for the {item} incident",
        "it is 3am but the {item} must be paged now",
    ],
}

MOD_CUES = {
    "category#0": [
        "buy cheap copies of the {item} now at this link",
        "limited offer on the {item}, click here today",
        "make fast money reselling the {item} scheme",
    ],
    "category#1": [
        "you are worthless and everyone knows it about the {item}",
        "I will find you over that {item} post",
        "this person is an idiot for liking the {item}",
    ],
    "category#2": [
        "the {item} cure is secret and doctors hide it",
        "the {item} experiment proves anything you want",
        "the {item} study shows impossible results {n} times over",
    ],
    "category#3": [
        "here is my garden this spring beside the {item}",
        "a bread recipe that pairs with the {item}",
        "thoughts on the new documentary about the {item}",
    ],
    "action#0": [
        "the {item} report is obvious, action is clear",
        "remove this now, the {item} post is over the line",
        "ban the account that posted the {item}",
    ],
    "action#1": [
        "the {item} report needs a human second look",
        "it is borderline whether the {item} post breaks rules",
        "review the {item} post manually before acting",
    ],
    "action#2": [
        "the {item} report looks like a false alarm",
        "nothing here about the {item} breaks the rules",
        "dismiss the flags on the {item} comment",
    ],
    "repeat_appeal#0": [
        "no one has contested the {item} flag",
        "this is a first report of the {item} item",
        "the {item} post has never been appealed",
    ],
    "repeat_appeal#1": [
        "the author already appealed the {item} decision {n} times",
        "this is a repeat appeal about the {item} thread",
        "they keep contesting the {item} removal",
    ],
}

WORKFLOWS = {
    "support": (
        [
            Question("department", "choice", "Which team should handle this ticket?",
                     ["billing", "technical", "shipping", "returns", "other"]),
            Question("urgency", "score", "How urgent is this request?",
                     ["not urgent", "soon", "blocking", "drop everything"]),
            Question("churn_risk", "noul", "Does the writer threaten to cancel or leave?",
                     ["no", "yes"]),
        ],
        SUPPORT_CUES,
    ),
    "incident": (
        [
            Question("severity", "score", "What severity is this incident?",
                     ["cosmetic", "minor", "major", "critical"]),
            Question("team", "choice", "Which team owns the fix?",
                     ["network", "database", "frontend", "infra"]),
            Question("page_oncall", "noul", "Should the on-call engineer be woken up?",
                     ["no", "yes"]),
        ],
        INCIDENT_CUES,
    ),
    "moderation": (
        [
            Question("category", "choice", "What kind of content is this?",
                     ["spam", "harassment", "misinformation", "clean"]),
            Question("action", "score", "What action does the post justify?",
                     ["remove now", "human review", "dismiss"]),
            Question("repeat_appeal", "noul", "Has the author already contested a prior decision?",
                     ["no", "yes"]),
        ],
        MOD_CUES,
    ),
}

# disjoint nouns per split: dev/test use nouns the model never saw in training.
# The pool is deliberately large (~50/split) so any single (template, noun)
# pair is rare — keying on noun identity stops paying off relative to keying
# on the shared template words.
SPLIT_NOUNS = {
    "train": [
        "laptop", "router", "invoice", "app", "subscription", "charger", "portal", "headset",
        "account", "billing page", "checkout", "mobile site", "database", "smartwatch",
        "keyboard", "monitor stand", "warranty", "delivery slot", "loyalty card", "gift card",
        "wifi plan", "phone line", "browser tab", "desktop", "mouse", "webcam", "microphone",
        "standing desk", "office chair", "printer", "scanner", "sim card", "data plan",
        "cloud storage", "backup drive", "sd card", "power bank", "dock", "ethernet cable",
        "service ticket", "support chat", "refund request", "order summary", "shipping label",
        "courier bag", "store pickup", "return slip", "trade-in", "installment plan", "promo code",
    ],
    "dev": [
        "tablet", "modem", "receipt", "widget", "membership", "cable", "dashboard", "earbuds",
        "profile", "pricing page", "cart", "desktop app", "cache", "keyboard case",
        "laptop sleeve", "screen protector", "adapter plug", "surge protector", "server rack",
        "cooling fan", "battery pack", "memory stick", "network switch", "wireless tag",
        "tracking link", "shipping box", "pickup locker", "store credit", "annual plan",
        "trial period", "renewal notice", "support pin", "service plan", "care package",
        "device list", "pairing code", "sync folder", "upload queue", "download link",
        "preview build", "stable release", "nightly image", "test channel", "beta program",
        "feedback form", "rating star", "review page", "wishlist", "price alert", "coupon",
    ],
    "test": [
        "monitor", "switch", "statement", "gadget", "plan", "adapter", "console", "speakers",
        "workspace", "settings page", "gateway", "browser build", "queue", "stylus",
        "e-reader", "drawing pad", "voltage regulator", "extension cord", "label printer",
        "barcode gun", "pos terminal", "card reader", "nfc tag", "kiosk screen", "handheld",
        "wearable band", "home hub", "smart plug", "video doorbell", "motion sensor",
        "service contract", "maintenance plan", "trade counter", "repair shop", "loaner device",
        "courier note", "cargo bin", "pallet jack", "stock room", "pick list", "scan form",
        "delivery van", "route map", "drop point", "signature pad", "claim number",
        "policy doc", "terms page", "status feed", "incident log",
    ],
}

# generic noun dropout: with small probability every {item} in an example is
# replaced by a content-free placeholder, forcing template-only solutions
# templates embed nouns as "the {item}", so placeholders stay article-free
GENERIC_NOUNS = ["item", "thing", "product", "order", "package"]

SUBJECTS = [
    "Hello, I", "Hi team, I", "Hey, we", "Please help, I", "URGENT, we", "Again, I",
    "Look, I", "Honestly I", "We", "I",
]
FILLERS = [
    "This is my {n} message about it.",
    "Ticket number {n}.",
    "Thanks in advance.",
    "Sorry for the extra detail.",
    "My colleague saw it too.",
    "It started after the last update.",
    "Weekends are fine for a reply.",
    "You can call any time.",
    "The weather here is awful while I type this.",
]


def _fill(rng, fragment, nouns):
    out = fragment.replace("{item}", rng.choice(nouns))
    out = out.replace("{n}", str(rng.randint(2, 30)))
    return out


def _sentence(rng, cue_pools, key, nouns):
    frag = _fill(rng, rng.choice(cue_pools[key]), nouns)
    if rng.random() < 0.5:
        frag = frag[0].upper() + frag[1:]
    tail = rng.random()
    end = "." if tail < 0.75 else ("!" if tail < 0.95 else "?")
    return f"{rng.choice(SUBJECTS)} {frag}{end}" if tail >= 0.5 or rng.random() < 0.4 else f"{frag}{end}"


def generate(n, workflow, rng, split="train"):
    questions, cues = WORKFLOWS[workflow]
    split_nouns = SPLIT_NOUNS[split]
    out = []
    for _ in range(n):
        nouns = GENERIC_NOUNS if rng.random() < 0.15 else split_nouns
        labels = [rng.randrange(len(q.options)) for q in questions]
        parts = []
        for q, lab in zip(questions, labels):
            key = f"{q.name}#{lab}"
            for _ in range(rng.choice([1, 1, 2, 2, 3])):
                parts.append(_sentence(rng, cues, key, nouns))
            if rng.random() < 0.45:  # denial decoy: an option NOT chosen
                others = [i for i in range(len(q.options)) if i != lab]
                if others:
                    neg = rng.choice(others)
                    decoy = _sentence(rng, cues, f"{q.name}#{neg}", nouns)
                    parts.append(decoy[:-1] + " — but that is not what happened.")
        for _ in range(rng.choice([1, 2, 2, 3])):
            parts.append(_fill(rng, rng.choice(FILLERS), nouns))
        rng.shuffle(parts)
        out.append(Example(" ".join(parts), workflow, tuple(labels)))
    return out

"""Instruction paraphrase augmentation — the test that separates reading from templates.

Why this exists: myna's question branch reads a *free-text instruction*. If every
train row and every eval row shows the same instruction string, a model can score
well by keying on that string and never attending to what it asks. So each schema
gets hand-written phrasings, the **suite's own wordings are never used for
training**, and dev/test are scored on those exact wordings. Held-out wording is
then a measurement of reading, not of memorization.

Two shapes of instruction, two mechanisms (SPEC §5 P1):

* **Templated cores** — 17 distinct strings over the nine sources that ask the
  same way every row (measured on the pilot: 10,629 distinct instruction strings,
  of which 10,600 are boolq/mnli per-row content). Each gets nine phrasings.
* **Per-row content** — `boolq` (the instruction *is* the question) and `mnli`
  (`Hypothesis: "<row text>" How does it relate to the premise?`). Here only the
  frame varies; the row's own text is reproduced **verbatim**, because paraphrasing
  the question or the hypothesis could move the gold label.

Phrasing index 8 of 9 is reserved: training draws 0..7, and index 8 is the unseen
wording the model is *also* scored on, so "it transferred" is measured twice — on
the suite's wording and on a wording neither side ever saw.

Nothing here generates text: no model, no thesaurus, no network. A hand-written
table is auditable, and an audit of meaning-preservation is exactly what this gate
needs. An instruction that is not in the table raises `UnknownSchema` rather than
passing through unparaphrased — a silent pass-through would train on the eval
wording and quietly void the whole gate.
"""

from __future__ import annotations

import random

from .data import Question

N_VARIANTS = 9  # 0..7 for training, 8 held out
TRAIN_INDEXES = range(N_VARIANTS - 1)
EVAL_INDEX = N_VARIANTS - 1


class UnknownSchema(Exception):
    """A train instruction has no phrasing table. Fail loud: see module docstring."""


# The suite randomizes these three suffixes onto instruction tails. They are part
# of the held-out wording, so we strip them and re-attach our own equivalents.
SUITE_SUFFIXES = (
    " Consider the whole message.",
    " Pick the single best fit.",
    " Use only the information given.",
)

# Our suffix axis, composed by variant index so the nine phrasings vary the tail
# as well as the question — otherwise all nine differ only in their first words.
EXTRA_SUFFIXES = (
    "",
    "",
    " Answer only from the observation.",
    " Weigh the whole observation before choosing.",
    " Pick the closest of the listed options.",
    " Do not use anything outside the observation.",
)

# ---- templated cores: 17 strings, nine phrasings each -------------------------
# None of these lists contains the suite's own core string, and each keeps the
# referents the gold depends on (the topic named, the star scale, the policy
# outcomes) rather than drifting to a looser synonym.

CORES: dict[str, dict[str, list[str]]] = {
    "agnews": {
        "What is the topic of this article?": [
            "Which topic does this article fall under?",
            "What subject is this article reporting on?",
            "Classify the article by its overall subject.",
            "Which of the listed subjects does this article cover?",
            "What section would this article run in?",
            "Under which heading would you file this article?",
            "Name the main subject of this article.",
            "What is this article chiefly about?",
            "Which subject matches the article as a whole?",
        ],
        "Is this article about world news?": [
            "Does this article cover world news?",
            "Is the subject here world news?",
            "Would you file this article under world news?",
            "This article is a world news story - true or false?",
            "Does the report belong in the world news section?",
            "Is this a piece of world news?",
            "World news: is that what the article reports?",
            "Can this article be described as world news?",
            "Is the content of this article world news?",
        ],
        "Is this article about sports?": [
            "Does this article cover sports?",
            "Is the subject here sports?",
            "Would you file this article under sports?",
            "This article is a sports story - true or false?",
            "Does the report belong in the sports section?",
            "Is this a piece of sports coverage?",
            "Sports: is that what the article reports?",
            "Can this article be described as sports?",
            "Is the content of this article about sports?",
        ],
        "Is this article about science and technology?": [
            "Does this article cover science and technology?",
            "Is the subject here science and technology?",
            "Would you file this article under science and technology?",
            "This article is about science and technology - true or false?",
            "Does the report belong in the science and technology section?",
            "Is this a science and technology story?",
            "Science and technology: is that what the article reports?",
            "Can this article be described as science and technology?",
            "Is the content of this article science and technology?",
        ],
        "Is this article about business?": [
            "Does this article cover business?",
            "Is the subject here business?",
            "Would you file this article under business?",
            "This article is a business story - true or false?",
            "Does the report belong in the business section?",
            "Is this a piece of business coverage?",
            "Business: is that what the article reports?",
            "Can this article be described as business?",
            "Is the content of this article about business?",
        ],
    },
    "amazon": {
        "How many stars did this product reviewer give?": [
            "What star rating did the reviewer give the product?",
            "Out of five stars, how did the reviewer score this product?",
            "How many stars does this product review award?",
            "The reviewer handed this product how many stars?",
            "Which star level on the legend matches this product review?",
            "Score the review on the star scale: how many stars?",
            "How generous was the star rating in this product review?",
            "What number of stars does this product review imply?",
            "How would you map this product review onto the star scale?",
        ],
    },
    "banking77": {
        "Which banking intent best describes this customer message?": [
            "What banking intent is the customer trying to fulfil?",
            "Which intent does this customer message express?",
            "Match the customer message to one of the listed banking intents.",
            "What is the customer asking the bank to do here?",
            "Which of the intent options fits this message?",
            "Name the intent behind this banking message.",
            "This message corresponds to which banking intent?",
            "What kind of banking request is the customer making?",
            "Which intent label fits the customer's message?",
        ],
    },
    "contrastive": {
        "How is this claim handled under the policy?": [
            "What does the policy require for this claim?",
            "According to the policy, how should this claim be treated?",
            "Which of the listed policy outcomes applies to this claim?",
            "Under the policy, what happens to this claim?",
            "The policy says this claim is: which of the three?",
            "Decide by the policy how this claim is dealt with.",
            "What treatment does the policy give this claim?",
            "Which policy decision covers this claim?",
            "Read the policy and classify the claim's outcome.",
        ],
        "Is this return request within the policy window?": [
            "Does the policy window still cover this return request?",
            "Is this return inside the allowed time frame in the policy?",
            "By the policy, is this return request still in time?",
            "Has the policy window closed on this return request?",
            "Is the timing of this return request allowed by the policy?",
            "Does this return fall inside the policy's stated window?",
            "Within policy time limits: is this return request eligible?",
            "Is this return request timely under the policy?",
            "Does the policy still permit this return at this point in time?",
        ],
        "Is the applicant eligible?": [
            "Does the applicant meet the eligibility rules?",
            "According to the policy, is this applicant eligible?",
            "Is the applicant qualified under the stated rules?",
            "Does the policy allow this applicant?",
            "Eligible or not: judge the applicant against the policy.",
            "Is this applicant covered by the rules?",
            "Does this applicant satisfy the criteria?",
            "Can the applicant be accepted under the policy?",
            "Is the applicant's case admissible under the rules?",
        ],
        "What happens to this order?": [
            "How is this order dealt with?",
            "Which of the listed outcomes applies to this order?",
            "What is done with this order?",
            "How does this order end up being treated?",
            "Decide the fate of this order from the options given.",
            "What is the disposition of this order?",
            "Which outcome does this order receive?",
            "How is the order processed in the end?",
            "What action is taken on this order?",
        ],
    },
    "dbpedia14": {
        "Which category does the subject of this encyclopedia text belong to?": [
            "What class of thing is the subject of this encyclopedia entry?",
            "Which listed category fits the subject of this text?",
            "Categorise the subject of this encyclopedia article.",
            "What kind of entity does this encyclopedia text describe?",
            "Under which category would you place the subject of this entry?",
            "The subject of this text belongs to which class?",
            "Name the category of the thing this encyclopedia text is about.",
            "Which label describes what this encyclopedia entry is about?",
            "Classify the subject of this encyclopedia text.",
        ],
    },
    "imdb": {
        "Is this movie review positive?": [
            "Does this movie review speak positively of the film?",
            "Is the tone of this film review positive?",
            "This review praises the movie - true or false?",
            "Does the reviewer like the movie?",
            "Is the sentiment of this film review positive?",
            "Would you call this movie review a positive one?",
            "Does this review view the film favourably?",
            "Positive take on the movie: does this review give one?",
            "Is this film review overall positive?",
        ],
    },
    "sst5": {
        "What is the sentiment of this review sentence?": [
            "How does the writer of this review sentence feel?",
            "Which sentiment on the legend matches this review sentence?",
            "Is the feeling in this review sentence positive, negative, or neutral?",
            "Read the review sentence and name its sentiment.",
            "What tone does this review sentence carry?",
            "Score the sentiment of this one review sentence.",
            "Does this review sentence express praise, complaint, or neither?",
            "Which sentiment label fits the review sentence?",
            "Classify the emotion expressed in this review sentence.",
        ],
    },
    "trec": {
        "What kind of answer does this question ask for?": [
            "What type of answer is this question looking for?",
            "Which answer kind fits the question?",
            "If you answered this question, what sort of thing would the answer be?",
            "Classify what the question expects as its answer.",
            "What is being asked for in this question?",
            "Which of the listed answer types does this question demand?",
            "What category of response does this question call for?",
            "Is the question after a person, a place, a number, or something else?",
            "What kind of thing would count as an answer here?",
        ],
    },
    "yelp": {
        "How many stars did this reviewer give?": [
            "What star rating did this reviewer leave?",
            "Out of five stars, how did this reviewer score it?",
            "How many stars is this review worth?",
            "The reviewer gave how many stars?",
            "Which star level on the legend matches this review?",
            "Score this review on the star scale.",
            "How generous was this reviewer's star rating?",
            "What number of stars does this review imply?",
            "How would you map this review onto the star scale?",
        ],
        "Would this reviewer recommend the business?": [
            "Does this reviewer recommend the business?",
            "Would this reviewer send someone else to the business?",
            "Is the business recommended by this reviewer?",
            "This reviewer would recommend the business - true or false?",
            "Does the reviewer endorse the business?",
            "Would the reviewer tell others to go here?",
            "Is this a recommending review of the business?",
            "Does this reviewer vouch for the business?",
            "Going by this review, is the business worth recommending?",
        ],
    },
}

# ---- per-row content: the frame varies, the row's text never does -------------

BOOLQ_FRAMES = [
    "Answer from the passage, yes or no: {q}",
    "Read the passage, then reply to this: {q}",
    "Does the passage settle this as yes or no? {q}",
    "Going only by the passage: {q}",
    "The passage above decides this. {q}",
    "Decide yes or no from the passage: {q}",
    "According to the passage, {q}",
    "Using only the passage, {q}",
    "From the passage alone, answer this. {q}",
]

# mnli's option legend says "hypothesis"; these frames deliberately do not, so a
# model that keyed on the word would not transfer — the statement is the thing to
# find by reading, not by matching the legend's vocabulary.
MNLI_FRAMES = [
    "Statement under test: {h}. Judged against the passage above, does it follow, "
    "conflict, or remain open?",
    "Claim to check: {h} Decide from the passage: entailed, neutral, or contradicted?",
    "Read the passage, then place {h}: supported, opposed, or neither.",
    "Does the passage entail, contradict, or leave open {h}",
    "Compared with the passage, {h} - restated, unrelated, or impossible?",
    "The sentence {h} is true given the passage, false, or undetermined?",
    "Taking the passage as fact, judge {h}: follows, conflicts, or neither.",
    "Against the passage, {h} - entailed, consistent-but-unknown, or ruled out?",
    "Classify this statement relative to the passage above: {h}",
]

CONTENT_FRAMES: dict[str, list[str]] = {
    "boolq": BOOLQ_FRAMES,
    "mnli": MNLI_FRAMES,
}

MNLI_HEAD = "Hypothesis: "
MNLI_TAIL = "How does it relate to the premise?"


def source_of(group_key: str) -> str:
    """The adapter keys groups as `<source>#<signature>` (see `real_data.load_split`)."""
    return group_key.split("#", 1)[0]


def strip_suite_suffix(instruction: str) -> tuple[str, bool]:
    for s in SUITE_SUFFIXES:
        if instruction.endswith(s):
            return instruction[: -len(s)].rstrip(), True
    return instruction, False


def mnli_parts(instruction: str) -> str:
    """The hypothesis span, verbatim including its quotes. Raises on any other shape."""
    if not instruction.startswith(MNLI_HEAD) or MNLI_TAIL not in instruction:
        raise UnknownSchema(f"mnli instruction outside the shipped frame: {instruction[:70]!r}")
    return instruction[len(MNLI_HEAD): instruction.index(MNLI_TAIL)].strip()


def variants(instruction: str, source: str) -> list[str]:
    """Nine phrasings of one train instruction; index 8 is the held-out one.

    For templated sources the core is looked up in `CORES`. For `boolq`/`mnli` the
    frame is swapped and the row's own text is inserted verbatim, so no variant can
    move a gold label. Never returns the suite's own wording.
    """
    core, _had_suffix = strip_suite_suffix(instruction)
    if source in CONTENT_FRAMES:
        frames = CONTENT_FRAMES[source]
        content = instruction if source == "boolq" else mnli_parts(instruction)
        out = [f.format(q=content) if source == "boolq" else f.format(h=content) for f in frames]
        return [s + EXTRA_SUFFIXES[i % len(EXTRA_SUFFIXES)] for i, s in enumerate(out)]
    table = CORES.get(source, {})
    if core not in table:
        raise UnknownSchema(f"no phrasing table for {source}: {core[:70]!r}")
    out = table[core]
    if any(v.strip() == instruction.strip() for v in out):
        raise AssertionError(f"{source}: a phrasing equals the suite wording: {core[:60]!r}")
    return [s.strip() + EXTRA_SUFFIXES[i % len(EXTRA_SUFFIXES)] for i, s in enumerate(out)]


def paraphrase_questions(questions: list[Question], source: str, index: int) -> list[Question]:
    """Same options, same type, same order — only the instruction text changes.

    The gold is an *option index*, so as long as options are untouched it cannot move;
    `tests/test_paraphrase.py` asserts exactly that rather than trusting the intent.
    """
    if not 0 <= index < N_VARIANTS:
        raise ValueError(f"variant index {index} outside 0..{N_VARIANTS - 1}")
    out = []
    for q in questions:
        v = variants(q.instruction, source)[index]
        out.append(Question(q.name, q.type, v, list(q.options)))
    return out


def draw_index(rng: random.Random) -> int:
    """A training phrasing — never the held-out one."""
    return rng.randrange(N_VARIANTS - 1)


class Paraphraser:
    """The eight training phrasings of every question set, built once per set.

    Memoized because a step draws one of them per question set, and the variant
    objects have to be *stable*: `train.question_tokens` keys its length cache on
    `id(questions)`, and the `--max-q-cells` budget is only honest if the number it
    reads is the number the batch will pad to. So each set's variants are pre-built,
    and the cache is warmed with the **longest** variant of the set — the budget then
    prices the worst the sampler could pick instead of the lucky draw it got.
    """

    def __init__(self):
        self._sets: dict[str, list[list[Question]]] = {}
        self.draws = 0
        self.draws = 0

    def variant_set(self, group_key: str, questions: list[Question]) -> list[list[Question]]:
        key = group_key
        if key not in self._sets:
            source = source_of(key)
            self._sets[key] = [
                paraphrase_questions(questions, source, i) for i in TRAIN_INDEXES
            ]
        return self._sets[key]

    def draw(self, group_key: str, questions: list[Question], rng: random.Random) -> list[Question]:
        sets = self.variant_set(group_key, questions)
        self.draws += 1  # counted so the trainer can prove the loop consulted it
        return sets[rng.randrange(len(sets))]

    def warm(self, groups) -> int:
        """Pre-build every set's variants and return how many were built.

        `groups`: {group_key: (questions, examples)} — the train split. Raises
        `UnknownSchema` on the first instruction with no table, which is the point:
        the coverage of this split is a precondition of the gate, not a detail.
        """
        built = 0
        for key, (questions, _exs) in groups.items():
            if key in self._sets:
                continue
            self.variant_set(key, questions)
            built += 1
        return built

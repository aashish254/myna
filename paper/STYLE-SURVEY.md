# Style survey: 46 papers, and the conventions this paper adopts

> **Format note.** The final deliverables are IEEEtran and elsarticle papers, not ACL. The
> structural lessons below still apply (method figure first, compact results table with a footing,
> reliability and risk--coverage, limitations that name what could not be verified); the
> ACL-specific page and figure notes describe the earlier draft.

**Purpose.** Before writing the Myna paper I surveyed 46 published papers in the four literatures the paper
sits between — typed/zero-shot decision models, calibration and selective prediction, efficient sequence
models, and cascaded/early-exit inference — and read nine of them closely enough to copy their *structural*
moves (not their prose). This document records what was read, how it was read, and which conventions the
Myna paper follows. It is a working document for the repository, not part of the submission.

**Method, so the survey is checkable.** Metadata (title, authors, abstract, date) came from the arXiv API on
2026-11-03. Section skeletons came from the full text served by ar5iv.labs.arxiv.org (falling back to
arxiv.org/html) by extracting h2/h3 headings; figure captions for the nine close reads came from the same
HTML. The extraction scripts are paper/scripts/survey_papers.py, paper/scripts/analyze_survey.py,
paper/scripts/build_skeleton_table.py and paper/scripts/deep_read_captions.py; the raw records are
paper/data/style_survey_raw.json. Two 2026 papers (LLM-as-Jev and General Decision Models) have no ar5iv
HTML, so only their abstracts were read. No claim below about a paper's structure is made from memory: every
skeleton in the table is the paper's own headings.

## What the 46 papers have in common

| statistic | value | what it means for us |
|---|---|---|
| abstracts | median **162.5 words** (range 91–279) | one paragraph, no citations, no "in this paper we..." throat-clearing |
| abstracts containing "we" | **45 / 46** | active voice, first person plural |
| abstracts containing a number | **33 / 46** | the result is in the abstract, not only in the table |
| top-level sections | median **6** | Intro · Related/Background · Method · Experiments · Analysis/Conclusion (± Limitations) |
| Related Work as a top-level section | **26 / 46** | either a section or folded into the intro; never a bibliography dump |
| "ablation" anywhere in headings | **9 / 46** | ablations are usually a subsection of Experiments, not a separate paper |
| Limitations as a top-level heading | **4 / 46** | undercounts: many recent ACL/EMNLP papers fold it into "Discussion" or "Conclusion" |
| Ethics as a top-level heading | **3 / 46** | required by ACL/EMNLP for accepted papers; often in the appendix |

The structural template that recurs, across venues, is:

1. **Introduction** — a task that matters, the specific gap, the method in one sentence, three or four
   contribution bullets, and a "we show" paragraph with the headline numbers.
2. **Related work / background** — grouped by idea, ending with a positioning paragraph that says what this
   paper does that the cited work does not.
3. **Method** — one overview figure, a formal statement of the contract, the model, the objective, and an
   algorithm block if the procedure is not obvious from equations.
4. **Experiments** — dataset table first, baselines with provenance, main table, then ablations and analysis.
5. **Discussion / limitations** — what the numbers do not license.
6. **Conclusion** — two paragraphs, no new claims.
7. **Appendix** — full tables, prompts, hyperparameters, commands, and the artifact map.

## Nine close reads and what each contributes

### PET — Schick & Schütze, EACL 2021 (arXiv 2001.07676)
Skeleton: Introduction · Related Work · Pattern-Exploiting Training · Experiments · Analysis · Conclusion.
Figure 1 is the method schematic: training examples are converted into cloze questions, one model per
pattern, then an ensemble. Table 1 reports **average accuracy and standard deviation** over five training-set
sizes; Table 2 is the head-to-head with two prior semi-supervised methods. Lesson: put the method schematic
first, and put the variance beside every average.

### Banking77 — Casanueva et al., EMNLP 2020 (arXiv 2003.04807)
Skeleton: Introduction · Methodology · New Dataset · Experimental Setup · Results and Discussion · Conclusion.
Table 1 is the **dataset statistics** table (classes, train/test sizes, average utterance length); Table 2 is
example utterances per class; Table 3 is accuracy at 10/30/full shots with the peak per column in bold; Table 4
is a hyperparameter sweep. Lesson: a dataset table with counts and an example table are not optional when the
contribution is a benchmark; bold the best per column, not the best overall.

### Guo et al., ICML 2017 (arXiv 1706.04599)
Skeleton: Introduction · Definitions · Observing Miscalibration · Calibration Methods · Results · Conclusion.
Figure 1 is the canonical **confidence histogram (top) + reliability diagram (bottom)**; Figure 2 isolates the
effect of depth, width, batch norm and weight decay on ECE; Table 1 reports ECE with **M = 15 bins** stated in
the caption. Lesson: a calibration claim needs a reliability diagram and the bin count; accuracy is not
calibration evidence.

### SelectiveNet — Geifman & El-Yaniv, ICML 2019 (arXiv 1901.09192)
Skeleton: Introduction · Problem Formulation · Related Work · SelectiveNet · Coverage–Accuracy · Experimental
Design · Experiments · Empirical Observations · Concluding Remarks. Tables 1–3 report **selective risk at
several target coverage rates** for CIFAR-10 and SVHN, not a single operating point. Lesson: report the whole
coverage–risk curve; a single "98% at 60% coverage" is a point on a curve and must be shown as such.

### CQR — Romano, Patterson & Candès, NeurIPS 2019 (arXiv 1905.03222)
Skeleton: Introduction · Related work · Technical approach · Empirical evaluation · Summary. The empirical
table reports **average coverage and average interval length** together, with the nominal level in the
caption. Lesson: conformal claims need both coverage and set size, and the nominal level must be visible;
undercoverage is a result, not a footnote.

### Mamba — Gu & Dao, COLM 2024 (arXiv 2312.00752)
Skeleton: Introduction · State Space Models · Selective State Space Models · Empirical Evaluation · Discussion ·
Conclusion. Figure 1 is a three-part overview of the state-space recurrence; Figure 2 contrasts a standard
copying task with the selective-copying task that motivates the method; Algorithm 1 and 2 give the recurrence.
Lesson: for an architecture paper, the first figure teaches the mechanism, the second teaches the *problem*,
and the algorithm block is the precise contract.

### FastBERT — Liu et al., ACL 2020 (arXiv 2004.02178)
Skeleton: Introduction · Related work · Methodology · Experimental results · Conclusion · Future work. Figure 1
contrasts classic distillation with the proposed adaptive inference; Figure 2 is the inference process with a
batch of four; Table 1 is **FLOPs per operation**; Table 2 is accuracy and FLOPs/speedup on twelve datasets.
Lesson: a systems claim needs a cost table beside the accuracy table, with the unit (FLOPs, ms, bytes) named.

### SetFit — Tunstall et al., EMNLP 2022 (arXiv 2209.11055)
Skeleton: Introduction · Related Work · SetFit · Experiments · Results · Multilingual Experiments · SetFit
Model Efficiency · Conclusion. Figure 1 is a **real-data teaser** (accuracy vs number of labeled examples),
not a cartoon; Figure 2 is the training block diagram; Table 1 lists model variants and parameter counts;
Table 2 reports mean ± standard deviation over six datasets. Lesson: the teaser figure should plot the paper's
actual claim, and parameter counts belong in the method/efficiency table.

### LLM.int8 — Dettmers et al., NeurIPS 2022 (arXiv 2208.07339)
Skeleton: Introduction · Background · Int8 Matrix Multiplication at Scale · Emergent Large Magnitude Features ·
Related work · Discussion and Limitations · Broader Impacts. Figure 1 is zero-shot accuracy across four
datasets for 16-bit, prior 8-bit and the proposed 8-bit; Figure 2 is the decomposition schematic; Table 1 is
perplexity across model sizes. Lesson: quantization papers show **quality and cost on the same axes**, and put
the limitation in the section title when the limitation is the finding.

## The conventions this paper adopts

1. **Title names the method and the three mechanisms**, not the venue fashion:
   *Myna: Cardinality-Aware Routing, Bundle Consistency, and Conformal Abstention for Non-Autoregressive
   Agent Decision Engines.*
2. **Abstract in five moves, 180–220 words**: the decision-setting problem; the three failure modes; what we
   build; three numbers with their conditions; what the numbers do not license. No citations.
3. **Introduction ends with four contribution bullets**, each pointing to a section and each carrying one
   number. The fourth bullet is the systems study, because the paper's second engine is a systems result.
4. **A "footing" paragraph before the main table** states in-domain vs zero-shot, parameter counts, and which
   engine each number belongs to. This is the paper's most important style decision: the repository's own
   discipline is that a number without its arm is not a result, and the paper inherits it.
5. **One architecture figure (Fig. 1), one diagnosis figure (Fig. 2), one ablation figure (Fig. 3), one
   consistency figure (Fig. 4), one calibration figure (Fig. 5), one systems figure (Fig. 6), one diagnostic
   figure (Fig. 7).** Every figure is generated from a committed JSON by paper/make_figures.py; no figure is
   a cartoon except Fig. 1, which is a schematic and is labelled as one.
6. **Dataset table first** (splits, rows, questions, cells, option cardinalities), then the main table, then
   ablations. The main table bolds the best per column and carries the aggregate rows (macro, overall, joint
   bundle exact, per-primitive).
7. **Intervals, not just points.** Every headline macro has a row-clustered 95% CI from
   bench/significance.py (20,000 replicates, seed 20261031); the caption names the unit (row, not question).
8. **Calibration is a reliability diagram plus ECE with the bin count**, and the conformal table reports
   nominal vs empirical coverage and set size. Undercoverage is reported, not hidden.
9. **Selective prediction is a curve**, with the operating point marked; the 60%-coverage number is read off
   the curve and the floor that realizes it is printed.
10. **Systems numbers carry their box**: platform, device, threads, warm-ups, reps, load average, dtype, and
    the checkpoint's parameter count. The v6 engine is explicitly excluded from the systems table because it
    re-encodes and has no browser build.
11. **Limitations is a subsection of Discussion, not a disclaimer**, and it names the four things this paper
    could not verify: the campaign's contradiction predicate, the 2,000-bundle CAL split, the none_absent
    decomposition, and v6's training objective.
12. **Appendix carries the full 16-cell table, the contradiction predicate and examples, the risk–coverage
    tables, the reproduction commands, and the artifact map.** The main text carries only what a reviewer
    needs to believe the claim.

## What we deliberately do not do

- No "state-of-the-art" claim: the comparison is in-domain v6 against zero-shot laya/vegaml, and the paper
  says so in the same sentence as the number.
- No teaser figure that is prettier than the result: the first data figure is the failure mode (packed
  collapse), not the win.
- No citation we did not open: every entry in refs.bib was resolved through the arXiv API or Crossref and
  its title is the API's title, not a remembered one. Two IDs in the first pass were wrong (ViT for LM-BFF,
  LayerNorm for BranchyNet); the survey script caught both because it compares the API title to the expected
  short name, and the corrected IDs are the ones in the table.
- No number in the abstract that does not appear in a table or figure with its arm named.

## The 46-paper table

Titles and venues are the API's; skeletons are the papers' own top-level headings. "— (abstract only)" means
no HTML full text was available (the two 2026 position/benchmark papers).

| # | Paper | Venue | Top-level sections (verified from full text) |
|---:|---|---|---|
| 1 | Attention Is All You Need | NeurIPS 2017 | Introduction · Background · Model Architecture · Why Self-Attention · Training · Results · Conclusion |
| 2 | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | NAACL 2019 | Introduction · Related Work · BERT · Experiments · Ablation Studies · Conclusion |
| 3 | DeBERTa: Decoding-enhanced BERT with Disentangled Attention | ICLR 2021 | Introduction · Background · The DeBERTa Architecture · Scale Invariant Fine-Tuning · Experiment · Conclusions · Acknowledgments |
| 4 | DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing | ICLR 2023 | Introduction · Background · DeBERTaV3 · Experiment · Conclusions |
| 5 | Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference | EACL 2021 | Introduction · Related Work · Pattern-Exploiting Training · Experiments · Analysis · Conclusion |
| 6 | Making Pre-trained Language Models Better Few-shot Learners | ACL 2021 | Introduction · Related Work · Problem Setup · Prompt-based Fine-tuning · Automatic Prompt Generation · Fine-tuning with Demonstrations · Experiments · Discussion · Conclusion |
| 7 | Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach | EMNLP 2019 | Introduction · Related Work · Benchmark the dataset · Benchmark the evaluation · An entailment model for 0shot-tc · Experiments · Summary |
| 8 | Efficient Intent Detection with Dual Sentence Encoders | EMNLP 2020 | Introduction · Methodology: Intent Detection with Dual Sentence Encoders · New Dataset: banking77 · Experimental Setup · Results and Discussion · Conclusion |
| 9 | GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding | ICLR 2019 | Introduction · Related Work · Tasks · Diagnostic Dataset · Baselines · Benchmark Results · Analysis · Conclusion |
| 10 | SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems | NeurIPS 2019 | Introduction · Related Work · SuperGLUE Overview · Using SuperGLUE · Experiments · Conclusion · Acknowledgments |
| 11 | A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference | ACL 2018 | Introduction · The Corpus · Baselines · Discussion and Analysis · Conclusion |
| 12 | BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions | ACL 2019 | Introduction · Related Work · The BoolQ Dataset · Training Yes/No QA Models · Results · Conclusion |
| 13 | On Calibration of Modern Neural Networks | ICML 2017 | Introduction · Definitions · Observing Miscalibration · Calibration Methods · Results · Conclusion |
| 14 | Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift | NeurIPS 2019 | Introduction · Background · Methods and Metrics · Experiments and Results · Takeaways and Recommendations |
| 15 | A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification | arXiv 2021 | Conformal Prediction · Examples of Conformal Procedures · Evaluating Conformal Prediction · Extensions of Conformal Prediction · Worked Examples · Full conformal prediction · Historical Notes on Conformal Prediction |
| 16 | Conformalized Quantile Regression | NeurIPS 2019 | Introduction · Quantile regression · Conformal Prediction · Conformalized quantile regression (CQR) · Related work: locally adaptive conformal prediction · Experiments · Conclusion |
| 17 | Distribution-Free Predictive Inference For Regression | JASA 2018 | Introduction · Conformal Inference · Statistical Accuracy · Empirical Study · Extensions of Conformal Inference · Model-Free Variable Importance: LOCO · Conclusion |
| 18 | Selective Classification for Deep Neural Networks | ICML 2017 | Introduction · Problem Setting · Selection with Guaranteed Risk Control · Confidence-Rate Functions for Neural Networks · Empirical Results · Concluding Remarks |
| 19 | SelectiveNet: A Deep Neural Network with an Integrated Reject Option | ICML 2019 | Introduction · Selective Prediction Problem Formulation · Related Work · SelectiveNet · Coverage Accuracy · Experimental Design and Details · Experiments · Empirical Observations · Concluding Remarks |
| 20 | Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles | NeurIPS 2017 | Introduction · Deep Ensembles: A Simple Recipe For Predictive Uncertainty Estimation · Experimental results · Discussion |
| 21 | What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? | NeurIPS 2017 | Introduction · Related Work · Combining Aleatoric and Epistemic Uncertainty in One Model · Experiments · Analysis: What Do Aleatoric and Epistemic Uncertainties Capture? · Conclusions |
| 22 | Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention | ICML 2020 | Introduction · Related Work · Linear Transformers · Experiments · Conclusions |
| 23 | Gated Linear Attention Transformers with Hardware-Efficient Training | ICML 2024 | Introduction · Background: Linear Attention · Hardware-Efficient Linear Attention · Gated Linear Attention · Empirical Study · Related Work |
| 24 | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | COLM 2024 | Introduction · State Space Models · Selective State Space Models · Empirical Evaluation · Discussion · Conclusion |
| 25 | RWKV: Reinventing RNNs for the Transformer Era | ACL Findings 2023 | Introduction · Background · RWKV · Trained Models and Computing Costs · Evaluations · Inference Experiments · Future Work · Conclusions · Limitations · Ethics Statement |
| 26 | Retentive Network: A Successor to Transformer for Large Language Models | arXiv 2023 | Introduction · Retentive Networks · Experiments · Conclusion |
| 27 | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | NeurIPS 2022 | Introduction · Background · FlashAttention : Algorithm, Analysis, and Extensions · Experiments · Limitations and Future Directions |
| 28 | Longformer: The Long-Document Transformer | ACL 2020 | Introduction · Related Work · Longformer · Autoregressive Language Modeling · Pretraining and Finetuning · Tasks · Longformer-Encoder-Decoder (LED) · Conclusion and Future Work |
| 29 | Big Bird: Transformers for Longer Sequences | NeurIPS 2020 | Introduction · BigBird Architecture · Theoretical Results about Sparse Attention Mechanism · Experiments: Natural Language Processing · Experiments: Genomics · Conclusion |
| 30 | Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context | ACL 2019 | Introduction · Related Work · Model · Experiments · Conclusions |
| 31 | BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks | ICPR 2016 | — (abstract only) |
| 32 | FastBERT: a Self-distilling BERT with Adaptive Inference Time | ACL 2020 | Introduction · Related work · Methodology · Experimental results · Conclusion · Future work |
| 33 | DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference | ACL 2020 | Introduction · Related Work · Early Exit for BERT inference · Experiments · Conclusions and Future Work |
| 34 | Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer | NeurIPS 2018 | Introduction · Learning to Defer · Formulating Adaptive Models within Decision Systems · Related Work · Experiments · Conclusion |
| 35 | A Semantic Loss Function for Deep Learning with Symbolic Knowledge | ICML 2018 | Introduction · Background and Notation · Semantic Loss · Semi-Supervised Classification · Learning with Complex Constraints · Related Work · Conclusions & Future Work |
| 36 | Self-Consistency Improves Chain of Thought Reasoning in Language Models | ICLR 2023 | Introduction · Self-Consistency over Diverse Reasoning Paths · Experiments · Related work · Conclusion and Discussion · Reproducibility Statement · Ethics Statement |
| 37 | Efficient Guided Generation for Large Language Models | arXiv 2023 | Introduction · LLM Sampling and Guided Generation · Iterative FSM Processing and Indexing · Extensions to Iterative Parsing · Discussion |
| 38 | Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? | EMNLP 2022 | Introduction · Related Work · Experimental Setup · Ground Truth Matters Little · Why does In-Context Learning work? · Discussion & Conclusion |
| 39 | DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter | arXiv 2019 | Introduction · Knowledge distillation · DistilBERT: a distilled version of BERT · Experiments · Related work · Conclusion and future work |
| 40 | Parameter-Efficient Transfer Learning for NLP | ICML 2019 | Introduction · Adapter tuning for NLP · Experiments · Related Work |
| 41 | LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale | NeurIPS 2022 | Introduction · Background · Int8 Matrix Multiplication at Scale · Emergent Large Magnitude Features in Transformers at Scale · Related work · Discussion and Limitations · Broader Impacts |
| 42 | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | ICLR 2023 | Introduction · Related Work · Background · The GPTQ Algorithm · Experimental Validation · Summary and Limitations · Ethics Statement · Reproducibility Statement |
| 43 | SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models | ICML 2023 | Introduction · Preliminaries · Review of Quantization Difficulty · SmoothQuant · Experiments · Related Work · Conclusion |
| 44 | Efficient Few-Shot Learning Without Prompts | EMNLP 2022 | Introduction · Related Work · SetFit: Sentence Transformer Fine-Tuning · Experiments · Results · Multilingual Experiments · SetFit Model Efficiency · Conclusion |
| 45 | LLM-as-Jev: LLMs Are Already Jev-Style Decision Models -- When and How to Fine-Tune Them | arXiv 2026 | — (abstract only) |
| 46 | General Decision Models: Benchmarking and Insights Beyond Jev | arXiv 2026 | — (abstract only) |

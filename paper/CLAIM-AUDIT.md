# Claim audit: the brief vs. the repository

> **Final-deliverable note (2026-11-03).** The IEEE and Elsevier papers in paper/ieee and
> paper/elsevier remove every Laya/VegaML column and mention at the author's request. This audit
> remains the evidence record for the earlier comparison; the final papers report only Myna's own
> arms (v5 Packed, v5 Unified, v6, and a local v6 re-score).

**Why this file exists.** The paper brief supplied with this task contains a mixture of numbers that are
exactly measured, numbers that are measured on a different arm than the sentence implies, and a few claims
that no committed artifact supports. The paper cannot inherit that mixture. This file goes claim by claim:
what the brief says, what the repository actually contains, and what the paper writes instead. Every
"evidence" cell names a committed file; where the evidence is a computation I ran, the script and its output
are named.

**Verdict vocabulary.**
- **Supported** — the number/claim is in a committed witness and means what the brief says.
- **Supported with a condition** — the number is real but belongs to a different arm, split, or model than
  the sentence implies; the condition must travel with the number.
- **Partially supported** — the direction is real but the magnitude, mechanism, or denominator in the brief
  is wrong.
- **Not supported** — no committed artifact contains it; the paper drops or explicitly marks it unverified.

---

## Option 1 — the flagship accuracy/consistency/calibration claims

| # | Brief claim | Repository evidence | Verdict | What the paper says |
|---|---|---|---|---|
| 1.1 | v5 Packed 0.8173 → v5 Unified 0.8502 → v6 0.8583; routing alone +0.0329 with the other 15 cells byte-identical | runs/witness-v5-master-paper.json (test 0.8173027), runs/witness-hybrid-v3.json (test 0.8501691), runs/witness-v6-master-paper.json (0.8582510); README.md:35–58 and 78–82 | **Supported** | Table 2 and Fig. 3; the +0.0329 is stated as 0.5259/16 from one cell (banking77) |
| 1.2 | banking77 0.0086 → 0.5345 → 0.6638 | witness-v5-master (0.0086207), witness-hybrid-v3 (0.5344828), witness-v6 (0.6637931); per-row option counts in paper/data/token_budget.json | **Supported** | Table 1/2 and Fig. 2; the cell is named as the only mover |
| 1.3 | Packing "works up to K ≤ 16 options" | paper/data/token_budget.json: every non-banking77 choice cell has K = 3–15 and **zero** overflow; the 77-way cell overflows on 50/116 rows | **Partially supported** | "All K ≤ 15 cells fit with zero dropped options; 16 is a design cap with one option of headroom, not a measured optimum" |
| 1.4 | 77 options leave "~3 tokens per option" | token_budget.json: mean option span 490.3 tokens, **6.36 tokens/option**, median state 25 tokens, 50/116 overflow, mean kept 75.9/77, max dropped 6 | **Not supported (number)** | "Option spans alone consume 490 tokens on average (6.4 tokens/option); 43% of rows overflow and the fallback collapses the context to one token. The collapse is budget exhaustion plus a packed head that does not scale, not a 3-token-per-option truncation." |
| 1.5 | A float32 softmax consistency BCE loss with CONSISTENCY_WEIGHT = 0.5 was trained alongside cross-entropy | No occurrence of CONSISTENCY_WEIGHT or a consistency loss in src/, bench/, the v6 checkpoint config, or any witness; the checkpoint config has only max_packed_options, use_coral_score_head, b77_none_mode, temperatures, conformal quantiles | **Not supported** | The paper describes the **measured** consistency, not a training loss. "We do not have the training code; the released checkpoint does not document a consistency objective, so we make no claim about one." |
| 1.6 | "Enforcing entailed readout across choice and noul primitives" | src/myna/v6_engine.py:507–533 scores each question independently; the GraphModule is not called in _ask; its gate_logit is -2.3757 (sigmoid 0.0850) and it is inert at inference | **Not supported** | "The readout is per-question. Cross-question consistency is a measured property of the shared encoder, not an enforced constraint; the graph module is present but inert, and we report that as a negative mechanism result." |
| 1.7 | v6 has 0/172 contradictions (0.00%) | witness-v6-master-paper.json records "0/172"; the campaign predicate is not in the repository. Under the explicit predicate in paper/CLAIM-AUDIT.md §Consistency, the released row outcomes give **3/172** (2 agnews + 1 yelp) | **Supported as recorded, not reproducible** | "The checkpoint records 0/172; we could not reproduce it because the predicate is unreleased. Under our transparent predicate we measure 3/172, and we report the discrepancy rather than the headline." |
| 1.8 | Joint bundle exact-match 83.76% vs Laya 55.95–61.82% and VegaML 44.98% | witness-v6 (0.8375850), witness-competitor (0.5595238 / 0.5918367 / 0.6181973 / 0.4498299) | **Supported** | Table 1 and Fig. 4; the same-row recomputation gives 83.93% / 56.63% / 51.19% and is shown separately |
| 1.9 | Split-conformal sets fit on a 2,000-bundle CAL split | README.md:20/84 and the model card claim 2,000; the only committed calibration split is data/decision-v2-pilot/calibration.jsonl = **448 rows / 568 questions**; no witness records a 2,000-bundle fit | **Not verifiable** | "Fitted on a held-out calibration split (the campaign reports 2,000 bundles; the committed calibration.jsonl used for the risk–coverage control has 448 rows/568 questions)." |
| 1.10 | ECE 0.0189, Brier 0.0434 | witness-v6 calibration_test: ece 0.0189072, brier 0.0433770; our 10-bin recomputation from runs/myna_v6_test.rowoutcomes.json gives 0.0189922 | **Supported** | Table 3; the reliability diagram is our 10-bin recomputation and the caption says so |
| 1.11 | Selective accuracy 98.73% at 60% coverage | witness-v6: 0.9872685 (853/864); re-ranking the released outcomes by top probability gives 98.15% (848/864); the campaign's selective routine is not released | **Supported with a condition** | "Reported 98.73%; our top-probability re-ranking of the same outcomes gives 98.15%; the 0.58-point difference is unresolved." |
| 1.12 | 99.04% on banking77 + dbpedia14 + trec at 60% coverage | witness-v6: g5_acc_60pct_slice 0.9903846 = 103/104; risk_coverage_v6_g5trio_test.md: 0.990 at 60.1% (209 questions) on test, 0.986 at 60% on calibration | **Supported with a condition** | Table 3 and Fig. 5; "103/104 on the gate's trio slice; 99.0% on the 209-question test trio; the calibration curve is in-sample for the confidence mechanism" |
| 1.13 | 0.9264 vs Laya 0.8937 on "Laya's own 6 in-distribution cells" | README.md:73–77; the 6 cells are the 5 agnews cells + boolq; the mean reproduces exactly: v6 0.926421, Laya typed 0.893733, Laya English 0.864859, VegaML 0.414734 | **Supported with a condition** | "On the six cells in laya's training mixture, v6 0.9264 vs laya 0.8937; v6 is in-domain on all 16, laya is zero-shot on 10" |
| 1.14 | 0.8583 published / 0.8620 local macro | witness-v6 0.8582510; runs/v6_test.report.json 0.8620375; runs/significance_v6.md 0.8620 [0.8430, 0.8800] | **Supported** | Table 1 footnote and §5: both numbers, with the local arm the higher of the two |

---

## Option 2 — systems claims

| # | Brief claim | Repository evidence | Verdict | What the paper says |
|---|---|---|---|---|
| 2.1 | 14.45M–16.9M GLA streaming engine vs 446M fine-tuned cross-encoder behind an identical typed API | src/myna/engine.py (v0 14,449,280), runs/v1b-kaggle-3600b (16,926,848), v6 446,631,182; src/myna/__init__.py dispatch | **Supported** | §3 and Table 4; the two engines are named Myna-Stream and Myna-v6 |
| 2.2 | O(1) state memory; observations 128–16,384 tokens compress into a fixed 576 KiB state | src/myna/engine.py:66–73; 589,824 bytes = 576 KiB; runs/browser_g3.json state_bytes_total 589824 | **Supported** | §3.2 and Fig. 6b; the state is per-layer [H, dk, dv] and independent of length |
| 2.3 | 3.0×–4.1× end-to-end and 3.6×–6.8× cached-state speedups over Laya on M5 CPU | runs/latency_matched.json: e2e 3.04/3.63/4.19/4.13, ask-only 6.78/4.90/5.15/4.56; README.md:536–540 gives per-row minima 3.04/3.21/4.12/3.60 and 6.78/4.67/4.75/3.65 | **Supported with a condition** | "3.04–4.13× end-to-end and 4.56–6.78× ask-only in the committed run; per-row minima over two runs 3.60–6.78×" |
| 2.4 | MLX parity 1.76×–2.22× over PyTorch MPS | runs/bench_mlx_int8.md: fp32 1.76–2.22×; int8 1.79–2.41× | **Supported** | Table 4 and Fig. 6c |
| 2.5 | In-browser WebAssembly/ONNX parity 1.03e-6 max probability drift | runs/browser_g3.json checks: "3 decisions agree with the engine... worst |Δp| 1.03e-6 against 0.0001"; ONNX export parity 8.94e-8 | **Supported** | §7.3; the browser number is 1.03e-6, the export number 8.94e-8 |
| 2.6 | int8 17.40–18.56 MiB; 6.48e-2 drift in ONNX; argmax flips in MLX | runs/bench_mlx_int8.md: 17.40 MiB, 9.24e-3 drift, 1/20 flip; runs/int8_quantize.log: 18.56 MiB, 6.48e-2 drift | **Supported** | Table 4; both engines and both drift numbers are named |
| 2.7 | int8 slows 5 of 6 linear shapes; 10–37 GB/s vs 55–257 GB/s fp32 | runs/mlx_int8_gem_probe.md: 5/6 slower, bandwidth columns | **Supported** | §7.4 |
| 2.8 | Cascaded fallback logs the committing engine and conformal set | src/myna/fallback.py and src/myna/serve.py; README.md fallback section | **Supported** (mechanics) | §3.4; but no accuracy claim is attached to the fallback |

---

## Option 3 — diagnostic/negative-results claims

| # | Brief claim | Repository evidence | Verdict | What the paper says |
|---|---|---|---|---|
| 3.1 | v0 synthetic 0.9525; v1b 0.4893–0.5339 on the real suite | runs/g6_v0_synthetic_s0.json 0.9525; runs/g6_v1b_synthetic_s0.json 0.3423; runs/v1b_kaggle_3600b.report.json 0.4893; runs/antiprior_off_s0.report.json 0.5339 | **Supported with a condition** | §8; the 0.9525 is synthetic, the 0.4893/0.5339 are real-suite, and the anti-prior off arm is VOID (stopped at 3314/3600) |
| 3.2 | Anti-prior removes 4 constant emitters (4 → 0) | SPEC.md §10c and §9.55/§9.56; runs/antiprior_on_s0c, runs/tier0_antiprior_on_s0c.json | **Supported with a condition** | §8.2; "4 → 0 constant emitters, and the macro falls 0.5339 → 0.4785"; the pair is a continuation and the off arm is VOID |
| 3.3 | Blank-instruction accuracy rises +0.0974 from v1b to antiprior_off | SPEC.md:3253 (0.3341 → 0.4315); the off arm is VOID | **Supported with a condition** | §8.2; reported as evidence of prompt selectivity, with the VOID line in the same paragraph |
| 3.4 | 4k truncated backprop flatlines at chance: 0.183 at 4k, 0.150 at 16k, identical at steps 799 and 1199 | runs/needle_lc1_stageC_final.json rows 0.1833/0.150; runs/needle_lc1_stageB_resume.json; SPEC.md §9.63 | **Supported** | Fig. 7c; the 128-token rung is also 0.150, so it is a flat line at the 0.167 floor, not a decay curve |
| 3.5 | "Proves the model learned source-level majority priors rather than instruction semantics" | The Tier-0 deltas are consistent with the shortcut story, but the off arm is VOID and the comparison is one seed | **Partially supported** | "Consistent with a prior shortcut; not a proof. One seed, one dose, one VOID arm." |
| 3.6 | "A 16.9M scratch trunk lacks the pre-trained lexical representations" | Not measured; the paper has no representation probe, only accuracy | **Not supported** | "We observe the gap; we do not claim its cause. The representation explanation is a hypothesis." |

---

## The consistency predicate used in the paper

The suite has 172 multi-question bundles: 92 agnews bundles with a topic question and two of the four
is_* boolean questions, and 80 yelp bundles with a rating and a recommend question. The gold labels are
logically consistent (own-class boolean true; other-class booleans false; rating 4–5 ↔ recommend, 1–2 ↔ not
recommend). We define an answer as a contradiction iff:

- agnews: the predicted topic is class c and the bundle asks is_c, and the predicted is_c is "no"; or
- yelp: the predicted rating is 4–5 and recommend is "no", or 1–2 and recommend is "yes".

Under this predicate, on the released row outcomes:

| engine | run | bundle exact | contradictions | macro |
|---|---|---|---|---|
| Myna v6 | runs/myna_v6_test.rowoutcomes.json | 987/1176 = 83.93% | 3/172 (2 agnews + 1 yelp) | 0.8620 |
| Laya 0.4.19 | runs/laya_0419_test.rowoutcomes.json | 666/1176 = 56.63% | 31/172 (5 + 26) | 0.6664 |
| VegaML 0.15.0 | runs/vegaml_0150_test.rowoutcomes.json | 602/1176 = 51.19% | 29/172 (26 + 3) | 0.6160 |

The campaign witnesses record 0/172, 1/172, 0/172 and 0/172 for the four competitor columns. Those
predicates are not in the repository, and they do not reproduce under the predicate above. The paper reports
the campaign column as "as recorded" and the same-row table as the reproducible comparison. It does not
claim zero contradictions.

---

## Other discrepancies worth naming in the paper

1. **The 2,000-bundle CAL split** is in the model card and README, not in a witness. The committed
   calibration split is 448 rows. The paper says "held-out calibration split" and footnotes the committed
   size.
2. **The v6 training code is not in the repository.** The checkpoint is released; the training script is
   not. Any claim about the objective, the consistency loss, or the optimizer is therefore unverifiable, and
   the paper makes none.
3. **The CORAL ordinal head is inert.** config["use_coral_score_head"] is false; the score primitive is a
   packed argmax over the five level options. The paper says so and does not report an ordinal result.
4. **The graph module is inert.** gate_logit = -2.3757, sigmoid 0.0850; _ask never calls it. The paper says
   so.
5. **banking77's none_absent decomposition does not reproduce.** The campaign records 7/12; this box gets
   0/12 while the cell aggregate matches (78/116 vs 77/116). The paper names it as the one decomposition it
   cannot stand behind.
6. **The local re-score is higher than the published column** (+5 of 1,440 questions, macro 0.8620 vs
   0.8583). The paper reports both and uses the published column for the campaign head-to-head, the local
   column for the paired significance analysis.

---

## What the paper will not say

- It will not say "zero contradictions".
- It will not say "state of the art".
- It will not say "conformal coverage is 90%" — the empirical coverage is 85.49% at the 90% target.
- It will not say the 16.9M streaming engine achieves v6's accuracy.
- It will not say v6 is a streaming or constant-state engine.
- It will not say the anti-prior experiment proves a mechanism; it is one seed, one dose, one VOID arm.
- It will not cite a paper it did not open; the references are the 46 surveyed papers plus the three
  software artifacts (laya, vegaml, the browser runtime).

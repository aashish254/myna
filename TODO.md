# TODO — executable mirror of [SPEC.md](SPEC.md)

Derived from SPEC §5 (plan phases P0–P8), §2.2 (acceptance gates G1–G7) and §7 (discipline).
Checked items are done **and witnessed** — an item is only ticked when a command prints the thing
the item claims, per SPEC §7.1.

**Training policy for this loop (user directive, 2026-09-26):** nothing trains on the MacBook.
All model training runs on **Kaggle**; the MacBook is used for architecture code, data, tests,
inference/eval, and every on-device/MLX/MPS measurement (those *must* stay here — SPEC §6).
So every "train" item below is split into *prepare/verify the Kaggle path locally* (this loop) and
*run it on Kaggle* (user's credentials — labelled `KAGGLE`).

## P0 — Repo and measurement hygiene
- [x] Fix duplicate-row batching in `draw_batch()` (`385e06c`), mutation-checked
- [x] Same fix in `rlcd.py`
- [x] Startup diagnostic: median pool size, share of sets with pool < batch
- [x] Fit temperature on `calibration.jsonl`, not dev (`59240e1`)
- [x] `--device auto` considers CUDA (`59240e1`)
- [x] Correct SPEC §6's stale budget line ("budget by option count") against §9.9's measurement
- [x] `--help` renders for every CLI (`myna.train`, `myna.rlcd`, `myna.serve`) — argparse `%`
      crash and the uvicorn-before-parse_args crash, both pinned by `tests/test_cli_help.py`
- [ ] `bench/diag_learn.py` / `diag_overfit.py` take positional argv and reject `--help`; their
      V1-B numbers are void anyway (SPEC §9.3) — retire or convert them when V1-B is re-derived
- [ ] Re-derive every V1-B conclusion from a post-`385e06c` run — `KAGGLE` (needs the retrain)
- [ ] Push the repo to a remote — user-gated, no remote exists (SPEC §5 P0: largest unmanaged risk)

## P1 — Data
- [x] Pull ~5k upstream rows/source from pinned HF revisions (`9e954f9`)
- [x] Re-express through the suite's own converters; parity test vs `kev.suite.select_unique`
- [x] Exact-state dedup on both keys, printing collision counts
- [x] Pilot corpus ≥55k states (57,904 states / 4,246,106 tokens measured)
- [x] **15a** `paraphrase.py`: 9 hand-written phrasings for each of the **17** stable schema cores
      (agnews 5, contrastive 4, yelp 2, seven sources 1); the suite's exact wording is in no pool —
      84,936 distinct training wordings over the pilot, **0 leaking into dev/test/train strings**
- [x] **15b** boolq/mnli: frame-only variation (9 frames each) with the per-row question or
      hypothesis **verbatim**, so no phrasing can move a gold; `mnli_parts` raises `UnknownSchema`
      on any other shape rather than guessing
- [x] **15c** Index 8 of 9 reserved: `TRAIN_INDEXES = range(8)`, `EVAL_INDEX = 8`, disjointness
      pinned by test, and `draw_index` witnessed over 4,000 draws never returning it
- [x] **15d** Wired into the trainer (`--paraphrase off|on`): both batch paths draw, variants are
      pre-built and stable so `question_tokens` identity-caching holds, and `worst_case_tokens`
      prices every set at its **longest** phrasing so `--max-q-cells` stays an upper bound.
      Witness: `paraphrase: 19 question sets re-worded …` + `paraphrase: N phrasing draws reached
      the batches over S steps` + `paraphrase_draws` in `metrics.json` — the warm-up banner alone
      would still print if the loop never consulted the table
- [x] **15e** Held-out gate: dev/test/calibration keep the exact suite strings, and `dev_unseen`
      (metrics.json + `=== dev, HELD-OUT phrasing ===`) scores the reserved phrasing via
      `held_out_eval`, whose single `EVAL_INDEX` call site is pinned by test. `--long-context`
      refuses `--paraphrase on` rather than claiming a gate it cannot run
- [x] **15f** Pilot coverage test: 17,112 sets / 19,596 question shapes / 57,904 rows /
      **73,804 labelled slots** paraphrased, 10,629 distinct instruction strings on disk
      (boolq 5,300 + mnli 5,300 of them per-row content), zero train↔eval wording overlap
- [x] **15g** `bench/mutation_paraphrase.py`: **29/29 mutations caught**, both batch paths, the
      budget, the index, the flag and the guard. Its first pass reported 0/29 because
      `pyproject`'s `pythonpath = ["src"]` outranks `PYTHONPATH`, i.e. it was testing the
      unmutated tree — hence the green-baseline and first-mutation guards (SPEC §9.12).
      Two real holes it then found and closed: boolq rows whose question ends with a suite
      suffix were being edited, and content frames whose nine variants all ended the same way

## P2 — Per-row question tensors
- [x] `[B,N,·]` batch contract with masks; shared form still accepted (`57befce`)
- [x] Equivalence tests, three directions; 13/13 mutations caught
- [x] Per-axis memory measured (`bench/mem_profile.py`): 1.6 MiB/question-token position,
      0.8 MiB/state-token position, 3 KiB/option cell
- [x] `--max-q-cells` budget + rows-per-forward table (`bench/pilot_topology.py`)

## P3 — Train and validate — **on Kaggle**
- [ ] **3a** `train.py` runs on a CUDA device without edits: verify the path locally at CPU scale
      with the same flags (`--row-batch`, `--max-q-cells`, `--grad-accum`), then `KAGGLE`
- [ ] **3b** `kaggle/` bundle: `requirements.txt` (torch cu121, python 3.11-compatible), dataset
      packaging of `data/decision-v2-pilot/`, one-command entrypoint, checkpoint to
      `/kaggle/working`, resumable, `EXPERIMENT_NAME`/seed pinned
- [ ] **3c** Python-3.11 compatibility: `requires-python >=3.13` conflicts with the Kaggle image;
      prove the package imports and trains on 3.11 (or pin a 3.13 image) — no silent fallback
- [ ] **3d** Memory-safe sizing as a *computed* startup line, not a guess: from free device bytes
      ÷ 0.8 MiB × p95 state tokens, print the chosen batch and refuse to exceed it
- [ ] **3e** Stop rule in the loop: hard-stop + save checkpoint if observed s/update rises above
      1.5× the median of the last 20, or free bytes fall below the projected need
      (SPEC §6 lesson: a projected rate is not a budget)
- [ ] **3f** Checkpoint cadence ≤ 50 updates, and a resume test that proves the cadence works
- [ ] **3g** Stratified reporting harness: per source × question-type table, groupable (9/11) vs
      per-row-instruction (boolq, mnli), majority/uniform floors in the same table — runnable on
      the v0 + any Kaggle checkpoint (inference here is allowed)
- [ ] **3h** 15.35M vs ~32M head-to-head on identical data — `KAGGLE`
- [ ] **3i** G1 verdict: decision-v2 **test ≥ 0.70** macro, ≥ +0.15 over the 0.4331 majority floor,
      per-source table published — `KAGGLE` result, reported here as measured or as a loss

## P4 — Latency reconciliation (G2)
- [ ] **4a** One box, one process, direct laya `Agent` call (not `Router`), matched window,
      question count, dtype, options/question — MacBook, inference only
- [ ] **4b** Decompose fixed per-call vs per-token vs per-question cost for myna
- [ ] **4c** Publish the ratio recomputed from 4a, or withdraw it (SPEC §9.1 discipline)

## P5 — Abstention and risk/coverage (G5, flagship)
- [ ] **5a** Threshold + abstain reason emitted from the engine (no silent fast path)
- [ ] **5b** Risk/coverage curve script over `calibration.jsonl` (448 rows / 568 questions)
- [ ] **5c** Fallback seam to a stronger model with the chosen model labelled per decision
- [ ] **5d** G5 pass condition measured on banking77 + dbpedia14 + trec — needs a real-trained
      checkpoint, so `KAGGLE`-gated; harness must be provably correct against a synthetic oracle
      first (mutation-checked)

## P6 — Browser/on-device deployment (G3)
- [ ] **6a** ONNX export of trunk+pointer head with torch-vs-ONNX parity ≤ 1e-4
- [ ] **6b** In-Chrome measurement via onnxruntime-web: download bytes fp32/int8, cold-load ms,
      p50 per decision on a real page (screenshot-verified at desktop + mobile widths)
- [ ] **6c** MLX int8 path re-measured after any parameter growth (MacBook-only)

## P7 — Long-context proof (G4)
- [ ] **7a** Needle recall-vs-length curve at 1k/4k/8k/16k on an existing checkpoint (inference)
- [ ] **7b** 4k truncated-backprop checkpoint — `KAGGLE` (code already built and tested)

## P8 — Write it up
- [ ] **8a** README headline table built only from committed artifacts, every cell labelled
      measured / projected / gated
- [ ] **8b** One reproduction command per table row, and a `make` / script target that runs them
- [ ] **8c** Disclose the upstream-data comparison prominently (myna-scratch, myna-trained, laya
      as three rows)
- [ ] **8d** Release gate: G1–G7 each pass or each explicitly marked not met

## Cross-cutting
- [ ] Every new gate mutation-checked, witness quoted in the commit message (SPEC §7.1)
- [ ] `pytest` green at every tick; 0 skips other than the KEV-gated parity test
- [ ] Corrections logged in SPEC §9 the moment a claim dies

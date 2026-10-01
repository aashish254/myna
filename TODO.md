# TODO — executable mirror of [SPEC.md](SPEC.md)

Derived from SPEC §5 (plan phases P0–P8), §2.2 (acceptance gates G1–G7) and §7 (discipline).
Checked items are done **and witnessed** — an item is only ticked when a command prints the thing
the item claims, per SPEC §7.1.

**Training policy for this loop:** the 2026-09-26 directive ("nothing trains on the MacBook; all
model training runs on **Kaggle**") was **lifted by the account holder on 2026-10-01** because the
Kaggle token no longer authenticates (SPEC §5 P0 `KAGGLE`, §9.50). The 10c pair therefore runs here
on the M5. Everything else about the old directive still holds: the on-device/MLX/MPS measurements
*must* stay on this box (SPEC §6), and only one training run at a time.
So every "train" item below is split into *prepare/verify the path locally* (this loop) and
*run it on Kaggle* (user's credentials — labelled `KAGGLE`, blocked until the token is re-minted).

## P0 — Repo and measurement hygiene
- [x] Fix duplicate-row batching in `draw_batch()` (`385e06c`), mutation-checked
- [x] Same fix in `rlcd.py`
- [x] Startup diagnostic: median pool size, share of sets with pool < batch
- [x] Fit temperature on `calibration.jsonl`, not dev (`59240e1`)
- [x] `--device auto` considers CUDA (`59240e1`)
- [x] Correct SPEC §6's stale budget line ("budget by option count") against §9.9's measurement
- [x] `--help` renders for every CLI (`myna.train`, `myna.rlcd`, `myna.serve`) — argparse `%`
      crash and the uvicorn-before-parse_args crash, both pinned by `tests/test_cli_help.py`
- [x] `bench/diag_learn.py` / `diag_overfit.py` converted to argparse (`--steps --lr --pool`, `--n`);
      they took positional `sys.argv`, so `--help` died in `int('--help')` — witnessed against
      `git show HEAD:bench/diag_overfit.py`, which still raises the `ValueError`. Now pinned by
      `tests/test_cli_help.py` alongside `myna.report --help`. Their V1-B *numbers* remain void and
      are re-derived post-`385e06c` (the KAGGLE item below), which is a different debt.
- [ ] Re-derive every V1-B conclusion from a post-`385e06c` run — `KAGGLE` (needs the retrain)
- [x] Push the repo to a remote — done 2026-09-30: public at `github.com/aashish254/myna`, v0 weights
      shipped as a Release asset not a git blob, and every blob in history scanned for credential
      shapes. The scan is `make secrets` now (`bench/scan_secrets.py`), because this line's earlier
      "845 blobs, 0 matches" turned out not to be reproducible by any command — a re-run of the same
      one-off recipe walked zero blobs and still printed green. SPEC §5 P0 carries the full correction.
- [x] `bench/scan_secrets.py` + **`make secrets`** — the credential scan stops being prose about a
      command. Walks `git cat-file --batch-all-objects` (not the refs), nine shapes, prints
      `blobs scanned` against `blobs in the db` and exits 1 on a mismatch, and splits a finding into
      **reachable-from-a-ref** (fails — that is the set a push transfers) and **unreachable** (printed
      as local debt, not red — otherwise a clean tree can never go green). The published claim is
      **no hit reachable from a ref, for 9 shapes**; the blob and byte counts are the command's output
      and not copied here, because committing this tool added blobs of its own and a number written in
      prose would have been stale before its own commit landed. The local-debt hits are the tool's
      superseded drafts — its first URL pattern, which matched its own source line, and its first test
      fixture, which hard-coded the fakes it tests. The loosest URL shape hit three
      `docs/screenshots/*.png` before it was narrowed to printable ASCII — read, PNG magic confirmed,
      fixed in the pattern rather than parked in an allowlist. Falsified by `tests/test_scan_secrets.py`
      (3 arms: clean repo prints its denominator, planted fakes are named by shape and path and exit 1,
      an `--amend`ed blob no ref points at is still found but classified local debt and exits 0).
      Suite at this tick: **556 passed, 1 skipped**.
- [x] **Make CI green** — `.github/workflows/ci.yml` failed all **7** runs this repo has ever
      recorded, at **29–48 s each**, because `tests/test_mlx_int8.py` did a bare module-scope
      `import mlx.core as mx`: `Interrupted: 1 error during collection`, exit 2, **zero tests
      executed** while this box printed 556 green beside them. The obvious fix — `importorskip` plus
      `uv sync --extra mlx` — was tried on a branch and the runner falsified half of it: MLX's Linux
      wheel installs the bindings without `libmlx.so`, so `import mlx.core` raises a *bare*
      `ImportError`, which `importorskip` re-raises rather than skipping. What landed instead:
      (a) `try / except ImportError` + `pytest.skip(allow_module_level=True)` in `test_mlx_int8.py`
      (8) and `test_mlx_parity.py` (3), measured both ways — `mlx` hidden, and `mlx` present but
      unloadable — each printing `MLX unavailable on this box: …`; (b) `bench/reproduce.py`
      `mlx_unloadable_here(tail, platform)`, which turns the four MLX rows' `--help` assertion into a
      printed note **only** off Darwin and **only** for those two error strings, so a missing base
      dependency or a dropped flag stays red (§9.43 intact); pinned by 5 arms in
      `tests/test_reproduce.py`, mutation-checked 4 mutants / 4 caught; (c) CI stays on plain
      `uv sync`, and README's Quickstart says what that leaves unread and why `--extra mlx` is a
      macOS-only instruction. The probe was wrong twice before it worked (`find_module`, gone in
      3.12, blocked nothing and reported 11 passed; then a bare `ImportError`, which
      `importorskip` re-raises) — SPEC §9.51.
      **Round 2 (§9.52):** PR #1 run 36779801999 executed the suite for the first time — **519.31 s,
      13 failed / 520 passed / 20 skipped** — so the collection defect is closed and the 13 are the
      box assumptions it had been hiding. Four causes, each closed with a real fix rather than a
      skip: onnx absent made the `int8-quantize` row's `--help` unaskable and took gate G7 with it
      (5 tests → CI installs `--extra browser`, and README's test-install line changed, since the
      documented `uv sync` + `uv run pytest` was the recipe that produced the red); five tests load
      `runs/myna-v0/model.pt`, which a clone does not have because the weights are a Release asset (→
      CI runs `gh release download v0-checkpoint`, which also makes §5 P0's release claim executable);
      `bench/bench_mlx.py` and `bench/diag_mlx_int8_gem.py` have no usage text to render off macOS (2
      → `macos_only` on exactly those two params, after a CPU-only rehearsal that prints the state it
      simulated and aborts if the patch did not take: **52 passed, 2 skipped**); and
      `test_explicit_device_never_rewritten` asserted `resolve_device("mps") == "mps"`, which is this
      laptop's hardware in a policy's clothes (1 → availability monkeypatched for both accelerators).
      `--extra browser`'s loadability on Linux stays labelled **projected** until the next run
      measures it; if it goes red on import, `int8-quantize` joins the platform-note path and the 13
      read as 8. **Closed 2026-10-01.** The projection resolved in the expected direction and PR #1
      landed on `main` by rebase; run **36785555798** at SHA `3ba18a3` printed **545 passed, 8
      skipped, 0 failed in 599.72 s**, exit 0, with the credential scan at 0 matches over its full
      222-blob clone. CI now runs `pytest -q -rs`, and the log names all 8 skips — 2 collapsed MLX
      files, 2 `macos_only` CLI harnesses, 3 `test_report.py` artifacts a clone cannot have, 1 KEV
      gate — which is the accounting §9.52 quotes rather than infers.

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
- [x] **3a** `train.py` runs on a CUDA device without edits: `--device auto` considers CUDA, an
      explicit unavailable name fails with the list of devices the machine has (never a silent
      downgrade), and the headroom read goes through `torch.cuda.mem_get_info()` — pinned by a
      monkeypatched-CUDA test that asserts *free* not total, and that a driver `RuntimeError`
      degrades to "no headroom reading" with `--batch` taken as given. The exact box flag set
      (`--row-batch --max-q-cells 2048 --accum-groups 2 --group-sample uniform --paraphrase on
      --free-gib 10 --mem-safety 0.5 --save-every 1 --stop-factor 1.5`) trains on CPU here.
      The GPU half is `KAGGLE`.
- [x] **3b** `kaggle/` bundle: `requirements.txt` (torch cu121-compatible pins, no `+cpu`/`cu121`
      wheel suffix that would fight the image), `run.py` one-command entrypoint, `package_dataset.py`
      for `data/decision-v2-pilot/`. Checkpoints go under `/kaggle/working/runs/$EXPERIMENT_NAME`
      (the constant is pinned by test, not just the monkeypatched default); no name → refusal;
      half-mounted corpus → refusal naming the missing split, never a fallback to synthetic data;
      `--resume` added exactly when `model_last.pt` exists; the trainer's non-zero stop exit reaches
      the notebook. Seed and vocab are pinned in `DEFAULTS`. The packager hashes every staged file
      against the corpus, writes the manifest **from the corpus**, and prints
      `kaggle datasets create` instead of running it.
      Witness: `tests/test_kaggle_bundle.py` → **16 passed**, including a real 2-update CPU training
      through the entrypoint (`run.json` + tee'd `train.log` + `model_last.pt` carrying optimizer
      and scheduler) and a deliberately lossy copy that fails the build.
      Gate: `bench/mutation_kaggle_bundle.py` → **25/25 mutations caught** (first pass: 21/25, and
      all four survivors were holes in the tests, not the code — SPEC §9.15).
- [x] **3c** Python 3.11: `requires-python` is `>=3.11` (was `>=3.13`, which made the Kaggle image a
      silent fallback), every `src/ tests/ bench/ kaggle/` file parses under
      `ast.parse(..., feature_version=(3, 11))`, and `bench/check_python311.py` runs the package on a
      real 3.11 interpreter — witness **`PASS: the package imports, compiles and trains on python
      3.11`** (3.11.15, torch 2.6.0), 2 CPU updates included. It exits 2 rather than skipping when no
      3.11 exists, because a compatibility check that silently skipped is worse than no check.
- [x] **3d** Memory plan computed at startup, not guessed: p95 state tokens measured over a
      4,000-row sample of the rows the loop draws (p95 because the batch pads to its longest row),
      `safety × free − (--max-q-cells × 1.6 MiB question branch) ÷ (p95 × 0.8 MiB)`, both prices from
      `bench/mem_profile.py`. Prints one of: fits / clamped-to-N with the reserve named / **refusal**
      when the question branch alone eats the budget ("lower `--max-q-cells`"). No headroom reading
      is an explicit line, never an invented number — SPEC §9.14 is the correction that added the
      reserve.
- [x] **3e** Stop rule: step time > 1.5× the median of the previous 20 updates, or free bytes <
      1.25× the projected per-step need → save `model_last.pt`, print `STOP at step N: <reason>`,
      and **exit non-zero**. Witnessed both ways: `--stop-factor 1e-4` over 30 steps makes the 21st
      update a violation by construction, and the default factor lets a normal run finish; the window
      is trailing and needs 21 samples (five samples cannot condemn a sixth).
- [x] **3f** Cadence ≤ 50 updates and a resume that proves it: `--save-every 25` default, snapshot
      carries weights + config + temperature + step + **optimizer + scheduler**. `--resume` starts at
      stored step + 1 (never replays the snapshotted update) and restores the cosine decay — pinned by
      a round-trip at step 17 (Adam moments, `sched.state_dict()` equality) and by a 6-update run
      that resumes at 4 of 6. A missing snapshot is a named refusal, not a `torch.load` traceback.
- [x] **3a–3f gate**: `bench/mutation_memory_plan.py` — **44 mutations over `src/myna/train.py`**, run
      inside a scratch copy of the repo (the `pythonpath = ["src"]` trap, §9.12), with green-baseline
      and first-mutation abort guards.
- [x] **3g** Stratified reporting harness: per source × question-type table, groupable (9/11) vs
      per-row-instruction (boolq, mnli), majority/uniform floors in the same table — runnable on
      the v0 + any Kaggle checkpoint (inference here is allowed). `src/myna/report.py` + `tests/test_report.py`
      (**29 tests**, three added by 8c) + `bench/mutation_report.py` (**67 mutations, all caught,
      exit 0** — `runs/mutation_report.log`; the 54→67 step is 8c's competitor line and the §9.32
      merge). Witnesses printed
      by the harness on the frozen pilot test split: 1440 rows over 16 cells, 16 kept at n≥30;
      **majority macro 0.433** (re-derives SPEC §4.2's published 0.4331 through the adapter) and
      **uniform macro 0.332** (row-weighted; §9.16 corrects the 0.3292 published figure and lists the
      three alternative weightings it could have been); laya joined per cell (0.673 macro over the
      shared-instruction stratum, 0.667 over all sixteen — §9.32 corrects the 0.682 this tick first
      printed, which came from a two-row cell being scored by its last row) reproducing its +0.0125 imdb / +0.0375 boolq margins; the stratum
      class measured from instruction strings — agnews 8 distinct instructions over 300 slots inside
      113 exact-signature sets, boolq 80/80, mnli 80/116 (§9.17); MACRO line model 0.338 vs floor 0.433
      → **G1 not met** on the void v1-rich checkpoint, which is the point of running it here.
- [x] **3g process rule**: a mutation battery runs against a frozen repo. Two earlier passes of the same
      battery reported 42/53 and 53/54 because `src` and `tests` were being edited mid-flight and the
      harness re-copies them per mutation. SPEC §9.18.
- [ ] **3h** Default-config model (15.35M at vocab 4096; v0's artifact is 14.45M — SPEC §2.1) vs the
      ≤ 32M allowance, head-to-head on identical data — `KAGGLE`
- [ ] **3i** G1 verdict: decision-v2 **test ≥ 0.70** macro, ≥ +0.15 over the 0.4331 majority floor,
      per-source table published — `KAGGLE` result, reported here as measured or as a loss

## P4 — Latency reconciliation (G2)
- [x] **4a** One box, one process, direct laya `Agent` call (not `Router`), matched window,
      question count, dtype, options/question — MacBook, inference only
      Witness: `bench/bench_latency_matched.py` → `runs/latency_matched.{json,md}` (+ run 1 kept as
      `_run1_superseded`). Both dtypes reported (`torch.float32` each, laya autocast off), same `str`
      and same `dict` objects to both engines, `Agent.system_one` on `55cf4c4`, ladder capped inside
      laya's 1024 window and `usage.input_tokens/questions` derived per row — no row truncated.
      `flock` on the output path refuses a second copy. 26 tests, 34/34 mutations caught.
- [x] **4b** Decompose fixed per-call vs per-token vs per-question cost for myna
      Witness: least squares on the ladder — ask 10.1 ms fixed + 9.6 ms/question with a state slope
      of −10.3 µs/token (≈ 0: the fixed-size state, measured); observe 360 µs/token with a *negative*
      intercept because the scan is quadratic inside a chunk. laya's additive fit R² 0.74 is the
      structural finding: state × questions.
- [x] **4c** Publish the ratio recomputed from 4a, or withdraw it (SPEC §9.1 discipline)
      Witness: **3.04× / 3.21× / 4.12× / 3.60×** end-to-end and **6.78× / 4.67× / 4.75× / 3.65×**
      streaming at 1/5/10/50 questions — each the *minimum* of the two committed runs (§9.23).
      Withdrawn: the ≥ 5× @ 50-questions target (§9.21) and the README's 33×–239× column (§9.20).

## P5 — Abstention and risk/coverage (G5, flagship)
- [x] **5a** Threshold + abstain reason emitted from the engine (no silent fast path)
      — `Myna(ckpt, device, abstain_below=t)`, default `None` so the latency and parity paths
      still answer everything. Under a floor the committed field is `None` and the answer
      carries `confidence`, runner-up `margin`, `abstain` and a `reason` quoting the measured
      numbers; `ask()` echoes `policy.{abstain_below,abstained}`; `myna.serve`
      `--abstain-below` and `/v1/health` carry it across HTTP. noul's confidence is the
      committed side `max(p,1-p)`, not `p(Yes)` — pinned by sharpening the real head
      (`temperature=1e-3`) rather than a dictionary. Witness: `tests/test_abstain.py` (19) +
      4 abstention tests in `tests/test_serve.py`
- [x] **5b** Risk/coverage curve script over `calibration.jsonl` (448 rows / 568 questions)
      — `bench/risk_coverage.py`, README table and `runs/risk_coverage.{md,json,log}`. Ranks on
      the committed-side probability *recomputed* from each printed distribution and
      cross-checked against the engine's own `confidence` field; the selected floor is then
      re-run on the engine and must abstain on exactly the rows the curve withheld
      (**227 = 227** at floor 0.693). v0: accuracy 0.349 at full coverage → 0.596 at 10%,
      no rung reaching 0.95, so the table is labelled a harness witness and not a pass (§9.24).
      Witness: `tests/test_risk_coverage.py` (21 synthetic-oracle tests, including G5's two
      boundaries landed on exactly)
- [x] **5c** Fallback seam to a stronger model with the chosen model labelled per decision
      — `src/myna/fallback.py`: `Decider` seam, `DECIDERS` registry (`myna`, `laya`),
      `Fallback.predict` re-asks the secondary **only** the abstained questions, labels every
      answer with the engine that committed, and reports `routing.still_abstained` when both
      refuse. `LayaDecider` goes through `Agent.system_one` with `laya_spec()` translating the
      schema (noul without `criteria`, no gold label crossing) — the same translation
      `bench/eval_laya_real.py` uses, and a test holds the two from drifting. Witness:
      `tests/test_fallback.py` (15)
- [x] **5d-gate** Harness provably correct against a synthetic oracle, mutation-checked second
      — `bench/mutation_p5.py`: **49/49 caught**, exit 0, on a frozen repo
      (`runs/mutation_p5.log`). The lies cluster three ways: a fast path going silent (the
      answer keeps its label, the router never routes, the policy echo reports `None`), a
      quantity read off the wrong side (noul confidence as `p(Yes)`, a routed answer labelled
      with the fast engine), and a boundary flipping quietly (`<` vs `<=` at a floor that is
      always read off a row sitting exactly on it).
- [ ] **5d** G5 pass condition measured on banking77 + dbpedia14 + trec — needs a real-trained
      checkpoint, so `KAGGLE`-gated. v0's own numbers are `NOT MET` and are published as such:
      the three gate sources sit at their guessing floors (0.000 / 0.025 / 0.150 at full
      coverage), which abstention cannot raise

## P6 — Browser/on-device deployment (G3)
- [x] **6a** ONNX export, parity measured — `src/myna/onnx_export.py`: `state_step.onnx`
      (256-token chunk + state stack → next state, so length is more calls and never a bigger
      graph) and `question.onnx` (branch off that state), pointer head left in JavaScript over the
      spans. Measured on v0: chained 853-token scan with a padded tail **3.13e-06 relative**
      (1.40e-03 absolute on a state of scale 449), branch **1.88e-05 relative**, masked-row
      inertness **exactly 0.0**, probabilities **1.86e-07**. Gate split by unit because 1e-4
      absolute on a ~4.5e2 state asks two BLAS implementations to agree bitwise (§9.26).
      Three forced findings: `dynamo=True` (the legacy exporter writes graphs onnxruntime refuses
      to load), `opset_actual` read back from the file (17 was a label; torch wrote 18), and
      **93.7 MiB fp32 = the trunk twice**, so ≤ 20 MB int8 needs shared weights, not quantisation.
      Witness: `runs/onnx_parity.json`, `runs/onnx_export.log`, `tests/test_onnx_export.py` (8),
      `bench/mutation_onnx.py` (**12/12**; its first pass caught the byte-budget lie that two
      rounds of size assertions had let through)
- [x] **6b** In-Chrome measurement via onnxruntime-web: download bytes fp32/int8, cold-load ms,
      p50 per decision on a real page (screenshot-verified at desktop + mobile widths). Both things
      6a proved necessary are closed: the trunk is **shared** (`share_weights`, dedup by content
      hash → one 54.32 MiB `weights.bin`, artifact **93.7 → 59.32 MiB**, requested exactly once in
      all four Chrome rows), and the widest real request — `banking77/intent`, **1,067 tokens / 77
      options** at graph width 1152 — reaches parity at **4.25e-07** on option probabilities
      (`runs/onnx_parity_widest.json`, `"measured": true`). The page re-implements the tokenizer
      (`browser/bpe.js`, from `tokenizer.json` alone) and the pointer head + abstention gate
      (`browser/head.js`, from `head.bin`), and `browser/parity.mjs` is the one witness the node
      gate (7 checks) and the tab (6) both import. Chrome 153, cold cache, `bench/browser_g3.mjs`:
      wire **73.09 MiB** with the **13.58 MiB** wasm runtime counted apart from the model, cold load
      254–394 ms, p50 **436/426 ms** on 10 threads vs **742/743 ms** single-thread, load average
      recorded beside every millisecond. int8: **18.56 MiB** artifact — inside ≤ 20 MB — and
      **6.48e-02** probability error against a 1e-4 bound, so bytes yes, agreement no (§9.28).
      Witness: `runs/browser_g3{,_int8}.{json,log}`, `docs/screenshots/`, `runs/int8_quantize.log`,
      `node browser/selftest.mjs` 7/7, `bench/mutation_browser.py` **24/24** (`runs/mutation_browser.log`;
      its first pass caught three survivors, two of them the instrument's own blind spots — §9.27)
- [x] **6c** MLX int8 path for Apple silicon, measured on this box (inference only) —
      `MynaMLX.quantized_()` / `save()` / `from_mlx_dir()` + `_linear` dispatch in
      `src/myna/mlx_model.py`. **Bytes: met** — `params.safetensors` 55.13 → **17.40 MiB**
      (**3.17×**, 50/51 weights at group 64 / bits 8, `trunk.tok.weight` left fp32 because it is
      gathered), reloaded bit-identical. **Latency: not moved** — MLX over torch-MPS, per-row
      minimum of two committed runs, fp32 **1.76–2.22×** and int8 **1.79–2.41×**, and the int8−fp32
      sign flips between runs on 4 of 7 state lengths while fp32's own spread reaches 15.2%, so no
      speed claim is made. **Agreement cost**: port 1.20e-03, quantisation 9.24e-03 (same device),
      1 flip in 20 choices on a 0.0008 reference margin. **Mechanism, measured**:
      `bench/diag_mlx_int8_gem.py` → 5/6 of the model's own linear shapes are *slower* quantised
      (0.43–1.06×) at 10–37 GB/s of weight read vs fp32's 55–257 — the linears are 99.4% of the
      per-state-token MACs and weight-only int8 changes none of them (§9.29).
      **Null result**: keeping `.gate.weight` in fp32 moves the drift 9.24e-03 → 9.53e-03, flips
      the same row, and costs the byte target (22.25 MiB).
      Witness: `runs/bench_mlx_int8.{md,json,log}` + `_run2` + `_keepgate`,
      `runs/mlx_int8_gem_probe.{md,log}`; gate `tests/test_mlx_int8.py` (8) and
      `bench/mutation_mlx.py` → **14/14 caught** (`runs/mutation_mlx.log`). fp32 remains the
      shipping artifact on both the browser and the Apple path; re-measure after parameter growth

## P7 — Long-context proof (G4)
- [x] **7a** Needle recall-vs-length curve on an existing checkpoint (inference) — swept
      128/1k/4k/8k/16k, 16 needles per rung, seed 0: **0.188 / 0.312 / 0.312 / 0.125 / 0.062**
      against a 0.167 uniform floor. G4 is **not judged from this run**: the shortest rung is at
      chance, so the longer rows are not decay, and `eval_lengths`' caller now prints
      `G4: NOT MEASURED here` rather than a table (`baseline_verdict`). §2.1's 16k row and the
      README's context column are restated as state/cost measurements only, with §9.25 recording how
      the one number covered two claims. Witness: `runs/needle_myna-v0.{md,json,log}`,
      `tests/test_longctx.py` (13), `bench/mutation_longctx.py` (**20/20**)
- [ ] **7b** 4k truncated-backprop checkpoint — `KAGGLE` (code already built and tested). Re-run 7a
      against it; G4's numbers are only meaningful once the 128-token rung clears the floor by the
      guard's margin

## P8 — Write it up
- [x] **8a** README headline table built only from committed artifacts, every cell labelled
      measured / projected / gated. README's architecture, latency, browser, MLX and v0-accuracy
      tables and PLAN's benchmark tables now carry per-cell ***m***/***p***/***g*** tags with the
      artifact each came from (`runs/bench_mlx_int8.json` `n_params` + `sizes.*.file_bytes`,
      `runs/browser_g3.json` `state_bytes_total: 589824`, `runs/latency_matched.md`,
      `runs/train-v0.log`, `runs/rlcd_v0.log`, `runs/risk_coverage.md`). Eight headline cells died
      in the process and SPEC §9.30 names each one with the arithmetic that killed it: "dev 0.968 /
      test 0.951 overall" (a 192-row mid-run prefix probe and a number no artifact prints — the
      macros of the nine committed rows are **0.9599 / 0.9523**), "1.7 points" of dev→test gap
      (**0.76**), "~3.7 h" (**15,693 s**, and "~3 h" elsewhere), "3,000 synthetic examples"
      (recorded nowhere; the printed rounding admits n = 1200/workflow as the smallest solution and
      that is tagged *p*), "ECE ≤ 0.04 everywhere" (test max **0.0415**, dev max **0.0438**),
      "15.35M measured" (the default config's count at vocab 4096; the artifact is **14,449,280** at
      vocab 1740), "dev accuracy up to 0.9657" (a different dev draw — `rlcd.py` regenerates it at
      600 rows/workflow, so only the before/after *pair* is a claim), and "14.7–17.4 ms" (the ladder
      minimum is **14.52**; 14.7 was the longest row, not the shortest). §9.11's quoted "dev ECE
      ≤ 0.042" was found to be the *test* max wearing the dev label.
      Two instrumentation gaps fixed forward: `train.py` prints its own argv and one
      `split <name>: N rows over M groups, per-group lo-hi` line per split
      (`tests/test_train_logging.py`, 4 tests, **4/4 mutations caught**: drop the sum, drop the
      range, delete the empty guard, swap min/max), and `bench_latency_matched.py` writes
      `meta.cmd` + `meta.load_avg_after` and prints them in its header, checked against a live
      `os.getloadavg()` so a stubbed zero triple fails (the two committed P4 runs predate §9.23 and
      are disclosed as such in PLAN.md rather than quietly re-run, because a third run could only
      lower a published minimum). `Observation.save_state`'s "~0.6 MB" is 576 KiB of state and
      592,533 B on disk, measured. Suite: **312 passed, 1 skipped**
- [ ] **8b** One reproduction command per table row, and a `make` / script target that runs them
      Command side done: `bench/reproduce.py` held **22 rows** at this tick (it carries **27 rows /
      58 figures** since 9f–9i and 8e), each binding one published table
      to one canonical command, its committed witness(es), and the literal figures the prose
      quotes out of them — **38 figures tied to a committed artifact, 1 row gated with no witness
      on purpose** (`g1-v1`; a gated number with a file behind it is how a projection gets read
      as a measurement). Every harness that writes a witness now prints the command that wrote it
      (`$ …` as the markdown/json header), so no artifact's provenance is hand-typed again
      (§9.30 forward). Gate: `tests/test_reproduce.py` (**25**) and
      `bench/mutation_reproduce.py` → **21/21** (`runs/mutation_reproduce.log`) — the pass that
      caught this item's own doc check being vacuous, logged as §9.31. `make repro` /
      `make repro-run ROW=id` are the target. **Still open:** one target cannot *run* the whole
      registry — 14 of the 22 rows were `here` and re-runnable on this box (18 of the 27 they have
      become are), while 2 retrain a
      checkpoint, 2 need a laya checkout on `PYTHONPATH`, 3 need Google Chrome and 1 is `KAGGLE`
      — so "a target that runs them all" is a Kaggle-side item, not a
      `make` line, and ticking it here would claim a one-command rebuild this repo cannot do.
      Suite at this tick: **337 passed, 1 skipped**.
- [x] **8c** Disclose the upstream-data comparison prominently, as three rows. README's new
      second section — *On the upstream corpus: three rows, including the one nobody likes* —
      carries **laya 0.667** (cell macro over the 16 cells both sides score, row-weighted merge
      per §9.32), **myna-scratch 0.346 / 0.351, mean 0.348** (the control: myna's default
      architecture, `torch.manual_seed`, the suite's own `tokenizer-8192.json`, frozen `test`
      split, no gradient step on any corpus), **myna-trained gated** on the Kaggle checkpoint,
      and the two floors (majority **0.433**, uniform **0.332**) beside every model number, plus
      the void pre-`385e06c` checkpoint's **0.338** printed *below its own control* in the same
      table rather than in a footnote. The control is falsifiable in one direction only: if a
      head or the pointer probe were reading the answer, a random init would sit above the
      uniform floor. `bench/eval_scratch.py` writes the trainer's `metrics.json` shape through
      `--metrics-out`, so `python -m myna.report` reaches the control by the *same* code path as
      a checkpoint and the two witnesses must print the same figure — they do, to the third
      decimal, and that agreement is a test rather than a sentence.
      Instrument: `tests/test_scratch.py` (**11**, ~8 s) and `bench/mutation_scratch.py`
      (**13/13 caught, exit 0** — `runs/mutation_scratch.log`). Four of those thirteen initially
      survived, and the reason was the fixture rather than the tests: question-sets of 1, 2 and 4
      rows can only score multiples of 0.1, and seeds 0 and 1 produced *identical accuracy maps*
      (different weights, same argmaxes), so "which draw did this artifact come from" and "the
      stored figure is rounded" were facts the data could not express. The fixture is twenty rows
      with a 7-row set now, the two per-seed tests run three draws, and both conditions are
      asserted as guards so the battery cannot silently go green again if they lapse. Two bugs
      came out of writing the tests to re-derive instead of restate: the §9.32 two-row merge, and
      a competitor line that averaged myna over all scored cells while averaging laya over the
      intersection.
      Witnesses: `runs/scratch_decision_v2_test.{md,json,log}`, `runs/scratch_metrics_test.json`,
      `runs/report_scratch_vs_laya.{json,log}`, `runs/report_void_vs_laya.{json,log}`,
      `runs/mutation_scratch.log`, `runs/mutation_report.log`. Three registry rows added
      (`scratch-control`, `scratch-vs-laya`, `void-vs-laya`) → **22 rows, 38 figures**; the row
      for the void checkpoint says so in its note, because an artifact whose input is uncommitted
      cannot be quoted as evidence.
      Suite at this tick: **354 passed, 1 skipped**.
- [x] **8d** Release gate: G1–G7 each pass or each explicitly marked not met. The answer is
      **no, and here is which**: `bench/gates.py` carries one verdict per gate — **3 met**
      (G2 the matched latency ratios, G3 the in-Chrome mechanics, G7 the checker itself),
      **2 not met** (G4, G5), **2 open** (G1, G6) — and `make gates` prints
      *7/7 verdicts hold — 3 of 7 met, 2 not met, 2 open — the release gate is NOT clear.*
      The binding, not the tally, is the work. A gate cites (i) registry rows from
      `bench/reproduce.py`, **imported and re-run** so a verdict inherits its witness's drift
      verbatim instead of restating it, and (ii) proofs — a named key path read out of a
      committed JSON artifact and compared to the value the verdict rests on:
      `g5.pass=false` for G5, `baseline.readable=false` for G4, `inference_only=true` plus
      both `*_dtype=torch.float32` for G2, `pass=true`/`rows/0/report/passed=6` for G3, and
      the registry's own green exit for G7. Four rules then refuse the easy lies: a `met`
      verdict may not rest on a field that prints `false`, and needs one that prints `true`;
      `not met` needs a field that prints `false`, because a failure is a claim too; a
      verdict may not rest on a `void` artifact (§9.23) or on a row this box cannot re-run;
      and **`open` is only allowed for a gate that names the run it cannot do** — so "open"
      means a missing artifact, never a measurement nobody took. That last rule is what keeps
      G1 honest: the control's 0.346 sits under its threshold and G1 is still `open`, because
      the checkpoint that decides it has no witness in this repo (§9.30).
      Instrument: `tests/test_gates.py` (**26** tests) and `bench/mutation_gates.py`
      (**17/17 caught, exit 0** — `runs/mutation_gates.log`). This battery had to link `.git`
      and the four docs into its scratch copy, because the checker under test reads the git
      index and the prose rather than only its own argv. One entry survived its first pass
      and the fix was to the test: dropping the status filter from the "locally reproducible"
      rule was invisible while the discriminating citation was a `gated-kaggle` row, because
      the registry already forbids those quotes, so `r["quotes"]` failed it either way — the
      two halves of the condition denoted the same outcome. The test now cites a `retrain`
      row (committed witness, unreproducible run) *and* a `gated-kaggle` row.
      What the prose had to give up: §2.2's G5 cell led with "harness met", so the verdict
      word is now first ("not met on the level, met on the machinery"); G7 moves from a bare
      `open` to **met as a binding, not a rebuild**, with the 14-of-22 split printed beside
      it rather than in a footnote; and G1's one-word `open` becomes the sentence that says
      which artifact is missing. README's copy of the table is generated, and §9.31 is why it
      is labelled as carrying no evidence — `--check` reads §2.2 and the JSON files, never the
      generated block. TODO 8b and §5's release line stay unticked on purpose: 8b is the
      one-target-runs-everything gap (Kaggle-side), and the release itself still waits on P0's
      push, on G1/G4's checkpoints and on any trained artifact for G6.
      Suite at this tick: **383 passed, 1 skipped**.
- [x] **8e** `--check` asks the tool, not just the file: a row's command must still accept the
      flags it publishes. *(this tick — the falsifiable half of 8b, which stays open)*
      8b's remaining gap is that one `make` target cannot *run* all 27 rows, and that stays a
      Kaggle-side item. What was buildable here is the assertion behind it: three of `--check`'s
      four questions are answered by reading disk, so a row could publish a flag its tool had
      renamed and stay green — the artifact predates the rename, the docs still quote the line,
      `git ls-files` still returns it. `python_target()` now resolves `-m module` and `script.py`
      past a `uv run` launcher and a `VAR=value` prefix, `help_output()` asks that tool for
      `--help` once per process, and `command_problem()` requires every published `--flag` to be
      in the answer. Nothing else executes, which is §9.40 applied to the registry: a gate that
      ran a row would overwrite the witness it is checking. A hung `--help` costs `HELP_TIMEOUT`
      and becomes a report rather than a traceback, and that is a test (`HELP_TIMEOUT = 0`) not a
      sentence.
      Measured, and it is a negative: **20 of 27 rows are python commands, all 20 answer, every
      flag accepted**, `--check` from **0.72 s to 5.85 s** (timed in one process by stubbing
      `command_problem` to `None`), still **27/27 rows, 58 figures**. The 7
      unchecked rows say what they are — 3 `npm` (gated for real by `mutation_browser.py`), 2
      `grep` over a committed log, `pytest tests/test_report.py`, and the `gated-kaggle` prose row
      whose apostrophe makes `shlex` raise, which is one of the new mutants. G7's own sentence now
      names the fourth assertion, so its prose was re-rendered into README (generated, pinned by a
      test) and copied by hand into SPEC §2.2 — and because `bench/gates.py` changed,
      `bench/mutation_gates.py` was re-run rather than inherited: **17/17 in 240 s**
      (`runs/mutation_gates.log`).
      Instrument:
      `tests/test_reproduce.py` **33** (was 25) and `bench/mutation_reproduce.py` **29/29 caught,
      exit 0** in **270 s** (`runs/mutation_reproduce.log`, 8 new mutants, each caught by the test
      written for it). One stale literal died on the way: the battery's own docstring still
      described a vacuous checker as one where "`--check` still says **19/19**" — a registry size
      three ticks out of date — now 27/27 in place. Logged as **§9.43**.
      Suite at this tick: **444 passed, 1 skipped**.
- [x] **8f** The registry's counts are counted, and `make gates` reads them back out of SPEC
      §2.2 — the gate table's one hand-copied row. *(this tick — the seam 8e named and left open)*
      8e wrote §9.43 into G7's prose in two places at once: `bench/gates.py`'s sentence, pinned by
      a test against `ROWS`, and this repo's §2.2 row, copied by hand and pinned by nothing. The
      numbers that now live in that row — `27 registry rows`, `58 quoted figures`, and the split
      `18 / 3 / 2 / 3 / 1` — are exactly the kind 9f already caught once (`22 rows` written beside
      a registry of 23), so agreement here was a memory, not a check.
      `registry_counts()` derives all seven phrases from `ROWS`, `count_phrase()` says the
      five-status half once and G7's note interpolates it, and `check_gate` — for any gate whose
      proof *is* the registry, which today is G7 alone — reads §2.2's cell off disk and names every
      phrase it no longer prints. Falsified in both directions, measured rather than asserted:
      deleting `58 quoted figures` from the row reds naming that token, and growing `ROWS` by one
      row reds on `28 registry rows`, which is what the new test does instead of describing — a
      literal baked into `registry_counts()` would pass the first and fail the second. Committed
      tree stays green: `make repro` **27/27 rows, 58 figures**, `make gates` 7/7 at
      **3 met / 3 not met / 1 open**.
      Instrument: `tests/test_gates.py` **29** (was 28) and `bench/mutation_gates.py`
      **21/21 caught, exit 0** in **334 s** (`runs/mutation_gates.log`, 4 new — two of them the
      theatre shapes: the comparison that runs and never reports, and §2.2 read from the wrong
      column so a count it cannot find is a count that always matches; the first two are each caught
      by the test written for them). README got one sentence and no re-render: the generated release
      block is byte-identical, which is the point of interpolating. Suite at this tick:
      **445 passed, 1 skipped in 219 s**. Logged as **§9.44**.

## P9 — The three defects the first valid checkpoint named (SPEC §5 P9)
- [x] **9a** `score` cells are trained with a nominal cost while every other component treats the
      legend as ordered — `typed_loss` charges "predicted 4, gold 3" like "predicted 1, gold 5". Add
      the squared-Cramér/EMD² path over the option index, normalised by `K−1`, selected by
      `--score-loss {ce,emd}` with **`ce` the default** so no published figure moves. The verdict is
      the next Kaggle run's; the code, its gate and its mutation battery are this box's. SPEC §5 P9.
      Done: `typed_loss` takes `(ordinal, opt_valid, score_loss)` and prices marked cells with the
      Cramér sum over the cell's own `K−1` inside one mean over live cells; both batch builders emit
      the mask from `q.type == "score"`, the loop's single call site passes it, `--score-loss`
      defaults to `ce`, and `--long-context` refuses `emd` rather than silently training `ce`. Three
      properties are the gate: a one-rung miss and an opposite-end miss are byte-identical under
      `ce` and strictly ordered under `emd`; the worst price is 1.0 at every legend length; at
      `K = 2` the number *is* the Brier score of the second option.
      `tests/test_ordinal_loss.py` 18 tests, three running one CPU update through a `typed_loss`
      spy (the trainer seeds no torch, so a cross-process loss comparison could not fail);
      `bench/mutation_ordinal.py` **19/19 caught** (`runs/mutation_ordinal.log`). Two findings: the
      sum is multiplied by `opt_valid` because a masked tail is only harmless by coincidence, and
      the `[N] → [B,N]` expand is **dead code** — torch broadcasting makes the shared and per-row
      forms the same tensor, so that mutation survived its own battery (§9.33).
      Suite at this tick: **404 passed, 1 skipped**; registry `--check` 22/22 rows and 38 figures;
      gates 7/7 verdicts and **3 met / 2 not met / 2 open — NOT clear**, unchanged, because nothing
      published was trained with `emd`.
- [x] **9b** The readout fix is **closed without being built**: fitting a per-cell threshold on the
      suite's own `calibration` split makes the binary cells worse, and the argmaxes were not
      collapsing as claimed (the diagnostic's artifact is Kaggle-side, so its numbers stay out of
      prose — §9.24/§9.30). What survives is the reporting half: a Brier column beside every `noul`
      accuracy in `myna.report`, labelled as reporting, not as a fix. Nothing here raises G1's
      ceiling. SPEC §5 P9.
      Done (the reporting half only): `brier_cells()` reads `evaluate()`'s `:brier` sidecars and
      weights them by the rows in each question-set; `roll_up` puts the result in a `brier` column
      beside the accuracy, with the group's macro under it. Three ways it can lie, three tests: a
      sidecar-less metrics file prints an **em dash** and the report says the column is empty
      because the file has none, "which is missing data and not a model that scores 0"; a sidecar
      whose set is unknown to the split carries **no weight** instead of a zero; and the printed
      count is `priced of noul`, never `noul of noul`. No threshold and no per-cell bias went into
      the readout — the dead half stays dead and the box is ticked because the surviving half
      shipped. `bench/eval_scratch.py` still strips `:brier`/`:ece` on purpose, so the one
      committed artifact that could show the em-dash path does, at `"priced_cells": 0` of 8.
- [x] **9c** Report the cells that lose to their own majority floor as that: `max(acc, majority)`
      beside `acc`, the delta labelled as the clip's worth, not the model's. Which cells qualify is
      9c's own measurement on the committed pilot split, not a carried-over count. *Printing* the
      clip is reporting; *acting* on it is a per-cell fallback rule and that is the user's
      leaderboard call, so 9c builds the report and stays out of the decision. SPEC §5 P9.
      Done, and it stayed out of the decision: `roll_up` carries `clip` and
      `clip_worth = max(0, majority − acc)`, the legend above the table defines the column where it
      prints and calls it an oracle, and `main()` names each floor-loser with its worth before
      saying "+0.090 of **arithmetic and 0 of model** … a leaderboard decision, not a learning
      improvement". Measured on the committed control over the pilot split: **13 of its 16 cells**
      answer below their own majority floor; clip macro **0.4365** against model **0.3462** and
      floor **0.4331** — three cells (both `yelp` rows, `mnli/relation`) beat their floor and keep
      their own number, which is why a per-cell maximum is not the floor renamed. `--out` carries
      `macro.clip`, `macro.clip_worth`, `macro.brier_noul` and `below_majority_floor`. Nothing in
      `src/myna/` reads any of them: no fallback consults `clip_worth`, and acting on it is 9d's
      company, i.e. the user's call.
      Two things the build surfaced, both logged: the committed artifacts' macro row had been
      sitting **six columns right of its header** the whole time and no value assertion could see
      it (§9.34), and the new columns needed a delta field seven wide or two figures printed as
      one. `table_text` now emits every line through one `_trow`, and `LBL` is asserted against the
      split's own widest cell name.
      Suite at this tick: **415 passed, 1 skipped**; `tests/test_report.py` 29 → **40** tests;
      `bench/mutation_report.py` **93/93 caught** (67 → 93; `runs/mutation_report.log`); registry
      `--check` 22/22 rows and **41 figures**; gates 7/7 verdicts and **3 met / 2 not met / 2 open
      — NOT clear**, unchanged, because a clip is not a result and nothing published was retrained.
- [x] **9f** The launch lane got generated instead of hand-typed, and the first valid GPU run
      got a committed witness — so G1 is a verdict instead of a promise. *(this tick)*
      Two halves, both paid for by the same mistake: the notebook cells in
      `KAGGLE_LAUNCH_INSTRUCTIONS.md` were written by hand, and every one of their wrong paths
      (`/kaggle/input/decision-v2-pilot` instead of `/kaggle/input/datasets/<owner>/<slug>`,
      `pip install` against a `/kaggle/working` that starts empty, a 1.5 h guess for a run that
      measured 5 h 37 m) was found by burning or nearly burning a session. `kaggle/campaign.py`
      now emits the cells and, with `--nb`, a pushable kernel directory: it walks the mount with
      `os.walk(followlinks=True)` because datasets arrive as **symlinks** and `Path.glob("**")`
      does not descend into them, refuses to guess the owner, prices every cell from the measured
      5.62 s/update, keeps the four measured-dead ablation cells out of the default lane, and
      writes `is_private: true` with nothing pushed — and the reason its open lane now states a
      measured slope instead of a remembered one is §9.38. `tests/test_kaggle_bundle.py` is **29
      tests** and `bench/mutation_kaggle_bundle.py` carries **42 mutants, 42 caught, exit 0**
      (`runs/mutation_kaggle_bundle.log`): the two figures this line used to carry — 27 tests, 39/39
      — described a battery that had never run to completion, which §9.40 is about. `kaggle/run.py` gained
      `--free-gib` (the T4 memory plan reads free bytes *before* the model loads) and
      `--warm-start`, and it now **refuses** `--resume` onto a snapshot that already reached its
      requested step — SPEC §5 P9 9e measured what that costs: a spent cosine schedule trains at
      lr ~0 and prints numbers that read as drift.
      The second half is the witness. `runs/v1b_kaggle_3600b.{train.log,metrics.json,report.json}`
      are committed (285 KB / 295 KB / 9 KB, allowed by `!runs/*.log` and `!runs/*.json`; the log
      was scanned before staging and its only "token" matches are `tokenizer` and `tokens`), which
      is what §9.30 required before the run's figures could be quoted. Registry `--check` is
      **26/26 rows, 55 figures**, and `bench/gates.py` moves G1 from `open` to **not met,
      measured**: **myna 0.4893 · laya 0.6668 · floor 0.4331**, +0.056 against the +0.15 required,
      its own report printing `g1.pass: false`. The tally is now **3 met / 3 not met / 1 open**,
      and the counts in G7's prose are pinned by a test — they had drifted to `22 rows` while the
      registry held 23, which is the failure mode of writing a number in three docs.
      Found while re-running the batteries for those edits: `bench/mutation_gates.py` and
      `bench/mutation_reproduce.py` had both been **aborting at baseline** — their scratch copies
      never contained the `Makefile` a gate test reads, and one of them restated `reproduce.DOCS`
      instead of importing it, so widening the doc set red-lined it — which means their committed
      `baseline copy: green` logs witnessed a tree that no longer existed (§9.39). Both scratches
      now derive the top-level file set from the filesystem, and both logs regenerate green:
      **gates 17/17, reproduce 21/21, exit 0 each**.
      Two remembered claims died on the way and are logged as **§9.37** (one run printed two
      "test macro" numbers — 0.3983 from the diagnostic's roll-up, 0.4893 from the committed
      report's; the gate quotes the one whose command is re-runnable) and **§9.38** ("dev was flat
      from step 2,250" — the committed log's tail is +0.0072 per 1,000 updates and 0.4355 is the
      run's high, so dose is still not the answer but for a measured reason: the 10k pair projects
      ≈ +0.046 for ≈ 31.2 GPU h against a 30 h/week quota).
- [x] **9g** A mutation battery that had never finished, and the reason its `39/39` was in the
      entry above. *(this tick)*
      `bench/mutation_kaggle_bundle.py` has carried `the dry run runs the job` since its first
      commit, and its docstring explained that the probe is cheap because a test asks for one
      update. The *first* `--dry-run` test did not ask for one update: it called the entrypoint at
      the default dose, so breaking the short-circuit made the checker launch a real 10,000-update
      CPU run. Measured, not inferred: the battery stopped after `[13/40]` with `myna.train …
      --steps 10000 --batch 32 --vocab 8192` alive inside it at 11 minutes, killed by exact PID, and
      `pytest_in`'s `timeout=1800` had no handler — so the end state of that mutant is either an
      hour of the MacBook's fan or a traceback that orphans the trainer. `_run_py` now appends
      `MINI_DOSE` (`--steps 1 --batch 1 --vocab 128`) to any `--dry-run` that did not name a dose,
      and `pytest_in` gives pytest its own process group and `killpg`s it on timeout (600 s).
      Capping the print removed `--steps 10000`, `--batch 32` and `--vocab 8192` from the
      command-line assertion, so `test_the_default_dose_is_the_one_the_docs_price` reads
      `run.DEFAULTS` without launching anything, and two new mutants (`"steps": 10_000`→`500`,
      `"batch": 32`→`512`) exist because a check that replaced a check has to be paid for — both are
      caught by that test and nothing else. Battery: **42 mutants, 42 caught, exit 0**, witnessed in
      the repo for the first time as `runs/mutation_kaggle_bundle.log`; `the dry run runs the job`
      fails at `[16/42]` in seconds. The same entry fixed the open lane's `target`, which was still
      printing "dev was flat from step 2250" as the reason to spend 31 GPU hours — §9.38's own
      finding, one screen away. Suite: **436 passed, 1 skipped**, and that is not 19 tests of new
      work: 17 of them skip when their extra or artifact is absent (`onnxruntime` alone is 14, and
      the worktree's venv does not have it — verified by importing from it), so a suite count only
      compares with the tree it was measured in. `make gates` 7/7 with **3 met / 3 not met / 1
      open**; `make repro` **26/26 rows, 55 figures**. Logged as **§9.40**.
- [x] **9h** The last "investigate before concluding" item on `v1b-kaggle-3600b` is closed with
      nothing in it: dbpedia14's dev→test drop was two aggregations, not two measurements. *(this tick)*
      §9.30 says a published figure needs a row that can re-run it, and 9f gave the test half of this
      delta a row (`v1b-kaggle-macro`) while the dev half lived only in a session's arithmetic — so the
      dev split is now rendered into the repo too: registry row **`v1b-kaggle-dev`**, witnesses
      `runs/v1b_kaggle_3600b.dev.report.{json,log}`, three quoted figures (`0.5355719231874607`,
      `"sets": 37`, and the `dbpedia14/category choice 116 37 0.647` table line read with its columns).
      Registry: **27/27 rows, 58 figures**, `make gates` 7/7 at **3 met / 3 not met / 1 open**, and
      README's generated block re-rendered from `gates.py`. G7's own sentence counted the registry, so
      9f/9g's `26/26, 55` lines stay as what they measured at their tick.
      The item said dev 0.5324 falls to test 0.2047 on 37 question-sets a side, which is too large a
      swing to call overfitting without looking. Both literals still reproduce, exactly, as the
      unweighted mean over `evaluate()`'s per-set keys of the committed `metrics.json` — and that cell
      is **116 rows in 37 sets, 36 of which hold one row**, with **one** signature shared between the
      splits, so 36 of the 37 terms are coin flips over different samples. Suite-wide that roll-up is
      **414 of 690 dev terms (60.0%) and 417 of 716 test (58.2%) holding one row each**, and dropping
      those terms lifts the same mean to 0.6756 dev / 0.6092 test — a basis that moves the headline by
      0.240 and 0.211 is not a rounding choice. `myna.report`'s own basis —
      row-weighted inside each cell, because its docstring says a 1-row group must not outvote a 116-row
      one — gives dev **0.647** → test **0.457** against a **0.129** majority floor, and −0.190 is at the
      tail of a spread, not an anomaly: over the 16 cells both splits score, mean |Δ| is 0.079 (sd 0.089)
      with five other cells past 0.10 and imdb/positive +0.125 the other way. Overall macro dev 0.5356 →
      test 0.4893. Witness: `runs/v1b_kaggle_3600b.dev.report.log`, which is the committed metrics
      re-rendered with `--split dev` — no GPU, no re-run of a gated lane. **G1 does not move**: 0.4893
      against 0.70 stays *not met* on entry 37's basis. One trap documented on the way, because it looks
      like waste and is not: the 37 sets come from 11/10 distinct legend key-sets since `_signature`
      hashes the criteria dict in the row's own order, and `parse_questions` resolves each gold through
      `list(crit).index(lab)` — canonicalising the key without canonicalising the options would group
      rows whose legends disagree on order and mis-index their labels. The same re-check went to
      `runs/ordinal_ab.json`, the repo's other negative: its `meta.macro_*` are `macro_acc` (the per-set
      roll-up), and on its own per-cell basis — which is `report.accuracy_cells`, the report's function —
      "macro moves opposite ways in the two seeds" is dev −0.0013/+0.0053 and test +0.0071/+0.0005, so the
      clause is basis-dependent while the verdict is not, because the resolution ratios (dev 0.5194, test
      0.8765) are priced cells against the unpriced churn floor all the way down. Logged as **§9.41**.
- [x] **9i** §9.40's sweep over all 14 mutation batteries, and the honest finding: no second
      instance. *(this tick)*
      9g fixed the probe inside one battery; the thing that ran the batteries was still uncapped, and
      that is the same defect one level up. Measured the hard way: part 1 looped eight batteries
      (`paraphrase report p5 longctx onnx mlx ordinal scratch`) with no per-battery deadline, so a hang
      would have burned its own uncaught `timeout=1800` and then queued every battery behind it. It
      reached the fifth — `onnx` — before being stopped at PID 32557 and its `onnxruntime` children
      (exact PIDs, never a broad `pkill`), and those four were re-launched under a `SECONDS`/`kill -0`
      watchdog at `CAP=2400`/`1800`, which was first tested on a dummy `sleep 600` (rc=143 after 8 s,
      process confirmed gone) rather than trusted. All 14 then finished in this session, and every
      number below is read off a log a process was watched closing, not off the record: paraphrase
      **29/29** · report **93/93** · p5 **49/49** · longctx **20/20** · onnx **22/22** (1,200 s — two
      graph re-exports per mutant) · mlx **14/14** · ordinal **19/19** · scratch **13/13** · gates
      **17/17** · reproduce **21/21** · browser **24/24** · latency-matched **34/34** · memory plan
      **44/44** · kaggle bundle **42/42**. **441 mutants, 441 caught; zero `MISSED`, zero
      `BAD-PATTERN`, zero `TIMEOUT`**, summed straight out of `runs/mutation_*.log` (14 files, 441
      caught), eleven of those files written by this session's three sweep parts and each closing with
      an `EXIT=0 (watched …)` line that names its wrapper.
- [x] **9j** §7.1 applied to the harness 9d added: 24 mutations of `bench/scope_pricing.py`, whose
      first pass caught 10 of them. *(this tick)*
      The gate says every new thing that publishes a claim gets mutation-checked, and the scope table
      publishes one — 9d shipped it with `tests/test_scope_pricing.py` and no battery, while three of
      its sibling table-harnesses (`risk_coverage`, `eval_scratch`, `bench_latency_matched`) all have
      one. The 24 mutants split four ways: the guard's individual comparisons, the arithmetic, the
      witness's assembly, the printed table. **10/24 caught**, and the survivors were not scattered —
      every arithmetic mutant died (8/8, because `test_every_witness_row_is_rederived` recomputes
      `price()` from the report's own cells), and 13 lived: 5 of the guard (drop the `majority_floor`
      comparison, read `myna` off `published["uniform"]`, delete the `g1.pass` check, widen
      `TOLERANCE` to 0.05 — each still *refuses* a doctored `macro.acc`, which is the only case the
      11 tests tried), 5 of how the witness is assembled (`cmd`, `published`, a `scopes` dict missing
      its control row, the two cell lists), 4 of the printed table, 1 of the missing-`--report`
      boundary. The shape says it: `main()` had never been run successfully by anything — no test
      checked it exits 0, none read stdout, none compared its JSON to `runs/scope_pricing.json`. A
      harness printing three rows instead of four, or having lost the line naming its own command,
      kept a green suite.
      Seven tests closed them in the reported order: the guard per compared field plus its verdict
      case, the boundary error by its message, a `main([])` run whose regenerated JSON must match the
      committed witness on every key except its own `--out`, and a `capsys` test that each row line
      carries that row's own cells/floor/margin/verdict under the headers naming them — plus the
      invariant that the dropped-cell count and the listed below-floor cells are the same 5. Second
      pass **24/24, exit 0** in **17 s** (`runs/mutation_scope_pricing.log`, baseline green with 18
      tests); `--help` was proven cheap before it joined `test_cli_help.py` (+3 cases), because a
      battery that runs on `--help` is §9.40 walking back in. Suite **466 passed / 1 skipped in
      193.95 s**, `make repro` **28/28 rows, 61 figures**, `make gates` 7/7 at **3 met / 3 not met /
      1 open**. The same pass found §3.4's map saying **312 passing** where `pytest --collect-only`
      says 467 — 154 tests behind, in the one section written in present tense, and with no read-back
      over it the way §2.2's counts now have (§9.44's decay, unpoliced where nobody thought to police).
      Logged as **§9.46**.
      The static audit had already said the same thing: no battery but the kaggle bundle even names
      `myna.train`, `run.py` or `--steps`, so `MINI_DOSE` is the only place a probe could reach
      training. Two batteries handle a timeout — the bundle with its process-group `killpg`, and
      `mutation_browser.py`, whose `node` selftest gets 180 s and a `TimeoutExpired` handler that
      *counts a hang as caught* with the reason printed — while the other twelve carry
      `timeout=1800` uncaught, which is a slow death rather than an unaffordable probe. The group-kill
      refactor therefore stays in the one file where a probe was genuinely unaffordable. Suite after the rewrites: **436 passed, 1 skipped in 194 s** (the count
      line read from the run, because all three of the session's background waiters fired early on a
      still-live `ps`), `make gates` 7/7 at **3 met / 3 not met / 1 open**, `make repro` **27/27 rows,
      58 figures**. Logged as **§9.42**.
- [x] **10a** Tier 0's diagnosis turned into a price: the label prior *is* the best-scoring policy
      the rows support. *(this tick)*
      §9.47 found eight of sixteen cells answering from an association, three of them agnews `noul`
      cells emitting one label on every test row and finishing on their own majority floor —
      0.6531 / 0.8085 / 0.7778. The reason was readable off the train split without a model: those
      four cells carry natural majorities of **0.7390–0.7545**, so a constant answer scores ~0.75
      and nothing the rows support scores better. Flattening the marginal a mini-batch is drawn from
      drops the constant answer to ~0.5, which is the whole mechanism of `--anti-prior`.
      Logged as **§9.48** alongside the two claims this tick re-derived out of the artifacts (the
      "ten of sixteen" count and the "~31 GPU hours" price of the pair).
- [x] **10b** `--anti-prior on|off` in the row batcher, its reach measured on the shipped rows, and
      its gate at 41/41. *(this tick)*
      Each row is weighted by the product, over the questions it answers, of `1 / (share of its gold
      label inside its (source, question) cell)`, rescaled so each source's weights sum to its row
      count — the task mix cannot move. Only cells at or above `ANTI_PRIOR_SKEW = 0.55` are treated
      (six of sixteen). `bench/anti_prior_audit.py` runs the *real* `draw_row_batch` at the
      *published* dose (batch 10, 2,048 cells/forward, 8 sets, 3,600 updates, seed 0), both arms plus
      `--compare` over every `COMBINE` rule, and `runs/anti_prior_audit.log` is the witness: the six
      flatten to 0.6184 / 0.6278 / 0.6247 / 0.6172 / 0.5023 / 0.5095, the control arm's largest move
      is +0.0077, per-source weight sums match their row counts to **0.00e+00**, and the arms' drawn
      source shares differ by at most **0.13 pts** — the only part of the mix that belongs to the
      weights (banking77 sits 5.18 points under its row share in *both* arms: that is
      `--max-q-cells`, and it is why a single-arm mix table would have been a bug). `prod` is the
      minimum in all six columns at this dose, which is the measured reason it is the shipped rule;
      at 40 updates × 2 sets it is the minimum in **1 of 6**, so the ranking test reads the committed
      witness and the reduced-dose test asserts only table-vs-payload.
      The two limits are in the artifact, not just here: flattening the *train* marginal does not
      flatten the *test* marginal the floors are computed from, and row-level weighting cannot hold a
      cell fixed when its rows carry several labels — nine of the ten untargeted cells flatten anyway
      (contrastive/decision 0.507 → 0.337) and **yelp/rating sharpens** (0.203 → 0.259) because its
      weight is computed on `recommend`. Reach over Tier 0's eight is three.
      Suite **508 passed / 1 skipped in 275.52 s** (the count read off the run, and the +42 over
      §3.4's stale 466 measured by collecting both trees and diffing per-file totals: 35 new in
      `test_anti_prior.py`, 3 in `test_kaggle_bundle.py`, 4 in `test_cli_help.py`), `make repro`
      **29/29 rows, 69 figures**, `make gates` 7/7 at **3 met / 3 not met / 1 open** — the new
      registry row is `anti-prior-audit`, and §3.4's map now carries the harness and its battery.
      The three claims this tick had to retract had already shipped: **"ten of sixteen"** (§9.47's
      count is eight), **"~31 GPU hours"** for the pair (that is the 10k lane; this pair is 11.2),
      and the flag's help text promising that the ten untargeted cells "change nothing" — the audit
      measures nine moving and one sharpening.
      `bench/mutation_anti_prior.py`: **41/41 caught** over three passes (25/28 → 28/28 → +13
      printed-face mutants, all 13 dying first try — §9.46's lesson inverted, because those tests
      assert against output `main()` produces inside the test), baseline green on 35 tests, 36 min
      (`runs/mutation_anti_prior.log`). And the reason this tick exists at all: `kaggle/run.py` had
      never plumbed the flag and `campaign.py` had no cell for it, so 10c was unexecutable from the
      staged bundle — the launcher gap §9.48 is named for.
      §7.1 then got paid against the launcher itself: `bench/mutation_kaggle_bundle.py` went 42 →
      **48 mutants, 48 caught, exit 0** (`runs/mutation_kaggle_bundle.log`), six of them over the
      P10 plumbing — three in `kaggle/run.py` (default draw flipped to `on`, the token dropped
      between `--score-loss` and `--paraphrase`, argparse defaulting to `on`) and three in
      `kaggle/campaign.py` (cell hardcoding `--anti-prior off`, `--list` without its `anti=`
      column, the `dead`/unspent labels swapped). Measured, not asserted: the hardcoded-arm mutant
      was run against the pair test **with the new `f"--anti-prior {arm}" in cell(name)` line
      removed**, and it went green — 32 passed — because the one-flag-apartness check normalizes
      `on` and `off` to the same token, so a generator emitting `off` twice still yields two
      identical commands. §9.46 from the other side again: a normalization that makes two things
      comparable deletes the difference the check is about.
- [ ] **10c** The pair `antiprior_off_s0` / `antiprior_on_s0` — **~11.2 GPU-hours**, unspent *on
      the T4*, and the user's call there. Kaggle auth is broken, so the pair is running on the local
      M5 lane since 2026-10-01 at the measured 6.74 s/update averaged over its first 2,220 updates
      (§9.50), and each arm's `model.pt` goes
      to a GitHub Release the pass that writes it — the V1-B weights are gone with `/private/tmp`, so
      a regenerated checkpoint that is not shipped is a repeat of the same loss.
      **Status 2026-10-01 12:56 NPT: `off` is done and shipped; `on` is training here.** The owner put
      it on the Mac ("continue it to 3,314 updates, dose-matched to the off control ... then download +
      Release the weights immediately and eval macro vs the control's 0.5339"), so the arm is
      `runs/antiprior_on_s0c/` — the step-500 snapshot continued through the canonical launcher with
      `--steps 3315 --resume`, driver 80297 / trainer 80299 / `caffeinate -i -s -w 80297` at 80330, and
      `runs/ckpt_backup_on_s0c.sh` (80600) shipping both snapshots untended. §9.56 is the entry, and
      3,315 rather than 3,314 is in it: the loop's last executed index is `steps − 1`, and the control
      stamped `last_step: 3314`. Nothing else may run on this box until it finishes — `--stop-factor
      3.0` is armed, and a job the size of the 293 s suite, a mutation battery or a Tier 0 re-run is
      the shape that trips it (§9.56(v) records the one 26 s slice that was taken deliberately, and that
      the interval containing it still printed at the control's own 6.58 s/update).
      Two arms × 3,600 updates at the measured 5.618 s/update (`kaggle-wall-clock`). Printed by
      `python kaggle/campaign.py --include-dead`; `--list` labels both *open, unspent* rather than
      borrowing the ablation cells' *measured, not resolved*. The control is run, not borrowed:
      0.4893 came off the `myna-code` version before any of the data-loader work, so judging one new
      arm against it would price the flag *and* the code drift. `tests/test_kaggle_bundle.py` asserts
      the two generated cells are byte-identical after normalizing the one flag value — for the two
      *generated Kaggle cells*; the local pair's continuation added `--steps` and `--resume`, which
      §9.56(iv) names rather than hiding. Default stays `off` at every layer.
      What is *not* promised: that removing the shortcut makes the model read. If the macro does not
      move, the finding is that the constant answer was never what the training signal was buying.
- [ ] **9d** `banking77/intent` and `mnli/relation` are inside or outside G1's scope — a decision
      for the user, recorded so a later run does not silently average them back in.
      **Priced on 2026-09-28 so the call has numbers beside it** — `uv run python -m
      bench.scope_pricing` → `runs/scope_pricing.json`: in-scope 0.4893 / floor 0.4331 / +0.0563;
      both cells out 0.5334 / 0.4666 / **+0.0668** (the floor rises 0.0335 of the 0.044 level
      gain, so the gated half moves by 0.0105); their whole sources out is the *same* 14 cells;
      all 5 below-floor cells out 0.5526 / 0.4577 / +0.0949, still 0.147 short of 0.70. No option
      passes either half of G1 — so this is a decision about what the claim means, and it is not
      a reason to spend the ~31.2 GPU h on the expectation of clearing the bar.

## Cross-cutting
- [ ] Every new gate mutation-checked, witness quoted in the commit message (SPEC §7.1)
- [ ] `pytest` green at every tick; 0 skips other than the KEV-gated parity test
- [ ] Corrections logged in SPEC §9 the moment a claim dies

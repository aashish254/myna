# Stage C checkpoint — GitHub Release draft

**PUBLISHED 2026-10-29.** Tag `lc1-4k-s0-stageC-weights` → commit `1fb0ceb3729293b2f8290d7a781c4d1faeb487ad`;
Release id **428002663** at https://github.com/aashish254/myna/releases/tag/lc1-4k-s0-stageC-weights,
published `2026-10-29T14:53:44Z`, three assets, not a draft, not a prerelease. This file is the record of
what was published, as published — the preparation wording is gone because it became false at publish time.

## 1. Tag and assets — as published

Tag: `lc1-4k-s0-stageC-weights` — the same `<arm>-weights` form as the two Releases the repo already
fetches (`antiprior_off_s0-weights`, `antiprior_on_s0c-weights`).

The asset set is the one `src/myna/weights.py:35` already contracts for: `ASSETS = ("model.pt",
"tokenizer.json")` plus the `SHA256SUMS` manifest, so `fetch_weights(tag="lc1-4k-s0-stageC-weights")`
works against it without a code change.

| file | bytes | sha256 | role |
|---|---|---|---|
| `model.pt` | 61,442,325 | `fb37a28b88bea29a0bb871dceb8f7b835b216e8c11a987830a420fb15d39b7e8` | **the released checkpoint** — what `Myna()` and the needle ladder load |
| `tokenizer.json` | 22,096 | `5455c56a7e1ece5e254f04075c4b16d3349dbc5fc8595715726a2e8bae8c4c39` | the vocab the checkpoint was trained and scored with (byte-identical in all three stage dirs) |
| `SHA256SUMS` | 156 | `7e0be345d9542e95e837e725b77f2eb1f1929e866aeddc6267710d29240952ad` | the manifest the helper verifies against — its two lines are the two digests above |

Not uploaded: `model_last.pt` (184,355,257 B,
`25b868e9f53660237d584da12066afc8d2b462f9392b8f2afa8d7ae9076674dd`). It is the resume artifact — same
weights plus Adam's two moment tensors and the scheduler — and a reader reproducing the ladder needs none
of that. It stays witnessed in `runs/lc1_4k_s0.stageC_final.log:12` and SPEC §9.64 (i) either way.

## 2. Verification performed on the staged files, 2026-10-29

Hashes recomputed here from `~/myna-staging/lc1-4k-s0/` and compared to the committed witnesses. Every
witnessed prefix matches a real file; nothing was renamed to make it fit.

| staged file | bytes | sha256 | witness it matches |
|---|---|---|---|
| `stageC_final/model.pt` | 61,442,325 | `fb37a28b88bea29a…` | SPEC §9.64 (i); `runs/lc1_4k_s0.stageC_final.log:13` |
| `stageC_final/model_last.pt` | 184,355,257 | `25b868e9f5366023…` | SPEC §9.64 (i) and (iv); `runs/lc1_4k_s0.stageC_final.log:12` |
| `stageC_final/tokenizer.json` | 22,096 | `5455c56a7e1ece5e…` | identical in `stageA/`, `stageB_resume/`, `resume_base_s424/` |
| `stageB_resume/model_last.pt` | 184,355,257 | `a2bbc2eb0c6ee560…` | SPEC §9.64 (iv); `runs/lc1_4k_s0.stageB_resume.log:9` |
| `resume_base_s424/model_last.pt` | 184,355,257 | `d3548a881b9280e0…` | SPEC §9.64 (i); `runs/lc1_4k_s0.stageB_resume.log:7` |
| `stageC_final/needle_stageC_final.json` | 1,298 | `636f17704758438e…` | **byte-identical** to the committed `runs/needle_lc1_stageC_final.json`, same hash |
| `stageC_final/needle_stageC_final.md` | 1,049 | `8c734bb72745cae9…` | **byte-identical** to the committed `runs/needle_lc1_stageC_final.md`, same hash |

Its `cmd` field names this directory as the checkpoint it evaluated
(`--ckpt /Users/aashish/myna-staging/lc1-4k-s0/stageC_final`), so the file that was released is the
file the witnessed ladder was run against, not a sibling.

`model_last.pt` also carries `"step": 1199` inside it — the same `last_step` the witnesses record — and
both files carry the identical `state_dict`.

## 3. Why one file is 61,442,325 B and the other 184,355,257 B

Measured by loading both, not inferred. `src/myna/train.py:808-810` writes them:

- `model.pt` = `{"state_dict", "cfg", "temperature"}`. Its tensor bodies sum to **61,415,936 B**
  (15,353,984 fp32 parameters × 4 B); the remaining **26,389 B** is serialization header and the `cfg`
  dict (`d_model 384, n_layers 6, n_heads 6, d_ff 1024, d_k 64, d_v 64, d_ptr 256, vocab 4096`).
- `model_last.pt` = the same three keys **plus** `step`, `optimizer` and `scheduler`. Its `state_dict` is
  the identical **61,415,936 B**, and Adam adds two more full-size fp32 tensors per parameter —
  `exp_avg` **15,353,984** values and `exp_avg_sq` **15,353,984** values = **122,831,872 B**.
  61,415,936 + 122,831,872 = **184,247,808 B** of tensor body; the file is 184,355,257 B, so the header
  plus the optimizer/scheduler/step metadata is **107,449 B**.

Plain terms: `model_last.pt` is exactly three copies of the network — the weights, and two gradient-memory
tables the optimizer needs only to keep training. `model.pt` is the one copy, which is all a reader needs to
run the model. That is the whole 122,912,932 B difference.

## 4. The Release body — as published

Published body: 4,499 characters, assembled programmatically from this file (this section's paragraph +
§1's assets and tag + §3's size explanation), and every one of the paragraph's 20 lines was diff-checked
against the draft before upload — **0 lines not verbatim**. Two stale-state words in the §1 and §3
fragments were corrected on the way in, because they would have been false on a live page:
"(proposed)" and "NOT proposed for the Release" (now "Not uploaded"). No number, no gate claim and no
sentence of the description itself was added, removed or softened; the live body still tests true for
"G4 is NOT MET, measured, not skipped", "0.18333 vs 0.90", "0.15000 vs 0.85" and "3 of 7 met, 3 not met, 1
open". The draft banner and the pre-publish half of §6 were excluded from the body as publish-time false.

> **lc1-4k-s0 Stage C — the 4k truncated-backpropagation checkpoint, G4 not met.** This is the arm
> `lc1-4k-s0`: 15,353,984 parameters at vocab 4096, trained on the M5 with `--long-context 4096
> --grad-tokens 1024 --batch 3`, where `split = 3072` tokens of each state are scanned under `no_grad`
> and only the final 1,024-token window carries gradient. It is the final snapshot of the resumed leg —
> 425 updates in leg 1 plus 775 in leg 2 = **1,200 executed updates, the full approved dose** — at
> `last_step` 1199 with `"stopped": null`, so no machine rule aborted it. `model.pt` (61,442,325 B, sha256
> `fb37a28b…`) is the checkpoint the needle ladder was run against. That ladder — 60 questions per rung,
> seed 0, `cpu`, rungs 128/1024/4096/8192/16384 — is `runs/needle_lc1_stageC_final.json`, and it reports
> **0.150 at 128, 0.2167 at 1024, 0.1833 at 4096, 0.150 at 8192, 0.150 at 16384**. Against G4's bar —
> needle-style typed decision ≥ 0.90 at 4k and ≥ 0.85 at 16k — this checkpoint is **0.7167 short at 4096
> (0.18333 vs 0.90) and 0.7000 short at 16384 (0.15000 vs 0.85): G4 is NOT MET, measured, not skipped.**
> The curve is flat at the 0.167 uniform floor rather than decaying, because the checkpoint does not answer
> the task at any length; the harness's own `baseline.readable` field is `false` for exactly that reason.
> Releasing these weights makes the ladder reproducible from a clone — download them, then re-run
> `bench/eval_needle.py --ckpt <dir> --device cpu --n 60 --seed 0 --lengths 128 1024 4096 8192 16384` and
> the accuracy column comes back cell-for-cell; `temperature` is recorded at 1.0 and never applied to the
> score, so it cannot differ between runs. It does **not** change any gate verdict: after this Release
> `bench/gates.py --check` still reads **3 of 7 met, 3 not met, 1 open**, with G1, G4 and G5 all **not
> met**, and the release gate is still not clear. The `ms` column is machine-scheduled and is not offered
> as a reproducible quantity. Publishing this checkpoint is a shipping decision, not evidence of capability.

## 5. What this Release does and does not settle

Closed one open *logistics* item — the Stage C weights existed only on this laptop under `~/myna-staging/`,
so the §9.64 ladder could be re-run by nobody but the person holding the machine. It
settles nothing about accuracy: the shortfall it documents is 0.7167 and 0.7000 at the two gate rungs, and
the mechanism question (does 4k truncated backprop give long-context decisions at all?) stays open with the
same answer it had before — at 1,200 updates it does not.

## 6. Published state, and what is still open

Published, verified through the JSON API rather than the HTML page (GitHub's web UI can bot-wall; the API
is the oracle), on 2026-10-29:

- branch `lc1-4k` pushed to `origin` as a new branch at `1fb0ceb` — `refs/heads/lc1-4k` = `1fb0ceb…`
- tag `lc1-4k-s0-stageC-weights` created on `1fb0ceb` and pushed — `refs/tags/…` = `1fb0ceb…`. **Lightweight**,
  matching the repo's own convention: all four existing `*-weights` / `*-checkpoint` tags are lightweight and
  only `v0.1.0` is annotated
- Release id **428002663**, `draft: false`, `prerelease: false`, `published_at` `2026-10-29T14:53:44Z`,
  three assets — GitHub's **server-computed** `sha256` digests for `model.pt` and `tokenizer.json` equal the
  staged-side hashes in §2, which is a stronger witness than the upload exit code
- `origin/main` untouched at `3fb3c31`; nothing merged; `pages-site` untouched at `62be5bd`; tokens and the
  8124 server untouched

Two deviations from the approved spec were caught on the API and fixed before this file said "published",
and both are on the page as the record rather than as an aside:

1. the manifest first uploaded as `stageC_SHA256SUMS` — `gh release create` ignored the `path#label`
   argument and took the local basename, which would have silently broken `fetch_weights`, since
   `src/myna/weights.py:36` looks for `SHA256SUMS` by name. Asset id 688890555 deleted, re-listed,
   re-uploaded from a file actually named `SHA256SUMS`, and the **served** 156 bytes re-hashed to confirm
   they equal the uploaded bytes.
2. `target_commitish` came back as `"main"`. The tag itself was always correct at `1fb0ceb`, so the page
   displayed the right commit, but that metadata meant a deleted-and-recreated tag would land on a commit
   that never saw this checkpoint. Corrected with `--target 1fb0ceb3…`; the API now reports the full commit
   sha.

**The claim this file's §4 published as an assertion has now been replayed, on 2026-10-29, from a fresh
clone** outside the working tree: clone at `1fb0ceb` on `lc1-4k`, `fetch_weights(tag=
"lc1-4k-s0-stageC-weights")` → `~/.cache/myna/lc1-4k-s0-stageC-weights`, both assets hashed against the live
manifest **before any model ran** (exact), then the ladder at n=60, seed 0, `cpu`, five rungs — **all five
accuracy cells returned bit-identical to `runs/needle_lc1_stageC_final.json`**. The `ms` column was not
compared. Three boundaries on that, recorded because they are what a reader cannot assume: same laptop as
the original (not a cross-machine determinism test); this project's existing virtualenv as the interpreter
(torch 2.14.0, python 3.13.14) rather than a fresh dependency install, though the clone's own `src/myna`
was proven to be the copy answering via `myna.__file__`; and `bench/reproduce.py` still has no registry row
for the lc1 ladder, so these five cells are guarded by this replay and the committed JSON, not by the audit
that guards the other 34 rows.

`src/myna/weights.py`'s `DEFAULT_TAG` still `antiprior_off_s0-weights`; changing the default arm is a
separate owner decision from shipping this one. The replay artifact itself lives at
`/tmp/myna-stageC-replay/` — outside the repo, and not a committed witness.

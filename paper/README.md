# Myna: two journal-format manuscripts

| Format | LaTeX | PDF | Word | Length |
|---|---|---|---|---|
| IEEE journal | `paper/ieee/myna_ieee.tex` | `paper/ieee/myna_ieee.pdf` | `paper/ieee/myna_ieee.docx` | 13 pp |
| Elsevier journal | `paper/elsevier/myna_elsevier.tex` | `paper/elsevier/myna_elsevier.pdf` | `paper/elsevier/myna_elsevier.docx` | 15 pp |

Both share one body (`paper/shared/body_main.tex` + `body_appendix.tex`), one bibliography
(`refs.bib`), and one figure set. The IEEE build uses IEEEtran; the Elsevier build uses
`elsarticle` in two-column final style.

## What this version is

A measurement study, not a leaderboard claim. Myna v6 is compared on the same 1,176 test bundles
with five external systems — three configurations of Laya 0.4.19, VegaML 0.15.0, a supervised
TF-IDF classifier, a train-prior majority classifier, and a zero-shot NLI scorer — and the paper
states the footing of every column (Myna is in-domain; the two engines are out-of-domain or
zero-shot on most cells).

| Arm | macro | overall | bundle exact |
|---|---:|---:|---:|
| Myna v6, routed | **0.8620** | 0.8632 | 0.8393 |
| Myna v6, packed everywhere | 0.8222 | 0.8118 | 0.7764 |
| TF-IDF + logistic regression | 0.7111 | 0.6840 | 0.6293 |
| DeBERTa-v3-large, per-cell (3 epochs, 2 seeds) | 0.7907 | 0.7684 | 0.7245 |
| DeBERTa-v3-large, per-cell (2 epochs, 3 seeds) | 0.7814 | 0.7560 | 0.7089 |
| DeBERTa-v3-large, per-cell (1 epoch, 2 seeds) | 0.7126 | 0.6799 | 0.6165 |
| Laya 0.4.19 (typed-decisions best) | 0.7050 | 0.6792 | 0.6182 |
| Laya 0.4.19 (raised + tournament) | 0.6829 | 0.6597 | 0.5918 |
| Laya 0.4.19 (English default) | 0.6624 | 0.6333 | 0.5595 |
| VegaML 0.15.0 | 0.6160 | 0.5833 | 0.5119 |
| Train-prior majority | 0.3726 | 0.2965 | 0.2041 |
| Zero-shot NLI | 0.3278 | 0.3465 | 0.3027 |

## What changed from the earlier draft

- **A scoring scare, investigated and retracted.** An internal note claimed an off-by-one in the
  ordinal cell of `bench/v6_audit.py`. It was wrong: the suite stores the score label as a 0-based
  class index (0..4), and the original comparison was correct. The numbers stand at 0.8620 macro and
  ~0.63 on the ordinal cells; the retraction is `paper/MEASUREMENT-CORRECTION.md`.
- **An in-domain baseline, same backbone and split.** DeBERTa-v3-large with one linear head per
  cell, question-conditioned: 0.7126 at one epoch (two seeds), 0.7814 at two (three seeds) and
  0.7907 at three (two seeds), so most of the
  first gap was underfitting. Myna leads the three-epoch control by 7.1 points, concentrated on
  multi-class choice cells (dbpedia14 0.72 vs 0.96, mnli 0.62 vs 0.86).
- **External engine baselines restored.** Laya and VegaML are back, evaluated on the same rows by
  their own released harnesses, with the footing stated beside each number.
- **An isolated routing ablation.** The packed scorer is forced on the v6 checkpoint
  (`--max-packed 1000`) and compared with the routed arm on the same weights: banking77 moves from
  0.0345 to 0.6724, every other cell is identical, and the macro difference is 0.0399. The paper
  presents routing as a budget repair for one cell, with its compute cost.
- **Consistency is a released predicate, not a claim.** `src/myna/consistency.py` ships the
  definition; Myna measures 3/172 against 10/172 (Laya) and 29/172 (VegaML). The graph head in the
  checkpoint is never called, and no architectural claim is attached to it.
- **Conformal sets are a negative result.** 83.68% coverage at a 90% target, 93.33% at 95%; the
  paper uses ECE, Brier, and risk–coverage instead and states that the sets are not an abstention
  mechanism on this suite.
- **The campaign notebooks are released.** `release/notebooks/` holds three copies of the v5 run
  and the v6 campaign, with digests and cell ranges. They settle what the earlier draft asserted:
  the training loss is per-question cross-entropy with an unused consistency constant, the graph
  gate is zero and the ablation is identical on/off, the 0/172 record comes from a narrow
  choice--boolean rule that cannot see the yelp case, the v6 temperatures and conformal quantiles
  are fitted on a CAL split drawn from train (`release/build_cal_split.py`, 1,999 bundles), and the
  CORAL ordinal head was trained and gated out on development.
- **A reference trainer is released.** `release/train_v6.py` trains the packed-readout arm through
  the engine's own renderer (parity tested), fits temperatures and conformal quantiles on the
  calibration split, and writes a checkpoint whose heads load with `strict=True`.

## Build

```bash
cd paper/ieee && tectonic -X compile myna_ieee.tex
cd paper/elsevier && tectonic -X compile myna_elsevier.tex
python paper/shared/make_docx.py          # both .docx files
uv run --with matplotlib python paper/make_figures_v2.py
```

The wrappers load `fontspec` with Times New Roman: `tectonic`'s default TeX Live does not carry
TeX Gyre, and without a Unicode font the XeTeX build silently loses bold and italic.

## Figures

`paper/make_figures_v2.py` regenerates `fig2_cardinality`, `fig3_external`, and
`fig4_calibration` from `runs/paper_numbers.json` and `runs/witness-competitor-head-to-head-1440.json`.
Figures 1, 5, 6, and 7 (architecture schematic, systems, diagnostics, decision sequence) are
campaign artifacts and are unchanged.

## Evidence

Every number is recomputed from a committed artifact by `bench/paper_numbers.py`, which refuses a
per-question record whose correctness flag disagrees with its own prediction and gold. The main
witnesses are `runs/paper_v6_audit.json`, `runs/paper_myna_v6_test.rowoutcomes.json`,
`runs/v6_packed_forced.*`, `runs/paper_numbers.json`, `runs/consistency_audit.json`,
`runs/paper_cell_intervals.json`, `runs/external_{majority,tfidf,nli}.json`, and the Laya/VegaML
row-outcome files. The reproducer is `release/README.md`; `bench/reproduce.py --check` gates the
registry row that quotes the corrected re-score.

## Other documents

- `paper/MEASUREMENT-CORRECTION.md`: the ordinal scoring error, its signature, and every quantity
  it changed.
- `paper/CLAIM-AUDIT.md`: claim-by-claim audit of the original brief against the repository. It
  predates the correction; where it quotes 0.8620, read the correction record.
- `paper/STYLE-SURVEY.md`: the 46-paper survey behind the structure and figure conventions.
- `release/README.md`: how to verify the paper from a clean checkout.

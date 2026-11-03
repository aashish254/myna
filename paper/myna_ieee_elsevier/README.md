# Myna: two journal-format papers

This directory now contains two independent versions of the same single-system study, both written
without any external engine baselines:

| Format | LaTeX | PDF | Word | Length |
|---|---|---|---|---|
| IEEE journal | paper/ieee/myna_ieee.tex | paper/ieee/myna_ieee.pdf | paper/ieee/myna_ieee.docx | 12 pages |
| Elsevier journal | paper/elsevier/myna_elsevier.tex | paper/elsevier/myna_elsevier.pdf | paper/elsevier/myna_elsevier.docx | 14 pages |

Both use the same body text, figures, tables, and curated bibliography. The IEEE version uses
IEEEtran journal style; the Elsevier version uses elsarticle in two-column final style.

## Author

Aashish Kumar Mahato
Independent Researcher, Kathmandu, Nepal
Graduate of Vellore Institute of Technology (VIT), Vellore, India.
This work was conducted independently after graduation.

## Build

    cd paper/ieee
    tectonic -X compile myna_ieee.tex

    cd paper/elsevier
    tectonic -X compile myna_elsevier.tex

Each folder is self-contained: it carries body_main.tex, body_appendix.tex, refs.bib, and its own
figures/ directory. Tectonic downloads IEEEtran.cls or elsarticle.cls on first use.

## What changed from the earlier draft

- All Laya and VegaML benchmark columns, mentions, and software references are removed. The paper
  reports only Myna's own arms: v5 Packed, v5 Unified, v6, and a local v6 re-score.
- Every comparison is now an internal ablation, and the paper says so in the abstract, the
  introduction, and the limitations.
- The systems section reports absolute Myna latency and same-model runtime comparisons (MLX vs
  PyTorch MPS, fp32 vs int8, browser vs torch). No cross-model speedup is claimed.
- References were reduced to papers a Myna-style project would actually cite: typed decision models,
  linear attention/SSMs, calibration, selective prediction, conformal prediction, consistency,
  early exit, quantization, and the dataset papers. Laya and VegaML are not cited.
- Author name, affiliation, and the independent-project footnote are on both title pages.

## Evidence

Every number is from a committed artifact or a script in the repository. The fresh audit command
was:

    uv run python -m bench.v6_audit --ckpt "myna-v5-prototype 2" --split test --every 1 \
      --out runs/paper_v6_audit.json --md runs/paper_v6_audit.md \
      --metrics runs/paper_v6_test.metrics.json --rowout runs/paper_myna_v6_test.rowoutcomes.json

It scored all 1,176 bundles in 806 s on CPU and reproduced all 1,440 question outcomes bit-for-bit,
with macro 0.8620 and overall 0.8632.

The main claims are:

- Routing repairs the 77-way banking77 cell from 0.0086 to 0.5345 (v5) and 0.6638 (v6), and lifts
  16-cell macro by 0.0329 and 0.0081; the other 15 cells stay at 0.8712 macro.
- Joint bundle exact match is 77.30% / 82.48% / 83.76% / 83.93% for the four arms.
- The checkpoint records 0/172 contradictions; under the explicit predicate the v6 arm has 3/172.
  The paper reports both and does not claim zero.
- ECE 0.0189 and Brier 0.0434; 90% conformal sets cover 85.49%, 95% sets cover 93.68%. The paper
  reports the undercoverage and treats the sets as a ranking signal, not a coverage guarantee.
- The streaming arm is 14.45M parameters with a 576 KiB state, 33.08–869.95 ms end-to-end at
  1–50 questions, 1.76–2.22x under MLX vs PyTorch MPS, and 1.03e-6 browser probability drift.

## Word files

The .docx files are single-column editable manuscripts generated from the LaTeX body with pandoc,
then styled in Times New Roman. They include the main text, tables, figures, appendix, and
references. The PDFs are the format-specific versions; the Word files do not reproduce IEEE or
Elsevier two-column layout.

## Earlier draft

paper/archive_acl/ holds the earlier ACL-style draft with the competitor columns. It is kept only as
a record and is not a deliverable of this version.

## Other documents

- paper/CLAIM-AUDIT.md: claim-by-claim audit of the original brief against the repository.
- paper/STYLE-SURVEY.md: the 46-paper survey that determined the structure and figure conventions.
- paper/shared/: shared body text, bibliography, figure-generation and DOCX-generation scripts.

# Draft 1 — TMLR submission package

Built 2026-11-05 from the shared body (`paper/shared/`), anonymized and typeset with the official
TMLR style file.

| file | what it is |
|---|---|
| `myna_tmlr.tex` | anonymized TMLR build; pulls `draft_body_main.tex` + `draft_body_appendix.tex` |
| `myna_tmlr.pdf` | 22 pp, "Anonymous authors", no ORCID, no contributions block |
| `draft_body_main.tex` | shared body with the author-contributions block removed |
| `draft_body_appendix.tex` | appendix copy |
| `tmlr.sty` | official style file (JmlrOrg/tmlr-style-file, `main`) |
| `refs.bib` | bibliography |

Rebuild with `tectonic -X compile myna_tmlr.tex`.

## Before you submit on OpenReview

1. **Anonymous repository link — required.** Reviewers asked for a verifiable artifact. Mirror the
   repo with Anonymous GitHub (`https://anonymous.4open.science/`) and add one sentence to the
   availability paragraph, e.g. `Code, data and per-question records: \url{https://anonymous.4open.science/r/XXXX}`.
   Do not link the named GitHub repo; it de-anonymizes the submission.
2. **Check the PDF for identifiers**: `pdftotext myna_tmlr.pdf - | grep -i -E 'mahato|dhakal|orcid|vellore'`
   should return nothing. It currently does.
3. **Fill the OpenReview form**: title, abstract (copy from the PDF), keywords, and the
   "Reproducibility" note if the venue asks for it.
4. **Supplement**: zip `paper_draft_v1/` sources if the form accepts attachments.

## What this draft already answers from the reviews

- Table II arithmetic corrected: the other-15-cells figure is 0.8747, which reproduces 0.8620/0.8222.
- Abstract leads with the same-backbone comparison (0.7126), not the external engines.
- Formalisation section with numbered equations.
- Conformal section reports released (83.68), refit per-primitive (86.94), disjoint halves (87.29),
  pooled refit (89.17) and Mondrian (83.96), plus the independence check.
- Streaming arm demoted to a compute result; VOID ablation removed.
- `noul` defined; K=78 explained; bootstrap counts reconciled; repository paths moved out of the body.

## Known gaps this draft still carries (draft-2 work)

- The in-domain baseline is one epoch; the 2-epoch run was still training at submission time.
- Myna-v6 has a single training seed.
- Paired row-clustered intervals against the external engines are not yet in the paper.
- Out-of-domain cell evaluation is absent.

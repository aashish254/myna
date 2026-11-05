# Submitting to TMLR — what to do, in order

You have never submitted before, so this is the whole procedure, not a summary. TMLR is a journal
that runs entirely on OpenReview. There is no editor you email; you create a submission, it gets
reviewers, and you get a decision in roughly 6–10 weeks.

---

## 0. What is in this folder

| file | what it is | do you upload it? |
|---|---|---|
| `upload/myna_tmlr.pdf` | the paper, anonymous, TMLR style | **yes** — as the PDF |
| `upload/myna_supplement.zip` | sources: tex, style, bib, figures | optional, but upload it |
| `upload/abstract.txt` | plain-text abstract for the form | paste into the form |
| `upload/authors.txt` | the real author list for the form | paste into the form |
| `upload/keywords.txt` | keywords | paste into the form |
| `SUBMISSION-GUIDE.md` | this file | no |
| `myna_tmlr.tex` + friends | build sources | no (already in the zip) |

**There is no "proof" to upload.** TMLR does not ask for degree certificates, affiliation letters or
ID. What it asks is the PDF, an abstract, authors, keywords, and optionally code/supplement.

---

## 1. One-time account setup (15 minutes)

1. Go to <https://openreview.net> and sign up with your email.
2. Confirm the email (the link can take a few minutes).
3. Open **Profile** → add your **name exactly as you want it published**, your **ORCID**
   (`0009-0007-8952-6357`), and your affiliation (**Independent Researcher**).
4. Add Prakrist Dhakal as a co-author on the submission later — he does not need an account to be
   listed, but if he has one, link it.
5. Go to the TMLR group: <https://openreview.net/group?id=TMLR>. If you see "New Submission", your
   account is active. If you don't, your email is not confirmed yet.

---

## 2. Before you click submit — three checks

- **Anonymity.** The PDF must contain no names, no ORCID, no GitHub username, no acknowledgments.
  Check with:
  ```bash
  pdftotext upload/myna_tmlr.pdf - | grep -i -E 'mahato|dhakal|orcid|vellore|github.com/aashish'
  ```
  It must print nothing. It currently prints nothing.
- **Anonymous code link.** Create a mirror at <https://anonymous.4open.science/> and add this one
  sentence to the "Data and code availability" paragraph in the paper:
  `Code, data and per-question records: \url{https://anonymous.4open.science/r/XXXX}`.
  Then rebuild (`tectonic -X compile myna_tmlr.tex`). Without this, reviewers cannot verify the
  reproducibility claims — the single easiest mark to lose.
- **Abstract length.** The form takes plain text; `upload/abstract.txt` is ready.

---

## 3. The submission form

Fill these fields exactly as follows:

| field | value |
|---|---|
| Title | `Myna: A Measurement Study of Typed Agent Decision Engines` |
| Authors | `Aashish Kumar Mahato, Prakrist Dhakal` (hidden from reviewers; TMLR records them) |
| Abstract | contents of `upload/abstract.txt` |
| Keywords | from `upload/keywords.txt` |
| TL;DR | one sentence, see below |
| Primary area | *Interpretability and Analysis* or *Other* — pick whichever the form offers; the paper is a measurement study |
| PDF | `upload/myna_tmlr.pdf` |
| Supplementary material | `upload/myna_supplement.zip` |
| Conflicts of interest | declare anyone you have worked with; if nobody, leave empty |
| Dual submission | confirm this is not under review elsewhere (it is not) |

**TL;DR suggestion:** `A measurement study of a typed decision engine: what routing, consistency and
conformal prediction actually buy on a frozen 16-cell suite, including the negative results.`

---

## 4. After you submit

- The paper appears on your OpenReview profile as "Under review".
- Reviewers are assigned automatically; expect the first reviews in **4–8 weeks**.
- You will get an author-response window. You do **not** upload a new PDF then; you write a reply.
- Decision is one of: accept, accept with minor revision, reject. A reject is common and is not a
  verdict on the work — TMLR rejects most first submissions.
- **Camera-ready** (only if accepted): put the names back —
  `\author{Aashish Kumar Mahato \\ Prakrist Dhakal}`, add ORCIDs and the contributions paragraph,
  rebuild, upload. That is the version that gets published.

---

## 5. If it is rejected

You keep the reviews. They are the most valuable thing you will get from this cycle. Draft 2 is
already planned around them: converged baseline across 3 seeds, a second Myna-v6 seed, an
out-of-domain cell, paired intervals for TF-IDF, and an external benchmark. Resubmission to TMLR is
allowed and common; the second attempt usually scores better because the reviews tell you exactly
which sentence each reviewer doubted.

---

## Appendix note: the camera-ready build

`myna_tmlr_camera.tex` is the same paper with your names, ORCIDs and the contributions paragraph
restored. It exists so that after acceptance you do not have to reconstruct the author block under
time pressure:

```bash
tectonic -X compile myna_tmlr_camera.tex   # produces myna_tmlr_camera.pdf
```

Use `myna_tmlr.pdf` for the submission (anonymous) and `myna_tmlr_camera.pdf` for the accepted
version. Do not upload the camera build by mistake — it breaks double-blind.

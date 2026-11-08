Dear Editor-in-Chief,

Please consider the enclosed manuscript, "Myna: A Measurement Study of Typed Agent Decision
Engines", for publication in Neural Networks as a research article.

The paper asks a narrow question — on a fixed option set, does an option-conditioned readout beat a
conventional classifier built on the same encoder — and answers it with a controlled measurement
rather than a leaderboard. We pair two engines behind one typed contract: a 14.45M-parameter
gated-linear-attention model that keeps a constant 576 KiB state, and a 446.63M-parameter
fine-tuned DeBERTa-v3-large with a cardinality-routed readout. On a frozen 16-cell suite of 1,176
bundles and 1,440 questions, the routed engine reaches 0.8620 macro accuracy, 6.7 points above a
fine-tuned per-cell classifier that we trained to convergence at six epochs over two seeds.

What we think makes the study useful is that it reports the negative results with the positive one.
Option routing above sixteen options helps exactly one cell and costs 11.4x latency; bundle
consistency comes from the encoder, not from the graph head; split-conformal prediction sets
under-cover their nominal level because calibration rows cluster within bundles, and refitting on a
larger split does not rescue them. Every figure is recomputed from released per-question records by
a verification script, so the numbers in the tables can be re-derived rather than trusted.

The work was conducted independently, without institutional affiliation on either author's part at
the time of submission. An earlier version was desk-rejected by TMLR without review, before any
reviewer read it; the present manuscript reports a two-seed converged control that the earlier
version did not have. We mention this only so the record is complete.

Code, the frozen suite, per-question records, training notebooks and the trained checkpoint are
publicly archived (anonymous artifact and a DOI are given in the Data and code availability
section), and the manuscript states its limits: a single benchmark suite, one training seed for the
accuracy engine, and no out-of-domain cell.

This manuscript is not under consideration elsewhere, and the authors declare no competing
interests. Both authors have approved the submission.

Yours sincerely,

Aashish Kumar Mahato (corresponding author)
Independent Researcher, Kathmandu, Nepal
ORCID 0009-0007-8952-6357
[email]

Prakrist Dhakal
Independent Researcher, North Carolina, USA
ORCID 0009-0000-9463-0094

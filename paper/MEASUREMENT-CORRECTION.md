# Retraction: there was no scoring off-by-one

**Status: withdrawn 2026-11-04.** This file previously claimed that the evaluation harness scored the
ordinal (`score`) cells one level off, and that the v6 macro was therefore 0.7777 rather than 0.8620.
That claim was wrong, and it is retracted here rather than quietly deleted, because it was committed,
quoted in the model card, and used to rewrite the manuscript for a day.

## What was actually true

The suite stores a `score` label as a **0-based class index** into the five-level `criteria` list.
All three score sources carry labels `0..4` with all five values present in the test split
(amazon/stars 13/21/19/13/14, sst5 13/20/16/18/13, yelp 13/12/18/22/15), and the upstream datasets
— `SetFit/sst5`, `Yelp/yelp_review_full`, `SetFit/amazon_reviews_multi_en` — all use 0-based labels.

The original harness compared `top == ranked[golds[i]]`. With `golds[i]` a 0-based index, that is
correct. There was no off-by-one.

The error came from reading the yelp criteria text ("1 star … 5 stars") as a 1-based star level and
assuming `label: 4` meant four stars, when it means index 4, i.e. five stars. The corrected numbers
that followed (macro 0.7777, score cells 0.1750, ECE 0.0637) were the artefact.

Two independent checks that the original numbers are the right ones:

- The score cells score 0.6375 / 0.6125 / 0.6250 under the original comparison — above the 0.20
  uniform floor, which is what a fine-tuned encoder should do. The "corrected" 0.1750 was *below*
  chance, which should have been the tell.
- The v6 notebook gates its ordinal head on "development score accuracy > 0.6542". That threshold
  only makes sense if score accuracy is around 0.65; under the bad mapping it would have been ~0.17.

## What is true now

- Macro **0.8620**, overall 0.8632, joint bundle exact 0.8393 (routed), packed-forced 0.8222.
- Ordinal cells 0.6375 / 0.6125 / 0.6250; the disabled CORAL head scores 0.6333 on the same cells
  against 0.6292 for the packed head — a wash, which is why the campaign left it off.
- ECE 0.0190, multiclass Brier 0.0449, selective accuracy 0.9815 at 60% coverage.
- Conformal 90% sets cover 83.68% (1,205/1,440); 95% sets cover 93.33% (1,344/1,440). The
  undercoverage is real and is reported as a negative result.

The manuscript no longer contains a correction section. The harness compares `ranked[gold]` for
`choice` and `score` alike, and `bench/coral_eval.py`, `bench/cal_refit.py` and
`release/colab_baselines.py` all treat the score label as the index it is.

## The lesson

A label's space is a property of the data, not of the column name. The check that would have caught
this in one minute is the one now used everywhere here: print the label distribution before assuming
its offset. `0..4` with every value present is an index; `1..5` would have been a level.

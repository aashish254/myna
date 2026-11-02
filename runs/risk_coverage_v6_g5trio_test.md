$ python -m bench.risk_coverage --ckpt 'myna-v5-prototype 2' --split test --source banking77,dbpedia14,trec --out runs/risk_coverage_v6_g5trio_test.md
test: 348 rows / 348 questions over 190 question-sets — pooled over banking77,dbpedia14,trec only
checkpoint myna-v5-prototype 2: 446.63M params, temperature {'choice': 0.9, 'score': 0.9, 'noul': 0.85}, abstain_below=None (curve applies its own floor)

risk / coverage — myna-v5-prototype 2 on test (348 questions)

  rung  realized      n   accuracy    risk   floor
  1.00     1.000    348      0.859   0.141   0.124
  0.90     0.899    313      0.917   0.083   0.518
  0.80     0.799    278      0.950   0.050   0.759
  0.70     0.701    244      0.975   0.025   0.903
  0.60     0.601    209      0.990   0.010   0.976
  0.50     0.500    174      0.994   0.006   0.994
  0.40     0.399    139      0.993   0.007   0.997
  0.30     0.299    104      1.000   0.000   0.999
  0.20     0.201     70      1.000   0.000   1.000
  0.10     0.101     35      1.000   0.000   1.000

per source, whole split and then only the rows at or above the 0.903 floor:
source           questions   acc@1.0  cov@floor  acc@floor
banking77              116     0.672      0.328      0.921
dbpedia14              116     0.957      0.914      0.981
trec                   116     0.948      0.862      0.990

G5 (§2.2): accuracy >= 0.95 at coverage >= 0.60: PASS — best coverage at target: 0.701 · max accuracy on this curve: 1.0

floor 0.903 re-run on the engine: it abstained on 104 of 348 questions, the curve puts 104 below that floor (coverage 0.701)

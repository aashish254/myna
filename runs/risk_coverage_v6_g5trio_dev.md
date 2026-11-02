$ python -m bench.risk_coverage --ckpt 'myna-v5-prototype 2' --split development --source banking77,dbpedia14,trec --out runs/risk_coverage_v6_g5trio_dev.md
development: 348 rows / 348 questions over 190 question-sets — pooled over banking77,dbpedia14,trec only
checkpoint myna-v5-prototype 2: 446.63M params, temperature {'choice': 0.9, 'score': 0.9, 'noul': 0.85}, abstain_below=None (curve applies its own floor)

risk / coverage — myna-v5-prototype 2 on development (348 questions)

  rung  realized      n   accuracy    risk   floor
  1.00     1.000    348      0.845   0.155   0.125
  0.90     0.899    313      0.901   0.099   0.507
  0.80     0.799    278      0.939   0.061   0.806
  0.70     0.701    244      0.955   0.045   0.894
  0.60     0.601    209      0.971   0.029   0.978
  0.50     0.500    174      0.971   0.029   0.995
  0.40     0.399    139      0.971   0.029   0.998
  0.30     0.299    104      0.962   0.038   0.999
  0.20     0.201     70      0.943   0.057   1.000
  0.10     0.101     35      0.886   0.114   1.000

per source, whole split and then only the rows at or above the 0.894 floor:
source           questions   acc@1.0  cov@floor  acc@floor
banking77              116     0.724      0.397      0.978
dbpedia14              116     0.931      0.897      0.952
trec                   116     0.879      0.819      0.947

G5 (§2.2): accuracy >= 0.95 at coverage >= 0.60: PASS — best coverage at target: 0.701 · max accuracy on this curve: 0.971

floor 0.894 re-run on the engine: it abstained on 103 of 348 questions, the curve puts 103 below that floor (coverage 0.704)

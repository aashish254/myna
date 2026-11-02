$ python -m bench.risk_coverage --ckpt 'myna-v5-prototype 2' --split calibration --source banking77,dbpedia14,trec --out runs/risk_coverage_v6_g5trio.md
calibration: 120 rows / 120 questions over 42 question-sets — pooled over banking77,dbpedia14,trec only
checkpoint myna-v5-prototype 2: 446.63M params, temperature {'choice': 0.9, 'score': 0.9, 'noul': 0.85}, abstain_below=None (curve applies its own floor)

risk / coverage — myna-v5-prototype 2 on calibration (120 questions)

  rung  realized      n   accuracy    risk   floor
  1.00     1.000    120      0.883   0.117   0.299
  0.90     0.900    108      0.917   0.083   0.692
  0.80     0.800     96      0.969   0.031   0.852
  0.70     0.700     84      0.976   0.024   0.958
  0.60     0.600     72      0.986   0.014   0.992
  0.50     0.500     60      1.000   0.000   0.997
  0.40     0.400     48      1.000   0.000   0.999
  0.30     0.300     36      1.000   0.000   1.000
  0.20     0.200     24      1.000   0.000   1.000
  0.10     0.100     12      1.000   0.000   1.000

per source, whole split and then only the rows at or above the 0.852 floor:
source           questions   acc@1.0  cov@floor  acc@floor
banking77               40     0.725      0.550      0.909
dbpedia14               40     1.000      0.975      1.000
trec                    40     0.925      0.875      0.971

G5 (§2.2): accuracy >= 0.95 at coverage >= 0.60: PASS — best coverage at target: 0.8 · max accuracy on this curve: 1.0

floor 0.852 re-run on the engine: it abstained on 24 of 120 questions, the curve puts 24 below that floor (coverage 0.800)

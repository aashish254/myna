$ python -m bench.risk_coverage --ckpt 'myna-v5-prototype 2' --split calibration --out runs/risk_coverage_v6.md
calibration: 448 rows / 568 questions over 168 question-sets
checkpoint myna-v5-prototype 2: 446.63M params, temperature {'choice': 0.9, 'score': 0.9, 'noul': 0.85}, abstain_below=None (curve applies its own floor)

risk / coverage — myna-v5-prototype 2 on calibration (568 questions)

  rung  realized      n   accuracy    risk   floor
  1.00     1.000    568      0.856   0.144   0.299
  0.90     0.900    511      0.896   0.104   0.561
  0.80     0.799    454      0.923   0.077   0.661
  0.70     0.701    398      0.947   0.053   0.795
  0.60     0.600    341      0.968   0.032   0.884
  0.50     0.500    284      0.989   0.011   0.943
  0.40     0.400    227      0.987   0.013   0.966
  0.30     0.299    170      0.988   0.012   0.987
  0.20     0.201    114      1.000   0.000   0.995
  0.10     0.100     57      1.000   0.000   0.999

per source, whole split and then only the rows at or above the 0.884 floor:
source           questions   acc@1.0  cov@floor  acc@floor
agnews                 120     0.950      0.833      0.960
amazon                  40     0.625      0.125      1.000
banking77               40     0.725      0.400      0.875
boolq                   40     0.875      0.500      1.000
contrastive             48     0.875      0.604      1.000
dbpedia14               40     1.000      0.975      1.000
imdb                    40     1.000      0.825      1.000
mnli                    40     0.850      0.625      0.920
sst5                    40     0.700      0.000        n/a
trec                    40     0.925      0.850      0.971
yelp                    80     0.775      0.500      0.950

G5 (§2.2): accuracy >= 0.95 at coverage >= 0.60: PASS — best coverage at target: 0.6 · max accuracy on this curve: 1.0

floor 0.884 re-run on the engine: it abstained on 227 of 568 questions, the curve puts 227 below that floor (coverage 0.600)

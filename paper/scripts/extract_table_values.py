import json
W5=json.load(open("runs/witness-v5-master-paper.json"))
WH=json.load(open("runs/witness-hybrid-v3.json"))
W6=json.load(open("runs/witness-v6-master-paper.json"))
for name, macro, b77 in [("packed",W5["test_16cell_audit"]["macro_acc"],W5["test_16cell_audit"]["per_cell"]["banking77/intent"]),
                          ("unified",WH["test"]["macro_acc"],WH["test"]["per_cell"]["banking77/intent"]),
                          ("v6",W6["test_audit"]["macro_acc"],W6["test_audit"]["per_cell"]["banking77/intent"])]:
    other=(macro*16-b77)/15
    print(name,"macro",macro,"b77",b77,"other15",round(other,10))
rc=json.load(open("runs/risk_coverage.json"))
print("v0 curve max", max(rc["curve"], key=lambda c:c["accuracy"]))
print("v0 g5", rc["g5"])
rc6=json.load(open("runs/risk_coverage_v6.json"))
print("v6 cal curve 60", [c for c in rc6["curve"] if abs(c["coverage"]-0.6)<0.02])
print("v6 cal g5", rc6["g5"])
for p in ["runs/risk_coverage_v6_g5trio_test.json","runs/risk_coverage_v6_g5trio_dev.json","runs/risk_coverage_v6_g5trio.json"]:
    d=json.load(open(p)); print(p, d["n_questions"], d["g5"], [c for c in d["curve"] if abs(c["coverage"]-0.6)<0.02])

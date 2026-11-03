import json
WC=json.load(open("runs/witness-competitor-head-to-head-1440.json"))
W5=json.load(open("runs/witness-v5-master-paper.json"))
WH=json.load(open("runs/witness-hybrid-v3.json"))
W6=json.load(open("runs/witness-v6-master-paper.json"))
cols={"Laya":WC["Laya (Typed-Decisions Best)"],"VegaML":WC["VegaML (vega-U 0.8B)"],"v5 Packed":W5["test_16cell_audit"],"v5 Unified":WH["test"],"v6":W6["test_audit"]}
six=["agnews/is_business","agnews/is_scitech","agnews/is_sports","agnews/is_world","agnews/topic","boolq/answer"]
for k,v in cols.items():
    other=(v["macro_acc"]*16-v["per_cell"]["banking77/intent"])/15
    sixm=sum(v["per_cell"][c] for c in six)/6
    print(k, "other15",round(other,4),"six",round(sixm,4))

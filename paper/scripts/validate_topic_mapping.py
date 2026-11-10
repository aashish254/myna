import json, collections
suite={}
for l in open("data/decision-v2-pilot/test.jsonl"):
    r=json.loads(l); suite[r["_meta"]["id"]]=r
d=json.load(open("runs/myna_v6_test.rowoutcomes.json"))
TOPIC=["sports","business","scitech","world"]
mis=0; n=0; goldmap=collections.Counter(); predmap=collections.Counter()
for r in d["rows"]:
    if "topic" not in r["questions"]: continue
    q=r["questions"]["topic"]; s=suite[r["meta"]["id"]]["questions"]["topic"]
    n+=1
    if (q["pred"]==q["gold"]) != bool(q["ok"]): mis+=1
    goldmap[(q["gold"], s["label"])]+=1
    predmap[(q["pred"], s["label"])]+=1
print("rows",n,"pred==gold vs ok mismatches",mis)
print("gold index vs suite label:", goldmap.most_common(10))
print("pred index vs suite label (first 10):", predmap.most_common(10))
PY

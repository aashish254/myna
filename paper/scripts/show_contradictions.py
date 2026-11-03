import json
suite={}
for l in open("data/decision-v2-pilot/test.jsonl"):
    r=json.loads(l); suite[r["_meta"]["id"]]=r
d=json.load(open("runs/myna_v6_test.rowoutcomes.json"))
for r in d["rows"]:
    rid=r["meta"]["id"]; q=r["questions"]
    if len(q)<2: continue
    if "topic" in q:
        crit=suite[rid]["questions"]["topic"]["criteria"]; labels=list(crit)
        t=labels[q["topic"]["pred"]]
        tb={"business":"is_business","scitech":"is_scitech","sports":"is_sports","world":"is_world"}.get(t)
        if tb in q and not bool(q[tb]["pred"]):
            print("AG",rid,"variant",r["meta"].get("variant"),"topic",t,q["topic"]["pred"],"bool",tb,q[tb]["pred"],"gold",q["topic"]["gold"])
    if "rating" in q:
        crit=suite[rid]["questions"]["rating"]["criteria"]; labels=list(crit)
        rt=int(labels[q["rating"]["pred"]].split()[0])
        rec=bool(q["recommend"]["pred"])
        if (rt>=4 and not rec) or (rt<=2 and rec):
            print("YELP",rid,"variant",r["meta"].get("variant"),"rating",rt,labels[q["rating"]["pred"]],"rec",rec,"gold",q["rating"]["gold"])

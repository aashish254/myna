import json, collections
rows=[json.loads(l) for l in open("data/decision-v2-pilot/test.jsonl")]
TOPIC=["sports","business","scitech","world"]
TB={"sports":"is_sports","business":"is_business","scitech":"is_scitech","world":"is_world"}
own=collections.Counter(); other=collections.Counter()
for r in rows:
    q=r["questions"]
    if "topic" not in q: continue
    tg=q["topic"].get("label")
    t=TOPIC[tg] if isinstance(tg,int) else str(tg).lower()
    for qn,qq in q.items():
        if not qn.startswith("is_"): continue
        g=qq.get("label")
        key=(t,qn,bool(g))
        if qn==TB.get(t): own[key]+=1
        else: other[key]+=1
print("own-class boolean gold:")
for k,v in sorted(own.items()): print(k,v)
print("other-class boolean gold:")
for k,v in sorted(other.items()): print(k,v)
yc=collections.Counter()
for r in rows:
    q=r["questions"]
    if "rating" in q and "recommend" in q:
        rt=q["rating"].get("label"); rec=q["recommend"].get("label")
        yc[(rt,bool(rec))]+=1
print("yelp gold rating/recommend:")
for k,v in sorted(yc.items()): print(k,v)

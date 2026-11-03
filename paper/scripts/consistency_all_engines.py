import json, collections
suite={}
for l in open("data/decision-v2-pilot/test.jsonl"):
    r=json.loads(l); suite[r["_meta"]["id"]]=r
TB={"business":"is_business","scitech":"is_scitech","sports":"is_sports","world":"is_world"}
engines={
 "Myna v6":"runs/paper_myna_v6_test.rowoutcomes.json",
 "Laya 0.4.19":"runs/laya_0419_test.rowoutcomes.json",
 "VegaML 0.15.0":"runs/vegaml_0150_test.rowoutcomes.json",
}
def topic_label(rid, pred):
    q=suite[rid]["questions"].get("topic")
    if q is None: return None
    crit=q["criteria"]
    labels=list(crit) if isinstance(crit,dict) else list(crit)
    if isinstance(pred,int):
        return labels[pred] if 0<=pred<len(labels) else None
    return str(pred).lower()
def bool_val(v):
    if isinstance(v,bool): return v
    if isinstance(v,(int,float)): return bool(v)
    return str(v).lower() in ("yes","true","1")
def rating_label(rid,pred):
    q=suite[rid]["questions"].get("rating")
    if q is None: return None
    crit=q["criteria"]
    labels=list(crit) if isinstance(crit,dict) else list(crit)
    if isinstance(pred,int):
        # v6 pred is an index into criteria
        return labels[pred] if 0<=pred<len(labels) else None
    return str(pred)
def rating_num(rid,pred):
    lab=rating_label(rid,pred)
    if lab is None: return None
    return int(str(lab).split()[0])
out={}
for name,p in engines.items():
    d=json.load(open(p)); rows=d["rows"]
    ex=sum(all(q["ok"] for q in r["questions"].values()) for r in rows)
    multi=[r for r in rows if len(r["questions"])>1]
    exm=sum(all(q["ok"] for q in r["questions"].values()) for r in multi)
    agc=[]; ylc=[]
    for r in multi:
        rid=r["meta"]["id"]; q=r["questions"]
        if "topic" in q:
            t=topic_label(rid,q["topic"]["pred"])
            b=TB.get(t)
            if b and b in q and not bool_val(q[b]["pred"]):
                agc.append((rid,t,b,q["topic"]["pred"],q[b]["pred"]))
        if "rating" in q and "recommend" in q:
            rt=rating_num(rid,q["rating"]["pred"]); rec=bool_val(q["recommend"]["pred"])
            if rt is not None and ((rt>=4 and not rec) or (rt<=2 and rec)):
                ylc.append((rid,rt,rec,q["rating"]["pred"]))
    out[name]={"bundle_exact":round(ex/len(rows),6),"bundle_exact_n":ex,"multi_exact":exm,
               "agnews_contradictions":len(agc),"yelp_contradictions":len(ylc),
               "total":len(agc)+len(ylc),"multi":len(multi),
               "ag_examples":agc[:3],"yelp_examples":ylc[:3]}
print(json.dumps(out,indent=1))
json.dump(out,open("paper/data/consistency_all_engines.json","w"),indent=1)

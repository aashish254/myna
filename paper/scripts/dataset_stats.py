import json, collections, statistics
splits=["train","development","calibration","test"]
out={}
for s in splits:
    p=f"data/decision-v2-pilot/{s}.jsonl"
    rows=[json.loads(l) for l in open(p)]
    qn=sum(len(r["questions"]) for r in rows)
    cells=collections.Counter()
    srcs=collections.Counter()
    ks=[]
    for r in rows:
        for name,q in r["questions"].items():
            cells[name]+=1
            srcs[r["_meta"]["source"]]+=1
            crit=q.get("criteria")
            if q["type"]=="choice":
                k=len(crit) if crit is not None else 0
            elif q["type"]=="score":
                k=len(crit) if crit is not None else 0
            else:
                k=2
            ks.append(k)
    out[s]={"rows":len(rows),"questions":qn,"cells":len(cells),"sources":len(srcs),
            "questions_by_name":dict(cells),"sources_count":dict(srcs),
            "k_min":min(ks),"k_max":max(ks),"k_median":statistics.median(ks),
            "k_gt16":sum(1 for k in ks if k>16)}
print(json.dumps(out,indent=1)[:5000])
json.dump(out,open("paper/data/dataset_stats.json","w"),indent=1)

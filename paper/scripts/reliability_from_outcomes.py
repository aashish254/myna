import json, math, statistics
d=json.load(open("runs/myna_v6_test.rowoutcomes.json"))
qs=[]
for r in d["rows"]:
    for name,q in r["questions"].items():
        qs.append((q["conf"], int(q["ok"]), name, r["meta"]["source"]))
qs.sort(key=lambda x:-x[0])
n=len(qs)
print("questions",n,"acc",sum(q[1] for q in qs)/n)
for cov in (0.9,0.8,0.7,0.6,0.5):
    k=round(cov*n); print("coverage",cov,"n",k,"acc",sum(q[1] for q in qs[:k])/k)
# ECE 10 equal-width bins
bins=[[] for _ in range(10)]
for c,ok,_,_ in qs:
    b=min(9,int(c*10)); bins[b].append((c,ok))
ece=0.0
print("bin n conf acc gap")
for i,b in enumerate(bins):
    if not b: continue
    conf=sum(x[0] for x in b)/len(b); acc=sum(x[1] for x in b)/len(b)
    ece += len(b)/n*abs(acc-conf)
    print(i,len(b),round(conf,4),round(acc,4),round(acc-conf,4))
print("ECE10",ece)
# Brier for noul? overall multiclass Brier not available; witness has 0.0434
# selective accuracy at 60% exact
k=round(0.6*n); print("selective60",sum(q[1] for q in qs[:k])/k,k)

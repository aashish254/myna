import json
a=json.load(open("runs/myna_v6_test.rowoutcomes.json"))
b=json.load(open("runs/paper_myna_v6_test.rowoutcomes.json"))
print("a",a["macro_acc"],a["n_rows"],"b",b["macro_acc"],b["n_rows"])
da={r["meta"]["id"]:r for r in a["rows"]}
db={r["meta"]["id"]:r for r in b["rows"]}
diff=[]
for k in da:
    qa, qb = da[k]["questions"], db[k]["questions"]
    for name in qa:
        if qa[name].get("ok") != qb[name].get("ok") or qa[name].get("pred") != qb[name].get("pred"):
            diff.append((k,name,qa[name],qb[name]))
print("question diffs",len(diff))
for d in diff[:10]: print(d)
print("a exact",sum(all(q["ok"] for q in r["questions"].values()) for r in a["rows"]))
print("b exact",sum(all(q["ok"] for q in r["questions"].values()) for r in b["rows"]))

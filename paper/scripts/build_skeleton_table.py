import json, re, statistics
d=json.load(open("paper/data/style_survey_raw.json"))
def clean(h):
    h=re.sub(r"^\d+(\.\d+)*\s*","",h).strip()
    return h
def top(heads):
    out=[]
    for h in heads:
        m=re.match(r"^(\d+)\s",h)
        if m:
            out.append(clean(h))
        elif h.lower().strip() in ("introduction","related work","background","method","methods","experiments","results","analysis","conclusion","conclusions","limitations","ethics statement","reproducibility statement","discussion"):
            out.append(clean(h))
    seen=[]
    for x in out:
        if x.lower() not in [s.lower() for s in seen]: seen.append(x)
    return seen
lines=["| # | Paper | Venue | Top-level sections (verified from full text) |","|---:|---|---|---|"]
for i,r in enumerate(d,1):
    t=top(r.get("headings",[]))
    lines.append(f"| {i} | {r['title']} | {r['venue_expected']} | {' · '.join(t) if t else '— (abstract only)'} |")
open("paper/data/style_skeleton_table.md","w").write("\n".join(lines))
print("\n".join(lines[:12]))
print("... total",len(lines)-2)
# aggregate
print("\nabstract median words", statistics.median([len(r['abstract'].split()) for r in d if r['abstract']]))
print("with we", sum(1 for r in d if re.search(r'\bwe\b',r['abstract'],re.I)), "/", len(d))
print("with number", sum(1 for r in d if re.search(r'\d',r['abstract'])))
print("RelatedWork top", sum(1 for r in d if any('related work' in x.lower() for x in top(r.get('headings',[])))))
print("Limitations top", sum(1 for r in d if any('limitation' in x.lower() for x in top(r.get('headings',[])))))
print("Ethics top", sum(1 for r in d if any('ethic' in x.lower() for x in top(r.get('headings',[])))))
print("Ablation anywhere", sum(1 for r in d if any('ablation' in x.lower() for x in r.get('headings',[]))))
print("median top-level", statistics.median([len(top(r.get('headings',[]))) for r in d]))

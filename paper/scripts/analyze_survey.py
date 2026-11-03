import json, re, statistics, collections
d=json.load(open("paper/data/style_survey_raw.json"))
def top(heads):
    out=[]
    for h in heads:
        if re.match(r"^[A-Z]?\d+(\.\d+)?\s", h) or re.match(r"^\d+\s", h):
            # top-level only: one number, no dot
            if re.match(r"^\d+\s", h): out.append(re.sub(r"^\d+\s*","",h))
        elif h.lower() in ("introduction","related work","method","methods","experiments","results","analysis","conclusion","conclusions","background","limitations","ethics statement","reproducibility statement"):
            out.append(h)
    # keep unique order
    seen=[]; 
    for x in out:
        if x.lower() not in [s.lower() for s in seen]: seen.append(x)
    return seen
rows=[]
for r in d:
    t=top(r.get("headings",[]))
    rows.append((r["short"], r["venue_expected"], t))
    print(f"{r['short']:28s} {r['venue_expected']:22s} {' | '.join(t[:10])}")
print("\n--- aggregate ---")
words=[len(r["abstract"].split()) for r in d if r["abstract"]]
print("abstracts",len(words),"median words",statistics.median(words),"min",min(words),"max",max(words))
we=sum(1 for r in d if re.search(r"\bwe\b", r["abstract"], re.I))
num=sum(1 for r in d if re.search(r"\d", r["abstract"]))
print("abstracts with 'we'",we,"with a number",num)
print("has Related Work top-level", sum(1 for _,_,t in rows if any("related work" in x.lower() for x in t)))
print("has Limitations", sum(1 for _,_,t in rows if any("limitation" in x.lower() for x in t)))
print("has Ethics", sum(1 for _,_,t in rows if any("ethic" in x.lower() for x in t)))
print("has Ablation", sum(1 for r in d if any("ablation" in h.lower() for h in r.get("headings",[]))))
print("top-level count median", statistics.median([len(t) for _,_,t in rows]))

#!/usr/bin/env python
"""External baselines on decision-v2-pilot: train-prior majority, TF-IDF+logreg, zero-shot NLI."""
from __future__ import annotations
import argparse, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from myna.real_data import _option_text, _state_text

def load_split(name):
    rows=[]
    with open(ROOT / f"data/decision-v2-pilot/{name}.jsonl") as f:
        for line in f:
            r=json.loads(line); rows.append(r)
    return rows

def norm_label(v):
    if isinstance(v,bool): return "yes" if v else "no"
    if isinstance(v,int): return str(v)
    return str(v)

def question_items(rows):
    """yield (cell, row_id, qname, spec, state_text, options, gold_label)"""
    for r in rows:
        state=_state_text(r["state"])
        for qname,spec in r["questions"].items():
            source=r["_meta"]["source"]
            cell=f"{source}/{qname}"
            qtype=spec["type"]
            if qtype=="choice":
                crit=spec["criteria"]
                if isinstance(crit,dict):
                    labels=list(crit); options=[_option_text(k,v) for k,v in crit.items()]
                else:
                    labels=[str(c) for c in crit]; options=[_state_text(c) for c in crit]
            elif qtype=="score":
                labels=[str(c) for c in spec["criteria"]]; options=[_state_text(c) for c in spec["criteria"]]
            else:
                labels=["no","yes"]; options=["no","yes"]
            gold=norm_label(spec.get("label"))
            yield cell, r["_meta"]["id"], qname, spec, state, options, labels, gold

def metrics(records):
    """records: list of (cell,row_id,qname,ok)"""
    per=defaultdict(list); byrow=defaultdict(list)
    for cell,rid,qn,ok in records:
        per[cell].append(ok); byrow[rid].append(ok)
    cells=sorted(per)
    macro=float(np.mean([np.mean(per[c]) for c in cells]))
    overall=float(np.mean([ok for _,_,_,ok in records]))
    exact=float(np.mean([all(v) for v in byrow.values()]))
    return {"macro":macro,"overall":overall,"bundle_exact":exact,"n":len(records),
            "cells":{c:{"n":len(per[c]),"acc":float(np.mean(per[c]))} for c in cells}}

def run_majority():
    train=load_split("train"); test=load_split("test")
    prior=defaultdict(Counter)
    for cell,rid,qn,spec,state,opts,labels,gold in question_items(train):
        prior[cell][gold]+=1
    top={c:cnt.most_common(1)[0][0] for c,cnt in prior.items()}
    rec=[]
    for cell,rid,qn,spec,state,opts,labels,gold in question_items(test):
        pred=top.get(cell,"yes")
        rec.append((cell,rid,qn,int(pred==gold)))
    return metrics(rec)

def run_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    train=load_split("train"); test=load_split("test")
    tr=defaultdict(list); te=defaultdict(list)
    for cell,rid,qn,spec,state,opts,labels,gold in question_items(train):
        text=state+" [SEP] "+str(spec.get("instructions",""))
        tr[cell].append((text,gold))
    for cell,rid,qn,spec,state,opts,labels,gold in question_items(test):
        text=state+" [SEP] "+str(spec.get("instructions",""))
        te[cell].append((rid,qn,text,gold))
    rec=[]; per_cell={}
    for cell in sorted(tr):
        X=[t for t,_ in tr[cell]]; y=[g for _,g in tr[cell]]
        if len(set(y))<2:
            pred=Counter(y).most_common(1)[0][0]
            for rid,qn,text,gold in te.get(cell,[]): rec.append((cell,rid,qn,int(pred==gold)))
            per_cell[cell]={"n":len(te.get(cell,[])),"acc":1.0 if te.get(cell) else None}
            continue
        vec=TfidfVectorizer(ngram_range=(1,2),max_features=120000,min_df=2,sublinear_tf=True,strip_accents="unicode")
        Xv=vec.fit_transform(X)
        clf=LogisticRegression(max_iter=1500,C=4.0,n_jobs=-1)
        clf.fit(Xv,y)
        n=0; ok=0
        for rid,qn,text,gold in te.get(cell,[]):
            p=clf.predict(vec.transform([text]))[0]
            hit=int(p==gold); rec.append((cell,rid,qn,hit)); n+=1; ok+=hit
        per_cell[cell]={"n":n,"acc":ok/n if n else None}
        print("tfidf",cell,n,round(ok/n,4) if n else None,flush=True)
    m=metrics(rec); m["method"]="tfidf"; m["per_cell"]=per_cell
    return m

def run_nli():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_name="cross-encoder/nli-deberta-v3-small"
    tok=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    ent_idx=[i for i,l in model.config.id2label.items() if "entail" in str(l).lower()][0]
    test=load_split("test")
    rec=[]; per_cell=defaultdict(lambda:[0,0])
    items=list(question_items(test))
    t0=time.time()
    for n,(cell,rid,qn,spec,state,opts,labels,gold) in enumerate(items):
        premise=state
        hyps=[str(spec.get("instructions",""))+" "+o for o in opts]
        scores=[]
        with torch.inference_mode():
            for i in range(0,len(hyps),32):
                batch=hyps[i:i+32]
                enc=tok([premise]*len(batch),batch,padding=True,truncation=True,max_length=256,return_tensors="pt")
                logits=model(**enc).logits
                probs=torch.softmax(logits,dim=-1)[:,ent_idx]
                scores.extend(probs.tolist())
        pred=labels[int(np.argmax(scores))] if scores else labels[0]
        hit=int(pred==gold); rec.append((cell,rid,qn,hit)); per_cell[cell][0]+=hit; per_cell[cell][1]+=1
        if (n+1)%50==0:
            print("nli",n+1,"/",len(items),round(time.time()-t0,1),"s",flush=True)
    m=metrics(rec); m["method"]="nli_zero_shot"; m["model"]=model_name
    m["per_cell"]={c:{"n":v[1],"acc":v[0]/v[1] if v[1] else None} for c,v in per_cell.items()}
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--method",choices=["majority","tfidf","nli","all"],default="all")
    ap.add_argument("--out",default="runs/external_baselines.json")
    args=ap.parse_args()
    out={}
    if args.method in ("majority","all"): out["majority"]=run_majority(); print("majority",out["majority"]["macro"])
    if args.method in ("tfidf","all"): out["tfidf"]=run_tfidf(); print("tfidf",out["tfidf"]["macro"])
    if args.method in ("nli","all"): out["nli"]=run_nli(); print("nli",out["nli"]["macro"])
    Path(ROOT/"runs").mkdir(exist_ok=True)
    with open(ROOT/args.out,"w") as f: json.dump(out,f,indent=1)
    print("wrote",args.out)

if __name__=="__main__":
    main()

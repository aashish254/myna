import json, re, time, urllib.request, html, sys

IDS = [
 ("1706.03762","Attention Is All You Need","NeurIPS 2017"),
 ("1810.04805","BERT","NAACL 2019"),
 ("2006.03654","DeBERTa","ICLR 2021"),
 ("2111.09543","DeBERTaV3","ICLR 2023"),
 ("2001.07676","PET","EACL 2021"),
 ("2012.15723","LM-BFF","ACL 2021"),
 ("1909.00161","NLI zero-shot","EMNLP 2019"),
 ("2003.04807","Banking77","EMNLP 2020"),
 ("1804.07461","GLUE","ICLR 2019"),
 ("1905.00537","SuperGLUE","NeurIPS 2019"),
 ("1704.05426","MNLI","ACL 2018"),
 ("1905.10044","BoolQ","ACL 2019"),
 ("1706.04599","Calibration","ICML 2017"),
 ("1906.02530","Uncertainty","NeurIPS 2019"),
 ("2107.07511","Conformal intro","arXiv 2021"),
 ("1905.03222","CQR","NeurIPS 2019"),
 ("1604.04173","Split conformal","JASA 2018"),
 ("1705.08500","Selective classification","ICML 2017"),
 ("1901.09192","SelectiveNet","ICML 2019"),
 ("1612.01474","Deep ensembles","NeurIPS 2017"),
 ("1703.04977","Aleatoric/epistemic","NeurIPS 2017"),
 ("2006.16236","Linear attention","ICML 2020"),
 ("2312.06635","GLA","ICML 2024"),
 ("2312.00752","Mamba","COLM 2024"),
 ("2305.13048","RWKV","ACL Findings 2023"),
 ("2307.08621","RetNet","arXiv 2023"),
 ("2205.14135","FlashAttention","NeurIPS 2022"),
 ("2004.05150","Longformer","ACL 2020"),
 ("2007.14062","BigBird","NeurIPS 2020"),
 ("1901.02860","Transformer-XL","ACL 2019"),
 ("1709.01686","BranchyNet","ICPR 2016"),
 ("2004.02178","FastBERT","ACL 2020"),
 ("2004.12993","DeeBERT","ACL 2020"),
 ("1711.06664","Learning to defer","NeurIPS 2018"),
 ("1711.11157","Semantic loss","ICML 2018"),
 ("2203.11171","Self-consistency","ICLR 2023"),
 ("2307.09702","Guided generation","arXiv 2023"),
 ("2202.12837","Demonstrations","EMNLP 2022"),
 ("1910.01108","DistilBERT","arXiv 2019"),
 ("1902.00751","Adapters","ICML 2019"),
 ("2208.07339","LLM.int8","NeurIPS 2022"),
 ("2210.17323","GPTQ","ICLR 2023"),
 ("2211.10438","SmoothQuant","ICML 2023"),
 ("2209.11055","SetFit","EMNLP 2022"),
 ("2610.02076","LLM-as-Jev","arXiv 2026"),
 ("2610.03935","General Decision Models","arXiv 2026"),
]

def fetch(url, timeout=60):
    req=urllib.request.Request(url, headers={"User-Agent":"myna-style-survey/1.0"})
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8","ignore")

meta={}
for i in range(0, len(IDS), 10):
    chunk=IDS[i:i+10]
    ids=",".join(x[0] for x in chunk)
    xml=fetch(f"https://export.arxiv.org/api/query?id_list={ids}&max_results=20")
    for entry in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        aid=re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>", entry)
        title=re.search(r"<title>(.*?)</title>", entry, re.S)
        summ=re.search(r"<summary>(.*?)</summary>", entry, re.S)
        pub=re.search(r"<published>(.*?)</published>", entry)
        jr=re.search(r"<arxiv:journal_ref[^>]*>(.*?)</arxiv:journal_ref>", entry, re.S)
        cm=re.search(r"<arxiv:comment[^>]*>(.*?)</arxiv:comment>", entry, re.S)
        if aid and title:
            meta[aid.group(1).split("v")[0]]={
                "id":aid.group(1).split("v")[0],
                "title":html.unescape(" ".join(title.group(1).split())),
                "abstract":html.unescape(" ".join(summ.group(1).split())) if summ else "",
                "published":pub.group(1)[:10] if pub else "",
                "journal_ref":html.unescape(" ".join(jr.group(1).split())) if jr else "",
                "comment":html.unescape(" ".join(cm.group(1).split())) if cm else "",
            }
    time.sleep(1.0)
print("metadata fetched", len(meta), file=sys.stderr)

out=[]
for aid, short, venue in IDS:
    rec=meta.get(aid, {"id":aid,"title":"","abstract":"","published":"","journal_ref":"","comment":""})
    rec.update({"short":short,"venue_expected":venue})
    raw=None
    for url in (f"https://ar5iv.labs.arxiv.org/html/{aid}", f"https://arxiv.org/html/{aid}"):
        try:
            raw=fetch(url, timeout=45); break
        except Exception as e:
            last=str(e)
    if raw:
        heads=[]
        for m in re.finditer(r"<(h[23])[^>]*>(.*?)</\1>", raw, re.S):
            t=re.sub(r"<[^>]+>"," ",m.group(2)); t=html.unescape(" ".join(t.split()))
            if t and len(t)<160 and not t.lower().startswith("references"): heads.append(t)
        rec["headings"]=heads
        rec["n_figures"]=len(re.findall(r"<figcaption", raw))
        rec["n_tables"]=len(re.findall(r"<figcaption[^>]*>[^<]*Table", raw))
        low=raw.lower()
        rec["has_limitations"]=("limitations" in low)
        rec["has_ethics"]=("ethics statement" in low)
        rec["html_len"]=len(raw)
    else:
        rec["headings"]=[]; rec["n_figures"]=0; rec["n_tables"]=0; rec["has_limitations"]=None; rec["has_ethics"]=None; rec["html_len"]=0
    out.append(rec)
    print(aid, short, "heads",len(rec["headings"]),"figcaps",rec["n_figures"], file=sys.stderr)
    time.sleep(1.0)

json.dump(out, open("paper/data/style_survey_raw.json","w"), indent=1)
print("wrote", len(out))

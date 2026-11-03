import json, re, urllib.request, html, time
ids={
 "2001.07676":"PET",
 "2003.04807":"Banking77",
 "1706.04599":"Guo calibration",
 "1901.09192":"SelectiveNet",
 "1905.03222":"CQR",
 "2312.00752":"Mamba",
 "2004.02178":"FastBERT",
 "2209.11055":"SetFit",
 "2208.07339":"LLM.int8",
}
for aid,name in ids.items():
    raw=urllib.request.urlopen(urllib.request.Request(f"https://ar5iv.labs.arxiv.org/html/{aid}",headers={"User-Agent":"myna/1.0"}),timeout=60).read().decode("utf-8","ignore")
    caps=[]
    for m in re.finditer(r"<figcaption[^>]*>(.*?)</figcaption>", raw, re.S):
        t=re.sub(r"<[^>]+>"," ",m.group(1)); t=html.unescape(" ".join(t.split()))
        if t: caps.append(t)
    print("\n=========",name,aid,"figs",len(caps))
    for c in caps[:4]: print(" -",c[:260])
    time.sleep(0.8)

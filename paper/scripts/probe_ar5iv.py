import urllib.request, re, html
url="https://ar5iv.labs.arxiv.org/html/2006.03654"
raw=urllib.request.urlopen(url, timeout=60).read().decode("utf-8","ignore")
for m in re.finditer(r"<(h[123])[^>]*>(.*?)</\1>", raw, re.S):
    t=re.sub(r"<[^>]+>"," ",m.group(2)); t=html.unescape(" ".join(t.split()))
    if t and len(t)<160: print(m.group(1), "|", t)
print("figures", len(re.findall(r"<figure", raw)), "tables", len(re.findall(r"<table", raw)))

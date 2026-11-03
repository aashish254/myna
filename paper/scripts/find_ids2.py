import urllib.request, urllib.parse, re, html, time
queries=["learning to defer","distribution-free predictive inference for regression","selective classification for deep neural networks"]
for q in queries:
    url="https://export.arxiv.org/api/query?"+urllib.parse.urlencode({"search_query":f'all:"{q}"',"max_results":5})
    raw=urllib.request.urlopen(url,timeout=45).read().decode("utf-8","ignore")
    print("\nQ:",q)
    for e in re.findall(r"<entry>(.*?)</entry>",raw,re.S)[:5]:
        aid=re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>",e)
        ti=re.search(r"<title>(.*?)</title>",e,re.S)
        print(" ",aid.group(1).split("v")[0] if aid else "?", "|", html.unescape(" ".join(ti.group(1).split()))[:140] if ti else "?")
    time.sleep(1)

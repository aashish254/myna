import urllib.request, urllib.parse, re, html, time
for q in ["Character-level Convolutional Networks for Text Classification",
          "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank",
          "Learning Question Classifiers"]:
    url="https://export.arxiv.org/api/query?"+urllib.parse.urlencode({"search_query":f'ti:"{q}"',"max_results":2})
    raw=urllib.request.urlopen(url,timeout=45).read().decode("utf-8","ignore")
    print("\nQ:",q)
    for e in re.findall(r"<entry>(.*?)</entry>",raw,re.S)[:2]:
        aid=re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>",e); ti=re.search(r"<title>(.*?)</title>",e,re.S)
        print(" ",aid.group(1).split("v")[0] if aid else "?", "|", html.unescape(" ".join(ti.group(1).split()))[:130] if ti else "?")
    time.sleep(1)

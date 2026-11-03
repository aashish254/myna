import urllib.request, urllib.parse, re, html, time
queries=[
 "Making Pre-trained Language Models Better Few-shot Learners",
 "Can You Trust Your Model's Uncertainty Evaluating Predictive Uncertainty Under Dataset Shift",
 "Conformalized Quantile Regression",
 "BranchyNet Fast Inference via Early Exiting from Deep Neural Networks",
 "Learning to Defer to an Expert for Aiding Fairness",
 "A Semantic Loss Function for Deep Learning with Symbolic Knowledge",
 "Split Conformal Prediction",
]
for q in queries:
    url="https://export.arxiv.org/api/query?"+urllib.parse.urlencode({"search_query":f'ti:"{q}"',"max_results":3})
    try:
        raw=urllib.request.urlopen(url,timeout=45).read().decode("utf-8","ignore")
    except Exception as e:
        print(q,"ERR",e); continue
    print("\nQ:",q)
    for e in re.findall(r"<entry>(.*?)</entry>",raw,re.S)[:3]:
        aid=re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>",e)
        ti=re.search(r"<title>(.*?)</title>",e,re.S)
        print(" ",aid.group(1).split("v")[0] if aid else "?", "|", html.unescape(" ".join(ti.group(1).split()))[:120] if ti else "?")
    time.sleep(1)

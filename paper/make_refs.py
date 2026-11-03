import json, re, time, urllib.request, html
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
# key, arXiv id, type, venue, year, doi
PAPERS=[
 ("schick2021pet","2001.07676","inproceedings","Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics (EACL)",2021,"10.18653/v1/2021.eacl-main.20"),
 ("gao2021lmbff","2012.15723","inproceedings","Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL-IJCNLP)",2021,""),
 ("yin2019benchmarking","1909.00161","inproceedings","Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP)",2019,""),
 ("casanueva2020banking77","2003.04807","inproceedings","Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",2020,""),
 ("wang2018glue","1804.07461","inproceedings","International Conference on Learning Representations (ICLR)",2019,""),
 ("wang2019superglue","1905.00537","inproceedings","Advances in Neural Information Processing Systems (NeurIPS)",2019,""),
 ("williams2018mnli","1704.05426","inproceedings","Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)",2018,"10.18653/v1/P18-1104"),
 ("clark2019boolq","1905.10044","inproceedings","Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)",2019,"10.18653/v1/P19-1472"),
 ("devlin2019bert","1810.04805","inproceedings","Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)",2019,"10.18653/v1/N19-1423"),
 ("he2021deberta","2006.03654","inproceedings","International Conference on Learning Representations (ICLR)",2021,""),
 ("he2023debertav3","2111.09543","inproceedings","International Conference on Learning Representations (ICLR)",2023,""),
 ("tunstall2022setfit","2209.11055","inproceedings","Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)",2022,""),
 ("li2026jev","2610.02076","misc","arXiv preprint",2026,""),
 ("duan2026gdm","2610.03935","misc","arXiv preprint",2026,""),
 ("min2022rethinking","2202.12837","inproceedings","Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)",2022,""),
 ("guo2017calibration","1706.04599","inproceedings","Proceedings of the 34th International Conference on Machine Learning (ICML)",2017,""),
 ("ovadia2019uncertainty","1906.02530","inproceedings","Advances in Neural Information Processing Systems (NeurIPS)",2019,""),
 ("angelopoulos2021gentle","2107.07511","misc","arXiv preprint",2021,""),
 ("romano2019cqr","1905.03222","inproceedings","Advances in Neural Information Processing Systems (NeurIPS)",2019,""),
 ("lei2018distribution","1604.04173","article","Journal of the American Statistical Association",2018,"10.1080/01621459.2017.1307116"),
 ("geifman2017selective","1705.08500","inproceedings","Proceedings of the 34th International Conference on Machine Learning (ICML)",2017,""),
 ("geifman2019selectivenet","1901.09192","inproceedings","Proceedings of the 36th International Conference on Machine Learning (ICML)",2019,""),
 ("lakshminarayanan2017ensembles","1612.01474","inproceedings","Advances in Neural Information Processing Systems (NeurIPS)",2017,""),
 ("kendall2017uncertainties","1703.04977","inproceedings","Advances in Neural Information Processing Systems (NeurIPS)",2017,""),
 ("vaswani2017attention","1706.03762","inproceedings","Advances in Neural Information Processing Systems (NeurIPS)",2017,""),
 ("katharopoulos2020transformers","2006.16236","inproceedings","Proceedings of the 37th International Conference on Machine Learning (ICML)",2020,""),
 ("yang2024gla","2312.06635","inproceedings","Proceedings of the 41st International Conference on Machine Learning (ICML)",2024,""),
 ("gu2024mamba","2312.00752","inproceedings","Conference on Language Modeling (COLM)",2024,""),
 ("peng2023rwkv","2305.13048","inproceedings","Findings of the Association for Computational Linguistics: ACL 2023",2023,""),
 ("sun2023retnet","2307.08621","misc","arXiv preprint",2023,""),
 ("dao2022flashattention","2205.14135","inproceedings","Advances in Neural Information Processing Systems (NeurIPS)",2022,""),
 ("beltagy2020longformer","2004.05150","inproceedings","Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)",2020,""),
 ("zaheer2020bigbird","2007.14062","inproceedings","Advances in Neural Information Processing Systems (NeurIPS)",2020,""),
 ("dai2019transformerxl","1901.02860","inproceedings","Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)",2019,""),
 ("teerapittayanon2016branchynet","1709.01686","inproceedings","International Conference on Pattern Recognition (ICPR)",2016,""),
 ("xin2020fastbert","2004.02178","inproceedings","Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)",2020,""),
 ("zhou2020deebert","2004.12993","inproceedings","Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)",2020,""),
 ("madras2018predict","1711.06664","inproceedings","Advances in Neural Information Processing Systems (NeurIPS)",2018,""),
 ("xu2018semantic","1711.11157","inproceedings","Proceedings of the 35th International Conference on Machine Learning (ICML)",2018,""),
 ("wang2023selfconsistency","2203.11171","inproceedings","International Conference on Learning Representations (ICLR)",2023,""),
 ("willard2023guided","2307.09702","misc","arXiv preprint",2023,""),
 ("dettmers2022llmint8","2208.07339","inproceedings","Advances in Neural Information Processing Systems (NeurIPS)",2022,""),
 ("frantar2023gptq","2210.17323","inproceedings","International Conference on Learning Representations (ICLR)",2023,""),
 ("xiao2023smoothquant","2211.10438","inproceedings","Proceedings of the 40th International Conference on Machine Learning (ICML)",2023,""),
 ("sanh2019distilbert","1910.01108","misc","arXiv preprint",2019,""),
 ("houlsby2019adapters","1902.00751","inproceedings","Proceedings of the 36th International Conference on Machine Learning (ICML)",2019,""),
]

def fetch(url):
    req=urllib.request.Request(url,headers={"User-Agent":"myna-refs/1.0"})
    return urllib.request.urlopen(req,timeout=60).read().decode("utf-8","ignore")

meta={}
for i in range(0,len(PAPERS),10):
    ids=",".join(p[1] for p in PAPERS[i:i+10])
    xml=fetch(f"https://export.arxiv.org/api/query?id_list={ids}&max_results=20")
    for e in re.findall(r"<entry>(.*?)</entry>",xml,re.S):
        aid=re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>",e)
        ti=re.search(r"<title>(.*?)</title>",e,re.S)
        names=re.findall(r"<name>(.*?)</name>",e,re.S)
        if aid:
            meta[aid.group(1).split("v")[0]]={"title":html.unescape(" ".join(ti.group(1).split())) if ti else "","authors":[html.unescape(n) for n in names]}
    time.sleep(1)

def esc(t):
    for a,b in [("&",r"\&"),("%",r"\%"),("#",r"\#"),("_",r"\_")]:
        t=t.replace(a,b)
    return t

entries=[]
for key,aid,typ,venue,year,doi in PAPERS:
    m=meta.get(aid,{"title":"","authors":[]})
    title=esc(m["title"])
    authors=" and ".join(m["authors"]) if m["authors"] else "Anonymous"
    if typ=="misc":
        entries.append(f"@misc{{{key},\n  title = {{{title}}},\n  author = {{{authors}}},\n  year = {{{year}}},\n  eprint = {{{aid}}},\n  archivePrefix = {{arXiv}},\n  url = {{https://arxiv.org/abs/{aid}}}\n}}")
    elif typ=="article":
        entries.append(f"@article{{{key},\n  title = {{{title}}},\n  author = {{{authors}}},\n  journal = {{{venue}}},\n  year = {{{year}}},\n  doi = {{{doi}}},\n  url = {{https://arxiv.org/abs/{aid}}}\n}}")
    else:
        d=f",\n  doi = {{{doi}}}" if doi else ""
        entries.append(f"@inproceedings{{{key},\n  title = {{{title}}},\n  author = {{{authors}}},\n  booktitle = {{{venue}}},\n  year = {{{year}}}{d},\n  url = {{https://arxiv.org/abs/{aid}}}\n}}")

extra=r"""@misc{laya,
  title = {laya: a typed-decision System-1 engine},
  author = {Nandhakishor M. and Convai Innovations},
  year = {2026},
  howpublished = {Software, version 0.3.20/0.4.19},
  url = {https://github.com/NandhaKishorM/laya}
}
@misc{vegaml,
  title = {vegaml (vega-U): a frozen-reader decision engine},
  author = {Nandakishor M.},
  year = {2026},
  howpublished = {Software, version 0.15.0},
  url = {https://github.com/NandhaKishorM/vegaml}
}
@misc{onnxruntime,
  title = {{ONNX Runtime}},
  author = {{Microsoft}},
  year = {2026},
  howpublished = {Software, version 1.30.0},
  url = {https://onnxruntime.ai}
}
@misc{mlx,
  title = {{MLX}: an array framework for Apple silicon},
  author = {{Apple Machine Learning Research}},
  year = {2026},
  howpublished = {Software},
  url = {https://github.com/ml-explore/mlx}
}
@misc{viola2001rapid,
  title = {Rapid Object Detection using a Boosted Cascade of Simple Features},
  author = {Viola, Paul and Jones, Michael},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2001},
  doi = {10.1109/CVPR.2001.990517}
}
"""
out="\n\n".join(entries)+"\n\n"+extra
(ROOT/"paper/refs.bib").write_text(out)
print("wrote refs.bib",len(entries)+5,"entries")

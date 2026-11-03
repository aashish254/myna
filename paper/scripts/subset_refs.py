import re
from pathlib import Path
src=Path("paper/refs.bib").read_text()
blocks=re.split(r"\n(?=@)", src)
keep={"schick2021pet","gao2021lmbff","yin2019benchmarking","casanueva2020banking77","tunstall2022setfit","li2026jev","duan2026gdm","devlin2019bert","he2021deberta","he2023debertav3","katharopoulos2020transformers","yang2024gla","gu2024mamba","peng2023rwkv","dao2022flashattention","beltagy2020longformer","teerapittayanon2016branchynet","xin2020fastbert","zhou2020deebert","guo2017calibration","ovadia2019uncertainty","geifman2017selective","geifman2019selectivenet","lei2018distribution","angelopoulos2021gentle","romano2019cqr","xu2018semantic","wang2023selfconsistency","willard2023guided","dettmers2022llmint8","frantar2023gptq","xiao2023smoothquant","williams2018mnli","clark2019boolq","zhang2015character"}
out=[]
for b in blocks:
    m=re.match(r"@\w+\{([^,]+),", b.strip())
    if m and m.group(1) in keep:
        out.append(b.strip())
extra=r"""@inproceedings{zhang2015character,
  title = {Character-level Convolutional Networks for Text Classification},
  author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2015},
  url = {https://arxiv.org/abs/1509.01626}
}
"""
# add zhang if not present
if not any("zhang2015character" in x for x in out):
    out.append(extra.strip())
Path("paper/shared").mkdir(exist_ok=True)
Path("paper/shared/refs.bib").write_text("\n\n".join(out)+"\n")
print("entries",len(out))
print([re.match(r"@\w+\{([^,]+),",x).group(1) for x in out])

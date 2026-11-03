import re
from pathlib import Path
src=Path("paper/shared/docx_source.tex").read_text()
body=Path("paper/shared/body_main.tex").read_text()
body=body.replace("\\myna","Myna").replace("\\stream","Myna-Stream").replace("\\vsix","Myna-v6")
refs={"fig:arch":"1","fig:cardinality":"2","fig:calibration":"4","sec:cardinality":"4","sec:systems":"7","tab:dataset":"1"}
body=body.replace("Both steps are one-cell repairs, and Figure~\\ref{fig:ablation} in the appendix shows the zero deltas rather than hiding them in an aggregate.",
                  "Both steps are one-cell repairs, and the per-cell deltas are zero on the other 15 cells rather than hidden in an aggregate.")
body=body.replace("The appendix plots the synthetic/real gap and the flat needle curve.",
                  "The synthetic/real gap and the flat needle curve are reported in the supplementary material.")
for k,v in refs.items():
    body=body.replace("\\ref{"+k+"}",v)
src=src.replace("\\input{body_main}", body)
src=src.replace("\\begin{widetable}","\\begin{table}").replace("\\end{widetable}","\\end{table}")
src=src.replace("\\begin{widefigure}","\\begin{figure}").replace("\\end{widefigure}","\\end{figure}")
src=src.replace("\\graphicspath{{../figures/}}","")
def fix(m):
    path=m.group(1)
    if path.startswith("paper/") or path.startswith("/"): return m.group(0)
    return m.group(0).replace("{"+path+"}","{paper/figures/"+path+"}")
src=re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", fix, src)
Path("/tmp/myna_docx.tex").write_text(src)
print("remaining refs",len(re.findall(r"\\ref\{",src)),"macros",src.count("\\myna")+src.count("\\stream")+src.count("\\vsix"))

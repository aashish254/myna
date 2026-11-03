import re
from pathlib import Path
src=Path("paper/shared/docx_source.tex").read_text()
main=Path("paper/shared/body_main.tex").read_text()
app=Path("paper/shared/body_appendix.tex").read_text()
for a,b in [("\\myna","Myna"),("\\stream","Myna-Stream"),("\\vsix","Myna-v6")]:
    main=main.replace(a,b); app=app.replace(a,b)
mrefs={"fig:arch":"1","fig:cardinality":"2","fig:calibration":"4","sec:cardinality":"4","sec:systems":"7","tab:dataset":"1"}
main=main.replace("Both steps are one-cell repairs, and Figure~\\ref{fig:ablation} in the appendix shows the zero deltas rather than hiding them in an aggregate.",
                  "Both steps are one-cell repairs, and the per-cell deltas are zero on the other 15 cells rather than hidden in an aggregate.")
main=main.replace("The appendix plots the synthetic/real gap and the flat needle curve.",
                  "The synthetic/real gap and the flat needle curve are reported in the appendix.")
for k,v in mrefs.items(): main=main.replace("\\ref{"+k+"}",v)
arefs={"tab:consistency":"3","fig:sequence":"6","fig:ablation":"7","tab:cells":"7"}
app=app.replace("\\appendix","\\section*{Appendix}")
for k,v in arefs.items(): app=app.replace("\\ref{"+k+"}",v)
math={
"$\\times$":"×","$K$":"K","$K>16$":"K > 16","$c$":"c","$^{\\dagger}$":"†","$\\mu$":"μ",
"$K\\le16$":"K ≤ 16","$1.03\\times10^{-6}$":"1.03 × 10⁻⁶","$x$":"x","$p(\\text{yes})$":"p(yes)",
"$d_{\\text{model}}=384$":"d_model = 384","$\\tau\\in\\{\\texttt{choice},\\texttt{score},\\texttt{noul}\\}$":"τ ∈ {choice, score, noul}",
"$\\sigma(-2.3757)=0.0850$":"σ(−2.3757) = 0.0850","$\\le$":"≤","$\\Delta$":"Δ",
"$S\\in\\mathbb{R}^{H\\times d_k\\times d_v}$":"S ∈ R^(H × dk × dv)","$S$":"S",
"$Q=\\{q_1,\\dots,q_m\\}$":"Q = {q1, …, qm}","$O_q$":"O_q","$K_q=|O_q|$":"Kq = |Oq|",
"$K\\le15$":"K ≤ 15","$K=2$":"K = 2","$8.94\\times10^{-8}$":"8.94 × 10⁻⁸","$15$":"15",
"$10^{-6}$":"10⁻⁶","$0.5259/16=0.0329$":"0.5259/16 = 0.0329","$0.5259/16$":"0.5259/16",
"$-10.3$":"−10.3","$-0.0554$":"−0.0554","$(\\text{source},\\text{question})$":"(source, question)",
"$y$":"y","$r=3$":"r = 3","$r$":"r","$n$":"n","$b_c=\\text{no}$":"b_c = no","$b_c$":"b_c",
"$\\{1,\\dots,5\\}$":"{1, …, 5}","$(r\\le2 \\land y)$":"(r ≤ 2 and y)","$(r\\ge4 \\land \\neg y)$":"(r ≥ 4 and not y)",
}
for k,v in math.items():
    src=src.replace(k,v); main=main.replace(k,v); app=app.replace(k,v)
body=main+"\n\n"+app
src=src.replace("\\input{body_main}", body)
src=src.replace("\\begin{widetable}","\\begin{table}").replace("\\end{widetable}","\\end{table}")
src=src.replace("\\begin{widefigure}","\\begin{figure}").replace("\\end{widefigure}","\\end{figure}")
src=src.replace("\\graphicspath{{../figures/}}","")
def fix(m):
    path=m.group(1)
    if path.startswith("paper/") or path.startswith("/"): return m.group(0)
    return m.group(0).replace("{"+path+"}","{paper/figures/"+path+"}")
src=re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", fix, src)
# Word cannot embed PDF images reliably; use the PNG twin for every figure.
src=re.sub(r"(paper/figures/[^\s{}]+\.)pdf", r"\1png", src)
Path("/tmp/myna_docx_full.tex").write_text(src)
print("math dollars left",src.count("$"))

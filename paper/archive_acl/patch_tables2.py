from pathlib import Path
T=Path("paper/tables")
def wrap_resize(name):
    p=T/name; s=p.read_text()
    s=s.replace("\\begin{tabular}","\\resizebox{\\columnwidth}{!}{\\begin{tabular}",1)
    s=s.replace("\\end{tabular}","\\end{tabular}}",1)
    p.write_text(s); print("wrapped",name)

for n in ["tab_calibration.tex","tab_systems.tex"]:
    wrap_resize(n)

def fullwidth(name):
    p=T/name; s=p.read_text()
    s=s.replace("\\begin{table}[t]","\\begin{table*}[t]",1)
    s=s.replace("\\end{table}","\\end{table*}",1)
    s=s.replace("\\footnotesize","\\small",1)
    p.write_text(s); print("fullwidth",name)

for n in ["tab_routing.tex","tab_consistency.tex","tab_diagnostics.tex"]:
    fullwidth(n)

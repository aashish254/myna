from pathlib import Path
p=Path("paper/tables/tab_main_compact.tex")
s=p.read_text()
s=s.replace("\\begin{table}[t]","\\begin{table*}[t]",1).replace("\\end{table}","\\end{table*}",1)
p.write_text(s)
print("widened compact table")

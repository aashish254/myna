from pathlib import Path
p1=Path("paper/shared/body.tex").read_text()
p2=Path("paper/shared/body2.tex").read_text()
p1=p1.rstrip()+"\n"
if p1.endswith("\\end{table}"):
    p1=p1[:-len("\\end{table}")]+"\\end{widetable}"
# safer: replace the final occurrence
p1=p1.replace("\\end{table}\n","\\end{widetable}\n") if p1.endswith("\\end{table}\n") else p1
Path("paper/shared/body.tex").write_text(p1+"\n"+p2)
print(p1[-80:])

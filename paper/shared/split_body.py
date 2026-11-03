from pathlib import Path
p=Path("paper/shared/body.tex").read_text()
idx=p.find("\\appendix")
print("appendix idx",idx)
main=p[:idx].rstrip()+"\n"
app=p[idx:]
# remove fig6 diagnostics block from main
start=main.find("\\begin{widefigure}\n\\centering\n\\includegraphics[width=0.94\\linewidth]{fig6_diagnostics.pdf}")
if start!=-1:
    end=main.find("\\end{widefigure}",start)+len("\\end{widefigure}")
    block=main[start:end]
    main=main[:start]+main[end:]
    print("moved fig6 block",len(block))
else:
    block=""
# add a sentence pointing to the appendix figure at end of section VIII, before Discussion
marker="\\section{Discussion and Limitations}"
main=main.replace(marker, "Figure~\\ref{fig:diagnostics} in the appendix plots the synthetic/real gap and the flat needle curve.\n\n"+marker,1)
Path("paper/shared/body_main.tex").write_text(main)
Path("paper/shared/body_appendix.tex").write_text(app.rstrip()+"\n\n"+block+"\n")
print("main lines",main.count("\n"),"app lines",app.count("\n"))

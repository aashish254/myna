from pathlib import Path
p=Path("paper/shared/body_main.tex").read_text()
start=p.find("\\section{Bundle Consistency}")
end=p.find("\\section{Calibration and Conformal Abstention}")
print(start,end)
p2=p[:start]+p[end:]
# remove consistency contribution bullet
cb_start=p2.find("\\item \\textbf{A reproducible bundle-consistency measurement.}")
if cb_start!=-1:
    cb_end=p2.find("\\item \\textbf{A calibration and abstention audit.}",cb_start)
    p2=p2[:cb_start]+p2[cb_end:]
# remove consistency related-work paragraph
rw_start=p2.find("\\paragraph{Consistency and structured readout.}")
if rw_start!=-1:
    rw_end=p2.find("\\paragraph{Cascades, edge deployment, and quantization.}",rw_start)
    p2=p2[:rw_start]+p2[rw_end:]
Path("paper/shared/body_main_v2.tex").write_text(p2)
print("v2 lines",p2.count("\n"),"consistency left",p2.count("Bundle Consistency"),p2.count("consistency"))

from pathlib import Path
T=Path("paper/tables")
def patch(name, pairs):
    p=T/name
    s=p.read_text()
    for a,b in pairs:
        s=s.replace(a,b)
    p.write_text(s)
    print("patched",name)

patch("tab_dataset.tex", [
 ("\\small","\\footnotesize"),
 ("\\begin{tabular}{lrrrrr}","\\begin{tabular}{lrrrr}"),
 ("Split & Rows & Questions & Cells & Sources & $K$ \\\\","Split & Rows & Questions & Cells & $K$ \\\\"),
 ("& 11 &","&"),
 ("The frozen \\textsc{decision-v2-pilot} suite.","The frozen \\textsc{decision-v2-pilot} suite (11 public sources)."),
])
patch("tab_routing.tex", [
 ("\\small","\\footnotesize"),
 ("\\setlength{\\tabcolsep}{3.5pt}","\\setlength{\\tabcolsep}{2.5pt}"),
 ("$K$ (banking77) & \\multicolumn{4}{l}{77--78 options; 490.3 option-span tokens; 6.36 tokens/option} \\\\",
  "$K$ (banking77) & \\multicolumn{4}{l}{77--78 options} \\\\\n & \\multicolumn{4}{l}{490.3 option-span tokens; 6.36 tokens/option} \\\\"),
 ("overflow & \\multicolumn{4}{l}{50/116 rows; max 6 options dropped; context collapsed to 1 token} \\\\",
  "overflow & \\multicolumn{4}{l}{50/116 rows; context collapses to 1 token} \\\\\n & \\multicolumn{4}{l}{max 6 options dropped} \\\\"),
])
patch("tab_consistency.tex", [
 ("\\small","\\footnotesize"),
 ("\\begin{tabular}{llrrr}","\\begin{tabular}{llrrr}"),
 ("bundle exact & contradictions & macro","exact & contra. & macro"),
])
patch("tab_calibration.tex", [("\\small","\\footnotesize")])
patch("tab_systems.tex", [
 ("\\small","\\footnotesize"),
 ("\\setlength{\\tabcolsep}{3pt}","\\setlength{\\tabcolsep}{2pt}"),
])
patch("tab_diagnostics.tex", [
 ("\\small","\\footnotesize"),
 ("4 constant emitters; VOID arm","4 emitters; VOID"),
 ("0 constant emitters","0 emitters"),
 ("128 rung 0.150; readable:false","readable:false"),
 ("same, 16k","readable:false"),
])

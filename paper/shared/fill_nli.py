from pathlib import Path
files=["paper/shared/body_main_v2.tex","paper/ieee/myna_ieee.tex","paper/elsevier/myna_elsevier.tex"]
vals={"[NLI-MACRO]":"0.3278","[NLI-OVERALL]":"0.3465","[NLI-BUNDLE]":"0.3027","[NLI-B77]":"0.2931","[NLI-OTHER15]":"0.3301"}
for f in files:
    p=Path(f); s=p.read_text()
    for a,b in vals.items(): s=s.replace(a,b)
    p.write_text(s)
    print(f, "placeholders left", s.count("[NLI-"))
# make v2 the shared main body
Path("paper/shared/body_main.tex").write_text(Path("paper/shared/body_main_v2.tex").read_text())
print("shared body replaced with v2")

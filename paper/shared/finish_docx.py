import sys
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

inp, outp, page = sys.argv[1], sys.argv[2], sys.argv[3]
doc=Document(inp)
for sec in doc.sections:
    sec.top_margin=Inches(1); sec.bottom_margin=Inches(1)
    sec.left_margin=Inches(1); sec.right_margin=Inches(1)
    if page=="a4":
        sec.page_width=Inches(8.27); sec.page_height=Inches(11.69)
    else:
        sec.page_width=Inches(8.5); sec.page_height=Inches(11)

# remove theme colours everywhere and force black
for part in (doc.styles.element, doc.element.body):
    for col in part.iter(qn("w:color")):
        for att in ("w:themeColor","w:themeShade","w:themeTint"):
            if col.get(qn(att)) is not None:
                del col.attrib[qn(att)]
        col.set(qn("w:val"),"000000")

for stname in ["Normal","Title","Heading 1","Heading 2","Heading 3"]:
    try:
        st=doc.styles[stname]
        st.font.name="Times New Roman"
        st.font.color.rgb=RGBColor(0,0,0)
    except Exception:
        pass
st=doc.styles["Normal"]; st.font.size=Pt(11)
st.paragraph_format.space_after=Pt(4); st.paragraph_format.line_spacing=1.15

for p in doc.paragraphs:
    name=p.style.name if p.style is not None else ""
    for r in p.runs:
        r.font.name="Times New Roman"; r.font.color.rgb=RGBColor(0,0,0)
    if name=="Title":
        p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs: r.font.size=Pt(17); r.font.bold=True
    elif name.startswith("Heading"):
        # Use a bold Normal paragraph so every viewer renders it black.
        p.style=doc.styles["Normal"]
        for r in p.runs:
            r.font.bold=True; r.font.size=Pt(14 if name=="Heading 1" else 12)
        p.paragraph_format.space_before=Pt(10); p.paragraph_format.space_after=Pt(4)
    else:
        for r in p.runs:
            if not r.font.size: r.font.size=Pt(11)
        if len(p.text)>120:
            p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY

for sh in doc.inline_shapes:
    try:
        w,h=sh.width, sh.height
        target=Inches(6.0)
        if w>target:
            sh.width=target; sh.height=int(h*target/w)
    except Exception:
        pass

for t in doc.tables:
    try: t.style="Table Grid"
    except Exception: pass
    t.autofit=True
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after=Pt(0)
                for r in p.runs:
                    r.font.name="Times New Roman"; r.font.size=Pt(8)

doc.save(outp)
print("saved",outp,"paras",len(doc.paragraphs),"tables",len(doc.tables),"images",len(doc.inline_shapes))

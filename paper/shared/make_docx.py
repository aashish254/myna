#!/usr/bin/env python
"""Generate the editable Word manuscripts from the shared LaTeX body.

    python paper/shared/make_docx.py

The PDFs are the format-specific deliverables. The .docx files are single-column editable
copies for reviewers who need to comment: same text, same tables, same figures, no IEEE or
Elsevier two-column layout. Pandoc does the heavy lifting; this script resolves the LaTeX
cross-references and citations first, because pandoc would otherwise leave `\\ref` and
`\\cite` unresolved in the output.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHARED = REPO / "paper" / "shared"


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def macro_expand(text: str) -> str:
    for a, b in [("\\myna", "Myna"), ("\\stream", "Myna-Stream"), ("\\vsix", "Myna-v6"),
                 ("\\eg", "e.g.,"), ("\\ie", "i.e.,")]:
        text = text.replace(a, b)
    return text


def build_label_map(body: str) -> dict[str, str]:
    """label -> the number a reader sees, in document order."""
    labels: dict[str, str] = {}
    sec = 0
    app = 0
    tab = 0
    fig = 0
    in_appendix = False
    for line in body.splitlines():
        if line.strip() == "\\appendix":
            in_appendix = True
            continue
        m = re.match(r"\s*\\section\*?\{", line)
        if m:
            if in_appendix:
                app += 1
                num = chr(ord("A") + app - 1)
            else:
                sec += 1
                num = str(sec)
            lab = re.search(r"\\label\{([^}]*)\}", line)
            if lab:
                labels[lab.group(1)] = num
            continue
        m = re.match(r"\s*\\begin\{(wide)?table\}", line)
        if m:
            tab += 1
        m = re.match(r"\s*\\begin\{(wide)?figure\}", line)
        if m:
            fig += 1
        lab = re.search(r"\\label\{((?:tab|fig):[^}]*)\}", line)
        if lab:
            key = lab.group(1)
            labels[key] = str(tab) if key.startswith("tab:") else str(fig)
    return labels


def resolve_refs(body: str, labels: dict[str, str]) -> str:
    def repl(m):
        key = m.group(1)
        if key not in labels:
            raise SystemExit(f"unresolved reference: {key}")
        return labels[key]

    body = re.sub(r"\\ref\{([^}]*)\}", repl, body)
    return body.replace("~\\ref", " \\ref")


def number_sections(body: str, labels: dict[str, str]) -> str:
    out = []
    sec = 0
    app = 0
    in_appendix = False
    for line in body.splitlines():
        if line.strip() == "\\appendix":
            in_appendix = True
            out.append("")
            continue
        m = re.match(r"\s*\\section\{(.*)\}\s*$", line)
        if m:
            if in_appendix:
                app += 1
                num = chr(ord("A") + app - 1)
            else:
                sec += 1
                num = str(sec)
            out.append(f"\\section*{{{num}. {m.group(1)}}}")
            continue
        out.append(line)
    return "\n".join(out)


def number_captions(body: str) -> str:
    """Prefix each caption with its object number, the way the PDFs show them."""
    tab = fig = 0
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"\s*\\begin\{(wide)?table\}", line):
            tab += 1
        if re.match(r"\s*\\begin\{(wide)?figure\}", line):
            fig += 1
        m = re.search(r"\\caption\{(.*)\}\s*$", line)
        if m:
            if tab and not fig:
                pass
    # Second pass with an explicit object-type scan, because a table and a figure can
    # appear between one another and the counters above are positional.
    out: list[str] = []
    kind = None
    tab = fig = 0
    for line in lines:
        if re.match(r"\s*\\begin\{(wide)?table\}", line):
            tab += 1
            kind = "table"
        if re.match(r"\s*\\begin\{(wide)?figure\}", line):
            fig += 1
            kind = "figure"
        m = re.search(r"\\caption\{(.*)\}\s*$", line)
        if m and kind:
            prefix = "Table" if kind == "table" else "Figure"
            n = tab if kind == "table" else fig
            line = line[:m.start()] + f"\\caption{{{prefix} {n}. {m.group(1)}}}" + line[m.end():]
        out.append(line)
    return "\n".join(out)


def parse_bib(path: Path) -> dict[str, str]:
    """key -> a one-line IEEE-ish reference, from the fields we actually carry."""
    entries: dict[str, str] = {}
    raw = path.read_text()
    for block in re.split(r"\n@", raw):
        m = re.search(r"\{([^,]+),", block)
        if not m:
            continue
        key = m.group(1).strip()
        fields = dict(re.findall(r"(\w+)\s*=\s*\{(.*?)\},?\n", block, flags=re.S))
        author = " and ".join(fields.get("author", "").split(" and "))
        author = re.sub(r"\{|\}", "", author)
        title = re.sub(r"\{|\}", "", fields.get("title", ""))
        venue = fields.get("journal") or fields.get("booktitle") or fields.get("publisher", "")
        venue = re.sub(r"\{|\}", "", venue)
        year = fields.get("year", "").strip()
        doi = fields.get("doi", "").strip()
        bits = [f"{author}. {title}." if author else f"{title}."]
        if venue:
            bits.append(venue)
        if year:
            bits.append(year)
        tail = ", ".join(bits[1:])
        ref = bits[0] + (" " + tail + "." if tail else "")
        if doi:
            ref += f" doi:{doi}."
        entries[key] = ref
    return entries


def resolve_cites(body: str, entries: dict[str, str]) -> tuple[str, list[str]]:
    order: list[str] = []

    def repl(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        nums = []
        for k in keys:
            if k not in order:
                order.append(k)
            nums.append(order.index(k) + 1)
        return "[" + ", ".join(str(n) for n in sorted(set(nums))) + "]"

    return re.sub(r"\\cite\{([^}]*)\}", repl, body), order


def strip_labels(text: str) -> str:
    return re.sub(r"\\label\{[^}]*\}\n?", "", text)


def environments(text: str) -> str:
    text = re.sub(r"\\begin\{widefigure\}\[[^\]]*\]", "\\\\begin{figure}", text)
    text = re.sub(r"\\begin\{widetable\}\[[^\]]*\]", "\\\\begin{table}", text)
    text = text.replace("\\begin{widefigure}", "\\begin{figure}")
    text = text.replace("\\end{widefigure}", "\\end{figure}")
    text = text.replace("\\begin{widetable}", "\\begin{table}")
    text = text.replace("\\end{widetable}", "\\end{table}")
    text = re.sub(r"\\includegraphics(\[[^\]]*\])?\{([^}]*)\}",
                  lambda m: f"\\includegraphics[width=0.9\\linewidth]{{paper/figures/{m.group(2)}}}",
                  text)
    return text


def main() -> None:
    body = strip_comments((SHARED / "body_main.tex").read_text())
    appendix = strip_comments((SHARED / "body_appendix.tex").read_text())
    body = macro_expand(body + "\n\n" + appendix)
    # `\label` sits on the line after its `\section` / `\caption` in the source; join the
    # pair so the numbering pass can see which object a label belongs to.
    body = re.sub(r"(\\(?:section|caption)\{[^}]*\})\s*\n\s*(\\label\{[^}]*\})",
                  r"\1\2", body)
    labels = build_label_map(body)
    body = resolve_refs(body, labels)
    body = number_sections(body, labels)
    body = number_captions(body)
    body = strip_labels(body)
    body = environments(body)
    entries = parse_bib(REPO / "paper" / "ieee" / "refs.bib")
    body, order = resolve_cites(body, entries)
    refs = "\n\n\\section*{References}\n\n" + "\n\n".join(
        f"[{i+1}] {entries[k]}" for i, k in enumerate(order))
    title = ("Myna: A Measurement Study of Typed Agent Decision Engines")
    abstract = (SHARED / "abstract.txt").read_text().strip()
    tex = f"""\\documentclass[11pt]{{article}}
\\usepackage{{amsmath,amssymb,graphicx,booktabs,multirow,url}}
\\title{{{title}}}
\\author{{Aashish Kumar Mahato\\\\Independent Researcher, Kathmandu, Nepal\\\\
\\small Graduate of Vellore Institute of Technology (VIT), Vellore, India. This work was conducted
independently after graduation.}}
\\date{{November 2026}}
\\begin{{document}}
\\maketitle
\\begin{{abstract}}
{abstract}
\\end{{abstract}}
{body}
{refs}
\\end{{document}}
"""
    tmp = REPO / "paper" / "shared" / "_docx_source_gen.tex"
    tmp.write_text(tex)
    for venue, page in (("ieee", "letter"), ("elsevier", "a4")):
        raw = REPO / "paper" / venue / "_raw.docx"
        out = REPO / "paper" / venue / ("myna_" + venue + ".docx")
        subprocess.run(["pandoc", str(tmp), "-f", "latex", "-t", "docx",
                        f"--resource-path={REPO}", "-o", str(raw)], check=True, cwd=str(REPO))
        subprocess.run(["uv", "run", "--with", "python-docx", "python",
                        str(SHARED / "finish_docx.py"), str(raw), str(out), page],
                       check=True, cwd=str(REPO))
        raw.unlink(missing_ok=True)
        print("wrote", out)


if __name__ == "__main__":
    main()

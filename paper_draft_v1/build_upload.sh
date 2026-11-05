#!/bin/sh
# Assemble the upload bundle: PDF, a flat source archive, and the plain-text form fields.
set -e
cd "$(dirname "$0")"
mkdir -p upload
cp myna_tmlr.pdf upload/myna_tmlr.pdf

# abstract and author fields, taken from the built source so they cannot drift
python3 - <<'PY'
from pathlib import Path
t = Path("myna_tmlr.tex").read_text()
a = t.index("\\begin{abstract}") + len("\\begin{abstract}")
b = t.index("\\end{abstract}")
import re
text = " ".join(t[a:b].split())
text = text.replace("\\vsix{}", "Myna-v6").replace("\\stream{}", "Myna-Stream")
text = text.replace("\\myna{}", "Myna").replace("\\,", " ").replace("\\%", "%")
text = re.sub(r"\\texttt\{([^}]*)\}", r"\1", text)
text = re.sub(r"\$([^$]*)\$", r"\1", text)
text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
text = re.sub(r"[{}]", "", text)
Path("upload/abstract.txt").write_text(text + "\n")
PY

rm -rf /tmp/myna_pkg && mkdir -p /tmp/myna_pkg
cp myna_tmlr.tex draft_body_main.tex draft_body_appendix.tex refs.bib tmlr.sty /tmp/myna_pkg/
cp ../paper/figures/*.pdf ../paper/figures/*.png /tmp/myna_pkg/ 2>/dev/null || true
rm -f upload/myna_supplement.zip
( cd /tmp/myna_pkg && zip -q -r "$OLDPWD/upload/myna_supplement.zip" . )
ls -la upload

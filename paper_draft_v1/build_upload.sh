#!/bin/sh
# Assemble the upload bundle: the PDF, the sources, and the plain-text form fields.
set -e
cd "$(dirname "$0")"
mkdir -p upload
cp myna_tmlr.pdf upload/myna_tmlr.pdf
rm -f upload/myna_supplement.zip
zip -q -r upload/myna_supplement.zip \
    myna_tmlr.tex draft_body_main.tex draft_body_appendix.tex refs.bib tmlr.sty \
    ../paper/figures -x '*.pdf.bak' >/dev/null
ls -la upload

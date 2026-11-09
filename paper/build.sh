#!/bin/bash
# Build main.pdf and supplement.pdf with cross-references between them .
# The \newlabel lines of each .aux go to main-labels.tex / supp-labels.tex, which the other
# document reads (\extlabels in the preamble); these files are also synced to Overleaf, so its single-document build resolves them.
set -e
cd "$(dirname "$0")"
for i in 1 2; do
  latexmk -pdf -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
  grep '^\\newlabel' main.aux > main-labels.tex
  latexmk -pdf -interaction=nonstopmode supplement.tex >/dev/null 2>&1 || true
  grep '^\\newlabel' supplement.aux > supp-labels.tex
done
latexmk -pdf -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
for d in main supplement; do
  echo "$d: $(pdfinfo $d.pdf | awk '/Pages/{print $2}') pages, $(grep -c 'Reference.*undefined\|Citation.*undefined' $d.log) undefined"
done

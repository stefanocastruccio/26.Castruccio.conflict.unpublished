#!/bin/bash
# Copy the paper (tex, bib, tables, figures, class files) to the Overleaf git repo and push.
set -e
S=/home/user/26.castruccio.conflict.unpublished/paper
O=/home/user/26.castruccio.conflicts.unpublished
cd "$O" && git pull -q --rebase
cp "$S"/main.tex "$S"/supplement.tex "$S"/refs.bib .
mkdir -p sections tables figures data
cp "$S"/sections/*.tex sections/; cp "$S"/tables/*.tex tables/; cp "$S"/figures/*.png figures/
cp "$S"/data/*.geojson data/ 2>/dev/null || true
git add -A
if git diff --cached --quiet; then echo "Overleaf: nothing to sync"; exit 0; fi
git commit -qm "Sync from the code repository ($(git -C "$S/.." rev-parse --short HEAD))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H85kxLsrzzXKJMbCCVADs7"
git push -q 2>&1 | grep -vi "moved\|location" || true
echo "Overleaf synced: $(git rev-parse --short HEAD)"

#!/bin/bash
# Download the released checkpoints from Hugging Face into this folder and check them against SHA256SUMS.
# The Hugging Face repository (liguang0115/justact) still uses the method's earlier name JustAct in nine file names; they are
# renamed here to the OnlyAct names that SHA256SUMS and the code use. The file contents are unchanged.
#   bash checkpoints/download.sh
set -euo pipefail
cd "$(dirname "$0")"
hf download liguang0115/justact --local-dir . --exclude README.md .gitattributes
find . -path ./.cache -prune -o -name '*justact*' -print | sort -r | while read -r f; do
  new="$(dirname "$f")/$(basename "$f" | sed 's/justact/onlyact/g')"
  if [ -e "$new" ]; then rm -rf "$f"; else mv "$f" "$new"; fi
done
sha256sum -c SHA256SUMS

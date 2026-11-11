#!/bin/bash
# Download the released checkpoints from Hugging Face (liguang0115/onlyact) into this folder and check them against SHA256SUMS.
#   bash checkpoints/download.sh
set -euo pipefail
cd "$(dirname "$0")"
hf download liguang0115/onlyact --local-dir . --exclude README.md .gitattributes
sha256sum -c SHA256SUMS

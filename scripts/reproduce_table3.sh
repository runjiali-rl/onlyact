#!/bin/bash
# Table 3: downstream fine-tuning of the three pretrained world models on DMLab, Minecraft, Habitat and nuScenes.
# Per domain and arm: generation before fine-tuning, an 800-update fine-tuning run (one 24 GB GPU and 24-26 GiB of free
# host memory; about 4 h on an RTX 4090), generation after 200 and 800 updates; then scoring and the table.
# STEPS="0" runs only the generation before fine-tuning (no training).
# Environment variables: DATA (DATA/downstream/<domain>/, prepared as in downstream/README.md), RUNS [runs], PY_WM [python],
#   DOMAINS [dmlab minecraft habitat nuscenes], STEPS [0 200 800].
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA:?set DATA}"
RUNS=${RUNS:-runs}; PY_WM=${PY_WM:-python}
DOMAINS=${DOMAINS:-"dmlab minecraft habitat nuscenes"}; STEPS=${STEPS:-"0 200 800"}
ARMS="adaworld_lam_retrained dreamdojo_lam_retrained onlyact"
O=$RUNS/table3
for D in $DOMAINS; do
  for A in $ARMS; do
    for N in $STEPS; do
      out=$O/cells/$D/${A}_step$N; [ -f $out/summary.json ] && continue
      if [ $N = 0 ]; then
        $PY_WM -m downstream.generate --config downstream/configs/$D.json --arm $A --data_root $DATA/downstream/$D --out $out
      else
        [ -f $O/train/$D/$A/step000800.safetensors ] || \
          $PY_WM -m downstream.train --config downstream/configs/$D.json --arm $A --data_root $DATA/downstream/$D --out $O/train/$D/$A
        $PY_WM -m downstream.generate --config downstream/configs/$D.json --arm $A --data_root $DATA/downstream/$D \
          --finetuned $O/train/$D/$A/step000$N.safetensors --out $out
      fi
    done
  done
  $PY_WM -m downstream.score --config downstream/configs/$D.json --cells $O/cells/$D --data_root $DATA/downstream/$D --out $O/scores/$D.json
done
if [ "$STEPS" = "0 200 800" ]; then
  $PY_WM -m downstream.table --scores $O/scores --out $O/table3_source.json
  $PY_WM tables/table3_downstream.py --source $O/table3_source.json --out $O/table3.tex
fi

#!/bin/bash
# Train every decoder of Table 4: 13 datasets x 4 representations x 4 seeds = 208 runs, one after another
# (about 10 GPU-hours on one RTX 4090). Build the tables afterwards with tables/table4_leakage.py.
#
#   EMB=<embeddings root: $EMB/<representation>/<dataset>/<task_id>.npy> \
#   FRAMES128=<128x128 frames root: $FRAMES128/<dataset>/<clip-list relative path>> \
#   OUT=runs/table4 bash leakage/run_table4.sh
#
# DATASETS, REPS and SEEDS restrict the loop; PYTHON selects the interpreter. Finished runs (result.json present) are
# skipped, so the script can be restarted; a run that stopped half-way must be deleted first.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${EMB:?set EMB to the embeddings root}" "${FRAMES128:?set FRAMES128 to the 128x128 frames root}" "${OUT:?set OUT to the output root}"
PYTHON=${PYTHON:-python}
DATASETS=${DATASETS:-"libero droid rlbench robocasa droid_iprl metaworld utaustin_mutex vp2_robosuite vp2_robodesk spatialvid_nav arkitscenes realestate10k procgen_plan"}
REPS=${REPS:-"adaworld_lam_retrained dreamdojo_lam_retrained vjepa2_frozen justact"}
SEEDS=${SEEDS:-"0 1 2 3"}

for d in $DATASETS; do
  for r in $REPS; do
    for s in $SEEDS; do
      run="$OUT/$d/$r/seed$s"
      if [ -f "$run/result.json" ]; then echo "done already: $run"; continue; fi
      if [ ! -d "$EMB/$r/$d" ]; then echo "no embeddings in $EMB/$r/$d, skipped"; continue; fi
      echo "[$(date +%H:%M:%S)] $d $r seed $s"
      "$PYTHON" -m leakage.train --dataset "$d" --representation "$r" --seed "$s" \
        --clips "data/probe/$d.json" --data_root "$FRAMES128/$d" --emb_dir "$EMB/$r/$d" --out_dir "$run"
    done
  done
done

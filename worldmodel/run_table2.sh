#!/usr/bin/env bash
# Table 2: generate the rollouts of the given world models on all ten evaluation cells, then score them.
#
#   OUT=runs/table2 EMB=/data/embeddings DATA_LIBERO=... DATA_RLBENCH=... DATA_SPATIALVID=... DATA_NUSCENES=... \
#       bash worldmodel/run_table2.sh [no_action adaworld_lam dreamdojo_lam onlyact]
#
# Each (world model, cell) is one generate.py process with its own seed, so finished cells (summary.json present) are
# skipped on a re-run and different world models can run in parallel (one process needs about 11 GB of GPU memory).
# Optional: CELLS="libero_self libero_transfer" restricts the cells; CKPT_DIR, BASE_CKPT override checkpoint locations.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$HERE")
: "${OUT:?set OUT}" "${EMB:?set EMB}" "${DATA_LIBERO:?}" "${DATA_RLBENCH:?}" "${DATA_SPATIALVID:?}" "${DATA_NUSCENES:?}"
PY=${PYTHON:-python}
CKPT_DIR=${CKPT_DIR:-$REPO/checkpoints/worldmodel}
BASE_CKPT=${BASE_CKPT:-$REPO/checkpoints/adaworld/adaworld.safetensors}
CELLS=${CELLS:-"libero_self libero_transfer rlbench_self rlbench_transfer spatialvid_self spatialvid_transfer nuscenes_dev_self nuscenes_dev_transfer nuscenes_conf_self nuscenes_conf_transfer"}
ARMS=${*:-"no_action adaworld_lam dreamdojo_lam onlyact"}
declare -A ROOT=([libero]=$DATA_LIBERO [rlbench]=$DATA_RLBENCH [spatialvid_nav]=$DATA_SPATIALVID [nuscenes_mini]=$DATA_NUSCENES)

for arm in $ARMS; do
  for cell in $CELLS; do
    dir=$OUT/$arm/$cell
    if [ -f "$dir/summary.json" ]; then echo "[table2] $arm/$cell done"; continue; fi
    ds=$($PY -c "import json, sys; print(json.load(open(sys.argv[1]))['dataset'])" "$REPO/data/worldmodel/$cell.json")
    mkdir -p "$dir"
    $PY "$HERE/generate.py" --arm "$arm" --cell "$REPO/data/worldmodel/$cell.json" --data_root "${ROOT[$ds]}" --emb_root "$EMB" \
      --out_dir "$dir" --base_ckpt "$BASE_CKPT" --ckpt "$CKPT_DIR/$arm.safetensors" \
      --whitening_stats "$CKPT_DIR/onlyact_whitening.safetensors" 2>&1 | tee "$dir/generate.log"
  done
done

$PY "$HERE/score.py" --runs_root "$OUT" --arms $ARMS --cells $CELLS \
  --data_root libero="$DATA_LIBERO" rlbench="$DATA_RLBENCH" spatialvid_nav="$DATA_SPATIALVID" nuscenes_mini="$DATA_NUSCENES"

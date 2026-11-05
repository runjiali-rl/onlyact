#!/bin/bash
# Table 2: action-conditioned world model (self PSNR and transfer direction), 4 world models x 10 cells = 1,120 rollouts
# (about 24 GPU-hours on one RTX 4090). Run scripts/reproduce_table1.sh first for LIBERO, RLBench and SpatialVID: their
# evaluation clips are clips of the probe lists, so their embeddings are already in RUNS/emb.
# Environment variables: DATA (DATA/probe/<dataset>/ and DATA/nuscenes_mini/), CKPT [checkpoints], RUNS [runs],
#   BASE_CKPT [CKPT/adaworld/adaworld.safetensors, the released AdaWorld weights], PY_ENC, PY_WM [python].
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA:?set DATA}"
CKPT=${CKPT:-checkpoints}; RUNS=${RUNS:-runs}; PY_ENC=${PY_ENC:-python}; PY_WM=${PY_WM:-python}
EMB=$RUNS/emb; NUS=$DATA/nuscenes_mini

# nuScenes evaluation clips (the transfer sources are among the self clips); the paper embedded them with a 4-thread LoRA merge
for c in nuscenes_dev_self nuscenes_conf_self; do
  $PY_ENC -m onlyact.extract --manifest data/worldmodel/$c.json --data_root $NUS --ckpt $CKPT/onlyact --out_dir $EMB/onlyact/nuscenes_mini \
    --merge_threads 4
  for a in adaworld dreamdojo; do
    $PY_WM baselines/extract.py --arch $a --manifest data/worldmodel/$c.json --data_root $NUS \
      --ckpt $CKPT/${a}_lam_retrained.safetensors --out_dir $EMB/${a}_lam_retrained/nuscenes_mini
  done
done

PYTHON=$PY_WM OUT=$RUNS/table2 EMB=$EMB CKPT_DIR=$CKPT/worldmodel BASE_CKPT=${BASE_CKPT:-$CKPT/adaworld/adaworld.safetensors} \
  DATA_LIBERO=$DATA/probe/libero DATA_RLBENCH=$DATA/probe/rlbench DATA_SPATIALVID=$DATA/probe/spatialvid_nav DATA_NUSCENES=$NUS \
  bash worldmodel/run_table2.sh no_action adaworld_lam dreamdojo_lam onlyact
$PY_WM tables/table2_worldmodel.py --scores $RUNS/table2/scores.json --out $RUNS/table2/table2.tex

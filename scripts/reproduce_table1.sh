#!/bin/bash
# Table 1: linear probing of action information on 13 datasets.
#   1. embeddings of every clip with OnlyAct, frozen V-JEPA 2, the two retrained latent-action models and the random control
#   2. a linear probe per (representation, dataset, seed 0 / 1 / 2)
#   3. the LaTeX table
# Environment variables (defaults in brackets):
#   DATA       data root written by data/probe/prepare/ (see data/probe/README.md): frames
#              DATA/probe/<dataset>/ laid out as data/probe/<dataset>.json                      (required)
#   EVAL_DATA  action targets EVAL_DATA/probe/<dataset>/<target>/<task_id>.npy, also written by
#              data/probe/prepare/                                                             [DATA/eval_data]
#   CKPT       checkpoints folder [checkpoints]          RUNS  output folder [runs]
#   PY_ENC     python of environment 1 (requirements/encoder.txt) [python]
#   PY_WM      python of environment 2 (requirements/worldmodel.txt) [python]
#   DATASETS   subset of datasets [all 13]
# Every step skips outputs that already exist, so the script can be restarted.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA:?set DATA}"
EVAL_DATA=${EVAL_DATA:-$DATA/eval_data}
CKPT=${CKPT:-checkpoints}; RUNS=${RUNS:-runs}; PY_ENC=${PY_ENC:-python}; PY_WM=${PY_WM:-python}
DATASETS=${DATASETS:-"vp2_robosuite vp2_robodesk procgen_plan utaustin_mutex spatialvid_nav arkitscenes realestate10k droid droid_iprl robocasa metaworld rlbench libero"}
REPS="onlyact vjepa2_frozen adaworld_lam_retrained dreamdojo_lam_retrained random"
EMB=$RUNS/emb; OUT=$RUNS/table1; mkdir -p $OUT/logs

for d in $DATASETS; do
  L=data/probe/$d.json; R=$DATA/probe/$d
  $PY_ENC -m onlyact.extract --manifest $L --data_root $R --ckpt $CKPT/onlyact --out_dir $EMB/onlyact/$d
  $PY_ENC -m onlyact.extract --manifest $L --data_root $R --frozen --out_dir $EMB/vjepa2_frozen/$d
  for a in adaworld dreamdojo; do
    $PY_WM baselines/extract.py --arch $a --manifest $L --data_root $R --ckpt $CKPT/${a}_lam_retrained.safetensors \
      --out_dir $EMB/${a}_lam_retrained/$d
  done
  [ -d $EMB/random/$d ] || $PY_ENC -m probe.make_random --clips $L --ref_dir $EMB/vjepa2_frozen/$d --out_dir $EMB/random/$d
done

for d in $DATASETS; do
  T=$($PY_ENC -c "import json; print(json.load(open('data/probe/datasets.json'))['$d']['target'])")
  for r in $REPS; do for s in 0 1 2; do
    o=$OUT/$r/${d}_seed$s.json; [ -f $o ] && continue
    $PY_ENC -m probe.probe --dataset $d --clips data/probe/$d.json --actions_dir $EVAL_DATA/probe/$d/$T \
      --emb_dir $EMB/$r/$d --seed $s --output $o > $OUT/logs/${r}_${d}_seed$s.log 2>&1
    echo "$r $d seed $s: $(grep -o 'Test R2=[-0-9.]*\|macro-F1=[-0-9.]*' $OUT/logs/${r}_${d}_seed$s.log | tail -1)"
  done; done
done

$PY_ENC tables/table1_probe.py --results $OUT --out $OUT/table1.tex

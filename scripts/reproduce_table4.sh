#!/bin/bash
# Table 4: scene leakage, 13 datasets x 4 representations x 4 decoder seeds = 208 decoders (about 10 GPU-hours on one
# RTX 4090). Needs the embeddings of scripts/reproduce_table1.sh and the 128x128 frame copies DATA/leakage128/<dataset>/
# laid out as data/probe/<dataset>.json. Writes the table and the appendix table (per-dataset SSIM and L1).
# Environment variables: DATA, RUNS [runs], PY_ENC [python].
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA:?set DATA}"
RUNS=${RUNS:-runs}; PY_ENC=${PY_ENC:-python}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} PYTHON=$PY_ENC EMB=$RUNS/emb FRAMES128=$DATA/leakage128 OUT=$RUNS/table4/runs bash leakage/run_table4.sh
$PY_ENC tables/table4_leakage.py --results_root $RUNS/table4/runs --out_dir $RUNS/table4

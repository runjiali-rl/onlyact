# Action-conditioned world model (Table 2)

Table 2 asks whether an action representation can drive a video world model. Four AdaWorld world models are trained with
the same recipe and data and differ only in the per-transition action vector they are conditioned on:

| world model (`--arm`) | action vector | width | config |
|---|---|---|---|
| `no_action` | all zeros (lower bound: the model sees only the context frames) | 32 | `configs/no_action.yaml` |
| `adaworld_lam` | AdaWorld-LAM (retrained), `adaworld_lam_retrained` | 32 | `configs/adaworld_lam.yaml` |
| `dreamdojo_lam` | DreamDojo-LAM (retrained), `dreamdojo_lam_retrained` | 32 | `configs/dreamdojo_lam.yaml` |
| `justact` | JustAct, `justact`, with per-provider whitening | 640 | `configs/justact.yaml` |

Each model is evaluated on ten cells (`data/worldmodel/<cell>.json`): a **self** cell rolls a clip out from its first 6
frames with its own actions and reports PSNR (dB) of the 19 predicted frames against the ground truth; a **transfer**
cell rolls the target clip out with the actions of another clip (the source) and reports the **transfer direction**, the
cosine between the optical-flow motion of the generated video and that of the source (1 = follows the source,
0 = unrelated). On LIBERO, RLBench and SpatialVID each source was chosen to move opposite to its target, so a model that
ignores the action and continues the target's own motion scores below 0; on nuScenes the source is a clip of another
driving log with no motion selection.

| dataset | self cell (clips) | transfer cell (pairs) | data root (`--data_root`) |
|---|---|---|---|
| LIBERO | `libero_self` (40) | `libero_transfer` (40, cross-scene) | `$DATA/probe/libero`, the frames of `data/probe/libero.json` |
| RLBench | `rlbench_self` (40) | `rlbench_transfer` (40, cross-task-family) | `$DATA/probe/rlbench`, the frames of `data/probe/rlbench.json` |
| SpatialVID | `spatialvid_self` (20) | `spatialvid_transfer` (20, cross-scene) | `$DATA/probe/spatialvid_nav`, the frames of `data/probe/spatialvid_nav.json` |
| nuScenes development | `nuscenes_dev_self` (28) | `nuscenes_dev_transfer` (28, cross-log) | `$DATA/nuscenes_mini`, the nuScenes v1.0-mini root (`samples/`, `sweeps/`) |
| nuScenes confirmation | `nuscenes_conf_self` (12) | `nuscenes_conf_transfer` (12, cross-log) | same |

Every cell file stores the clips (frame paths relative to the data root; nuScenes also stores each frame's sha256), the
ordered rollout list `pairs` (`source == target` in self cells) and a `description` of how the clips were chosen. All
clips were chosen from ground-truth frames only, before any model was run on them, and none is in the training pool.

## Data

No frames are distributed with this repository. The LIBERO, RLBench and SpatialVID frames are the Table 1 frames,
downloaded and prepared with `data/probe/prepare/` as described in [`data/probe/README.md`](../data/probe/README.md).

nuScenes v1.0-mini: create an account on https://www.nuscenes.org/nuscenes#download (registration required), accept the
nuScenes Terms of Use (https://www.nuscenes.org/terms-of-use), download `v1.0-mini.tgz` (4.2 GB) and extract it:

```bash
mkdir -p $DATA/nuscenes_mini && tar -xf v1.0-mini.tgz -C $DATA/nuscenes_mini
```

This gives `$DATA/nuscenes_mini/{maps,samples,sweeps,v1.0-mini}/`. The cells read only the front-camera frames,
`samples/CAM_FRONT/` (key frames) and `sweeps/CAM_FRONT/` (intermediate frames), and do not need the nuScenes devkit.
`scripts/reproduce_table2.sh` reads `$DATA/probe/<dataset>/` and `$DATA/nuscenes_mini/`.

## Files

```
worldmodel/
  train.py              train one world model (LoRA on the AdaWorld UNet, 20,000 steps)
  generate.py           rollouts of one world model on one cell -> results.jsonl, grid_XX.png, summary.json
  score.py              self PSNR and transfer direction for all cells -> scores.json and the table
  compute_whitening.py  per-provider whitening statistics of the JustAct input
  run_table2.sh         generate + score all cells for the given world models
  loading.py            AdaWorld base weights (action pathway zeroed), LoRA, released checkpoints
  clip_lists.py         clip-list reader (relative paths, optional .json.gz)
  configs/              the four world-model configs
  vwm/                  AdaWorld world-model code (Apache-2.0, see NOTICE_ADAWORLD.md), plus
                        vwm/modules/encoders/external_action.py (action embedders) and vwm/data/external_action.py
  tools/                export_checkpoint.py, decode_pool_videos.py, make_cell_list.py, make_train_list.py
```

## Requirements

Python 3.12, one CUDA GPU with at least 12 GB for generation and 24 GB for training. Versions used for the paper:
`torch==2.10.0` (CUDA 12.8), `pytorch-lightning==2.6.1`, `xformers==0.0.35`, `peft==0.18.1`, `piq==0.8.0`,
`open_clip_torch==3.3.0`, `opencv-python==4.13.0.92`, `numpy==2.3.5`, `scipy==1.17.1`, `omegaconf==2.3.0`,
`safetensors==0.7.0`, `kornia==0.8.2`, `einops==0.8.2`, `pillow==12.0.0`, `decord==0.6.0` (only for
`tools/decode_pool_videos.py`). Run the commands from `worldmodel/`.

## Checkpoints

| file | content | size | sha256 |
|---|---|---|---|
| `checkpoints/adaworld/adaworld.safetensors` | released AdaWorld weights, download from https://huggingface.co/Little-Podi/AdaWorld (not redistributed) | 11,463,790,956 B | `d128a44d2b7b3467fbcb56997f0d9d73fae8148a6faad7ed7765e8b459096673` |
| `checkpoints/worldmodel/no_action.safetensors` | LoRA (rank 128) + `label_emb`, 516 tensors, 55.7 M parameters | 222,949,304 B | `3318e60eb6b93cd3ce96214a63502df5257d4d6af22f4a4e49e0ede43d5f580c` |
| `checkpoints/worldmodel/adaworld_lam.safetensors` | same layout | 222,949,304 B | `be632e503fc301b657c01d9e4e496e43c413a7bd37fc99cb48ae5027954694d4` |
| `checkpoints/worldmodel/dreamdojo_lam.safetensors` | same layout | 222,949,304 B | `a7a95651b928413169299a9824d0605488328aec51807eddb450bdff412fda3a` |
| `checkpoints/worldmodel/justact.safetensors` | same layout, 640-d action input, 61.5 M parameters | 245,985,256 B | `d9bdd027aea6dc900b112c3a4a68fe8a07def7174e59aca91649b5ad34fd0c4c` |
| `checkpoints/worldmodel/justact_whitening.safetensors` | per-provider whitening statistics of JustAct | 3,292,672 B | `6034ecdba90bf51f91a2dffbeb1dc04ded62ee254ab5fd81d431c48965694bd0` |

A world-model checkpoint holds only the trained tensors; the rest of the model is the AdaWorld base. When a model is
built, `open_clip` downloads OpenCLIP ViT-H-14 (`laion2b_s32b_b79k`, 3.7 GB, overwritten by the AdaWorld weights) and
`piq` downloads the LPIPS weights once.

## Action embeddings

The world models read precomputed action vectors, `<emb_root>/<representation>/<dataset>/<task_id>.npy`, float16, shape
`(n_frames - 1, D)`, row j = frame pair (j, j + 1). Dataset names: `libero`, `rlbench`, `spatialvid_nav`,
`nuscenes_mini` (the `dataset` field of each cell file) and `worldmodel_pool` for training. The LIBERO, RLBench and
SpatialVID clips of the cells are clips of `data/probe/{libero,rlbench,spatialvid_nav}.json` with identical frame lists,
so the Table 1 embeddings can be reused. Otherwise extract them from the cell files, e.g.

```bash
python -m justact.extract --manifest data/worldmodel/nuscenes_dev_self.json --data_root $NUSCENES \
    --ckpt checkpoints/justact --out_dir $EMB/justact/nuscenes_mini
python baselines/extract.py --arch dreamdojo --manifest data/worldmodel/nuscenes_dev_self.json --data_root $NUSCENES \
    --ckpt checkpoints/dreamdojo_lam_retrained.safetensors --out_dir $EMB/dreamdojo_lam_retrained/nuscenes_mini
```

(from the repository root; see `justact/` and `baselines/README.md`). The no-action model reads no embeddings.

## Generating and scoring Table 2

One cell of one world model:

```bash
python generate.py --arm justact --cell ../data/worldmodel/libero_transfer.json --data_root $LIBERO \
    --emb_root $EMB --out_dir runs/table2/justact/libero_transfer
```

Paper settings are the defaults: seed 32, 6 context frames, 19 predicted frames, 25 Euler steps of AdaWorld's sampler
(shifted EDM discretisation), classifier-free guidance 1.1 (the unconditional branch zeroes the image conditions and the
action), 256 x 256 pixels, context noise level 0.1. Checkpoints default to `checkpoints/...` in the repository root
(`--base_ckpt`, `--ckpt`, `--whitening_stats`).

All 40 runs and the table:

```bash
OUT=runs/table2 EMB=$EMB DATA_LIBERO=$LIBERO DATA_RLBENCH=$RLBENCH DATA_SPATIALVID=$SPATIALVID DATA_NUSCENES=$NUSCENES \
    bash run_table2.sh                     # or one world model per GPU: bash run_table2.sh justact
python score.py --runs_root runs/table2 --data_root libero=$LIBERO rlbench=$RLBENCH spatialvid_nav=$SPATIALVID nuscenes_mini=$NUSCENES
```

Outputs of `generate.py` in `<runs_root>/<arm>/<cell>/`:

| file | content |
|---|---|
| `results.jsonl` | one line per rollout in cell order: `source`, `target`, `psnr`, `ssim`, `lpips` (means over the 19 predicted frames, floating point, [0, 1] frames, centre crop and bicubic resize to 256), `psnr_per_frame`, `sharp_per_frame` (Laplacian-variance ratio prediction / ground truth) |
| `grid_XX.png` | rollout XX: ground-truth frames 6-24 (top row) and predicted frames (bottom row), 256 px each |
| `summary.json` | means and standard deviations over the cell, per-frame means, settings and checkpoint sha256 |

`score.py` writes `<runs_root>/scores.json` and prints the table: self cells report the mean of `psnr` over
`results.jsonl`; transfer cells report the mean `dir_src` over the scorable pairs. `dir_src` is computed from the grids
and the source clip's frames: Farneback optical flow at 128 x 128, per-frame mean flow over pixels moving more than 0.3 px
(zero if fewer than 1 % move), mean cosine over the frames where both motion vectors exceed 0.2 px; a pair with fewer
than 3 such frames is unscored (`n_scored` counts the scored pairs). It also stores `dir_tgt` (the same against the target)
and the PSNR of transfer rollouts against the target.

Reproducibility. The random stream is seeded once per `generate.py` process and consumed clip after clip, so identical
numbers need the same cell file and order (`--limit N` reproduces the first N rollouts of a cell). On the same GPU type and
software versions the rollouts are bit-identical; other GPUs or library versions give small differences. The no-action
model ignores the source clip, so its transfer rollouts equal its self rollouts of the same targets.

Expected time on one RTX 4090: about 74 s per rollout, 1,120 rollouts in total (about 24 GPU-hours including model
loading, about 11 GB of GPU memory per process, so two processes fit on a 24 GB card); scoring takes 10-15 CPU-minutes.

## Training

The training pool `data/worldmodel/train_pool.json.gz` lists 19,946 clips of 25 consecutive frames from nine datasets
(VPT 5,000, CALVIN 3,000, SpatialVID 3,000, EgoDex 2,000, AgiBot 2,000, Something-Something v2 1,946, DROID 1,000,
Atari 1,000, LIBERO 1,000), the training datasets of the encoder, downloaded and processed as described in
[`justact/training/README.md`](../justact/training/README.md). Each clip contains one of the frame windows the action encoders were trained on. Its paths
are relative to one pool root: frame-backed datasets are read in place (`<pool>/<dataset>/...`, link each dataset's frame
root there), and the 8,946 video-backed clips are decoded once into `<pool>/decoded/`:

```bash
python tools/decode_pool_videos.py --data_root $POOL \
    --video_root spatialvid=$SPATIALVID_VIDEOS agibot=$AGIBOT_VIDEOS egodex=$EGODEX_VIDEOS sthv2=$STHV2_VIDEOS
```

Extract the pool embeddings of each representation into `$EMB/<representation>/worldmodel_pool/` (as above, with
`--data_root $POOL` and the decompressed list, `gunzip -k data/worldmodel/train_pool.json.gz`, as `--manifest`). For JustAct, compute the whitening statistics on
the pool (or use the released file), then train and export:

```bash
python compute_whitening.py --train_list ../data/worldmodel/train_pool.json.gz \
    --emb_dir $EMB/justact/worldmodel_pool --out ../checkpoints/worldmodel/justact_whitening.safetensors
python train.py --base configs/justact.yaml --name justact --data_root $POOL --emb_root $EMB \
    --whitening_stats ../checkpoints/worldmodel/justact_whitening.safetensors
python tools/export_checkpoint.py logs/justact/checkpoints/trainable_step020000_end.pt ../checkpoints/worldmodel/justact.safetensors
```

and likewise `--base configs/{no_action,adaworld_lam,dreamdojo_lam}.yaml` (the no-action model needs no `--emb_root`).
Recipe, identical for the four models: start from the released AdaWorld weights with the action pathway zeroed
(AdaWorld's latent-action columns of `label_emb` and of every cross-attention key / value projection set to zero, input
widths 512 + D and 1024 + D), LoRA of rank 128 on `to_q, to_k, to_v, to_out.0` plus the `label_emb` MLP trained,
20,000 steps, batch 4, 256 x 256 pixels, AdamW at 5e-5 with a 1,000-step linear warm-up, bf16 mixed precision, gradient
clipping 0.3, seed 32. The action embedders have no parameters. A run takes about 5 hours and 19 GB of GPU memory on one
RTX 4090 (about 20 GPU-hours for the four models). Training uses six data-loader workers, cuDNN benchmarking and bf16, so
a re-trained checkpoint is statistically similar to, not bit-identical with, the released one. Config values can be
overridden with dot-list arguments, e.g. `lightning.trainer.max_steps=50` for a smoke test.

Per-provider whitening (`compute_whitening.py`, applied inside `WhitenedExternalActionEmbedder` in training and
evaluation): x -> ((x - mean) / std) @ W, with mean / std per dimension and W = Q diag(g) Q^T from the eigendecomposition
of the covariance of the standardised pool rows; g = sqrt(D / sum(e / (e + r))) / sqrt(e + r) with the smallest ridge r
that keeps every gain at most 2, so the total variance stays D = 640 and no direction is amplified more than 2x. It is
computed from the 478,704 pool transitions only (no evaluation data). Released statistics: r = 1.576, gains 0.369-2.0;
the number of directions holding 90 % of the variance goes from 116 to 253. The two 32-d latent-action inputs and the
zero vector are used as they are.

# Retrained latent-action baselines

AdaWorld-LAM (retrained) and DreamDojo-LAM (retrained) are the latent action models of AdaWorld and DreamDojo, trained
from random initialisation on the same frame pairs and with the same budget (160,000 pair views) as the JustAct encoder.
Each keeps its upstream network, objective (beta-VAE: reconstruction of frame t+1 plus beta times a Kullback-Leibler term)
and optimiser; only the training data changes. Their 32-dimensional latent actions are the retrained baseline rows of
Tables 1-4 (representations `adaworld_lam_retrained` and `dreamdojo_lam_retrained`).

| | AdaWorld-LAM (retrained) | DreamDojo-LAM (retrained) |
|---|---|---|
| network | 16 + 16 spatio-temporal blocks, width 1024, latent 32, 474.7 M parameters | 24 + 24 blocks, width 1024, latent 32, 709.8 M parameters |
| upstream configuration | `third_party/adaworld_lam/lam/config/lam.yaml` (beta 2e-4) | `third_party/dreamdojo_lam/external/lam_project/config/lam_0.yaml` (beta 1e-6) |
| training input | square centre crop, bicubic resize to 256 x 256 | 4:3 centre crop, bilinear resize to 240 x 320 |
| schedule | 20,000 steps x 8 pairs | 80,000 steps x 2 pairs |
| optimiser | AdamW, learning rate 2.5e-5, weight decay 0.01, gradient clipping 0.3, bfloat16 autocast, seed 0 | same |
| run settings | `configs/adaworld.yaml` | `configs/dreamdojo.yaml` |
| checkpoint | `checkpoints/adaworld_lam_retrained.safetensors` | `checkpoints/dreamdojo_lam_retrained.safetensors` |

## Files

```
baselines/
  extract.py       latent actions of a clip list -> <emb_root>/<representation>/<dataset>/<task_id>.npy
  train.py         training on the JustAct training-pair cache
  lam_models.py    upstream code locations, configuration reader, network construction and checkpoint input / output
  configs/         run settings of the two released checkpoints
  third_party/     upstream network code and configuration, unmodified (see NOTICE.md)
```

## Requirements

Python 3.12 with `torch==2.10.0` (CUDA 12.8), `numpy==2.3.5`, `pillow==12.0.0`, `pyyaml==6.0.3`, `einops==0.8.2`,
`safetensors==0.7.0`, and for training also `piq==0.8.0` (reconstruction quality during validation). These are the
versions the released checkpoints and embeddings were produced with. Run the commands below from the repository root.

## Checkpoints

State dict only (float32, the network's own parameter names, no metadata):

| file | size | sha256 |
|---|---|---|
| `checkpoints/adaworld_lam_retrained.safetensors` | 1,898,902,400 B | `2dc9850d4bb640715fc9ee8b2e0203681e4d924755b1ea812169984131e4a8e0` |
| `checkpoints/dreamdojo_lam_retrained.safetensors` | 2,839,214,792 B | `ff892f5f311a574694edc9cc4e33188c4aade847f94611b72197f4db09beedf7` |

## Extracting latent actions

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/extract.py --arch adaworld \
    --manifest data/probe/libero.json --data_root <frames root of libero> \
    --ckpt checkpoints/adaworld_lam_retrained.safetensors \
    --out_dir <emb_root>/adaworld_lam_retrained/libero
```

Output: one `<task_id>.npy` per clip, float16, shape `(n_frames - 1, 32)`; row j is the latent action of the frame pair
(j, j + 1) of the clip's `target_frames`. The latent action is the encoder's posterior mean `z_mu`, computed in float32,
not normalised (typical row norms: about 0.5-0.7 for AdaWorld, about 7 for DreamDojo). Existing files are skipped, so an
interrupted run can be restarted with the same command.

Preprocessing. The whole frame is resized without the training crop: AdaWorld to 256 x 256 with PIL bicubic (skipped when
the frame is already 256 x 256), DreamDojo to 320 x 240 (width x height) with PIL Lanczos; pixels in [0, 1]. Each pair is
one 2-frame clip, 16 pairs per forward pass (`--batch_size 16`, the paper setting). Because the crop is skipped, frames
that are not square (AdaWorld) or not 4:3 (DreamDojo) are read slightly off the training distribution; the same holds for
the authors' released models evaluated with this protocol.

Select the GPU with `CUDA_VISIBLE_DEVICES`: the upstream network code places its positional table on the current CUDA
device. `--device cpu` also works (about 1 pair per second on 16 cores); CPU and GPU results agree to float16 rounding.

All 13 probe datasets of Table 1 (also used by Table 4), for both models:

```bash
EMB=<emb_root>
for ds in libero rlbench droid robocasa droid_iprl metaworld utaustin_mutex vp2_robosuite vp2_robodesk \
          spatialvid_nav arkitscenes realestate10k procgen_plan; do
  for arch in adaworld dreamdojo; do
    CUDA_VISIBLE_DEVICES=0 python baselines/extract.py --arch $arch --manifest data/probe/$ds.json \
        --data_root "<frames root of $ds>" --ckpt checkpoints/${arch}_lam_retrained.safetensors \
        --out_dir $EMB/${arch}_lam_retrained/$ds
  done
done
```

World model (Table 2): the LIBERO, RLBench and SpatialVID evaluation clips are clips of the probe lists, so the folders
above already hold them. The nuScenes evaluation clips and the training pool need their own runs (the folder name is the
clip list's `dataset` field):

```bash
for arch in adaworld dreamdojo; do
  for cell in nuscenes_dev_transfer nuscenes_conf_transfer; do      # the self cells use the same clips
    CUDA_VISIBLE_DEVICES=0 python baselines/extract.py --arch $arch --manifest data/worldmodel/$cell.json \
        --data_root <nuScenes root> --ckpt checkpoints/${arch}_lam_retrained.safetensors \
        --out_dir $EMB/${arch}_lam_retrained/nuscenes_mini
  done
  CUDA_VISIBLE_DEVICES=0 python baselines/extract.py --arch $arch --manifest data/worldmodel/train_pool.json.gz \
      --data_root <world-model pool root> --ckpt checkpoints/${arch}_lam_retrained.safetensors \
      --out_dir $EMB/${arch}_lam_retrained/worldmodel_pool
done
```

On one RTX 4090 extraction runs at about 50 pairs per second for either model: about 1.8 hours per model for the 316,482
probe pairs and about 2.7 hours per model for the 478,704 pairs of the world-model training pool.

## Training

Input: the training-pair cache of the JustAct encoder (`python -m justact.training.build_sample_cache ...`, 168,000 pairs
from nine datasets). The trainer reads `index.jsonl` and the anchor frames `frames/<i // 1000:03d>/<i:06d>_{a0,a1}.jpg`;
2 % of the records (seed 0) are held out for validation.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 python baselines/train.py --config baselines/configs/adaworld.yaml \
    --cache_dir <cache> --out_dir runs/adaworld_lam_retrained
CUDA_VISIBLE_DEVICES=0 python baselines/train.py --config baselines/configs/dreamdojo.yaml \
    --cache_dir <cache> --out_dir runs/dreamdojo_lam_retrained
```

The final weights are `<out_dir>/ckpt_step20000.safetensors` (AdaWorld) and `<out_dir>/ckpt_step80000.safetensors`
(DreamDojo), in the format `extract.py` loads. Network size, beta, learning rate, weight decay and gradient clipping are read
from the upstream YAML files; `configs/*.yaml` hold the run settings (steps, batch, seed, data workers, logging).
`--num_workers 4` is part of the recipe: the brightness augmentation is drawn inside the data-loader workers, so the
number of workers decides the augmentation values.

On one RTX 4090 (24 GB) AdaWorld trains in about 85 minutes (peak memory 13.6 GiB) and DreamDojo in about 4.6 hours
(13.3 GiB). Final validation reconstruction of frame t+1: AdaWorld 24.7 dB peak signal-to-noise ratio and 0.763 structural
similarity, DreamDojo 25.3 dB and 0.780, i.e. both models reconstruct the next frame to a similar quality.

The released DreamDojo checkpoint comes from a run that was continued from its step-5,000 resume point. A resumed run
draws a new data order from the restored generator, so the two-step command below follows the released checkpoint's data
order; a single uninterrupted run visits the pairs in a different order after step 5,000:

```bash
python baselines/train.py --config baselines/configs/dreamdojo.yaml --cache_dir <cache> --out_dir <out> --total_steps 5000
python baselines/train.py --config baselines/configs/dreamdojo.yaml --cache_dir <cache> --out_dir <out> \
    --resume <out>/ckpt_last.pt
```

GPU training is not guaranteed to be bitwise reproducible, so a re-trained model matches the released one statistically,
not bit for bit.

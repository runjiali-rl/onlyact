# Scene leakage (Table 4)

How much of the *scene* can be read back from an action representation? A decoder, identical for every
representation, learns to reconstruct the frame after the action from the representation alone. It is then
evaluated on scenes it never saw during training. If the reconstruction of unseen scenes is poor (high LPIPS
and L1, low SSIM), the representation carries little scene information beyond the action.

| file | content |
|---|---|
| `train.py` | trains and evaluates one decoder (one dataset x representation x seed) and writes `result.json` |
| `decoder.py` | decoder: Linear(D -> 512x4x4), 5 up-blocks (ConvTranspose2d stride 2, GroupNorm, GELU, Conv3x3, GroupNorm, GELU), Conv3x3, Tanh; 128x128 RGB output; about 15 M parameters |
| `dataset.py` | one sample per frame pair: embedding row *i* -> frame *i + 1* (the post-action frame), scaled to [-1, 1] |
| `split.py` | held-out-scene split (20 % of the scenes) |
| `metrics.py` | LPIPS, SSIM, PSNR |
| `run_table4.sh` | all 208 runs of the table |
| `../tables/table4_leakage.py` | Table 4 and the appendix table (per-dataset SSIM and L1) from the `result.json` files |

## Protocol

- **Representations** (`--representation`): `justact` (D = 640), `vjepa2_frozen` (1024),
  `adaworld_lam_retrained` (32), `dreamdojo_lam_retrained` (32). One row per consecutive frame pair, as written by the
  extraction scripts.
- **Training**: L1 + 0.5 x LPIPS (`lpips` 0.1.4, AlexNet, v0.1 linear weights), AdamW (learning rate 1e-4, weight
  decay 1e-4, betas 0.9 / 0.95), cosine schedule, 10 epochs, batch 64, gradient-norm clip 1.0, 2 data-loader workers.
- **Split**: 20 % of the scenes of the clip list are held out. The seed (0, 1, 2, 3) selects the held-out scenes as
  well as the initialisation and the data order. For a given dataset and seed, all four representations share the same split.
- **Evaluation**: after every epoch, on every held-out sample. LPIPS uses the same network as the loss. SSIM uses an
  11x11 Gaussian window with sigma 1.5 and a data range of 2. L1 is the mean absolute error on the [-1, 1] scale.
  Table 4 reports the last epoch, averaged over the 4 seeds. The table's Average is the unweighted mean over the
  13 datasets.

What counts as one scene (the `scene_key` of the clip list; for LIBERO and RLBench, which have no `scene_key`, the scene comes from `scene_id`):

| dataset | one scene = | held out / total scenes |
|---|---|---|
| LIBERO | a LIBERO scene, e.g. `KITCHEN_SCENE3`; several tasks share one | 5 / 23 |
| DROID, DROID-IPRL | one episode (a single lab per dataset) | 30 / 150 |
| RLBench | one task | 4 / 18 |
| RoboCasa | one kitchen layout + style | 53 / 263 |
| MetaWorld | one task | 10 / 50 |
| MUTEX | one instruction | 10 / 50 |
| Robosuite, RoboDesk | one demonstration | 60 / 300 |
| SpatialVID, RE10K | one video clip | 60 / 300 |
| ARKitScenes | one scan | 56 / 280 |
| Procgen | one episode (level) | 60 / 300 |

## Inputs

- Clip lists: `data/probe/<dataset>.json`, the same files as for Table 1.
- 128x128 frames: `$FRAMES128/<dataset>/<path from the clip list>`. These are 128x128 copies of the clip frames, with the
  same relative paths as the clip list. LIBERO and RLBench frames are already 128x128. A frame of a different size
  is resized bilinearly when loaded. No frames are distributed with this repository: the preparation scripts of
  `data/probe/prepare/` write them to `$DATA/leakage128/<dataset>/` (`FRAMES128=$DATA/leakage128`), see
  [`data/probe/README.md`](../data/probe/README.md).
- Embeddings: `$EMB/<representation>/<dataset>/<task_id>.npy`, float16, shape (n_frames - 1, D).

## Commands

Run from the repository root.

```bash
# one decoder
python -m leakage.train --dataset libero --representation justact --seed 0 \
    --clips data/probe/libero.json --data_root $FRAMES128/libero \
    --emb_dir $EMB/justact/libero --out_dir runs/table4/libero/justact/seed0

# all 208 decoders (13 datasets x 4 representations x 4 seeds), restartable
EMB=$EMB FRAMES128=$FRAMES128 OUT=runs/table4 bash leakage/run_table4.sh

# Table 4 + appendix table
python tables/table4_leakage.py --results_root runs/table4 --out_dir runs/table4/tables
```

Each run folder holds:
- `scene_split.json`: training and held-out clips, and the held-out scenes;
- `config.json`;
- `train_log.jsonl`: the loss every 50 steps and one `{"val": ...}` record per epoch;
- `recon_ep<k>.png`: 16 held-out frames, with the originals on top and the reconstructions below;
- `result.json`.

`result.json` records the dataset, representation, seed, `epochs`, `epochs_completed`, the md5 of
`scene_split.json` (`split_md5`), the sample and scene counts, and `final_val` (LPIPS, SSIM, L1, PSNR and their spread
over held-out samples, from the last epoch). The table builder averages a seed only if the run completed all of its epochs
and its `split_md5` equals that of the `vjepa2_frozen` run for the same dataset and seed.
Pass `--save_checkpoint` to keep the final decoder weights (`last.pt`, about 60 MB).

## Compute

The decoder is small, but every step also runs LPIPS. All 208 runs take about **10 GPU-hours** on one RTX 4090, run one
after another. A typical run takes 1-2 minutes; the longest are RLBench (about 10 minutes per run), LIBERO (about 6) and
MetaWorld (about 6). The output is about 6 MB per run, without checkpoints.

## Reproducibility

The seed fixes the split, the initialisation and the data order. The CUDA kernels of the decoder (for example the
backward pass of the transposed convolutions) are not deterministic, so a re-run differs from the published numbers by
about 0.001-0.004 LPIPS per table cell. That is smaller than the spread between seeds (standard deviation
0.002-0.037 per dataset).
Environment used for the paper: Python 3.12, torch 2.10, torchvision 0.25, lpips 0.1.4, numpy 2.3, Pillow 12.
torchvision downloads the AlexNet weights (`IMAGENET1K_V1`) on first use.

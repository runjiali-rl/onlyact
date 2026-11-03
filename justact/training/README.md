# Training the JustAct encoder

This folder trains the JustAct encoder, the checkpoint used in every table of the paper. V-JEPA 2 ViT-L
(`facebook/vjepa2-vitl-fpc64-256`) is adapted with LoRA and trained contrastively on frame pairs from nine video
datasets. The code implements a single recipe, `configs/justact.yaml`.

| | |
|---|---|
| input | a frame pair (f_t, f_t+k), k = 1-3, encoded as one two-frame V-JEPA 2 clip |
| embedding | mean-pooled last hidden state (1024-d) -> fixed random linear projection to 640-d -> L2 normalisation |
| trained parameters | LoRA rank 16, alpha 32, dropout 0.05 on `query, key, value, proj, fc1, fc2` (7,118,848 trained parameters; the checkpoint also holds 1,349,632 adapter parameters on the V-JEPA 2 predictor, which the encoder never runs, so they keep their initial values) |
| projection | `nn.Linear(1024, 640)` with PyTorch's default initialisation from seed 0, **not optimised** |
| positive | the anchor pair with its moving part (from RAFT optical flow) shifted to a random position on a still frame of another scene of the same dataset |
| negatives | the other pairs of the batch and a hard negative from a disjoint window of the same clip |
| loss | symmetric InfoNCE, temperature 0.07 |
| optimisation | AdamW (betas 0.9 / 0.95, weight decay 0), learning rate 1e-4, 500 warm-up steps then cosine decay, 10,000 steps, 2 samples x 8 accumulation = 16 per step, gradient-norm clipping 1.0, seed 0 |
| data | 168,000 cached samples from 9 datasets, about 0.95 passes |
| compute | one RTX 4090 (24 GB), 0.55 s per step, about 1.5 h |

The projection is part of the embedding but is never updated; only the LoRA adapters are optimised. Its gradient is
still computed and enters the gradient-norm clipping (see `train.py`), exactly as in the run that produced the
released checkpoint.

## Files

| file | role |
|---|---|
| `build_sample_cache.py` | draws the 168,000 training samples from the 9-dataset manifest and stores their four frames as JPEG |
| `precompute_flow_cache.py` | RAFT-large optical flow a0 -> a1 of every cached sample |
| `flow_cache.py` | format of the flow cache and the foreground (moving part) rule |
| `augment.py` | photometric jitter; background replacement with core shift |
| `cache_dataset.py` | builds (anchor, positive, hard negative) from the caches |
| `model.py` | training-time encoder (V-JEPA 2 + LoRA + projection), saves the checkpoint |
| `loss.py` | symmetric InfoNCE with hard negatives |
| `train.py`, `config.py`, `configs/justact.yaml` | the trainer and its recipe |

Run every command from the repository root (the folder that contains `justact/`).

## Environment

Python 3.12 with torch 2.10 / torchvision 0.25 (CUDA 12.8), transformers with `VJEPA2Model` (the released
checkpoint was trained with the development build at commit `7aa888b7fa477d13153ffbfe107dfbd6c696014a`), peft 0.19.1,
safetensors, numpy, scipy, opencv-python, pillow, decord (video decoding for the cache build) and pyyaml.
The V-JEPA 2 weights are downloaded from Hugging Face (revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f`).

## 1. Datasets and the 9-dataset manifest

The training data are VPT (Minecraft), CALVIN, LIBERO, DROID, Atari, AgiBot World, EgoDex, SpatialVID and
Something-Something v2. No data is distributed with this repository: download each dataset from its source and process
it with the scripts in `data/training/`, all into one data root `<data>`. Each script's docstring gives its output
layout and, where it has one, the command with the settings used for the paper; the commands of the LIBERO and DROID
scripts are below the table. Each dataset keeps its own licence.

| dataset | public source | licence (as stated by the source) | scripts (`data/training/`) | notes |
|---|---|---|---|---|
| VPT | OpenAI's VPT contractor recordings, index file https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_6xx_Jun_29.json (github.com/openai/Video-Pre-Training) | MIT (`LICENSE` of the VPT repository; no separate licence stated for the recordings) | `vpt_process.py`, `vpt_build_index.py` | 101-frame segments, up to 3 per episode; `vpt_segment_ids.txt` lists the 10,050 segments used |
| CALVIN | `task_ABC_D` split, http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip (github.com/mees/calvin) | MIT (README of the CALVIN repository) | `calvin_process.py` | one task per language-annotated window |
| LIBERO | LIBERO-100 HDF5 files (`libero_10`, `libero_90`) of github.com/Lifelong-Robot-Learning/LIBERO, on Hugging Face as `yifengzhu-hf/LIBERO-datasets` | datasets CC BY 4.0, code MIT (README of the LIBERO repository; the Hugging Face card says Apache-2.0) | `libero_process.py`, `libero_build_manifest.py` | 41 frames per demo; `libero_task_ids.txt` lists the 1,403 demos used, from the same 52 HDF5 files as the LIBERO probe clips (download command in `data/probe/README.md`) |
| DROID | raw release 1.0.1, `gs://gresearch/robotics/droid_raw/1.0.1` (droid-dataset.github.io; needs `gsutil`) | CC BY 4.0 (DROID paper, arxiv.org/abs/2403.12945) | `droid_download_and_process.py`, `droid_build_manifest.py` | `droid_task_ids.txt` lists the 606 episodes used; `--scene_id TRI/2023-10-26` (the default) reproduces the single scene id all DROID clips had in the paper's manifest |
| Atari | Hugging Face `edbeeching/prj_gia_dataset_atari_2B_atari_<game>_1111` (`dataset.npy`), downloaded by the script | no licence stated by the source | `atari_process.py` | Pong, Breakout, Space Invaders, Ms. Pac-Man, Seaquest |
| AgiBot World | Hugging Face `agibot-world/AgiBotWorld-Alpha` (gated), downloaded by the script | CC BY-NC-SA 4.0 (dataset card) | `agibot_download_and_process.py` | re-encoded to 256x256 H.264 at 16 frames per second |
| EgoDex | training parts 1-5, https://ml-site.cdn-apple.com/datasets/egodex/partN.zip (github.com/apple/ml-egodex), streamed by the script | "CC-by-NC-ND terms" (README of the EgoDex repository) | `egodex_download_and_process.py` | re-encoded to 256x256 H.264 at 16 frames per second |
| SpatialVID | Hugging Face `SpatialVID/SpatialVID` (gated), video groups 1-272, downloaded by the script | CC BY-NC-SA 4.0 (dataset card) | `spatialvid_download_and_process.py` | re-encoded to 256x256 H.264 at 16 frames per second |
| Something-Something v2 | Qualcomm, https://www.qualcomm.com/developer/software/something-something-v-2-dataset | see the source (licence agreement on its download page) | none | the official `.webm` videos, used as they are, in `<data>/sthv2/videos/20bn-something-something-v2/` (the default of `build_union_manifest.py`) |

LIBERO and DROID (`<libero>` holds `libero_10/` and `libero_90/`):

```bash
python data/training/libero_process.py --libero_root <libero> --splits libero_10 libero_90 --data_root <data>
python data/training/libero_build_manifest.py --data_root <data> --task_ids data/training/libero_task_ids.txt
python data/training/droid_download_and_process.py --data_root <data> --task_ids data/training/droid_task_ids.txt
python data/training/droid_build_manifest.py --data_root <data> --task_ids data/training/droid_task_ids.txt
```

Then build the manifest:

```bash
python data/training/build_union_manifest.py --data_root <data> --mp4_min_frames 16 \
    --datasets calvin atari libero droid vpt agibot egodex spatialvid --out <data>/union_8ds/manifest_union.json
python data/training/build_union_manifest.py --data_root <data> --mp4_min_frames 16 \
    --datasets sthv2 --out <data>/union_9ds/manifest_sthv2_only.json
python data/training/merge_sthv2_manifest.py --union <data>/union_8ds/manifest_union.json \
    --sthv2 <data>/union_9ds/manifest_sthv2_only.json --out <data>/union_9ds/manifest_union.json
python data/training/subsample_manifest.py --input <data>/union_9ds/manifest_union.json \
    --output <data>/union_9ds/manifest_union_sub.json      # caps AgiBot / EgoDex 100K, SpatialVID 200K, Something-Something v2 50K, seed 0
```

The paper's subsampled manifest has 480,360 clips: SpatialVID 200,000, AgiBot 100,000, EgoDex 100,000,
Something-Something v2 50,000, CALVIN 17,870, VPT 10,050, LIBERO 1,403, DROID 606, Atari 431. Paths in the manifests
are relative to `<data>`. A rebuild gives the same clips (the script docstrings list the few exceptions), but the order of the video clips (and so which ones the caps
keep and the final shuffle) follows the file-system listing order; the sample list below makes the training set
exact regardless.

## 2. Sample cache (168,000 samples, about 21 GB)

Each sample is one clip and two windows of it: the anchor pair (a0, a1), 1-3 frames apart, and the hard-negative pair
(h0, h1) from a non-overlapping window. Frames are stored with the short side capped at 292 pixels (the V-JEPA 2
processor's resize target), JPEG quality 95.

**Exact paper samples.** `data/training/samples_168k.jsonl.gz` lists the 168,000 samples the released checkpoint was
trained on (clip id, dataset, scene id and the four frame indices; no frames). Rebuild exactly these samples; the
manifest is only used to look up the clips by id, so pass the full manifest from before subsampling:

```bash
python -m justact.training.build_sample_cache --manifest <data>/union_9ds/manifest_union.json --data_root <data> \
    --samples data/training/samples_168k.jsonl.gz --out_dir <cache> --workers 4 --max_short_side 292
```

**New draw.** Draw the samples from your manifest with the paper's procedure (seed 0, gaps 1-3, dataset weights
VPT 0.25, CALVIN 0.15, LIBERO 0.05, DROID 0.05, Atari 0.05, AgiBot 0.10, EgoDex 0.10, SpatialVID 0.15,
Something-Something v2 0.10). The draw depends on the task order of the manifest, so it gives the paper's samples only
for the paper's manifest.

```bash
python -m justact.training.build_sample_cache --manifest <data>/union_9ds/manifest_union_sub.json --data_root <data> \
    --out_dir <cache> --n_samples 168000 --gap_lo 1 --gap_hi 3 --seed 0 --workers 4 --max_short_side 292 \
    --dataset_weights "vpt=0.25,calvin=0.15,libero=0.05,droid=0.05,atari=0.05,agibot=0.10,egodex=0.10,spatialvid=0.15,sthv2=0.10"
```

`--only_draw` writes the drawn sample list (`<cache>/samples.jsonl`) without decoding any frame. The build is
resumable; with 4 workers it takes about 3 h, mostly video decoding.

## 3. Optical-flow cache (about 5.8 GB)

```bash
python -m justact.training.precompute_flow_cache --cache_dir <cache>     # one GPU, about 17 min on an RTX 4090
```

RAFT-large (torchvision `Raft_Large_Weights.C_T_SKHT_V2`), 12 flow updates, float16 autocast, at 256x256, stored at
128x128 as int8. The flow is used only to build the positive: its foreground mask decides which pixels are moved onto
the new background, and the mean flow outside the moving part (above 0.5 px) marks camera motion, for which no
composite is built.

## 4. Train

```bash
python -m justact.training.train --cache_dir <cache> --out_dir runs/justact
```

Output: `runs/justact/config.json`, `runs/justact/train_log.jsonl` (step 1 and every 20 steps: loss, learning rate,
anchor-to-positive accuracy, positive / hard-negative cosine similarities and their gap, and `synth_ok` / `synth_n`,
the number of composited positives / samples per micro-batch, averaged over the step), checkpoints `runs/justact/step_<N>/` every 2,000 steps and
`runs/justact/final/` with `adapter_config.json`, `adapter_model.safetensors` and `projection.safetensors`. Copy
`final/` to `checkpoints/justact/` to use it with the extraction code (`justact/extract.py`).

What a healthy run looks like: at step 1 the anchor-to-positive accuracy is about 0.5 and the positive is slightly
less similar to the anchor than the hard negative (the composited positive starts out harder than the same-clip
negative); the accuracy passes 0.9 within the first 300 steps; at step 10,000 the loss is about 0.01, the accuracy
1.0 and the positive / hard-negative gap about 0.66. On average 43 % of the positives are composited
(`synth_ok / synth_n` between 0.25 and 0.63 in 90 % of the logged steps); the rest fall back to a photometric copy of
the anchor (camera motion, static pairs, or pairs moving over more than 90 % of the frame).

Reproducibility: with the same cache, flow cache, seed and software versions the run is deterministic up to GPU
kernel non-determinism. The model is built from the global seed before the data loader draws its shuffle order, and
every training view is a pure function of its index; `--num_workers` does not change the result.
`--stop_after N` ends the run after N steps without changing the learning-rate schedule.

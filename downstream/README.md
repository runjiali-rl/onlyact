# Downstream world-model fine-tuning (Table 3)

Each pretrained world model of Table 2 (the AdaWorld world model conditioned on AdaWorld-LAM (retrained),
DreamDojo-LAM (retrained) or OnlyAct action representations) is fine-tuned on a small labelled set of a new domain and
evaluated on held-out test cases after 0, 200 and 800 updates.

| Domain | Actions | Labelled fine-tuning clips | Test cases | Recipe (`configs/<domain>.json`, block `recipe`) |
|---|---|---|---|---|
| DMLab | look left / look right / forward | 300 (100 per action) | 300 (300 test episodes) | standardised interface, interface learning rate 2.42e-3 |
| Minecraft | forward+jump / camera left 20° / camera right 20° (+jump) | 300 (100 per action) | 300 | standardised interface, interface learning rate 2.42e-3 |
| Habitat (HM3D) | move forward / turn left / turn right | 300 (100 per action, scene 00337) | 300 (scenes 00770, 00861) | native interface, interface learning rate 5e-5 |
| nuScenes | ego motion (forward, lateral) in metres | 100 (4 driving logs) | 60 (12 other driving logs) | continuous head, learning rate 5e-5 |

Two recipes are used, and each domain config states its recipe explicitly:

- **Standardised interface (DMLab, Minecraft).** The action book stores `u`, and the condition is `mean + std * u`,
  where `mean` and `std` are per-dimension statistics of the arm's own world-model training features (after the world
  model's input transform). The interface learning rate 2.42e-3 is then a step measured in the arm's own standard deviations,
  so every arm gets the same relative step whatever the scale of its features.
- **Native interface / continuous head (Habitat, nuScenes).** The book rows (Habitat), or the head
  `Linear(2, d) - SiLU - Linear(d, d)` (nuScenes), output the condition directly. Learning rate 5e-5.

Both recipes share the rest of the setup. The whole merged UNet is trainable (1.56 billion parameters, FP32 weights) at 5e-6.
They use 800 AdamW updates (betas 0.9 / 0.999, weight decay 0.01) with a global batch of 32 (32 micro-batches of one clip),
a 100-update linear warm-up, gradient clipping at 0.3, an EMA of the UNet (decay 0.9999 with the num-updates correction),
bf16 autocast, activation checkpointing and seed 32. One fixed seed controls the clip order, the augmentation and the
diffusion noise, so the 200- and 800-update checkpoints come from the same run.

**Step 0 ("0 updates") uses no gradient update, but it does use the labelled clips.** Each book row is the mean of the
arm's action features over the 100 labelled clips of that action, after the world model's own input transform. On
nuScenes the step-0 head is fitted by 3000 SGD steps (learning rate 0.01, batch 10, mean-squared error) from the clip's ego
motion to that transformed feature. Fine-tuning starts from this step-0 interface.

## Metrics

Each test case has six real context frames (0-5), and one next frame (6) is generated. The sampler is EulerEDM with the
EDM-shift discretisation, 5 steps, classifier-free guidance 1.05 and float16 autocast. The random state is reset to the
case's own seed before every generated action, so all actions and all arms share the same noise.

- **PSNR**: the PSNR (dB) of the frame generated under the commanded action, measured against the real frame 6.
- **Control**: for the discrete domains, the PSNR under the commanded action minus the mean PSNR under the two other
  actions. For nuScenes, it is the commanded PSNR minus the PSNR under four actions drawn from the training clips. The
  draw uses `numpy Generator(PCG64(seed)).integers(0, 100, 4)`, and a draw equal to the commanded action counts as 0.
  A Control of 0 means the action is ignored.
- **Follow**: for the discrete domains, this is the accuracy of an inverse-dynamics classifier (`idm.py`). The classifier
  is trained only on real frames and receives (real frame 5, generated frame) for every generated action; chance is 0.333.
  The "Real frames" row is the same classifier applied to (real frame 5, real frame 6). For nuScenes, Follow is a
  per-case Spearman correlation: the five commanded forward displacements against the displacement that a forward-motion
  readout (`readout.py`) reads from (real frame 5, generated frame). Chance is 0.
- **Delta**: OnlyAct minus the better baseline. A Delta is not significant when its 95 % paired bootstrap interval
  contains 0. The bootstrap resamples whole episodes (driving logs on nuScenes) with 20 000 draws and seed 20261020. It
  covers test-case sampling only, not the variance between training seeds.

## Files

```
downstream/
  configs/{dmlab,minecraft,habitat,nuscenes}.json   lists, recipe, generation settings, classifier, arms
  prepare.py      frames from the public sources (DMLab / Minecraft TECO episodes, nuScenes CAM_FRONT, HM3D renders)
  interfaces.py   step-0 book / head from the arms' action features; standardisation statistics
  train.py        fine-tuning (both recipes);  offload.py  CPU-offloaded AdamW + EMA
  generate.py     test rollouts of one cell -> results.jsonl + preds/NNN.npy (uint8)
  idm.py          inverse-dynamics classifier (training + input);  readout.py  nuScenes forward-motion readout
  score.py        Follow / Control / PSNR per cell + bootstrap contrasts;  control.py  nuScenes four-draw margin
  table.py        the four score files -> the numbers of Table 3
  model.py, data.py, common.py
data/downstream/<domain>/   train.json, test.json (+ dmlab/idm_episodes.json, nuscenes/dev.json), in the order used
checkpoints/downstream/     idm_{dmlab,minecraft,habitat}.safetensors, readout_nuscenes.safetensors,
                            step0/<domain>_<arm>.safetensors, standardise/<arm>.safetensors
```

Arms: `onlyact`, `adaworld_lam_retrained` and `dreamdojo_lam_retrained`. The pretrained world models are read from
`worldmodel/configs/<world model>.yaml` and `checkpoints/worldmodel/<world model>.safetensors` (OnlyAct also needs
`checkpoints/worldmodel/onlyact_whitening.safetensors`), together with the public AdaWorld weights in
`checkpoints/adaworld/adaworld.safetensors`. Each path can be overridden with `--wm_config`, `--wm_ckpt`, `--whitening_stats`
or `--base_ckpt`.

Software: Python 3.12 and torch 2.10.0 (CUDA 12.8). The other packages are piq 0.8.0 (PSNR / LPIPS; LPIPS downloads its
VGG weights on first use), peft 0.18.1, omegaconf, safetensors 0.7.0, numpy 2.3.5, pillow, plus opencv-python for
Minecraft preparation. They come on top of the world-model requirements.

## Data

No frames are distributed. The lists give the frame identifiers (TECO episode and frame indices; nuScenes
`source_filenames` / `frame_tokens`; Habitat scenes, `poses` and render settings) with paths relative to a data root,
and every frame carries a pixel hash. `prepare.py` makes the frames from the public sources below and checks every
frame against its hash, and the loaders check them again. `--no_check_hashes` turns the check off. The commands write to
`$DATA/downstream/<domain>/`, the data root that `scripts/reproduce_table3.sh` reads.

```bash
python -m downstream.prepare clips --list data/downstream/dmlab/train.json --source $TECO/dmlab --out $DATA/downstream/dmlab              # and test.json
python -m downstream.prepare clips --list data/downstream/minecraft/train.json --source $TECO/minecraft --out $DATA/downstream/minecraft  # and test.json
python -m downstream.prepare nuscenes --list data/downstream/nuscenes/train.json --source $NUSCENES --out $DATA/downstream/nuscenes       # and test / dev
python -m downstream.prepare habitat --list data/downstream/habitat/train.json --source $HM3D --out $DATA/downstream/habitat              # and test.json
```

**DMLab and Minecraft (TECO).** The TECO datasets (https://github.com/wilson1yan/teco) are on archive.org as one tar
archive per domain, split into parts with one archive.org item per part: DMLab in `dmlab_dataset_aa` to `_ac` (3 parts,
57.6 GB) and Minecraft in `minecraft_marsh_dataset_aa` to `_ak` (11 parts, 225.2 GB). All parts are needed, because the
archive is cut at fixed byte offsets. Only the folders that the lists use have to be extracted:

```bash
mkdir -p $TECO && cd $TECO
for p in aa ab ac; do wget -c https://archive.org/download/dmlab_dataset_$p/dmlab.tar.part$p; done
cat dmlab.tar.parta? | tar -xf - dmlab/train/49 dmlab/test/0                 # -> $TECO/dmlab/{train/49,test/0}/<episode>.npz
for p in aa ab ac ad ae af ag ah ai aj ak; do wget -c https://archive.org/download/minecraft_marsh_dataset_$p/minecraft.tar.part$p; done
cat minecraft.tar.parta? | tar -xf - minecraft/train/4_39 minecraft/test/0   # -> $TECO/minecraft/.../<episode>.{mp4,npz}
```

The MD5 of every part is listed at `https://archive.org/metadata/<item>`, and the parts can be deleted after extraction.
`$TECO/dmlab` is also the `--data_root` of the DMLab inverse-dynamics classifier. Licence: neither the TECO repository
nor the archive.org items state one; see the source.

**nuScenes.** The lists use the front camera of the full nuScenes v1.0-trainval release (not v1.0-mini), both key frames
(`samples/CAM_FRONT/`) and intermediate frames (`sweeps/CAM_FRONT/`). Register at https://www.nuscenes.org/nuscenes#download,
accept the Terms of Use, and download the "File blobs" of the trainval parts: part 1 (`v1.0-trainval01_blobs.tgz`) for the
train and dev lists, parts 2 to 7 and 9 for the test list (27.5 to 33.5 GB each, 242 GB in total). The metadata and the
nuScenes devkit are not needed. `$NUSCENES` is the folder with `samples/` and `sweeps/`:

```bash
mkdir -p $NUSCENES
for k in 1 2 3 4 5 6 7 9; do tar -xzf v1.0-trainval0${k}_blobs.tgz -C $NUSCENES samples/CAM_FRONT sweeps/CAM_FRONT; done
```

Licence: CC BY-NC-SA 4.0 (non-commercial use), as stated in the nuScenes paper, under the nuScenes Terms of Use
(https://www.nuscenes.org/terms-of-use).

**Habitat (HM3D example scenes).** The three scenes (00337, 00770, 00861) are the public HM3D v0.2 example scenes of
https://github.com/matterport/habitat-matterport-3dresearch (folder `example/`, 162 MB):

```bash
mkdir -p $HM3D && cd $HM3D
for f in hm3d-example-habitat-v0.2.tar hm3d-example-configs.tar; do
  wget https://github.com/matterport/habitat-matterport-3dresearch/raw/9b74d5aa58e371c9bc16b2ccb64748cddf9bf1c6/example/$f
done
sha256sum -c <<'EOF'
bbafcfb13e84f404e95fb99022f0d6702571be795f3f711ad396d0bae5b33ef7  hm3d-example-habitat-v0.2.tar
b81722f17f6710b33e45e33439f107eacd9c881bdc5e3040034846cb0d8810f4  hm3d-example-configs.tar
EOF
tar -xf hm3d-example-habitat-v0.2.tar && tar -xf hm3d-example-configs.tar
```

This gives `$HM3D/hm3d_example_basis.scene_dataset_config.json` and `$HM3D/<scene>/<id>.basis.{glb,navmesh}`. habitat-sim's
downloader fetches the same two archives (`python -m habitat_sim.utils.datasets_download --uids hm3d_example_habitat
hm3d_example_configs --data-path data/`, then `--source data/scene_datasets/hm3d/example`). Licence: "available
exclusively for academic, non-commercial uses", under the Matterport End User License Agreement for Academic Use of Model
Data (https://matterport.com/matterport-end-user-license-agreement-academic-use-model-data).

The frames are rendered with habitat-sim 0.3.3, whose conda packages are built for Python 3.9 only, so `prepare.py habitat`
runs in its own environment. It needs a GPU with EGL (headless build, no Bullet physics) and a CPU build of torch for the
resize:

```bash
conda create -n habitat -c conda-forge -c aihabitat python=3.9 numpy=1.26.4 habitat-sim=0.3.3 headless
conda activate habitat
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu    # the last torch release for Python 3.9
cd onlyact && python -m downstream.prepare habitat --list data/downstream/habitat/train.json --source $HM3D --out $DATA/downstream/habitat
```

Each frame is rendered at its stored pose with the settings of the list header (`render`: 128 x 128 pinhole RGB
camera 1.25 m above the stored position, 90° field of view), then bicubic-resized to 256 x 256 as for nuScenes. All 600 clips take
under a minute on an RTX 4090. habitat-sim prints a harmless "SSD Load Failure" line per scene (the example scenes have
no semantic annotations). On an RTX 4090 (NVIDIA driver 595.84), rendering and resizing as above reproduce all 4200
listed frames exactly, and the resize gives identical pixels with torch 2.4.1, 2.5.0, 2.7.0, 2.8.0, 2.9.1 and 2.10.0.
Other GPUs or drivers may rasterise a few pixels differently. If `prepare.py habitat` stops with "rendered pixels
differ", rerun it with `--no_check_hashes` and pass `--no_check_hashes` to `train.py`, `generate.py` and `score.py`; the Habitat
numbers may then differ slightly from the paper.

## Running

The commands below are for one domain and one arm; repeat them for every domain and arm.

```bash
cd onlyact            # repository root
D=dmlab; A=onlyact    # domain, arm

# 0 updates
python -m downstream.generate --config downstream/configs/$D.json --arm $A --data_root $DATA/downstream/$D --out runs/table3/cells/$D/${A}_step0

# fine-tuning (saves step000200 / step000400 / step000800.safetensors), then generation from the saved checkpoints
python -m downstream.train --config downstream/configs/$D.json --arm $A --data_root $DATA/downstream/$D --out runs/table3/train/$D/$A
for N in 200 800; do
  python -m downstream.generate --config downstream/configs/$D.json --arm $A --data_root $DATA/downstream/$D \
      --finetuned runs/table3/train/$D/$A/step000$N.safetensors --out runs/table3/cells/$D/${A}_step$N
done

# scoring (after all arms of the domain), then the table
python -m downstream.score --config downstream/configs/$D.json --cells runs/table3/cells/$D --data_root $DATA/downstream/$D --out runs/table3/scores/$D.json
python -m downstream.table --scores runs/table3/scores --out runs/table3/table3_source.json
```

For nuScenes, `generate.py` also reads the training list to get the action bank of the draws. When the training clips are
in a different folder from the test clips, pass `--train_data_root`.

The step-0 interfaces and classifiers can be rebuilt instead of downloaded:

```bash
python -m downstream.interfaces book --config downstream/configs/$D.json --arm $A --features $EMB/$A/${D}_train --out checkpoints/downstream/step0/${D}_$A.safetensors
python -m downstream.interfaces head --config downstream/configs/nuscenes.json --arm $A --features $EMB/$A/nuscenes_train --data_root $DATA/downstream/nuscenes --out checkpoints/downstream/step0/nuscenes_$A.safetensors
python -m downstream.interfaces standardise --arm $A --pool_list <world-model training clip list> --features $EMB/$A/pool --out checkpoints/downstream/standardise/$A.safetensors
python -m downstream.idm --config downstream/configs/dmlab.json --data_root $TECO/dmlab --out checkpoints/downstream/idm_dmlab.safetensors
python -m downstream.idm --config downstream/configs/minecraft.json --data_root $DATA/downstream/minecraft --out checkpoints/downstream/idm_minecraft.safetensors
python -m downstream.idm --config downstream/configs/habitat.json --data_root $DATA/downstream/habitat --out checkpoints/downstream/idm_habitat.safetensors
python -m downstream.readout --config downstream/configs/nuscenes.json --data_root $DATA/downstream/nuscenes --out checkpoints/downstream/readout_nuscenes.safetensors
```

`--features` is a folder of `<task_id>.npy` files, one (6, d) float16 array per clip, written by the arm's extractor over
the listed training clips. The book uses row 5, the last transition.

## Compute

- **Fine-tuning.** Each run needs one 24 GB GPU and **24-26 GiB of free host memory**: the AdamW moments and the EMA copy
  of the UNet live on the CPU (about 17 GiB). Do not run two fine-tuning runs on one host unless it has the memory for both.
  One 800-update run takes about 4 h on an RTX 4090 and 7 h on an RTX A5500. All 12 runs together take about 50 to 80
  GPU-hours.
- **Checkpoints.** Each saved checkpoint is 6.1 GB (32-d arms) or 6.2 GB (OnlyAct), FP32.
- **Generation.** One 300-case discrete cell takes 14 min on an RTX 4090 (26 min on an RTX A5500), and one 60-case
  nuScenes cell takes 9 min. A cell's predictions take 180 MB.
- **Scoring.** All four domains score in a few minutes. The classifiers and the readout are small and also run on a CPU.

## Reproducibility

- **Generation is deterministic.** Repeated generation from a fixed checkpoint on the same GPU and software stack is
  bit-identical. Across GPU models, the PSNR of a case moves by at most about 0.02 dB.
- **Fine-tuning is not bit-reproducible.** Backward passes are not deterministic, so retrained models differ slightly.
  Retraining reproduced the 800-update PSNR and Control to about 0.01 dB.
- **nuScenes Follow is sensitive.** It is a rank correlation over five often nearly tied readout values, so it moves more
  than the other metrics between retrainings.

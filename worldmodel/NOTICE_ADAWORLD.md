# Third-party code in `worldmodel/`

## AdaWorld (Apache License 2.0)

The world model is AdaWorld (Gao et al., "AdaWorld: Learning Adaptable World Models with Latent Actions", ICML 2025),
https://github.com/Little-Podi/AdaWorld. Its code is distributed under the Apache License 2.0; a copy of the licence is in
`LICENSE_ADAWORLD`. AdaWorld itself builds on Vista (https://github.com/OpenDriveLab/Vista, Apache License 2.0) and on
Stability AI's generative-models (https://github.com/Stability-AI/generative-models, MIT License); parts of
`vwm/modules/diffusionmodules/sampling.py` are ported from k-diffusion (https://github.com/crowsonkb/k-diffusion, MIT License).

Files copied from AdaWorld's `worldmodel/` folder:

| File | Changes |
|---|---|
| `vwm/__init__.py`, `vwm/util.py`, `vwm/lr_scheduler.py` | none |
| `vwm/models/__init__.py`, `vwm/models/autoencoder.py` | none |
| `vwm/models/diffusion.py` | `configure_optimizers` passes only the UNet parameters with `requires_grad` (LoRA weights and `label_emb`) to the optimizer |
| `vwm/modules/__init__.py`, `vwm/modules/attention.py`, `vwm/modules/video_attention.py`, `vwm/modules/ema.py` | none |
| `vwm/modules/autoencoding/*.py`, `vwm/modules/distributions/distributions.py` | none |
| `vwm/modules/diffusionmodules/*.py` | none |
| `vwm/modules/encoders/modules.py` | none |
| `sample_utils.py` | only `init_sampling`, `get_guider`, `get_discretization`, `get_sampler` and `get_batch` are kept, unchanged |
| `train.py` | reduced to the configuration used here: start from the released AdaWorld weights with the action pathway zeroed, LoRA, trainable-only checkpoints; resume, Stable Video Diffusion initialisation, image logging and debugging code removed |
| `LICENSE_ADAWORLD` | AdaWorld's `LICENSE`, unchanged |

New files (this repository): `vwm/modules/encoders/external_action.py`, `vwm/data/external_action.py`, `loading.py`,
`clip_lists.py`, `generate.py`, `score.py`, `compute_whitening.py`, `configs/*.yaml` (derived from AdaWorld's
`configs/training/adaworld.yaml`), `tools/*.py`.

## Weights (not redistributed)

* AdaWorld released weights, `adaworld.safetensors` from https://huggingface.co/Little-Podi/AdaWorld (model card: Apache-2.0).
  AdaWorld is fine-tuned from Stable Video Diffusion; check the Stability AI licence terms of Stable Video Diffusion as well.
* OpenCLIP ViT-H-14 (`laion2b_s32b_b79k`), downloaded by `open_clip` when the model is built and then overwritten by the
  AdaWorld weights: https://github.com/mlfoundations/open_clip.
* LPIPS weights downloaded by `piq` (LPIPS is reported in `results.jsonl` but not used in the paper's table):
  https://github.com/photosynthesis-team/piq.

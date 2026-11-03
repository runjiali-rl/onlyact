# Third-party code in `baselines/`

## AdaWorld latent action model

- Source: AdaWorld, <https://github.com/Little-Podi/AdaWorld> (Gao et al., "Learning Adaptable World Models with Latent Actions", arXiv:2503.18938).
- Licence: Apache License 2.0, copy in `third_party/adaworld_lam/LICENSE`.
- Vendored, unmodified, at their upstream paths under `third_party/adaworld_lam/`:
  - `lam/lam/modules/__init__.py`, `lam.py`, `blocks.py`, `embeddings.py` (the latent action network)
  - `lam/config/lam.yaml` (network size, beta, optimiser and trainer settings)

## DreamDojo latent action model

- Source: DreamDojo, <https://github.com/NVIDIA/DreamDojo> (NVIDIA, arXiv:2602.06949).
- Licence: the DreamDojo source code is released under the Apache License 2.0, copy in `third_party/dreamdojo_lam/LICENSE`.
- Vendored, unmodified, at their upstream paths under `third_party/dreamdojo_lam/`:
  - `external/lam/modules/__init__.py`, `lam.py`, `blocks.py`, `embeddings.py` (inference copy of the network, used by `extract.py`)
  - `external/lam_project/lam/modules/__init__.py`, `lam.py`, `blocks.py`, `embeddings.py` (training copy of the network, used by `train.py`)
  - `external/lam_project/config/lam_0.yaml` (network size, beta, optimiser and trainer settings)

## Notes

- The upstream files carry no per-file licence headers; the licence of each upstream repository applies to them.
- None of the vendored files is changed. On a CPU-only run, `extract.py` replaces the forward method of the upstream
  `PositionalEncoding` at run time (the upstream code moves its table with `.cuda()`); the added values are the same.
- `train.py` contains code derived from the upstream training code (Apache License 2.0): the loss follows
  `lam/model.py :: LAM.shared_step` and the input preprocessing follows `lam/dataset.py :: VideoDataset` of each repository.
- Weights. `adaworld_lam_retrained.safetensors` and `dreamdojo_lam_retrained.safetensors` were trained from random
  initialisation with this code; no released weights were used. The authors' released latent action model weights are not
  redistributed here (AdaWorld: Apache License 2.0, see the AdaWorld repository; DreamDojo: NVIDIA Open Model License, see
  the DreamDojo model card).

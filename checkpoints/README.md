# Checkpoints

Download the files below into this folder, keeping the sub-folders. `SHA256SUMS` lists their checksums
(`cd checkpoints && sha256sum -c SHA256SUMS`).

Download: `<CHECKPOINT_URL>`

| file | size | content | used by |
|---|---|---|---|
| `justact/adapter_config.json`, `justact/adapter_model.safetensors` | 34 MB | JustAct LoRA adapters for V-JEPA 2 ViT-L (PEFT format) | everything |
| `justact/projection.safetensors` | 2.6 MB | the fixed random 1024 -> 640 projection (`weight`, `bias`) | everything |
| `adaworld_lam_retrained.safetensors` | 1.9 GB | AdaWorld latent action model trained on our frame pairs | Tables 1, 2, 4 |
| `dreamdojo_lam_retrained.safetensors` | 2.8 GB | DreamDojo latent action model trained on our frame pairs | Tables 1, 2, 4 |
| `worldmodel/{no_action,adaworld_lam,dreamdojo_lam,justact}.safetensors` | 223-246 MB each | trained weights (LoRA + action conditioning) of the four AdaWorld world models | Tables 2, 3 |
| `worldmodel/justact_whitening.safetensors` | 3.3 MB | per-provider whitening statistics of the JustAct input of the world model | Tables 2, 3 |
| `downstream/idm_{dmlab,minecraft,habitat}.safetensors`, `downstream/readout_nuscenes.safetensors` | about 1 MB each | inverse-dynamics classifiers (Follow) and the nuScenes forward-motion readout, with their training logs | Table 3 |
| `downstream/step0/*.safetensors`, `downstream/standardise/*.safetensors` | small | action interfaces before fine-tuning (books / heads) and the standardisation statistics | Table 3 |

Weights that are not redistributed here and are downloaded from their authors:

- **V-JEPA 2 ViT-L** (`facebook/vjepa2-vitl-fpc64-256`, revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f`, MIT licence). It is
  fetched from the Hugging Face Hub on first use.
- **AdaWorld world model** (`Little-Podi/AdaWorld`, file `adaworld.safetensors`, Apache License 2.0). It is the base of the four
  world models and is needed for Tables 2 and 3: place it at `checkpoints/adaworld/adaworld.safetensors`. The world-model code
  also loads the OpenCLIP ViT-H-14 image encoder (`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`) from the Hugging Face Hub.

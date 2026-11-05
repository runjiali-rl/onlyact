"""Training-time OnlyAct encoder: V-JEPA 2 ViT-L with LoRA adapters and a fixed random 640-d projection.

A frame pair (f_t, f_t+k) is one two-frame clip (a single temporal tubelet). It goes through the V-JEPA 2 video
processor (resize the short side to 292, centre-crop 256, ImageNet normalisation) and the V-JEPA 2 encoder only (the
predictor is not run). The encoder's last hidden state is mean-pooled (1024-d), mapped by the projection to 640-d and
L2-normalised.

The projection is nn.Linear(1024, 640) with PyTorch's default initialisation, created right after the LoRA adapters
from the global seed. It is never optimised: only the LoRA adapters are in the optimiser. Its parameters keep
requires_grad=True, so their gradient is computed and enters the global gradient-norm clipping of the trainer.

save() writes the adapter with PEFT (adapter_config.json, adapter_model.safetensors) and the projection as
projection.safetensors ("weight", "bias"); this is the checkpoint layout onlyact.encoder.OnlyActEncoder loads.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file

BASE_MODEL = "facebook/vjepa2-vitl-fpc64-256"
BASE_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
LORA_TARGETS = ("query", "key", "value", "proj", "fc1", "fc2")


def _unwrap_peft(m):
    """The VJEPA2Model inside a PEFT model."""
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        return m.base_model.model
    return m


class OnlyActTrainModel(nn.Module):
    def __init__(self, model_name: str = BASE_MODEL, revision: Optional[str] = BASE_REVISION, lora_r: int = 16,
                 lora_alpha: int = 32, lora_dropout: float = 0.05, lora_targets: Sequence[str] = LORA_TARGETS,
                 proj_dim: int = 640, dtype: torch.dtype = torch.bfloat16, device: Optional[str] = None):
        super().__init__()
        from transformers import AutoVideoProcessor, VJEPA2Model

        # The construction order fixes the random-number stream: LoRA initialisation, then the projection.
        self.processor = AutoVideoProcessor.from_pretrained(model_name, revision=revision)
        base = VJEPA2Model.from_pretrained(model_name, revision=revision, dtype=dtype)
        self._dtype = dtype
        self.emb_dim = int(base.config.hidden_size)
        lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                              target_modules=list(lora_targets), bias="none")
        # Every module whose name matches a target gets an adapter, including the V-JEPA 2 predictor (never run, so its
        # adapters keep their initial values) and the patch-embedding convolution ("proj").
        self.backbone = get_peft_model(base, lora_cfg)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backbone.to(device)
        self.backbone.train()
        self.proj_dim = int(proj_dim)
        self.projection = nn.Linear(self.emb_dim, self.proj_dim)
        self.projection.to(device)

    @staticmethod
    def items_to_clips(items: List[Dict[str, Any]]) -> List[np.ndarray]:
        """[{"image": [PIL_t, PIL_t+k]}, ...] -> list of uint8 arrays (2, H, W, 3)."""
        clips = []
        for it in items:
            imgs = it["image"]
            frames = [np.asarray(im.convert("RGB")) for im in imgs[:2]]
            if frames[0].shape != frames[1].shape:
                from PIL import Image as _Image
                w, h = imgs[0].size
                frames[1] = np.asarray(imgs[1].convert("RGB").resize((w, h), _Image.BILINEAR))
            clips.append(np.stack(frames, axis=0))
        return clips

    def encode_pixels(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) -> (B, 1024) mean-pooled encoder features in float32."""
        device = next(self.backbone.parameters()).device
        pv = pixel_values_videos.to(device=device, dtype=self._dtype)
        enc = _unwrap_peft(self.backbone).encoder
        hidden = enc(pixel_values_videos=pv).last_hidden_state  # (B, N, D)
        return hidden.mean(dim=1).float()

    def forward(self, items: List[Dict[str, Any]]) -> torch.Tensor:
        """Unit-norm (B, 640) embeddings of a list of frame-pair items."""
        clips = self.items_to_clips(items)
        pv = self.processor(clips, return_tensors="pt")["pixel_values_videos"]
        pooled = self.projection(self.encode_pixels(pv))
        return F.normalize(pooled, p=2, dim=-1)

    def save(self, out_dir: str) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(out_dir)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.projection.state_dict().items()},
                  str(Path(out_dir) / "projection.safetensors"))

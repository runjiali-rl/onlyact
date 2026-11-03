"""The JustAct encoder: V-JEPA 2 ViT-L with merged LoRA adapters and a fixed 640-d projection.

A frame pair (f_t, f_t+1) is fed to V-JEPA 2 as one two-frame clip (a single temporal tubelet). The encoder's
last hidden state is mean-pooled (1024-d), mapped by the projection to 640-d and L2-normalised.
The projection is a random linear map fixed at initialisation; only the LoRA adapters are trained.
Without a checkpoint the same code gives the frozen V-JEPA 2 baseline (1024-d, no projection).
"""
from __future__ import annotations

import os

import numpy as np
import torch
from safetensors.torch import load_file

BASE_MODEL = "facebook/vjepa2-vitl-fpc64-256"
BASE_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"


class JustActEncoder:
    """Encodes batches of frame pairs. ckpt_dir=None gives frozen V-JEPA 2."""

    def __init__(self, ckpt_dir=None, base_model=BASE_MODEL, revision=BASE_REVISION, device="cuda:0", merge_threads=6):
        from transformers import AutoModel, AutoVideoProcessor

        self.device = device
        self.processor = AutoVideoProcessor.from_pretrained(base_model, revision=revision)
        model = AutoModel.from_pretrained(base_model, revision=revision, dtype=torch.bfloat16)
        self.projection = None
        if ckpt_dir is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, ckpt_dir)
            # The adapters are merged into the bfloat16 weights on the CPU. The rounding of the merged weights depends on
            # the number of CPU threads; the paper's embeddings were produced with 6.
            n_threads = torch.get_num_threads()
            torch.set_num_threads(merge_threads)
            model = model.merge_and_unload()
            torch.set_num_threads(n_threads)
            proj = load_file(os.path.join(ckpt_dir, "projection.safetensors"))
            out_dim, in_dim = proj["weight"].shape
            self.projection = torch.nn.Linear(in_dim, out_dim)
            self.projection.load_state_dict(proj)
            self.projection = self.projection.to(device).eval()
        self.model = model.to(device).eval()

    @property
    def dim(self):
        return self.projection.out_features if self.projection is not None else self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, videos):
        """videos: list of uint8 arrays of shape (2, H, W, 3). Returns (B, D) float32, unit-norm rows."""
        inputs = self.processor(videos, return_tensors="pt")
        pixel = inputs["pixel_values_videos"].to(device=self.device, dtype=self.model.dtype)
        hidden = self.model(pixel_values_videos=pixel).last_hidden_state  # (B, N, 1024)
        vec = hidden.mean(dim=1).float()
        if self.projection is not None:
            vec = self.projection(vec)
        return torch.nn.functional.normalize(vec, dim=-1)


def load_rgb(path):
    """Load an image as an RGB array; None if the file is missing or corrupt."""
    try:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))
    except Exception as e:
        print(f"[warn] unreadable frame {path}: {e}")
        return None


def encode_frame_sequence(encoder, frame_paths, batch_size=64, gap=1):
    """Embed every pair (frame i, frame i + gap) of a sequence. Returns float16 (len - gap, D)."""
    pairs = [(frame_paths[i], frame_paths[i + gap]) for i in range(len(frame_paths) - gap)]
    out = []
    for s in range(0, len(pairs), batch_size):
        videos = []
        for a, b in pairs[s:s + batch_size]:
            ia, ib = load_rgb(a), load_rgb(b)
            if ia is None and ib is None:
                raise RuntimeError(f"both frames unreadable: {a}, {b}")
            # A corrupt frame is replaced by its partner (the pair then encodes a static transition).
            if ia is None:
                ia = ib
            if ib is None:
                ib = ia
            videos.append(np.stack([ia, ib]))
        out.append(encoder.encode(videos).cpu().numpy().astype(np.float16))
    return np.concatenate(out, axis=0)

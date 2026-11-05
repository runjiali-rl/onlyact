"""Action embedders that feed a precomputed per-transition vector to the AdaWorld world model (Table 2).

They replace AdaWorld's latent-action encoder. Each returns the two tensors the conditioner expects from an action
embedder: a (B, 1, D) cross-attention token and a (B, D) vector.

ExternalActionEmbedder          passes the vector through unchanged. Used by the no-action, AdaWorld-LAM (retrained) and
                                DreamDojo-LAM (retrained) world models (D = 32).
WhitenedExternalActionEmbedder  applies per-provider whitening first. Used by the OnlyAct world model (D = 640). The map
                                x -> ((x - mean) / std) @ W is fixed and parameter-free: mean, std and the symmetric matrix W
                                are measured once on the provider's own training-pool embeddings (compute_whitening.py) and
                                the same map is applied in training and in evaluation.

Neither embedder has trainable parameters or draws random numbers. The whitening statistics are non-persistent buffers,
so they are read from the statistics file at construction and never stored in a world-model checkpoint.
"""
import hashlib

import torch
import torch.nn as nn
from safetensors.torch import load_file

from vwm.modules.encoders.modules import AbstractEmbModel


class ExternalActionEmbedder(AbstractEmbModel):
    def __init__(self, in_dim: int, scale: float = 1.0):
        super().__init__()
        self.in_dim = self.out_dim = int(in_dim)
        self.scale = float(scale)
        self.mapping = nn.Identity()

    def forward(self, x: torch.Tensor):
        if x.ndim == 3 and x.shape[1] == 1:
            x = x[:, 0]
        assert x.ndim == 2 and x.shape[1] == self.in_dim, f"ext_action must be (B, {self.in_dim}), got {tuple(x.shape)}"
        z = self.mapping(x.float()) * self.scale
        if not torch.isfinite(z).all():
            raise FloatingPointError("ExternalActionEmbedder produced a non-finite vector")
        return [z.unsqueeze(1), z]  # cross-attention token, vector


class WhitenedExternalActionEmbedder(ExternalActionEmbedder):
    def __init__(self, in_dim: int, whitening_stats: str, whitening_sha256: str = "", std_floor: float = 1e-6):
        super().__init__(in_dim)
        if whitening_sha256:
            with open(whitening_stats, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            if digest != whitening_sha256:
                raise ValueError(f"whitening statistics {whitening_stats}: sha256 {digest} != expected {whitening_sha256}")
        stats = load_file(whitening_stats)
        mean, std, w = stats["mean"].float(), stats["std"].float(), stats["transform"].float()
        if mean.shape != (self.in_dim,) or std.shape != (self.in_dim,) or w.shape != (self.in_dim, self.in_dim):
            raise ValueError(f"whitening statistics {whitening_stats} do not match in_dim {self.in_dim}")
        if not torch.isfinite(w).all():
            raise ValueError("whitening transform is not finite")
        std = torch.where(std < std_floor, torch.ones_like(std), std)
        self.register_buffer("in_mean", mean, persistent=False)
        self.register_buffer("in_std", std, persistent=False)
        self.register_buffer("whitening_transform", w, persistent=False)
        print(f"[WhitenedExternalActionEmbedder] per-provider whitening from {whitening_stats}")

    def standardise(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and x.shape[1] == 1:
            x = x[:, 0]
        return (x.float() - self.in_mean) / self.in_std

    def forward(self, x: torch.Tensor):
        # The whitening runs in float32 under training mixed precision and evaluation autocast alike.
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = self.standardise(x) @ self.whitening_transform.float()
        return super().forward(z)

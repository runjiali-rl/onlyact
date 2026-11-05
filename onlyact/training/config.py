"""Training configuration of the OnlyAct encoder. The defaults are the released recipe (configs/onlyact.yaml)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import List


@dataclass
class TrainConfig:
    # encoder
    model_name: str = "facebook/vjepa2-vitl-fpc64-256"
    model_revision: str = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: List[str] = field(default_factory=lambda: ["query", "key", "value", "proj", "fc1", "fc2"])
    proj_dim: int = 640
    # objective
    tau: float = 0.07
    # optimisation
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    warmup_steps: int = 500
    total_steps: int = 10_000
    grad_clip: float = 1.0
    micro_batch: int = 2
    grad_accum: int = 8
    seed: int = 0
    # data
    val_frac: float = 0.02
    epoch_multiplier: int = 4
    bg_max_motion_px: float = 0.5
    feather_px: float = 0.0
    core_shift_min_px: int = 16
    num_workers: int = 2
    # logging and checkpoints
    log_every: int = 20
    save_every: int = 2000

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        known = {f_.name for f_ in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
        cfg = cls(**raw)
        cfg.lr = float(cfg.lr)
        cfg.betas = [float(b) for b in cfg.betas]
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

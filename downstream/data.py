"""Frame loading and fine-tuning batches for Table 3.

Two frame sources, both giving float32 RGB in [0, 1] at 256 x 256, (7, 3, 256, 256):
  clip  (DMLab, Minecraft): a seven-frame native-resolution uint8 clip (.npz, key "video"); each frame is centre-cropped
        to a square, bicubic-resized to 256 in float32 and clamped (no re-quantisation);
  png   (Habitat, nuScenes): seven 256 x 256 RGB PNGs, divided by 255 in float32.
Training batches apply the world model's original augmentation (brightness jitter, random context length, context
augmentation level, conditioning-frame noise) with per-example seeds, one example per micro-batch.
"""
from __future__ import annotations

import hashlib
import random
from collections import OrderedDict

import numpy as np

from .common import seed_for, stable_hash


def preprocess_rgb256(rgb):
    """Native uint8 HWC RGB -> float32 (3, 256, 256): centre square crop, bicubic resize, clamp to [0, 1]."""
    import torch
    import torch.nn.functional as F
    h, w = rgb.shape[:2]; side = min(h, w)
    x = torch.from_numpy(np.ascontiguousarray(rgb[(h-side)//2:(h-side)//2+side, (w-side)//2:(w-side)//2+side])).float().permute(2, 0, 1)[None] / 255
    if side != 256:
        x = F.interpolate(x, size=256, mode="bicubic", align_corners=False)
    return x[0].clamp(0, 1).contiguous()


def _content_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_clip_frames(task, check=True):
    import torch
    clip = task["clip"]
    with np.load(clip["abs_path"], allow_pickle=False) as payload:
        video = payload["video"]
    if video.dtype != np.uint8 or video.shape[0] != 7:
        raise ValueError("expected a seven-frame uint8 clip: " + clip["abs_path"])
    if check and "video_sha256" in clip and _content_sha(video) != clip["video_sha256"]:
        raise ValueError("clip pixels differ from the released clip (check the preparation): " + clip["abs_path"])
    return torch.stack([preprocess_rgb256(frame) for frame in video])


def load_png_frames(task, check=True):
    import torch
    from PIL import Image
    frames = task["target_frames"]; paths = frames["abs_paths"][:7]
    if check and "pixels_sha256" in frames:
        for p, h in zip(paths, frames["pixels_sha256"][:7]):
            with Image.open(p) as im:
                if _content_sha(np.array(im.convert("RGB"))) != h:
                    raise ValueError("frame pixels differ from the released frame (check the preparation): " + p)
    rows = []
    for p in paths:
        with Image.open(p) as im:
            rows.append(np.array(im, dtype=np.float32) / np.float32(255.))
    return torch.from_numpy(np.stack(rows)).permute(0, 3, 1, 2).contiguous()


def frame_loader(cfg, check=True):
    fn = {"clip": load_clip_frames, "png": load_png_frames}[cfg["frames"]]
    return lambda task: fn(task, check)


class _FrameCache:
    """Bounded LRU cache of decoded clips (128 MiB), only to avoid re-reading files."""

    def __init__(self, load, max_bytes=128 * 1024 * 1024):
        self.load, self.cache, self.bytes, self.max_bytes = load, OrderedDict(), 0, max_bytes

    def get(self, key, task):
        if key in self.cache:
            self.cache.move_to_end(key); return self.cache[key]
        frames = self.load(task); size = frames.numel() * frames.element_size()
        if size <= self.max_bytes:
            while self.bytes + size > self.max_bytes:
                _, old = self.cache.popitem(last=False); self.bytes -= old.numel() * old.element_size()
            self.cache[key] = frames; self.bytes += size
        return frames


class DiscreteTrainingData:
    """DMLab / Minecraft / Habitat: integer action of the last transition (index 5), target frame 6."""

    def __init__(self, tasks, load):
        self.tasks = list(tasks); self.frames = _FrameCache(load)

    def sample(self, index, seed):
        import torch
        rng = random.Random(seed); generator = torch.Generator(device="cpu").manual_seed(seed)
        frames = (self.frames.get(index, self.tasks[index]) + torch.rand(1, generator=generator) * .2 - .1).clamp(0, 1)
        length = rng.choices(range(1, 7))[0]
        prev = frames[:6].clone(); prev[:6-length] = 0
        prev = prev * 2 - 1; target = frames[6] * 2 - 1
        return {"img_seq": torch.cat([prev, target[None]]),
                "cond_frames_without_noise": prev[-1],
                "cond_frames": prev[-1] + .02 * torch.randn(prev[-1].shape, generator=generator),
                "raw_action": torch.tensor([self.tasks[index]["raw_action"]], dtype=torch.long),
                "context_len": torch.tensor([length], dtype=torch.float32),
                "context_aug": torch.tensor([rng.choices(range(8))[0] / 10], dtype=torch.float32)}

    def batch(self, indices, seed):
        import torch
        rows = [self.sample(int(index), int(stable_hash("paper-batch", seed, i)[:8], 16)) for i, index in enumerate(indices)]
        return {key: torch.stack([r[key] for r in rows]) for key in rows[0]}


class ContinuousTrainingData:
    """nuScenes: ego-motion action (forward, lateral metres) of the last transition (index 5), target frame 6."""

    def __init__(self, tasks, load):
        self.tasks = list(tasks); self.frames = _FrameCache(load)

    def sample(self, index, seed):
        import torch
        g = torch.Generator(device="cpu").manual_seed(seed)
        frames = (self.frames.get(index, self.tasks[index]) + (torch.rand(1, generator=g) * .2 - .1)).clamp(0, 1)
        length = int(torch.randint(1, 7, (1,), generator=g).item()); prev = frames[:6].clone()
        prev[:6-length] = 0.; prev = prev * 2 - 1; target = frames[6] * 2 - 1
        return {"img_seq": torch.cat([prev, target[None]]), "cond_frames_without_noise": prev[-1],
                "cond_frames": prev[-1] + .02 * torch.randn(prev[-1].shape, generator=g),
                "raw_action": torch.tensor(self.tasks[index]["raw_action"], dtype=torch.float32),
                "context_len": torch.tensor([length], dtype=torch.float32),
                "context_aug": torch.randint(0, 8, (1,), generator=g).float() / 10}

    def batch(self, indices, seed):
        import torch
        rows = [self.sample(int(i), seed_for(seed, "sample", j)) for j, i in enumerate(indices)]
        return {k: torch.stack([r[k] for r in rows]) for k in rows[0]}


def training_data(cfg, tasks, check=True):
    cls = ContinuousTrainingData if cfg["action_type"] == "continuous" else DiscreteTrainingData
    return cls(tasks, frame_loader(cfg, check))

"""Training data for the AdaWorld world model with external actions (Table 2 world models).

Each sample follows AdaWorld's VideoDataset.build_data_dict: 7 frames = a 6-slot short-term memory with a random context
length 1..6 (unused slots zero-padded) plus one target frame, together with context_len and context_aug. AdaWorld's
latent-action input is replaced by a precomputed per-transition vector `ext_action`, read from <emb_dir>/<task_id>.npy,
whose row j embeds the frame pair (j, j + gap). The predicted transition is frame s+(c-1)*gap -> frame s+c*gap and uses row
s+(c-1)*gap. With representation None every vector is zero (the no-action world model).

Frames come from a clip list (see clip_lists.py) whose paths are relative to a data root given at run time.
"""
import os
from random import choices, randint
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from clip_lists import load_clip_list


class ExternalActionClipDataset(Dataset):
    def __init__(
            self,
            manifest: str,
            data_root: str,
            emb_dir: Optional[str],
            representation: Optional[str],
            action_dim: int,
            gap: int = 1,
            resolution: int = 256,
            n_context_frames: int = 6,
            color_aug: bool = True,
            name: str = ""
    ):
        super().__init__()
        self.name = name or os.path.basename(manifest)
        self.emb_dir, self.representation, self.action_dim, self.gap = emb_dir, representation, int(action_dim), int(gap)
        self.resolution, self.n_context_frames, self.color_aug = resolution, n_context_frames, color_aug
        span = self.n_context_frames * self.gap  # frames s .. s+span must exist
        tasks = [{"task_id": t["task_id"], "abs_paths": t["target_frames"]["abs_paths"], "scene_id": t.get("scene_id")}
                 for t in load_clip_list(manifest, data_root)["tasks"]]
        tasks = [t for t in tasks if len(t["abs_paths"]) >= 2 and len(t["abs_paths"]) > span]
        if self.representation is not None:
            kept, missing, bad = [], 0, 0
            for t in tasks:
                if not os.path.isfile(os.path.join(emb_dir, t["task_id"] + ".npy")):
                    missing += 1
                    continue
                kept.append(t)
            # the layout is checked on the first files here and on every read in __getitem__
            for t in kept[:16]:
                a = np.load(os.path.join(emb_dir, t["task_id"] + ".npy"), mmap_mode="r")
                if a.ndim != 2 or a.shape[1] != self.action_dim or a.shape[0] != len(t["abs_paths"]) - self.gap:
                    bad += 1
            if bad:
                raise ValueError(f"{self.name}/{representation}: {bad} of the first 16 embedding files violate the layout "
                                 f"(n_frames - gap, {self.action_dim}) with gap {self.gap}")
            if missing:
                print(f"[external_action] {self.name}/{representation}: {missing} clips without an embedding file dropped")
            tasks = kept
        if not tasks:
            raise ValueError(f"{self.name}/{representation}: no usable clip")
        self.tasks = tasks
        print(f"[external_action] {self.name}/{representation}: {len(tasks)} clips, gap {self.gap}, action_dim {self.action_dim}")

    def __len__(self) -> int:
        return len(self.tasks)

    def _load_frame(self, path: str) -> torch.Tensor:
        im = Image.open(path).convert("RGB")
        x = torch.from_numpy(np.asarray(im)).float() / 255.0  # (H, W, 3)
        h, w = x.shape[:2]
        if h != w:  # centre crop to a square
            s = min(h, w)
            x = x[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        x = x.permute(2, 0, 1)[None]  # (1, 3, S, S)
        if x.shape[-1] != self.resolution:
            x = F.interpolate(x, self.resolution, mode="bicubic")
        return x[0]

    def __getitem__(self, idx: int) -> Dict:
        t = self.tasks[idx]
        n = len(t["abs_paths"])
        span = self.n_context_frames * self.gap
        s = randint(0, n - 1 - span)
        frames = [self._load_frame(t["abs_paths"][s + k * self.gap]) for k in range(self.n_context_frames + 1)]
        if self.color_aug:
            j = torch.rand(1) * 0.2 - 0.1
            frames = [(f + j).clamp(0, 1) for f in frames]
        context_len = randint(1, self.n_context_frames)
        filled = [torch.zeros_like(frames[0])] * (self.n_context_frames - context_len) + frames
        prev_frames = torch.stack(filled[:self.n_context_frames]) * 2.0 - 1.0
        next_frame = filled[self.n_context_frames] * 2.0 - 1.0
        row = s + (context_len - 1) * self.gap  # transition filled[5] -> filled[6] = frames[c-1] -> frames[c]
        if self.representation is None:
            act = torch.zeros(self.action_dim)
        else:
            a = np.load(os.path.join(self.emb_dir, t["task_id"] + ".npy"), mmap_mode="r")
            if a.shape[0] != n - self.gap or a.shape[1] != self.action_dim:
                raise ValueError(f"{t['task_id']}: embedding shape {a.shape} != ({n - self.gap}, {self.action_dim})")
            act = torch.from_numpy(np.array(a[row], dtype=np.float32))
            if not torch.isfinite(act).all():
                raise ValueError(f"{t['task_id']} row {row}: non-finite embedding")
        return {
            "img_seq": torch.cat([prev_frames, next_frame[None]]),  # (7, 3, R, R)
            "cond_frames_without_noise": prev_frames[-1],
            "cond_frames": prev_frames[-1] + 0.02 * torch.randn_like(prev_frames[-1]),
            "ext_action": act,  # (D,)
            "context_len": torch.Tensor([context_len]),
            "context_aug": torch.Tensor(choices(range(8))) / 10
        }


class MultiSourceExternalActionDataset(Dataset):
    """Samples a source with probability proportional to len ** 0.43 (AdaWorld's 'pi' rule), then a random clip of it."""

    def __init__(self, sources: List[Dict], samples_per_epoch: int, sampling_strategy: str = "pi", **kwargs):
        self.samples_per_epoch = samples_per_epoch
        self.subsets = [ExternalActionClipDataset(**src, **kwargs) for src in sources]
        if sampling_strategy == "sample":
            probs = [len(d) for d in self.subsets]
        elif sampling_strategy == "dataset":
            probs = [1 for _ in self.subsets]
        elif sampling_strategy == "pi":
            probs = [len(d) ** 0.43 for d in self.subsets]
        else:
            raise ValueError(f"Unavailable sampling strategy: {sampling_strategy}")
        tot = sum(probs)
        self.sample_probs = [p / tot for p in probs]
        print("[external_action] sources:", [(d.name, len(d), round(p, 3)) for d, p in zip(self.subsets, self.sample_probs)])

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict:
        subset = choices(self.subsets, self.sample_probs)[0]
        return subset[randint(0, len(subset) - 1)]


class ExternalActionDataSampler(LightningDataModule):
    def __init__(
            self,
            sources: List[Dict],
            representation: Optional[str],
            action_dim: int,
            gap: int = 1,
            batch_size: int = 1,
            num_workers: int = 8,
            resolution: int = 256,
            n_context_frames: int = 6,
            prefetch_factor: int = 4,
            shuffle: bool = True,
            samples_per_epoch: int = 60000,
            sampling_strategy: str = "pi",
            color_aug: bool = True
    ):
        super().__init__()
        self.batch_size, self.num_workers, self.shuffle = batch_size, num_workers, shuffle
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.train_dataset = MultiSourceExternalActionDataset(
            [dict(s) for s in sources], samples_per_epoch=samples_per_epoch, sampling_strategy=sampling_strategy,
            representation=representation, action_dim=action_dim, gap=gap, resolution=resolution,
            n_context_frames=n_context_frames, color_aug=color_aug)

    def _loader(self, shuffle: bool):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor, persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self._loader(self.shuffle)

    def val_dataloader(self):
        return self._loader(False)

    def test_dataloader(self):
        return self._loader(False)

"""Training / validation samples of the scene-leakage decoder (Table 4).

One sample per (clip, frame pair): the embedding row of the pair and the frame AFTER the action as a 3x128x128 tensor in
[-1, 1]. Embedding files are <emb_dir>/<task_id>.npy with one row per consecutive frame pair, so row i describes frames
(i, i + 1) and its target is target_frames.abs_paths[i + 1].
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LeakageDataset(Dataset):
    def __init__(self, task_ids, tasks, emb_dir, resolution=128):
        """task_ids: clips of this split; tasks: the clip list loaded by justact.data.load_manifest (frame paths resolved)."""
        tasks_by_id = {t["task_id"]: t for t in tasks}
        self.tasks = [tasks_by_id[tid] for tid in task_ids if tid in tasks_by_id]
        self.embed_dir = emb_dir
        self.resolution = resolution

        self.index = []          # (task position, pair index) per sample
        self.task_stride = []
        self._embed_cache = {}
        for ti, t in enumerate(self.tasks):
            emb_path = os.path.join(self.embed_dir, f"{t['task_id']}.npy")
            if not os.path.isfile(emb_path):
                raise FileNotFoundError(f"missing embedding {emb_path}")
            shape = np.load(emb_path, mmap_mode="r").shape
            if len(shape) != 2 or shape[0] < 1:
                raise ValueError(f"{emb_path}: expected a (pairs, D) array, got shape {shape}")
            n_pairs = shape[0]
            n_abs = len(t["target_frames"]["abs_paths"])
            # one row per consecutive pair gives stride 1; fewer rows than pairs are spread evenly over the clip
            if n_abs <= n_pairs:
                stride = 1
                usable = min(n_pairs, n_abs - 1)
            else:
                stride = (n_abs - 1) // n_pairs
                usable = min(n_pairs, (n_abs - 1) // max(stride, 1))
            if usable < 1:
                raise ValueError(f"{t['task_id']}: no usable frame pair")
            for pi in range(usable):
                self.index.append((ti, pi))
            self.task_stride.append(stride)

        probe = np.load(os.path.join(self.embed_dir, f"{self.tasks[0]['task_id']}.npy"), mmap_mode="r")
        self.embed_dim = int(probe.shape[1])

    def __len__(self):
        return len(self.index)

    def _load_embedding(self, task_idx):
        arr = self._embed_cache.get(task_idx)
        if arr is None:
            tid = self.tasks[task_idx]["task_id"]
            arr = np.load(os.path.join(self.embed_dir, f"{tid}.npy"))
            if len(self._embed_cache) > 256:
                self._embed_cache.clear()
            self._embed_cache[task_idx] = arr
        return arr

    def __getitem__(self, idx):
        task_idx, pair_idx = self.index[idx]
        task = self.tasks[task_idx]

        emb = self._load_embedding(task_idx)[pair_idx].astype(np.float32)

        stride = self.task_stride[task_idx]
        post_idx = min(stride * (pair_idx + 1),
                       len(task["target_frames"]["abs_paths"]) - 1)
        post_path = task["target_frames"]["abs_paths"][post_idx]
        img = Image.open(post_path).convert("RGB")
        if img.size != (self.resolution, self.resolution):
            img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
        frame = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        return {"embedding": torch.from_numpy(emb), "frame": frame}


def collate(batch):
    return {
        "embedding": torch.stack([b["embedding"] for b in batch]),
        "frame": torch.stack([b["frame"] for b in batch]),
    }

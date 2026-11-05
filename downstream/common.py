"""Shared helpers for the Table 3 scripts: random-number streams, the fixed sample order, hashing and clip lists.

Every random number used by fine-tuning and generation comes from a stream derived here, so a run is fully determined by
the seed in the domain config (32) and the clip order in the lists under data/downstream/.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

ARMS = ("adaworld_lam_retrained", "dreamdojo_lam_retrained", "onlyact")
BASELINES = ("adaworld_lam_retrained", "dreamdojo_lam_retrained")
OURS = "onlyact"
REPO = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_for(seed, stream, index=0):
    """Seed of one named random stream (data augmentation, diffusion noise, sample order, head fitting)."""
    return int.from_bytes(hashlib.sha256(f"adapt-v1:{seed}:{stream}:{index}".encode()).digest()[:4], "little")


def index_plan(count, steps, batch, seed, stream):
    """(steps, batch) clip indices: concatenated shuffled epochs over all clips, so no label is dropped."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed_for(seed, stream)); ids = []
    while len(ids) < steps * batch:
        ids.extend(torch.randperm(count, generator=g).tolist())
    return torch.tensor(ids[:steps * batch], dtype=torch.int64).reshape(steps, batch)


def tensor_digest(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def sampling_seed(task_id):
    """Per-test-case generation seed; reset before every generated action, so all actions and arms share noise."""
    return int.from_bytes(hashlib.sha256(f"paper-eval-v1:32:{task_id}:6".encode()).digest()[:4], "little")


def stable_hash(*parts):
    return hashlib.sha256(json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def load_config(path):
    with open(path) as f:
        return json.load(f)


def repo_path(p):
    """Config paths are relative to the repository root."""
    return p if os.path.isabs(p) else str(REPO / p)


def load_clips(path, data_root):
    """Clip list -> tasks with absolute frame / clip paths (lists store paths relative to the data root)."""
    with open(repo_path(path)) as f:
        tasks = json.load(f)["tasks"]
    for t in tasks:
        if "clip" in t:
            t["clip"]["abs_path"] = os.path.join(data_root, t["clip"]["path"])
        if "target_frames" in t:
            t["target_frames"]["abs_paths"] = [os.path.join(data_root, p) for p in t["target_frames"]["paths"]]
    return tasks


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)

"""Held-out-scene split of a clip list for the scene-leakage decoder (Table 4).

The clips are grouped into scenes. A seeded shuffle of the sorted scene keys selects val_frac of the scenes, and every clip
of those scenes is held out for validation. The decoder seed is also the split seed, so the 4 decoder seeds of Table 4 use
4 different sets of held-out scenes; for a given dataset and seed all representations share the same split.
"""
import json
import os
import re

import numpy as np

_LIBERO_SCENE = re.compile(r"(libero_\d+/[A-Z_]+SCENE\d+)")


def scene_key(task):
    """Scene of a clip: its "scene_key" if set; otherwise the LIBERO scene of its "scene_id" (several LIBERO tasks share
    one scene, e.g. libero_10/KITCHEN_SCENE3); otherwise "scene_id" itself (RLBench: the task)."""
    if task.get("scene_key"):
        return str(task["scene_key"])
    sid = task.get("scene_id", "")
    m = _LIBERO_SCENE.match(sid)
    return m.group(1) if m else sid


def build_scene_split(tasks, seed=0, val_frac=0.2):
    """Deterministic scene-level split of a list of clips. Task order is kept inside "train" and "val"."""
    scenes = sorted({scene_key(t) for t in tasks})
    rng = np.random.default_rng(seed)
    rng.shuffle(scenes)
    n_val = max(1, int(round(len(scenes) * val_frac)))
    val_scenes = set(scenes[:n_val])
    split = {"train": [], "val": []}
    for t in tasks:
        k = "val" if scene_key(t) in val_scenes else "train"
        split[k].append(t["task_id"])
    split["val_scenes"] = sorted(val_scenes)
    split["meta"] = {
        "seed": seed, "val_frac": val_frac,
        "n_scenes_total": len(scenes), "n_val_scenes": len(val_scenes),
        "n_train_tasks": len(split["train"]), "n_val_tasks": len(split["val"]),
    }
    return split


def save_split(split, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)

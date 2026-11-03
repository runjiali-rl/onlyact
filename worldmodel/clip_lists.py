"""Clip lists for the world-model experiments (Table 2).

A clip list is a JSON file (optionally gzip-compressed, *.json.gz) with the clip-list schema used across this repository:
{"tasks": [{"task_id", "scene_id", "target_frames": {"paths": [...], "frame_indices": [...], "n_frames"}}, ...]}.
"paths" are relative to a dataset root given at run time; load_clip_list() adds "target_frames.abs_paths".

Evaluation cells (data/worldmodel/<cell>.json) also carry "cell", "dataset", "mode" ("self" or "transfer") and "pairs",
the ordered list of {"source", "target"} rollouts. For self cells source == target. The rollout order matters: the
sampler's random stream is consumed clip after clip, so the same numbers need the same order.
"""
import gzip
import json
import os
from typing import Dict, Optional


def read_clip_list(path: str) -> Dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def load_clip_list(path: str, data_root: Optional[str]) -> Dict:
    clip_list = read_clip_list(path)
    for t in clip_list["tasks"]:
        tf = t["target_frames"]
        if "paths" in tf:
            if data_root is None:
                raise ValueError(f"{path} stores frame paths relative to a data root: pass --data_root")
            tf["abs_paths"] = [os.path.join(data_root, p) for p in tf["paths"]]
    return clip_list


def rollout_pairs(cell: Dict):
    """Ordered (source, target) task ids of an evaluation cell."""
    return [(p["source"], p["target"]) for p in cell["pairs"]]

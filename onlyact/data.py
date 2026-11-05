"""Clip lists (manifests) with frame paths relative to a data root.

A clip list is a JSON file {"tasks": [task, ...]}. Each task has a "task_id", the held-out unit fields
"scene_id" / "scene_key", and "target_frames" = {"paths": [...], "frame_indices": [...], "n_frames": int}.
"paths" are relative to the dataset root given at run time; load_manifest() adds the resolved
"abs_paths" that the extraction and evaluation code read. Task order is meaningful and is kept.
"""
from __future__ import annotations

import json
import os


def resolve_tasks(tasks, data_root):
    for t in tasks:
        tf = t.get("target_frames", {})
        if "paths" in tf:
            if data_root is None:
                raise ValueError("this clip list stores relative frame paths: pass --data_root")
            tf["abs_paths"] = [os.path.join(data_root, p) for p in tf["paths"]]
    return tasks


def load_manifest(path, data_root=None):
    """Return the list of tasks of a clip list, with target_frames.abs_paths resolved against data_root."""
    with open(path) as f:
        manifest = json.load(f)
    tasks = manifest["tasks"] if isinstance(manifest, dict) else manifest
    return resolve_tasks(tasks, data_root)

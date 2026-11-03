"""Shared helpers of the probe-data preparation scripts (Table 1 frames and action targets, Table 4 128x128 frames).

Every preparation script reads the clip list data/probe/<dataset>.json (task order, relative frame paths, frame
indices and a per-clip `source` record with the public identifiers of the raw data) and writes

    <out>/probe/<dataset>/<path of the clip list>        frames read by the encoders (Tables 1 and 2)
    <out>/leakage128/<dataset>/<path of the clip list>   128x128 frames read by the leakage decoder (Table 4)
    <actions_out>/probe/<dataset>/<target>/<task_id>.npy action targets of the linear probe (Table 1)

<actions_out> defaults to <out>/eval_data, so EVAL_DATA=<out>/eval_data works with scripts/reproduce_table1.sh.
Files that already exist are kept (pass --overwrite to rewrite them); every file is written atomically.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))


def parser(dataset, raw_help: str) -> argparse.ArgumentParser:
    """dataset: a dataset name, or a list of names selected with --dataset."""
    name = dataset if isinstance(dataset, str) else " / ".join(dataset)
    ap = argparse.ArgumentParser(description=f"Prepare the {name} probe clips (frames, 128x128 frames, action targets).")
    if not isinstance(dataset, str):
        ap.add_argument("--dataset", required=True, choices=list(dataset))
    if raw_help:
        ap.add_argument("--raw", required=True, help=raw_help)
    ap.add_argument("--out", required=True, help="data root: writes <out>/probe/<dataset>/ and <out>/leakage128/<dataset>/")
    ap.add_argument("--actions_out", default=None, help="action root (default <out>/eval_data)")
    ap.add_argument("--clips", default=None, help="clip list (default data/probe/<dataset>.json)")
    ap.add_argument("--task_ids", nargs="*", default=None, help="prepare only these clips")
    ap.add_argument("--limit", type=int, default=0, help="prepare only the first N clips of the list (0 = all)")
    ap.add_argument("--overwrite", action="store_true", help="rewrite files that already exist")
    ap.set_defaults(dataset=dataset if isinstance(dataset, str) else None)
    return ap


def clips_path(args) -> str:
    return args.clips or os.path.join(REPO, "data", "probe", f"{args.dataset}.json")


class Outputs:
    """Output locations of one dataset."""

    def __init__(self, args, dataset: str):
        self.dataset = dataset
        self.overwrite = args.overwrite
        self.frames = os.path.join(args.out, "probe", dataset)
        self.frames128 = os.path.join(args.out, "leakage128", dataset)
        target = json.load(open(os.path.join(REPO, "data", "probe", "datasets.json")))[dataset]["target"]
        self.actions = os.path.join(args.actions_out or os.path.join(args.out, "eval_data"), "probe", dataset, target)
        os.makedirs(self.actions, exist_ok=True)

    def frame(self, rel: str) -> str:
        return os.path.join(self.frames, rel)

    def frame128(self, rel: str) -> str:
        return os.path.join(self.frames128, rel)

    def action(self, task_id: str) -> str:
        return os.path.join(self.actions, f"{task_id}.npy")

    def todo(self, *paths: str) -> bool:
        return self.overwrite or not all(os.path.exists(p) for p in paths)

    def link_frames128_to_frames(self) -> None:
        """For datasets whose frames are already 128x128 (LIBERO, RLBench): leakage128/<dataset> -> probe/<dataset>."""
        os.makedirs(os.path.dirname(self.frames128), exist_ok=True)
        if not os.path.lexists(self.frames128):
            os.symlink(os.path.relpath(self.frames, os.path.dirname(self.frames128)), self.frames128)


def load_tasks(args) -> list[dict]:
    tasks = json.load(open(clips_path(args)))["tasks"]
    if args.task_ids:
        keep = set(args.task_ids)
        tasks = [t for t in tasks if t["task_id"] in keep]
        missing = keep - {t["task_id"] for t in tasks}
        if missing:
            raise SystemExit(f"task ids not in {clips_path(args)}: {sorted(missing)[:5]}")
    if args.limit:
        tasks = tasks[: args.limit]
    return tasks


def _mkdir_for(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_pil(im, path: str, **kw) -> None:
    """Atomic PIL save (the temporary name keeps the extension, so the format is the same)."""
    _mkdir_for(path)
    root, ext = os.path.splitext(path)
    tmp = f"{root}.tmp{ext}"
    im.save(tmp, **kw)
    os.replace(tmp, path)


def save_cv2(img, path: str) -> None:
    import cv2
    _mkdir_for(path)
    root, ext = os.path.splitext(path)
    tmp = f"{root}.tmp{ext}"
    if not cv2.imwrite(tmp, img):
        raise IOError(f"cv2.imwrite failed for {path}")
    os.replace(tmp, path)


def save_bytes(data: bytes, path: str) -> None:
    _mkdir_for(path)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def save_npy(path: str, arr: np.ndarray) -> None:
    _mkdir_for(path)
    tmp = path[:-4] + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def save_thumbnail_frame(frame: np.ndarray, path: str, path128: str) -> None:
    """VP2 and Procgen frames: long side at most 256 (PIL thumbnail, bicubic), 128x128 copy bicubic from that image."""
    from PIL import Image
    im = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    if max(im.size) > 256:
        im.thumbnail((256, 256), Image.BICUBIC)
    save_pil(im, path)
    save_pil(im.resize((128, 128), Image.BICUBIC), path128)


def log_done(dataset: str, n: int, out: Outputs) -> None:
    print(f"[{dataset}] {n} clips -> {out.frames} | {out.frames128} | {out.actions}", flush=True)

#!/usr/bin/env python3
"""Prepare training data of the OnlyAct encoder: DROID training manifest.

Reads the episode folders written by droid_download_and_process.py under
<data_root>/<subdir>/<lab>/success/<date>/<timestamp>/ and writes one task per episode.
Each task uses the frames of the first exterior camera (varied_camera_1) and samples
120 evenly spaced frame indices over the episode (all frames if it is shorter).

With --task_ids, only the listed episodes are kept, in the order of the list.
droid_task_ids.txt holds the 606 episodes used for training (chosen by an
object-grounding filter that is not part of this release). Each line is
"<task_id>\\t<n_source_frames>": the second column is the number of camera frames the
120 indices are spread over, so the frame lists match the training run even if a
re-extraction yields a different frame count.

The training manifest of the paper stores the same scene id, "TRI/2023-10-26", for all
606 DROID tasks, and that is the default here (it decides which samples can donate a background
during training). --scene_id per_episode writes "<lab>/<date>" for each task instead.

    python data/training/droid_build_manifest.py --data_root <data> --task_ids data/training/droid_task_ids.txt

Output (default <data_root>/<subdir>/manifest_training.json):
    {"stage": "droid_training", "dataset": "droid", "tasks": [{"task_id", "dataset", "scene_id",
     "task_type", "ep_dir", "target_frames": {"paths": [relative to data_root],
     "frame_indices", "n_frames"}}]}
"""

import argparse
import glob
import json
import os

import numpy as np

N_TARGET_FRAMES = 120
# Camera folders in fallback order; varied_camera_1 is used whenever it has frames.
CAMERA_NAMES = ["hand_camera", "varied_camera_1", "varied_camera_2"]
PRIMARY_CAMERA = "varied_camera_1"


def droid_task_id(lab: str, date: str, timestamp: str) -> str:
    """Task id of an episode (same rule as droid_download_and_process.py)."""
    task_id = f"droid_{lab}_{date}_{timestamp}"
    return task_id.replace(":", "-").replace(" ", "_").replace("+", "_")


def discover_episodes(episodes_root: str, labs: list | None = None) -> list:
    """All finished episodes ({lab}/success/{date}/{timestamp}/_done with trajectory.h5)."""
    episodes = []
    pattern = os.path.join(episodes_root, "*", "success", "*", "*", "_done")
    for done_path in sorted(glob.glob(pattern)):
        ep_dir = os.path.dirname(done_path)
        parts = ep_dir.split(os.sep)
        timestamp = parts[-1]
        date = parts[-2]
        lab = parts[-4]
        if labs and lab not in labs:
            continue
        if not os.path.exists(os.path.join(ep_dir, "trajectory.h5")):
            continue
        episodes.append({"ep_dir": ep_dir, "lab": lab, "date": date, "timestamp": timestamp})
    return episodes


def primary_frames(ep_dir: str) -> list:
    """Sorted frame paths of varied_camera_1, else of the first camera that has frames."""
    camera_frames = {}
    for cam in CAMERA_NAMES:
        cam_dir = os.path.join(ep_dir, cam)
        if os.path.isdir(cam_dir):
            frames = sorted(glob.glob(os.path.join(cam_dir, "frame_*.jpg")))
            if frames:
                camera_frames[cam] = frames
    if not camera_frames:
        return []
    cam = PRIMARY_CAMERA if PRIMARY_CAMERA in camera_frames else list(camera_frames.keys())[0]
    return camera_frames[cam]


def sample_target_frames(n_frames: int, n_target: int = N_TARGET_FRAMES) -> list:
    """n_target evenly spaced indices over [0, n_frames - 1] (all indices if fewer frames)."""
    if n_frames <= n_target:
        return list(range(n_frames))
    return np.linspace(0, n_frames - 1, n_target, dtype=int).tolist()


def read_task_list(path: str) -> list:
    """Rows (task_id, n_source_frames or None) from a whitespace-separated text file."""
    rows = []
    with open(path) as f:
        for line in f:
            cols = line.split()
            if not cols:
                continue
            rows.append((cols[0], int(cols[1]) if len(cols) > 1 else None))
    return rows


def build_task(ep: dict, frames: list, data_root: str, scene_id: str | None) -> dict:
    indices = sample_target_frames(len(frames))
    return {
        "task_id": droid_task_id(ep["lab"], ep["date"], ep["timestamp"]),
        "dataset": "droid",
        "scene_id": scene_id if scene_id != "per_episode" else f"{ep['lab']}/{ep['date']}",
        "task_type": "manipulation",
        "ep_dir": os.path.relpath(ep["ep_dir"], data_root),
        "target_frames": {
            "paths": [os.path.relpath(frames[i], data_root) for i in indices],
            "frame_indices": indices,
            "n_frames": len(indices),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build the DROID training manifest.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder that holds all processed datasets")
    parser.add_argument("--subdir", type=str, default="droid_processed",
                        help="DROID folder inside data_root")
    parser.add_argument("--labs", nargs="*", default=None,
                        help="Labs to include (default: all)")
    parser.add_argument("--task_ids", type=str, default=None,
                        help="Task list (task_id [n_source_frames] per line); keeps and orders tasks by it")
    parser.add_argument("--scene_id", type=str, default="TRI/2023-10-26",
                        help="Scene id written for every task (default: the paper's value); 'per_episode' = <lab>/<date>")
    parser.add_argument("--out", type=str, default=None,
                        help="Output manifest (default <data_root>/<subdir>/manifest_training.json)")
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    episodes_root = os.path.join(data_root, args.subdir)
    out_path = args.out or os.path.join(episodes_root, "manifest_training.json")

    episodes = discover_episodes(episodes_root, args.labs)
    print(f"found {len(episodes)} finished episodes under {args.subdir}")
    by_id = {droid_task_id(ep["lab"], ep["date"], ep["timestamp"]): ep for ep in episodes}

    tasks = []
    if args.task_ids:
        rows = read_task_list(args.task_ids)
        missing = [tid for tid, _ in rows if tid not in by_id]
        if missing:
            raise SystemExit(f"{len(missing)} listed task ids not found, e.g. {missing[:3]}")
        n_changed = 0
        for tid, n_src in rows:
            frames = primary_frames(by_id[tid]["ep_dir"])
            if n_src is not None:
                if len(frames) < n_src:
                    raise SystemExit(f"{tid}: {len(frames)} frames on disk, {n_src} listed")
                n_changed += len(frames) != n_src
                frames = frames[:n_src]
            if not frames:
                raise SystemExit(f"{tid}: no camera frames")
            tasks.append(build_task(by_id[tid], frames, data_root, args.scene_id))
        print(f"kept {len(tasks)} tasks listed in {args.task_ids} "
              f"({n_changed} episodes have more frames on disk than listed; the listed count is used)")
    else:
        for ep in episodes:
            frames = primary_frames(ep["ep_dir"])
            if frames:
                tasks.append(build_task(ep, frames, data_root, args.scene_id))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"stage": "droid_training", "dataset": "droid", "tasks": tasks}, f)
    print(f"wrote {len(tasks)} tasks -> {out_path}")


if __name__ == "__main__":
    main()

"""Prepare training data of the OnlyAct encoder: LIBERO training manifest.

Reads <data_root>/<subdir>/demos_manifest.json (written by libero_process.py)
and writes one task per demo with 41 target frames sampled uniformly over the
whole demo. With --task_ids, only the listed demos are kept, in the order of
the list. libero_task_ids.txt holds the 1,403 demos used for training (chosen
by an object-grounding filter that is not part of this release).

    python data/training/libero_build_manifest.py --data_root <data> --task_ids data/training/libero_task_ids.txt

Output (default <data_root>/<subdir>/manifest_training.json):
    {"stage": "libero_training", "tasks": [{"task_id", "dataset", "scene_id", ...,
     "target_frames": {"paths": [relative to data_root], "frame_indices", "n_frames"}}]}
"""

import argparse
import json
import os

N_TARGET_FRAMES = 41
MIN_DEMO_LEN = N_TARGET_FRAMES


def build_task(demo: dict) -> dict:
    """Return the training task of one demo, or None if the demo is too short."""
    paths = demo["all_frames"]["paths"]
    n_total = len(paths)
    if n_total < MIN_DEMO_LEN:
        return None
    # 41 indices spread uniformly from the first to the last frame.
    tgt_indices = [int(round(i * (n_total - 1) / (N_TARGET_FRAMES - 1)))
                   for i in range(N_TARGET_FRAMES)]
    return {
        "task_id": demo["task_id"],
        "dataset": "libero",
        "scene_id": demo["scene_id"],
        "task_type": "manipulation",
        "description": demo["description"],
        "split": demo.get("split", ""),
        "hdf5_file": demo.get("hdf5_file", ""),
        "demo_key": demo["demo_key"],
        "demo_idx": demo.get("demo_idx", 0),
        "target_frames": {
            "paths": [paths[i] for i in tgt_indices],
            "frame_indices": tgt_indices,
            "n_frames": N_TARGET_FRAMES,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build the LIBERO training manifest.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder that holds all processed datasets")
    parser.add_argument("--subdir", type=str, default="libero_processed",
                        help="LIBERO folder inside data_root")
    parser.add_argument("--demos", type=str, default=None,
                        help="Demo list from libero_process.py "
                             "(default <data_root>/<subdir>/demos_manifest.json)")
    parser.add_argument("--task_ids", type=str, default=None,
                        help="Text file with one task id per line; keeps and orders tasks by it")
    parser.add_argument("--out", type=str, default=None,
                        help="Output manifest (default <data_root>/<subdir>/manifest_training.json)")
    args = parser.parse_args()

    demos_path = args.demos or os.path.join(args.data_root, args.subdir, "demos_manifest.json")
    out_path = args.out or os.path.join(args.data_root, args.subdir, "manifest_training.json")

    with open(demos_path) as f:
        demos = json.load(f)["tasks"]

    tasks = []
    for demo in demos:
        task = build_task(demo)
        if task is not None:
            tasks.append(task)
    print(f"{len(tasks)} / {len(demos)} demos have at least {MIN_DEMO_LEN} frames")

    if args.task_ids:
        with open(args.task_ids) as f:
            keep = [line.strip() for line in f if line.strip()]
        by_id = {t["task_id"]: t for t in tasks}
        missing = [tid for tid in keep if tid not in by_id]
        if missing:
            raise SystemExit(f"{len(missing)} listed task ids not found, e.g. {missing[:3]}")
        tasks = [by_id[tid] for tid in keep]
        print(f"kept {len(tasks)} tasks listed in {args.task_ids}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"stage": "libero_training", "dataset": "libero", "tasks": tasks}, f)
    print(f"wrote {len(tasks)} tasks -> {out_path}")


if __name__ == "__main__":
    main()

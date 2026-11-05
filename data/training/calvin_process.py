"""Prepare training data of the OnlyAct encoder: CALVIN frames and manifest.

Input: the CALVIN task_ABC_D split (http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip,
see https://github.com/mees/calvin). CALVIN stores one .npz file per time step (rgb_static 200x200, rel_actions 7-D)
and the language-annotated windows in lang_annotations/auto_lang_ann.npy. Each
annotated window becomes one task:
  - its frames (subsampled to at most --num_target_frames) are saved as PNG,
  - its rel_actions (delta position 3 + delta Euler 3 + gripper 1) are saved as .npy,
  - a manifest entry lists the frames with paths relative to --data_root.

Output layout under <data_root>/<subdir>/:
    frames/<task_id>/frame_<j:04d>.png
    actions/<task_id>/actions.npy, actions/index.json
    manifest_<split>.json

Tasks that already have actions.npy are not re-extracted: their existing frames
are added to the manifest as they are.

Usage:
    python calvin_process.py --calvin_dir <CALVIN>/task_ABC_D \
        --data_root <DATA_ROOT> --split training
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_frame(calvin_dir, frame_idx):
    """Load a single CALVIN frame's npz data."""
    path = os.path.join(calvin_dir, f"episode_{frame_idx:07d}.npz")
    if not os.path.exists(path):
        return None
    return np.load(path)


def main():
    parser = argparse.ArgumentParser(description="Process CALVIN into the training manifest format")
    parser.add_argument("--calvin_dir", required=True,
                        help="CALVIN dataset root (contains training/ and validation/)")
    parser.add_argument("--data_root", required=True,
                        help="Folder that holds all processed datasets")
    parser.add_argument("--subdir", default="calvin_processed",
                        help="CALVIN output folder inside data_root")
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--camera", default="rgb_static",
                        help="Camera to use: rgb_static (200x200) or rgb_gripper (84x84)")
    parser.add_argument("--num_target_frames", type=int, default=41,
                        help="Number of frames per task (subsample if episode is longer)")
    parser.add_argument("--max_tasks", type=int, default=0,
                        help="Max tasks to process (0=all)")
    parser.add_argument("--min_episode_len", type=int, default=20,
                        help="Minimum episode length to include")
    args = parser.parse_args()

    output_dir = os.path.join(args.data_root, args.subdir)

    def rel(p):
        return os.path.relpath(p, args.data_root)

    split_dir = os.path.join(args.calvin_dir, args.split)
    if not os.path.isdir(split_dir):
        print(f"ERROR: split directory not found: {split_dir}")
        return

    # Language annotations
    lang_path = os.path.join(split_dir, "lang_annotations", "auto_lang_ann.npy")
    if not os.path.exists(lang_path):
        print(f"ERROR: language annotations not found: {lang_path}")
        return

    ann = np.load(lang_path, allow_pickle=True).item()
    descriptions = ann["language"]["ann"]
    task_names = ann["language"]["task"]
    indices = ann["info"]["indx"]

    print(f"CALVIN {args.split}: {len(descriptions)} language-annotated episodes")

    frames_dir = os.path.join(output_dir, "frames")
    actions_dir = os.path.join(output_dir, "actions")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(actions_dir, exist_ok=True)

    tasks = []
    action_index = {}
    n_processed = 0

    for i in tqdm(range(len(descriptions)), desc="Processing"):
        if args.max_tasks > 0 and n_processed >= args.max_tasks:
            break

        start_idx, end_idx = indices[i]
        episode_len = end_idx - start_idx + 1

        if episode_len < args.min_episode_len:
            continue

        task_name = task_names[i]
        description = descriptions[i]
        task_id = f"calvin_{task_name}_{start_idx:07d}"

        task_frames_dir = os.path.join(frames_dir, task_id)
        task_actions_dir = os.path.join(actions_dir, task_id)

        # Already processed: reuse the frames on disk
        action_path = os.path.join(task_actions_dir, "actions.npy")
        if os.path.exists(action_path):
            frame_paths = sorted(
                os.path.join(task_frames_dir, f)
                for f in os.listdir(task_frames_dir) if f.endswith(".png")
            ) if os.path.isdir(task_frames_dir) else []
            if frame_paths:
                tasks.append(_build_task_entry(
                    task_id, task_name, description, [rel(p) for p in frame_paths],
                    list(range(len(frame_paths)))
                ))
                action_index[task_id] = rel(action_path)
                n_processed += 1
                continue

        os.makedirs(task_frames_dir, exist_ok=True)
        os.makedirs(task_actions_dir, exist_ok=True)

        # Subsample frame indices to num_target_frames
        all_frame_indices = list(range(start_idx, end_idx + 1))
        if len(all_frame_indices) > args.num_target_frames:
            step = len(all_frame_indices) / args.num_target_frames
            selected = [all_frame_indices[int(j * step)] for j in range(args.num_target_frames)]
        else:
            selected = all_frame_indices

        # Extract frames and actions
        frame_paths = []
        all_actions = []
        valid = True

        for j, frame_idx in enumerate(selected):
            data = load_frame(split_dir, frame_idx)
            if data is None:
                valid = False
                break

            img = data[args.camera]  # (200, 200, 3) uint8
            frame_path = os.path.join(task_frames_dir, f"frame_{j:04d}.png")
            if not os.path.exists(frame_path):
                Image.fromarray(img).save(frame_path)
            frame_paths.append(frame_path)

            all_actions.append(data["rel_actions"].astype(np.float32))

        if not valid or len(frame_paths) < args.min_episode_len:
            continue

        actions_array = np.stack(all_actions)  # (T, 7)
        np.save(action_path, actions_array)

        tasks.append(_build_task_entry(
            task_id, task_name, description, [rel(p) for p in frame_paths],
            list(range(len(frame_paths)))
        ))
        action_index[task_id] = rel(action_path)
        n_processed += 1

    manifest = {
        "stage": 1,
        "dataset": "calvin",
        "dataset_info": {
            "calvin_dir": os.path.basename(os.path.normpath(args.calvin_dir)),
            "split": args.split,
            "camera": args.camera,
            "n_tasks": len(tasks),
            "num_target_frames": args.num_target_frames,
        },
        "tasks": tasks,
    }

    manifest_path = os.path.join(output_dir, f"manifest_{args.split}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Action index: task_id -> actions.npy path relative to data_root
    index_path = os.path.join(actions_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(action_index, f, indent=2)

    print(f"\nProcessed {len(tasks)} tasks -> {manifest_path}")


def _build_task_entry(task_id, task_name, description, frame_paths, frame_indices):
    return {
        "task_id": task_id,
        "scene_id": task_name,
        "task_type": "manipulation",
        "description": description,
        "hdf5_file": "",
        "demo_key": "",
        "demo_idx": 0,
        "split": "calvin",
        "target_frames": {
            "paths": frame_paths,
            "frame_indices": frame_indices,
            "n_frames": len(frame_paths),
        },
        "context_frames": {
            "paths": [frame_paths[0]] if frame_paths else [],
            "n_frames": 1,
        },
    }


if __name__ == "__main__":
    main()

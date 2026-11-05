"""Prepare training data of the OnlyAct encoder: LIBERO frame extraction.

Reads the LIBERO HDF5 demonstration files (one file per task, 50 demos each)
from <libero_root>/<split>/*.hdf5 and writes, under <data_root>/<subdir>/:

    frames/<split>/<task_name>/demo_<k:03d>/frame_<t:04d>.png   agentview RGB
    demos_manifest.json                                       one entry per demo

Frame paths in demos_manifest.json are relative to <data_root>.
libero_build_manifest.py turns this file into the training manifest.

Source: the LIBERO-100 HDF5 files (libero_10, libero_90) of github.com/Lifelong-Robot-Learning/LIBERO,
also on Hugging Face as yifengzhu-hf/LIBERO-datasets.

    python data/training/libero_process.py --libero_root <libero> --splits libero_10 libero_90 --data_root <data>
"""

import argparse
import json
import os
import time

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def parse_instruction_from_metadata(hdf5_file) -> str:
    """Return the language instruction stored in the HDF5 metadata, or ""."""
    problem_info = hdf5_file["data"].attrs.get("problem_info", "")
    if problem_info:
        try:
            info = json.loads(problem_info)
            instruction = info.get("language_instruction", "")
            if instruction:
                return instruction
        except (json.JSONDecodeError, TypeError):
            pass
    return ""


def extract_frames(hdf5_path: str, demo_key: str, data_root: str, subdir: str,
                   task_name: str, demo_idx: int) -> list:
    """Write the agentview_rgb frames of one demo as PNGs.

    Returns the frame paths relative to data_root, in time order.
    """
    rel_dir = os.path.join(subdir, "frames", task_name, f"demo_{demo_idx:03d}")
    av_dir = os.path.join(data_root, rel_dir)
    os.makedirs(av_dir, exist_ok=True)

    rel_paths = []
    with h5py.File(hdf5_path, "r") as f:
        av_rgb = f["data"][demo_key]["obs"]["agentview_rgb"]  # (T, 128, 128, 3)
        n_frames = av_rgb.shape[0]
        for i in range(n_frames):
            av_fname = f"frame_{i:04d}.png"
            av_path = os.path.join(av_dir, av_fname)
            if not os.path.exists(av_path):
                # LIBERO renders upside down; flip vertically, then RGB -> BGR for OpenCV.
                img = np.flipud(av_rgb[i])
                cv2.imwrite(av_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            rel_paths.append(os.path.join(rel_dir, av_fname))
    return rel_paths


def main():
    parser = argparse.ArgumentParser(description="Extract LIBERO agentview frames and list all demos.")
    parser.add_argument("--libero_root", type=str, required=True,
                        help="Folder with the LIBERO HDF5 files (or with one sub-folder per split)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder that holds all processed datasets")
    parser.add_argument("--subdir", type=str, default="libero_processed",
                        help="Output folder inside data_root")
    parser.add_argument("--splits", nargs="*", default=None,
                        help="Split sub-folders to process (e.g. libero_10 libero_90). "
                             "If not given, HDF5 files are read directly from libero_root.")
    parser.add_argument("--max_demos_per_task", type=int, default=0,
                        help="Max demos per HDF5 file (0 = all)")
    args = parser.parse_args()

    out_dir = os.path.join(args.data_root, args.subdir)
    os.makedirs(out_dir, exist_ok=True)

    # Collect HDF5 files (sorted by file name within each split, splits in the given order).
    hdf5_files = []
    if args.splits:
        for split in args.splits:
            split_dir = os.path.join(args.libero_root, split)
            if not os.path.isdir(split_dir):
                print(f"Warning: split directory not found: {split_dir}")
                continue
            for fn in sorted(os.listdir(split_dir)):
                if fn.endswith(".hdf5"):
                    hdf5_files.append((split, os.path.join(split_dir, fn), fn))
    else:
        for fn in sorted(os.listdir(args.libero_root)):
            if fn.endswith(".hdf5"):
                hdf5_files.append(("", os.path.join(args.libero_root, fn), fn))

    if not hdf5_files:
        print(f"No HDF5 files found in {args.libero_root}")
        return

    print(f"Processing {len(hdf5_files)} HDF5 files")
    t0 = time.time()
    all_demos = []
    total_frames = 0

    for split, hdf5_path, filename in tqdm(hdf5_files, desc="LIBERO", unit="file"):
        task_name = filename.replace("_demo.hdf5", "")
        if split:
            task_name = f"{split}/{task_name}"

        with h5py.File(hdf5_path, "r") as f:
            instruction = parse_instruction_from_metadata(f)
            # Lexicographic order of the demo keys (demo_0, demo_1, demo_10, ...);
            # demo_idx below is the position in this order, not the number in the key.
            demo_keys = sorted(f["data"].keys())
            n_demos_total = len(demo_keys)
            if not instruction:
                # Fall back to the task name taken from the file name.
                instruction = task_name.split("/")[-1].replace("_", " ")

        if args.max_demos_per_task > 0:
            demo_keys = demo_keys[:args.max_demos_per_task]

        for demo_idx, demo_key in enumerate(demo_keys):
            rel_paths = extract_frames(hdf5_path, demo_key, args.data_root, args.subdir,
                                       task_name, demo_idx)
            total_frames += len(rel_paths)
            all_demos.append({
                "task_id": f"{task_name.replace('/', '_')}_demo{demo_idx:03d}",
                "scene_id": task_name,
                "task_type": "manipulation",
                "description": instruction,
                "split": split,
                "hdf5_file": filename,
                "demo_key": demo_key,
                "demo_idx": demo_idx,
                "n_demos_total": n_demos_total,
                "all_frames": {
                    "paths": rel_paths,
                    "n_frames": len(rel_paths),
                },
            })

    manifest = {
        "stage": "libero_demos",
        "dataset": "libero",
        "dataset_info": {
            "splits": args.splits or [],
            "n_hdf5_files": len(hdf5_files),
            "n_demos": len(all_demos),
            "n_frames": total_frames,
        },
        "tasks": all_demos,
    }
    output_path = os.path.join(out_dir, "demos_manifest.json")
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done: {len(hdf5_files)} files, {len(all_demos)} demos, {total_frames} frames "
          f"in {time.time() - t0:.1f}s -> {output_path}")


if __name__ == "__main__":
    main()

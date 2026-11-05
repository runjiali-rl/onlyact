"""Prepare training data of the OnlyAct encoder: union manifest over the datasets.

Combines the per-dataset outputs into one manifest of training clips:
  - frame-backed manifests (calvin, atari, libero, droid) are passed through,
  - the VPT segment index (vpt) becomes one task per segment,
  - video folders (agibot, egodex, spatialvid: *.mp4; sthv2: *.webm) become one
    task per video, with the frame count read from the container header by ffprobe.

Every input is given relative to --data_root (defaults follow the folder layout
written by the per-dataset scripts). The output uses paths relative to --data_root:
    {"tasks": [{"task_id", "dataset", "_union_dataset", "split", "task_type",
                "description", "scene_id" (frame-backed datasets) / "ep_dir" (vpt),
                "target_frames": {"paths": [...], "frame_indices": [...], "n_frames": N}
                              or {"mp4_path": ..., "n_frames": N},
                "context_frames": {... one frame ...}}, ...]}

Usage (settings of the training data):
    # eight datasets
    python build_union_manifest.py --data_root <DATA_ROOT> --mp4_min_frames 16 \
        --datasets calvin atari libero droid vpt agibot egodex spatialvid \
        --out <DATA_ROOT>/union_8ds/manifest_union.json
    # Something-Something v2 alone
    python build_union_manifest.py --data_root <DATA_ROOT> --mp4_min_frames 16 \
        --datasets sthv2 --out <DATA_ROOT>/union_9ds/manifest_sthv2_only.json
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Input of each dataset, relative to --data_root
DEFAULTS = {
    "calvin":     "calvin_processed/manifest_training.json",
    "atari":      "atari_processed/manifest_atari.json",
    "libero":     "libero_processed/manifest_training.json",
    "droid":      "droid_processed/manifest_training.json",
    "vpt":        "vpt_minecraft_stride1/index.json",
    "agibot":     "agibot_processed",
    "egodex":     "egodex_processed",
    "spatialvid": "spatialvid_processed",
    "sthv2":      "sthv2/videos/20bn-something-something-v2",
}


def _ensure_dataset_field(task: dict, name: str) -> dict:
    task.setdefault("dataset", name)
    # The trainer reads the dataset tag from _union_dataset.
    task["_union_dataset"] = name
    task.setdefault("split", name)
    if "task_type" not in task:
        task["task_type"] = "manipulation" if name in ("calvin", "libero", "droid", "agibot", "egodex") else name
    return task


def build_frame_dataset(name: str, manifest_path: str) -> list:
    """calvin / atari / libero / droid: tasks already in trainer schema, just tag them."""
    with open(manifest_path) as f:
        m = json.load(f)
    tasks = m["tasks"]
    out = []
    for t in tasks:
        tf = t.get("target_frames")
        if not (isinstance(tf, dict) and "paths" in tf and "n_frames" in tf):
            continue
        if len(tf["paths"]) != tf["n_frames"]:
            # Trust the frame list over the stored count.
            tf["n_frames"] = len(tf["paths"])
        out.append(_ensure_dataset_field(t, name))
    print(f"  [{name:10s}] kept {len(out):>7d}/{len(tasks):>7d} tasks")
    return out


def build_vpt(index_path: str, data_root: str) -> list:
    """One task per VPT segment, listing all its frames."""
    with open(index_path) as f:
        m = json.load(f)
    out = []
    skipped = 0
    for ep in m["episodes"]:
        ep_dir = ep["ep_dir"]  # relative to data_root
        frames_dir = os.path.join(data_root, ep_dir, "frames")
        if not os.path.isdir(frames_dir):
            skipped += 1
            continue
        try:
            files = sorted(
                f for f in os.listdir(frames_dir)
                if f.startswith("frame_") and f.endswith(".jpg")
            )
        except OSError:
            skipped += 1
            continue
        if not files:
            skipped += 1
            continue
        paths = [os.path.join(ep_dir, "frames", f) for f in files]
        n = len(paths)
        eid = ep["episode_id"]
        out.append({
            "task_id": f"vpt_{eid}".replace("/", "_"),
            "dataset": "vpt",
            "_union_dataset": "vpt",
            "split": "vpt",
            "task_type": "minecraft",
            "description": "minecraft gameplay segment",
            "ep_dir": ep_dir,
            "episode_id": eid,
            "target_frames": {
                "paths": paths,
                "frame_indices": list(range(n)),
                "n_frames": n,
            },
            "context_frames": {
                "paths": [paths[0]],
                "frame_indices": [0],
                "n_frames": 1,
            },
        })
    print(f"  [vpt       ] kept {len(out):>7d}/{len(m['episodes']):>7d} episodes "
          f"(skipped {skipped})")
    return out


def _ffprobe_n_frames(video_path: str) -> int:
    """Frame count = header duration x frame rate (no decoding).
    Uses the stream duration, or the container duration when the stream has none (webm)."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=duration,r_frame_rate:format=duration",
                "-of", "default=noprint_wrappers=1:nokey=0",
                video_path,
            ],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode()
        dur_stream = None
        dur_format = None
        fps = 16.0
        for line in out.splitlines():
            k, _, v = line.partition("=")
            v = v.strip()
            if k == "duration":
                # First occurrence is the stream duration, second the container duration.
                if v == "N/A" or v == "":
                    continue
                try:
                    f = float(v)
                except ValueError:
                    continue
                if dur_stream is None:
                    dur_stream = f
                else:
                    dur_format = f
            elif k == "r_frame_rate" and "/" in v:
                num, den = v.split("/")
                if float(den) > 0:
                    fps = float(num) / float(den)
        dur = dur_stream if dur_stream is not None else dur_format
        if dur is None:
            return 0
        return int(round(dur * fps))
    except Exception:
        return 0


def _probe_one(args):
    idx, path = args
    return idx, path, _ffprobe_n_frames(path)


def build_video_dataset(name: str, root: str, data_root: str, workers: int = 32,
                        min_frames: int = 8, glob_pattern: str = "**/*.mp4",
                        chunksize: int = 200) -> list:
    """Glob the videos under root, ffprobe each in parallel, one task per video."""
    root_p = Path(root)
    print(f"  [{name:10s}] scanning ...", flush=True)
    # Glob order is file-system dependent (it fixes the task order of this dataset).
    paths = [str(p) for p in root_p.glob(glob_pattern)]
    print(f"  [{name:10s}] found {len(paths)} videos, probing with {workers} workers", flush=True)
    nframes = [0] * len(paths)
    n_done = 0
    t0 = time.time()
    # Spawned workers do not inherit the parent's memory.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        for i, _p, nf in ex.map(_probe_one, enumerate(paths), chunksize=chunksize):
            nframes[i] = nf
            n_done += 1
            if n_done % 20000 == 0:
                rate = n_done / max(time.time() - t0, 0.001)
                print(f"    [{name}] probed {n_done}/{len(paths)} ({rate:.0f}/s)", flush=True)
    out = []
    dropped = 0
    for path, nf in zip(paths, nframes):
        if nf < min_frames:
            dropped += 1
            continue
        # task_id from the path relative to the dataset folder, without extension
        rel = os.path.relpath(path, root)
        rel_noext = os.path.splitext(rel)[0]
        task_id = f"{name}_" + rel_noext.replace("/", "_")
        rel_to_data_root = os.path.relpath(path, data_root)
        out.append({
            "task_id": task_id,
            "dataset": name,
            "_union_dataset": name,
            "split": name,
            "task_type": "video",
            "description": f"{name} video clip",
            "target_frames": {
                "mp4_path": rel_to_data_root,
                "n_frames": nf,
            },
            "context_frames": {
                "mp4_path": rel_to_data_root,
                "frame_indices": [0],
                "n_frames": 1,
            },
        })
    print(f"  [{name:10s}] kept {len(out):>7d}/{len(paths):>7d} videos "
          f"(dropped {dropped} for n_frames<{min_frames})")
    return out


def main():
    ap = argparse.ArgumentParser(description="Build the union training manifest")
    ap.add_argument("--data_root", required=True,
                    help="Folder that holds all processed datasets")
    ap.add_argument("--out", required=True, help="Output manifest path")
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULTS.keys()),
                    help="Datasets to include, in this order")
    ap.add_argument("--mp4_workers", type=int, default=32,
                    help="Parallel ffprobe workers for video datasets")
    ap.add_argument("--mp4_min_frames", type=int, default=8,
                    help="Drop videos with fewer than this many frames")
    for k in DEFAULTS:
        ap.add_argument(f"--{k}", default=DEFAULTS[k],
                        help=f"{k} input relative to data_root (default: {DEFAULTS[k]})")
    args = ap.parse_args()

    all_tasks = []
    counts = {}
    for name in args.datasets:
        path = os.path.join(args.data_root, getattr(args, name))
        print(f"\n>>> building {name} from {getattr(args, name)}")
        if name in ("calvin", "atari", "libero", "droid"):
            tasks = build_frame_dataset(name, path)
        elif name == "vpt":
            tasks = build_vpt(path, args.data_root)
        elif name in ("agibot", "egodex", "spatialvid"):
            tasks = build_video_dataset(name, path, args.data_root,
                                        workers=args.mp4_workers,
                                        min_frames=args.mp4_min_frames)
        elif name == "sthv2":
            tasks = build_video_dataset(name, path, args.data_root,
                                        workers=args.mp4_workers,
                                        min_frames=args.mp4_min_frames,
                                        glob_pattern="**/*.webm")
        else:
            print(f"  unknown dataset {name}, skipping")
            continue
        counts[name] = len(tasks)
        all_tasks.extend(tasks)

    print("\n=== summary ===")
    for n, c in counts.items():
        print(f"  {n:12s} {c:>9d}")
    print(f"  {'TOTAL':12s} {len(all_tasks):>9d}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "union",
        "datasets": list(counts.keys()),
        "dataset_counts": counts,
        "tasks": all_tasks,
    }
    print(f"\nwriting {out_path} ({len(all_tasks)} tasks) ...")
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"  done. file size: {out_path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

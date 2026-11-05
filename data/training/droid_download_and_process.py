#!/usr/bin/env python3
"""Prepare training data of the OnlyAct encoder: DROID download and frame extraction.

Downloads raw DROID episodes from the public bucket gs://gresearch/robotics/droid_raw/1.0.1
(needs `gsutil`), keeps trajectory.h5 and the episode metadata, and decodes every mono MP4
to 256x144 JPEG frames with ffmpeg. With --task_ids only the listed episodes are fetched
(droid_task_ids.txt holds the 606 episodes used for training).

Output layout under <data_root>/<subdir> (default subdir: droid_processed):
    <lab>/success/<date>/<timestamp>/
        trajectory.h5  metadata.json  _done
        hand_camera/frame_%05d.jpg  varied_camera_1/frame_%05d.jpg  varied_camera_2/frame_%05d.jpg

droid_build_manifest.py turns these folders into the training manifest.

    python data/training/droid_download_and_process.py --data_root <data> --task_ids data/training/droid_task_ids.txt
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

GCS_BUCKET = "gs://gresearch/robotics/droid_raw/1.0.1"
ALL_LABS = [
    "AUTOLab", "CLVR", "GuptaLab", "ILIAD", "IPRL", "IRIS",
    "PennPAL", "RAD", "RAIL", "REAL", "RPL", "TRI", "WEIRD",
]

# Exterior cameras are 1280x720 and the wrist camera 672x376; all are resized to 256x144.
TARGET_W = 256
TARGET_H = 144


def droid_task_id(lab: str, date: str, timestamp: str) -> str:
    """Task id used in the training manifest (same rule as droid_build_manifest.py)."""
    task_id = f"droid_{lab}_{date}_{timestamp}"
    return task_id.replace(":", "-").replace(" ", "_").replace("+", "_")


def list_all_episodes_from_manifest() -> list[dict]:
    """List all episodes from the bucket's manifest.json."""
    log.info("Downloading manifest.json...")
    result = subprocess.run(
        ["gsutil", "cat", f"{GCS_BUCKET}/manifest.json"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        log.error(f"Failed to download manifest: {result.stderr}")
        return []

    manifest = json.loads(result.stdout)
    episodes = []
    for entry in manifest:
        # path: .../r2d2_full_raw/{LAB}/{outcome}/{date}/{timestamp}/trajectory_im128.h5
        raw_path = entry["path"]
        parts = raw_path.split("/")
        try:
            idx = parts.index("r2d2_full_raw")
            lab = parts[idx + 1]
            outcome = parts[idx + 2]
            date = parts[idx + 3]
            timestamp = parts[idx + 4]
        except (ValueError, IndexError):
            log.warning(f"Cannot parse path: {raw_path}")
            continue

        gcs_dir = f"{GCS_BUCKET}/{lab}/{outcome}/{date}/{timestamp}"
        episodes.append({
            "lab": lab,
            "outcome": outcome,
            "date": date,
            "timestamp": timestamp,
            "gcs_dir": gcs_dir,
        })

    log.info(f"Manifest has {len(episodes)} episodes")
    return episodes


def get_episode_id(ep: dict) -> str:
    """Relative folder of an episode: {lab}/{outcome}/{date}/{timestamp}."""
    return f"{ep['lab']}/{ep['outcome']}/{ep['date']}/{ep['timestamp']}"


def is_episode_processed(output_dir: Path, ep: dict) -> bool:
    """An episode is done when its _done marker exists."""
    return (output_dir / get_episode_id(ep) / "_done").exists()


def _gsutil_cp(src: str, dst: str, timeout: int = 300) -> bool:
    """Download a single file from the bucket. Returns True on success."""
    r = subprocess.run(
        ["gsutil", "-q", "cp", src, dst],
        capture_output=True, text=True, timeout=timeout,
    )
    return r.returncode == 0


def _read_camera_serials(traj_path: Path) -> list[str]:
    """Camera serial numbers from trajectory.h5 (observation/camera_type group)."""
    import h5py
    try:
        with h5py.File(str(traj_path), "r") as f:
            if "observation/camera_type" in f:
                return list(f["observation/camera_type"].keys())
            # Fallback: serials from camera_extrinsics keys such as "13062452_left".
            if "observation/camera_extrinsics" in f:
                serials = set()
                for key in f["observation/camera_extrinsics"].keys():
                    serial = key.split("_")[0]
                    if serial.isdigit():
                        serials.add(serial)
                return list(serials)
    except Exception as e:
        log.debug(f"  Cannot read camera serials from {traj_path}: {e}")
    return []


def download_episode_files_batch(episodes: list[dict], tmp_dir: Path) -> None:
    """Download trajectory.h5, the mono MP4 of every camera and the metadata JSON."""
    log.info(f"Downloading {len(episodes)} episodes to {tmp_dir}...")

    for i, ep in enumerate(episodes):
        gcs_dir = ep["gcs_dir"]
        ep_tmp = tmp_dir / get_episode_id(ep)
        ep_tmp.mkdir(parents=True, exist_ok=True)

        traj_dst = str(ep_tmp / "trajectory.h5")
        if not _gsutil_cp(f"{gcs_dir}/trajectory.h5", traj_dst):
            log.warning(f"  No trajectory.h5 for {get_episode_id(ep)}, skipping")
            continue

        cam_serials = _read_camera_serials(ep_tmp / "trajectory.h5")

        for serial in cam_serials:
            mp4_src = f"{gcs_dir}/recordings/MP4/{serial}.mp4"
            mp4_dst = str(ep_tmp / f"{serial}.mp4")
            if not _gsutil_cp(mp4_src, mp4_dst):
                log.debug(f"  MP4 not found: {serial}.mp4")

        # The metadata file name contains a user id, so it is fetched by wildcard.
        meta_src = f"{gcs_dir}/metadata_*.json"
        meta_dst = str(ep_tmp / "metadata.json")
        r = subprocess.run(
            ["gsutil", "-q", "cp", meta_src, meta_dst],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            log.debug(f"  Metadata not found for {get_episode_id(ep)}")

        if (i + 1) % 50 == 0:
            log.info(f"  Downloaded {i + 1}/{len(episodes)} episodes...")


def extract_frames_from_mp4(mp4_path: Path, output_dir: Path, camera_name: str) -> int:
    """Decode all frames of an MP4 to 256x144 JPEG files frame_00000.jpg, frame_00001.jpg, ..."""
    frames_dir = output_dir / camera_name
    frames_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-i", str(mp4_path),
        "-vf", f"scale={TARGET_W}:{TARGET_H}",
        "-q:v", "5",  # JPEG quality (2 = best, 31 = worst)
        "-start_number", "0",
        str(frames_dir / "frame_%05d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log.warning(f"ffmpeg failed for {mp4_path.name}: {result.stderr[-200:]}")
        return 0

    return len(list(frames_dir.glob("frame_*.jpg")))


def process_episode(ep: dict, tmp_dir: Path, output_dir: Path) -> dict:
    """Copy trajectory.h5 and metadata, extract frames of every camera, write _done."""
    ep_id = get_episode_id(ep)
    ep_tmp = tmp_dir / ep_id
    ep_out = output_dir / ep_id
    ep_out.mkdir(parents=True, exist_ok=True)

    result = {"episode_id": ep_id, "success": False, "n_frames": {}, "error": None}

    try:
        traj_src = ep_tmp / "trajectory.h5"
        if traj_src.exists():
            shutil.copy2(str(traj_src), str(ep_out / "trajectory.h5"))
        else:
            result["error"] = "trajectory.h5 not found"
            return result

        meta_src = ep_tmp / "metadata.json"
        if meta_src.exists():
            shutil.copy2(str(meta_src), str(ep_out / "metadata.json"))

        # Camera folder names from the serials in the metadata.
        meta_path = ep_out / "metadata.json"
        cam_map = {}  # serial -> camera folder
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if "wrist_cam_serial" in meta:
                cam_map[meta["wrist_cam_serial"]] = "hand_camera"
            if "ext1_cam_serial" in meta:
                cam_map[meta["ext1_cam_serial"]] = "varied_camera_1"
            if "ext2_cam_serial" in meta:
                cam_map[meta["ext2_cam_serial"]] = "varied_camera_2"

        for mp4_path in sorted(ep_tmp.glob("*.mp4")):
            serial = mp4_path.stem
            camera_name = cam_map.get(serial, f"camera_{serial}")
            n = extract_frames_from_mp4(mp4_path, ep_out, camera_name)
            result["n_frames"][camera_name] = n

        (ep_out / "_done").write_text(json.dumps({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_frames": result["n_frames"],
            "cameras": list(result["n_frames"].keys()),
        }))
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        log.error(f"Error processing {ep_id}: {e}")

    return result


def process_batch(episodes: list[dict], tmp_dir: Path, output_dir: Path,
                  ffmpeg_workers: int = 4) -> list[dict]:
    """Download a batch of episodes, then extract their frames in parallel."""
    download_episode_files_batch(episodes, tmp_dir)

    results = []
    with ThreadPoolExecutor(max_workers=ffmpeg_workers) as pool:
        futures = {
            pool.submit(process_episode, ep, tmp_dir, output_dir): ep
            for ep in episodes
        }
        for future in as_completed(futures):
            ep = futures[future]
            try:
                result = future.result()
                results.append(result)
                if not result["success"]:
                    log.warning(f"  FAIL: {result['episode_id']}: {result['error']}")
            except Exception as e:
                log.error(f"  Exception for {get_episode_id(ep)}: {e}")
                results.append({
                    "episode_id": get_episode_id(ep),
                    "success": False, "error": str(e),
                })

    return results


def main():
    parser = argparse.ArgumentParser(description="Download DROID episodes and extract frames.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder that holds all processed datasets")
    parser.add_argument("--subdir", type=str, default="droid_processed",
                        help="DROID folder inside data_root")
    parser.add_argument("--tmp_dir", type=str, default=None,
                        help="Temporary download folder (default <data_root>/<subdir>/.tmp)")
    parser.add_argument("--labs", nargs="+", default=None,
                        help=f"Labs to process (default: all). Choices: {ALL_LABS}")
    parser.add_argument("--task_ids", type=str, default=None,
                        help="Task id list (first column per line); only these episodes are processed")
    parser.add_argument("--batch_size", type=int, default=200,
                        help="Episodes per download batch")
    parser.add_argument("--ffmpeg_workers", type=int, default=4,
                        help="Parallel ffmpeg workers")
    parser.add_argument("--resume", action="store_true",
                        help="Skip episodes that already have a _done marker")
    parser.add_argument("--dry_run", action="store_true",
                        help="List episodes without downloading")
    parser.add_argument("--include_failures", action="store_true",
                        help="Also process episodes recorded as failures")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Max episodes to process (for testing)")
    args = parser.parse_args()

    output_dir = Path(args.data_root) / args.subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else output_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    labs = args.labs or ALL_LABS
    for lab in labs:
        if lab not in ALL_LABS:
            log.error(f"Unknown lab: {lab}. Choices: {ALL_LABS}")
            sys.exit(1)

    all_episodes = list_all_episodes_from_manifest()
    if not all_episodes:
        log.error("No episodes found. Check bucket access.")
        sys.exit(1)

    episodes = [ep for ep in all_episodes if ep["lab"] in labs]
    log.info(f"Selected {len(episodes)} episodes from labs: {labs}")

    if not args.include_failures:
        episodes = [ep for ep in episodes if ep["outcome"] == "success"]
        log.info(f"After filtering success only: {len(episodes)} episodes")

    if args.task_ids:
        with open(args.task_ids) as f:
            keep = {line.split()[0] for line in f if line.strip()}
        episodes = [ep for ep in episodes
                    if droid_task_id(ep["lab"], ep["date"], ep["timestamp"]) in keep]
        log.info(f"Kept {len(episodes)} episodes listed in {args.task_ids} ({len(keep)} ids)")

    if args.resume:
        before = len(episodes)
        episodes = [ep for ep in episodes if not is_episode_processed(output_dir, ep)]
        log.info(f"Resume: {before - len(episodes)} already processed, {len(episodes)} remaining")

    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
        log.info(f"Limited to {len(episodes)} episodes (--max_episodes)")

    if args.dry_run:
        by_lab = {}
        for ep in episodes:
            by_lab.setdefault(ep["lab"], []).append(ep)
        for lab in sorted(by_lab):
            log.info(f"  {lab}: {len(by_lab[lab])} episodes")
        log.info(f"Total: {len(episodes)} episodes")
        return

    total = len(episodes)
    n_success = 0
    n_fail = 0
    t_start = time.time()

    for batch_start in range(0, total, args.batch_size):
        batch = episodes[batch_start:batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1
        n_batches = (total + args.batch_size - 1) // args.batch_size
        log.info(f"=== Batch {batch_num}/{n_batches}: {len(batch)} episodes ===")

        results = process_batch(batch, tmp_dir, output_dir, args.ffmpeg_workers)
        n_success += sum(1 for r in results if r["success"])
        n_fail += sum(1 for r in results if not r["success"])

        # Remove the downloaded MP4s of this batch.
        for ep in batch:
            ep_tmp = tmp_dir / get_episode_id(ep)
            if ep_tmp.exists():
                shutil.rmtree(ep_tmp)

        processed = batch_start + len(batch)
        elapsed = time.time() - t_start
        log.info(f"Progress: {processed}/{total} | OK: {n_success} | Fail: {n_fail} | "
                 f"{elapsed / 60:.1f} min")

    log.info(f"Done: {n_success} succeeded, {n_fail} failed")


if __name__ == "__main__":
    main()

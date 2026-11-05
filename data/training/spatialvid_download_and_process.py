#!/usr/bin/env python3
"""Prepare training data of the OnlyAct encoder (SpatialVID part).

Downloads the SpatialVID video shards videos/group_NNNN.tar.gz from Hugging
Face (https://huggingface.co/datasets/SpatialVID/SpatialVID) and re-encodes
every clip to 256x256 H.264 at 16 frames per second. The annotation shards are
not downloaded.

Each shard holds SpatialVID/videos/{group_NNNN}/{uuid}.mp4. Shards are processed
one at a time:
  1. download one tar.gz into --work_dir;
  2. stream-extract each .mp4 member;
  3. transcode it with libx264 (crf 23, preset fast): scale the shorter side
     to 256, then center-crop to 256x256;
  4. delete the extracted file and, at the end, the shard.

Output layout (the union manifest builder globs these files):
  <data_root>/<subdir>/{group_NNNN}/{uuid}.mp4

SpatialVID is gated: accept the licence on the dataset page and log in with
`huggingface-cli login` (or set HF_TOKEN) before running.

Usage (the training data uses the first 272 of the 545 shards,
group_0001 ... group_0272):
  python spatialvid_download_and_process.py --data_root <DATA_ROOT> \\
      --work_dir <WORK_DIR> --start 0 --end 272
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    sys.exit("Install huggingface_hub: pip install 'huggingface_hub>=0.20'")

REPO_ID = "SpatialVID/SpatialVID"
REPO_TYPE = "dataset"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("spatialvid")


def list_video_shards(api: HfApi) -> list[str]:
    """Enumerate all videos/group_NNNN.tar.gz shards in the repo (sorted)."""
    paths = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    return sorted(p for p in paths
                  if p.startswith("videos/group_") and p.endswith(".tar.gz"))


def parse_member_path(name: str) -> tuple[str, str] | None:
    """Map a tar member SpatialVID/videos/{group_NNNN}/{uuid}.mp4 to
    (group_id, clip_file) or None for any other member."""
    if not name.endswith(".mp4"):
        return None
    parts = name.split("/")
    if len(parts) != 4:
        return None
    if parts[0] != "SpatialVID" or parts[1] != "videos":
        return None
    group_id = parts[2]
    clip_file = parts[3]
    if not group_id.startswith("group_"):
        return None
    return group_id, clip_file


def ffmpeg_transcode(src: Path, dst: Path, target_fps: int, target_size: int,
                     crf: int = 23) -> bool:
    """Resize and re-encode one mp4. Returns True on success."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Temporary name keeps the .mp4 suffix so ffmpeg can infer the container.
    tmp = dst.parent / f".part_{dst.name}"
    # Scale the shorter side to target_size (even width), then center-crop
    # to a target_size x target_size square.
    vf = (
        f"scale=-2:{target_size}:flags=bicubic,"
        f"crop={target_size}:{target_size}"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
        "-i", str(src),
        "-vf", vf,
        "-r", str(target_fps),
        "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an", "-f", "mp4",
        str(tmp),
    ]
    try:
        # Clips whose transcode takes longer than 120 s are skipped.
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
    except subprocess.CalledProcessError as e:
        log.error("ffmpeg failed on %s\n%s", src.name, (e.stderr or "")[-500:])
        if tmp.exists():
            tmp.unlink()
        return False
    except subprocess.TimeoutExpired:
        log.error("ffmpeg timeout on %s", src.name)
        if tmp.exists():
            tmp.unlink()
        return False
    tmp.rename(dst)
    return True


def process_shard(
    tar_path: Path,
    output_root: Path,
    scratch_root: Path,
    target_fps: int,
    target_size: int,
    workers: int,
    crf: int = 23,
) -> tuple[int, int, int, int]:
    """Stream-extract mp4s from a tar.gz shard and transcode each.
    Returns (ok, fail, skipped, seen)."""
    extract_dir = scratch_root / (tar_path.stem.replace(".tar", "") + "_extract")
    if extract_dir.exists():
        shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = n_skip = n_seen = 0
    # At most workers * 2 extracted clips wait for transcoding at any time.
    pending_slots = threading.Semaphore(workers * 2)
    submitted: list = []

    try:
        with tarfile.open(tar_path, "r|gz") as tar, ThreadPoolExecutor(max_workers=workers) as pool:
            for member in tar:
                if not member.isfile():
                    continue
                parsed = parse_member_path(member.name)
                if parsed is None:
                    continue
                group_id, clip_file = parsed
                n_seen += 1
                dst = output_root / group_id / clip_file
                # Existing non-empty outputs are kept (resume).
                if dst.exists() and dst.stat().st_size > 0:
                    n_skip += 1
                    continue

                pending_slots.acquire()
                tar.extract(member, extract_dir, set_attrs=False)
                src = extract_dir / member.name

                def _job(src=src, dst=dst):
                    try:
                        ok = ffmpeg_transcode(src, dst, target_fps, target_size, crf=crf)
                    finally:
                        try:
                            src.unlink()
                        except OSError:
                            pass
                        pending_slots.release()
                    return ok

                submitted.append(pool.submit(_job))

            for fut in as_completed(submitted):
                if fut.result():
                    n_ok += 1
                else:
                    n_fail += 1
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)

    if n_seen == 0:
        log.warning("  no matching .mp4 members in %s", tar_path.name)

    return n_ok, n_fail, n_skip, n_seen


def delete_downloaded_file(path: Path) -> None:
    """Delete a downloaded shard (and its target if it is a symlink)."""
    try:
        if path.is_symlink():
            target = path.resolve()
            path.unlink()
            if target.exists():
                target.unlink()
        elif path.exists():
            path.unlink()
    except OSError as e:
        log.warning("could not delete %s: %s", path.name, e)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=Path, required=True,
                   help="Folder that holds all processed training datasets.")
    p.add_argument("--subdir", default="spatialvid_processed",
                   help="Output folder under --data_root.")
    p.add_argument("--work_dir", type=Path, required=True,
                   help="Scratch folder for shard downloads and extraction.")
    p.add_argument("--target_fps", type=int, default=16)
    p.add_argument("--target_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel ffmpeg workers per shard.")
    p.add_argument("--crf", type=int, default=23,
                   help="libx264 CRF (the training data uses 23).")
    p.add_argument("--shards", nargs="*", default=None,
                   help="Process only these shard names (e.g. group_0001). Default: all.")
    p.add_argument("--start", type=int, default=0,
                   help="Start index in the sorted shard list (inclusive).")
    p.add_argument("--end", type=int, default=None,
                   help="End index in the sorted shard list (exclusive). Default: all.")
    p.add_argument("--keep_tar", action="store_true",
                   help="Keep each shard after processing.")
    p.add_argument("--dry_run", action="store_true",
                   help="List the shards that would be processed and exit.")
    args = p.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH.")

    output_root = args.data_root / args.subdir
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    log.info("Listing video shards in %s ...", REPO_ID)
    try:
        all_shards = list_video_shards(api)
    except HfHubHTTPError as e:
        log.error("Could not list repo files (gated?): %s", e)
        log.error("Visit https://huggingface.co/datasets/%s and accept the licence.", REPO_ID)
        return 2
    log.info("Found %d total video shards", len(all_shards))

    if args.shards:
        wanted = set(args.shards)
        shards = [s for s in all_shards if Path(s).name.replace(".tar.gz", "") in wanted]
        log.info("Filtered to %d shard(s) via --shards", len(shards))
    else:
        shards = all_shards[args.start:args.end]
        log.info("Processing %d shards (range %d:%s)",
                 len(shards), args.start, args.end if args.end is not None else "end")

    if args.dry_run:
        for s in shards:
            print(s)
        return 0

    total_ok = total_fail = total_skip = 0
    run_start = time.time()

    for idx, repo_path in enumerate(shards, 1):
        log.info("[%d/%d] %s", idx, len(shards), repo_path)

        t0 = time.time()
        try:
            local_path = Path(hf_hub_download(
                REPO_ID,
                filename=repo_path,
                repo_type=REPO_TYPE,
                local_dir=str(work_dir),
            ))
        except Exception as e:
            log.error("  download failed: %s", e)
            total_fail += 1
            continue
        size_gb = local_path.stat().st_size / 1e9
        log.info("  downloaded %.1f GB in %.0fs", size_gb, time.time() - t0)

        t1 = time.time()
        try:
            n_ok, n_fail, n_skip, n_seen = process_shard(
                local_path, output_root, work_dir,
                args.target_fps, args.target_size, args.workers,
                crf=args.crf,
            )
        except Exception as e:
            log.exception("  processing failed: %s", e)
            n_ok, n_fail, n_skip, n_seen = 0, 1, 0, 0
        log.info("  processed in %.0fs: seen=%d ok=%d fail=%d skipped=%d",
                 time.time() - t1, n_seen, n_ok, n_fail, n_skip)
        total_ok += n_ok
        total_fail += n_fail
        total_skip += n_skip

        if not args.keep_tar:
            delete_downloaded_file(local_path)

        # Stop if the first shard yields no clips (unexpected archive layout).
        if idx == 1 and total_ok == 0 and total_skip == 0:
            log.error("First shard produced 0 outputs; check the archive layout.")
            return 2

    log.info("Done in %.0fs. videos ok=%d fail=%d skipped=%d",
             time.time() - run_start, total_ok, total_fail, total_skip)
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

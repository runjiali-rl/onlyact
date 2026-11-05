#!/usr/bin/env python3
"""Prepare training data of the OnlyAct encoder (AgiBot World part).

Downloads the AgiBotWorld-Alpha observation shards from Hugging Face
(https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha) and
re-encodes the RGB camera streams to 256x256 H.264 at 16 frames per second.

Each shard observations/{task_id}/{episode_range}.tar holds many episodes with
RGB streams under {episode_id}/videos/{camera}.mp4. Shards are processed one at
a time:
  1. download one tar into --work_dir;
  2. stream its members and extract only the selected RGB .mp4 files;
  3. transcode each video with libx264 (crf 23, preset fast): scale the
     shorter side to 256, then center-crop to 256x256;
  4. delete the tar and the extracted files.

Output layout (the union manifest builder globs these files):
  <data_root>/<subdir>/{task_id}/{episode_id}/{camera}.mp4

AgiBotWorld-Alpha is gated: accept the licence on the dataset page and log in
with `huggingface-cli login` (or set HF_TOKEN) before running.

Usage (all shards, the three default cameras):
  python agibot_download_and_process.py --data_root <DATA_ROOT> --work_dir <WORK_DIR>
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

REPO_ID = "agibot-world/AgiBotWorld-Alpha"
REPO_TYPE = "dataset"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("agibot")


def list_tar_files(api: HfApi) -> list[str]:
    """Enumerate all observations/{task}/*.tar shards in the dataset repo."""
    paths = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    return sorted(p for p in paths if p.startswith("observations/") and p.endswith(".tar"))


def ffmpeg_transcode(src: Path, dst: Path, target_fps: int, target_size: int) -> bool:
    """Resize and re-encode one mp4. Returns True on success."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Temporary name keeps the .mp4 suffix so ffmpeg can infer the container.
    tmp = dst.parent / f".part_{dst.name}"
    # Scale the shorter side to target_size, then center-crop to a square
    # (640x480 -> 342x256 -> 256x256).
    vf = (
        f"scale=-2:{target_size}:flags=bicubic,"
        f"crop={target_size}:{target_size}"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
        "-i", str(src),
        "-vf", vf,
        "-r", str(target_fps),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        "-f", "mp4",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        log.error("ffmpeg failed on %s\n%s", src.name, (e.stderr or "")[-800:])
        if tmp.exists():
            tmp.unlink()
        return False
    tmp.rename(dst)
    return True


DEFAULT_CAMS = ("head_color.mp4", "hand_left_color.mp4", "hand_right_color.mp4")


def parse_member_path(name: str, allowed_cams: frozenset[str]) -> tuple[str, str] | None:
    """Map a tar member name {episode_id}/videos/{camera}.mp4 to
    (episode_id, camera) or None for any other member."""
    if name.startswith("./"):
        name = name[2:]
    parts = name.split("/")
    if len(parts) != 3 or parts[1] != "videos":
        return None
    cam = parts[2]
    if cam not in allowed_cams:
        return None
    return parts[0], cam


def process_tar(
    tar_path: Path,
    task_id: str,
    output_root: Path,
    scratch_root: Path,
    target_fps: int,
    target_size: int,
    workers: int,
    allowed_cams: frozenset[str],
) -> tuple[int, int, int]:
    """Extract RGB mp4s from a tar and transcode each. Returns (ok, fail, skipped)."""
    extract_dir = scratch_root / (tar_path.stem + "_extract")
    if extract_dir.exists():
        shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = n_skip = 0
    n_seen = 0
    submitted: list = []
    # At most workers * 2 extracted videos wait for transcoding at any time.
    pending_slots = threading.Semaphore(workers * 2)

    try:
        with tarfile.open(tar_path, "r|*") as tar, ThreadPoolExecutor(max_workers=workers) as pool:
            # Stream members in archive order.
            for member in tar:
                if not member.isfile():
                    continue
                parsed = parse_member_path(member.name, allowed_cams)
                if parsed is None:
                    continue
                episode_id, cam_file = parsed
                n_seen += 1
                dst = output_root / task_id / episode_id / cam_file
                # Existing non-empty outputs are kept (resume).
                if dst.exists() and dst.stat().st_size > 0:
                    n_skip += 1
                    continue

                pending_slots.acquire()
                tar.extract(member, extract_dir, set_attrs=False)
                src = extract_dir / member.name

                def _job(src=src, dst=dst):
                    try:
                        ok = ffmpeg_transcode(src, dst, target_fps, target_size)
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
        log.warning("  no matching cameras found in %s", tar_path.name)

    return n_ok, n_fail, n_skip


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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=Path, required=True,
                   help="Folder that holds all processed training datasets.")
    p.add_argument("--subdir", default="agibot_processed",
                   help="Output folder under --data_root.")
    p.add_argument("--work_dir", type=Path, required=True,
                   help="Scratch folder for shard downloads and extraction.")
    p.add_argument("--target_fps", type=int, default=16)
    p.add_argument("--target_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel ffmpeg workers per shard.")
    p.add_argument("--task_filter", nargs="*", default=None,
                   help="Process only these task ids (e.g. 327 352). Default: all.")
    p.add_argument("--max_tars", type=int, default=None,
                   help="Process at most N shards (quick test).")
    p.add_argument("--keep_tar", action="store_true",
                   help="Keep each shard after processing.")
    p.add_argument("--dry_run", action="store_true",
                   help="List the shards and exit without downloading.")
    p.add_argument("--cams", nargs="+", default=list(DEFAULT_CAMS),
                   help="Camera mp4 file names to keep (default: the three main RGB streams).")
    args = p.parse_args()
    allowed_cams = frozenset(args.cams)

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH.")

    output_root = args.data_root / args.subdir
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    log.info("Listing tar shards in %s ...", REPO_ID)
    try:
        tars = list_tar_files(api)
    except HfHubHTTPError as e:
        log.error("Could not list repo files (gated dataset?): %s", e)
        log.error("Run `huggingface-cli login` and accept the licence on")
        log.error("  https://huggingface.co/datasets/%s", REPO_ID)
        return 2

    if args.task_filter:
        keep = set(str(t) for t in args.task_filter)
        tars = [t for t in tars if t.split("/")[1] in keep]
    log.info("Found %d tar shards to process", len(tars))
    if args.max_tars is not None:
        tars = tars[: args.max_tars]
        log.info("Limited to first %d for this run", len(tars))

    if args.dry_run:
        for t in tars:
            print(t)
        return 0

    total_ok = total_fail = total_skip = 0
    run_start = time.time()

    for idx, repo_path in enumerate(tars, 1):
        log.info("[%d/%d] %s", idx, len(tars), repo_path)
        task_id = repo_path.split("/")[1]  # observations/{task_id}/{file}.tar

        # 1) Download the shard into work_dir as a regular file.
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

        # 2) Extract and transcode.
        t1 = time.time()
        try:
            n_ok, n_fail, n_skip = process_tar(
                local_path, task_id, output_root, work_dir,
                args.target_fps, args.target_size, args.workers, allowed_cams,
            )
        except Exception as e:
            log.exception("  processing failed: %s", e)
            n_ok, n_fail, n_skip = 0, 1, 0
        log.info("  processed in %.0fs: ok=%d fail=%d skipped=%d",
                 time.time() - t1, n_ok, n_fail, n_skip)
        total_ok += n_ok
        total_fail += n_fail
        total_skip += n_skip

        # 3) Delete the shard.
        if not args.keep_tar:
            delete_downloaded_file(local_path)

    log.info("Done in %.0fs total. videos ok=%d fail=%d skipped=%d",
             time.time() - run_start, total_ok, total_fail, total_skip)
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

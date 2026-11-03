#!/usr/bin/env python3
"""Prepare training data of the JustAct encoder: VPT Minecraft frames and actions.

Downloads OpenAI's VPT contractor recordings (public Azure blob storage) and
extracts fixed-length segments of JPEG frames plus encoded keyboard / mouse
actions. The input is one of the public VPT index files, e.g.
    https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_6xx_Jun_29.json
which holds {"basedir": ..., "relpaths": ["data/6.10/<player-episode>.mp4", ...]};
the mp4 is at basedir + relpath and its per-frame actions at the same path with
.jsonl.

Per source episode:
    1. download .mp4 and .jsonl to --tmp_dir,
    2. read the frame count with ffprobe,
    3. cut into consecutive segments of --segment_len frames (at most
       --max_segments_per_episode),
    4. per segment pick --frames_per_segment frames with stride --sample_stride,
       extract them with ffmpeg (JPEG, -q:v 3), encode the matching JSONL lines
       into actions.npy,
    5. delete the downloads.
The run stops once --max_segments segments are ready (episodes already being
processed at that moment are finished).

Output layout under <data_root>/<subdir>/:
    episodes/<subversion>/<player_episode>/seg<K:02d>/
        frames/frame_<i:05d>.jpg   (native resolution)
        actions.npy                (frames_per_segment, 25) float32
        actions_raw.jsonl          raw JSON lines of the sampled frames
        metadata.json              segment info
        _frames_ready              marker
    episodes/<subversion>/<player_episode>/_episode_complete

Run vpt_build_index.py afterwards to write index.json.

Requires ffmpeg and ffprobe on PATH and curl.

Usage (settings of the training data):
    python vpt_process.py --index all_6xx_Jun_29.json --data_root <DATA_ROOT> \
        --segment_len 101 --frames_per_segment 101 --sample_stride 1 \
        --max_segments_per_episode 3 --max_segments 10012 --download_workers 32
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE_BIN = shutil.which("ffprobe") or "ffprobe"

# Encoded action schema: 21 binary keys + 2 mouse delta + 2 mouse buttons = 25 dims
ACTION_KEYS = [
    "key.keyboard.w", "key.keyboard.a", "key.keyboard.s", "key.keyboard.d",
    "key.keyboard.space", "key.keyboard.left.shift", "key.keyboard.left.control",
    "key.keyboard.q", "key.keyboard.e", "key.keyboard.escape", "key.keyboard.tab",
    "key.keyboard.1", "key.keyboard.2", "key.keyboard.3", "key.keyboard.4",
    "key.keyboard.5", "key.keyboard.6", "key.keyboard.7", "key.keyboard.8",
    "key.keyboard.9", "key.keyboard.f",
]
ACTION_DIM = len(ACTION_KEYS) + 2 + 2   # keys + (dx, dy) + (left_btn, right_btn)


# ---------------------------------------------------------------------------
# Download / video helpers
# ---------------------------------------------------------------------------

def http_download(url: str, dst: Path, timeout: int = 900,
                  min_bytes: int = 1024) -> bool:
    """Download with curl (3 retries, size check). Returns True on success."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    cmd = [
        "curl", "-sfL", "--retry", "3", "--retry-delay", "2",
        "--connect-timeout", "30", "--max-time", str(timeout),
        "-o", str(tmp), url,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
    except subprocess.TimeoutExpired:
        log.warning(f"  curl TIMEOUT {url}")
        if tmp.exists():
            tmp.unlink()
        return False
    if r.returncode != 0:
        log.warning(f"  curl rc={r.returncode} {url}: stderr={r.stderr[:300]}")
        if tmp.exists():
            tmp.unlink()
        return False
    size = tmp.stat().st_size if tmp.exists() else 0
    if size < min_bytes:
        log.warning(f"  download too small: {size} bytes -- {url}")
        if tmp.exists():
            tmp.unlink()
        return False
    tmp.rename(dst)
    return True


def ffprobe_nframes(mp4: Path) -> int:
    """Video frame count: container nb_frames, else duration * fps, else a decode count."""
    cmd_fast = [
        FFPROBE_BIN, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(mp4),
    ]
    try:
        r = subprocess.run(cmd_fast, capture_output=True, text=True, timeout=30)
        val = (r.stdout or "").strip()
        if val and val != "N/A":
            n = int(val)
            if n > 0:
                return n
    except Exception:
        pass
    cmd_duration = [
        FFPROBE_BIN, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration,r_frame_rate",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(mp4),
    ]
    try:
        r = subprocess.run(cmd_duration, capture_output=True, text=True, timeout=30)
        lines = [l for l in (r.stdout or "").splitlines() if l.strip()]
        if len(lines) >= 2:
            dur = float(lines[0])
            num, den = lines[1].split("/")
            fps = float(num) / float(den)
            n = int(dur * fps)
            if n > 0:
                return n
    except Exception:
        pass
    cmd_slow = [
        FFPROBE_BIN, "-v", "error", "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(mp4),
    ]
    try:
        r = subprocess.run(cmd_slow, capture_output=True, text=True, timeout=300)
        val = (r.stdout or "").strip()
        return int(val) if val and val != "N/A" else 0
    except Exception:
        return 0


def extract_indexed_frames(mp4: Path, out_dir: Path, frame_indices: list) -> list:
    """Extract the given frame indices from the mp4. Returns the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # One ffmpeg select filter: select='eq(n,i1)+eq(n,i2)+...'
    expr = "+".join(f"eq(n\\,{i})" for i in frame_indices)
    cmd = [
        FFMPEG_BIN, "-y", "-loglevel", "error",
        "-i", str(mp4),
        "-vf", f"select='{expr}'",
        "-vsync", "vfr",
        "-q:v", "3",
        "-start_number", "0",
        str(out_dir / "frame_%05d.jpg"),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        log.warning(f"ffmpeg failed for {mp4.name}: {r.stderr[:300]}")
        return []
    written = sorted(out_dir.glob("frame_*.jpg"))
    return written


# ---------------------------------------------------------------------------
# JSONL action parsing
# ---------------------------------------------------------------------------

def read_jsonl_lines(path: Path) -> list:
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({})
    return out


def encode_action(entry: dict) -> np.ndarray:
    """Encode one frame's action dict into a fixed-size float32 vector."""
    vec = np.zeros(ACTION_DIM, dtype=np.float32)
    kb = entry.get("keyboard") or {}
    keys = set(kb.get("keys") or [])
    for i, k in enumerate(ACTION_KEYS):
        if k in keys:
            vec[i] = 1.0
    mouse = entry.get("mouse") or {}
    vec[len(ACTION_KEYS) + 0] = float(mouse.get("dx") or 0.0)
    vec[len(ACTION_KEYS) + 1] = float(mouse.get("dy") or 0.0)
    buttons = set(mouse.get("buttons") or [])
    vec[len(ACTION_KEYS) + 2] = 1.0 if 0 in buttons else 0.0  # left click
    vec[len(ACTION_KEYS) + 3] = 1.0 if 1 in buttons else 0.0  # right click
    return vec


# ---------------------------------------------------------------------------
# One episode: download + extract frames + encode actions (worker thread)
# ---------------------------------------------------------------------------

def _frames_complete(frames_dir: Path, n: int) -> bool:
    if not frames_dir.is_dir():
        return False
    return (frames_dir / f"frame_{n-1:05d}.jpg").exists()


def prefetch_episode(
    relpath: str,
    basedir: str,
    output_root: Path,
    tmp_root: Path,
    segment_len: int,
    frames_per_segment: int,
    max_segments_per_ep: int,
    sample_stride: int = 10,
) -> dict:
    """Download one episode and write frames + actions for all its planned segments.

    Each finished segment gets a `_frames_ready` marker. Returns a summary dict.
    """
    subversion = relpath.split("/")[1]
    ep_stem = Path(relpath).stem
    ep_out_dir = output_root / "episodes" / subversion / ep_stem
    ep_tmp = tmp_root / subversion / ep_stem

    summary = {
        "episode_id": f"{subversion}/{ep_stem}",
        "relpath": relpath,
        "subversion": subversion,
        "n_frames": 0,
        "n_segments": 0,
        "ready_segs": [],
        "already_done_segs": [],
        "error": None,
    }

    # Whole episode already finished: skip the download.
    if (ep_out_dir / "_episode_complete").exists():
        summary["already_done_segs"] = sorted(ep_out_dir.glob("seg*/_frames_ready"))
        summary["n_segments"] = len(summary["already_done_segs"])
        return summary

    ep_tmp.mkdir(parents=True, exist_ok=True)
    mp4_path = ep_tmp / "video.mp4"
    jsonl_path = ep_tmp / "actions.jsonl"

    try:
        t_dl = time.time()
        if not http_download(basedir + relpath, mp4_path):
            summary["error"] = "mp4 download failed"
            return summary
        mp4_size = mp4_path.stat().st_size
        log.info(f"  [{summary['episode_id']}] mp4 {mp4_size/1e6:.1f}MB {time.time()-t_dl:.1f}s")
        if not http_download(basedir + relpath.replace(".mp4", ".jsonl"), jsonl_path):
            summary["error"] = "jsonl download failed"
            return summary

        n_frames = ffprobe_nframes(mp4_path)
        summary["n_frames"] = n_frames
        if n_frames < frames_per_segment:
            summary["error"] = f"too few frames ({n_frames})"
            return summary

        jsonl_entries = read_jsonl_lines(jsonl_path)

        # Plan consecutive segments
        plans = []
        start = 0
        while start + segment_len <= n_frames and len(plans) < max_segments_per_ep:
            plans.append((start, start + segment_len))
            start += segment_len
        if not plans and n_frames >= frames_per_segment:
            plans.append((0, n_frames))
        summary["n_segments"] = len(plans)

        for k, (a, b) in enumerate(plans):
            seg_out = ep_out_dir / f"seg{k:02d}"
            seg_out.mkdir(parents=True, exist_ok=True)
            frames_dir = seg_out / "frames"

            if (seg_out / "_frames_ready").exists() and _frames_complete(frames_dir, frames_per_segment):
                summary["already_done_segs"].append(seg_out)
                continue

            if sample_stride == 0:
                selected = np.linspace(a, b - 1, frames_per_segment,
                                       dtype=np.int64).tolist()
            else:
                sel = np.arange(a, a + frames_per_segment * sample_stride,
                                sample_stride, dtype=np.int64)
                sel = sel[:frames_per_segment]
                sel = np.clip(sel, a, b - 1)
                selected = sel.tolist()

            if not _frames_complete(frames_dir, frames_per_segment):
                if frames_dir.exists():
                    shutil.rmtree(frames_dir, ignore_errors=True)
                out_paths = extract_indexed_frames(mp4_path, frames_dir, selected)
                if len(out_paths) < frames_per_segment:
                    log.warning(f"  seg {k}: ffmpeg emitted {len(out_paths)}/{frames_per_segment}")
                    shutil.rmtree(frames_dir, ignore_errors=True)
                    continue

            actions_arr = np.zeros((frames_per_segment, ACTION_DIM), dtype=np.float32)
            with open(seg_out / "actions_raw.jsonl", "w") as f:
                for i, frame_idx in enumerate(selected):
                    entry = jsonl_entries[frame_idx] if frame_idx < len(jsonl_entries) else {}
                    f.write(json.dumps(entry) + "\n")
                    actions_arr[i] = encode_action(entry)
            np.save(seg_out / "actions.npy", actions_arr)

            with open(seg_out / "metadata.json", "w") as f:
                json.dump({
                    "seg_start_frame": int(a),
                    "seg_end_frame": int(b),
                    "frames_per_segment": int(frames_per_segment),
                    "n_pairs": int(frames_per_segment - 1),
                    "selected_frames": [int(x) for x in selected],
                    "action_dim": int(ACTION_DIM),
                }, f, indent=2)

            (seg_out / "_frames_ready").write_text(
                json.dumps({"at": time.strftime("%Y-%m-%d %H:%M:%S")})
            )
            summary["ready_segs"].append(seg_out)

    except Exception as e:
        summary["error"] = f"exception: {e}"
        log.exception(f"prefetch error for {summary['episode_id']}")
    finally:
        shutil.rmtree(ep_tmp, ignore_errors=True)

    return summary


# ---------------------------------------------------------------------------
# Main loop: parallel episodes until max_segments segments are ready
# ---------------------------------------------------------------------------

def run_download(relpaths, basedir, output_root, tmp_root,
                 segment_len, frames_per_segment, max_segments_per_ep,
                 max_segments, download_workers, sample_stride=10):
    log.info(f"=== download + frames: {download_workers} workers, "
             f"target {max_segments} segs ===")

    def count_ready() -> int:
        ep_root = output_root / "episodes"
        if not ep_root.exists():
            return 0
        return sum(1 for _ in ep_root.rglob("_frames_ready"))

    pre_ready = count_ready()
    log.info(f"pre-existing ready segments: {pre_ready}")
    if pre_ready >= max_segments:
        log.info("already at/over target; nothing to do")
        return

    t0 = time.time()
    relpath_iter = iter(enumerate(relpaths))
    in_flight: dict = {}
    stats = {"ok_eps": 0, "dl_fail": 0, "short": 0, "noop": 0, "new_segs": 0}

    def submit_more(pool, target):
        while len(in_flight) < target:
            try:
                i, rp = next(relpath_iter)
            except StopIteration:
                return
            fut = pool.submit(
                prefetch_episode, rp, basedir, output_root, tmp_root,
                segment_len, frames_per_segment, max_segments_per_ep,
                sample_stride,
            )
            in_flight[fut] = (i, rp)

    target_in_flight = download_workers + 1
    with ThreadPoolExecutor(max_workers=download_workers,
                            thread_name_prefix="vpt-prefetch") as pool:
        submit_more(pool, target_in_flight)
        ready = pre_ready
        while in_flight and ready < max_segments:
            done_set, _ = wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)
            for fut in done_set:
                i, rp = in_flight.pop(fut)
                try:
                    summary = fut.result()
                except Exception as e:
                    log.exception(f"prefetch failed for {rp}: {e}")
                    continue

                if summary["error"] == "mp4 download failed":
                    stats["dl_fail"] += 1
                elif summary["error"] and "too few" in summary["error"]:
                    stats["short"] += 1
                elif summary["ready_segs"]:
                    stats["ok_eps"] += 1
                    stats["new_segs"] += len(summary["ready_segs"])
                    ep_dir = Path(summary["ready_segs"][0]).parent
                    if len(list(ep_dir.glob("seg*/_frames_ready"))) >= summary["n_segments"]:
                        (ep_dir / "_episode_complete").write_text(
                            json.dumps({"at": time.strftime("%Y-%m-%d %H:%M:%S")})
                        )
                else:
                    stats["noop"] += 1

                ready = count_ready()
                log.info(
                    f"[{i+1}/{len(relpaths)}] {rp} nfr={summary['n_frames']} "
                    f"plan={summary['n_segments']} ready={len(summary['ready_segs'])} "
                    f"err={summary['error']} | tot_ready={ready}/{max_segments}"
                )
                submit_more(pool, target_in_flight)
            if ready >= max_segments:
                break
        # Queued episodes are cancelled; running ones finish before the pool exits.
        for fut in list(in_flight.keys()):
            fut.cancel()
    log.info(f"DONE: {stats}, final ready={count_ready()}, "
             f"elapsed {(time.time()-t0)/60:.1f} min")


def main():
    ap = argparse.ArgumentParser(description="Download VPT episodes and extract frame segments")
    ap.add_argument("--index", required=True,
                    help="VPT index JSON (e.g. all_6xx_Jun_29.json)")
    ap.add_argument("--data_root", required=True,
                    help="Folder that holds all processed datasets")
    ap.add_argument("--subdir", default="vpt_minecraft_stride1",
                    help="VPT output folder inside data_root")
    ap.add_argument("--tmp_dir", default=None,
                    help="Scratch folder for downloads (default: <system temp>/vpt_work)")
    ap.add_argument("--segment_len", type=int, default=2000,
                    help="Frames per segment of the source video")
    ap.add_argument("--frames_per_segment", type=int, default=101,
                    help="Frames kept per segment")
    ap.add_argument("--max_segments_per_episode", type=int, default=10,
                    help="Cap on segments per source episode")
    ap.add_argument("--max_segments", type=int, default=5000,
                    help="Stop once this many segments are ready")
    ap.add_argument("--shuffle_seed", type=int, default=17,
                    help="Seed of the deterministic shuffle of the relpath list")
    ap.add_argument("--episode_ids", default=None,
                    help="Optional text file of episode or segment ids (e.g. "
                         "vpt_segment_ids.txt); only these episodes are processed "
                         "(shuffled order kept)")
    ap.add_argument("--download_workers", type=int, default=4,
                    help="Parallel episodes (download + frame extraction)")
    ap.add_argument("--sample_stride", type=int, default=10,
                    help="Stride on source frames between kept frames "
                         "(1 = consecutive); 0 = np.linspace over the segment")
    args = ap.parse_args()

    output_root = Path(args.data_root) / args.subdir
    output_root.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(args.tmp_dir or os.path.join(tempfile.gettempdir(), "vpt_work"))
    tmp_root.mkdir(parents=True, exist_ok=True)

    with open(args.index) as f:
        raw = json.load(f)
    basedir = raw["basedir"]
    relpaths = list(raw["relpaths"])

    # Deterministic shuffle of the episode list
    rng = np.random.default_rng(args.shuffle_seed)
    order = np.arange(len(relpaths))
    rng.shuffle(order)
    relpaths = [relpaths[i] for i in order]

    if args.episode_ids:
        # Lines may be episode ids or segment ids ('<subversion>/<player_episode>/seg<K>').
        with open(args.episode_ids) as f:
            keep = {"/".join(line.strip().split("/")[:2]) for line in f if line.strip()}
        relpaths = [rp for rp in relpaths
                    if f"{rp.split('/')[1]}/{Path(rp).stem}" in keep]
        log.info(f"restricted to {len(relpaths)} episodes listed in {args.episode_ids}")

    t_global = time.time()
    run_download(
        relpaths=relpaths,
        basedir=basedir,
        output_root=output_root,
        tmp_root=tmp_root,
        segment_len=args.segment_len,
        frames_per_segment=args.frames_per_segment,
        max_segments_per_ep=args.max_segments_per_episode,
        max_segments=args.max_segments,
        download_workers=args.download_workers,
        sample_stride=args.sample_stride,
    )
    log.info(f"Total elapsed: {(time.time() - t_global) / 3600:.2f}h")


if __name__ == "__main__":
    main()

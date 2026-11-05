#!/usr/bin/env python3
"""Prepare training data of the OnlyAct encoder (EgoDex part).

Streams the EgoDex training parts 1-5 from Apple's public server
(https://ml-site.cdn-apple.com/datasets/egodex/partN.zip) and re-encodes every
video to 256x256 H.264 at 16 frames per second. The paired .hdf5 pose files
are not downloaded, and no full zip is ever stored on disk. For each part:
  1. read the zip's central directory with HTTP Range requests;
  2. list the .mp4 members;
  3. for each member, fetch its bytes with an HTTP Range request into
     --work_dir, transcode it with libx264 (crf 23, preset fast; shorter side
     scaled to 256, then center-cropped to 256x256) and delete the scratch file.

Zip layout: partN/{task_name}/{episode_id}.mp4 (+ .hdf5).

Output layout (the union manifest builder globs these files):
  <data_root>/<subdir>/{task_name}/{episode_id}.mp4

An output that already exists (non-empty) is kept, so a member whose
{task_name}/{episode_id}.mp4 was written by an earlier part is not rewritten.

Usage (training data: parts 1-5, the default):
  python egodex_download_and_process.py --data_root <DATA_ROOT> --work_dir <WORK_DIR>
"""
from __future__ import annotations

import argparse
import io
import logging
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import requests
    import requests.adapters
except ImportError:
    sys.exit("Install requests: pip install requests")


BASE_URL = "https://ml-site.cdn-apple.com/datasets/egodex"
DEFAULT_PARTS = ("part1", "part2", "part3", "part4", "part5")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("egodex")


# ---------------------------------------------------------------------------
# HTTP utilities: remote-zip streaming
# ---------------------------------------------------------------------------

def _head_total(url: str) -> int:
    """Resolve Content-Length, following redirects."""
    r = requests.head(url, allow_redirects=True, timeout=30)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def http_range_get(session: requests.Session, url: str, start: int, end_inclusive: int,
                   read_timeout: float = 120.0, max_retries: int = 3) -> bytes:
    """Fetch bytes [start, end_inclusive] via HTTP Range, with retries."""
    headers = {"Range": f"bytes={start}-{end_inclusive}"}
    expected = end_inclusive - start + 1
    last_err = None
    for attempt in range(max_retries):
        try:
            r = session.get(url, headers=headers, timeout=(15.0, read_timeout))
            if r.status_code not in (200, 206):
                raise IOError(f"unexpected HTTP {r.status_code}")
            data = r.content
            if len(data) != expected:
                raise IOError(f"short fetch: got {len(data)} expected {expected}")
            return data
        except (requests.RequestException, IOError) as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt, 10))
    raise IOError(f"range {start}-{end_inclusive} failed: {last_err}")


class HTTPSeekableFile(io.RawIOBase):
    """Read-only, seekable file object backed by HTTP Range requests.

    Lets `zipfile.ZipFile` read the central directory at the end of the remote
    zip without downloading the whole file. Keeps the last fetched window to
    serve small nearby reads.
    """

    def __init__(self, url: str, total: int, session: requests.Session | None = None):
        self.url = url
        self._total = total
        self._pos = 0
        self._session = session or requests.Session()
        self._cache_start = -1
        self._cache_data = b""

    @property
    def total_size(self) -> int:
        return self._total

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._total + offset
        else:
            raise ValueError(f"bad whence {whence}")
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self._total - self._pos
        if size <= 0:
            return b""
        end = min(self._pos + size, self._total)
        if (self._cache_start <= self._pos
                and self._pos + size <= self._cache_start + len(self._cache_data)):
            off = self._pos - self._cache_start
            data = self._cache_data[off:off + size]
            self._pos += len(data)
            return data
        # Fetch at least 256 KB per request.
        fetch_size = max(size, 256 * 1024)
        fetch_end = min(self._pos + fetch_size, self._total) - 1
        data = http_range_get(self._session, self.url, self._pos, fetch_end)
        self._cache_start = self._pos
        self._cache_data = data
        out = data[:size]
        self._pos += len(out)
        return out

    def close(self) -> None:
        # The session is owned by the caller.
        super().close()


def fetch_member_bytes(session: requests.Session, url: str, info: zipfile.ZipInfo) -> bytes:
    """Fetch and decompress a single zip member via HTTP Range.

    Reads the 30-byte local file header to locate the data start, then the
    compressed data itself.
    """
    hdr = http_range_get(session, url, info.header_offset, info.header_offset + 29)
    if hdr[:4] != b"PK\x03\x04":
        raise IOError(f"bad local file header sig for {info.filename}")
    fn_len = int.from_bytes(hdr[26:28], "little")
    extra_len = int.from_bytes(hdr[28:30], "little")
    data_offset = info.header_offset + 30 + fn_len + extra_len

    if info.compress_size == 0:
        compressed = b""
    else:
        compressed = http_range_get(session, url, data_offset,
                                    data_offset + info.compress_size - 1)
    if info.compress_type == zipfile.ZIP_STORED:
        return compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        import zlib
        return zlib.decompress(compressed, -15)
    else:
        raise ValueError(f"unsupported compression method {info.compress_type}")


def _make_session() -> requests.Session:
    """A requests.Session with connection pooling for many short Range fetches."""
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=4, pool_maxsize=8,
        max_retries=0,  # retries are handled in http_range_get
    )
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


# ---------------------------------------------------------------------------
# Transcode
# ---------------------------------------------------------------------------

def ffmpeg_transcode(src: Path, dst: Path, target_fps: int, target_size: int) -> bool:
    """Resize and re-encode one mp4. Returns True on success."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Temporary name keeps the .mp4 suffix so ffmpeg can infer the container.
    tmp = dst.parent / f".part_{dst.name}"
    # Scale the shorter side to target_size, then center-crop to a square.
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


def parse_zip_member(name: str) -> tuple[str, str] | None:
    """Map a zip member partN/{task_name}/{episode_id}.mp4 to
    (task_name, episode_file) or None for any other member."""
    if not name.endswith(".mp4"):
        return None
    parts = name.split("/")
    if len(parts) < 3:
        return None
    # Drop the leading partN/; the folders in between form the task name.
    task_name = "/".join(parts[1:-1])
    if not task_name:
        return None
    ep_file = parts[-1]
    if ep_file.startswith("."):  # hidden files
        return None
    return task_name, ep_file


# ---------------------------------------------------------------------------
# Zip processing
# ---------------------------------------------------------------------------

def process_zip_remote(
    url: str,
    output_root: Path,
    scratch_root: Path,
    target_fps: int,
    target_size: int,
    workers: int,
    max_videos: int | None = None,
) -> tuple[int, int, int, int]:
    """Stream a remote zip: list members from the central directory, fetch
    each .mp4 member via HTTP Range, transcode, discard.
    Returns (ok, fail, skipped, seen). At most workers * 2 members are held in
    scratch_root at a time."""
    extract_dir = scratch_root / "members_inflight"
    if extract_dir.exists():
        shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = n_skip = n_seen = 0
    pending_slots = threading.Semaphore(workers * 2)

    # 1. Total size of the remote zip.
    log.info("  HEAD %s", url)
    cd_session = _make_session()
    total = _head_total(url)
    log.info("  remote zip size: %.1f GB", total / 1e9)

    # 2. Read only the end-of-central-directory record and the central directory.
    http_file = HTTPSeekableFile(url, total, session=cd_session)
    t_cd0 = time.time()
    with zipfile.ZipFile(http_file) as zf:
        all_infos = zf.infolist()
    log.info("  zip has %d entries, central directory read in %.1fs",
             len(all_infos), time.time() - t_cd0)

    # 3. Keep the .mp4 members, sorted by offset in the zip.
    candidates = []
    for info in all_infos:
        parsed = parse_zip_member(info.filename)
        if parsed is None:
            continue
        candidates.append((info, *parsed))
    candidates.sort(key=lambda t: t[0].header_offset)
    log.info("  matched %d candidate .mp4 members", len(candidates))

    # 4. One HTTP session per worker thread.
    tls = threading.local()

    def get_session() -> requests.Session:
        s = getattr(tls, "session", None)
        if s is None:
            s = _make_session()
            tls.session = s
        return s

    submitted: list = []
    t_proc0 = time.time()
    counts = {"ok": 0, "fail": 0, "bytes": 0}
    counts_lock = threading.Lock()
    stop_progress = threading.Event()

    def progress_thread() -> None:
        """Log progress every 30 s."""
        while not stop_progress.is_set():
            time.sleep(30.0)
            if stop_progress.is_set():
                return
            with counts_lock:
                mb = counts["bytes"] / 1e6
                fok, ff = counts["ok"], counts["fail"]
            mbps = mb / max(0.001, time.time() - t_proc0)
            log.info("    progress: %d ok / %d fail of ~%d (%.1f MB/s avg)",
                     fok, ff, len(candidates), mbps)

    prog = threading.Thread(target=progress_thread, daemon=True, name="prog")
    prog.start()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for info, task_name, ep_file in candidates:
            n_seen += 1
            if max_videos is not None and n_seen > max_videos:
                break
            dst = output_root / task_name / ep_file
            # Existing non-empty outputs are kept (resume).
            if dst.exists() and dst.stat().st_size > 0:
                n_skip += 1
                continue

            pending_slots.acquire()

            def _job(info=info, task_name=task_name, ep_file=ep_file, dst=dst):
                src = extract_dir / f".m_{threading.get_ident()}_{info.header_offset}.mp4"
                ok = False
                try:
                    try:
                        data = fetch_member_bytes(get_session(), url, info)
                        with counts_lock:
                            counts["bytes"] += len(data)
                        src.write_bytes(data)
                        ok = ffmpeg_transcode(src, dst, target_fps, target_size)
                    except Exception as e:
                        log.error("  failed on %s: %s", info.filename, e)
                        ok = False
                finally:
                    try:
                        if src.exists():
                            src.unlink()
                    except OSError:
                        pass
                    with counts_lock:
                        if ok:
                            counts["ok"] += 1
                        else:
                            counts["fail"] += 1
                    pending_slots.release()
                return ok

            submitted.append(pool.submit(_job))

        for fut in as_completed(submitted):
            if fut.result():
                n_ok += 1
            else:
                n_fail += 1

    stop_progress.set()
    shutil.rmtree(extract_dir, ignore_errors=True)

    if n_seen == 0:
        log.warning("  no matching mp4 members found in zip")

    return n_ok, n_fail, n_skip, n_seen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=Path, required=True,
                   help="Folder that holds all processed training datasets.")
    p.add_argument("--subdir", default="egodex_processed",
                   help="Output folder under --data_root.")
    p.add_argument("--work_dir", type=Path, required=True,
                   help="Scratch folder for in-flight zip members.")
    p.add_argument("--parts", nargs="+", default=list(DEFAULT_PARTS),
                   help="Parts to process (default: part1 ... part5).")
    p.add_argument("--target_fps", type=int, default=16)
    p.add_argument("--target_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel fetch + ffmpeg workers.")
    p.add_argument("--max_videos_per_zip", type=int, default=None,
                   help="Cap videos per zip (quick test).")
    p.add_argument("--dry_run", action="store_true",
                   help="Only print the size of each part and exit.")
    args = p.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH.")

    output_root = args.data_root / args.subdir
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for part in args.parts:
            url = f"{BASE_URL}/{part}.zip"
            try:
                size = _head_total(url)
                print(f"{part}.zip\t{size / 1e9:6.1f} GB\t{url}")
            except Exception as e:
                print(f"{part}.zip\tERROR\t{e}")
        return 0

    total_ok = total_fail = total_skip = 0
    run_start = time.time()

    for idx, part in enumerate(args.parts, 1):
        url = f"{BASE_URL}/{part}.zip"
        log.info("[%d/%d] %s", idx, len(args.parts), part)

        t1 = time.time()
        try:
            n_ok, n_fail, n_skip, n_seen = process_zip_remote(
                url, output_root, work_dir,
                args.target_fps, args.target_size, args.workers,
                max_videos=args.max_videos_per_zip,
            )
        except Exception as e:
            log.exception("  processing failed: %s", e)
            n_ok, n_fail, n_skip, n_seen = 0, 1, 0, 0
        log.info("  zip done in %.0fs: seen=%d ok=%d fail=%d skipped=%d",
                 time.time() - t1, n_seen, n_ok, n_fail, n_skip)
        total_ok += n_ok
        total_fail += n_fail
        total_skip += n_skip

        # Stop if the first zip yields no clips (unexpected archive layout).
        if idx == 1 and total_ok == 0 and total_skip == 0:
            log.error("First zip produced 0 outputs; check the archive layout.")
            return 2

    log.info("Done in %.0fs. videos ok=%d fail=%d skipped=%d",
             time.time() - run_start, total_ok, total_fail, total_skip)
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

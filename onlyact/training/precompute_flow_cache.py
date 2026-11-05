"""Precompute the optical flow a0 -> a1 of every cached sample (RAFT-large) for the background-replaced positives.

Format: see flow_cache.py. Flow from torchvision's raft_large with the C_T_SKHT_V2 weights, inputs squashed to
256x256 (cv2.INTER_AREA when shrinking, cv2.INTER_CUBIC when enlarging) and scaled to [-1, 1], 12 flow updates,
float16 autocast, then average-pooled 2x to 128x128 and quantised to int8. mag_fg / mag_bg are computed from the
dequantised flow, i.e. exactly what the data loader recomputes.

Resumable (existing files are skipped; each file is written under a temporary name and renamed). One RTX 4090: about
17 minutes for the 168,000 samples (5.8 GB).

    python -m onlyact.training.precompute_flow_cache --cache_dir <cache>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import flow_cache as FC

cv2.setNumThreads(1)


def frame_path(cache_dir, i, key):
    return os.path.join(cache_dir, "frames", f"{i // 1000:03d}", f"{i:06d}_{key}.jpg")


def read_frame_256(path, res=FC.RAFT_RES):
    """JPEG -> RGB uint8 [res, res, 3] squashed to a square; None if unreadable."""
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None
    if im.shape[0] != res or im.shape[1] != res:
        interp = cv2.INTER_AREA if im.shape[0] >= res else cv2.INTER_CUBIC
        im = cv2.resize(im, (res, res), interpolation=interp)
    return np.ascontiguousarray(im[:, :, ::-1])


def read_index(cache_dir, datasets=None):
    """index.jsonl -> sorted list of (i, dataset)."""
    keep = set(datasets) if datasets else None
    out = []
    with open(os.path.join(cache_dir, "index.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if keep is None or d["dataset"] in keep:
                out.append((int(d["i"]), d["dataset"]))
    out.sort()
    return out


def load_raft(device):
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    return raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(device).eval()


@torch.no_grad()
def raft_flow_128(raft, f0_u8, f1_u8, device, updates=12, dtype=torch.float16):
    """[B,256,256,3] uint8 x2 -> float32 [B,2,128,128] flow (a0 -> a1) in 256-grid pixels, average-pooled 2x."""
    x0 = torch.from_numpy(f0_u8).to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1
    x1 = torch.from_numpy(f1_u8).to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1
    with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        fl = raft(x0, x1, num_flow_updates=updates)[-1]
    fl = F.avg_pool2d(fl.float(), kernel_size=2)  # the mean of displacement vectors keeps the 256-grid units
    return fl.cpu().numpy()


def quantise(flow_px):
    """float32 [2,128,128] -> (int8 [2,128,128], scale) with scale = max|flow| / 127 (1.0 for an all-zero flow)."""
    m = float(np.abs(flow_px).max())
    scale = np.float32(m / 127.0) if m > 0 else np.float32(1.0)
    q = np.clip(np.rint(flow_px / scale), -127, 127).astype(np.int8)
    return q, scale


def write_npz_atomic(path, q, scale, mag_fg, mag_bg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + f".tmp{os.getpid()}.npz"
    try:
        np.savez(tmp, flow=q, scale=np.float32(scale), mag_fg=np.float32(mag_fg), mag_bg=np.float32(mag_bg))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return os.path.getsize(path)


def clean_stale_tmp(flow_dir, max_age_s=600):
    """Remove temporary files older than max_age_s left by an interrupted run."""
    n = 0
    now = time.time()
    for p in glob.glob(os.path.join(flow_dir, "*", "*.tmp*.npz")):
        try:
            if now - os.path.getmtime(p) > max_age_s:
                os.remove(p)
                n += 1
        except OSError:
            pass
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="process at most this many not-yet-cached samples (0 = all)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8, help="threads for JPEG decoding and resizing")
    ap.add_argument("--datasets", default="", help="comma list (default: all)")
    ap.add_argument("--start", type=int, default=0, help="shard offset into the i-sorted list")
    ap.add_argument("--stride", type=int, default=1, help="shard stride (two processes: --start 0/1 --stride 2)")
    ap.add_argument("--flow_updates", type=int, default=12)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16", help="autocast dtype")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report_every", type=int, default=200)
    ap.add_argument("--torch_threads", type=int, default=4)
    a = ap.parse_args()

    torch.set_num_threads(a.torch_threads)
    torch.backends.cudnn.benchmark = True
    device = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    flow_dir = os.path.join(a.cache_dir, FC.FLOW_SUBDIR)
    os.makedirs(flow_dir, exist_ok=True)
    n_tmp = clean_stale_tmp(flow_dir)
    if n_tmp:
        print(f"[init] removed {n_tmp} stale temporary files")

    datasets = [s for s in a.datasets.split(",") if s.strip()] or None
    idx = read_index(a.cache_dir, datasets)
    idx = idx[a.start::a.stride] if a.stride > 1 else idx
    n_total = len(idx)
    todo = [(i, ds) for i, ds in idx if not os.path.isfile(FC.flow_path(a.cache_dir, i))]
    n_skipped = n_total - len(todo)
    if a.limit > 0:
        todo = todo[:a.limit]
    print(f"[init] {n_total} samples, {n_skipped} already cached, {len(todo)} to do")
    if not todo:
        print("[done] nothing to do")
        return

    raft = load_raft(device)
    pool = ThreadPoolExecutor(a.workers)

    def submit(bs):
        return bs, [pool.submit(read_frame_256, frame_path(a.cache_dir, i, "a0")) for i, _ in bs], \
                   [pool.submit(read_frame_256, frame_path(a.cache_dir, i, "a1")) for i, _ in bs]

    batches = [todo[b:b + a.batch] for b in range(0, len(todo), a.batch)]
    n_done = n_fail = 0
    t_start = time.time()
    next_report = a.report_every
    cur = submit(batches[0])
    try:
        for bi in range(len(batches)):
            nxt = submit(batches[bi + 1]) if bi + 1 < len(batches) else None
            bs, fa, fb = cur
            f0 = [f.result() for f in fa]
            f1 = [f.result() for f in fb]
            keep = [k for k in range(len(bs)) if f0[k] is not None and f1[k] is not None]
            n_fail += len(bs) - len(keep)
            for k in range(len(bs)):
                if f0[k] is None or f1[k] is None:
                    print(f"[warn] unreadable frame for i={bs[k][0]} ({bs[k][1]}); skipped, retried on a re-run")
            if keep:
                flow = raft_flow_128(raft, np.stack([f0[k] for k in keep]), np.stack([f1[k] for k in keep]), device,
                                     a.flow_updates, dtype)
                for q_i, k in enumerate(keep):
                    i = bs[k][0]
                    q, scale = quantise(flow[q_i])
                    _, mag_fg, mag_bg = FC.fg_stats(FC.dequantise(q, scale))
                    write_npz_atomic(FC.flow_path(a.cache_dir, i), q, scale, mag_fg, mag_bg)
                    n_done += 1
            if n_done + n_fail >= next_report:
                el = time.time() - t_start
                rate = (n_done + n_fail) / max(el, 1e-6)
                print(f"[{n_done + n_fail}/{len(todo)}] {el:.0f}s  {rate:.1f} samples/s  "
                      f"ETA {(len(todo) - n_done - n_fail) / max(rate, 1e-9) / 60:.1f} min  fail {n_fail}", flush=True)
                next_report += a.report_every
            cur = nxt
    except KeyboardInterrupt:
        print("\n[interrupt] finished files are on disk; re-run to resume.")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    print(f"[done] written {n_done}  failed {n_fail}  already cached {n_skipped}  wall {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()

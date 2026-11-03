"""On-disk format of the per-sample optical-flow cache and the foreground rule derived from it.

The flow cache is written by precompute_flow_cache.py and read by the training data loader (cache_dataset.py),
which uses it to build the background-replaced positive of each training pair.

    <cache_dir>/flow128/<i//1000:03d>/<i:06d>.npz
        flow    int8    [2, 128, 128]  RAFT-large flow a0 -> a1, computed at 256x256 (frames squashed to a square,
                                       aspect not preserved), average-pooled 2x to 128x128, in pixels of the 256 grid,
                                       quantised as round(flow / scale). Channel 0 = dx (columns), channel 1 = dy (rows).
        scale   float32 scalar         max(|flow|) / 127, so flow_px = flow.astype(float32) * scale
        mag_fg  float32 scalar         mean flow magnitude inside the foreground mask (256-grid pixels)
        mag_bg  float32 scalar         mean flow magnitude outside the foreground mask (camera-motion indicator)

Foreground rule (derived from the flow at load time, never stored): magnitude > max(1.0 px, 0.35 * 99th percentile of
the magnitude), then binary dilation by 5 cells on the 128 grid (4-connected cross, scipy's default structuring element).
"""
from __future__ import annotations

import os
import zipfile

import numpy as np

try:  # optional dependency; the fallback below gives the same result
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None

FLOW_SUBDIR = "flow128"
FLOW_RES = 128          # stored grid
RAFT_RES = 256          # grid the flow is expressed in (pixels)
FG_MIN_PX = 1.0         # absolute magnitude floor of the threshold (256-grid px)
FG_FRAC = 0.35          # fraction of the 99th percentile
FG_PCT = 99.0
FG_DILATE = 5           # dilation iterations (cells of the 128 grid)


def flow_path(cache_dir: str, i: int) -> str:
    return os.path.join(cache_dir, FLOW_SUBDIR, f"{i // 1000:03d}", f"{i:06d}.npz")


def dequantise(flow_i8: np.ndarray, scale) -> np.ndarray:
    """int8 [2,128,128] + scale -> float32 flow in 256-grid pixels."""
    return flow_i8.astype(np.float32) * np.float32(scale)


def magnitude(flow_px: np.ndarray) -> np.ndarray:
    """[2,H,W] float -> [H,W] float32 magnitude."""
    return np.sqrt(flow_px[0].astype(np.float32) ** 2 + flow_px[1].astype(np.float32) ** 2)


def _dilate_cross(mask: np.ndarray, n: int) -> np.ndarray:
    """Binary dilation with a 4-connected cross, n iterations, zero border (same as scipy's default)."""
    m = mask.copy()
    for _ in range(n):
        d = m.copy()
        d[1:, :] |= m[:-1, :]
        d[:-1, :] |= m[1:, :]
        d[:, 1:] |= m[:, :-1]
        d[:, :-1] |= m[:, 1:]
        m = d
    return m


def dilate(mask: np.ndarray, n: int = FG_DILATE) -> np.ndarray:
    if n <= 0 or not mask.any():
        return mask.astype(bool, copy=True)
    if _ndi is not None:
        return _ndi.binary_dilation(mask, iterations=n)
    return _dilate_cross(mask.astype(bool), n)


def fg_threshold(mag: np.ndarray) -> float:
    return float(max(FG_MIN_PX, FG_FRAC * float(np.percentile(mag, FG_PCT))))


def fg_mask(flow_px: np.ndarray) -> np.ndarray:
    """Dilated foreground mask. flow_px: float [2,128,128] in 256-grid pixels -> bool [128,128]."""
    mag = magnitude(flow_px)
    core = mag > fg_threshold(mag)
    return dilate(core, FG_DILATE)


def core_mask(flow_px: np.ndarray) -> np.ndarray:
    """The foreground rule without the dilation: the moving part itself. bool [128,128]."""
    mag = magnitude(flow_px)
    return mag > fg_threshold(mag)


def fg_stats(flow_px: np.ndarray):
    """(mask, mag_fg, mag_bg) with the dilated mask; an empty side contributes 0.0."""
    mag = magnitude(flow_px)
    mask = fg_mask(flow_px)
    mag_fg = float(mag[mask].mean()) if mask.any() else 0.0
    inv = ~mask
    mag_bg = float(mag[inv].mean()) if inv.any() else 0.0
    return mask, mag_fg, mag_bg


def load_flow_raw(cache_dir: str, i: int):
    """-> dict(flow=int8[2,128,128], scale, mag_fg, mag_bg) or None if the file is absent or corrupt."""
    p = flow_path(cache_dir, i)
    if not os.path.isfile(p):
        return None
    try:
        with np.load(p) as z:
            return {"flow": z["flow"], "scale": np.float32(z["scale"]), "mag_fg": np.float32(z["mag_fg"]),
                    "mag_bg": np.float32(z["mag_bg"])}
    except (zipfile.BadZipFile, OSError, KeyError, ValueError, EOFError):
        return None


def load_flow(cache_dir: str, i: int):
    """-> (flow_px float32 [2,128,128] in 256-grid pixels, mag_fg, mag_bg) or None."""
    r = load_flow_raw(cache_dir, i)
    if r is None:
        return None
    return dequantise(r["flow"], r["scale"]), float(r["mag_fg"]), float(r["mag_bg"])

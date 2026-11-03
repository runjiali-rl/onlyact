"""Image operations used to build JustAct training views.

Two operations:
  - photo_jitter: seeded photometric jitter (brightness, contrast, saturation, hue). Both frames of a pair receive the
    same parameters, so the motion between them is unchanged.
  - background replacement with core shift: the moving part of a pair (the "core", from the pair's optical flow) is
    cut out, moved by one random integer offset (the same for both frames) and pasted onto a still frame of another
    clip. Inside the core the pixels are the original pixels moved by the offset, so the motion field is preserved
    exactly while the background and the absolute image position change.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

_FLOW_GRID_PX = 256.0   # the flow cache stores displacements in pixels of a 256x256 grid


def _pil_to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


def _np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)


# ---------------------------------------------------------------- photometric jitter

def _sample_jitter_params(seed: int, b: float, c: float, s: float, h: float) -> Tuple[float, float, float, float]:
    g = torch.Generator().manual_seed(seed)
    def u(lo, hi): return torch.empty(1).uniform_(lo, hi, generator=g).item()
    bf = 1.0 + u(-b, b)
    cf = 1.0 + u(-c, c)
    sf = 1.0 + u(-s, s)
    hf = u(-h, h)
    return bf, cf, sf, hf


def _apply_jitter(img: Image.Image, bf: float, cf: float, sf: float, hf: float) -> Image.Image:
    img = TF.adjust_brightness(img, max(bf, 0.05))
    img = TF.adjust_contrast(img, max(cf, 0.05))
    img = TF.adjust_saturation(img, max(sf, 0.05))
    img = TF.adjust_hue(img, max(min(hf, 0.5), -0.5))
    return img


def photo_jitter(img: Image.Image, seed: int, brightness: float = 0.3, contrast: float = 0.3,
                 saturation: float = 0.3, hue: float = 0.05) -> Image.Image:
    """Photometric jitter of one image; the parameters are a pure function of `seed`."""
    bf, cf, sf, hf = _sample_jitter_params(seed, brightness, contrast, saturation, hue)
    return _apply_jitter(img, bf, cf, sf, hf)


# ---------------------------------------------------------------- background replacement with core shift

def mask_cover_destination(mask128: np.ndarray, flow_px: np.ndarray) -> np.ndarray:
    """Union of the mask with the cells its pixels move to (each masked cell splatted forward by its own flow), so one
    mask covers the moving part in both frames."""
    m = mask128.astype(bool)
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return m
    scale = m.shape[0] / _FLOW_GRID_PX  # 128 / 256 = 0.5 cells per 256-grid pixel
    xd = np.clip(np.rint(xs + flow_px[0][ys, xs] * scale).astype(np.int64), 0, m.shape[1] - 1)
    yd = np.clip(np.rint(ys + flow_px[1][ys, xs] * scale).astype(np.int64), 0, m.shape[0] - 1)
    out = m.copy()
    out[yd, xd] = True
    return out


def mask_to_frame(mask128: np.ndarray, w: int, h: int, feather_px: float = 4.0) -> np.ndarray:
    """bool [128, 128] -> float32 [h, w] soft mask in [0, 1]: bilinear resize, then a Gaussian feather with
    sigma = feather_px / 2 (no feather for feather_px = 0)."""
    m = cv2.resize(mask128.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    if feather_px > 0:
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=float(feather_px) / 2.0)
    return np.clip(m, 0.0, 1.0)


def composite_alpha(mask128: np.ndarray, flow_px: np.ndarray, w: int, h: int, feather_px: float = 4.0) -> np.ndarray:
    """float32 [h, w] blending alpha of the protected region: 1 keeps the original pixel, 0 takes the background."""
    m = mask_cover_destination(mask128, flow_px)
    return mask_to_frame(m, w, h, feather_px)


def core_alpha_bbox(alpha: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Inclusive bounding box (x0, y0, x1, y1) of alpha > 0, or None when alpha is zero everywhere."""
    a = np.asarray(alpha)
    if a.ndim == 3:
        a = a[..., 0]
    rows = np.flatnonzero((a > 0).any(axis=1))
    cols = np.flatnonzero((a > 0).any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


def draw_core_offset(alpha: np.ndarray, rng, min_px: int = 0, max_tries: int = 8) -> Tuple[int, int]:
    """Draw an integer offset (dx, dy), uniform over the offsets that keep the bounding box of alpha > 0 inside the
    frame (so nothing of the core is clipped). An offset with max(|dx|, |dy|) < min_px is re-drawn, up to max_tries
    attempts; the last attempt is accepted as drawn. Each attempt makes exactly two calls on the random.Random `rng`:
    rng.randint for dx, then rng.randint for dy."""
    a = np.asarray(alpha)
    if a.ndim == 3:
        a = a[..., 0]
    h, w = a.shape[:2]
    bb = core_alpha_bbox(a)
    x0, y0, x1, y1 = bb if bb is not None else (0, 0, w - 1, h - 1)
    lo_x, hi_x = -int(x0), int(w - 1 - x1)
    lo_y, hi_y = -int(y0), int(h - 1 - y1)
    mp = int(min_px)
    dx = dy = 0
    for _ in range(max(1, int(max_tries))):
        dx = rng.randint(lo_x, hi_x)
        dy = rng.randint(lo_y, hi_y)
        if mp <= 0 or max(abs(dx), abs(dy)) >= mp:
            break
    return int(dx), int(dy)


def shift_zero_fill(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Move `arr` ([h, w] or [h, w, c]) by (dx, dy) whole pixels (+dx = right, +dy = down); what enters is zero."""
    a = np.asarray(arr)
    h, w = a.shape[:2]
    dx, dy = int(dx), int(dy)
    out = np.zeros_like(a)
    if abs(dx) >= w or abs(dy) >= h:
        return out
    ys, yd = (0, dy) if dy >= 0 else (-dy, 0)
    xs, xd = (0, dx) if dx >= 0 else (-dx, 0)
    hh, ww = h - abs(dy), w - abs(dx)
    out[yd:yd + hh, xd:xd + ww] = a[ys:ys + hh, xs:xs + ww]
    return out


def translate_core(frame0: Image.Image, frame1: Image.Image, bg: Image.Image, alpha: np.ndarray,
                   offset: Tuple[int, int]) -> Tuple[Image.Image, Image.Image]:
    """Composite c_i = bg * (1 - shift(alpha)) + shift(frame_i) * shift(alpha), with the same integer shift for both
    frames. `bg` is resized to the frame size (INTER_AREA when shrinking, INTER_LINEAR when enlarging); `alpha` is
    composite_alpha() of the pair at frame resolution. Returns PIL RGB images at frame0's size."""
    f0 = _pil_to_np(frame0)
    f1 = _pil_to_np(frame1)
    h, w = f0.shape[:2]
    if f1.shape[:2] != (h, w):
        f1 = cv2.resize(f1, (w, h), interpolation=cv2.INTER_LINEAR)
    b = _pil_to_np(bg)
    if b.shape[:2] != (h, w):
        shrink = b.shape[0] > h or b.shape[1] > w
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    b = b.astype(np.float32)
    a = np.asarray(alpha, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    if a.shape != (h, w):
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    dx, dy = int(offset[0]), int(offset[1])
    a = np.clip(shift_zero_fill(a, dx, dy), 0.0, 1.0)[..., None]
    s0 = shift_zero_fill(f0, dx, dy).astype(np.float32)
    s1 = shift_zero_fill(f1, dx, dy).astype(np.float32)
    c0 = np.clip(a * s0 + (1.0 - a) * b + 0.5, 0, 255).astype(np.uint8)
    c1 = np.clip(a * s1 + (1.0 - a) * b + 0.5, 0, 255).astype(np.uint8)
    return _np_to_pil(c0), _np_to_pil(c1)

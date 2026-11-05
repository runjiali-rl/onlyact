"""Training views of the OnlyAct encoder, read from the pre-drawn sample cache.

Each cached sample (build_sample_cache.py) holds four frames of one clip: the anchor pair (a0, a1), 1-3 frames apart,
and the hard-negative pair (h0, h1) from a disjoint window of the same clip. Its optical flow a0 -> a1 is in the flow
cache (precompute_flow_cache.py). One training sample is three pairs:

    anchor    (a0, a1) with photometric jitter
    positive  (a0, a1) with its moving core shifted to a random position and pasted onto the first frame of a sample
              with another scene id of the same dataset, then photometric jitter. When the pair cannot be composited
              (flow file missing, camera motion above the guard, nothing moving, or more than 90 % of the frame moving)
              the positive is the anchor pair with independent photometric jitter.
    hard neg  (h0, h1) with photometric jitter

All randomness of a sample comes from a random.Random seeded by (seed, cached sample id, pass over the cache), and the
photometric jitter of each pair from a seed drawn from it, so a sample is a pure function of its index.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

from .augment import composite_alpha, draw_core_offset, photo_jitter, translate_core
from .flow_cache import FLOW_SUBDIR, core_mask, load_flow


@dataclass
class TrainSample:
    anchor: Tuple[Image.Image, Image.Image]
    positive: Tuple[Image.Image, Image.Image]
    hard_neg: Tuple[Image.Image, Image.Image]
    task_id: str
    scene_id: str
    dataset_id: str
    # outcome of the background replacement: bg_ok | bg_missing | bg_ego | bg_nofg | bg_allfg
    synth_status: str = ""


def collate(samples: List[TrainSample]) -> Dict[str, Any]:
    """Batch layout: items = [anchor_1..N | positive_1..N | hard_neg_1..N], each item {"image": [frame0, frame1]}."""
    n = len(samples)
    items = ([{"image": list(s.anchor)} for s in samples]
             + [{"image": list(s.positive)} for s in samples]
             + [{"image": list(s.hard_neg)} for s in samples])
    out: Dict[str, Any] = {
        "items": items,
        "n": n,
        "task_ids": [s.task_id for s in samples],
        "scene_ids": [s.scene_id for s in samples],
        "dataset_ids": [s.dataset_id for s in samples],
    }
    if any(s.synth_status for s in samples):
        out["synth_status"] = [s.synth_status for s in samples]
    return out


class CachedPairDataset(Dataset):
    """Virtual length = n_samples x epoch_multiplier; index -> (sample, pass), so every pass over the cache draws new
    augmentation seeds."""

    def __init__(self, cache_dir: str, split: str = "train", val_frac: float = 0.02, seed: int = 0,
                 epoch_multiplier: int = 1, bg_max_motion_px: float = 0.5, feather_px: float = 0.0,
                 core_shift_min_px: int = 16, flow_cache_dir: str = ""):
        self.cache_dir = cache_dir
        self.records: List[Dict[str, Any]] = []
        with open(os.path.join(cache_dir, "index.jsonl")) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        self.records.sort(key=lambda r: r["i"])
        # Deterministic split by a seeded shuffle of the sample order.
        order = list(range(len(self.records)))
        random.Random(seed).shuffle(order)
        n_val = int(len(order) * val_frac)
        keep = set(order[:n_val]) if split == "val" else set(order[n_val:])
        self.records = [r for j, r in enumerate(self.records) if j in keep]
        self.base_seed = int(seed)
        self.epoch_multiplier = max(1, int(epoch_multiplier))
        self._virtual_len = len(self.records) * self.epoch_multiplier
        # Background donors: the records of each dataset, in sample order.
        self._by_dataset: Dict[str, list] = {}
        for r_ in self.records:
            self._by_dataset.setdefault(str(r_.get("dataset", "")), []).append(r_)
        self.bg_max_motion_px = float(bg_max_motion_px)
        self.feather_px = float(feather_px)
        self.core_shift_min_px = int(core_shift_min_px)
        self.flow_cache_dir = flow_cache_dir or cache_dir
        fdir = os.path.join(self.flow_cache_dir, FLOW_SUBDIR)
        if not os.path.isdir(fdir):
            print(f"[data] no flow cache at {fdir}: every positive falls back to a photometric copy of the anchor")
        print(f"[data] {split}: {len(self.records)} cached samples x {self.epoch_multiplier} passes from {cache_dir}")

    def __len__(self) -> int:
        return self._virtual_len

    def _frame(self, rec: Dict[str, Any], key: str) -> Image.Image:
        i = int(rec["i"])
        p = os.path.join(self.cache_dir, "frames", f"{i // 1000:03d}", f"{i:06d}_{key}.jpg")
        return Image.open(p).convert("RGB")

    @staticmethod
    def _scene_key(rec: Dict[str, Any]) -> str:
        return f'{rec.get("dataset", "")}/{rec.get("scene_id", "")}'

    def _background(self, rec: Dict[str, Any], rng: random.Random) -> Image.Image:
        """First frame of a random sample of the same dataset with a different scene id (8 tries, then any sample of
        the dataset)."""
        pool = self._by_dataset.get(str(rec.get("dataset", ""))) or self.records
        me = self._scene_key(rec)
        for _ in range(8):
            other = pool[rng.randrange(len(pool))]
            if self._scene_key(other) != me:
                return self._frame(other, "a0")
        return self._frame(pool[rng.randrange(len(pool))], "a0")

    def _replace_background(self, p0: Image.Image, p1: Image.Image, rec: Dict[str, Any], rng: random.Random):
        """-> (c0, c1, status). c0 / c1 are None unless status == 'ok'. The rng is used only on 'ok'."""
        fl = load_flow(self.flow_cache_dir, int(rec["i"]))
        if fl is None:
            return None, None, "missing"
        flow_px, _mag_fg, mag_bg = fl
        if float(mag_bg) > self.bg_max_motion_px:
            return None, None, "ego"                      # the camera moves: the background carries the motion
        m = core_mask(flow_px)
        frac = float(m.mean())
        if frac <= 0.0:
            return None, None, "nofg"                     # nothing moves
        if frac > 0.9:
            return None, None, "allfg"                    # no background left to replace
        bg = self._background(rec, rng)
        w, h = p0.size
        alpha = composite_alpha(m, flow_px, w, h, self.feather_px)
        off = draw_core_offset(alpha, rng, min_px=self.core_shift_min_px)
        c0, c1 = translate_core(p0, p1, bg, alpha, off)
        return c0, c1, "ok"

    def __getitem__(self, idx: int) -> TrainSample:
        last_exc: Optional[Exception] = None
        for attempt in range(8):
            try:
                return self._build(idx + attempt * 104729)
            except Exception as e:
                last_exc = e
                if attempt == 0:
                    print(f"[data] skipping idx={idx} due to {type(e).__name__}: {e}")
        raise RuntimeError(f"retry exhausted at idx={idx}: {last_exc}")

    def _build(self, idx: int) -> TrainSample:
        n = len(self.records)
        rec = self.records[idx % n]
        epoch = idx // n
        rng = random.Random((self.base_seed * 1_000_003) ^ (int(rec["i"]) * 7919) ^ (epoch * 104_729 + 1))
        f0, f1 = self._frame(rec, "a0"), self._frame(rec, "a1")
        hn0_raw, hn1_raw = self._frame(rec, "h0"), self._frame(rec, "h1")
        pf0, pf1 = f0, f1
        c0, c1, status = self._replace_background(f0, f1, rec, rng)
        if c0 is not None:
            pf0, pf1 = c0, c1
        seed_a, seed_p, seed_h = (rng.randint(0, 2**31 - 1) for _ in range(3))
        a0, a1 = photo_jitter(f0, seed_a), photo_jitter(f1, seed_a)
        p0, p1 = photo_jitter(pf0, seed_p), photo_jitter(pf1, seed_p)
        hn0, hn1 = photo_jitter(hn0_raw, seed_h), photo_jitter(hn1_raw, seed_h)
        return TrainSample(
            anchor=(a0, a1), positive=(p0, p1), hard_neg=(hn0, hn1),
            task_id=rec["task_id"], scene_id=rec.get("scene_id") or rec["task_id"],
            dataset_id=rec.get("dataset", "calvin"), synth_status="bg_" + status,
        )

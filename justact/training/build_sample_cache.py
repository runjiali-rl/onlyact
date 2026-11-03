"""Build the sample cache the JustAct encoder is trained from (168,000 samples drawn from the 9-dataset manifest).

Each sample is one clip of the manifest and two windows of it: the anchor pair (a0, a1) = frames (a_lo, a_hi) with a
gap of gap_lo..gap_hi frames, and the hard-negative pair (h0, h1) = frames (h_lo, h_hi) from a window that does not
overlap the anchor's. The four frames are decoded once and stored as JPEG files.

Sampling (identical to the paper's cache):
  1. keep the manifest tasks with n_frames >= 2 * gap_hi + 4, shuffle them with random.Random(seed), hold out the first
     2 % (validation) and draw from the rest;
  2. draw n_samples tasks with replacement, numpy.random.default_rng(seed).choice with per-task probability
     weight(dataset) / n_tasks(dataset); the k-th draw of a task gets step = k;
  3. windows from random.Random((seed * 1_000_003) ^ (task_index * 7919) ^ step).
--samples rebuilds a given list of samples instead (task_id + windows, e.g. the samples.jsonl written by --only_draw),
looking the tasks up in the manifest by task_id.

Layout:
    <out_dir>/index.jsonl                   one record per sample {i, task_id, dataset, scene_id, a_lo, a_hi, h_lo, h_hi, size}
    <out_dir>/frames/<i//1000:03d>/<i:06d>_{a0,a1,h0,h1}.jpg
    <out_dir>/meta.json

    python -m justact.training.build_sample_cache --manifest <manifest.json> --data_root <data> --out_dir <cache> \
        --n_samples 168000 --gap_lo 1 --gap_hi 3 --seed 0 --workers 4 --max_short_side 292 \
        --dataset_weights "vpt=0.25,calvin=0.15,libero=0.05,droid=0.05,atari=0.05,agibot=0.10,egodex=0.10,spatialvid=0.15,sthv2=0.10"

Manifest: {"tasks": [{"task_id", "dataset", "scene_id" (or "ep_dir"), "target_frames": {"paths": [...]} or
{"mp4_path": ..., "n_frames": N}}]}; paths are relative to --data_root. Video frames are decoded with decord.
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import random
import time
from collections import Counter
from functools import lru_cache

import numpy as np
from PIL import Image

_TASKS = None   # tasks of the training split (shared with the forked workers)
_ARGS = None


def _parse_weights(s: str) -> dict:
    out = {}
    for part in s.split(","):
        if not part.strip():
            continue
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out


def _dataset_of(task) -> str:
    return task.get("_union_dataset") or task.get("dataset", "calvin")


def _scene_of(task) -> str:
    return task.get("scene_id") or task.get("ep_dir") or task["task_id"]


def _frame_source(task):
    """-> ("images", [paths]) or ("video", path, n_frames)."""
    tf = task["target_frames"]
    paths = tf.get("paths") or tf.get("abs_paths")
    if paths:
        return ("images", paths)
    return ("video", tf["mp4_path"], int(tf["n_frames"]))


def _n_frames(task) -> int:
    src = _frame_source(task)
    return len(src[1]) if src[0] == "images" else src[2]


def pair_window(n_frames: int, gap_lo: int, gap_hi: int, rng: random.Random):
    k = rng.randint(gap_lo, gap_hi)
    if n_frames - k - 1 <= 0:
        k = max(1, min(gap_hi, n_frames - 2))
    t = rng.randint(0, n_frames - k - 1)
    return t, t + k


def disjoint_window(anchor_lo: int, anchor_hi: int, n_frames: int, gap_lo: int, gap_hi: int, rng: random.Random):
    """A (t', t'+k') window that does not overlap [anchor_lo, anchor_hi] (16 tries, then the far end of the clip)."""
    for _ in range(16):
        k = rng.randint(gap_lo, gap_hi)
        if n_frames - k - 1 <= 0:
            continue
        t = rng.randint(0, n_frames - k - 1)
        if t + k < anchor_lo or t > anchor_hi:
            return t, t + k
    k = gap_lo
    if anchor_lo >= n_frames // 2:
        t = 0
    else:
        t = max(0, n_frames - k - 1)
    return t, min(n_frames - 1, t + k)


def training_tasks(manifest_tasks, gap_hi: int, seed: int, val_frac: float = 0.02):
    """Filter by length, shuffle with the seed and drop the validation share."""
    min_frames = 2 * gap_hi + 4
    tasks = [t for t in manifest_tasks if t["target_frames"]["n_frames"] >= min_frames]
    random.Random(seed).shuffle(tasks)
    n_val = int(len(tasks) * val_frac)
    return tasks[n_val:]


def draw_samples(tasks, n_samples: int, weights: dict, seed: int, gap_lo: int, gap_hi: int):
    """-> list of sample records (without frames) in sample order."""
    tags = [_dataset_of(t) for t in tasks]
    counts = Counter(tags)
    present = {k: v for k, v in weights.items() if k in counts and v > 0}
    w = np.array([present.get(tg, 0.0) / counts[tg] for tg in tags], dtype=np.float64)
    w /= w.sum()
    task_draw = np.random.default_rng(seed).choice(len(tasks), size=n_samples, replace=True, p=w)
    seen: Counter = Counter()
    out = []
    for i, ti in enumerate(task_draw.tolist()):
        step = seen[ti]
        seen[ti] += 1
        task = tasks[ti]
        n = _n_frames(task)
        rng = random.Random((seed * 1_000_003) ^ (ti * 7919) ^ step)
        a_lo, a_hi = pair_window(n, gap_lo, gap_hi, rng)
        h_lo, h_hi = disjoint_window(a_lo, a_hi, n, gap_lo, gap_hi, rng)
        out.append({"i": i, "task_index": ti, "task_id": task["task_id"], "dataset": tags[ti],
                    "scene_id": _scene_of(task), "a_lo": a_lo, "a_hi": a_hi, "h_lo": h_lo, "h_hi": h_hi})
    return out


@lru_cache(maxsize=4)
def _video_reader(path: str):
    import decord
    decord.bridge.set_bridge("native")
    return decord.VideoReader(path, num_threads=1)


def _load_frame(src, idx: int, data_root: str) -> Image.Image:
    if src[0] == "images":
        return Image.open(os.path.join(data_root, src[1][idx])).convert("RGB")
    vr = _video_reader(os.path.join(data_root, src[1]))
    i = min(max(int(idx), 0), len(vr) - 1)
    return Image.fromarray(vr[i].asnumpy(), "RGB")


def _build_one(job):
    rec, task_key = job
    task = _TASKS[task_key]
    src = _frame_source(task)
    i = rec["i"]
    sub = os.path.join(_ARGS["out_dir"], "frames", f"{i // 1000:03d}")
    os.makedirs(sub, exist_ok=True)
    try:
        frames = {k: _load_frame(src, rec[w], _ARGS["data_root"])
                  for k, w in (("a0", "a_lo"), ("a1", "a_hi"), ("h0", "h_lo"), ("h1", "h_hi"))}
    except Exception as e:  # unreadable file: the sample is left out of the index
        return {"i": i, "error": f"{type(e).__name__}: {e}", "task_id": rec["task_id"]}
    mss = int(_ARGS["max_short_side"] or 0)
    if mss > 0:
        # Cap the short side at the V-JEPA 2 processor's resize target (292, then centre-crop 256); never upscale.
        for k, im in list(frames.items()):
            w, h = im.size
            short = min(w, h)
            if short > mss:
                r = mss / short
                frames[k] = im.resize((max(1, round(w * r)), max(1, round(h * r))), Image.BILINEAR, reducing_gap=3.0)
    for k, im in frames.items():
        im.save(os.path.join(sub, f"{i:06d}_{k}.jpg"), quality=_ARGS["jpeg_quality"], subsampling=0)
    out = {key: rec[key] for key in ("i", "task_id", "dataset", "scene_id", "a_lo", "a_hi", "h_lo", "h_hi")}
    out["size"] = list(frames["a0"].size)
    return out


def _sort_key(task):
    src = _frame_source(task)
    return src[1] if src[0] == "video" else src[1][0]


def _read_jsonl(path: str):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    global _TASKS, _ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data_root", default="", help="root of the relative frame / video paths of the manifest")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=168_000)
    ap.add_argument("--gap_lo", type=int, default=1)
    ap.add_argument("--gap_hi", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset_weights", type=str, default="")
    ap.add_argument("--samples", default="", help="rebuild exactly these samples (jsonl or jsonl.gz) instead of drawing")
    ap.add_argument("--only_draw", action="store_true", help="write <out_dir>/samples.jsonl without decoding frames")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--chunksize", type=int, default=16)
    ap.add_argument("--max_short_side", type=int, default=292, help="cap the stored short side (0 = keep native)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    _ARGS = vars(args)

    t0 = time.time()
    with open(args.manifest) as f:
        manifest_tasks = json.load(f)["tasks"]
    if args.samples:
        samples = _read_jsonl(args.samples)
        by_id = {t["task_id"]: t for t in manifest_tasks}
        missing = sorted({s["task_id"] for s in samples if s["task_id"] not in by_id})
        if missing:
            raise SystemExit(f"{len(missing)} task ids of {args.samples} are not in the manifest, e.g. {missing[:3]}")
        _TASKS = by_id
        jobs = [(s, s["task_id"]) for s in samples]
        print(f"[cache] {len(samples)} listed samples; manifest read in {time.time() - t0:.0f}s", flush=True)
    else:
        if not args.dataset_weights:
            raise SystemExit("--dataset_weights is required when drawing samples")
        tasks = training_tasks(manifest_tasks, args.gap_hi, args.seed)
        samples = draw_samples(tasks, args.n_samples, _parse_weights(args.dataset_weights), args.seed,
                               args.gap_lo, args.gap_hi)
        _TASKS = tasks
        jobs = [(s, s["task_index"]) for s in samples]
        print(f"[cache] {len(tasks)} training tasks; drew {len(samples)} samples, per dataset "
              f"{dict(Counter(s['dataset'] for s in samples))}; {time.time() - t0:.0f}s", flush=True)
    del manifest_tasks

    if args.only_draw:
        keys = ("i", "task_id", "dataset", "scene_id", "a_lo", "a_hi", "h_lo", "h_hi")
        with open(os.path.join(args.out_dir, "samples.jsonl"), "w") as f:
            for s in samples:
                f.write(json.dumps({k: s[k] for k in keys}) + "\n")
        print(f"[cache] wrote {os.path.join(args.out_dir, 'samples.jsonl')}")
        return

    index_path = os.path.join(args.out_dir, "index.jsonl")
    done = set()
    if os.path.isfile(index_path):   # resume
        for line in open(index_path):
            try:
                done.add(json.loads(line)["i"])
            except Exception:
                pass
        print(f"[cache] resuming: {len(done)} samples already in the index", flush=True)
    jobs = [j for j in jobs if j[0]["i"] not in done]
    jobs.sort(key=lambda j: _sort_key(_TASKS[j[1]]))   # path-sorted order: near-sequential reads

    n_err = 0
    t1 = time.time()
    ctx = mp.get_context("fork")   # the workers share the task list with the parent
    with open(index_path, "a") as f, ctx.Pool(args.workers) as pool:
        for k, rec in enumerate(pool.imap_unordered(_build_one, jobs, chunksize=args.chunksize), 1):
            if "error" in rec:
                n_err += 1
                if n_err <= 20:
                    print(f"[cache] skip i={rec['i']} {rec['task_id']}: {rec['error']}", flush=True)
                continue
            f.write(json.dumps(rec) + "\n")
            if k % 2000 == 0:
                f.flush()
                rate = k / (time.time() - t1)
                print(f"[cache] {k}/{len(jobs)}  {rate:.1f} samples/s  ETA {(len(jobs) - k) / max(rate, 1e-6) / 60:.0f} min  "
                      f"errors={n_err}", flush=True)
    skip = ("manifest", "data_root", "out_dir", "samples") + (("n_samples", "dataset_weights") if args.samples else ())
    meta = {"args": {k: v for k, v in _ARGS.items() if k not in skip},
            "n_samples": len(samples), "n_errors": n_err, "built_s": time.time() - t0}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[cache] done {meta}", flush=True)


if __name__ == "__main__":
    main()

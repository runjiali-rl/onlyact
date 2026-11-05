r"""Per-pair latent actions from a retrained AdaWorld or DreamDojo latent action model.

Writes the AdaWorld-LAM (retrained) and DreamDojo-LAM (retrained) embeddings read by Tables 1-4:
  <out_dir>/<task_id>.npy   float16, shape (n_frames - gap, 32)
with --out_dir = <emb_root>/<representation>/<dataset>, representation adaworld_lam_retrained or
dreamdojo_lam_retrained.
Row j is the latent action of the frame pair (target_frames[j], target_frames[j + gap]): the posterior mean
z_mu of the model's encoder (deterministic, not normalised), computed in float32.

Preprocessing (the whole frame is resized, without the training crop):
  adaworld   PIL RGB -> 256 x 256 bicubic (skipped when already 256 x 256) -> [0, 1]
  dreamdojo  PIL RGB -> 320 x 240 (width x height) Lanczos -> [0, 1]
Each pair is one 2-frame clip (B, 2, H, W, 3); --batch_size pairs per forward pass (16, as for the paper).
An unreadable frame is replaced by the first readable frame of the clip. Existing outputs are skipped.

Usage (select the GPU with CUDA_VISIBLE_DEVICES; the upstream network code uses the current CUDA device):
  python baselines/extract.py --arch adaworld --manifest data/probe/libero.json --data_root <frames root> \
      --ckpt checkpoints/adaworld_lam_retrained.safetensors --out_dir <emb_root>/adaworld_lam_retrained/libero
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

import lam_models

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from onlyact.data import load_manifest, resolve_tasks  # noqa: E402

ADAWORLD_RES = 256
DREAMDOJO_H, DREAMDOJO_W = 240, 320


def load_frame_adaworld(path: str) -> np.ndarray | None:
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[extract] unreadable frame {path}: {e}", flush=True)
        return None
    if img.size != (ADAWORLD_RES, ADAWORLD_RES):
        img = img.resize((ADAWORLD_RES, ADAWORLD_RES), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_frame_dreamdojo(path: str) -> np.ndarray | None:
    try:
        im = Image.open(path).convert("RGB").resize((DREAMDOJO_W, DREAMDOJO_H), Image.LANCZOS)
    except Exception as e:
        print(f"[extract] unreadable frame {path}: {e}", flush=True)
        return None
    return np.array(im).astype(np.float32) / 255.0


LOADERS = {"adaworld": load_frame_adaworld, "dreamdojo": load_frame_dreamdojo}


def read_tasks(path: str, data_root: str | None):
    """Tasks of a clip list (.json, or gzip-compressed .json.gz), in file order, with abs_paths resolved."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            manifest = json.load(f)
        return resolve_tasks(manifest["tasks"] if isinstance(manifest, dict) else manifest, data_root)
    return load_manifest(path, data_root)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True, choices=sorted(lam_models.ARCHS))
    ap.add_argument("--manifest", required=True, help="clip list (.json or .json.gz), e.g. data/probe/libero.json")
    ap.add_argument("--data_root", default=None, help="root that the clip list's frame paths are relative to")
    ap.add_argument("--out_dir", required=True, help="<emb_root>/<representation>/<dataset>")
    ap.add_argument("--ckpt", default=None, help="default: checkpoints/<arch>_lam_retrained.safetensors in the repository root")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (0 = all)")
    ap.add_argument("--workers", type=int, default=6, help="frame-reading threads")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    rep = lam_models.ARCHS[a.arch]["representation"]
    ckpt = a.ckpt or os.path.join(REPO_ROOT, "checkpoints", f"{rep}.safetensors")
    out_dir = a.out_dir
    os.makedirs(out_dir, exist_ok=True)
    latent_dim = lam_models.read_upstream_config(a.arch)["model_kwargs"]["latent_dim"]
    load_frame = LOADERS[a.arch]

    tasks = read_tasks(a.manifest, a.data_root)
    tasks = tasks[:a.limit] if a.limit else tasks
    todo = [t for t in tasks if not os.path.isfile(os.path.join(out_dir, f"{t['task_id']}.npy"))]
    print(f"[extract] {a.arch}: {len(tasks)} clips, {len(todo)} to do -> {out_dir} (gap {a.gap}, ckpt {ckpt})", flush=True)
    if not todo:
        return
    if torch.device(a.device).type == "cpu":
        lam_models.allow_cpu(a.arch)
    m = lam_models.build_extraction_model(a.arch, ckpt, a.device)
    pool = ThreadPoolExecutor(a.workers)
    t0 = time.time()
    npairs = 0
    nwritten = 0
    for n, t in enumerate(todo):
        paths = t["target_frames"]["abs_paths"]
        if len(paths) < 1 + a.gap:
            print(f"[extract] skip {t['task_id']}: {len(paths)} frames", flush=True)
            continue
        frames = list(pool.map(load_frame, paths))
        if any(f is None for f in frames):
            ref = next((f for f in frames if f is not None), None)
            if ref is None:
                print(f"[extract] skip {t['task_id']}: no readable frame", flush=True)
                continue
            frames = [f if f is not None else ref for f in frames]
        starts = list(range(0, len(frames) - a.gap))
        rows = []
        with torch.no_grad():
            for b in range(0, len(starts), a.batch_size):
                sel = starts[b:b + a.batch_size]
                x = torch.from_numpy(np.stack([np.stack([frames[i], frames[i + a.gap]]) for i in sel])).to(a.device)
                z = m.encode(x)["z_mu"].reshape(len(sel), latent_dim)
                if getattr(m, "mu_record", None) is not None:
                    m.mu_record = None  # AdaWorld's encode() appends every z_mu to this buffer in eval mode
                rows.append(z.float().cpu().numpy())
        Z = np.nan_to_num(np.concatenate(rows))
        assert Z.shape == (len(starts), latent_dim), Z.shape
        tmp = os.path.join(out_dir, f".{t['task_id']}.tmp.npy")
        np.save(tmp, Z.astype(np.float16))
        os.replace(tmp, os.path.join(out_dir, f"{t['task_id']}.npy"))
        npairs += len(Z)
        nwritten += 1
        if (n + 1) % 25 == 0 or n + 1 == len(todo):
            print(f"[extract] {n + 1}/{len(todo)} clips, {npairs} pairs, "
                  f"{npairs / max(time.time() - t0, 1e-6):.1f} pairs/s", flush=True)
    print(f"[extract] done: wrote {nwritten}/{len(todo)} clips in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

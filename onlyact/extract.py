"""Extract per-frame-pair embeddings with OnlyAct or frozen V-JEPA 2 (inputs of Tables 1, 2 and 4).

For every task of a clip list, writes <out_dir>/<task_id>.npy: float16, shape (n_frames - gap, D), one unit-norm row per
pair (frame i, frame i + gap). D = 640 for OnlyAct, 1024 for frozen V-JEPA 2. Existing files are skipped, so a run can be resumed.

    python -m onlyact.extract --manifest data/probe/libero.json --data_root $DATA/libero \
        --ckpt checkpoints/onlyact --out_dir $EMB/onlyact/libero
    python -m onlyact.extract --manifest data/probe/libero.json --data_root $DATA/libero \
        --frozen --out_dir $EMB/vjepa2_frozen/libero
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from tqdm import tqdm

from .data import load_manifest
from .encoder import BASE_MODEL, BASE_REVISION, OnlyActEncoder, encode_frame_sequence


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="clip list (JSON)")
    p.add_argument("--data_root", default=None, help="root that the clip list's relative frame paths start from")
    p.add_argument("--out_dir", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", help="OnlyAct checkpoint folder (adapter_config.json, adapter_model.safetensors, projection.safetensors)")
    g.add_argument("--frozen", action="store_true", help="frozen V-JEPA 2 baseline (no adapter, no projection)")
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--revision", default=BASE_REVISION)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gap", type=int, default=1, help="frame gap of each pair (1 = consecutive frames)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--merge_threads", type=int, default=6,
                   help="CPU threads used to merge the LoRA adapters (changes the bfloat16 rounding of the merged weights); "
                        "the paper used 6, and 4 for the nuScenes world-model clips")
    args = p.parse_args()

    tasks = load_manifest(args.manifest, args.data_root)
    encoder = OnlyActEncoder(None if args.frozen else args.ckpt, args.base_model, args.revision, args.device, args.merge_threads)
    os.makedirs(args.out_dir, exist_ok=True)
    todo = [t for t in tasks if not os.path.exists(os.path.join(args.out_dir, f"{t['task_id']}.npy"))]
    print(f"[extract] tasks={len(tasks)} done={len(tasks) - len(todo)} todo={len(todo)} dim={encoder.dim} out={args.out_dir}")

    n_failed = 0
    for t in tqdm(todo):
        paths = [q for q in t["target_frames"]["abs_paths"] if os.path.exists(q)]
        if len(paths) < args.gap + 1:
            print(f"[skip] {t['task_id']}: fewer than {args.gap + 1} frames on disk")
            continue
        try:
            emb = encode_frame_sequence(encoder, paths, args.batch_size, args.gap)
        except Exception as e:
            n_failed += 1
            print(f"[skip] {t['task_id']}: {e}")
            continue
        np.save(os.path.join(args.out_dir, f"{t['task_id']}.npy"), emb)
    print(f"[extract] finished, {n_failed} tasks failed")


if __name__ == "__main__":
    main()

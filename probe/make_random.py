"""Random control for Table 1: one standard-normal (n_pairs, 640) float16 array per clip.

Arrays are drawn in clip-list order from a single numpy default_rng(0) stream; n_pairs is the row count of the clip's
frozen V-JEPA 2 embedding file (so the random control has exactly the pairs of the other representations).

    python -m probe.make_random --clips data/probe/libero.json --ref_dir $EMB/vjepa2_frozen/libero --out_dir $EMB/random/libero
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--ref_dir", required=True, help="frozen V-JEPA 2 embeddings of the same clips (row counts)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dim", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tasks = json.load(open(args.clips))["tasks"]
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for t in tasks:
        ref = os.path.join(args.ref_dir, t["task_id"] + ".npy")
        n = int(np.load(ref, mmap_mode="r").shape[0]) if os.path.exists(ref) else max(1, t["target_frames"]["n_frames"] - 1)
        np.save(os.path.join(args.out_dir, t["task_id"] + ".npy"),
                rng.standard_normal((n, args.dim), dtype=np.float32).astype(np.float16))
    print(f"wrote {len(tasks)} random arrays to {args.out_dir}")


if __name__ == "__main__":
    main()

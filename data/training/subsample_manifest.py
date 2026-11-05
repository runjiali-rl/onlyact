"""Prepare training data of the OnlyAct encoder: cap the large video datasets.

Randomly subsamples AgiBot World, EgoDex, SpatialVID and Something-Something v2
to fixed caps (other datasets are kept whole), then shuffles the task list.
This yields the training manifest (480,360 tasks with the default caps).

Procedure (seed --seed, one random.Random generator):
  1. group tasks by dataset, in order of first appearance in --input,
  2. for each capped dataset with more tasks than its cap: shuffle, keep the first cap,
  3. concatenate the groups and shuffle the whole list.
The result depends on the task order of --input.

Usage:
    python subsample_manifest.py --input <DATA_ROOT>/union_9ds/manifest_union.json \
        --output <DATA_ROOT>/union_9ds/manifest_union_sub.json
"""

import argparse
import json
import os
import random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description="Subsample the large video datasets of a union manifest")
    ap.add_argument("--input", required=True, help="Union manifest")
    ap.add_argument("--output", required=True, help="Output manifest path")
    ap.add_argument("--cap_agibot", type=int, default=100000)
    ap.add_argument("--cap_egodex", type=int, default=100000)
    ap.add_argument("--cap_spatialvid", type=int, default=200000)
    ap.add_argument("--cap_sthv2", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    with open(args.input) as f:
        m = json.load(f)
    tasks = m["tasks"]
    print(f"[load] {len(tasks)} tasks")

    by_ds = defaultdict(list)
    for t in tasks:
        ds = t.get("_union_dataset") or t.get("dataset", "?")
        by_ds[ds].append(t)
    caps = {"agibot": args.cap_agibot, "egodex": args.cap_egodex,
            "spatialvid": args.cap_spatialvid, "sthv2": args.cap_sthv2}
    new_tasks = []
    counts = {}
    for ds, lst in by_ds.items():
        if ds in caps and len(lst) > caps[ds]:
            rng.shuffle(lst)
            lst = lst[:caps[ds]]
        counts[ds] = len(lst)
        new_tasks.extend(lst)
        print(f"  {ds:12s}: kept {len(lst):>7d} (orig {len(by_ds[ds]):>7d})")
    rng.shuffle(new_tasks)
    print(f"[total] {len(new_tasks)}")

    payload = {**m, "tasks": new_tasks, "dataset_counts": counts}
    with open(args.output, "w") as f:
        json.dump(payload, f)
    print(f"[write] {args.output}  ({os.path.getsize(args.output) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

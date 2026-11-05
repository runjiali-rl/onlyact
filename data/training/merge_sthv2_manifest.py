"""Prepare training data of the OnlyAct encoder: append Something-Something v2.

Concatenates the eight-dataset union manifest and the Something-Something v2
manifest (both written by build_union_manifest.py) into the nine-dataset union
manifest. Task order: all tasks of --union, then all tasks of --sthv2.

Usage:
    python merge_sthv2_manifest.py \
        --union <DATA_ROOT>/union_8ds/manifest_union.json \
        --sthv2 <DATA_ROOT>/union_9ds/manifest_sthv2_only.json \
        --out   <DATA_ROOT>/union_9ds/manifest_union.json
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Append the Something-Something v2 tasks to a union manifest")
    ap.add_argument("--union", required=True, help="Eight-dataset union manifest")
    ap.add_argument("--sthv2", required=True, help="Something-Something v2 manifest")
    ap.add_argument("--out", required=True, help="Output manifest path")
    args = ap.parse_args()

    with open(args.union) as f:
        m8 = json.load(f)
    print(f"[load] union: {len(m8['tasks'])} tasks, datasets: {m8.get('datasets')}")

    with open(args.sthv2) as f:
        ms = json.load(f)
    print(f"[load] sthv2: {len(ms['tasks'])} tasks")

    all_tasks = m8["tasks"] + ms["tasks"]
    payload = {
        "stage": "union",
        "datasets": list(m8.get("datasets", [])) + ["sthv2"],
        "dataset_counts": {**m8.get("dataset_counts", {}), "sthv2": len(ms["tasks"])},
        "tasks": all_tasks,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f)
    print(f"[write] {out}  total tasks = {len(all_tasks)}, size {out.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

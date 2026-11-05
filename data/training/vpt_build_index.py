"""Prepare training data of the OnlyAct encoder: VPT segment index.

Scans <data_root>/<subdir>/episodes/ for segments written by vpt_process.py
(marker file `_frames_ready`) and writes index.json, the VPT input of
build_union_manifest.py:
    {"episodes": [{"ep_dir": "<subdir>/episodes/<subversion>/<player_episode>/seg<K>",
                   "episode_id": "<subversion>/<player_episode>/seg<K>"}, ...],
     "stats": {"n_segments": N}}
ep_dir is relative to data_root. Segments are listed in sorted path order.

With --segment_ids (one episode_id per line, e.g. vpt_segment_ids.txt, the
10,050 segments used for training), only the listed segments are written and
every listed segment must exist.

Usage:
    python vpt_build_index.py --data_root <DATA_ROOT> --segment_ids vpt_segment_ids.txt
"""

import argparse
import json
import os
from pathlib import Path


def build_index(data_root: Path, subdir: str) -> dict:
    output_root = data_root / subdir
    ep_root = output_root / "episodes"
    entries = []
    if ep_root.exists():
        for marker in sorted(ep_root.rglob("_frames_ready")):
            seg_dir = marker.parent
            entries.append({
                "ep_dir": str(seg_dir.relative_to(data_root)),
                "episode_id": str(seg_dir.relative_to(ep_root)),
            })
    return {"episodes": entries, "stats": {"n_segments": len(entries)}}


def main():
    ap = argparse.ArgumentParser(description="Write the VPT segment index")
    ap.add_argument("--data_root", required=True,
                    help="Folder that holds all processed datasets")
    ap.add_argument("--subdir", default="vpt_minecraft_stride1",
                    help="VPT folder inside data_root")
    ap.add_argument("--segment_ids", default=None,
                    help="Optional text file of episode_ids to keep")
    ap.add_argument("--out", default=None,
                    help="Output path (default <data_root>/<subdir>/index.json)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    idx = build_index(data_root, args.subdir)
    print(f"found {idx['stats']['n_segments']} segments")

    if args.segment_ids:
        with open(args.segment_ids) as f:
            keep = [line.strip() for line in f if line.strip()]
        found = {e["episode_id"] for e in idx["episodes"]}
        missing = [s for s in keep if s not in found]
        if missing:
            raise SystemExit(f"{len(missing)} listed segments not found, e.g. {missing[:3]}")
        keep_set = set(keep)
        idx["episodes"] = [e for e in idx["episodes"] if e["episode_id"] in keep_set]
        idx["stats"]["n_segments"] = len(idx["episodes"])
        print(f"kept {len(idx['episodes'])} segments listed in {args.segment_ids}")

    out = args.out or os.path.join(args.data_root, args.subdir, "index.json")
    with open(out, "w") as f:
        json.dump(idx, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

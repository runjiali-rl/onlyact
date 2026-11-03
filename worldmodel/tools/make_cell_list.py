"""Write one Table 2 evaluation cell file (data/worldmodel/<cell>.json) from a manifest with absolute frame paths.

Used to produce the released cell files. Input: a manifest {"tasks": [...]} whose tasks carry target_frames.abs_paths,
and the ordered rollout list (a JSON list of task ids for a self cell, or a JSON list of {"source", "target"} for a
transfer cell). Output: the same tasks with frame paths made relative to --strip_prefix (the dataset root), the ordered
"pairs" (source == target for self cells), and a short description of how the clips were chosen. The rollout order is
copied unchanged; --check_results verifies it against results.jsonl files of earlier runs.

    python tools/make_cell_list.py --cell libero_self --dataset libero --mode self --order ids.json \
        --manifest manifest.json --strip_prefix /data/libero/frames --description "..." --out ../data/worldmodel/libero_self.json
"""
import argparse
import json
import os

KEEP_FRAME_FIELDS = ("frame_indices", "n_frames", "sha256")


def relative(path: str, prefix: str) -> str:
    prefix = prefix.rstrip("/") + "/"
    if not path.startswith(prefix):
        raise ValueError(f"{path} is not under {prefix}")
    return path[len(prefix):]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--dataset", required=True, help="dataset name used in the embedding layout <emb_root>/<rep>/<dataset>/")
    ap.add_argument("--mode", required=True, choices=["self", "transfer"])
    ap.add_argument("--order", required=True, help="self: JSON list of task ids; transfer: JSON list of {source, target}")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--strip_prefix", required=True, help="absolute dataset root removed from every frame path")
    ap.add_argument("--description", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check_results", nargs="*", default=[], help="results.jsonl files whose (source, target) order must match")
    a = ap.parse_args()

    with open(a.manifest) as f:
        m = json.load(f)
    by_id = {t["task_id"]: t for t in (m["tasks"] if isinstance(m, dict) else m)}
    with open(a.order) as f:
        order = json.load(f)
    pairs = [{"source": t, "target": t} for t in order] if a.mode == "self" else [{"source": p["source"], "target": p["target"]} for p in order]
    if a.mode == "self":
        assert all(isinstance(t, str) for t in order), "a self cell takes a list of task ids"

    ids = []
    for p in pairs:  # targets first in rollout order, then sources
        if p["target"] not in ids:
            ids.append(p["target"])
    for p in pairs:
        if p["source"] not in ids:
            ids.append(p["source"])
    tasks = []
    for tid in ids:
        t = by_id[tid]
        tf = t["target_frames"]
        entry = {"task_id": tid, "scene_id": t.get("scene_id")}
        if t.get("scene_key") is not None:
            entry["scene_key"] = t["scene_key"]
        frames = {"paths": [relative(p, a.strip_prefix) for p in tf["abs_paths"]]}
        frames.update({k: tf[k] for k in KEEP_FRAME_FIELDS if k in tf})
        if frames.get("n_frames", len(frames["paths"])) != len(frames["paths"]):
            raise ValueError(f"{tid}: n_frames does not match the number of paths")
        entry["target_frames"] = frames
        tasks.append(entry)

    for rp in a.check_results:
        with open(rp) as f:
            got = [(r["source"], r["target"]) for r in map(json.loads, f)]
        if got != [(p["source"], p["target"]) for p in pairs]:
            raise ValueError(f"rollout order differs from {rp}")

    cell = {"cell": a.cell, "dataset": a.dataset, "mode": a.mode, "description": a.description,
            "n_rollouts": len(pairs), "pairs": pairs, "tasks": tasks}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(cell, f, indent=1)
        f.write("\n")
    print(f"[make_cell_list] {a.cell}: {len(pairs)} rollouts, {len(tasks)} clips -> {a.out}"
          + (f" (order matches {len(a.check_results)} results files)" if a.check_results else ""))


if __name__ == "__main__":
    main()

"""Write the world-model training clip list (data/worldmodel/train_pool.json.gz) from a pool manifest with absolute paths.

The training pool has 19,946 clips of 25 consecutive frames from nine datasets (VPT 5,000, CALVIN 3,000, SpatialVID 3,000,
EgoDex 2,000, AgiBot 2,000, Something-Something v2 1,946, DROID 1,000, Atari 1,000, LIBERO 1,000). Every clip contains one
of the frame windows the action encoders were trained on, and no clip overlaps the Table 2 evaluation clips. Frame paths in
the output are relative to one pool data root:
  <dataset>/...                    frame-backed datasets (vpt, calvin, atari, droid, libero): the dataset's own frame files
  decoded/<dataset>/<task_id>/f_NN.jpg   video-backed datasets (spatialvid, agibot, egodex, sthv2): 25 frames decoded from
                                   "video" starting at frame "start" (tools/decode_pool_videos.py)
Task order is kept (the training sampler and compute_whitening.py depend on it).

    python tools/make_train_list.py --pool_manifest pool.json --out ../data/worldmodel/train_pool.json.gz \
        --frame_prefix vpt=/data/vpt/episodes ... --decoded_prefix /data/pool/frames \
        --video_manifest union.json --video_prefix spatialvid=/data/spatialvid ...
"""
import argparse
import gzip
import json


def strip(path, prefix):
    prefix = prefix.rstrip("/") + "/"
    if not path.startswith(prefix):
        raise ValueError(f"{path} is not under {prefix}")
    return path[len(prefix):]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool_manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame_prefix", nargs="+", required=True, metavar="DATASET=PREFIX",
                    help="absolute frame root of each frame-backed dataset")
    ap.add_argument("--decoded_prefix", required=True, help="absolute root of the decoded frames of video-backed datasets")
    ap.add_argument("--video_manifest", required=True, help="manifest that maps source_task_id to its source video")
    ap.add_argument("--video_prefix", nargs="+", required=True, metavar="DATASET=PREFIX", help="absolute video root per dataset")
    ap.add_argument("--dataset_name", default="worldmodel_pool", help="name used in <emb_root>/<representation>/<dataset_name>/")
    a = ap.parse_args()
    frame_prefix = dict(x.split("=", 1) for x in a.frame_prefix)
    video_prefix = dict(x.split("=", 1) for x in a.video_prefix)

    pool = json.load(open(a.pool_manifest))
    need = {t["source_task_id"] for t in pool["tasks"] if t["dataset"] in video_prefix}
    vm = json.load(open(a.video_manifest))
    videos = {t["task_id"]: (t.get("target_frames") or {}).get("mp4_path") for t in (vm["tasks"] if isinstance(vm, dict) else vm)
              if t["task_id"] in need}
    del vm
    tasks = []
    for t in pool["tasks"]:
        ds, tf = t["dataset"], t["target_frames"]
        scene = t["scene_id"]
        if scene.startswith("/"):  # some datasets use the episode folder as scene id: store it relative as well
            scene = f"{ds}/{strip(scene, frame_prefix[ds])}"
        e = {"task_id": t["task_id"], "dataset": ds, "scene_id": scene, "source_task_id": t["source_task_id"], "start": t["start"]}
        if ds in frame_prefix:
            paths = [f"{ds}/{strip(p, frame_prefix[ds])}" for p in tf["abs_paths"]]
        else:
            paths = [f"decoded/{ds}/{strip(p, a.decoded_prefix + '/' + ds)}" for p in tf["abs_paths"]]
            assert all(p == f"decoded/{ds}/{t['task_id']}/f_{k:02d}.jpg" for k, p in enumerate(paths)), t["task_id"]
            e["video"] = f"{ds}/{strip(videos[t['source_task_id']], video_prefix[ds])}"
            assert tf["frame_indices"] == list(range(t["start"], t["start"] + len(paths))), t["task_id"]
        e["target_frames"] = {"paths": paths, "frame_indices": tf["frame_indices"], "n_frames": tf["n_frames"]}
        assert tf["n_frames"] == len(paths) == 25
        tasks.append(e)
    text = json.dumps(tasks)
    for bad in ('"/', "/home", "/scratch", "/tmp"):
        if bad in text:
            raise ValueError(f"absolute path left in the output ({bad})")
    out = {"dataset": a.dataset_name, "n_tasks": len(tasks),
           "description": "World-model training pool: 25-frame clips (gap 1) that each contain one encoder training window; "
                          "paths are relative to the pool data root (see tools/make_train_list.py).",
           "tasks": tasks}
    with open(a.out, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
        gz.write(json.dumps(out, separators=(",", ":")).encode())  # fixed header: byte-identical output on every run
    print(f"[make_train_list] {len(tasks)} clips -> {a.out}")


if __name__ == "__main__":
    main()

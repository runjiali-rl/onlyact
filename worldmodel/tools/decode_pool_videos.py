"""Decode the frames of the video-backed clips of the world-model training pool (SpatialVID, AgiBot, EgoDex,
Something-Something v2) into <data_root>/decoded/<dataset>/<task_id>/f_NN.jpg, as listed in data/worldmodel/train_pool.json.gz.

Each clip is the 25 consecutive frames starting at "start" of its source "video" (path relative to --video_root
<dataset>=PATH), decoded with decord at native size and stored as JPEG quality 95 without chroma subsampling, the format
the pool was trained on. Frame-backed datasets (VPT, CALVIN, Atari, DROID, LIBERO) are read in place: link their frame
roots as <data_root>/<dataset>. Existing complete clips are skipped, so the script can be resumed.

    python tools/decode_pool_videos.py --data_root $POOL --workers 6 \
        --video_root spatialvid=/data/spatialvid agibot=/data/agibot egodex=/data/egodex sthv2=/data/sthv2/videos
"""
import argparse
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from clip_lists import read_clip_list  # noqa: E402


def decode(job):
    video, start, paths = job
    if all(os.path.exists(p) for p in paths):
        return None
    import decord
    from PIL import Image
    os.makedirs(os.path.dirname(paths[0]), exist_ok=True)
    vr = decord.VideoReader(video, num_threads=1)
    if len(vr) < start + len(paths):
        return f"{video}: {len(vr)} frames < {start + len(paths)}"
    arr = vr.get_batch(list(range(start, start + len(paths)))).asnumpy()
    for k, p in enumerate(paths):
        Image.fromarray(arr[k]).save(p + ".tmp.jpg", format="JPEG", quality=95, subsampling=0)
        os.replace(p + ".tmp.jpg", p)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_list", default=os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "worldmodel", "train_pool.json.gz"))
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--video_root", nargs="+", required=True, metavar="DATASET=PATH")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    roots = dict(x.split("=", 1) for x in a.video_root)
    jobs = []
    for t in read_clip_list(a.train_list)["tasks"]:
        if "video" in t:
            ds = t["dataset"]
            jobs.append((os.path.join(roots[ds], t["video"][len(ds) + 1:]), t["start"],
                         [os.path.join(a.data_root, p) for p in t["target_frames"]["paths"]]))
    print(f"[decode] {len(jobs)} video-backed clips")
    with Pool(a.workers) as pool:
        errors = [e for e in pool.imap_unordered(decode, jobs, chunksize=4) if e]
    print(f"[decode] done, {len(errors)} errors", *errors[:10], sep="\n")


if __name__ == "__main__":
    main()

"""UT Austin MUTEX probe clips (Table 1, Table 4).

Input (--raw): a local copy of the Hugging Face dataset IPEC-COMMUNITY/utaustin_mutex_lerobot (LeRobot v2.1 export of
the Open X-Embodiment version of MUTEX; 1,500 episodes, 50 tasks, 20 frames per second, AV1 video). Only the files
named by the clips are needed (0.16 GB):
    python data/probe/prepare/utaustin_mutex.py --list_files > files.txt
    hf download IPEC-COMMUNITY/utaustin_mutex_lerobot $(cat files.txt) --repo-type dataset --revision <revision> \
        --local-dir <raw>
Each clip's `source` names the episode, its parquet file and its video (camera observation.images.image). The 200
clips are 4 episodes of each of the 50 tasks (seeded round robin over tasks); the scene of a clip is its task text.

Frames: the first n frames of the episode (n = the clip's frame count, at most 160), decoded in order with PyAV,
128x128, saved unchanged as PNG f_<k>.png; the 128x128 copies are the same images.
Action target `actions` (n, 7) float32: the first n rows of the `action` column (end-effector deltas x, y, z, roll,
pitch, yaw and gripper), ordered by frame index.
"""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

DATASET = "utaustin_mutex"
NATIVE_RES = 256


def main():
    if "--list_files" in sys.argv:
        tasks = json.load(open(os.path.join(common.REPO, "data", "probe", f"{DATASET}.json")))["tasks"]
        print("\n".join(f for t in tasks for f in (t["source"]["data"], t["source"]["video"])))
        return
    args = common.parser(DATASET, "local copy of IPEC-COMMUNITY/utaustin_mutex_lerobot").parse_args()
    out = common.Outputs(args, DATASET)
    import av  # the videos are AV1; decord cannot open them
    tasks = common.load_tasks(args)
    for t in tasks:
        s = t["source"]
        tf = t["target_frames"]
        n = tf["n_frames"]
        pairs = [(out.frame(rel), out.frame128(rel)) for rel in tf["paths"]]
        if out.todo(*[p for ab in pairs for p in ab]):
            container = av.open(os.path.join(args.raw, s["video"]))
            container.streams.video[0].thread_type = "NONE"
            k = 0
            for frame in container.decode(video=0):          # the k-th decoded frame is frame_index k
                if k >= n:
                    break
                im = Image.fromarray(frame.to_ndarray(format="rgb24"), "RGB")
                w, h = im.size
                if min(w, h) > NATIVE_RES:
                    r = NATIVE_RES / min(w, h)
                    im_n = im.resize((max(1, round(w * r)), max(1, round(h * r))), Image.BILINEAR, reducing_gap=3.0)
                else:
                    im_n = im
                common.save_pil(im_n, pairs[k][0])
                common.save_pil(im.resize((128, 128), Image.BILINEAR), pairs[k][1])
                k += 1
            container.close()
            assert k == n, (t["task_id"], k, n)
        a = out.action(t["task_id"])
        if out.todo(a):
            d = pq.read_table(os.path.join(args.raw, s["data"])).to_pydict()
            order = np.argsort(np.asarray(d["frame_index"]))
            A = np.asarray([d["action"][j] for j in order], dtype=np.float32)
            common.save_npy(a, A[:n])
    common.log_done(DATASET, len(tasks), out)


if __name__ == "__main__":
    main()

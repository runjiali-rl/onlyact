"""MetaWorld MT50 probe clips (Table 1, Table 4).

Input (--raw): a local copy of the Hugging Face dataset lerobot/metaworld_mt50 (LeRobot v3.0; frames are stored as
480x480 PNG bytes inside data/chunk-000/file-<k>.parquet). Only the parquet files named by the clips are needed
(127 files, 12 GB):
    python data/probe/prepare/metaworld.py --list_files > files.txt
    hf download lerobot/metaworld_mt50 $(cat files.txt) --repo-type dataset --revision <revision> --local-dir <raw>
Each clip's `source` names the parquet file and the episode index (10 episodes per task, 500 clips).

Frames: every step of the episode, 480x480 -> 256x256 (PIL bilinear, reducing_gap 3.0) as ep<episode>/frame_<k>.png.
128x128 copies: PIL bilinear from the 480x480 source.
Action target `actions_clip` (T, 4) float32: the `action` column (end-effector delta x, y, z and gripper) clipped to
MetaWorld's action range [-1, 1].
"""
import io
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

DATASET = "metaworld"
NATIVE_RES = 256
COLS = ["episode_index", "frame_index", "task_index", "action", "observation.image"]


def main():
    if "--list_files" in sys.argv:
        tasks = json.load(open(os.path.join(common.REPO, "data", "probe", f"{DATASET}.json")))["tasks"]
        print("\n".join(sorted({t["source"]["file"] for t in tasks})))
        return
    args = common.parser(DATASET, "local copy of lerobot/metaworld_mt50").parse_args()
    out = common.Outputs(args, DATASET)
    tasks = common.load_tasks(args)
    by_file = {}
    for t in tasks:
        by_file.setdefault(t["source"]["file"], []).append(t)
    for fn, group in by_file.items():
        tb = pq.read_table(os.path.join(args.raw, fn), columns=COLS)
        ep_col = tb.column("episode_index").to_numpy()
        for t in group:
            e = int(t["source"]["episode_index"])
            sub = tb.take(np.nonzero(ep_col == e)[0]).to_pydict()
            order = np.argsort(np.asarray(sub["frame_index"]))
            tf = t["target_frames"]
            assert len(order) == tf["n_frames"], (t["task_id"], len(order))
            for k, j in enumerate(order):
                rel = tf["paths"][k]
                p, p128 = out.frame(rel), out.frame128(rel)
                if not out.todo(p, p128):
                    continue
                im = Image.open(io.BytesIO(sub["observation.image"][j]["bytes"])).convert("RGB")
                w, h = im.size
                if min(w, h) > NATIVE_RES:
                    r = NATIVE_RES / min(w, h)
                    im_n = im.resize((max(1, round(w * r)), max(1, round(h * r))), Image.BILINEAR, reducing_gap=3.0)
                else:
                    im_n = im
                common.save_pil(im_n, p)
                common.save_pil(im.resize((128, 128), Image.BILINEAR), p128)
            a = out.action(t["task_id"])
            if out.todo(a):
                A = np.asarray([sub["action"][j] for j in order], dtype=np.float32)
                common.save_npy(a, np.clip(A, -1.0, 1.0).astype(np.float32))
    common.log_done(DATASET, len(tasks), out)


if __name__ == "__main__":
    main()

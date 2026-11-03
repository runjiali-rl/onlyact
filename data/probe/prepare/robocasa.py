"""RoboCasa365 probe clips (Table 1, Table 4).

Input (--raw): a local copy of the Hugging Face dataset nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos (the
LeRobot export of the atomic tasks). Only the video and parquet file of each clip are needed (552 files, 35 MB):
    python -c "import json; print('\\n'.join(t['source'][k] for t in json.load(open('data/probe/robocasa.json'))['tasks'] for k in ('video', 'data')))" > files.txt
    hf download nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos $(cat files.txt) --repo-type dataset \
        --revision <revision> --local-dir <raw>
Each clip's `source` names the episode's video (left agent-view camera) and parquet file inside that copy. The 276
clips are every episode of the 47 atomic tasks that were downloaded, one clip per episode.

Frames: the 60 consecutive frames in the middle of the episode (the clip's frame indices), decoded with decord,
256x256, saved unchanged as PNG f_<k>.png. 128x128 copies: PIL bilinear resize.
Action target `actions` (T, 11) float32 over the whole episode: the parquet `action` column (12-D) without
dimension 3 (constant over all 276 episodes). The probe reads the rows at the clip's frame indices.
"""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

DATASET = "robocasa"
KEPT_DIMS = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]   # action dimensions with non-zero spread over the 276 episodes


def main():
    args = common.parser(DATASET, "local copy of nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos").parse_args()
    out = common.Outputs(args, DATASET)
    import decord
    decord.bridge.set_bridge("native")
    tasks = common.load_tasks(args)
    for t in tasks:
        s = t["source"]
        tf = t["target_frames"]
        pairs = [(fi, out.frame(rel), out.frame128(rel)) for fi, rel in zip(tf["frame_indices"], tf["paths"])]
        if out.todo(*[p for _, a, b in pairs for p in (a, b)]):
            vr = decord.VideoReader(os.path.join(args.raw, s["video"]), num_threads=1)
            for fi, p, p128 in pairs:
                im = Image.fromarray(vr[fi].asnumpy(), "RGB")
                assert min(im.size) <= 292 and p.endswith(".png"), (p, im.size)
                common.save_pil(im, p)
                common.save_pil(im.resize((128, 128), Image.BILINEAR), p128)
        a = out.action(t["task_id"])
        if out.todo(a):
            df = pd.read_parquet(os.path.join(args.raw, s["data"]), columns=["action"])
            act = np.stack(df["action"].to_numpy()).astype(np.float32)          # (T, 12)
            assert act.shape[1] == 12 and max(tf["frame_indices"]) < len(act)
            common.save_npy(a, act[:, KEPT_DIMS].astype(np.float32))
    common.log_done(DATASET, len(tasks), out)


if __name__ == "__main__":
    main()

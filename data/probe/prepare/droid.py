"""DROID (IRIS lab) and DROID-IPRL probe clips (Table 1, Table 4).

Input (--raw): DROID episodes in the layout written by data/training/droid_download_and_process.py (raw DROID
release gs://gresearch/robotics/droid_raw/1.0.1, every MP4 decoded to 256x144 JPEG frames with ffmpeg):
    <raw>/<lab>/success/<date>/<time>/{trajectory.h5, varied_camera_1/frame_<i:05d>.jpg}
Fetch only the probe episodes with
    python data/probe/prepare/droid.py --dataset <droid|droid_iprl> --write_task_ids ids.txt
    python data/training/droid_download_and_process.py --data_root <dir> --labs <IRIS|IPRL> --task_ids ids.txt
Each clip's `source` names the episode (its path in the DROID release) and the camera (exterior camera 1).

Frames: the JPEG frames at the clip's frame indices (60 per clip; consecutive steps, or a stride when the episode was
long), saved unchanged as PNG f_<k>.png (their short side is below 292). 128x128 copies: PIL bilinear resize.
Action target `actions_agg` (T, 7) float32 over the whole episode: action/cartesian_velocity (6) and
action/gripper_position (1) of trajectory.h5; at every sampled index fi[k] the velocity is replaced by its mean over
[fi[k], fi[k+1]) so that a strided pair is described by the motion of the whole interval (the gripper command is
taken at the start). The probe reads the rows at the clip's frame indices.
"""
import json
import os
import sys

import h5py
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def episode_actions(traj_path, frame_indices):
    with h5py.File(traj_path, "r") as f:
        cv = np.asarray(f["action/cartesian_velocity"], dtype=np.float32)                 # (T, 6)
        gp = np.asarray(f["action/gripper_position"], dtype=np.float32).reshape(-1, 1)    # (T, 1)
    n = min(len(cv), len(gp))
    act = np.concatenate([cv[:n], gp[:n]], axis=1)
    assert max(frame_indices) < len(act)
    agg = act.copy()
    for k in range(len(frame_indices) - 1):
        lo, hi = int(frame_indices[k]), int(frame_indices[k + 1])
        if hi > lo:
            agg[lo, :6] = act[lo:hi, :6].mean(axis=0)
    return agg


def write_task_ids(argv):
    """--dataset <name> --write_task_ids <file>: the clip ids, one per line, for droid_download_and_process.py --task_ids."""
    ds, path = argv[argv.index("--dataset") + 1], argv[argv.index("--write_task_ids") + 1]
    tasks = json.load(open(os.path.join(common.REPO, "data", "probe", f"{ds}.json")))["tasks"]
    with open(path, "w") as f:
        f.write("".join(t["task_id"] + "\n" for t in tasks))
    print(f"[{ds}] wrote {len(tasks)} episode ids to {path}")


def main():
    if "--write_task_ids" in sys.argv:
        return write_task_ids(sys.argv[1:])
    args = common.parser(["droid", "droid_iprl"], "DROID root written by data/training/droid_download_and_process.py").parse_args()
    tasks = common.load_tasks(args)
    out = common.Outputs(args, args.dataset)
    for t in tasks:
        s = t["source"]
        ep_dir = os.path.join(args.raw, s["episode"])
        tf = t["target_frames"]
        for fi, rel in zip(tf["frame_indices"], tf["paths"]):
            p, p128 = out.frame(rel), out.frame128(rel)
            if out.todo(p, p128):
                im = Image.open(os.path.join(ep_dir, s["camera"], f"frame_{fi:05d}.jpg")).convert("RGB")
                assert min(im.size) <= 292 and rel.endswith(".png"), (rel, im.size)
                common.save_pil(im, p)
                common.save_pil(im.resize((128, 128), Image.BILINEAR), p128)
        a = out.action(t["task_id"])
        if out.todo(a):
            common.save_npy(a, episode_actions(os.path.join(ep_dir, "trajectory.h5"), tf["frame_indices"]))
    common.log_done(args.dataset, len(tasks), out)


if __name__ == "__main__":
    main()

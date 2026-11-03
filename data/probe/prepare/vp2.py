"""VP2 Robosuite and RoboDesk probe clips (Table 1, Table 4).

Input (--raw): the VP2 video-prediction training data (Tian et al., "A Control-Centric Benchmark for Video
Prediction"; Stanford Digital Repository https://purl.stanford.edu/cw843qn4148, also on Hugging Face as
s-tian/VP2, folder vp2_training_datasets/), laid out as
    <raw>/robosuite/5k_slice_rendered_256.hdf5                  Robosuite PushCenterMulti, 5,000 rendered demonstrations
    <raw>/robodesk/robodesk_<task>/noise_0.1_256.hdf5           RoboDesk, 7 tasks
Each clip's `source` names the file, the demonstration (data/<demo>) and the camera. The 300 clips per dataset are a
seeded random draw of demonstrations (RoboDesk: stratified over the 7 tasks); one demonstration is one scene.

Frames: the 35 frames of obs/<camera> (256x256), saved as PNG frame_<k>.png; 128x128 copies: PIL bicubic.
Action target `actions` (35, A) float32: data/<demo>/actions (Robosuite: end-effector delta x, y, z and gripper,
whose gripper is always 0; RoboDesk: 5-D).
"""
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main():
    args = common.parser(["vp2_robosuite", "vp2_robodesk"], "VP2 training data root (robosuite/, robodesk/)").parse_args()
    out = common.Outputs(args, args.dataset)
    tasks = common.load_tasks(args)
    by_file = {}
    for t in tasks:
        by_file.setdefault(t["source"]["file"], []).append(t)
    for fn, group in by_file.items():
        with h5py.File(os.path.join(args.raw, fn), "r") as f:
            for t in group:
                s = t["source"]
                g = f["data"][s["demo"]]
                tf = t["target_frames"]
                pairs = [(out.frame(rel), out.frame128(rel)) for rel in tf["paths"]]
                if out.todo(*[p for ab in pairs for p in ab]):
                    frames = g["obs"][s["camera"]][:]
                    assert len(frames) == tf["n_frames"], (t["task_id"], len(frames))
                    for fr, (p, p128) in zip(frames, pairs):
                        common.save_thumbnail_frame(fr, p, p128)
                a = out.action(t["task_id"])
                if out.todo(a):
                    common.save_npy(a, g["actions"][:].astype(np.float32))
    common.log_done(args.dataset, len(tasks), out)


if __name__ == "__main__":
    main()

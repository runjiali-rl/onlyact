"""LIBERO probe clips (Table 1, Table 2 LIBERO rows, Table 4).

Input (--raw): the LIBERO demonstration files as distributed by the LIBERO authors, one folder per suite:
    <raw>/libero_10/<task>_demo.hdf5   <raw>/libero_90/<task>_demo.hdf5
Each clip's `source` names the suite, the file and the demonstration key (`data/<demo_key>`).

Frames: obs/agentview_rgb (128x128), flipped vertically (LIBERO renders upside down), written with OpenCV as
<suite>/<task>/demo_<k>/frame_<t>.png for the time steps t of the clip (41 per demonstration). The frames are
already 128x128, so leakage128/libero links to probe/libero.
Action target `actions`: data/<demo_key>/actions, the full (T, 7) float32 array (end-effector position delta,
rotation delta, gripper); the probe reads the rows at the clip's frame indices.
"""
import os
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

DATASET = "libero"


def main():
    args = common.parser(DATASET, "folder with libero_10/ and libero_90/ demonstration HDF5 files").parse_args()
    out = common.Outputs(args, DATASET)
    tasks = common.load_tasks(args)
    by_file = {}
    for t in tasks:
        s = t["source"]
        by_file.setdefault((s["suite"], s["file"]), []).append(t)
    for (suite, fn), group in by_file.items():
        with h5py.File(os.path.join(args.raw, suite, fn), "r") as f:
            for t in group:
                g = f["data"][t["source"]["demo_key"]]
                tf = t["target_frames"]
                paths = [out.frame(p) for p in tf["paths"]]
                if out.todo(*paths):
                    av = g["obs"]["agentview_rgb"]                              # (T, 128, 128, 3) RGB
                    for p, i, rel in zip(paths, tf["frame_indices"], tf["paths"]):
                        assert rel.endswith(f"frame_{i:04d}.png"), rel
                        img = np.flipud(av[i])
                        common.save_cv2(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), p)
                ap = out.action(t["task_id"])
                if out.todo(ap):
                    common.save_npy(ap, g["actions"][:].astype(np.float32))
    out.link_frames128_to_frames()
    common.log_done(DATASET, len(tasks), out)


if __name__ == "__main__":
    main()

"""SpatialVID camera-navigation probe clips (Table 1, Table 2 SpatialVID rows, Table 4).

Input (--raw), from the Hugging Face dataset SpatialVID/SpatialVID (gated: accept its licence first):
    <raw>/videos/<group>/<clip_id>.mp4                   the clip re-encoded to 256x256 at 16 frames per second by
                                                          data/training/spatialvid_download_and_process.py (the
                                                          videos the encoder is trained on; groups 1-8 are needed)
    <raw>/annotations/<group>/<clip_id>/poses.npy        camera poses, from annotations/<group>.tar.gz
Extract the poses of the probe clips from downloaded annotation archives with
    python data/probe/prepare/spatialvid_nav.py --extract_annotations <folder with group_000N.tar.gz> --raw <raw>
Each clip's `source` names the group, the clip id, the annotation rows (one pose every int(fps / 5) source frames,
about 0.2 s) and, for each row, the frame of the 16 frames-per-second video that shows it.

Frames: those video frames (41-60 per clip), decoded in order with PyAV, 256x256, saved as PNG f_<k>.png; 128x128
copies: PIL bilinear.
Action target `actions_padded` (n, 6) float32: the camera's ego-motion between consecutive rows i, j = i + 1, in
camera i's axes (OpenCV: +x right, +y down, +z forward). poses.npy rows are [tx, ty, tz, qx, qy, qz, qw] of the
world-to-camera transform [R | t]; translation = R_i (C_j - C_i) with C = -R^T t, rotation = axis-angle of
R_i R_j^T (radians). The scale of the translation is that of SpatialVID's reconstruction. The last row is NaN padding
(the probe drops the row of the last frame).
"""
import os
import sys
import tarfile

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

DATASET = "spatialvid_nav"


def ego_actions(R, t):
    """R (n, 3, 3), t (n, 3) world-to-camera -> (n - 1, 6) float32 ego-motion in camera i's axes."""
    R_rel = R[1:] @ np.transpose(R[:-1], (0, 2, 1))
    t_rel = t[1:] - np.einsum("nij,nj->ni", R_rel, t[:-1])
    d = -np.einsum("nji,nj->ni", R_rel, t_rel)
    rv = Rotation.from_matrix(np.transpose(R_rel, (0, 2, 1))).as_rotvec()
    return np.concatenate([d, rv], axis=1).astype(np.float32)


def extract_annotations(tar_dir, raw, tasks):
    """Stream annotations/<group>.tar.gz and keep poses.npy of the probe clips."""
    want = {}
    for t in tasks:
        want.setdefault(t["source"]["group"], set()).add(t["source"]["clip_id"])
    for g, clips in sorted(want.items()):
        n = 0
        with tarfile.open(os.path.join(tar_dir, f"{g}.tar.gz"), mode="r|gz") as tf:
            for m in tf:
                parts = m.name.split("/")
                if m.isfile() and len(parts) >= 2 and parts[-1] == "poses.npy" and parts[-2] in clips:
                    common.save_bytes(tf.extractfile(m).read(), os.path.join(raw, "annotations", g, parts[-2], "poses.npy"))
                    n += 1
        print(f"[{DATASET}] {g}: {n} / {len(clips)} pose files", flush=True)


def main():
    if "--extract_annotations" in sys.argv:
        argv = sys.argv[1:]
        tar_dir, raw = argv[argv.index("--extract_annotations") + 1], argv[argv.index("--raw") + 1]
        import json
        tasks = json.load(open(os.path.join(common.REPO, "data", "probe", f"{DATASET}.json")))["tasks"]
        return extract_annotations(tar_dir, raw, tasks)
    args = common.parser(DATASET, "folder with videos/<group>/<clip_id>.mp4 and annotations/<group>/<clip_id>/poses.npy").parse_args()
    out = common.Outputs(args, DATASET)
    import av
    tasks = common.load_tasks(args)
    for t in tasks:
        s = t["source"]
        tf = t["target_frames"]
        need = {int(v): k for k, v in enumerate(s["video_frames"])}
        pairs = [(out.frame(rel), out.frame128(rel)) for rel in tf["paths"]]
        assert len(need) == len(pairs) == tf["n_frames"]
        if out.todo(*[p for ab in pairs for p in ab]):
            got = {}
            with av.open(os.path.join(args.raw, "videos", s["group"], f"{s['clip_id']}.mp4")) as c:
                for fi, fr in enumerate(c.decode(video=0)):
                    if fi in need:
                        got[fi] = fr.to_image().convert("RGB")
                    if len(got) == len(need):
                        break
            assert len(got) == len(need), (t["task_id"], len(got), len(need))
            for fi, k in need.items():
                common.save_pil(got[fi], pairs[k][0])
                common.save_pil(got[fi].resize((128, 128), Image.BILINEAR), pairs[k][1])
        a = out.action(t["task_id"])
        if out.todo(a):
            pose = np.load(os.path.join(args.raw, "annotations", s["group"], s["clip_id"], "poses.npy")).astype(np.float64)
            rows = s["annotation_rows"]
            A = ego_actions(Rotation.from_quat(pose[rows, 3:7]).as_matrix(), pose[rows, :3])
            common.save_npy(a, np.concatenate([A, np.full((1, 6), np.nan, A.dtype)], 0))
    common.log_done(DATASET, len(tasks), out)


if __name__ == "__main__":
    main()

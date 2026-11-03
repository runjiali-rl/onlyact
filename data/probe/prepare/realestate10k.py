"""RealEstate10K camera-navigation probe clips (Table 1, Table 4).

Input (--raw): the test-split chunks test/<NNNNNN>.torch of the Hugging Face dataset lhmd/re10k_torch (the
pixelSplat preprocessing of RealEstate10K: per video the official 18-number camera rows together with the 640x360
JPEG frames). Only the chunks named by the clips are needed (48 chunks, about 5 GB):
    python data/probe/prepare/realestate10k.py --list_files > files.txt
    hf download lhmd/re10k_torch $(cat files.txt) --repo-type dataset --local-dir <raw>
Each clip's `source` names the chunk, the video key and the sampled frame indices (every 3rd frame, 0.1 s apart,
41-60 per clip); the video is one scene.

Frames: the sampled JPEG frames, 640x360 -> 256x144 (PIL bilinear, reducing_gap 3.0), saved as PNG f_<k>.png;
128x128 copies: PIL bilinear from the 640x360 frame.
Action target `actions_padded` (n, 6) float32: the camera's ego-motion between consecutive sampled frames i, j in
camera i's axes (OpenCV: +x right, +y down, +z forward). The last 12 camera numbers are the row-major 3x4
world-to-camera [R | t]; translation = R_i (C_j - C_i) with C = -R^T t (structure-from-motion units, not metres),
rotation = axis-angle of R_i R_j^T (radians). The last row repeats the one before it (the probe never reads it).
"""
import io
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

DATASET = "realestate10k"
NATIVE_LONG = 256


def relative_actions(cam):
    """cam (n, 18) float64 -> (n - 1, 6) float32 [translation, rotation vector of R_rel = R_j R_i^T]."""
    M = cam[:, 6:].reshape(-1, 3, 4)
    R, t = M[:, :, :3], M[:, :, 3]
    R_rel = R[1:] @ np.transpose(R[:-1], (0, 2, 1))
    t_rel = t[1:] - np.einsum("nij,nj->ni", R_rel, t[:-1])
    d = -np.einsum("nji,nj->ni", R_rel, t_rel)
    rv = Rotation.from_matrix(R_rel).as_rotvec()
    return np.concatenate([d, rv], axis=1).astype(np.float32)


def main():
    if "--list_files" in sys.argv:
        tasks = json.load(open(os.path.join(common.REPO, "data", "probe", f"{DATASET}.json")))["tasks"]
        print("\n".join(sorted({t["source"]["chunk"] for t in tasks})))
        return
    args = common.parser(DATASET, "folder with test/<NNNNNN>.torch chunks of lhmd/re10k_torch").parse_args()
    out = common.Outputs(args, DATASET)
    import torch
    tasks = common.load_tasks(args)
    by_chunk = {}
    for t in tasks:
        by_chunk.setdefault(t["source"]["chunk"], {})[t["source"]["key"]] = t
    for chunk_name, wanted in by_chunk.items():
        chunk = torch.load(os.path.join(args.raw, chunk_name), map_location="cpu", weights_only=False)
        found = 0
        for e in chunk:
            t = wanted.get(e["key"])
            if t is None:
                continue
            found += 1
            sel = t["source"]["frame_indices"]
            assert len(e["images"]) == t["source"]["n_source_frames"], t["task_id"]
            tf = t["target_frames"]
            pairs = [(out.frame(rel), out.frame128(rel)) for rel in tf["paths"]]
            if out.todo(*[p for ab in pairs for p in ab]):
                for j, (p, p128) in zip(sel, pairs):
                    im = Image.open(io.BytesIO(e["images"][j].numpy().tobytes())).convert("RGB")
                    w, h = im.size
                    if max(w, h) > NATIVE_LONG:
                        r = NATIVE_LONG / max(w, h)
                        im_n = im.resize((max(1, round(w * r)), max(1, round(h * r))), Image.BILINEAR, reducing_gap=3.0)
                    else:
                        im_n = im
                    common.save_pil(im_n, p)
                    common.save_pil(im.resize((128, 128), Image.BILINEAR), p128)
            a = out.action(t["task_id"])
            if out.todo(a):
                A = relative_actions(e["cameras"].numpy().astype(np.float64)[sel])
                A[:, 3:] *= -1.0              # rotation of R_i R_j^T = the camera's own rotation from i to j
                common.save_npy(a, np.concatenate([A, A[-1:]], 0).astype(A.dtype))
        assert found == len(wanted), (chunk_name, found, len(wanted))
    common.log_done(DATASET, len(tasks), out)


if __name__ == "__main__":
    main()

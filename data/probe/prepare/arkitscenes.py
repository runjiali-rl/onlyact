"""ARKitScenes camera-navigation probe clips (Table 1, Table 4).

Input (--raw): scans of the ARKitScenes raw subset, Training fold, from Apple's downloader (github.com/apple/ARKitScenes):
    python download_data.py raw --split Training --video_id_csv <ids.csv> --download_dir <dir> \
        --raw_dataset_assets lowres_wide.traj lowres_wide --keep_zip
which gives <dir>/raw/Training/<video_id>/{lowres_wide.traj, lowres_wide.zip or lowres_wide/}; pass
--raw <dir>/raw/Training. `--write_video_ids ids.csv` writes the id list. With --remote the PNG frames are instead
read by HTTP byte ranges from Apple's server (about 7 MB per scan instead of the 150-190 MB zip).
Each clip's `source` names the scan and the 60 indices of the trajectory poses it uses (stride 3 = 0.3 s, a window
centred in the scan); the scan is one scene.

Device orientation: the sky direction is found from the mean camera up and right vectors of the whole trajectory;
frames are rotated upright and the poses are right-multiplied by the same rotation, so the actions are expressed in
the axes of the upright image.
Frames: for each pose, the lowres_wide PNG (256x192) with the nearest timestamp, decoded with OpenCV, rotated upright,
written as <video_id>/f_<k>.png; 128x128 copies: cv2.resize with INTER_AREA.
Action target `actions` (60, 6) float32: row k = camera motion from pose k to pose k + 1 in pose k's upright camera
axes (OpenCV: +x right, +y down, +z forward): T_rel = inv(C_k) C_{k+1} with C the upright camera-to-world transform,
translation T_rel[:3, 3] in metres, rotation = Rodrigues(T_rel[:3, :3]) in radians. The last row is zero padding.
Each .traj line is "timestamp rx ry rz tx ty tz", the axis-angle rotation and translation of world-to-camera.
"""
import io
import math
import os
import sys
import time
import zipfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

DATASET = "arkitscenes"
URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training"
NATIVE_RES = 256
MAX_DT = 0.05   # largest allowed |frame timestamp - pose timestamp|, seconds


# ---------------------------------------------------------------- trajectory and device orientation
def parse_trajectory(path):
    """[(timestamp, 4x4 camera-to-world)] from a .traj file."""
    out = []
    with open(path) as f:
        for line in f:
            tok = line.strip().split()
            if len(tok) != 7:
                continue
            r_w2c, _ = cv2.Rodrigues(np.array([float(tok[1]), float(tok[2]), float(tok[3])]))
            w2c = np.eye(4)
            w2c[:3, :3] = r_w2c
            w2c[:3, 3] = np.array([float(tok[4]), float(tok[5]), float(tok[6])])
            out.append((float(tok[0]), np.linalg.inv(w2c)))
    return out


def scene_orientation(poses):
    """(sky direction, 4x4 rotation that maps the upright camera to the device camera, cv2 rotate code or None)."""
    up_sum, right_sum = np.zeros(4), np.zeros(4)
    for _, c2w in poses:
        up_sum += c2w @ np.array([0.0, -1.0, 0.0, 0.0])      # device up = -y
        right_sum += c2w @ np.array([1.0, 0.0, 0.0, 0.0])    # device right = +x
    up, right = up_sum / len(poses), right_sum / len(poses)
    world_up = np.array([0.0, 0.0, 1.0, 0.0])
    up_angle = np.degrees(np.arccos(np.clip(np.dot(world_up, up), -1.0, 1.0)))
    right_angle = np.degrees(np.arccos(np.clip(np.dot(world_up, right), -1.0, 1.0)))
    if abs(up_angle - 90.0) < abs(right_angle - 90.0):      # landscape
        sky, angle = ("LEFT", math.pi / 2.0) if right_angle > 90.0 else ("RIGHT", -math.pi / 2.0)
    else:                                                    # portrait
        sky, angle = ("DOWN", math.pi) if up_angle > 90.0 else ("UP", 0.0)
    if angle == 0.0:
        rotated_to_cam = np.eye(4)
    else:
        c, s = math.cos(angle), math.sin(angle)
        cam_to_rotated = np.eye(4)
        cam_to_rotated[:2, :2] = [[c, -s], [s, c]]
        rotated_to_cam = np.linalg.inv(cam_to_rotated)
    code = {"UP": None, "RIGHT": cv2.ROTATE_90_COUNTERCLOCKWISE, "LEFT": cv2.ROTATE_90_CLOCKWISE, "DOWN": cv2.ROTATE_180}[sky]
    return sky, rotated_to_cam, code


def actions(poses, idxs, rotated_to_cam, rotate_code):
    C = [poses[i][1] if rotate_code is None else poses[i][1] @ rotated_to_cam for i in idxs]
    A = np.zeros((len(idxs), 6), dtype=np.float32)
    for k in range(len(idxs) - 1):
        rel = np.linalg.inv(C[k]) @ C[k + 1]
        A[k, :3] = rel[:3, 3]
        rv, _ = cv2.Rodrigues(rel[:3, :3])
        A[k, 3:] = rv.ravel()
    return A


def centred_window(n_poses, n_frames=60, stride=3):
    span = (n_frames - 1) * stride
    start = (n_poses - 1 - span) // 2
    return [start + k * stride for k in range(n_frames)]


# ---------------------------------------------------------------- frames
class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP byte ranges with a read-ahead buffer (zipfile only seeks and reads)."""

    def __init__(self, url, session, readahead=1 << 15, retries=4):
        self.url, self.s, self.readahead, self.retries = url, session, readahead, retries
        self.pos, self._buf, self._buf_start = 0, b"", -1
        self.size = int(self._retry(lambda: self.s.head(url, timeout=30)).headers["Content-Length"])

    def _retry(self, fn):
        for i in range(self.retries):
            try:
                r = fn()
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                return r
            except Exception:
                if i == self.retries - 1:
                    raise
                time.sleep(1.5 * (i + 1))

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = off if whence == io.SEEK_SET else self.pos + off if whence == io.SEEK_CUR else self.size + off
        self.pos = max(0, min(self.pos, self.size))
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if not (0 <= self._buf_start <= self.pos and self.pos + n <= self._buf_start + len(self._buf)):
            end = min(self.pos + max(n, self.readahead), self.size) - 1
            self._buf = self._retry(lambda: self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=60)).content
            self._buf_start = self.pos
            n = min(n, len(self._buf))
        off = self.pos - self._buf_start
        data = self._buf[off:off + n]
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def frame_source(raw, vid, remote, session=None):
    """{timestamp: reader()} of the scan's lowres_wide PNG frames (file names <video_id>_<timestamp>.png)."""
    folder = os.path.join(raw, vid, "lowres_wide") if raw else None
    if folder and os.path.isdir(folder):
        names = {n: (lambda p=os.path.join(folder, n): open(p, "rb").read()) for n in os.listdir(folder)}
    else:
        zf = zipfile.ZipFile(HttpRangeFile(f"{URL}/{vid}/lowres_wide.zip", session) if remote
                             else os.path.join(raw, vid, "lowres_wide.zip"))
        names = {os.path.basename(i.filename): (lambda i=i: zf.read(i)) for i in zf.infolist()}
    out = {}
    for base, reader in names.items():
        stem = base[:-4]
        if base.lower().endswith(".png") and "_" in stem:
            try:
                out[float(stem.rsplit("_", 1)[1])] = reader
            except ValueError:
                pass
    return out


def main():
    if "--write_video_ids" in sys.argv:
        import json
        path = sys.argv[sys.argv.index("--write_video_ids") + 1]
        tasks = json.load(open(os.path.join(common.REPO, "data", "probe", f"{DATASET}.json")))["tasks"]
        with open(path, "w") as f:
            f.write("video_id,fold\n" + "".join(f"{t['source']['video_id']},{t['source']['fold']}\n" for t in tasks))
        return
    ap = common.parser(DATASET, "folder with <video_id>/lowres_wide.traj and lowres_wide.zip (or lowres_wide/)")
    ap.add_argument("--remote", action="store_true", help="read the PNG frames by HTTP byte ranges from Apple's server")
    args = ap.parse_args()
    out = common.Outputs(args, DATASET)
    session = None
    if args.remote:
        import requests
        session = requests.Session()
    tasks = common.load_tasks(args)
    for t in tasks:
        s = t["source"]
        vid, idxs = s["video_id"], s["pose_indices"]
        poses = parse_trajectory(os.path.join(args.raw, vid, "lowres_wide.traj"))
        assert idxs == centred_window(len(poses)), t["task_id"]
        sky, rotated_to_cam, code = scene_orientation(poses)
        assert sky == s["sky_direction"], (t["task_id"], sky)
        tf = t["target_frames"]
        pairs = [(out.frame(rel), out.frame128(rel)) for rel in tf["paths"]]
        if out.todo(*[p for ab in pairs for p in ab]):
            frames = frame_source(args.raw, vid, args.remote, session)
            mt = np.array(sorted(frames))
            for i, (p, p128) in zip(idxs, pairs):
                ts = poses[i][0]
                j = int(np.argmin(np.abs(mt - ts)))
                assert abs(float(mt[j]) - ts) <= MAX_DT, (t["task_id"], ts)
                img = cv2.imdecode(np.frombuffer(frames[float(mt[j])](), np.uint8), cv2.IMREAD_COLOR)
                up = img if code is None else cv2.rotate(img, code)
                h, w = up.shape[:2]
                assert max(h, w) <= NATIVE_RES, (t["task_id"], up.shape)
                common.save_cv2(up, p)
                common.save_cv2(cv2.resize(up, (128, 128), interpolation=cv2.INTER_AREA), p128)
        a = out.action(t["task_id"])
        if out.todo(a):
            common.save_npy(a, actions(poses, idxs, rotated_to_cam, code))
    common.log_done(DATASET, len(tasks), out)


if __name__ == "__main__":
    main()

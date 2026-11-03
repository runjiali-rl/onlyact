"""RLBench probe clips (Table 1, Table 2 RLBench rows, Table 4).

Input (--raw): the pre-generated RLBench demonstrations of PerAct (18 tasks, train split, 100 episodes per task,
128x128 cameras), extracted so that each episode is
    <raw>/<task>/all_variations/episodes/episode<N>/{front_rgb/<i>.png, low_dim_obs.pkl, variation_number.pkl}
Each clip's `source` names the task, the episode number and the variation (the clip id is
rlbench_<task>_<episode:04d>_v<variation:02d>).

Frames: 41 time steps sub-sampled evenly from the episode (int(j * T / 41) when T > 41) of front_rgb, copied byte for
byte as frame_<j>.png. They are 128x128, so leakage128/rlbench links to probe/rlbench.
Action target `actions_delta` (41, 7) float32: from the gripper pose [x, y, z, qx, qy, qz, qw] and gripper_open of the
41 sampled observations, row t = [x_{t+1} - x_t (3), rotation vector of q_t^-1 q_{t+1} (3), gripper_open_t]; the last
row repeats the one before it (the probe never reads it).
low_dim_obs.pkl is unpickled with minimal stand-ins for the RLBench classes, so RLBench itself is not needed.
"""
import os
import pickle
import shutil
import sys
import types

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

DATASET = "rlbench"
N_FRAMES = 41


class Observation:
    """Plain container for an unpickled rlbench.backend.observation.Observation."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Demo:
    """Plain container for an unpickled rlbench.demo.Demo."""

    def __init__(self, observations=None, random_seed=None, num_reset_attempts=None):
        self._observations = observations or []
        self.random_seed = random_seed
        self.num_reset_attempts = num_reset_attempts

    def __len__(self):
        return len(self._observations)

    def __getitem__(self, i):
        return self._observations[i]

    def __iter__(self):
        return iter(self._observations)


def install_stubs():
    rlbench_mod = types.ModuleType("rlbench")
    backend_mod = types.ModuleType("rlbench.backend")
    obs_mod = types.ModuleType("rlbench.backend.observation")
    demo_mod = types.ModuleType("rlbench.demo")
    obs_mod.Observation = Observation
    demo_mod.Demo = Demo
    backend_mod.observation = obs_mod
    rlbench_mod.demo, rlbench_mod.backend = demo_mod, backend_mod
    sys.modules.setdefault("rlbench", rlbench_mod)
    sys.modules.setdefault("rlbench.backend", backend_mod)
    sys.modules["rlbench.backend.observation"] = obs_mod
    sys.modules["rlbench.demo"] = demo_mod


def camera_frames(episode_dir, camera="front_rgb"):
    cam = os.path.join(episode_dir, camera)
    frames = sorted((int(os.path.splitext(n)[0]), os.path.join(cam, n)) for n in os.listdir(cam) if n.endswith(".png"))
    return [p for _, p in frames]


def subsample_indices(n_total, n_target):
    if n_total <= n_target:
        return list(range(n_total))
    step = n_total / n_target
    return [int(j * step) for j in range(n_target)]


def absolute_actions(demo):
    """(T, 8) float32: gripper pose (position 3 + quaternion xyzw 4) and gripper_open of every observation."""
    out = []
    for o in demo:
        gp = np.asarray(o.gripper_pose, dtype=np.float32)
        go = float(getattr(o, "gripper_open", 0.0))
        out.append(np.concatenate([gp, [go]], axis=0))
    return np.stack(out).astype(np.float32)


def delta_actions(arr32):
    arr = arr32.astype(np.float64)
    pos, quat, grip = arr[:, :3], arr[:, 3:7], arr[:, 7:8]
    quat = quat / np.clip(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8, None)
    R = Rotation.from_quat(quat)
    dpos = np.diff(pos, axis=0)
    drot = (R[:-1].inv() * R[1:]).as_rotvec()
    out = np.concatenate([dpos, drot, grip[:-1]], axis=1).astype(np.float32)
    return np.concatenate([out, out[-1:]], axis=0)


def main():
    args = common.parser(DATASET, "folder with <task>/all_variations/episodes/episode<N>/ (PerAct RLBench, train split)").parse_args()
    out = common.Outputs(args, DATASET)
    install_stubs()
    tasks = common.load_tasks(args)
    for t in tasks:
        s = t["source"]
        ep_dir = os.path.join(args.raw, s["task"], "all_variations", "episodes", f"episode{s['episode']}")
        with open(os.path.join(ep_dir, "variation_number.pkl"), "rb") as f:
            assert int(pickle.load(f)) == s["variation"], t["task_id"]
        with open(os.path.join(ep_dir, "low_dim_obs.pkl"), "rb") as f:
            demo = pickle.load(f)
        frames = camera_frames(ep_dir)
        n_total = min(len(frames), len(demo))
        sel = subsample_indices(n_total, N_FRAMES)
        tf = t["target_frames"]
        assert len(sel) == tf["n_frames"], (t["task_id"], len(sel))
        for j, rel in zip(sel, tf["paths"]):
            p = out.frame(rel)
            if out.todo(p):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                shutil.copyfile(frames[j], p + ".tmp")
                os.replace(p + ".tmp", p)
        ap = out.action(t["task_id"])
        if out.todo(ap):
            common.save_npy(ap, delta_actions(absolute_actions(demo[:n_total])[sel]))
    out.link_frames128_to_frames()
    common.log_done(DATASET, len(tasks), out)


if __name__ == "__main__":
    main()

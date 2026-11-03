"""Procgen probe clips (heist, jumper, maze, caveflyer; Table 1, Table 4).

No download: every clip is re-rendered with the procgen package (pip install procgen; it needs gym3 and runs on the
CPU). Each clip's `source` holds the game, the level (start_level, a single level per episode), the distribution
mode and the 40 discrete actions of the episode (15-way Procgen action set). The environment is created with
num_levels=1 and render_mode="rgb_array"; the 41 frames are the 512x512 info["rgb"] renders after reset and after
each action. The level and its dynamics are fixed by start_level, so replaying the actions reproduces the episode;
an episode that ends early is an error.

Frames: 512x512 -> 256x256 (PIL thumbnail, bicubic) as frame_<k>.png; 128x128 copies: PIL bicubic from the
256x256 image.
Action target `actions` (40,) int64: the episode's actions (the probe is a 15-way classifier here).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

DATASET = "procgen_plan"


def render_episode(env_name, start_level, distribution_mode, actions):
    from procgen import ProcgenGym3Env
    env = ProcgenGym3Env(env_name=env_name, num=1, num_levels=1, start_level=int(start_level), use_sequential_levels=False,
                         distribution_mode=distribution_mode, render_mode="rgb_array")
    try:
        frames = [env.get_info()[0]["rgb"]]
        for a in actions:
            env.act(np.asarray([a], dtype=np.int32))
            _, _, first = env.observe()
            if bool(first[0]):
                raise RuntimeError(f"{env_name} level {start_level}: episode ended before the last action")
            frames.append(env.get_info()[0]["rgb"])
        return frames
    finally:
        env.close()


def main():
    args = common.parser(DATASET, None).parse_args()
    out = common.Outputs(args, DATASET)
    tasks = common.load_tasks(args)
    for t in tasks:
        s = t["source"]
        tf = t["target_frames"]
        pairs = [(out.frame(rel), out.frame128(rel)) for rel in tf["paths"]]
        if out.todo(*[p for ab in pairs for p in ab]):
            frames = render_episode(s["env"], s["start_level"], s["distribution_mode"], s["actions"])
            assert len(frames) == tf["n_frames"]
            for fr, (p, p128) in zip(frames, pairs):
                common.save_thumbnail_frame(fr, p, p128)
        a = out.action(t["task_id"])
        if out.todo(a):
            common.save_npy(a, np.asarray(s["actions"], dtype=np.int64))
    common.log_done(DATASET, len(tasks), out)


if __name__ == "__main__":
    main()

"""Materialise the Table 3 frames listed in data/downstream/<domain>/*.json from the public source datasets.

  clips     DMLab / Minecraft: cut the seven-frame clip of every listed task out of the TECO episode
            (DMLab: <episode>.npz with "video" (T, 64, 64, 3) and "actions" (T,), actions aligned by dropping the last;
             Minecraft: <episode>.mp4 decoded with OpenCV + <episode>.npz "actions" (T+1,), aligned by dropping the first)
            and save <task_id>.npz with "video" (7, H, W, 3) uint8 and "actions_transition" (6,) int64.
  nuscenes  nuScenes CAM_FRONT JPEGs (1600 x 900) -> centre square crop, float32 / 255, bicubic resize to 256, clamp,
            x255, round -> <frame_token>.png.
  habitat   HM3D example scenes: render every listed frame with habitat-sim at its stored pose (settings in the list
            header, "render"), 128 x 128 RGB -> float32 / 255, bicubic resize to 256, clamp, x255, round
            -> <task_id>/<frame>.png. Needs habitat-sim 0.3.3 and torch in the same environment (see README).
Every output is checked against the pixel hash stored in the list (--no_check_hashes skips the check).

  python -m downstream.prepare clips --list data/downstream/dmlab/train.json --source TECO/dmlab --out DATA/dmlab
  python -m downstream.prepare nuscenes --list data/downstream/nuscenes/test.json --source NUSCENES --out DATA/nuscenes
  python -m downstream.prepare habitat --list data/downstream/habitat/test.json --source HM3D_EXAMPLE --out DATA/habitat
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from .common import repo_path


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _decode_mp4(path):
    import cv2
    cap = cv2.VideoCapture(str(path)); frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok: break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return np.stack(frames)


def clips(tasks, source, out, check=True):
    cache = {}
    for t in tasks:
        src = t["source"]
        if src["video"] not in cache:
            if src["video"].endswith(".npz"):
                with np.load(os.path.join(source, src["video"])) as z:
                    video, actions = z["video"], np.asarray(z["actions"])[:-1]
            else:
                video = _decode_mp4(os.path.join(source, src["video"]))
                with np.load(os.path.join(source, src["actions"])) as z:
                    actions = np.asarray(z["actions"])[1:]
            cache = {src["video"]: (video, actions.astype(np.int64))}
        video, actions = cache[src["video"]]
        idx = t["frame_indices"]; clip = np.ascontiguousarray(video[idx]); at = actions[idx[0]:idx[0] + 6]
        assert at.tolist() == t["actions_transition"], t["task_id"]
        if check and _sha(clip) != t["clip"]["video_sha256"]:
            raise ValueError(f"{t['task_id']}: decoded pixels differ from the released clip (decoder / source version?)")
        np.savez_compressed(os.path.join(out, t["clip"]["path"]), video=clip, actions_transition=at)


def shared_rgb(path):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    with Image.open(path) as im:
        im.load(); tensor = torch.from_numpy(np.array(im.convert("RGB"))).float() / 255.
    h, w = tensor.shape[:2]; s = min(h, w)
    tensor = tensor[(h-s)//2:(h-s)//2+s, (w-s)//2:(w-s)//2+s]
    tensor = F.interpolate(tensor.permute(2, 0, 1)[None], 256, mode="bicubic")[0].clamp(0, 1)
    return (tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def nuscenes(tasks, source, out, check=True):
    from PIL import Image
    for t in tasks:
        for name, rel, h in zip(t["source_filenames"], t["target_frames"]["paths"], t["target_frames"]["pixels_sha256"]):
            dest = os.path.join(out, rel)
            if os.path.exists(dest): continue
            rgb = shared_rgb(os.path.join(source, name))
            if check and _sha(rgb) != h:
                raise ValueError(f"{name}: pixels differ from the released frame (image decoder version?)")
            Image.fromarray(rgb).save(dest)


def _habitat_scene(source, scene, render):
    """One simulator per scene: no physics, one pinhole RGB camera, values from the list header."""
    import habitat_sim
    sim_cfg = habitat_sim.SimulatorConfiguration(); sim_cfg.enable_physics = False
    sim_cfg.scene_dataset_config_file = os.path.join(source, "hm3d_example_basis.scene_dataset_config.json")
    sim_cfg.scene_id = os.path.join(source, scene, scene.split("-", 1)[1] + ".basis.glb")
    camera = habitat_sim.CameraSensorSpec(); camera.uuid = "rgb"
    camera.sensor_type, camera.sensor_subtype = habitat_sim.SensorType.COLOR, habitat_sim.SensorSubType.PINHOLE
    camera.resolution = list(render["resolution"]); camera.position = list(render["sensor_position_metres"])
    camera.orientation = [0., 0., 0.]; camera.hfov = render["hfov_degrees"]
    camera.near, camera.far = render["near_metres"], render["far_metres"]
    agent = habitat_sim.agent.AgentConfiguration(); agent.sensor_specifications = [camera]
    agent.height, agent.radius = render["agent_height_metres"], render["agent_radius_metres"]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent]))
    return sim, sim.initialize_agent(0)


def habitat(tasks, source, out, render, check=True):
    import habitat_sim
    import torch
    from habitat_sim.utils.common import quat_from_coeffs
    from PIL import Image
    from .data import preprocess_rgb256
    sim = scene = None
    try:
        for t in tasks:
            frames = t["target_frames"]
            todo = [(pose, rel, h) for pose, rel, h in zip(t["poses"], frames["paths"], frames["pixels_sha256"])
                    if not os.path.exists(os.path.join(out, rel))]
            if not todo: continue
            if t["scene_id"] != scene:
                if sim is not None: sim.close()
                scene = t["scene_id"]; sim, agent = _habitat_scene(source, scene, render)
            for pose, rel, h in todo:
                # The stored pose is the body pose; the camera sits at sensor_position_metres relative to it.
                state = habitat_sim.AgentState(np.asarray(pose["position_metres"], dtype=np.float32),
                                               quat_from_coeffs(pose["rotation_xyzw"]))
                agent.set_state(state, reset_sensors=True)
                rgb = np.ascontiguousarray(sim.get_sensor_observations()["rgb"][:, :, :3])
                png = preprocess_rgb256(rgb).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
                if check and _sha(png) != h:
                    raise ValueError(f"{rel}: rendered pixels differ from the released frame (GPU / driver? see README)")
                dest = os.path.join(out, rel); os.makedirs(os.path.dirname(dest), exist_ok=True)
                Image.fromarray(png).save(dest)
    finally:
        if sim is not None: sim.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["clips", "nuscenes", "habitat"]); ap.add_argument("--list", required=True)
    ap.add_argument("--source", required=True, help="TECO episode folder of the domain, the nuScenes root (with samples/, "
                    "sweeps/), or the HM3D example folder (with hm3d_example_basis.scene_dataset_config.json)")
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_check_hashes", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    with open(repo_path(a.list)) as f:
        listed = json.load(f)
    tasks = listed["tasks"][:a.limit] if a.limit else listed["tasks"]
    if a.what == "habitat":
        habitat(tasks, a.source, a.out, listed["render"], check=not a.no_check_hashes)
    else:
        (clips if a.what == "clips" else nuscenes)(tasks, a.source, a.out, check=not a.no_check_hashes)
    print(f"prepared {len(tasks)} clips in {a.out}")


if __name__ == "__main__":
    main()

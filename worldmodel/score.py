"""Score Table 2 rollouts written by generate.py (CPU only).

Self cells: peak signal-to-noise ratio (PSNR, dB) against the ground truth, the mean over clips of the per-clip values in
results.jsonl (each the mean over the 19 predicted frames, computed in floating point during generation).

Transfer cells: transfer direction, i.e. does the generated video move like the SOURCE clip whose actions drive it?
Farneback optical flow at 128 x 128 between consecutive frames of [target frame ctx-1 + the predicted frames] (read from
grid_XX.png) and of the source clip's frames ctx-1 .. ctx+horizon-1. Per frame, the global motion vector is the mean flow
over pixels that move more than 0.3 px (zero if fewer than 1 % move). dir_src = mean cosine between the generated and the
source vectors over the frames where both are longer than 0.2 px; a pair with fewer than 3 such frames is left unscored
and excluded from the cell mean. 1 = follows the source, 0 = unrelated, negative = follows the target's own motion (the
pairs were chosen so that source and target move in opposite directions). dir_tgt is the same against the target clip.

    python score.py --runs_root runs/table2 --data_root libero=$DATA/libero rlbench=$DATA/rlbench \
        spatialvid_nav=$DATA/spatialvid_nav nuscenes_mini=$DATA/nuscenes_mini

expects <runs_root>/<arm>/<cell>/ from generate.py and writes <runs_root>/scores.json (per-clip and per-cell values).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from clip_lists import load_clip_list, read_clip_list, rollout_pairs  # noqa: E402

ARMS = ("no_action", "adaworld_lam", "dreamdojo_lam", "onlyact")
ARM_NAMES = {"no_action": "No action", "adaworld_lam": "AdaWorld-LAM (retrained)", "dreamdojo_lam": "DreamDojo-LAM (retrained)",
             "onlyact": "OnlyAct"}
CELLS = ("libero_self", "libero_transfer", "rlbench_self", "rlbench_transfer", "spatialvid_self", "spatialvid_transfer",
         "nuscenes_dev_self", "nuscenes_dev_transfer", "nuscenes_conf_self", "nuscenes_conf_transfer")
START, CTX, HOR, GAP, FRES = 0, 6, 19, 1, 128
THR, GMIN, MIN_FRAMES = 0.3, 0.2, 3


def load_frame(path, res):
    im = Image.open(path).convert("RGB"); w, h = im.size
    if h != w:
        s = min(h, w); im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return np.asarray(im.resize((res, res), Image.BICUBIC)) if im.size[0] != res else np.asarray(im)


def grey(x):  # (R, R, 3) uint8 -> (FRES, FRES) uint8
    g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, (FRES, FRES), interpolation=cv2.INTER_AREA) if g.shape[0] != FRES else g


def flows(seq):  # list of (R, R, 3) uint8 -> (T - 1, FRES, FRES, 2)
    gs = [grey(x) for x in seq]
    return np.stack([cv2.calcOpticalFlowFarneback(gs[i], gs[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0) for i in range(len(gs) - 1)])


def global_motion(fl, thr=THR):  # per-frame mean flow over moving pixels
    mag = np.linalg.norm(fl, axis=-1)
    g = []
    for t in range(fl.shape[0]):
        mv = mag[t] > thr
        g.append(fl[t][mv].mean(0) if mv.mean() >= 0.01 else np.zeros(2))
    return np.array(g)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def mean_direction(gr, gref, gmin=GMIN):
    use = [t for t in range(HOR) if np.linalg.norm(gr[t]) > gmin and np.linalg.norm(gref[t]) > gmin]
    return (float(np.mean([cos(gr[t], gref[t]) for t in use])) if len(use) >= MIN_FRAMES else None), len(use)


def read_grid(path):
    g = np.asarray(Image.open(path).convert("RGB")); R = g.shape[0] // 2
    return g[R:].reshape(R, -1, R, 3).transpose(1, 0, 2, 3), R  # predicted frames, resolution


def score_psnr(run_dir):
    rows = [json.loads(line) for line in open(os.path.join(run_dir, "results.jsonl"))]
    return rows, float(np.mean([r["psnr"] for r in rows]))


def score_direction(run_dir, cell, cache):
    tasks = {t["task_id"]: t for t in cell["tasks"]}
    out = []
    for j, (src, tgt) in enumerate(rollout_pairs(cell)):
        gp = os.path.join(run_dir, f"grid_{j:02d}.png")
        if not os.path.exists(gp):
            raise FileNotFoundError(f"{gp} missing: the run is incomplete")
        pred, R = read_grid(gp)
        if (src, tgt, R) not in cache:
            sp, tp = tasks[src]["target_frames"]["abs_paths"], tasks[tgt]["target_frames"]["abs_paths"]
            s_frames = [load_frame(sp[START + k * GAP], R) for k in range(CTX - 1, CTX + HOR)]
            t_frames = [load_frame(tp[START + k * GAP], R) for k in range(CTX - 1, CTX + HOR)]
            cache[(src, tgt, R)] = (global_motion(flows(s_frames)), global_motion(flows(t_frames)), t_frames[0])
        gs, gt, t5 = cache[(src, tgt, R)]
        gr = global_motion(flows([t5] + list(pred)))
        ds, ns = mean_direction(gr, gs)
        dt, nt = mean_direction(gr, gt)
        out.append({"pair": j, "source": src, "target": tgt, "dir_src": ds, "dir_tgt": dt, "n_src": ns, "n_tgt": nt})
    v = [r["dir_src"] for r in out if r["dir_src"] is not None]
    return out, (float(np.mean(v)) if v else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_root", required=True, help="folder with <arm>/<cell>/ rollouts from generate.py")
    ap.add_argument("--cells_dir", default=os.path.join(os.path.dirname(HERE), "data", "worldmodel"))
    ap.add_argument("--data_root", nargs="+", default=[], metavar="DATASET=PATH",
                    help="data root per dataset (needed for transfer cells), e.g. libero=/data/libero")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--out", default="", help="default: <runs_root>/scores.json")
    a = ap.parse_args()
    roots = dict(x.split("=", 1) for x in a.data_root)

    scores = {"settings": {"start": START, "ctx": CTX, "horizon": HOR, "gap": GAP, "flow_resolution": FRES, "move_threshold": THR,
                           "min_vector": GMIN, "min_frames": MIN_FRAMES}, "cells": {}}
    for cname in a.cells:
        cpath = os.path.join(a.cells_dir, f"{cname}.json")
        dataset = read_clip_list(cpath)["dataset"]
        cell = load_clip_list(cpath, roots[dataset]) if dataset in roots else read_clip_list(cpath)
        cache, scores["cells"][cname] = {}, {}
        for arm in a.arms:
            run_dir = os.path.join(a.runs_root, arm, cname)
            if not os.path.exists(os.path.join(run_dir, "results.jsonl")):
                print(f"[score] {arm}/{cname}: no results.jsonl, skipped")
                continue
            rows, psnr = score_psnr(run_dir)
            if [(r["source"], r["target"]) for r in rows] != rollout_pairs(cell):
                raise ValueError(f"{run_dir}: rollouts differ from the cell's order")
            entry = {"n": len(rows), "psnr": psnr, "per_clip_psnr": [r["psnr"] for r in rows]}
            if cell["mode"] == "transfer":
                if cell["dataset"] not in roots:
                    raise ValueError(f"{cname}: pass --data_root {cell['dataset']}=PATH to score transfer direction")
                per_pair, d = score_direction(run_dir, cell, cache)
                entry.update({"direction": d, "n_scored": sum(r["dir_src"] is not None for r in per_pair), "per_pair": per_pair})
                print(f"[score] {arm}/{cname}: direction {d:.3f} over {entry['n_scored']}/{len(rows)} pairs (PSNR vs target {psnr:.2f})")
            else:
                print(f"[score] {arm}/{cname}: PSNR {psnr:.2f} dB over {len(rows)} clips")
            scores["cells"][cname][arm] = entry
    out = a.out or os.path.join(a.runs_root, "scores.json")
    with open(out, "w") as f:
        json.dump(scores, f, indent=1)

    # Table 2 layout: self cells report PSNR, transfer cells report direction
    print("\n" + " | ".join(["world model".ljust(26)] + [c.replace("_", " ") for c in a.cells]))
    for arm in a.arms:
        vals = []
        for c in a.cells:
            e = scores["cells"].get(c, {}).get(arm)
            if e is None:
                vals.append("-")
            elif c.endswith("_transfer"):
                vals.append("-" if e.get("direction") is None else f"{e['direction']:.3f}")
            else:
                vals.append(f"{e['psnr']:.2f}")
        print(" | ".join([ARM_NAMES.get(arm, arm).ljust(26)] + vals))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

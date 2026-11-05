"""Score the generated test rollouts of one domain (paper Table 3).

Reads <cells>/<arm>_step<n>/{results.jsonl, preds/} written by generate.py and writes one JSON with, per cell:
  psnr                 PSNR (dB) of the commanded action's frame against the real frame 6 (mean over cases);
  m3_margin            Control, discrete domains: commanded-action PSNR minus the mean PSNR of the other actions;
  mc4_psnr             Control, nuScenes: commanded-action PSNR minus the PSNR of four random training-clip actions;
  m1_action_following  Follow, discrete domains: fraction of (case, generated action) pairs that the inverse-dynamics
                       classifier, applied to (real frame 5, generated frame), assigns to the generated action (chance 1/3);
  follow               Follow, nuScenes: per-case Spearman correlation between the commanded forward displacements of the
                       five generated actions and the readout's displacement from (real frame 5, generated frame) (chance 0);
plus LPIPS and motion-region PSNR, and per number of updates the contrasts OnlyAct minus each baseline: paired bootstrap
over whole episodes (discrete: one case per episode; nuScenes: driving logs), 20 000 draws, seed 20261020, percentile
95 % interval. "ceiling" (discrete) = the classifier's accuracy on (real frame 5, real frame 6) of the test cases.

  python -m downstream.score --config downstream/configs/dmlab.json --cells runs/table3/cells/dmlab --data_root DATA/dmlab \
      --out runs/table3/scores/dmlab.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from .common import ARMS, BASELINES, OURS, REPO, load_clips, load_config, write_json

DISCRETE_METRICS = ["psnr", "lpips", "m1", "motion_psnr", "m3"]
NUSCENES_METRICS = ["psnr", "lpips", "motion_psnr", "mc4_psnr", "mc4_lpips", "follow"]


def boot(a, b, eps, key, n=20000, seed=20261020):
    """Paired whole-episode bootstrap of mean(a[key] - b[key]); cases with a missing value in either arm are dropped for both."""
    d, e = [], []
    for x, y, ep in zip(a, b, eps):
        if x[key] is None or y[key] is None: continue
        d.append(x[key] - y[key]); e.append(ep)
    d, e = np.array(d), np.array(e); ue = np.unique(e); idx = {u: np.where(e == u)[0] for u in ue}
    rng = np.random.default_rng(seed); means = np.empty(n)
    for i in range(n):
        pick = rng.choice(ue, len(ue), replace=True); sel = np.concatenate([idx[u] for u in pick]); means[i] = d[sel].mean()
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], "n": int(len(d)), "episodes": int(len(ue))}


def find_cells(root):
    cells = {}
    for d in sorted(Path(root).iterdir()):
        m = re.fullmatch(r"(.+)_step(\d+)", d.name)
        if m and m.group(1) in ARMS and (d / "results.jsonl").exists():
            cells[(m.group(1), int(m.group(2)))] = d
    return cells


# ---------------------------------------------------------------- discrete domains (DMLab, Minecraft, Habitat)

def real_frames(cfg, tasks, cache=None, check=True):
    """Real frames 5 and 6 of every test case at 64 px, uint8 (area-downsampled from the generation-time 256 px frames)."""
    if cache and os.path.exists(cache):
        z = np.load(cache); assert list(z["task_ids"]) == [t["task_id"] for t in tasks]
        return torch.from_numpy(z["f5"]), torch.from_numpy(z["f6"])
    from .data import frame_loader
    load = frame_loader(cfg, check)
    q = lambda x: (torch.nn.functional.interpolate(x[None], size=64, mode="area")[0] * 255).round().clamp(0, 255).to(torch.uint8)
    f5, f6 = [], []
    for t in tasks:
        fr = load(t); f5.append(q(fr[5])); f6.append(q(fr[6]))
    f5, f6 = torch.stack(f5), torch.stack(f6)
    if cache:
        np.savez(cache, task_ids=np.array([t["task_id"] for t in tasks]), f5=f5.numpy(), f6=f6.numpy())
    return f5, f6


def up(x64):
    """uint8 64 px -> float 256 px (nearest); idm_input area-downsamples it back to the same 64 px values."""
    return torch.nn.functional.interpolate(x64.float()[None] / 255, size=256, mode="nearest")[0]


@torch.no_grad()
def score_discrete_cell(cell_dir, actions, idm, ids, f5, dev):
    from .idm import idm_input
    rows = [json.loads(l) for l in open(cell_dir / "results.jsonl")]
    assert [r["task_id"] for r in rows] == ids, f"case order differs in {cell_dir}"
    per = []
    for r in rows:
        P = torch.from_numpy(np.load(cell_dir / "preds" / f"{r['index']:03d}.npy")).float() / 255
        prev = up(f5[r["index"]]).expand(len(actions), -1, -1, -1)
        pred_cls = idm(idm_input(prev, P).to(dev)).argmax(1).cpu().numpy()
        correct = [int(pred_cls[j] == j) for j in range(len(actions))]
        others = [v for k, v in r["psnr_by_action"].items() if int(k) != r["raw_action"]]
        per.append({"episode_id": r["episode_id"], "psnr": r["psnr"], "lpips": r["lpips"], "motion_psnr": r["motion_psnr"],
                    "m1": float(np.mean(correct)), "conf": [int(c) for c in pred_cls], "m3_cmd": r["psnr_by_action"][str(r["raw_action"])],
                    "m3_other": float(np.mean(others)), "m3": r["psnr_by_action"][str(r["raw_action"])] - float(np.mean(others))})
    return per


def discrete_summary(P):
    mp = [p["motion_psnr"] for p in P if p["motion_psnr"] is not None]
    conf = np.zeros((3, 3), int)
    for p in P:
        for j, k in enumerate(p["conf"]): conf[j, k] += 1
    return {"psnr": float(np.mean([p["psnr"] for p in P])), "lpips": float(np.mean([p["lpips"] for p in P])),
            "m1_action_following": float(np.mean([p["m1"] for p in P])), "m1_confusion_rows_generated_cols_idm": conf.tolist(),
            "m2_motion_psnr": float(np.mean(mp)), "m2_excluded_cases": int(len(P) - len(mp)),
            "m3_cmd_psnr": float(np.mean([p["m3_cmd"] for p in P])), "m3_other_psnr": float(np.mean([p["m3_other"] for p in P])),
            "m3_margin": float(np.mean([p["m3"] for p in P]))}


def score_discrete(cfg, cells, tasks, classifier, dev, cache, check=True):
    from .idm import load_idm
    actions = [int(a) for a in cfg["actions"]]; ids = [t["task_id"] for t in tasks]
    idm = load_idm(classifier, len(actions), dev); f5, f6 = real_frames(cfg, tasks, cache, check)
    y = np.array([actions.index(int(t["raw_action"])) for t in tasks]); x = torch.cat([f5, f6], 1).float() / 255
    with torch.no_grad():
        pr = idm(torch.cat([x[:, :3], x[:, 3:], x[:, 3:] - x[:, :3]], 1).to(dev)).argmax(1).cpu().numpy()
    res = {"domain": cfg["domain"], "ceiling": {"idm_real_test_accuracy": float((pr == y).mean()), "n": int(len(y))}, "cells": {}, "contrasts": {}}
    per = {k: score_discrete_cell(d, actions, idm, ids, f5, dev) for k, d in cells.items()}
    for (arm, n), P in sorted(per.items()):
        res["cells"][f"{arm}/step{n}"] = discrete_summary(P)
    eps = [p["episode_id"] for p in next(iter(per.values()))]  # resampling unit: episode (one test case each)
    for n in sorted({n for _, n in per}):
        for other in BASELINES:
            A, B = per.get((OURS, n)), per.get((other, n))
            if A and B:
                res["contrasts"][f"step{n}/{OURS}-{other}"] = {k: boot(A, B, eps, k) for k in DISCRETE_METRICS}
    return res


# ---------------------------------------------------------------- nuScenes

def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def score_nuscenes_cell(cell_dir, tasks, net, res):
    from .readout import load_png, readout_input
    rows = [json.loads(l) for l in open(cell_dir / "results.jsonl")]
    assert [r["task_id"] for r in rows] == [t["task_id"] for t in tasks], f"case order differs in {cell_dir}"
    per = []
    for r, t in zip(rows, tasks):
        prev = (load_png(t["target_frames"]["abs_paths"][5]).float() / 255)[None]; P = torch.from_numpy(np.load(cell_dir / "preds" / f"{r['index']:03d}.npy")).float() / 255
        read = net(readout_input(prev.expand(len(P), -1, -1, -1), P, res))[:, 0].numpy(); cmd = [a[0] for a in r["generated_actions"]]
        follow = spearman(read, cmd) if len({round(x, 6) for x in cmd}) >= 3 else None
        per.append({"episode_id": r["log_token"], "case_id": r["task_id"], "psnr": r["psnr"], "lpips": r["lpips"], "motion_psnr": r["motion_psnr"],
                    "mc4_psnr": r["mc4"]["empirical_delta1_psnr_mc4"], "mc4_lpips": r["mc4"]["empirical_delta1_lpips_mc4"], "follow": follow,
                    "factual_read": float(read[0]), "factual_cmd": float(cmd[0])})
    return per


def nuscenes_summary(P):
    s = {m: float(np.mean([p[m] for p in P if p[m] is not None])) for m in NUSCENES_METRICS}
    s["follow_cases"] = sum(p["follow"] is not None for p in P)
    s["factual_readout_spearman_across_cases"] = spearman([p["factual_read"] for p in P], [p["factual_cmd"] for p in P])
    return s


def score_nuscenes(cfg, cells, tasks, readout):
    from .readout import load_readout
    net, res_px = load_readout(readout)
    res = {"domain": cfg["domain"], "cells": {}, "contrasts": {}}
    per = {k: score_nuscenes_cell(d, tasks, net, res_px) for k, d in cells.items()}
    for (arm, n), p in sorted(per.items()):
        res["cells"][f"{arm}/step{n}"] = nuscenes_summary(p)
    for n in sorted({n for _, n in per}):
        for other in BASELINES:
            A, B = per.get((OURS, n)), per.get((other, n))
            if A and B:
                eps = [p["episode_id"] for p in A]  # resampling unit: driving log
                res["contrasts"][f"step{n}/{OURS}-{other}"] = {m: boot(A, B, eps, m) for m in NUSCENES_METRICS}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--cells", required=True, help="folder of <arm>_step<n> cell folders")
    ap.add_argument("--data_root", required=True, help="folder the test list's clip / frame paths are relative to")
    ap.add_argument("--classifier", help="classifier / readout (default: the config's)")
    ap.add_argument("--real_frames", help="optional cache file of the real frames 5 / 6 at 64 px (discrete domains)")
    ap.add_argument("--no_check_hashes", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config); tasks = load_clips(cfg["test_list"], a.data_root); cells = find_cells(a.cells)
    ck = a.classifier or str(REPO / cfg["follow"]["checkpoint"])
    res = score_nuscenes(cfg, cells, tasks, ck) if cfg["action_type"] == "continuous" else score_discrete(cfg, cells, tasks, ck, a.device, a.real_frames, not a.no_check_hashes)
    write_json(a.out, res)
    for k, v in res["cells"].items():
        f = v.get("m1_action_following", v.get("follow")); c = v.get("m3_margin", v.get("mc4_psnr"))
        print(f"[{cfg['domain']}] {k:32s} PSNR {v['psnr']:.3f} | Follow {f:.3f} | Control {c:+.3f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

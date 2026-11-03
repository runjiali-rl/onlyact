"""Forward-motion readout for the nuScenes Follow metric of Table 3.

The continuous-action counterpart of the inverse-dynamics classifier (downstream/idm.py, same network with one output):
regresses the ego forward displacement (metres) of a transition from [frame t, frame t+1, frame t+1 - frame t] at 64 x 64
(256-pixel PNGs, area-downsampled, quantised to uint8). Trained only on real frames: every transition of the 100
fine-tuning clips (600 transitions); Adam 1e-3, batch 64, 40 epochs, seed 0, horizontal-flip augmentation (forward motion
is left-right symmetric), last epoch kept, on the CPU. The development check (Spearman correlation on the 480 transitions
of 20 held-out clips of other driving logs) is only logged.

  python -m downstream.readout --config downstream/configs/nuscenes.json --data_root DATA/nuscenes \
      --out checkpoints/downstream/readout_nuscenes.safetensors
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .common import load_clips, load_config
from .idm import IDM


def load_png(p):
    from PIL import Image
    with Image.open(p) as im:
        return torch.from_numpy(np.array(im.convert("RGB"), dtype=np.uint8)).permute(2, 0, 1)


def q(x, res):
    """(N, 3, 256, 256) float [0, 1] -> (N, 3, res, res) uint8, area average."""
    return (F.interpolate(x, size=res, mode="area") * 255).round().clamp(0, 255).to(torch.uint8)


def readout_input(f_t, f_t1, res):
    """f_t, f_t1: (N, 3, 256, 256) float in [0, 1] -> (N, 9, res, res) float, as the training data."""
    a, b = q(f_t, res).float() / 255, q(f_t1, res).float() / 255
    return torch.cat([a, b, b - a], 1)


def transitions(tasks, res):
    X, y = [], []
    for t in tasks:
        fr = torch.stack([load_png(p) for p in t["target_frames"]["abs_paths"]]).float() / 255
        n = len(t["actions_ego_xy"]); X.append(readout_input(fr[:n], fr[1:n + 1], res)); y += [a[0] for a in t["actions_ego_xy"]]
    return torch.cat(X), torch.tensor(y, dtype=torch.float32)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


def load_readout(path):
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(str(path), framework="pt") as f:
        res = int(f.metadata()["res"])
    net = IDM(1); net.load_state_dict(load_file(str(path))); net.eval()
    return net, res


def main():
    from safetensors.torch import save_file
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--data_root", required=True)
    ap.add_argument("--dev_data_root", help="folder of the development clips (default --data_root)")
    ap.add_argument("--res", type=int, default=64); ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--out", required=True)
    a = ap.parse_args(); torch.set_num_threads(8); cfg = load_config(a.config)
    torch.manual_seed(0)
    Xtr, ytr = transitions(load_clips(cfg["train_list"], a.data_root), a.res)
    Xd, yd = transitions(load_clips(cfg["readout"]["development_list"], a.dev_data_root or a.data_root), a.res)
    net = IDM(1); opt = torch.optim.Adam(net.parameters(), 1e-3); g = torch.Generator().manual_seed(0); log = []
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]; x = Xtr[idx]
            flip = torch.rand(len(idx), generator=g) < 0.5; x = torch.where(flip[:, None, None, None], x.flip(-1), x)
            loss = F.mse_loss(net(x)[:, 0], ytr[idx]); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): pd = net(Xd)[:, 0].numpy(); pt = net(Xtr)[:, 0].numpy()
        log.append({"epoch": ep + 1, "train_spearman": spearman(pt, ytr.numpy()), "development_spearman": spearman(pd, yd.numpy()),
                    "development_mae_m": float(np.abs(pd - yd.numpy()).mean())}); print(json.dumps(log[-1]), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_file({k: v.contiguous() for k, v in net.state_dict().items()}, a.out, metadata={"res": str(a.res), "epochs": str(a.epochs)})
    with open(os.path.splitext(a.out)[0] + "_train_log.json", "w") as f:
        json.dump({"res": a.res, "epochs": a.epochs, "train_transitions": len(Xtr), "development_transitions": len(Xd), "log": log}, f, indent=1)


if __name__ == "__main__":
    main()

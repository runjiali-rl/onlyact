"""Inverse-dynamics classifier for the Follow metric of Table 3 (DMLab, Minecraft, Habitat).

Trained only on real frames and then used as a fixed measuring instrument for every arm. Input: [frame t, frame t+1,
frame t+1 - frame t] at 64 x 64 (frames go through the same 256-pixel preprocessing as generation, are area-downsampled
to 64 and quantised to uint8); network: 4 stride-2 3x3 convolutions (32, 64, 128, 128 channels, ReLU), global average
pooling, linear layer. Adam (learning rate 1e-3), batch 256, 20 epochs, seed 0, last epoch kept; 10 % of the episodes are
held out for the reported accuracy.
Training transitions:
  DMLab     every transition of the 400 training-split episodes of the TECO DMLab pool (data/downstream/dmlab/idm_episodes.json);
  Minecraft every transition of the 300 labelled fine-tuning clips;
  Habitat   every transition of the 300 labelled fine-tuning clips.

  python -m downstream.idm --config downstream/configs/dmlab.json --data_root DATA/dmlab_pool --out checkpoints/downstream/idm_dmlab.safetensors
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import load_clips, load_config, repo_path


class IDM(nn.Module):
    def __init__(self, n_outputs):
        super().__init__()
        ch = [9, 32, 64, 128, 128]
        self.convs = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1) for i in range(4)])
        self.head = nn.Linear(128, n_outputs)

    def forward(self, x):
        for c in self.convs:
            x = F.relu(c(x))
        return self.head(x.mean(dim=(2, 3)))


def to64(x256):
    """(N, 3, 256, 256) float in [0, 1] -> (N, 3, 64, 64) by area averaging."""
    return F.interpolate(x256, size=64, mode="area")


def from_uint8(xb):
    """(B, 6, 64, 64) uint8 [frame t, frame t+1] -> (B, 9, 64, 64) float [t, t+1, t+1 - t]."""
    x = xb.float() / 255; a, b = x[:, :3], x[:, 3:]
    return torch.cat([a, b, b - a], dim=1)


def idm_input(f_t, f_t1):
    """f_t, f_t1: (N, 3, 256, 256) float in [0, 1] -> (N, 9, 64, 64), quantised at 64 px exactly as the training data."""
    q = lambda x: (to64(x) * 255).round().clamp(0, 255).to(torch.uint8)
    return from_uint8(torch.cat([q(f_t), q(f_t1)], dim=1))


def load_idm(path, n_actions, dev="cuda"):
    from safetensors.torch import load_file
    net = IDM(n_actions); net.load_state_dict(load_file(str(path))); return net.to(dev).eval()


def _clip_transitions(tasks):
    for t in tasks:
        with np.load(t["clip"]["abs_path"]) as z:
            v, a = z["video"], z["actions_transition"]
        yield t["episode_id"], v, np.asarray(a)


def _episode_transitions(episodes, root):
    for rel in episodes:
        with np.load(os.path.join(root, rel)) as z:
            v, a = z["video"], z["actions"]
        yield rel, v, a[:-1]


def _png_transitions(tasks):
    from PIL import Image
    for t in tasks:
        frames = np.stack([np.array(Image.open(p).convert("RGB")) for p in t["target_frames"]["abs_paths"]])
        yield t["episode_id"], frames, np.asarray(t["actions_transition"])


def build_arrays(cfg, data_root, valid_actions):
    """X (N, 6, 64, 64) uint8 = [frame t, frame t+1], labels (index into valid_actions), episode of each transition."""
    from .data import preprocess_rgb256
    src = cfg["idm"]["train_source"]
    if src == "episodes":
        with open(repo_path(cfg["idm"]["episode_list"])) as f:
            source = _episode_transitions(json.load(f)["episodes"], data_root)
    elif src == "clips":
        source = _clip_transitions(load_clips(cfg["train_list"], data_root))
    else:
        source = _png_transitions(load_clips(cfg["train_list"], data_root))
    xs, ys, eps = [], [], []
    for ep, video, acts in source:
        if src == "png":
            f256 = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255
        else:
            f256 = torch.stack([preprocess_rgb256(np.ascontiguousarray(f)) for f in video])
        f64 = (to64(f256) * 255).round().clamp(0, 255).to(torch.uint8)
        x = torch.cat([f64[:-1], f64[1:]], dim=1)
        keep = np.isin(acts, valid_actions)
        xs.append(x[torch.from_numpy(keep)]); ys.append(torch.from_numpy(np.searchsorted(valid_actions, acts[keep])))
        eps += [ep] * int(keep.sum())
    return torch.cat(xs), torch.cat(ys).long(), np.array(eps)


@torch.no_grad()
def accuracy(net, X, Y, dev):
    net.eval(); c = 0
    for i in range(0, len(Y), 1024):
        c += int((net(from_uint8(X[i:i + 1024]).to(dev)).argmax(1).cpu() == Y[i:i + 1024]).sum())
    return c / max(1, len(Y))


def train(cfg, data_root, out, dev):
    from safetensors.torch import save_file
    valid_actions = np.array(sorted(int(x) for x in cfg["actions"]))
    torch.manual_seed(0); np.random.seed(0)
    t0 = time.time(); X, Y, E = build_arrays(cfg, data_root, valid_actions)
    ue = np.unique(E); rng = np.random.default_rng(0); hold = set(rng.choice(ue, max(1, len(ue) // 10), replace=False).tolist())
    hmask = torch.from_numpy(np.array([e in hold for e in E]))
    Xtr, Ytr, Xho, Yho = X[~hmask], Y[~hmask], X[hmask], Y[hmask]
    net = IDM(len(valid_actions)).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    log = {"domain": cfg["domain"], "valid_actions": valid_actions.tolist(), "n_train": int(len(Ytr)), "n_holdout": int(len(Yho)),
           "class_counts_train": np.bincount(Ytr.numpy(), minlength=len(valid_actions)).tolist(), "episodes": int(len(ue)), "epochs": []}
    for ep in range(20):
        net.train(); perm = torch.randperm(len(Ytr))
        for i in range(0, len(perm), 256):
            j = perm[i:i + 256]; xb, yb = from_uint8(Xtr[j]).to(dev), Ytr[j].to(dev)
            loss = F.cross_entropy(net(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
        acc = accuracy(net, Xho, Yho, dev); log["epochs"].append({"epoch": ep + 1, "holdout_acc": acc, "last_loss": float(loss)})
        print(f"[idm] {cfg['domain']} epoch {ep + 1}: hold-out accuracy {acc:.3f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}, out)
    log["seconds"] = time.time() - t0
    with open(os.path.splitext(out)[0] + "_train_log.json", "w") as f:
        json.dump(log, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--data_root", required=True,
                    help="DMLab: root of the TECO DMLab episodes; Minecraft / Habitat: root of the labelled training clips")
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); assert not os.path.exists(a.out), "refusing to overwrite"
    train(load_config(a.config), a.data_root, a.out, a.device)


if __name__ == "__main__":
    main()

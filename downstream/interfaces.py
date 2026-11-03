"""Step-0 action interfaces and the standardisation statistics for Table 3.

Step 0 ("0 updates") uses no gradient update of the world model but does use the labelled clips:
  book  (discrete domains): for each labelled training clip, the arm's action feature of the last transition (row 5 of
        the per-clip (6, d) feature file written by the arm's extractor) is passed once through the pretrained world
        model's own action input transform; the book row of an action is the mean over its 100 clips.
  head  (nuScenes): Linear(2, d) - SiLU - Linear(d, d) fitted by 3000 SGD steps (learning rate 0.01, batch 10, mean-squared
        error) from the clip's (forward, lateral) ego motion to the same transformed feature, on the 100 training clips.
  standardise: per-dimension mean / std (float64) of the arm's world-model training features after the input transform
        (used by the standardised interface of DMLab / Minecraft).

  python -m downstream.interfaces book --config downstream/configs/dmlab.json --arm justact --features EMB/justact/dmlab_train \
      --out checkpoints/downstream/step0/dmlab_justact.safetensors
  python -m downstream.interfaces head --config downstream/configs/nuscenes.json --arm justact --features EMB/justact/nuscenes_train \
      --out checkpoints/downstream/step0/nuscenes_justact.safetensors
  python -m downstream.interfaces standardise --arm justact --pool_list POOL.json --features EMB/justact/pool \
      --out checkpoints/downstream/standardise/justact.safetensors
(--features: folder of <task_id>.npy, (6, d) float16 per clip, from the arm's extractor; the pool is the world model's
training clip list of Table 2; the whitening file must hold "mean", "std" and "transform".)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .common import REPO, index_plan, load_clips, load_config, seed_all

CKPT = REPO / "checkpoints" / "downstream"


def raw_features(tasks, features):
    """Row 5 (the last transition) of every clip's (6, d) feature file, in clip-list order, as float32."""
    return np.stack([np.array(np.load(os.path.join(features, t["task_id"] + ".npy"), allow_pickle=False)[5], dtype=np.float32) for t in tasks])


def transformed(raw, embedder):
    with torch.no_grad():
        tokens, final = embedder(torch.as_tensor(raw, dtype=torch.float32))
    assert final.shape == raw.shape and torch.equal(tokens[:, 0], final)
    return final.detach()


def step0_book(final, labels, actions):
    labels = torch.as_tensor(labels)
    return torch.stack([final[labels == a].mean(0) for a in actions])


def fit_head(actions_xy, targets, dim, seed=32, steps=3000, batch=10, lr=0.01):
    """Returns the fitted head state dict (CPU, FP32)."""
    from .model import ActionHead
    seed_all(seed); head = ActionHead(dim); opt = torch.optim.SGD(head.parameters(), lr=lr)
    plan = index_plan(len(actions_xy), steps, batch, seed, "init_order")
    for indices in plan:
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(head(actions_xy[indices])[1], targets[indices])
        loss.backward(); opt.step()
    return head.state_dict()


def load_step0(cfg, arm, path=None):
    from safetensors.torch import load_file
    path = path or CKPT / "step0" / f"{cfg['domain']}_{arm}.safetensors"
    state = load_file(str(path))
    return state["book"] if cfg["action_type"] == "discrete" else state


def load_standardise(arm, path=None):
    from safetensors.torch import load_file
    st = load_file(str(path or CKPT / "standardise" / f"{arm}.safetensors"))
    return {"mean": st["mean"], "std": st["std"]}


def load_whitening(path):
    from safetensors.torch import load_file
    return {k: v.numpy() for k, v in load_file(str(path)).items()}


def standardise_stats(rows_iter, whitening=None):
    """Two passes in float64: mean, then per-dimension variance, of the transformed pool features."""
    if whitening is not None:
        mu, sd, W = (np.asarray(whitening[k], dtype=np.float64) for k in ("mean", "std", "transform"))
        sd = np.where(sd < 1e-6, 1.0, sd); inj = lambda X: ((X - mu) / sd) @ W
    else:
        inj = lambda X: X
    s, n = None, 0
    for X in rows_iter():
        X = inj(X.astype(np.float64)); s = X.sum(0) if s is None else s + X.sum(0); n += len(X)
    m = s / n; v = np.zeros_like(m)
    for X in rows_iter():
        X = inj(X.astype(np.float64)); v += ((X - m) ** 2).sum(0)
    return m, np.sqrt(v / n)


def main():
    from safetensors.torch import save_file
    from .model import original_action_embedder, world_model_files
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["book", "head", "standardise"])
    ap.add_argument("--config"); ap.add_argument("--arm", required=True); ap.add_argument("--features", required=True)
    ap.add_argument("--wm_config", help="default: worldmodel/configs/<world model>.yaml")
    ap.add_argument("--whitening_stats", help="JustAct only; default: checkpoints/worldmodel/justact_whitening.safetensors")
    ap.add_argument("--data_root", default="."); ap.add_argument("--pool_list")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); torch.set_num_threads(4)
    if a.what == "standardise":
        tasks = load_clips(a.pool_list, a.data_root)
        rows = lambda: (np.load(os.path.join(a.features, t["task_id"] + ".npy")) for t in tasks)
        wpath = a.whitening_stats or REPO / "checkpoints" / "worldmodel" / "justact_whitening.safetensors"
        m, sd = standardise_stats(rows, load_whitening(wpath) if a.arm == "justact" else None)
        save_file({"mean": torch.from_numpy(m), "std": torch.from_numpy(sd)}, a.out); return
    cfg = load_config(a.config); dim = cfg["arms"][a.arm]["native_dim"]
    tasks = load_clips(cfg["train_list"], a.data_root)
    files = world_model_files(cfg, a.arm, wm_config=a.wm_config, whitening_stats=a.whitening_stats)
    final = transformed(raw_features(tasks, a.features), original_action_embedder(files, dim))
    if a.what == "book":
        save_file({"book": step0_book(final, [t["raw_action"] for t in tasks], cfg["actions"]).contiguous()}, a.out)
    else:
        xy = torch.tensor([t["raw_action"] for t in tasks], dtype=torch.float32)
        save_file({k: v.contiguous() for k, v in fit_head(xy, final.contiguous(), dim).items()}, a.out)


if __name__ == "__main__":
    main()

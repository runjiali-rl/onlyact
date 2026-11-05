"""Linear probe of action information (Table 1).

A linear layer is trained to predict the ground-truth action of each frame pair from a frozen representation
(continuous actions: mean-squared error on z-scored targets, reported as test R^2; Procgen: 15-way cross-entropy,
reported as macro-F1). Inputs are z-scored with statistics of a 20,000-sample training draw. The held-out split is a
deterministic function of the clip list and --split_seed; --seed reseeds every random generator before the fit, so
the result of one representation does not depend on anything probed before it.

    python -m probe.probe --dataset libero --clips data/probe/libero.json \
        --actions_dir $EVAL_DATA/probe/libero/actions --emb_dir $EMB/onlyact/libero \
        --seed 0 --output results/table1/onlyact/libero_seed0.json
"""
import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.join(HERE, "..", "data", "probe", "datasets.json")

EE7 = ["ee_dx", "ee_dy", "ee_dz", "ee_rx", "ee_ry", "ee_rz", "gripper"]
CAM6 = ["cam_tx", "cam_ty", "cam_tz", "cam_rx", "cam_ry", "cam_rz"]
DIM_NAMES = {  # names of the action dimensions in the result files
    "libero": EE7, "droid": EE7, "rlbench": EE7, "utaustin_mutex": EE7,
    "robocasa": ["ee_dx", "ee_dy", "ee_dz", "ee_drx", "ee_dry", "ee_drz", "gripper"],
    "metaworld": ["ee_dx", "ee_dy", "ee_dz", "gripper"],
    "vp2_robosuite": ["ee_dx", "ee_dy", "ee_dz", "gripper"],
    "vp2_robodesk": ["a0", "a1", "a2", "a3", "a4"],
    "realestate10k": CAM6, "spatialvid_nav": CAM6, "arkitscenes": CAM6,
}


def build_index(clips_path, actions_dir, emb_dir, task_type, scene_field="scene_id"):
    """Flat list of (file_idx, pair_idx, action, task_id, scene) over all tasks that have an embedding and an action file.
    The action of pair j is the action at the pair's first frame (actions are indexed by the clip's frame_indices)."""
    with open(clips_path) as f:
        tasks = json.load(f)["tasks"]
    file_paths, entries = [], []
    for task in tqdm(tasks, desc="  indexing"):
        task_id = task["task_id"]
        act_path = os.path.join(actions_dir, f"{task_id}.npy")
        emb_path = os.path.join(emb_dir, f"{task_id}.npy")
        if not os.path.exists(act_path) or not os.path.exists(emb_path):
            continue
        try:
            n_emb = len(np.load(emb_path, mmap_mode="r"))
        except Exception:
            continue
        actions_arr = np.load(act_path)
        if task_type == "discrete":
            pair_actions = actions_arr.astype(np.int64)[:-1]
        else:
            actions_arr = actions_arr.astype(np.float32)
            frame_indices = task.get("target_frames", {}).get("frame_indices", list(range(len(actions_arr))))
            pair_actions = actions_arr[frame_indices][:-1]
        n_pairs = min(n_emb, len(pair_actions))
        if n_pairs == 0:
            continue
        file_idx = len(file_paths)
        file_paths.append(emb_path)
        scene_id = task.get(scene_field) or task.get("scene_id", "")
        for j in range(n_pairs):
            entries.append((file_idx, j, pair_actions[j], task_id, scene_id))
    emb_dim = int(np.load(file_paths[0]).shape[-1]) if file_paths else 0
    return {"file_paths": file_paths, "entries": entries, "emb_dim": emb_dim}


class EmbeddingDataset(Dataset):
    """Streams embeddings from .npy files using mmap. Caches loaded files."""

    def __init__(self, file_paths, entries, emb_dim, task_type, cache_size=200):
        self.file_paths = file_paths
        self.entries = entries  # list of (file_idx, frame_idx, action, task_id, scene_id)
        self.emb_dim = emb_dim
        self.task_type = task_type
        # LRU cache: file_idx → loaded array
        self._cache = {}
        self._cache_order = []
        self.cache_size = cache_size

    def __len__(self):
        return len(self.entries)

    def _get_file(self, file_idx):
        if file_idx in self._cache:
            return self._cache[file_idx]
        # Load via mmap (no copy)
        arr = np.load(self.file_paths[file_idx], mmap_mode="r")
        self._cache[file_idx] = arr
        self._cache_order.append(file_idx)
        if len(self._cache_order) > self.cache_size:
            old_idx = self._cache_order.pop(0)
            del self._cache[old_idx]
        return arr

    def __getitem__(self, idx):
        file_idx, frame_idx, action, task_id, scene_id = self.entries[idx]

        arr = self._get_file(file_idx)
        emb = np.array(arr[frame_idx], dtype=np.float32)

        if self.task_type == "discrete":
            action_t = torch.tensor(action, dtype=torch.long)
        else:
            action_t = torch.tensor(action, dtype=torch.float32)

        return torch.from_numpy(emb), action_t




def reseed_all(seed):
    """Reseed every generator the probe draws from: Python `random`, the numpy global generator, the torch CPU
    generator and all CUDA generators. The RandomSampler generator and the DataLoader worker base seed of every
    iterator are drawn from the torch default generator, so they follow."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




def stratified_split(entries, test_fraction=0.2, seed=42, split_mode="demo_within_scene"):
    """demo_within_scene: hold out test_fraction of the demonstrations inside every scene.
    scene: hold out test_fraction of whole scenes (episodes, clips, tasks or kitchens, depending on the dataset)."""
    rng = np.random.RandomState(seed)

    # Use scene_id if available
    has_scene = any(e[4] for e in entries[:100])
    if has_scene:
        scene_to_tids = defaultdict(set)
        for e in entries:
            scene_to_tids[e[4]].add(e[3])
    else:
        scene_to_tids = defaultdict(set)
        for e in entries:
            tid = e[3]
            parts = tid.rsplit("_demo", 1)
            if len(parts) > 1:
                key = parts[0]
            else:
                m = re.match(r"^(.*)_\d+$", tid)
                key = m.group(1) if m else tid
            scene_to_tids[key].add(tid)

    test_tasks = set()
    if split_mode == "scene":
        scenes = sorted(scene_to_tids)
        rng.shuffle(scenes)
        n_test_scenes = max(1, int(len(scenes) * test_fraction))
        for sc in scenes[:n_test_scenes]:
            test_tasks.update(scene_to_tids[sc])
    else:
        for scene, tids in scene_to_tids.items():
            tids_list = sorted(tids)
            rng.shuffle(tids_list)
            n_test = max(1, int(len(tids_list) * test_fraction))
            test_tasks.update(tids_list[:n_test])

    train_indices = []
    test_indices = []
    for i, e in enumerate(entries):
        if e[3] in test_tasks:
            test_indices.append(i)
        else:
            train_indices.append(i)
    return np.array(train_indices), np.array(test_indices)




def train_continuous_probe(file_paths, entries, emb_dim, train_indices, test_indices,
                           device="cuda", epochs=50, lr=1e-2, weight_decay=1e-3,
                           batch_size=4096, num_workers=4, dim_names=None, y_std_floor=0.01,
                           drop_dead_dims=True):
    """Train linear regression probe with DataLoader.

    y_std_floor: action dimensions whose standard deviation (on a 20,000-sample training draw) is below this are
        constant in the data and are excluded from training and reporting.
    """
    train_ds = EmbeddingDataset(file_paths, [entries[i] for i in train_indices],
                                 emb_dim, task_type="continuous")
    test_ds = EmbeddingDataset(file_paths, [entries[i] for i in test_indices],
                                emb_dim, task_type="continuous")

    # Compute normalization on a sample
    print(f"  Computing normalization stats...")
    sample_loader = DataLoader(train_ds, batch_size=2048, shuffle=True,
                                num_workers=num_workers, persistent_workers=False)
    sample_embs = []
    sample_targets = []
    for x, y in sample_loader:
        sample_embs.append(x)
        sample_targets.append(y)
        if sum(s.shape[0] for s in sample_embs) >= 20000:
            break
    sample_embs = torch.cat(sample_embs)[:20000]
    sample_targets = torch.cat(sample_targets)[:20000]
    x_mean = sample_embs.mean(0).to(device)
    x_std = sample_embs.std(0).to(device) + 1e-6

    full_out_dim = sample_targets.shape[1] if sample_targets.ndim > 1 else 1
    raw_y_std = sample_targets.std(0) if sample_targets.ndim > 1 else sample_targets.std()
    # Identify "live" dims: std above the floor
    if sample_targets.ndim > 1 and drop_dead_dims:
        live_mask = (raw_y_std.float() >= y_std_floor)
        live_idx = live_mask.nonzero(as_tuple=True)[0].cpu().numpy().tolist()
        dropped_idx = [i for i in range(full_out_dim) if i not in live_idx]
        if dropped_idx:
            print(f"  Dropping {len(dropped_idx)}/{full_out_dim} low-variance dims "
                  f"(std<{y_std_floor}): {dropped_idx}")
        sample_targets = sample_targets[:, live_idx]
    else:
        live_idx = list(range(full_out_dim))
        dropped_idx = []

    y_mean = sample_targets.mean(0).to(device)
    # Floor std so that any lingering low-variance dim cannot blow up normalization
    y_std = torch.clamp(sample_targets.std(0).float(), min=y_std_floor).to(device)

    out_dim = sample_targets.shape[1] if sample_targets.ndim > 1 else 1
    live_idx_t = torch.tensor(live_idx, dtype=torch.long, device=device)

    model = nn.Linear(emb_dim, out_dim).to(device)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop with DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True,
                               persistent_workers=num_workers > 0)

    def _slice_live(y):
        return y.index_select(-1, live_idx_t) if y.ndim > 1 and len(live_idx) < full_out_dim else y

    loss_history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            yb = _slice_live(yb)
            xb_norm = (xb - x_mean) / x_std
            yb_norm = (yb - y_mean) / y_std
            pred = model(xb_norm)
            loss = F.mse_loss(pred, yb_norm)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"    epoch {epoch+1:3d}/{epochs}: train_loss = {avg_loss:.6f}")

    # Eval
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)

    def eval_loop(loader, max_n=None):
        preds, targs = [], []
        n = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = _slice_live(yb.to(device, non_blocking=True))
                xb_norm = (xb - x_mean) / x_std
                pred_norm = model(xb_norm)
                pred = pred_norm * y_std + y_mean
                preds.append(pred.cpu().numpy())
                targs.append(yb.cpu().numpy())
                n += xb.shape[0]
                if max_n and n >= max_n:
                    break
        return np.concatenate(preds), np.concatenate(targs)

    print(f"  Evaluating test...")
    y_pred_test, y_test_raw = eval_loop(test_loader)
    print(f"  Evaluating train (subsample)...")
    y_pred_train, y_train_eval = eval_loop(train_eval_loader, max_n=200000)

    def r2_np(yt, yp):
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean(0)) ** 2)
        return 1 - ss_res / max(ss_tot, 1e-12)

    def mse_np(yt, yp):
        return float(np.mean((yt - yp) ** 2))

    def r2_macro_np(yt, yp, min_std=y_std_floor):
        """Average per-dim R², skipping dims whose ground-truth std in `yt`
        is below `min_std` (unstable / degenerate dim). Robust to heterogeneous
        action spaces where a few dims have zero test-split variance."""
        if yt.ndim == 1:
            return r2_np(yt, yp)
        per = []
        kept, skipped = [], []
        for i in range(yt.shape[1]):
            if yt[:, i].std() < min_std:
                skipped.append(i)
                continue
            per.append(r2_np(yt[:, i], yp[:, i]))
            kept.append(i)
        return (float(np.mean(per)) if per else 0.0, kept, skipped)

    r2_test_micro_raw = r2_np(y_test_raw, y_pred_test)
    mse_test = mse_np(y_test_raw, y_pred_test)
    r2_train_micro_raw = r2_np(y_train_eval, y_pred_train)
    mse_train = mse_np(y_train_eval, y_pred_train)
    r2_test_macro, macro_kept, macro_skipped = r2_macro_np(y_test_raw, y_pred_test)
    r2_train_macro, _, _ = r2_macro_np(y_train_eval, y_pred_train)
    # Reported R^2: variance-weighted ("micro") R^2 over the dimensions with non-degenerate test variance.
    def _safe_micro(yt, yp, kept):
        if yt.ndim == 1 or not kept:
            return r2_np(yt, yp)
        yt_k = yt[:, kept]; yp_k = yp[:, kept]
        return r2_np(yt_k, yp_k)
    r2_test_micro = _safe_micro(y_test_raw, y_pred_test, macro_kept)
    r2_train_micro = _safe_micro(y_train_eval, y_pred_train, macro_kept)
    r2_test = r2_test_micro
    r2_train = r2_train_micro

    default_libero = ["ee_dx", "ee_dy", "ee_dz", "ee_rx", "ee_ry", "ee_rz", "gripper"]
    names_full = list(dim_names) if dim_names else default_libero
    # Pad / truncate to full_out_dim
    if len(names_full) < full_out_dim:
        names_full = names_full + [f"dim{i}" for i in range(len(names_full), full_out_dim)]
    elif len(names_full) > full_out_dim:
        names_full = names_full[:full_out_dim]
    names_live = [names_full[i] for i in live_idx]
    per_dim = {}
    if y_test_raw.ndim > 1:
        for i, name in enumerate(names_live):
            per_dim[name] = {
                "r2": round(float(r2_np(y_test_raw[:, i], y_pred_test[:, i])), 4),
                "mse": round(float(mse_np(y_test_raw[:, i], y_pred_test[:, i])), 6),
                "r2_train": round(float(r2_np(y_train_eval[:, i], y_pred_train[:, i])), 4),
                "mse_train": round(float(mse_np(y_train_eval[:, i], y_pred_train[:, i])), 6),
            }

    return {
        "r2": round(float(r2_test), 4),
        "r2_macro": round(float(r2_test_macro), 4),
        "r2_micro": round(float(r2_test_micro), 4),
        "r2_micro_raw": round(float(r2_test_micro_raw), 4),
        "mse": round(float(mse_test), 6),
        "r2_train": round(float(r2_train), 4),
        "r2_train_macro": round(float(r2_train_macro), 4),
        "r2_train_micro": round(float(r2_train_micro), 4),
        "r2_train_micro_raw": round(float(r2_train_micro_raw), 4),
        "mse_train": round(float(mse_train), 6),
        "n_train": int(len(train_indices)),
        "n_test": int(len(test_indices)),
        "embedding_dim": int(emb_dim),
        "full_out_dim": int(full_out_dim),
        "live_dims": [int(i) for i in live_idx],
        "dropped_dims": [int(i) for i in dropped_idx],
        "macro_kept": [int(i) for i in macro_kept],
        "macro_skipped": [int(i) for i in macro_skipped],
        "per_dim": per_dim,
        "loss_history": [round(l, 6) for l in loss_history],
    }




def train_discrete_probe(file_paths, entries, emb_dim, train_indices, test_indices,
                         device="cuda", epochs=50, lr=1e-2, weight_decay=1e-3,
                         batch_size=4096, num_workers=4):
    """Train logistic regression probe with DataLoader."""
    train_ds = EmbeddingDataset(file_paths, [entries[i] for i in train_indices],
                                 emb_dim, task_type="discrete")
    test_ds = EmbeddingDataset(file_paths, [entries[i] for i in test_indices],
                                emb_dim, task_type="discrete")

    # Sample for normalization
    print(f"  Computing normalization stats...")
    sample_loader = DataLoader(train_ds, batch_size=2048, shuffle=True,
                                num_workers=num_workers)
    sample_embs = []
    for x, _ in sample_loader:
        sample_embs.append(x)
        if sum(s.shape[0] for s in sample_embs) >= 20000:
            break
    sample_embs = torch.cat(sample_embs)[:20000]
    x_mean = sample_embs.mean(0).to(device)
    x_std = sample_embs.std(0).to(device) + 1e-6

    n_classes = int(max(e[2] for e in entries) + 1)
    n_classes = max(n_classes, 2)

    model = nn.Linear(emb_dim, n_classes).to(device)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True,
                               persistent_workers=num_workers > 0)

    loss_history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            xb_norm = (xb - x_mean) / x_std
            logits = model(xb_norm)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"    epoch {epoch+1:3d}/{epochs}: train_loss = {avg_loss:.6f}")

    # Eval
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)

    def eval_loop(loader, max_n=None):
        preds, targs = [], []
        n = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                xb_norm = (xb - x_mean) / x_std
                logits = model(xb_norm)
                preds.append(logits.argmax(-1).cpu().numpy())
                targs.append(yb.numpy())
                n += xb.shape[0]
                if max_n and n >= max_n:
                    break
        return np.concatenate(preds), np.concatenate(targs)

    print(f"  Evaluating...")
    y_pred_test, y_test = eval_loop(test_loader)
    y_pred_train, y_train_eval = eval_loop(train_eval_loader, max_n=200000)

    acc = float((y_pred_test == y_test).mean())
    train_acc = float((y_pred_train == y_train_eval).mean())

    classes = np.unique(np.concatenate([y_test, y_pred_test]))
    f1_per_class = []
    for c in classes:
        tp = float(((y_pred_test == c) & (y_test == c)).sum())
        fp = float(((y_pred_test == c) & (y_test != c)).sum())
        fn = float(((y_pred_test != c) & (y_test == c)).sum())
        p = tp / max(tp + fp, 1e-12)
        r = tp / max(tp + fn, 1e-12)
        f1_per_class.append(2 * p * r / max(p + r, 1e-12))
    f1_macro = float(np.mean(f1_per_class))

    train_actions = np.array([entries[i][2] for i in train_indices])
    mode_action = Counter(train_actions).most_common(1)[0][0]
    chance = float((y_test == mode_action).mean())

    # Per-scene
    test_scenes = np.array([entries[i][4] for i in test_indices[:len(y_test)]])
    per_scene = {}
    for scene in sorted(set(test_scenes)):
        m = test_scenes == scene
        if m.sum() < 5:
            continue
        per_scene[scene] = {
            "acc": round(float((y_pred_test[m] == y_test[m]).mean()), 4),
            "n_samples": int(m.sum()),
        }

    return {
        "accuracy": round(acc, 4),
        "train_accuracy": round(train_acc, 4),
        "f1_macro": round(f1_macro, 4),
        "chance_baseline": round(chance, 4),
        "improvement_over_chance": round(acc - chance, 4),
        "n_train": int(len(train_indices)),
        "n_test": int(len(test_indices)),
        "n_classes": int(n_classes),
        "embedding_dim": int(emb_dim),
        "per_scene": per_scene,
        "loss_history": [round(l, 6) for l in loss_history],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="key of data/probe/datasets.json (sets target, task and split)")
    ap.add_argument("--clips", required=True, help="clip list, e.g. data/probe/<dataset>.json")
    ap.add_argument("--actions_dir", required=True, help="folder with <task_id>.npy action targets")
    ap.add_argument("--emb_dir", required=True, help="folder with <task_id>.npy embeddings of one representation")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0, help="probe seed (the paper uses 0, 1, 2)")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = json.load(open(DATASETS))[args.dataset]

    reseed_all(args.seed)
    idx = build_index(args.clips, args.actions_dir, args.emb_dir, cfg["task"], scene_field=cfg["scene_field"])
    if not idx["entries"]:
        raise SystemExit(f"no data for {args.emb_dir}")
    print(f"  {len(idx['entries'])} samples, dim={idx['emb_dim']}, {len(idx['file_paths'])} files")
    train_indices, test_indices = stratified_split(idx["entries"], seed=args.split_seed, split_mode=cfg["split_mode"])
    print(f"  Train: {len(train_indices)}, Test: {len(test_indices)}")
    common = dict(device=args.device, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                  batch_size=args.batch_size, num_workers=args.num_workers)
    if cfg["task"] == "continuous":
        result = train_continuous_probe(idx["file_paths"], idx["entries"], idx["emb_dim"], train_indices, test_indices,
                                        dim_names=DIM_NAMES.get(cfg["action_space"]), y_std_floor=cfg["y_std_floor"], **common)
        print(f"  Train R2={result['r2_train']:.4f}, Test R2={result['r2']:.4f}")
    else:
        result = train_discrete_probe(idx["file_paths"], idx["entries"], idx["emb_dim"], train_indices, test_indices, **common)
        print(f"  Test accuracy={result['accuracy']:.4f}, macro-F1={result['f1_macro']:.4f}, chance={result['chance_baseline']:.4f}")
    result["probe"] = {"dataset": args.dataset, "target": cfg["target"], "seed": args.seed, "split_seed": args.split_seed,
                       "metric": "r2" if cfg["task"] == "continuous" else "f1_macro"}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

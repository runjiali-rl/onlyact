"""Per-provider whitening statistics for the JustAct world model of Table 2.

The JustAct action vector (640-d, unit norm) enters the world model through a fixed, parameter-free whitening map
    x -> ((x - mean) / std) @ W,
measured once on the provider's own embeddings of ALL transitions of the world-model training pool (no evaluation data):
  1. mean / std per dimension (float64 accumulation, stored as float32; std below 1e-6 is replaced by 1);
  2. the covariance C of the standardised rows, its eigendecomposition C = Q diag(e) Q^T, and gains
     g = sqrt(D / sum(e / (e + r))) / sqrt(e + r) with the smallest ridge r >= 0 (found by bisection) such that no gain
     exceeds 2. W = Q diag(g) Q^T is symmetric, keeps the total variance equal to D (= 640), and balances the
     directions without amplifying any of them more than 2x. For the released statistics r = 1.576, gains 0.369 .. 2.0,
     and the number of directions holding 90 % of the variance rises from 116 to 253.
Output: a safetensors file with mean, std, transform (= W), inverse (float32) and eigenvalues, gains, ridge (float64),
plus a JSON report next to it.

    python compute_whitening.py --train_list ../data/worldmodel/train_pool.json.gz \
        --emb_dir $EMB/justact/worldmodel_pool --out checkpoints/worldmodel/justact_whitening.safetensors

The released statistics were computed with --device cuda. A CPU run reproduces mean and std bit for bit; 2 of the
409,600 transform entries differ, by at most 1.4e-14.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from clip_lists import read_clip_list  # noqa: E402


def per_dimension_stats(tasks, emb_dir):
    s = ss = None
    n = files = 0
    for t in tasks:
        p = os.path.join(emb_dir, t["task_id"] + ".npy")
        if not os.path.exists(p):
            continue
        a = np.load(p).astype(np.float64)
        files += 1
        n += len(a)
        if s is None:
            s, ss = a.sum(0), (a * a).sum(0)
        else:
            s += a.sum(0)
            ss += (a * a).sum(0)
    assert files > 0, f"no embedding files in {emb_dir}"
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 0.0))
    return mean.astype(np.float32), std.astype(np.float32), n, files


def bounded_gains(eigenvalues, cap=2.0):
    e = np.asarray(eigenvalues, np.float64)
    assert np.isfinite(e).all() and (e >= 0).all() and e.sum() > 0 and cap > 1

    def gains(r):
        if r == 0:
            return 1 / np.sqrt(e) if e.min() > 0 else np.full_like(e, np.inf)
        return np.sqrt(len(e) / np.sum(e / (e + r))) / np.sqrt(e + r)

    if gains(0).max() <= cap:
        return gains(0), 0.0
    low, high = 0.0, max(float(e.max()), 1.0)
    while gains(high).max() > cap:
        high *= 2
    for _ in range(100):
        mid = (low + high) / 2
        if gains(mid).max() > cap:
            low = mid
        else:
            high = mid
    return gains(high), high


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_list", required=True, help="world-model training clip list (task order is kept)")
    ap.add_argument("--emb_dir", required=True, help="<emb_root>/<representation>/<dataset> of the training pool")
    ap.add_argument("--out", required=True, help="output .safetensors")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_gain", type=float, default=2.0)
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if os.path.exists(a.out):
        raise FileExistsError(f"refusing to overwrite {a.out}")
    tasks = read_clip_list(a.train_list)["tasks"]

    mu, sd, expected_n, expected_files = per_dimension_stats(tasks, a.emb_dir)
    print(f"[whitening] per-dimension statistics: {expected_files} clips, {expected_n} rows, dim {len(mu)}", flush=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    d = len(mu)
    mean_t = torch.tensor(mu, dtype=torch.float64, device=a.device)
    std_t = torch.tensor(sd, dtype=torch.float64, device=a.device)
    total = torch.zeros(d, dtype=torch.float64, device=a.device)
    products = torch.zeros((d, d), dtype=torch.float64, device=a.device)
    n, buffer = 0, []
    for i, task in enumerate(tasks):
        x = np.load(os.path.join(a.emb_dir, task["task_id"] + ".npy"))
        assert x.ndim == 2 and x.shape[1] == d and np.isfinite(x).all(), task["task_id"]
        assert len(x) == len(task["target_frames"]["paths"]) - 1, task["task_id"]
        n += len(x)
        buffer.append(x)
        if len(buffer) == 256 or i == len(tasks) - 1:
            z = (torch.tensor(np.concatenate(buffer), dtype=torch.float64, device=a.device) - mean_t) / std_t
            total += z.sum(0)
            products += z.T @ z
            buffer.clear()
    assert len(tasks) == expected_files and n == expected_n
    zmean = total / n
    cov = (products / n - torch.outer(zmean, zmean)).cpu().numpy()
    cov = (cov + cov.T) / 2
    e, q = np.linalg.eigh(cov)
    assert e.min() > -1e-8
    e = e.clip(0)
    g, ridge = bounded_gains(e, a.max_gain)
    w = (q * g) @ q.T
    inverse = (q / g) @ q.T
    assert g.min() > 0 and g.max() <= a.max_gain + 1e-10
    assert abs(np.sum(e * g * g) - d) < 1e-7 * d
    w32, inv32 = w.astype(np.float32), inverse.astype(np.float32)
    reconstruction_error = float(np.max(np.abs(w32 @ inv32 - np.eye(d))))
    assert reconstruction_error < 2e-5
    output_trace = float(np.trace(w32.T @ cov @ w32))
    assert abs(output_trace - d) < 1e-5 * d

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tensors = {"mean": mu, "std": sd.astype(np.float32), "transform": w32, "inverse": inv32,
               "eigenvalues": e, "gains": g, "ridge": np.asarray(ridge, dtype=np.float64)}
    save_file({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in tensors.items()}, a.out)
    pc90 = lambda v: int(np.searchsorted(v.cumsum(), .9 * v.sum()) + 1)  # noqa: E731
    report = {"width": d, "n_rows": n, "n_files": len(tasks), "ridge": ridge, "gain_min": float(g.min()), "gain_max": float(g.max()),
              "input_trace": float(e.sum()), "output_trace": output_trace,
              "directions_for_90pct_variance_before": pc90(np.sort(e)[::-1]),
              "directions_for_90pct_variance_after": pc90(np.sort(e * g * g)[::-1]),
              "inverse_max_error_float32": reconstruction_error, "device": a.device}
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

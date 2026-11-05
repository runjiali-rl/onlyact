"""Generate the Table 2 world-model rollouts for one evaluation cell and score them against the ground truth.

Each rollout starts from the TARGET clip's first --ctx frames and predicts --horizon frames autoregressively, one frame
per sampler call, conditioned on the SOURCE clip's per-transition action vectors. In a self cell source == target
(the clip's own actions); in a transfer cell the source is a different clip with opposite motion, and the generated
motion is scored against the source's motion with score.py. Predicting frame ctx+i uses the action row of the frame pair
(ctx-1+i, ctx+i), the same alignment as in training.

Sampling follows AdaWorld: Euler EDM sampler with the shifted EDM discretisation, --steps denoising steps, classifier-free
guidance --cfg whose unconditional branch zeroes the image conditions and the action. Paper setting: seed 32, 6 context
frames, 19 predicted frames, 25 steps, guidance 1.1, 256 x 256 pixels. The random stream is seeded once per process and
consumed clip after clip, so identical numbers need the same cell file (same rollout order) and the same software stack.

Outputs in --out_dir:
  results.jsonl  one line per rollout: source, target, psnr / ssim / lpips (mean over the predicted frames, computed in
                 floating point on [0, 1] frames), psnr_per_frame, sharp_per_frame (Laplacian-variance ratio pred / gt)
  grid_XX.png    ground-truth row (frames ctx .. ctx+horizon-1) above the predicted row, one per rollout
  summary.json   means, standard deviations and the settings used

    python generate.py --arm onlyact --cell ../data/worldmodel/libero_transfer.json --data_root $DATA/libero \
        --emb_root $EMB --out_dir runs/table2/onlyact/libero_transfer
"""
import argparse
import hashlib
import json
import os
import sys
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from clip_lists import load_clip_list, rollout_pairs  # noqa: E402
from loading import inject_lora, load_adaworld_base, load_config, load_trainable, slot_widths  # noqa: E402
from sample_utils import get_batch, init_sampling  # noqa: E402
from vwm.util import instantiate_from_config  # noqa: E402

ARMS = ("no_action", "adaworld_lam", "dreamdojo_lam", "onlyact")


def load_frame(path: str, res: int) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    x = torch.from_numpy(np.array(im)).float() / 255.0
    h, w = x.shape[:2]
    if h != w:
        s = min(h, w); x = x[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    x = x.permute(2, 0, 1)[None]
    if x.shape[-1] != res:
        x = F.interpolate(x, res, mode="bicubic")
    return x[0].clamp(0, 1)


def rollout(model, sampler, frames: List[torch.Tensor], rows: torch.Tensor, ctx: int, horizon: int, n_ctx_slots: int,
            aug_level: float, cfg_keys: List[str]):
    """frames: ctx + horizon frames in [0, 1], each (3, R, R); rows: (horizon, D) action vectors; returns (horizon, 3, R, R)."""
    dev = "cuda"
    zeros = torch.zeros_like(frames[0])
    filled = [zeros] * (n_ctx_slots - ctx) + frames[:ctx + 1]
    prev = torch.stack(filled[:n_ctx_slots]) * 2.0 - 1.0
    nxt = filled[n_ctx_slots] * 2.0 - 1.0
    img_seq = torch.cat([prev, nxt[None]]).to(dev)
    value_dict = {
        "img_seq": img_seq,
        "cond_frames_without_noise": prev[-1][None].to(dev),
        "cond_frames": (prev[-1] + 0.02 * torch.randn_like(prev[-1]))[None].to(dev),
        "ext_action": rows[0][None].to(dev),
        "context_len": torch.Tensor([ctx]).to(dev),
        "context_aug": torch.Tensor([aug_level]).to(dev),
    }
    keys = list(set(e.input_key for e in model.conditioner.embedders))
    z = model.encode_first_stage(value_dict["img_seq"])
    res = frames[0].shape[-1]
    out = torch.zeros(horizon, 3, res, res, device=dev)

    def denoiser(x, sigma, c):
        return model.denoiser(model.model, x, sigma, c)

    for i in range(horizon):
        batch, batch_uc = get_batch(keys, value_dict)
        c, uc = model.conditioner.get_unconditional_conditioning(batch, batch_uc=batch_uc, force_uc_zero_embeddings=cfg_keys)
        randn = torch.randn_like(z)
        samples_z = sampler(denoiser, randn, x_ori=z, cond=c, uc=uc)
        samples_x = model.decode_first_stage(samples_z)
        samples = torch.clamp((samples_x + 1.0) / 2.0, 0.0, 1.0)
        out[i] = samples[-1]
        if i < horizon - 1:
            value_dict["context_len"] = torch.Tensor([min(ctx + i + 1, n_ctx_slots)]).to(dev)
            value_dict["ext_action"] = rows[i + 1][None].to(dev)
            value_dict["cond_frames"] = samples_z[[-1]] / model.scale_factor
            value_dict["cond_frames_without_noise"] = torch.clamp(samples_x[[-1]], -1.0, 1.0)
            for e in model.conditioner.embedders:
                if hasattr(e, "skip_encode"):
                    e.skip_encode = True
            z = torch.cat([samples_z[1:], torch.zeros_like(samples_z[:1])], dim=0)
    for e in model.conditioner.embedders:
        if hasattr(e, "skip_encode"):
            e.skip_encode = False
    return out


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=ARMS, help="which world model (selects configs/<arm>.yaml)")
    ap.add_argument("--cell", required=True, help="evaluation cell, data/worldmodel/<cell>.json")
    ap.add_argument("--data_root", required=True, help="root that the cell's relative frame paths start from")
    ap.add_argument("--emb_root", default="", help="action embeddings, <emb_root>/<representation>/<dataset>/<task_id>.npy")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", default="", help="default: configs/<arm>.yaml")
    ap.add_argument("--base_ckpt", default=os.path.join(REPO, "checkpoints", "adaworld", "adaworld.safetensors"),
                    help="released AdaWorld weights (Hugging Face Little-Podi/AdaWorld)")
    ap.add_argument("--ckpt", default="", help="world-model checkpoint, default checkpoints/worldmodel/<arm>.safetensors")
    ap.add_argument("--whitening_stats", default=os.path.join(REPO, "checkpoints", "worldmodel", "onlyact_whitening.safetensors"),
                    help="per-provider whitening statistics (OnlyAct world model only)")
    ap.add_argument("--lora_rank", type=int, default=128)
    ap.add_argument("--lora_alpha", type=int, default=0, help="0 = rank")
    ap.add_argument("--lora_targets", default="to_q,to_k,to_v,to_out.0")
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=6, help="initial context frames (1..6)")
    ap.add_argument("--horizon", type=int, default=19)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=1.1)
    ap.add_argument("--aug_level", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="> 0: only the first N rollouts of the cell (they are identical "
                                                           "to the first N of a full run)")
    a = ap.parse_args()
    config = a.config or os.path.join(HERE, "configs", f"{a.arm}.yaml")
    ckpt = a.ckpt or os.path.join(REPO, "checkpoints", "worldmodel", f"{a.arm}.safetensors")

    # Order matters from here on: model construction, LoRA initialisation and the LPIPS network all draw from the seeded
    # CPU random stream before the first rollout.
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    cfg = load_config(config, a.whitening_stats)
    representation = cfg.data.params.representation
    action_dim = int(cfg.data.params.action_dim)
    model = instantiate_from_config(cfg.model)
    adm_w, ctx_w = slot_widths(cfg)
    n, missing, unexpected = load_adaworld_base(model, a.base_ckpt, adm_w, ctx_w)
    print(f"[generate] base: {n} tensors; missing {len(missing)} (e.g. {missing[:3]}); unexpected {len(unexpected)}")
    if a.lora_rank > 0:
        inject_lora(model, a.lora_rank, a.lora_alpha, a.lora_targets)
    load_trainable(model, ckpt)
    model = model.cuda().eval()
    n_slots = int(cfg.model.params.n_context_frames)
    sampler = init_sampling(steps=a.steps, cfg_scale=a.cfg, n_context_frames=n_slots)
    cfg_keys = ["cond_frames_without_noise", "cond_frames", "ext_action"]

    cell = load_clip_list(a.cell, a.data_root)
    tasks = {t["task_id"]: t for t in cell["tasks"]}
    pairs = rollout_pairs(cell)
    if a.limit > 0:
        pairs = pairs[:a.limit]
    need = a.start + (a.ctx + a.horizon) * a.gap
    if representation is not None and not a.emb_root:
        raise ValueError(f"representation {representation}: pass --emb_root")
    emb_dir = os.path.join(a.emb_root, representation, cell["dataset"]) if representation is not None else None

    def emb_rows(task_id: str) -> torch.Tensor:
        if representation is None:
            return torch.zeros(a.horizon, action_dim)
        arr = np.load(os.path.join(emb_dir, task_id + ".npy"), mmap_mode="r")
        idx = [a.start + (a.ctx - 1 + i) * a.gap for i in range(a.horizon)]
        if idx[-1] >= arr.shape[0] or arr.shape[1] != action_dim:
            raise ValueError(f"{task_id}: embedding shape {arr.shape} too short for rows up to {idx[-1]} or wrong width")
        return torch.from_numpy(np.array(arr[idx], dtype=np.float32))

    import piq
    lpips_fn = piq.LPIPS()
    lap_k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)

    def lap_var(x):  # per-frame Laplacian variance of the grey image (sharpness); x: (T, 3, H, W) in [0, 1] on cpu
        g = x.float().mean(1, keepdim=True)
        return torch.nn.functional.conv2d(g, lap_k).flatten(1).var(dim=1)

    results, t0 = [], time.time()
    with torch.no_grad(), torch.autocast("cuda"):
        for j, (src, tgt) in enumerate(pairs):
            paths = tasks[tgt]["target_frames"]["abs_paths"]
            if len(paths) < need or (src != tgt and len(tasks[src]["target_frames"]["abs_paths"]) < need):
                print(f"[generate] skip {src}->{tgt}: fewer than {need} frames"); continue
            frames = [load_frame(paths[a.start + k * a.gap], a.resolution) for k in range(a.ctx + a.horizon)]
            rows = emb_rows(src)
            pred = rollout(model, sampler, frames, rows, a.ctx, a.horizon, n_slots, a.aug_level, cfg_keys).float().cpu()
            gt = torch.stack(frames[a.ctx:a.ctx + a.horizon])
            r = {"source": src, "target": tgt, "psnr": piq.psnr(gt, pred).mean().item(), "ssim": piq.ssim(gt, pred).mean().item(),
                 "lpips": lpips_fn(gt, pred).mean().item(),
                 "psnr_per_frame": [round(v, 3) for v in piq.psnr(gt, pred, reduction="none").tolist()],
                 "sharp_per_frame": [round(v, 4) for v in (lap_var(pred) / lap_var(gt).clamp_min(1e-10)).tolist()]}
            results.append(r)
            print(f"[generate] {j + 1}/{len(pairs)} {tgt[:48]}: PSNR {r['psnr']:.2f} SSIM {r['ssim']:.3f} LPIPS {r['lpips']:.3f} "
                  f"({(time.time() - t0) / (j + 1):.1f} s/clip)")
            grid = torch.cat([torch.cat(list(gt), dim=2), torch.cat(list(pred), dim=2)], dim=1)  # gt row over pred row
            Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(os.path.join(a.out_dir, f"grid_{j:02d}.png"))
    with open(os.path.join(a.out_dir, "results.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    summ = {k: {"mean": float(np.mean([r[k] for r in results])), "sd": float(np.std([r[k] for r in results], ddof=1)) if len(results) > 1 else None}
            for k in ("psnr", "ssim", "lpips")} if results else {}
    if results:  # horizon buckets: predicted frames 1-6, 7-12, 13-end (mean over clips of the per-frame values)
        pf, sf = np.array([r["psnr_per_frame"] for r in results]), np.array([r["sharp_per_frame"] for r in results])
        summ["psnr_per_frame"] = [round(float(v), 3) for v in pf.mean(0)]; summ["sharp_per_frame"] = [round(float(v), 4) for v in sf.mean(0)]
        summ["horizon_buckets"] = {"psnr": [round(float(pf[:, lo:hi].mean()), 3) for lo, hi in ((0, 6), (6, 12), (12, pf.shape[1]))],
                                   "sharp": [round(float(sf[:, lo:hi].mean()), 4) for lo, hi in ((0, 6), (6, 12), (12, sf.shape[1]))]}
    summ.update({"n": len(results), "arm": a.arm, "cell": cell.get("cell"), "mode": cell.get("mode"), "dataset": cell["dataset"],
                 "representation": representation, "ckpt_sha256": sha256(ckpt),
                 "whitening_sha256": sha256(a.whitening_stats) if a.arm == "onlyact" else None,
                 "steps": a.steps, "cfg": a.cfg, "ctx": a.ctx, "horizon": a.horizon, "resolution": a.resolution, "seed": a.seed,
                 "aug_level": a.aug_level, "gap": a.gap, "start": a.start})
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print("[generate] SUMMARY", json.dumps(summ))


if __name__ == "__main__":
    main()

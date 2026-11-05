"""Train the AdaWorld and DreamDojo latent action models from scratch on the OnlyAct training pairs.

Produces the AdaWorld-LAM (retrained) and DreamDojo-LAM (retrained) checkpoints behind the retrained
latent-action baseline rows of Tables 1-4. Each model keeps its upstream network, objective and optimiser;
only the training data changes: the frame pairs the OnlyAct encoder is trained on, at the same budget
(160,000 pair views).

  network     vendored unmodified (third_party/, see lam_models.py)
  objective   upstream LAM.shared_step: mean-squared error between frame t+1 and its reconstruction
              (mean over all elements) + beta x Kullback-Leibler divergence of the latent (summed over the
              latent dimensions, averaged over the batch); beta from the upstream YAML
  optimiser   AdamW, learning rate / weight decay / gradient clipping from the upstream YAML,
              bfloat16 autocast, random initialisation, seed 0
  input       each repository's training preprocessing (see CachePairDataset)

Input (--cache_dir), the training-pair cache written by the OnlyAct cache builder:
  index.jsonl                                one JSON record per pair: {"i", "task_id", "dataset", ...}
  frames/<i // 1000:03d>/<i:06d>_a0.jpg      frame t
  frames/<i // 1000:03d>/<i:06d>_a1.jpg      frame t + gap (gap 1-3)
Output (--out_dir):
  config.json, train_log.jsonl, summary.json
  ckpt_step<N>.safetensors   weights only, the file extract.py loads
  ckpt_last.pt               resume point (weights, optimiser, random-number and data-order generator states)
  recon_step<N>.png          rows: frame t / frame t+1 / reconstruction of frame t+1, on fixed held-out pairs

Usage:
  python baselines/train.py --config baselines/configs/adaworld.yaml --cache_dir <cache> --out_dir <out>
  python baselines/train.py --config baselines/configs/dreamdojo.yaml --cache_dir <cache> --out_dir <out>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import piq
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import lam_models

AMP_DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------------------------------------
# Objective (upstream lam/model.py :: LAM.shared_step, identical in both repositories)
# ---------------------------------------------------------------------------------------------------------
def lam_beta_vae_loss(outputs: Dict[str, Tensor], videos: Tensor, beta: float) -> Tuple[Tensor, Tensor, Tensor]:
    gt_future_frames = videos[:, 1:]
    mse_loss = ((gt_future_frames - outputs["recon"]) ** 2).mean()
    # z_var is the log-variance of the latent.
    kl_loss = -0.5 * torch.sum(
        1 + outputs["z_var"] - outputs["z_mu"] ** 2 - outputs["z_var"].exp(), dim=1
    ).mean()
    loss = mse_loss + beta * kl_loss
    return loss, mse_loss, kl_loss


# ---------------------------------------------------------------------------------------------------------
# Training pairs
# ---------------------------------------------------------------------------------------------------------
class CachePairDataset(Dataset):
    """One item = the 2-frame clip (frame t, frame t+gap) of one cache record, preprocessed the way the
    architecture's upstream training dataset (lam/dataset.py :: VideoDataset) preprocesses a clip.

    Split: records sorted by "i", index order shuffled with random.Random(seed); the first val_frac of that
    order is the validation split (the same split rule as the OnlyAct encoder's cache loader).

    Geometry, pixels in [0, 1], channels last:
      adaworld   centre crop to a square of side min(H, W), bicubic resize to `resolution` square
                 (skipped when already that size)
      dreamdojo  centre crop to width / height = crop_aspect (640 / 480), bilinear resize to
                 resize_hw = (240, 320)
    Augmentation (both): one brightness offset per clip, uniform in [-0.1, 0.1], then clamp to [0, 1].
    As upstream, there is no clamp after the resize, so validation pixels can overshoot [0, 1] slightly.
    """

    def __init__(
        self,
        cache_dir: str,
        split: str = "train",
        val_frac: float = 0.02,
        seed: int = 0,
        resolution: int = 256,
        output_format: str = "t h w c",
        color_aug: bool = True,
        crop_aspect: Optional[float] = None,
        resize_hw: Optional[Tuple[int, int]] = None,
        interp_mode: str = "bicubic",
    ) -> None:
        self.cache_dir = cache_dir
        self.resolution = int(resolution)
        self.output_format = output_format
        self.color_aug = bool(color_aug)
        self.crop_aspect = (float(crop_aspect) if crop_aspect else None)
        self.resize_hw = (int(resize_hw[0]), int(resize_hw[1])) if resize_hw else None
        self.interp_mode = str(interp_mode)
        self.records: List[Dict[str, Any]] = []
        with open(os.path.join(cache_dir, "index.jsonl")) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        self.records.sort(key=lambda r: r["i"])
        order = list(range(len(self.records)))
        random.Random(seed).shuffle(order)
        n_val = int(len(order) * val_frac)
        keep = set(order[:n_val]) if split == "val" else set(order[n_val:])
        self.records = [r for j, r in enumerate(self.records) if j in keep]

    def __len__(self) -> int:
        return len(self.records)

    def _frame_path(self, i: int, key: str) -> str:
        return os.path.join(self.cache_dir, "frames", f"{i // 1000:03d}", f"{i:06d}_{key}.jpg")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        i = int(rec["i"])
        imgs = [Image.open(self._frame_path(i, key)).convert("RGB") for key in ("a0", "a1")]
        if imgs[1].size != imgs[0].size:  # guard only; both frames of a pair come from the same video
            imgs[1] = imgs[1].resize(imgs[0].size, Image.BICUBIC)
        frames = [torch.from_numpy(np.asarray(im, dtype=np.uint8)) for im in imgs]
        video = torch.stack(frames).float() / 255.0  # (T, H, W, C) in [0, 1]
        if self.crop_aspect is not None:
            # DreamDojo: centre crop to a fixed width / height ratio.
            target_ratio = self.crop_aspect
            if video.shape[2] / video.shape[1] > target_ratio:
                target_height = video.shape[1]
                target_width = int(video.shape[1] * target_ratio)
            elif video.shape[2] / video.shape[1] < target_ratio:
                target_height = int(video.shape[2] / target_ratio)
                target_width = video.shape[2]
            else:
                target_height = video.shape[1]
                target_width = video.shape[2]
            h_crop = (video.shape[1] - target_height) // 2
            w_crop = (video.shape[2] - target_width) // 2
            video = video[:, h_crop:h_crop + target_height, w_crop:w_crop + target_width]
        elif video.shape[1] != video.shape[2]:
            # AdaWorld: square centre crop.
            square_len = min(video.shape[1], video.shape[2])
            h_crop = (video.shape[1] - square_len) // 2
            w_crop = (video.shape[2] - square_len) // 2
            video = video[:, h_crop:h_crop + square_len, w_crop:w_crop + square_len]
        target = self.resize_hw if self.resize_hw else (self.resolution, self.resolution)
        if (video.shape[-3], video.shape[-2]) != target:
            video = rearrange(video, "t h w c -> c t h w")
            video = F.interpolate(video, target, mode=self.interp_mode)
            video = rearrange(video, f"c t h w -> {self.output_format}")
        else:
            video = rearrange(video, f"t h w c -> {self.output_format}")
        if self.color_aug:
            video = (video + torch.rand(1) * 0.2 - 0.1).clamp(0, 1)
        return {"videos": video, "i": i, "dataset": str(rec.get("dataset", "")),
                "task_id": str(rec.get("task_id", ""))}


def collate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "videos": torch.stack([x["videos"] for x in items]),
        "i": torch.tensor([x["i"] for x in items], dtype=torch.long),
        "dataset": [x["dataset"] for x in items],
        "task_id": [x["task_id"] for x in items],
    }


def dataset_kwargs(arch: str, ucfg: Dict[str, Any]) -> Dict[str, Any]:
    spec = lam_models.ARCHS[arch]
    return dict(resolution=ucfg["resolution"], crop_aspect=spec["crop_aspect"], resize_hw=spec["resize_hw"],
                interp_mode=spec["interp_mode"], output_format=ucfg["output_format"])


def dataset_shares(ds: CachePairDataset) -> Dict[str, float]:
    """Share of each source dataset among the training records (records are drawn uniformly)."""
    cnt = Counter(str(r.get("dataset", "")) for r in ds.records)
    n = max(1, len(ds.records))
    return {k: v / n for k, v in sorted(cnt.items())}


def make_loaders(arch: str, ucfg: Dict[str, Any], args) -> Tuple[DataLoader, DataLoader, torch.Generator]:
    """Training loader: uniform shuffle over the training records, driven by one generator seeded with
    --seed (its state goes into the resume checkpoint). --num_workers also fixes which worker, and so
    which random stream, draws each pair's brightness offset."""
    tr = CachePairDataset(args.cache_dir, split="train", val_frac=args.val_frac, seed=args.seed,
                          color_aug=True, **dataset_kwargs(arch, ucfg))
    va = CachePairDataset(args.cache_dir, split="val", val_frac=args.val_frac, seed=args.seed,
                          color_aug=False, **dataset_kwargs(arch, ucfg))
    data_gen = torch.Generator().manual_seed(int(args.seed))
    tl = DataLoader(tr, batch_size=args.micro_batch, shuffle=True, generator=data_gen,
                    num_workers=args.num_workers, collate_fn=collate, drop_last=True,
                    pin_memory=True, persistent_workers=args.num_workers > 0,
                    prefetch_factor=(args.prefetch if args.num_workers > 0 else None))
    vl = DataLoader(va, batch_size=args.micro_batch, shuffle=False, num_workers=max(0, args.num_workers // 2),
                    collate_fn=collate)
    return tl, vl, data_gen


# ---------------------------------------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------------------------------------
def save_resume_ckpt(path: str, model, opt, step: int, meta: Dict[str, Any], data_gen: torch.Generator) -> None:
    tmp = path + ".tmp"
    torch.save({
        "state_dict": {f"lam.{k}": v for k, v in model.state_dict().items()},
        "optimizer": opt.state_dict(),
        "step": int(step),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (torch.cuda.get_rng_state() if torch.cuda.is_available() else None),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "data_rng": data_gen.get_state(),
        "meta": meta,
    }, tmp)
    os.replace(tmp, path)


def load_resume_ckpt(path: str, model, opt, data_gen: torch.Generator) -> int:
    """Restores weights, optimiser, all random-number states and the data-order generator. The training
    loader is then re-created, so the data order after a resume is drawn from the restored generator."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k[len("lam."):]: v for k, v in ck["state_dict"].items() if k.startswith("lam.")}
    model.load_state_dict(sd, strict=True)
    opt.load_state_dict(ck["optimizer"])
    torch.set_rng_state(ck["torch_rng"].to(torch.uint8))
    if ck.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(ck["cuda_rng"].to(torch.uint8))
    np.random.set_state(ck["numpy_rng"])
    random.setstate(ck["python_rng"])
    data_gen.set_state(ck["data_rng"].to(torch.uint8))
    print(f"[resume] restored step {ck['step']} from {path}", flush=True)
    return int(ck["step"])


# ---------------------------------------------------------------------------------------------------------
# Monitoring: reconstruction sheet and validation
# ---------------------------------------------------------------------------------------------------------
@torch.no_grad()
def contact_sheet(path: str, model: torch.nn.Module, batch: Dict[str, Any], device: str, n_pairs: int = 6) -> None:
    """Rows: frame t | frame t+1 | reconstruction of frame t+1 (eval mode, so the latent is z_mu)."""
    was_training = model.training
    model.eval()
    x = batch["videos"][:n_pairs].to(device, non_blocking=True)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        out = model({"videos": x})
    if getattr(model, "mu_record", None) is not None:
        model.mu_record = None  # encode() appends every z_mu to this buffer in eval mode
    gt_in = x[:, 0].float().clamp(0, 1).cpu()
    gt_next = x[:, 1].float().clamp(0, 1).cpu()
    pred = out["recon"][:, 0].float().clamp(0, 1).cpu()
    grid = torch.cat([torch.cat(list(r), dim=1) for r in (gt_in, gt_next, pred)], dim=0)  # (3H, N*W, C)
    img = Image.fromarray((grid.numpy() * 255.0).round().astype(np.uint8))
    tmp = path + ".tmp.png"
    img.save(tmp)
    os.replace(tmp, path)
    if was_training:
        model.train()


@torch.no_grad()
def validate(model, loader, device, beta, max_batches: int = 0) -> Dict[str, float]:
    """Loss, peak signal-to-noise ratio and structural similarity of the reconstruction on the first
    `max_batches` batches of the validation split (0 = all)."""
    was_training = model.training
    model.eval()
    tot = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "psnr": 0.0, "ssim": 0.0}
    nb = 0
    nrec = 0
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x = batch["videos"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            out = model({"videos": x})
            loss, mse, kl = lam_beta_vae_loss(out, x, beta)
        if getattr(model, "mu_record", None) is not None:
            model.mu_record = None
        nrec += int(x.shape[0])
        tot["loss"] += float(loss)
        tot["recon"] += float(mse)
        tot["kl"] += float(kl)
        gt = x[:, 1:].float().clamp(0, 1).reshape(-1, *x.shape[2:]).permute(0, 3, 1, 2)
        rc = out["recon"].float().clamp(0, 1).reshape(-1, *out["recon"].shape[2:]).permute(0, 3, 1, 2)
        tot["psnr"] += float(piq.psnr(gt, rc).mean())
        tot["ssim"] += float(piq.ssim(gt, rc).mean())
        nb += 1
    if was_training:
        model.train()
    out = {k: v / max(nb, 1) for k, v in tot.items()}
    out.update(n_batches=nb, n_records=nrec, n_val_total=len(loader.dataset))
    return out


# ---------------------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------------------
def train(args) -> Dict[str, Any]:
    arch = args.arch
    ucfg = lam_models.read_upstream_config(arch)
    if ucfg["num_frames"] != 2:
        raise SystemExit(f"{ucfg['config_yaml']} asks for {ucfg['num_frames']} frames; the cache holds frame pairs")
    dev = args.device
    os.makedirs(args.out_dir, exist_ok=True)
    # Random initialisation, drawn on the device.
    net_cls = lam_models.training_network_class(arch)
    with torch.device(dev):
        model = net_cls(**ucfg["model_kwargs"])
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=ucfg["lr"], weight_decay=ucfg["weight_decay"])

    tl, vl, data_gen = make_loaders(arch, ucfg, args)
    shares = dataset_shares(tl.dataset)
    print(f"[data] {len(tl.dataset)} train records, {len(vl.dataset)} validation records "
          f"({args.val_frac:.1%} held out, seed {args.seed})", flush=True)
    print(f"[data] dataset shares: {{{', '.join(f'{k}:{v:.4f}' for k, v in shares.items())}}}", flush=True)

    meta = dict(arch=arch, n_params=n_par, beta=ucfg["beta"], lr=ucfg["lr"], weight_decay=ucfg["weight_decay"],
                gradient_clip_val=ucfg["gradient_clip_val"], precision="bf16", micro_batch=args.micro_batch,
                grad_accum=args.grad_accum, effective_batch=args.micro_batch * args.grad_accum,
                dataset_shares=shares, seed=args.seed, config_yaml=ucfg["config_yaml"])
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(dict(meta=meta, upstream=ucfg, args=vars(args)), f, indent=2, default=str)
    print(f"[cfg] {json.dumps({k: v for k, v in meta.items() if k != 'dataset_shares'}, default=str)}", flush=True)

    step = 0
    if args.resume:
        if not os.path.isfile(args.resume):
            raise SystemExit(f"--resume {args.resume}: no such file")
        step = load_resume_ckpt(args.resume, model, opt, data_gen=data_gen)
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    log = open(log_path, "a", buffering=1)
    sheet_steps = {int(s) for s in (args.sheet_steps or "").split(",") if s.strip()}
    sheet_batch = None
    if sheet_steps:
        sb_loader = DataLoader(vl.dataset, batch_size=args.sheet_pairs, shuffle=False, num_workers=0,
                               collate_fn=collate)
        sheet_batch = next(iter(sb_loader))  # fixed held-out pairs, identical at every sheet step

    model.train()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    it = iter(tl)
    n_samples = 0
    if 0 in sheet_steps and step == 0 and sheet_batch is not None:
        contact_sheet(os.path.join(args.out_dir, "recon_step0.png"), model, sheet_batch, dev, args.sheet_pairs)
    while step < args.total_steps:
        t0 = time.time()
        acc = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(tl)
                batch = next(it)
            x = batch["videos"].to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                out = model({"videos": x})
                loss, mse, kl = lam_beta_vae_loss(out, x, ucfg["beta"])
            scaled = loss / args.grad_accum
            scaled.backward()
            acc["loss"] += float(loss) / args.grad_accum
            acc["recon"] += float(mse) / args.grad_accum
            acc["kl"] += float(kl) / args.grad_accum
            n_samples += int(x.shape[0])
        gnorm = float("nan")
        if ucfg["gradient_clip_val"] > 0:
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), ucfg["gradient_clip_val"]))
        opt.step()
        step += 1
        dt = time.time() - t0
        rec = dict(kind="train", step=step, loss=acc["loss"], recon=acc["recon"], kl=acc["kl"],
                   grad_norm=gnorm, samples=args.micro_batch * args.grad_accum,
                   samples_per_s=round(args.micro_batch * args.grad_accum / max(dt, 1e-9), 3),
                   s_per_step=round(dt, 4),
                   peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
                   wall_s=round(time.time() - t_start, 2), lr=ucfg["lr"])
        log.write(json.dumps(rec) + "\n")
        if step % args.log_every == 0 or step == 1 or step == args.total_steps:
            print(f"[train] step {step}/{args.total_steps} loss {acc['loss']:.6f} "
                  f"(reconstruction {acc['recon']:.6f} + {ucfg['beta']:g} x Kullback-Leibler {acc['kl']:.4f}) "
                  f"{rec['samples_per_s']} samples/s peak {rec['peak_gib']:.2f} GiB "
                  f"{rec['wall_s'] / 60:.1f} min", flush=True)
        if args.val_every and step % args.val_every == 0:
            v = validate(model, vl, dev, ucfg["beta"], args.val_batches)
            v.update(kind="val", step=step, wall_s=round(time.time() - t_start, 2))
            log.write(json.dumps(v) + "\n")
            print(f"[val]   step {step} loss {v['loss']:.6f} reconstruction {v['recon']:.6f} "
                  f"peak signal-to-noise ratio {v['psnr']:.2f} dB structural similarity {v['ssim']:.4f} "
                  f"({v['n_records']} of {v['n_val_total']} validation records)", flush=True)
        if step in sheet_steps and sheet_batch is not None:
            contact_sheet(os.path.join(args.out_dir, f"recon_step{step}.png"), model, sheet_batch, dev,
                          args.sheet_pairs)
        if args.save_every and step % args.save_every == 0:
            lam_models.save_weights(os.path.join(args.out_dir, f"ckpt_step{step}.safetensors"), model)
            save_resume_ckpt(os.path.join(args.out_dir, "ckpt_last.pt"), model, opt, step, meta, data_gen)

    final = os.path.join(args.out_dir, f"ckpt_step{step}.safetensors")
    lam_models.save_weights(final, model)
    save_resume_ckpt(os.path.join(args.out_dir, "ckpt_last.pt"), model, opt, step, meta, data_gen)
    v = validate(model, vl, dev, ucfg["beta"], args.val_batches)
    v.update(kind="val", step=step, wall_s=round(time.time() - t_start, 2), final=True)
    log.write(json.dumps(v) + "\n")
    print(f"[val]   final step {step} loss {v['loss']:.6f} reconstruction {v['recon']:.6f} "
          f"peak signal-to-noise ratio {v['psnr']:.2f} dB structural similarity {v['ssim']:.4f} "
          f"({v['n_records']} of {v['n_val_total']} validation records)", flush=True)
    log.close()
    del model, opt
    torch.cuda.empty_cache()
    # The final weights must load into the network extract.py builds (for DreamDojo, the inference copy).
    m = lam_models.build_extraction_model(arch, final, dev)
    del m
    torch.cuda.empty_cache()
    summary = dict(arch=arch, steps=step, samples_seen=n_samples, wall_min=round((time.time() - t_start) / 60, 2),
                   final_val=v, final_ckpt=final, meta=meta, log=log_path)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


# ---------------------------------------------------------------------------------------------------------
RUN_KEYS = ("arch", "total_steps", "micro_batch", "grad_accum", "seed", "val_frac", "num_workers", "prefetch",
            "log_every", "val_every", "val_batches", "save_every", "sheet_steps", "sheet_pairs")


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="run settings, e.g. baselines/configs/adaworld.yaml")
    ap.add_argument("--cache_dir", required=True, help="training-pair cache (index.jsonl + frames/)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", default="", help="ckpt_last.pt of an interrupted run")
    ap.add_argument("--arch", choices=sorted(lam_models.ARCHS))
    ap.add_argument("--total_steps", type=int)
    ap.add_argument("--micro_batch", type=int)
    ap.add_argument("--grad_accum", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--val_frac", type=float)
    ap.add_argument("--num_workers", type=int)
    ap.add_argument("--prefetch", type=int)
    ap.add_argument("--log_every", type=int)
    ap.add_argument("--val_every", type=int)
    ap.add_argument("--val_batches", type=int)
    ap.add_argument("--save_every", type=int)
    ap.add_argument("--sheet_steps", help="comma-separated steps at which a reconstruction sheet is written")
    ap.add_argument("--sheet_pairs", type=int)
    if known.config:
        with open(known.config) as f:
            cfg = yaml.safe_load(f) or {}
        unknown = sorted(set(cfg) - set(RUN_KEYS))
        if unknown:
            raise SystemExit(f"{known.config}: unknown keys {unknown}")
        ap.set_defaults(**cfg)
    args = ap.parse_args(argv)
    missing = [k for k in RUN_KEYS if getattr(args, k) is None]
    if missing:
        raise SystemExit(f"missing run settings {missing} (set them in --config)")
    args.sheet_steps = str(args.sheet_steps)
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()

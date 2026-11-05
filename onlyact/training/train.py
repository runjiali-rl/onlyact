"""Contrastive training of the OnlyAct encoder (the checkpoint used in every table of the paper).

V-JEPA 2 ViT-L is adapted with LoRA (rank 16) through a fixed random 640-d projection. Each micro-batch holds
micro_batch training samples from the sample cache; every sample gives three frame-pair clips (anchor, background-
replaced positive, hard negative from a disjoint window of the same clip). The loss is the symmetric InfoNCE of
loss.py. AdamW, linear warm-up then cosine decay, gradient-norm clipping at 1.0, gradient accumulation.

    python -m onlyact.training.train --cache_dir <cache> --out_dir runs/onlyact

Writes <out_dir>/config.json, <out_dir>/train_log.jsonl (step 1 and every log_every steps), checkpoints
<out_dir>/step_<N>/ every save_every steps and <out_dir>/final/ (adapter_config.json, adapter_model.safetensors,
projection.safetensors). One RTX 4090 (24 GB): about 0.55 s per optimiser step, 1.5 h for the 10,000 steps.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .cache_dataset import CachedPairDataset, collate
from .config import TrainConfig
from .loss import info_nce_with_hard_negatives
from .model import OnlyActTrainModel

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "onlyact.yaml"


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    p = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def build_loader(cfg: TrainConfig, cache_dir: str, flow_cache_dir: str = "") -> DataLoader:
    ds = CachedPairDataset(cache_dir, split="train", val_frac=cfg.val_frac, seed=cfg.seed,
                           epoch_multiplier=cfg.epoch_multiplier, bg_max_motion_px=cfg.bg_max_motion_px,
                           feather_px=cfg.feather_px, core_shift_min_px=cfg.core_shift_min_px,
                           flow_cache_dir=flow_cache_dir)
    # The shuffle order is drawn from the global torch generator when the loader is first iterated, i.e. after the
    # model was built from the same generator.
    return DataLoader(ds, batch_size=cfg.micro_batch, shuffle=True, num_workers=cfg.num_workers,
                      collate_fn=collate, pin_memory=False, persistent_workers=(cfg.num_workers > 0), drop_last=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--cache_dir", required=True, help="sample cache (build_sample_cache.py)")
    ap.add_argument("--flow_cache_dir", default="", help="folder holding flow128/ (default: --cache_dir)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_workers", type=int, default=None, help="data loader workers (does not change the result)")
    ap.add_argument("--stop_after", type=int, default=0,
                    help="stop after this many optimiser steps; the learning-rate schedule still spans total_steps")
    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    torch.manual_seed(cfg.seed)

    print(f"[init] loading {cfg.model_name} + LoRA r={cfg.lora_r}")
    model = OnlyActTrainModel(cfg.model_name, revision=cfg.model_revision, lora_r=cfg.lora_r,
                              lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, lora_targets=cfg.lora_targets,
                              proj_dim=cfg.proj_dim, device=args.device)
    print(f"[init] parameters with requires_grad: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Only the LoRA adapters are optimised. The projection keeps requires_grad=True: its gradient enters the
    # gradient-norm clipping below together with the adapters' gradients. It is in no optimiser group, so it is never
    # updated, and optimizer.zero_grad() never resets its gradient, which therefore accumulates over all steps.
    lora_params = [p for _, p in model.backbone.named_parameters() if p.requires_grad]
    clip_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": lora_params, "lr": cfg.lr}], weight_decay=cfg.weight_decay,
                                  betas=tuple(cfg.betas))
    print(f"[init] optimiser: {len(lora_params)} LoRA tensors; projection fixed")

    loader = build_loader(cfg, args.cache_dir, args.flow_cache_dir)
    train_iter = iter(loader)
    log_path = out_dir / "train_log.jsonl"
    print(f"[init] logging to {log_path}")

    end_step = min(cfg.total_steps, args.stop_after) if args.stop_after > 0 else cfg.total_steps
    step = 0
    t_start = time.time()
    optimizer.zero_grad(set_to_none=True)
    while step < end_step:
        accum_loss = 0.0
        accum = {"acc_a2p": 0.0, "sim_pos": 0.0, "sim_hard_neg": 0.0, "gap": 0.0, "synth_n": 0.0, "synth_ok": 0.0}
        for _ in range(cfg.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(loader)
                batch = next(train_iter)
            embs = model(batch["items"])
            n = batch["n"]
            a, p, h = embs[:n], embs[n:2 * n], embs[2 * n:3 * n]
            loss, diag = info_nce_with_hard_negatives(a, p, h, tau=cfg.tau)
            (loss / cfg.grad_accum).backward()
            accum_loss += float(loss.item())
            accum["acc_a2p"] += float(diag["acc_a2p"].item())
            accum["sim_pos"] += float(diag["sim_pos"].item())
            accum["sim_hard_neg"] += float(diag["sim_hard_neg"].item())
            accum["gap"] += float(diag["gap_pos_vs_hn"].item())
            status = batch.get("synth_status", [])
            accum["synth_n"] += len(status)
            accum["synth_ok"] += sum(1 for s in status if str(s).endswith("_ok"))
        accum_loss /= cfg.grad_accum
        for k in accum:
            accum[k] /= cfg.grad_accum

        lr = cosine_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        torch.nn.utils.clip_grad_norm_(clip_params, cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % cfg.log_every == 0 or step == 1:
            elapsed = time.time() - t_start
            msg = {"step": step, "loss": accum_loss, "lr": lr, **accum, "sec_per_step": elapsed / step}
            ok = accum["synth_ok"] / max(1.0, accum["synth_n"])
            print(f"[train] step {step}/{cfg.total_steps} loss={accum_loss:.4f} acc={accum['acc_a2p']:.3f} "
                  f"gap={accum['gap']:.3f} lr={lr:.2e} secs/step={elapsed / step:.1f} synth_ok={ok:.2f}", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(msg) + "\n")
        if step % cfg.save_every == 0:
            model.save(str(out_dir / f"step_{step:06d}"))
            print(f"[ckpt] {out_dir / f'step_{step:06d}'}")

    model.save(str(out_dir / "final"))
    print(f"[done] final checkpoint: {out_dir / 'final'}")


if __name__ == "__main__":
    main()

"""Fine-tune one pretrained world model on the labelled clips of one domain (paper Table 3).

The recipe is read from the domain config (downstream/configs/<domain>.json, block "recipe"):
  DMLab, Minecraft  standardised interface: the action book stores u, condition = mean + std * u (per-dimension statistics
                    of the arm's own world-model training features), interface learning rate 2.42e-3 per standard deviation;
  Habitat           native interface: the book holds the conditions directly, interface learning rate 5e-5;
  nuScenes          continuous action head (2 -> d -> d), learning rate 5e-5, starting from the fitted step-0 head.
Shared by all: whole merged UNet trainable at 5e-6, 800 AdamW updates (betas 0.9 / 0.999, weight decay 0.01), global batch
32 = 32 micro-batches of one clip, 100-update linear warm-up, gradient clipping 0.3, EMA of the UNet (decay 0.9999 with the
num-updates correction), bf16 autocast, activation checkpointing, seed 32. Saved: EMA UNet + live interface, FP32
safetensors, at the updates listed in "save_steps" (a standardised book is saved as native rows mean + std * u).

Needs one 24 GB GPU and 24-26 GiB of free host memory; about 4 h on an RTX 4090 (7 h on an RTX A5500).

  python -m downstream.train --config downstream/configs/dmlab.json --arm justact --data_root DATA/dmlab \
      --out runs/table3/dmlab/justact
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from .common import index_plan, load_clips, load_config, seed_all, seed_for, tensor_digest, write_json
from .data import training_data
from .interfaces import load_step0, load_standardise
from .model import build_pretrained, world_model_files, enable_checkpointing, install_interface, trainable_groups
from .offload import CPUAdamW, accumulated_update

INTERFACE_KEY = "conditioner.embedders.2.codebook.weight"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--data_root", required=True, help="folder the train list's clip / frame paths are relative to")
    ap.add_argument("--wm_config", help="default: worldmodel/configs/<world model>.yaml")
    ap.add_argument("--wm_ckpt", help="default: checkpoints/worldmodel/<world model>.safetensors")
    ap.add_argument("--base_ckpt", help="default: checkpoints/adaworld/adaworld.safetensors")
    ap.add_argument("--whitening_stats", help="JustAct only; default: checkpoints/worldmodel/justact_whitening.safetensors")
    ap.add_argument("--step0", help="step-0 interface (default: checkpoints/downstream/step0/<domain>_<arm>.safetensors)")
    ap.add_argument("--standardise", help="standardisation statistics (default: checkpoints/downstream/standardise/<arm>.safetensors)")
    ap.add_argument("--out", required=True); ap.add_argument("--no_check_hashes", action="store_true")
    ap.add_argument("--max_steps", type=int, default=0, help="stop after N updates (smoke test only)")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); torch.set_num_threads(4)
    cfg = load_config(a.config); rc = cfg["recipe"]; arm = cfg["arms"][a.arm]; dim = arm["native_dim"]
    out = Path(a.out); assert not out.exists(), "output folder exists (no implicit resume)"
    tasks = load_clips(cfg["train_list"], a.data_root)
    init = load_step0(cfg, a.arm, a.step0)
    standardise = load_standardise(a.arm, a.standardise) if rc["interface"] == "standardised" else None

    seed_all(rc["seed"])
    files = world_model_files(cfg, a.arm, a.wm_config, a.wm_ckpt, a.base_ckpt, a.whitening_stats)
    model = build_pretrained(files, dim)
    install_interface(model, cfg, init, dim, standardise)
    enable_checkpointing(model)
    model = model.cuda().train()
    named, groups = trainable_groups(model, rc["unet_lr"], rc["interface_lr"])
    opt = CPUAdamW(named, groups, betas=rc["betas"], eps=rc["eps"], weight_decay=rc["weight_decay"], warmup_steps=rc["warmup_steps"],
                   warmup_initial_factor=rc["warmup_initial_factor"], ema_decay=rc["ema_decay"], ema=True)
    accumulation = rc["global_batch"]
    plan = index_plan(len(tasks), rc["optimizer_steps"], accumulation, rc["seed"], "paper_full_wm_order"); plan_sha = tensor_digest(plan)
    data = training_data(cfg, tasks, check=not a.no_check_hashes)
    out.mkdir(parents=True)
    write_json(out / "start.json", {"domain": cfg["domain"], "arm": a.arm, "recipe": rc, "index_plan_sha256": plan_sha,
                                    **files, "train_list": cfg["train_list"]})
    t0 = time.monotonic()
    with open(out / "training_metrics.jsonl", "w") as mf:
        for step, indices in enumerate(plan, 1):
            if a.max_steps and step > a.max_steps:
                break

            def micro():
                for m in range(accumulation):
                    idx = (step - 1) * accumulation + m
                    batch = data.batch(indices[m:m + 1], seed_for(rc["seed"], "paper_data_augmentation", idx))
                    batch = {k: v.cuda() for k, v in batch.items()}
                    seed_all(seed_for(rc["seed"], "paper_diffusion", idx))
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, _ = model.shared_step(batch)
                    yield loss
            losses, stats = accumulated_update(opt, micro(), accumulation, rc["gradient_clip"])
            row = {"step": step, "denoising_loss": sum(losses) / accumulation, "elapsed_seconds": time.monotonic() - t0, **stats}
            mf.write(json.dumps(row) + "\n"); mf.flush(); print(json.dumps(row), flush=True)
            if step in rc["save_steps"]:
                save(opt, model, rc, out / f"step{step:06d}.safetensors", step)
    write_json(out / "complete.json", {"optimizer_steps": rc["optimizer_steps"], "index_plan_sha256": plan_sha,
                                       "checkpoints": [f"step{s:06d}.safetensors" for s in rc["save_steps"]]})


def save(opt, model, rc, path, step):
    from safetensors.torch import save_file
    state = opt.inference_state(use_ema=True)
    if rc["interface"] == "standardised":
        book = model.conditioner.embedders[2]
        state[INTERFACE_KEY] = (book.mu.detach().cpu() + book.sigma.detach().cpu() * state[INTERFACE_KEY].clone()).float().contiguous()
    assert all(torch.isfinite(v).all().item() for v in state.values())
    save_file({k: v.contiguous() for k, v in state.items()}, str(path), metadata={"optimizer_steps": str(step)})


if __name__ == "__main__":
    main()

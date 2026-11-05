"""Train and evaluate one scene-leakage decoder (Table 4): one dataset x one representation x one seed.

A decoder (leakage/decoder.py, the same for every representation) learns to reconstruct the frame after the action from
the action representation alone, with L1 + 0.5 x LPIPS for 10 epochs at 128x128. It is validated after every epoch on
clips from scenes held out from its training (leakage/split.py; the seed picks the held-out scenes, the initialisation
and the data order). Table 4 reports the last epoch's validation LPIPS / SSIM / L1, averaged over seeds 0-3.

    python -m leakage.train --dataset libero --representation onlyact \
        --clips data/probe/libero.json --data_root $FRAMES128/libero \
        --emb_dir $EMB/onlyact/libero --seed 0 --out_dir runs/table4/libero/onlyact/seed0

--data_root is the folder of the 128x128 frames, laid out with the clip list's relative frame paths.
Writes to --out_dir: scene_split.json, config.json, train_log.jsonl (training losses every 50 steps and one {"val": ...}
record per epoch), recon_ep<epoch>.png (16 held-out frames, top row ground truth, bottom row reconstruction) and
result.json (read by tables/table4_leakage.py). --save_checkpoint also keeps the final decoder weights (last.pt).
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from onlyact.data import load_manifest

from .dataset import LeakageDataset, collate
from .decoder import FrameDecoder
from .metrics import lpips_score, psnr_score, ssim_score
from .split import build_scene_split, save_split

RESOLUTION = 128


def build_loaders(args):
    tasks = load_manifest(args.clips, args.data_root)
    split = build_scene_split(tasks, seed=args.seed, val_frac=args.val_frac)
    split_path = os.path.join(args.out_dir, "scene_split.json")
    save_split(split, split_path)

    train_ds = LeakageDataset(split["train"], tasks, args.emb_dir, resolution=RESOLUTION)
    val_ds = LeakageDataset(split["val"], tasks, args.emb_dir, resolution=RESOLUTION)
    print(f"[data] train={len(train_ds)} samples  val={len(val_ds)} samples  "
          f"dim={train_ds.embed_dim}  val_scenes={split['meta']['n_val_scenes']}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate,
                            pin_memory=True, persistent_workers=args.num_workers > 0)
    return train_loader, val_loader, train_ds, val_ds, split, split_path


@torch.no_grad()
def evaluate(model, loader, device):
    """Mean (and spread) of the per-image metrics over the whole held-out set."""
    model.eval()
    lp_list, ss_list, l1_list, ps_list = [], [], [], []
    seen = 0
    for batch in loader:
        emb = batch["embedding"].to(device, non_blocking=True)
        frame = batch["frame"].to(device, non_blocking=True)
        pred = model(emb)
        lp_list.append(lpips_score(pred, frame, device=device).detach().cpu())
        ss_list.append(ssim_score(pred, frame).detach().cpu())
        l1_list.append((pred - frame).abs().mean(dim=(1, 2, 3)).detach().cpu())
        ps_list.append(psnr_score(pred, frame).detach().cpu())
        seen += emb.size(0)
    model.train()
    return {
        "lpips": float(torch.cat(lp_list).mean()),
        "lpips_std": float(torch.cat(lp_list).std()),
        "ssim": float(torch.cat(ss_list).mean()),
        "ssim_std": float(torch.cat(ss_list).std()),
        "l1": float(torch.cat(l1_list).mean()),
        "psnr": float(torch.cat(ps_list).mean()),
        "psnr_std": float(torch.cat(ps_list).std()),
        "n": seen,
    }


def save_viz(model, loader, device, path, n=16):
    model.eval()
    with torch.no_grad():
        for batch in loader:
            emb = batch["embedding"].to(device)
            frame = batch["frame"].to(device)
            pred = model(emb)
            pairs = torch.cat([frame[:n], pred[:n]], dim=0)
            save_image((pairs.clamp(-1, 1) + 1) / 2, path, nrow=n)
            break
    model.train()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="dataset name, recorded in result.json")
    p.add_argument("--representation", required=True, help="representation name, recorded in result.json")
    p.add_argument("--clips", required=True, help="clip list, e.g. data/probe/<dataset>.json")
    p.add_argument("--data_root", required=True, help="folder of the 128x128 frames (clip-list relative paths)")
    p.add_argument("--emb_dir", required=True, help="folder with <task_id>.npy embeddings of one representation")
    p.add_argument("--out_dir", required=True, help="new folder for this run")
    p.add_argument("--seed", type=int, default=0, help="decoder seed = split seed (the paper uses 0, 1, 2, 3)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lpips_weight", type=float, default=0.5)
    p.add_argument("--val_frac", type=float, default=0.2, help="fraction of scenes held out")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_checkpoint", action="store_true", help="save the final decoder as last.pt")
    args = p.parse_args()

    for name in ("train_log.jsonl", "result.json"):
        if os.path.exists(os.path.join(args.out_dir, name)):
            raise SystemExit(f"{args.out_dir} already holds a run ({name}); use a new --out_dir")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, val_loader, train_ds, val_ds, split, split_path = build_loaders(args)
    embed_dim = train_ds.embed_dim

    model = FrameDecoder(in_dim=embed_dim, out_res=RESOLUTION).to(device)
    print(f"[model] params={model.num_params/1e6:.2f}M  in_dim={embed_dim}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))
    total_steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    # LPIPS network of the training loss (frozen)
    import lpips
    lpips_net = lpips.LPIPS(net="alex", verbose=False).to(device)
    for pp in lpips_net.parameters():
        pp.requires_grad_(False)
    lpips_net.eval()

    global_step = 0
    metrics = None
    n_val_records = 0
    flog = open(os.path.join(args.out_dir, "train_log.jsonl"), "w")

    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in train_loader:
            emb = batch["embedding"].to(device, non_blocking=True)
            frame = batch["frame"].to(device, non_blocking=True)
            pred = model(emb)

            l1 = (pred - frame).abs().mean()
            lp = lpips_net(pred, frame).mean()
            loss = l1 + args.lpips_weight * lp

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            if global_step % args.log_every == 0:
                msg = {"step": global_step, "epoch": epoch,
                       "loss": float(loss), "l1": float(l1), "lpips": float(lp),
                       "lr": opt.param_groups[0]["lr"],
                       "t_min": (time.time() - t0) / 60}
                print(f"[train] step {global_step:6d}  ep {epoch:3d}  "
                      f"loss={msg['loss']:.4f}  l1={msg['l1']:.4f}  lp={msg['lpips']:.4f}  "
                      f"lr={msg['lr']:.2e}")
                flog.write(json.dumps(msg) + "\n"); flog.flush()

            global_step += 1

        metrics = evaluate(model, val_loader, device)
        metrics["epoch"] = epoch
        metrics["step"] = global_step
        print(f"[val]   ep {epoch:3d}  LPIPS={metrics['lpips']:.4f}±{metrics['lpips_std']:.3f}  "
              f"SSIM={metrics['ssim']:.4f}±{metrics['ssim_std']:.3f}  "
              f"L1={metrics['l1']:.4f}  n={metrics['n']}")
        flog.write(json.dumps({"val": metrics}) + "\n"); flog.flush()
        n_val_records += 1

        save_viz(model, val_loader, device, os.path.join(args.out_dir, f"recon_ep{epoch:03d}.png"), n=16)

    flog.close()
    train_minutes = (time.time() - t0) / 60
    if args.save_checkpoint:
        torch.save({"model": model.state_dict(), "args": vars(args), "metrics": metrics, "embed_dim": embed_dim},
                   os.path.join(args.out_dir, "last.pt"))

    with open(split_path, "rb") as f:
        split_md5 = hashlib.md5(f.read()).hexdigest()
    result = {
        "dataset": args.dataset, "representation": args.representation, "seed": args.seed,
        "epochs": args.epochs, "epochs_completed": n_val_records,
        "split_md5": split_md5,
        "n_scenes_total": split["meta"]["n_scenes_total"], "n_val_scenes": split["meta"]["n_val_scenes"],
        "n_train_samples": len(train_ds), "n_val_samples": len(val_ds), "embed_dim": embed_dim,
        "train_minutes": train_minutes,
        "final_val": metrics,
    }
    tmp = os.path.join(args.out_dir, "result.json.tmp")
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, os.path.join(args.out_dir, "result.json"))
    print(f"[done] final val LPIPS = {metrics['lpips']:.4f}  SSIM = {metrics['ssim']:.4f}  L1 = {metrics['l1']:.4f}")


if __name__ == "__main__":
    main()

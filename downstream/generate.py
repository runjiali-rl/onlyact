"""Generate the test rollouts of one Table 3 cell (domain x arm x number of fine-tuning updates).

For every test case: six real context frames (0-5), one generated next frame (6) at 256 x 256; EulerEDM sampler with the
EDM-shift discretisation, 5 denoising steps, classifier-free guidance 1.05, context augmentation level 0.1, float16
autocast. The random state is reset to the case's seed (first 4 bytes of SHA256("paper-eval-v1:32:<task_id>:6")) before
EVERY generated action, so all actions and all arms share the same noise.
  discrete domains: the next frame is generated for every action of the vocabulary;
  nuScenes        : for the commanded (factual) action and for four "draws" = training-clip actions picked by
                    numpy Generator(PCG64(seed)).integers(0, 100, 4) (a draw equal to the factual action reuses it).
Writes <out>/results.jsonl (per-case PSNR of every generated action against the real frame 6, LPIPS and motion-region
PSNR of the commanded action, nuScenes control margins) and <out>/preds/NNN.npy (uint8 frames, actions in vocabulary
order; nuScenes: factual first). Scoring: downstream/score.py.

  python -m downstream.generate --config downstream/configs/dmlab.json --arm justact --data_root DATA/dmlab \
      [--finetuned runs/table3/dmlab/justact/step000800.safetensors] --out runs/table3/cells/dmlab/justact_step800
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .common import load_clips, load_config, sampling_seed, seed_all, sha256, write_json
from .control import empirical_control, make_case
from .data import frame_loader
from .interfaces import load_step0
from .model import build_pretrained, world_model_files, for_inference, install_interface, load_finetuned

MOTION_THR, MIN_MASK = 0.05, 0.01


def motion_psnr(pred, gt, prev):
    """PSNR over the pixels that change between real frames 5 and 6 (None when under 1 % of the image moves)."""
    mask = (gt - prev).abs().amax(0) > MOTION_THR
    frac = float(mask.float().mean())
    if frac < MIN_MASK:
        return None, frac
    mse = float(((pred - gt) ** 2)[:, mask].mean())
    return (10 * np.log10(1.0 / max(mse, 1e-12))), frac


def sample_next_frame(model, sampler, context, action, continuous):
    """Six float RGB context frames in [0, 1] + one action -> generated frame 6 (float32, CPU, [0, 1])."""
    from sample_utils import get_batch
    device = next(model.parameters()).device
    for embedder in model.conditioner.embedders:
        if hasattr(embedder, "skip_encode"): embedder.skip_encode = False
    previous = context.to(device) * 2 - 1
    z = model.encode_first_stage(torch.cat([previous, torch.zeros_like(previous[:1])]))
    raw = torch.tensor([action], dtype=torch.float32, device=device) if continuous else torch.tensor([[action]], dtype=torch.int64, device=device)
    value = {"cond_frames_without_noise": previous[[-1]], "cond_frames": previous[[-1]],
             "context_len": torch.tensor([6.], device=device), "context_aug": torch.tensor([.1], device=device), "raw_action": raw}
    keys = sorted({e.input_key for e in model.conditioner.embedders})
    batch, batch_uc = get_batch(keys, value)
    condition, uncondition = model.conditioner.get_unconditional_conditioning(batch, batch_uc=batch_uc,
        force_uc_zero_embeddings=["cond_frames_without_noise", "cond_frames", "raw_action"])

    def denoiser(x, sigma, c):
        return model.denoiser(model.model, x, sigma, c)
    try:
        sampled = sampler(denoiser, torch.randn_like(z), x_ori=z, cond=condition, uc=uncondition)
        decoded = model.decode_first_stage(sampled)
        assert decoded.shape == (7, 3, 256, 256) and torch.isfinite(decoded).all().item()
        return ((decoded[-1] + 1) / 2).clamp(0, 1).float().cpu()
    finally:
        for embedder in model.conditioner.embedders:
            if hasattr(embedder, "skip_encode"): embedder.skip_encode = False


def to_uint8(P):
    return (P * 255).round().clamp(0, 255).to(torch.uint8).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--data_root", required=True, help="folder the test list's clip / frame paths are relative to")
    ap.add_argument("--train_data_root", help="nuScenes only: folder of the training clips (default --data_root)")
    ap.add_argument("--wm_config", help="default: worldmodel/configs/<world model>.yaml")
    ap.add_argument("--wm_ckpt", help="default: checkpoints/worldmodel/<world model>.safetensors")
    ap.add_argument("--base_ckpt", help="default: checkpoints/adaworld/adaworld.safetensors")
    ap.add_argument("--whitening_stats", help="JustAct only; default: checkpoints/worldmodel/justact_whitening.safetensors")
    ap.add_argument("--step0", help="step-0 interface (default: checkpoints/downstream/step0/<domain>_<arm>.safetensors)")
    ap.add_argument("--finetuned", help="fine-tuned checkpoint from train.py; omit for the 0-update cell")
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0, help="first N cases only (smoke test)")
    ap.add_argument("--no_check_hashes", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); torch.set_num_threads(4)
    cfg = load_config(a.config); arm = cfg["arms"][a.arm]; dim = arm["native_dim"]; ev = cfg["generation"]
    continuous = cfg["action_type"] == "continuous"; actions = [int(x) for x in cfg["actions"]] if not continuous else None
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False); (out / "preds").mkdir()
    tasks = load_clips(cfg["test_list"], a.data_root)
    if a.limit: tasks = tasks[:a.limit]
    load = frame_loader(cfg, check=not a.no_check_hashes)
    bank = None
    if continuous:  # action bank of the draws: the 100 training clips' last-transition actions, in list order
        bank = [{"raw_action": t["actions_ego_xy"][5]} for t in load_clips(cfg["train_list"], a.train_data_root or a.data_root)]
    init = load_step0(cfg, a.arm, a.step0)

    seed_all(32)
    files = world_model_files(cfg, a.arm, a.wm_config, a.wm_ckpt, a.base_ckpt, a.whitening_stats)
    model = build_pretrained(files, dim)
    install_interface(model, cfg, init, dim)
    restored = load_finetuned(model, a.finetuned) if a.finetuned else 0
    model = for_inference(model).cuda()
    from sample_utils import init_sampling
    import piq
    sampler = init_sampling(steps=ev["steps"], cfg_scale=ev["cfg_scale"], n_context_frames=ev["context_frames"])
    lpips = piq.LPIPS(reduction="none").eval()
    target_first = ev["metric_argument_order"] == "target_first"
    write_json(out / "start.json", {"domain": cfg["domain"], "arm": a.arm, "finetuned": a.finetuned,
                                    "finetuned_sha256": sha256(a.finetuned) if a.finetuned else None, "restored_tensors": restored,
                                    "generation": ev, "test_list": cfg["test_list"]})

    def psnr(gt, p):
        if target_first: return float(piq.psnr(gt[None], p[None], reduction="none").item())
        return float(piq.psnr(p[None], gt[None], data_range=1., reduction="none").item())

    def lp(gt, p):
        return float((lpips(gt[None], p[None]) if target_first else lpips(p[None], gt[None])).item())

    with open(out / "results.jsonl", "w") as stream:
        for index, task in enumerate(tasks):
            frames = load(task); gt, prev = frames[6], frames[5]; seed = sampling_seed(task["task_id"])
            if continuous:
                row = continuous_case(model, sampler, task, frames, seed, bank, psnr, lp, index, out)
            else:
                preds = []
                for act in actions:
                    seed_all(seed)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                        preds.append(sample_next_frame(model, sampler, frames[:6], int(act), False))
                P = torch.stack(preds); np.save(out / "preds" / f"{index:03d}.npy", to_uint8(P))
                ci = actions.index(int(task["raw_action"])); pc = P[ci]
                with torch.inference_mode():
                    psnr_all = [psnr(gt, P[j]) for j in range(len(actions))]
                    lpv = lp(gt, pc)
                mp, frac = motion_psnr(pc, gt, prev)
                row = {"index": index, "task_id": task["task_id"], "episode_id": task["episode_id"],
                       **({"scene_id": task["scene_id"]} if "scene_id" in task else {}), "raw_action": int(task["raw_action"]),
                       "sampling_seed": seed, "psnr": psnr_all[ci], "lpips": lpv, "psnr_by_action": dict(zip(map(str, actions), psnr_all)),
                       "motion_psnr": mp, "motion_mask_frac": frac}
            stream.write(json.dumps(row) + "\n"); stream.flush()
            if (index + 1) % 25 == 0: print(json.dumps({"arm": a.arm, "completed": index + 1, "total": len(tasks)}), flush=True)
    write_json(out / "complete.json", {"count": len(tasks), "results_sha256": sha256(out / "results.jsonl")})


def continuous_case(model, sampler, task, frames, seed, bank, psnr, lp, index, out):
    target, prev = frames[6], frames[5]
    factual = [float(x) for x in task["actions_ego_xy"][5]]
    draw = [int(i) for i in np.random.Generator(np.random.PCG64(seed)).integers(0, 100, size=4)]
    case = make_case(task, bank, draw)
    actions = [factual] + [list(x) for x in case["unique_new_actions"]]; preds, scores = [], []
    for act in actions:
        seed_all(seed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            p = sample_next_frame(model, sampler, frames[:6], act, True)
        with torch.inference_mode():
            scores.append({"psnr": psnr(target, p), "lpips": lp(target, p)})
        preds.append(p)
    mp, frac = motion_psnr(preds[0], target, prev)
    mc4 = empirical_control(case, scores[0], [{"unique_prediction_index": i, "raw_action": actions[i + 1], "metrics": scores[i + 1]} for i in range(len(actions) - 1)])
    np.save(out / "preds" / f"{index:03d}.npy", to_uint8(torch.stack(preds)))
    return {"task_id": task["task_id"], "index": index, "log_token": task["log_token"], "scene_token": task["scene_token"], "sampling_seed": seed,
            "raw_action": factual, "psnr": scores[0]["psnr"], "lpips": scores[0]["lpips"], "motion_psnr": mp, "motion_fraction": frac,
            "generated_actions": actions, "scores_by_action": scores, "draw_bank_indices": draw, "mc4": mc4}


if __name__ == "__main__":
    main()

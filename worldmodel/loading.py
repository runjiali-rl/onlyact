"""Building the Table 2 world models: AdaWorld base weights, LoRA adapters and the released trainable weights.

All four world models start from the released AdaWorld weights (Hugging Face Little-Podi/AdaWorld, adaworld.safetensors)
with the action pathway zeroed: AdaWorld's latent-action columns of the vector-conditioning layer (label_emb.0.0) and of
every cross-attention key / value projection are set to zero, the remaining columns are copied, and the input widths are
set to the width of the external action vector (vector input 512 + D, cross-attention context 1024 + D). AdaWorld's
latent-action encoder and EMA weights are not loaded. Training then adds rank-128 LoRA adapters to the UNet attention
projections and trains them together with label_emb; a released world-model checkpoint holds exactly those tensors.
"""
from typing import List, Optional

import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file

WHITENED_EMBEDDER = "vwm.modules.encoders.external_action.WhitenedExternalActionEmbedder"
LORA_TARGETS = "to_q,to_k,to_v,to_out.0"


def set_whitening_stats(cfg, path: Optional[str]) -> bool:
    """Point the whitened action embedder of a config at a statistics file. Returns True if the config has one."""
    found = False
    for emb in cfg.model.params.conditioner_config.params.emb_models:
        if emb.target == WHITENED_EMBEDDER:
            if not path:
                raise ValueError("this world model uses per-provider whitening: pass --whitening_stats")
            emb.params.whitening_stats = path
            found = True
    return found


def slot_widths(cfg):
    net = cfg.model.params.network_config.params
    return int(net.get("adm_in_channels", 544)), int(net.get("context_dim", 1056))


def load_adaworld_base(model, path: str, adm_w: int, ctx_w: int):
    """Stream the released AdaWorld weights into `model` with the action pathway zeroed (host peak = model + one tensor)."""
    own = dict(model.named_parameters())
    own.update(dict(model.named_buffers()))
    missing, unexpected, n, n_remap = set(own), [], 0, 0
    with torch.no_grad(), safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith("model_ema") or ".lam." in k:
                continue
            if k not in own:
                unexpected.append(k)
                continue
            t = f.get_tensor(k)
            if k == "model.diffusion_model.label_emb.0.0.weight" and t.shape[1] == 544:
                nw = torch.zeros(t.shape[0], adm_w, dtype=t.dtype)
                nw[:, adm_w - 512:] = t[:, 32:]
                t = nw
                n_remap += 1
            elif ("to_k.weight" in k or "to_v.weight" in k) and t.shape[1] == 1056:
                nw = torch.zeros(t.shape[0], ctx_w, dtype=t.dtype)
                nw[:, :1024] = t[:, :1024]
                t = nw
                n_remap += 1
            if tuple(t.shape) != tuple(own[k].shape):
                raise RuntimeError(f"shape mismatch for {k}: {tuple(t.shape)} vs model {tuple(own[k].shape)}")
            own[k].copy_(t)
            missing.discard(k)
            n += 1
            del t
    print(f"[loading] AdaWorld base: {n} tensors from {path}, {n_remap} action-slot tensors zeroed / remapped to widths "
          f"{adm_w}/{ctx_w}")
    return n, sorted(missing), unexpected


def inject_lora(model, rank: int, alpha: int = 0, targets: str = LORA_TARGETS) -> List[str]:
    from peft import LoraConfig, inject_adapter_in_model
    target_modules = [t for t in targets.split(",") if t]
    inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=alpha or rank, lora_dropout=0.0, bias="none",
                                       target_modules=target_modules), model.model.diffusion_model)
    return target_modules


def load_trainable(model, path: str) -> int:
    """Copy a released world-model checkpoint (LoRA weights + label_emb, safetensors) into the model's parameters."""
    sd = load_file(path)
    own = dict(model.named_parameters())
    absent = [k for k in sd if k not in own]
    if absent:
        raise RuntimeError(f"{len(absent)} checkpoint tensors have no slot in the model (LoRA rank / targets mismatch?): "
                           f"{absent[:3]}")
    with torch.no_grad():
        for k, v in sd.items():
            own[k].copy_(v)
    print(f"[loading] world-model checkpoint: {len(sd)} tensors from {path}")
    return len(sd)


def load_config(path: str, whitening_stats: Optional[str] = None):
    cfg = OmegaConf.load(path)
    set_whitening_stats(cfg, whitening_stats)
    return cfg

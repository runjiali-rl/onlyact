"""Pretrained world model -> fine-tunable model with a new action interface (Table 3).

Construction (identical for fine-tuning and generation):
  1. the AdaWorld UNet / autoencoder / CLIP from the world-model config, public AdaWorld weights loaded with the action
     pathway zeroed and the condition widths set to the arm's action width;
  2. the arm's pretrained world model: LoRA (rank 128 on to_q / to_k / to_v / to_out.0) + action input layer (label_emb),
     restored exactly, then the LoRA merged into the FP32 weights;
  3. the action slot (conditioner embedder 2) replaced by the new interface:
       discrete   : an action book, one row per action, initialised with the step-0 book (per-action mean of the arm's
                    own action features of the labelled clips, after the world model's input transform);
       continuous : a two-layer head Linear(2, d) - SiLU - Linear(d, d) from (forward, lateral) metres, initialised with
                    the fitted step-0 head.
For fine-tuning the whole UNet and the interface are trainable (FP32 weights, activation checkpointing).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from .common import REPO

LORA = dict(r=128, lora_alpha=128, lora_dropout=0., bias="none", target_modules=["to_q", "to_k", "to_v", "to_out.0"])


def worldmodel_path():
    """Make the world-model packages of worldmodel/ (vwm, loading, sample_utils) importable."""
    root = REPO / "worldmodel"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def world_model_files(cfg, arm, wm_config=None, wm_ckpt=None, base_ckpt=None, whitening_stats=None):
    """Default locations of the arm's pretrained world model (worldmodel/configs, checkpoints/worldmodel, checkpoints/adaworld)."""
    name = cfg["arms"][arm]["world_model"]
    return {"wm_config": wm_config or str(REPO / "worldmodel" / "configs" / f"{name}.yaml"),
            "wm_ckpt": wm_ckpt or str(REPO / "checkpoints" / "worldmodel" / f"{name}.safetensors"),
            "base_ckpt": base_ckpt or str(REPO / "checkpoints" / "adaworld" / "adaworld.safetensors"),
            "whitening_stats": whitening_stats or str(REPO / "checkpoints" / "worldmodel" / "justact_whitening.safetensors")}


def _config(wm_config, whitening_stats):
    """World-model config with the CLIP image encoder built empty on CPU (its weights come from the base checkpoint)."""
    worldmodel_path()
    from loading import load_config
    cfg = load_config(wm_config, whitening_stats)
    clips = [c for c in cfg.model.params.conditioner_config.params.emb_models if c.get("input_key") == "cond_frames_without_noise"]
    assert len(clips) == 1
    clips[0].params.open_clip_config.params.version = None
    clips[0].params.open_clip_config.params.init_device = "cpu"
    return cfg


def _persistent_names(model):
    out = set()
    for prefix, module in model.named_modules():
        for name, buf in module._buffers.items():
            if buf is not None and name not in module._non_persistent_buffers_set:
                out.add((prefix + "." if prefix else "") + name)
    return out


def pretrained_parameter_names(model):
    """Tensors of the pretrained world model on top of the base weights: LoRA and the action input layer."""
    return [n for n, _ in model.named_parameters() if n.startswith("model.diffusion_model.") and ("lora_" in n or ".label_emb." in n)]


def merge_lora(unet):
    """Merge every LoRA layer into its FP32 base Linear and remove the wrapper (checked to equal base + delta)."""
    from peft.tuners.lora.layer import LoraLayer
    merged = 0
    for name, layer in list(unet.named_modules()):
        if not isinstance(layer, LoraLayer):
            continue
        adapters = list(layer.active_adapters); base = layer.get_base_layer()
        assert adapters == ["default"] and isinstance(base, nn.Linear) and base.weight.dtype == torch.float32
        expected = base.weight.detach().clone()
        for adapter in adapters:
            expected.add_(layer.get_delta_weight(adapter))
        layer.merge(safe_merge=True, adapter_names=adapters)
        assert torch.equal(base.weight.detach(), expected), "merged weight differs from base + LoRA: " + name
        parent_name, _, attr = name.rpartition(".")
        setattr(unet.get_submodule(parent_name) if parent_name else unet, attr, base); merged += 1
    assert merged and not any("lora_" in n or ".base_layer." in n for n, _ in unet.named_parameters())
    return merged


def build_pretrained(files, native_dim):
    """Steps 1-2: the arm's pretrained world model with the LoRA merged (action slot still the original one)."""
    worldmodel_path()
    from safetensors.torch import load_file
    from vwm.util import instantiate_from_config
    from loading import inject_lora, load_adaworld_base, load_trainable, slot_widths
    cfg = _config(files["wm_config"], files["whitening_stats"]); adm_w, ctx_w = slot_widths(cfg)
    assert adm_w == native_dim + 512 and ctx_w == native_dim + 1024, "config / action width mismatch"
    model = instantiate_from_config(cfg.model)
    persistent = _persistent_names(model)
    _, missing, _ = load_adaworld_base(model, files["base_ckpt"], adm_w, ctx_w)
    assert not set(missing) & (set(dict(model.named_parameters())) | persistent), "base checkpoint does not cover the model"
    inject_lora(model, LORA["r"], LORA["lora_alpha"], ",".join(LORA["target_modules"]))
    keys = set(load_file(files["wm_ckpt"]))
    assert keys == set(pretrained_parameter_names(model)), "world-model checkpoint keys differ from LoRA + action input layer"
    load_trainable(model, files["wm_ckpt"])
    merge_lora(model.model.diffusion_model)
    return model


def original_action_embedder(files, native_dim):
    """The pretrained world model's own action input transform (per-provider whitening for JustAct, identity for the LAMs)."""
    worldmodel_path()
    from vwm.util import instantiate_from_config
    rows = [c for c in _config(files["wm_config"], files["whitening_stats"]).model.params.conditioner_config.params.emb_models
            if c.get("input_key") == "ext_action"]
    assert len(rows) == 1
    e = instantiate_from_config(rows[0]); assert e.in_dim == native_dim
    return e.eval()


class ActionBook(nn.Module):
    """Discrete interface: one trainable row per action; the row is the final native condition."""

    def __init__(self, weights, actions, ucg_rate=0.1):
        super().__init__()
        assert weights.ndim == 2 and weights.dtype == torch.float32 and len(actions) == weights.shape[0]
        self.codebook = nn.Embedding.from_pretrained(weights.clone(), freeze=False)
        self.register_buffer("action_ids", torch.tensor(actions, dtype=torch.int64))
        self.input_key = "raw_action"; self.is_trainable = True; self.ucg_rate = float(ucg_rate); self.legacy_ucg_val = None

    def rows(self, action):
        match = action.reshape(-1, 1) == self.action_ids.reshape(1, -1)
        assert (match.sum(1) == 1).all().item(), "action not in vocabulary"
        return self.codebook(match.long().argmax(1))

    def forward(self, action):
        z = self.rows(action)
        return [z.unsqueeze(1), z]


class StandardisedActionBook(ActionBook):
    """Standardised interface: the book stores u and the condition is mean + std * u, with the per-dimension mean / std of
    the arm's own world-model training features. A learning rate on u is then a step in units of the arm's own standard
    deviation, the same relative step for every arm. Folded back to native rows (mean + std * u) when saved."""

    def __init__(self, native_rows, actions, ucg_rate, mean, std):
        super().__init__(native_rows.float(), actions, ucg_rate)
        self.register_buffer("mu", mean.float()); self.register_buffer("sigma", std.float())
        u = ((native_rows.float() - self.mu) / self.sigma).contiguous()
        self.codebook = nn.Embedding.from_pretrained(u.clone(), freeze=False)

    def forward(self, action):
        z = self.mu + self.sigma * self.rows(action)
        return [z.unsqueeze(1), z]


class ActionHead(nn.Module):
    """Continuous interface (nuScenes): (forward, lateral) metres -> final native condition."""

    def __init__(self, dim, ucg_rate=0.1):
        super().__init__(); self.native_dim = int(dim)
        self.mapping = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.input_key = "raw_action"; self.is_trainable = True; self.ucg_rate = float(ucg_rate); self.legacy_ucg_val = None

    def forward(self, x):
        assert x.ndim == 2 and x.shape[1] == 2 and torch.isfinite(x).all().item()
        z = self.mapping(x.float())
        return [z.unsqueeze(1), z]


def install_interface(model, cfg, init, native_dim, standardise=None):
    """Step 3: replace the action slot. `init` = step-0 book (A, d) or step-0 head state dict."""
    old = model.conditioner.embedders[2]
    assert old.input_key == "ext_action" and old.in_dim == native_dim
    if cfg["action_type"] == "continuous":
        head = ActionHead(native_dim, old.ucg_rate); head.load_state_dict(init, strict=True); model.conditioner.embedders[2] = head
    else:
        model.conditioner.embedders[2] = ActionBook(init, [int(a) for a in cfg["actions"]], old.ucg_rate)
        if standardise is not None:
            book = model.conditioner.embedders[2]
            model.conditioner.embedders[2] = StandardisedActionBook(book.codebook.weight.detach().cpu(), [int(a) for a in cfg["actions"]],
                                                                    book.ucg_rate, standardise["mean"], standardise["std"])
    return model


def enable_checkpointing(model):
    for _, module in model.model.diffusion_model.named_modules():
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = True


def trainable_groups(model, unet_lr, interface_lr):
    """Whole UNet + the new interface, FP32; two learning-rate groups."""
    for p in model.parameters():
        p.requires_grad_(False)
    unet = list(model.model.diffusion_model.named_parameters()); new = list(model.conditioner.embedders[2].named_parameters())
    named = [("model.diffusion_model." + n, p) for n, p in unet] + [("conditioner.embedders.2." + n, p) for n, p in new]
    for _, p in named:
        assert p.dtype == torch.float32
        p.requires_grad_(True)
    groups = [{"name": "full_unet", "params": [p for _, p in unet], "lr": unet_lr},
              {"name": "interface", "params": [p for _, p in new], "lr": interface_lr}]
    return named, groups


def for_inference(model):
    for module in model.modules():
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = False
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def load_finetuned(model, path):
    """Restore a fine-tuned checkpoint (EMA UNet + interface, FP32) written by train.py."""
    from safetensors.torch import load_file
    state = load_file(str(path)); own = dict(model.named_parameters())
    expected = {"model.diffusion_model." + n for n, _ in model.model.diffusion_model.named_parameters()}
    expected |= {"conditioner.embedders.2." + n for n, _ in model.conditioner.embedders[2].named_parameters()}
    assert set(state) == expected, "fine-tuned checkpoint keys differ from UNet + interface"
    with torch.no_grad():
        for k, v in state.items():
            assert v.dtype == torch.float32 and v.shape == own[k].shape and torch.isfinite(v).all().item(), k
            own[k].copy_(v)
    return len(state)

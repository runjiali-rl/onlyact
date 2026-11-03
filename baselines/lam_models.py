"""Shared pieces of the two retrained latent-action baselines, AdaWorld-LAM and DreamDojo-LAM.

Used by train.py and extract.py (the retrained baseline rows of Tables 1-4): where the vendored upstream
network code lives, how its hyper-parameters are read from the upstream YAML, and how a network is built
and filled from a safetensors checkpoint.

The upstream code is vendored unmodified under third_party/ (see NOTICE.md):
  adaworld   third_party/adaworld_lam/lam/lam/modules       package `lam`; used for training and extraction
  dreamdojo  third_party/dreamdojo_lam/external/lam_project/lam/modules
                 package `lam`, the training copy: forward() also decodes frame t+1 and encode() samples the
                 latent in training mode
             third_party/dreamdojo_lam/external/lam/modules
                 package `external.lam.modules`, the inference copy used for extraction: encode() only, the
                 latent is always z_mu. Both copies declare the same parameters, so one checkpoint fits both.
Both packages called `lam` are imported by top-level name, so one process only ever loads one architecture.
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Dict

import torch
import yaml
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY = os.path.join(HERE, "third_party")

ARCHS: Dict[str, Dict[str, Any]] = {
    "adaworld": dict(
        config_yaml=os.path.join(THIRD_PARTY, "adaworld_lam", "lam", "config", "lam.yaml"),
        train_path=os.path.join(THIRD_PARTY, "adaworld_lam", "lam"),
        train_module="lam.modules.lam",
        extract_path=os.path.join(THIRD_PARTY, "adaworld_lam", "lam"),
        extract_module="lam.modules.lam",
        extract_blocks_module="lam.modules.blocks",
        # The extraction network is built on the host and moved to the device after loading.
        build_on_device=False,
        # Training geometry of AdaWorld lam/dataset.py: square centre crop, bicubic resize to
        # data.resolution (256) of the upstream YAML.
        crop_aspect=None,
        resize_hw=None,
        interp_mode="bicubic",
        representation="adaworld_lam_retrained",
    ),
    "dreamdojo": dict(
        config_yaml=os.path.join(THIRD_PARTY, "dreamdojo_lam", "external", "lam_project", "config", "lam_0.yaml"),
        train_path=os.path.join(THIRD_PARTY, "dreamdojo_lam", "external", "lam_project"),
        train_module="lam.modules",
        extract_path=os.path.join(THIRD_PARTY, "dreamdojo_lam"),
        extract_module="external.lam.modules",
        extract_blocks_module="external.lam.modules.blocks",
        # The extraction network is built directly on the device.
        build_on_device=True,
        # Training geometry of DreamDojo lam_project/lam/dataset.py (the upstream YAML has no resolution):
        # centre crop to width / height = 640 / 480, bilinear resize to (height, width) = (240, 320).
        crop_aspect=640.0 / 480.0,
        resize_hw=(240, 320),
        interp_mode="bilinear",
        representation="dreamdojo_lam_retrained",
    ),
}


def read_upstream_config(arch: str) -> Dict[str, Any]:
    """Network size, beta, optimiser and gradient clipping from the vendored upstream YAML."""
    with open(ARCHS[arch]["config_yaml"]) as f:
        raw = yaml.safe_load(f)
    m = raw.get("model", {}) or {}
    d = raw.get("data", {}) or {}
    t = raw.get("trainer", {}) or {}
    opt = m.get("optimizer") or {}
    opt_args = (opt.get("init_args") or {}) if isinstance(opt, dict) else {}
    return dict(
        model_kwargs=dict(
            in_dim=int(m.get("image_channels", 3)),
            model_dim=int(m["lam_model_dim"]),
            latent_dim=int(m["lam_latent_dim"]),
            patch_size=int(m["lam_patch_size"]),
            enc_blocks=int(m["lam_enc_blocks"]),
            dec_blocks=int(m["lam_dec_blocks"]),
            num_heads=int(m["lam_num_heads"]),
            dropout=float(m.get("lam_dropout", 0.0)),
        ),
        beta=float(m["beta"]),
        # The YAML names torch.optim.AdamW; "1e-2" is read as a string by YAML, hence float().
        lr=float(opt_args.get("lr", 2.5e-5)),
        weight_decay=float(opt_args.get("weight_decay", 1e-2)),
        gradient_clip_val=float(t.get("gradient_clip_val", 0.0) or 0.0),
        resolution=int(d.get("resolution", 0)),
        num_frames=int(d.get("num_frames", 2)),
        output_format=str(d.get("output_format", "t h w c")),
        config_yaml=os.path.relpath(ARCHS[arch]["config_yaml"], HERE),
    )


def _import(path: str, module: str):
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module(module)


def training_network_class(arch: str):
    spec = ARCHS[arch]
    return getattr(_import(spec["train_path"], spec["train_module"]), "LatentActionModel")


def extraction_network_class(arch: str):
    spec = ARCHS[arch]
    return getattr(_import(spec["extract_path"], spec["extract_module"]), "LatentActionModel")


def allow_cpu(arch: str) -> None:
    """The upstream positional encoding adds its table with `.cuda()`, which needs a GPU. For a CPU run the
    same table is added on the input's device instead (identical values). GPU runs do not call this."""
    spec = ARCHS[arch]
    blocks = _import(spec["extract_path"], spec["extract_blocks_module"])

    def forward(self, x):
        return x + self.pos_enc[:x.shape[2]].to(x.device)

    blocks.PositionalEncoding.forward = forward


def build_extraction_model(arch: str, ckpt: str, device: str) -> torch.nn.Module:
    """The network extract.py runs, filled from a state-dict-only safetensors checkpoint, in eval mode.

    The device placement follows the upstream code paths exactly: the positional-encoding table is a plain
    tensor attribute computed at construction, so AdaWorld (built on the host) adds a host-computed table
    and DreamDojo (built on the device) a device-computed one."""
    spec = ARCHS[arch]
    net_cls = extraction_network_class(arch)
    kwargs = read_upstream_config(arch)["model_kwargs"]
    if spec["build_on_device"]:
        with torch.device(device):
            m = net_cls(**kwargs)
    else:
        m = net_cls(**kwargs)
    sd = load_file(ckpt)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    n_par = sum(p.numel() for p in m.parameters())
    print(f"[lam] {arch}: loaded {len(sd)} tensors into a {n_par / 1e6:.1f} M-parameter latent action model | "
          f"{len(missing)} missing, {len(unexpected)} unexpected", flush=True)
    assert not missing and not unexpected, f"checkpoint does not match the network: {list(missing)[:6]} {list(unexpected)[:6]}"
    del sd
    if not spec["build_on_device"]:
        m = m.to(device)
    return m.eval()


def save_weights(path: str, model: torch.nn.Module) -> None:
    """State dict only, float32, the network's own parameter names (no prefix): the format extract.py loads."""
    sd = {k: v.detach().to(torch.float32).cpu().contiguous() for k, v in model.state_dict().items()}
    tmp = path + ".tmp"
    save_file(sd, tmp)
    os.replace(tmp, path)

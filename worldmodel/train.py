"""Train one Table 2 world model: AdaWorld conditioned on an external per-transition action vector.

Recipe (identical for the four world models; only the action input differs): start from the released AdaWorld weights
with the action pathway zeroed (loading.py), add rank-128 LoRA adapters to the UNet attention projections
(to_q, to_k, to_v, to_out.0) and train them together with the vector-conditioning MLP (label_emb) for 20,000 steps at
batch 4, 256 x 256 pixels, learning rate 5e-5 with a 1,000-step warm-up, bf16 mixed precision, gradient clipping 0.3,
seed 32, on the 19,946-clip training pool. Every --save_trainable_every steps and at the end only the trainable tensors
are written to <logdir>/<name>/checkpoints/trainable_step*.pt (tools/export_checkpoint.py turns one into the released
.safetensors format). One run takes about 5 hours and 19 GB on one 24 GB GPU.

    python train.py --base configs/onlyact.yaml --name onlyact --data_root $POOL --emb_root $EMB

Config values can be overridden with OmegaConf dot-list arguments, e.g. lightning.trainer.max_steps=50 for a smoke test.
Adapted from AdaWorld's worldmodel/train.py (Apache-2.0).
"""
import argparse
import datetime
import os
import sys

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_only

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from clip_lists import read_clip_list  # noqa: E402
from loading import LORA_TARGETS, load_adaworld_base, set_whitening_stats, slot_widths  # noqa: E402
from vwm.util import instantiate_from_config  # noqa: E402


class TrainableOnlyCheckpoint(Callback):
    """Saves only the parameters with requires_grad (LoRA weights and label_emb)."""

    def __init__(self, ckptdir: str, every_n_steps: int):
        super().__init__()
        self.ckptdir, self.every = ckptdir, int(every_n_steps)
        self.t0 = datetime.datetime.now()

    @rank_zero_only
    def _save(self, pl_module, step: int, tag: str = ""):
        os.makedirs(self.ckptdir, exist_ok=True)
        sd = {n: p.detach().cpu() for n, p in pl_module.named_parameters() if p.requires_grad}
        path = os.path.join(self.ckptdir, f"trainable_step{step:06d}{tag}.pt")
        torch.save({"state_dict": sd, "global_step": step,
                    "elapsed_min": (datetime.datetime.now() - self.t0).total_seconds() / 60}, path)
        peak = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0
        print(f"[trainable-ckpt] step {step}: {len(sd)} tensors -> {path} (peak GPU allocated {peak:.0f} MiB)")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step > 0 and step % self.every == 0 and not getattr(self, "_last", None) == step:
            self._last = step
            self._save(pl_module, step)

    def on_train_end(self, trainer, pl_module):
        self._save(pl_module, trainer.global_step, "_end")


class SetupCallback(Callback):
    """Creates the run folders and saves the resolved configs (from AdaWorld)."""

    def __init__(self, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.now, self.logdir, self.ckptdir, self.cfgdir = now, logdir, ckptdir, cfgdir
        self.config, self.lightning_config = config, lightning_config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config, os.path.join(self.cfgdir, f"{self.now}-project.yaml"))
            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, f"{self.now}-lightning.yaml"))


def get_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-b", "--base", required=True, help="world-model config, configs/<arm>.yaml")
    p.add_argument("-n", "--name", required=True, help="run name; the run folder is <logdir>/<name>")
    p.add_argument("-l", "--logdir", default="logs")
    p.add_argument("-s", "--seed", type=int, default=32)
    p.add_argument("--n_devices", type=int, default=1)
    p.add_argument("--adaworld_ckpt", default=os.path.join(REPO, "checkpoints", "adaworld", "adaworld.safetensors"),
                   help="released AdaWorld weights (Hugging Face Little-Podi/AdaWorld); loaded with the action pathway zeroed")
    p.add_argument("--train_list", default=os.path.join(REPO, "data", "worldmodel", "train_pool.json.gz"),
                   help="training clip list (relative frame paths)")
    p.add_argument("--data_root", required=True, help="root that the training clip list's frame paths start from")
    p.add_argument("--emb_root", default="", help="action embeddings, <emb_root>/<representation>/<dataset>/<task_id>.npy")
    p.add_argument("--whitening_stats", default=os.path.join(REPO, "checkpoints", "worldmodel", "onlyact_whitening.safetensors"),
                   help="per-provider whitening statistics (OnlyAct world model only)")
    p.add_argument("--lora_rank", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=0, help="0 = rank")
    p.add_argument("--lora_targets", default=LORA_TARGETS, help="comma-separated module-name suffixes that receive LoRA")
    p.add_argument("--save_trainable_every", type=int, default=2000)
    return p


def build_config(opt, unknown):
    """The world-model config with the run-time inputs filled in: training clip list, data root, embeddings, whitening."""
    config = OmegaConf.merge(OmegaConf.load(opt.base), OmegaConf.from_dotlist(unknown))
    src = config.data.params.sources[0]
    representation = config.data.params.representation
    src.manifest, src.data_root = opt.train_list, opt.data_root
    if representation is not None:
        if not opt.emb_root:
            raise ValueError(f"representation {representation}: pass --emb_root")
        src.emb_dir = os.path.join(opt.emb_root, representation, read_clip_list(opt.train_list)["dataset"])
    set_whitening_stats(config, opt.whitening_stats)
    return config


def main():
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    opt, unknown = get_parser().parse_known_args()
    logdir = os.path.join(opt.logdir, opt.name)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed, workers=True)

    config = build_config(opt, unknown)

    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config["accelerator"] = "gpu"
    trainer_config["devices"] = ",".join(str(i) for i in range(opt.n_devices)) + ","
    print(f"Running on GPUs {trainer_config['devices']}")
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # Model: AdaWorld with the action pathway zeroed, then LoRA; only LoRA and label_emb train
    model = instantiate_from_config(config.model)
    adm_w, ctx_w = slot_widths(config)
    _, missing, unexpected = load_adaworld_base(model, opt.adaworld_ckpt, adm_w, ctx_w)
    print(f"Missing keys: {[k for k in missing if '.lam.' not in k]}")
    print(f"Unexpected keys: {unexpected}")
    from peft import LoraConfig, inject_adapter_in_model
    targets = [t for t in opt.lora_targets.split(",") if t]
    inject_adapter_in_model(LoraConfig(r=opt.lora_rank, lora_alpha=opt.lora_alpha or opt.lora_rank, lora_dropout=0.0,
                                       bias="none", target_modules=targets), model.model.diffusion_model)
    n_lora = n_cond = 0
    for n_, p_ in model.model.named_parameters():
        is_lora = "lora_" in n_
        is_cond = ".label_emb." in n_
        p_.requires_grad = bool(is_lora or is_cond)
        n_lora += p_.numel() if is_lora else 0
        n_cond += p_.numel() if is_cond else 0
    n_emb = sum(p_.numel() for e_ in model.conditioner.embedders if e_.is_trainable for p_ in e_.parameters() if p_.requires_grad)
    n_all = sum(p_.numel() for p_ in model.parameters())
    print(f"[lora] rank {opt.lora_rank} on {targets}: trainable LoRA {n_lora / 1e6:.2f}M, label_emb {n_cond / 1e6:.2f}M, "
          f"embedders {n_emb / 1e6:.3f}M of {n_all / 1e6:.0f}M total parameters")

    # Trainer: CSV logger, learning-rate monitor, no full-model checkpoints (save_top_k 0), trainable-only checkpoints
    logger = pl.loggers.CSVLogger(name="testtube", save_dir=logdir)
    modelckpt_cfg = OmegaConf.merge({"dirpath": ckptdir, "filename": "{epoch:02}", "verbose": True, "save_last": False,
                                     "save_top_k": -1}, lightning_config.get("modelcheckpoint", {}).get("params", {}))
    callbacks = [SetupCallback(now, logdir, ckptdir, cfgdir, config, lightning_config),
                 pl.callbacks.LearningRateMonitor(logging_interval="step"),
                 pl.callbacks.ModelCheckpoint(**modelckpt_cfg),
                 TrainableOnlyCheckpoint(ckptdir, opt.save_trainable_every)]
    trainer = Trainer(**vars(trainer_opt), logger=logger, callbacks=callbacks, plugins=[])
    trainer.logdir = logdir

    data = instantiate_from_config(config.data)
    model.learning_rate = config.model.base_learning_rate
    print(f"Setting learning rate to {model.learning_rate:.2e} (batch size {config.data.params.batch_size}, "
          f"accumulate_grad_batches {lightning_config.trainer.get('accumulate_grad_batches', 1)})")
    trainer.fit(model, data)


if __name__ == "__main__":
    main()

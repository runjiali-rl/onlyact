"""AdamW with its moments and the exponential-moving-average (EMA) copy of the UNet kept on the CPU in FP32 (Table 3).

The 1.56-billion-parameter UNet is trained in FP32 on a 24 GB GPU: after each accumulated update the gradient of one
tensor at a time is moved to the CPU, clipped (global norm, computed first), stepped with torch.optim.AdamW
(foreach=False) and copied back. Host memory: two Adam moments + one EMA copy of the UNet, about 17 GiB.
"""
from __future__ import annotations

import torch
from torch import nn


class CPUAdamW:
    def __init__(self, named_parameters, groups, *, betas, eps, weight_decay, warmup_steps, warmup_initial_factor, ema_decay, ema=True):
        self.named = list(named_parameters)
        self.by_id = {id(p): n for n, p in self.named}
        self.base_lrs = {}
        for group in groups:
            self.base_lrs.update({self.by_id[id(p)]: group["lr"] for p in group["params"]})
        assert set(self.base_lrs) == {n for n, _ in self.named}
        self.warmup_steps, self.warmup_initial_factor, self.ema_decay = warmup_steps, warmup_initial_factor, ema_decay
        self.optimizer = torch.optim.AdamW([nn.Parameter(torch.empty(0))], lr=1., betas=tuple(betas), eps=eps, weight_decay=weight_decay, foreach=False)
        self.optimizer.param_groups[0]["params"] = []
        self.states = {}; self.steps = 0
        self.ema = {} if ema else None
        if ema:
            for n, p in self.named:
                if n.startswith("model.diffusion_model."):
                    self.ema[n] = p.detach().cpu().clone()

    def warmup_factor(self, completed_updates):
        """Linear warm-up from warmup_initial_factor to 1 over warmup_steps updates."""
        w0 = self.warmup_initial_factor
        return w0 + (1 - w0) * completed_updates / self.warmup_steps if completed_updates < self.warmup_steps else 1.0

    def zero_grad(self):
        for _, p in self.named:
            p.grad = None

    @torch.no_grad()
    def step(self, max_norm):
        missing = []
        for name, p in self.named:
            if p.grad is None:
                missing.append(name)
            elif not torch.isfinite(p.grad).all().item():
                raise FloatingPointError("non-finite accumulated gradient: " + name)
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in self.named], max_norm, error_if_nonfinite=True, foreach=False)
        factor = self.warmup_factor(self.steps)
        decay = min(self.ema_decay, (2 + self.steps) / (11 + self.steps))
        for name, p in self.named:
            value = nn.Parameter(p.detach().cpu().clone())
            if p.grad is not None:
                value.grad = p.grad.detach().cpu().clone()
                p.grad = None
                self.optimizer.param_groups[0].update(params=[value], lr=self.base_lrs[name] * factor)
                if name in self.states:
                    self.optimizer.state[value] = self.states[name]
                self.optimizer.step()
                self.states[name] = self.optimizer.state.pop(value)
                self.optimizer.param_groups[0]["params"] = []
            if self.ema is not None and name in self.ema:
                self.ema[name].sub_((1 - decay) * (self.ema[name] - value))
            p.copy_(value, non_blocking=False)
            del value
        self.steps += 1
        return {"step": self.steps, "grad_norm_before_clip": float(norm), "missing_gradients": len(missing),
                "lr_factor": factor, "ema_decay_this_update": decay if self.ema is not None else None}

    def inference_state(self, use_ema):
        """EMA UNet (if use_ema) + live interface tensors, on the CPU."""
        return {n: (self.ema[n] if use_ema and n in self.ema else p.detach().cpu()) for n, p in self.named}


def accumulated_update(optimizer, losses, accumulation, max_norm):
    """One optimizer update from `accumulation` micro-batch losses (each backpropagated as loss / accumulation)."""
    optimizer.zero_grad()
    values = []
    for loss in losses:
        if not torch.isfinite(loss).item():
            raise FloatingPointError("non-finite loss")
        values.append(float(loss.detach()))
        (loss / accumulation).backward()
    assert len(values) == accumulation
    return values, optimizer.step(max_norm)

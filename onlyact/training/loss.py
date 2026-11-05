"""Symmetric InfoNCE with hard negatives, the training objective of the OnlyAct encoder.

Inputs are L2-normalised embeddings, each [N, D]:
    anchors    A_i  the anchor pair
    positives  P_i  the background-replaced, core-shifted view of the anchor pair
    hard_negs  H_i  a pair from a disjoint window of the same clip

The bank is [A | P | H] (3N columns). For anchor A_i the target column is P_i and every other column except A_i itself
is a negative (other anchors, other positives, all hard negatives including H_i). The positive half is the same with
the roles of A and P swapped. The loss is the mean of the two cross-entropies.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def info_nce_with_hard_negatives(anchors: torch.Tensor, positives: torch.Tensor, hard_negs: torch.Tensor,
                                 tau: float = 0.07) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    assert anchors.shape == positives.shape == hard_negs.shape
    n, _ = anchors.shape
    device = anchors.device
    bank = torch.cat([anchors, positives, hard_negs], dim=0)  # [3N, D]

    def _xent(query: torch.Tensor, pos_off: int, self_off: int) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = query @ bank.T / tau
        self_idx = self_off + torch.arange(n, device=device)
        logits.scatter_(1, self_idx.unsqueeze(1), float("-inf"))   # a query is never its own negative
        labels = pos_off + torch.arange(n, device=device)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        return loss, acc

    loss_a, acc_a = _xent(anchors, pos_off=n, self_off=0)
    loss_p, acc_p = _xent(positives, pos_off=0, self_off=n)
    loss = 0.5 * (loss_a + loss_p)

    sim_pos_raw = (anchors * positives).sum(dim=-1).mean()
    with torch.no_grad():
        sim_pos = sim_pos_raw.detach()
        sim_hn = (anchors * hard_negs).sum(dim=-1).mean()
    return loss, {
        "loss_a2p": loss_a.detach(),
        "loss_p2a": loss_p.detach(),
        "acc_a2p": acc_a,
        "acc_p2a": acc_p,
        "sim_pos": sim_pos,
        "sim_hard_neg": sim_hn,
        "gap_pos_vs_hn": sim_pos - sim_hn,
    }

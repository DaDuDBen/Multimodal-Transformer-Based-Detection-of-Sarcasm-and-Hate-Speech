"""Losses for multitask multimodal training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - p_t).pow(self.gamma) * bce
        return focal.mean()


def info_nce_loss(v_proj: torch.Tensor, t_proj: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Contrastive alignment loss between image/text pooled embeddings."""
    v = F.normalize(v_proj.mean(dim=1), dim=-1)
    t = F.normalize(t_proj.mean(dim=1), dim=-1)
    logits = v @ t.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


class UncertaintyWeighting(nn.Module):
    """Kendall uncertainty weighting for two losses."""

    def __init__(self):
        super().__init__()
        self.log_sigma_sarcasm = nn.Parameter(torch.zeros(1))
        self.log_sigma_hate = nn.Parameter(torch.zeros(1))

    def forward(self, sarcasm_loss: torch.Tensor, hate_loss: torch.Tensor) -> torch.Tensor:
        s1 = torch.exp(self.log_sigma_sarcasm)
        s2 = torch.exp(self.log_sigma_hate)
        return 0.5 * sarcasm_loss / (s1**2) + 0.5 * hate_loss / (s2**2) + torch.log(s1 * s2)

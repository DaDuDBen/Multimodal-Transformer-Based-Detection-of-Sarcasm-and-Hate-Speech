"""Cross-modal fusion blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):
    """Symmetric cross-attention with residuals and mean pooling."""

    def __init__(self, d_model: int = 512, n_head: int = 8, dropout: float = 0.3):
        super().__init__()
        self.text_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.img_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.text_norm = nn.LayerNorm(d_model)
        self.img_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if v.ndim != 3 or t.ndim != 3:
            raise ValueError(f"Expected 3D tensors [B, N, D], got v={v.shape}, t={t.shape}")
        if v.shape[0] != t.shape[0] or v.shape[-1] != t.shape[-1]:
            raise ValueError(f"Batch/model dims must match, got v={v.shape}, t={t.shape}")

        t_attended, _ = self.text_attn(query=t, key=v, value=v)
        v_attended, _ = self.img_attn(query=v, key=t, value=t)

        t_out = self.text_norm(t + self.dropout(t_attended))
        v_out = self.img_norm(v + self.dropout(v_attended))

        t_pool = t_out.mean(dim=1)
        v_pool = v_out.mean(dim=1)
        return torch.cat([t_pool, v_pool], dim=1)


class ConcatFusion(nn.Module):
    """Late fusion baseline: mean pool and concatenate."""

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.cat([v.mean(dim=1), t.mean(dim=1)], dim=1)

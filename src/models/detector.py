"""Main multimodal detector architecture."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPVisionModel

from .fusion import ConcatFusion, CrossModalFusion


@dataclass
class ModelConfig:
    vision_model_name: str = "openai/clip-vit-base-patch16"
    text_model_name: str = "vinai/bertweet-large"
    d_model: int = 512
    n_head: int = 8
    dropout: float = 0.3
    fusion_type: str = "cross_attention"  # cross_attention | concat
    vision_hidden_size: int = 768
    text_hidden_size: int = 1024


class MultimodalDetector(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.img_enc = CLIPVisionModel.from_pretrained(cfg.vision_model_name)
        self.txt_enc = AutoModel.from_pretrained(cfg.text_model_name)

        self.proj_v = nn.Linear(cfg.vision_hidden_size, cfg.d_model)
        self.proj_t = nn.Linear(cfg.text_hidden_size, cfg.d_model)

        if cfg.fusion_type == "cross_attention":
            self.fusion = CrossModalFusion(d_model=cfg.d_model, n_head=cfg.n_head, dropout=cfg.dropout)
        elif cfg.fusion_type == "concat":
            self.fusion = ConcatFusion()
        else:
            raise ValueError(f"Unsupported fusion_type: {cfg.fusion_type}")

        self.shared_mlp = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.head_sarcasm = nn.Linear(cfg.d_model, 1)
        self.head_hate = nn.Linear(cfg.d_model, 1)

    def encode(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        v_feat = self.img_enc(pixel_values=images).last_hidden_state
        t_feat = self.txt_enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return v_feat, t_feat

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        v_feat, t_feat = self.encode(images, input_ids, attention_mask)
        v_proj = F.relu(self.proj_v(v_feat))
        t_proj = F.relu(self.proj_t(t_feat))
        fused = self.fusion(v_proj, t_proj)
        shared = self.shared_mlp(fused)
        sarcasm_logit = self.head_sarcasm(shared).squeeze(-1)
        hate_logit = self.head_hate(shared).squeeze(-1)
        return sarcasm_logit, hate_logit, v_proj, t_proj

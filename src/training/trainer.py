"""Training loop for staged multitask optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm

from src.training.losses import FocalLoss, UncertaintyWeighting, info_nce_loss


@dataclass
class TrainingConfig:
    encoder_lr: float = 1e-5
    head_lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    grad_clip: float = 1.0
    gamma: float = 2.0
    alpha: float = 0.25
    alignment_weight: float = 0.1


class Trainer:
    def __init__(self, model: torch.nn.Module, cfg: TrainingConfig, device: str = "cpu"):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.focal = FocalLoss(gamma=cfg.gamma, alpha=cfg.alpha)
        self.uncertainty = UncertaintyWeighting().to(device)

    def build_optimizer(self) -> AdamW:
        encoder_params = list(self.model.img_enc.parameters()) + list(self.model.txt_enc.parameters())
        head_params = (
            list(self.model.proj_v.parameters())
            + list(self.model.proj_t.parameters())
            + list(self.model.fusion.parameters())
            + list(self.model.shared_mlp.parameters())
            + list(self.model.head_sarcasm.parameters())
            + list(self.model.head_hate.parameters())
            + list(self.uncertainty.parameters())
        )
        return AdamW(
            [
                {"params": encoder_params, "lr": self.cfg.encoder_lr},
                {"params": head_params, "lr": self.cfg.head_lr},
            ],
            weight_decay=self.cfg.weight_decay,
        )

    def set_stage(self, stage: int) -> None:
        freeze = stage == 1
        for p in self.model.img_enc.parameters():
            p.requires_grad = not freeze
        for p in self.model.txt_enc.parameters():
            p.requires_grad = not freeze

    def train_epoch(self, dataloader, optimizer, scheduler=None) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in tqdm(dataloader, desc="train", leave=False):
            images = batch["images"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            sarcasm_labels = batch["sarcasm_label"].to(self.device).float()
            hate_labels = batch["hate_label"].to(self.device).float()

            optimizer.zero_grad()
            sarcasm_logits, hate_logits, v_proj, t_proj = self.model(images, input_ids, attention_mask)

            l_s = self.focal(sarcasm_logits, sarcasm_labels)
            l_h = self.focal(hate_logits, hate_labels)
            l_align = info_nce_loss(v_proj, t_proj)
            multitask_loss = self.uncertainty(l_s, l_h)
            loss = multitask_loss + self.cfg.alignment_weight * l_align

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            total_batches += 1

        mean_loss = total_loss / max(total_batches, 1)
        return {"loss": mean_loss}

    def fit(self, stage1_loader, stage2_loader, stage1_epochs: int = 5, stage2_epochs: int = 15):
        history = {"stage1": [], "stage2": []}
        optimizer = self.build_optimizer()
        total_steps = stage2_epochs * max(len(stage2_loader), 1)
        scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=max(self.cfg.warmup_steps, 1))

        self.set_stage(stage=1)
        for _ in range(stage1_epochs):
            history["stage1"].append(self.train_epoch(stage1_loader, optimizer, scheduler=None))

        self.set_stage(stage=2)
        for _ in range(stage2_epochs):
            history["stage2"].append(self.train_epoch(stage2_loader, optimizer, scheduler=scheduler if total_steps > 0 else None))

        return history

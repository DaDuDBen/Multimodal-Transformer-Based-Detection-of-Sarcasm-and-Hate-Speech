#!/usr/bin/env python3
"""Train the multimodal detector with two-stage schedule."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import DataConfig, MultimodalParquetDataset
from src.models.detector import ModelConfig, MultimodalDetector
from src.training.trainer import Trainer, TrainingConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--train-parquet", required=True)
    p.add_argument("--val-parquet", required=True)
    p.add_argument("--output", default="reports/train_history.json")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    data_cfg = DataConfig(
        text_model_name=cfg["model"]["text_model_name"],
        vision_model_name=cfg["model"]["vision_model_name"],
        max_length=cfg["data"].get("max_length", 128),
    )
    model_cfg = ModelConfig(**cfg["model"])
    train_cfg = TrainingConfig(**cfg["training"])

    train_ds = MultimodalParquetDataset(args.train_parquet, data_cfg)
    _ = MultimodalParquetDataset(args.val_parquet, data_cfg)  # kept for future validation hooks
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultimodalDetector(model_cfg)
    trainer = Trainer(model, train_cfg, device=device)
    history = trainer.fit(
        stage1_loader=train_loader,
        stage2_loader=train_loader,
        stage1_epochs=cfg["training"].get("stage1_epochs", 5),
        stage2_epochs=cfg["training"].get("stage2_epochs", 15),
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    ckpt = cfg.get("checkpoint_path", "reports/model.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"Saved history -> {args.output}")
    print(f"Saved checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()

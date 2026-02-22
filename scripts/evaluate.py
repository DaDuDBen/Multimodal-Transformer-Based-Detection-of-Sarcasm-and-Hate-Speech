#!/usr/bin/env python3
"""Evaluate trained model on a parquet split."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import DataConfig, MultimodalParquetDataset
from src.evaluation.metrics import evaluate_multitask
from src.models.detector import ModelConfig, MultimodalDetector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--parquet", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="reports/eval_metrics.json")
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

    ds = MultimodalParquetDataset(args.parquet, data_cfg)
    loader = DataLoader(ds, batch_size=cfg["evaluation"].get("batch_size", 8), shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultimodalDetector(model_cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    sarcasm_true, sarcasm_prob, hate_true, hate_prob = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            s_logit, h_logit, _, _ = model(images, input_ids, attention_mask)

            sarcasm_true.extend(batch["sarcasm_label"].cpu().numpy().tolist())
            hate_true.extend(batch["hate_label"].cpu().numpy().tolist())
            sarcasm_prob.extend(torch.sigmoid(s_logit).cpu().numpy().tolist())
            hate_prob.extend(torch.sigmoid(h_logit).cpu().numpy().tolist())

    metrics = evaluate_multitask(
        np.array(sarcasm_true),
        np.array(sarcasm_prob),
        np.array(hate_true),
        np.array(hate_prob),
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

"""Dataset/dataloader utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, CLIPImageProcessor


@dataclass
class DataConfig:
    text_model_name: str = "vinai/bertweet-large"
    vision_model_name: str = "openai/clip-vit-base-patch16"
    max_length: int = 128


class MultimodalParquetDataset(Dataset):
    def __init__(self, parquet_path: str, cfg: DataConfig):
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name, use_fast=True)
        self.image_processor = CLIPImageProcessor.from_pretrained(cfg.vision_model_name)
        self.max_length = cfg.max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        pixels = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "images": pixels,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "sarcasm_label": torch.tensor(float(row["sarcasm_label"]), dtype=torch.float32),
            "hate_label": torch.tensor(float(row["hate_label"]), dtype=torch.float32),
        }

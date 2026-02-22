"""Unified record schema for multimodal coursework experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UnifiedExample:
    image_path: str
    text: str
    sarcasm_label: float
    hate_label: float
    source_dataset: str

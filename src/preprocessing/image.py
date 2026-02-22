"""Image preprocessing with letterboxing and light augmentation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


@dataclass
class ImagePreprocessConfig:
    target_size: int = 224
    train: bool = False
    horizontal_flip_p: float = 0.5
    jitter_factor: float = 0.1


def letterbox(image: Image.Image, size: int = 224) -> Image.Image:
    """Resize while preserving aspect ratio and pad to square canvas."""
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.BICUBIC)

    canvas = Image.new("RGB", (size, size), color=(0, 0, 0))
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def apply_light_augmentation(image: Image.Image, cfg: ImagePreprocessConfig) -> Image.Image:
    """Apply constrained augmentation suitable for meme data."""
    if random.random() < cfg.horizontal_flip_p:
        image = ImageOps.mirror(image)

    brightness = 1.0 + random.uniform(-cfg.jitter_factor, cfg.jitter_factor)
    contrast = 1.0 + random.uniform(-cfg.jitter_factor, cfg.jitter_factor)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return image


def preprocess_image(path: str | Path, cfg: ImagePreprocessConfig | None = None) -> Image.Image:
    """Load and preprocess an image path."""
    cfg = cfg or ImagePreprocessConfig()
    with Image.open(path) as img:
        output = letterbox(img, size=cfg.target_size)

    if cfg.train:
        output = apply_light_augmentation(output, cfg)
    return output

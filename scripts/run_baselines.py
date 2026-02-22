#!/usr/bin/env python3
"""Generate baseline experiment configs for text-only/image-only/concat."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

BASELINES = {
    "text_only": {"fusion_type": "concat", "mask_image": True, "mask_text": False},
    "image_only": {"fusion_type": "concat", "mask_image": False, "mask_text": True},
    "late_fusion_concat": {"fusion_type": "concat", "mask_image": False, "mask_text": False},
    "proposed_cross_attention": {
        "fusion_type": "cross_attention",
        "mask_image": False,
        "mask_text": False,
    },
}


def main() -> None:
    out = Path("reports/baselines.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(BASELINES, indent=2), encoding="utf-8")
    print(f"Wrote baseline spec to {out}")


if __name__ == "__main__":
    main()

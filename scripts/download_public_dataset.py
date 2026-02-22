#!/usr/bin/env python3
"""Download a public Hugging Face dataset split into local parquet files."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset id")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/raw/downloaded.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_dataset(args.dataset, split=args.split)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(output))
    print(f"Saved {len(ds)} rows to {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare unified dataset and stratified splits for steps 3-4."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split


from src.preprocessing.text import preprocess_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV/JSONL/Parquet file")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--source", default="custom", help="Dataset source name")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--image-col", default="image_path")
    parser.add_argument("--sarcasm-col", default="sarcasm_label")
    parser.add_argument("--hate-col", default="hate_label")
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input extension: {suffix}")


def make_unified(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    unified = pd.DataFrame(
        {
            "image_path": df[args.image_col],
            "text": df[args.text_col].fillna(""),
            "sarcasm_label": df[args.sarcasm_col].fillna(0).astype(float),
            "hate_label": df[args.hate_col].fillna(0).astype(float),
            "source_dataset": args.source,
        }
    )

    processed = unified["text"].map(preprocess_text)
    unified["text"] = processed.map(lambda x: x["text"])
    unified["sarcasm_markers"] = processed.map(lambda x: "|".join(x["markers"]))
    unified["stratify_key"] = (
        unified["sarcasm_label"].round().astype(int).astype(str)
        + "_"
        + unified["hate_label"].round().astype(int).astype(str)
    )
    return unified


def split_dataset(df: pd.DataFrame, train_size: float, val_size: float, seed: int):
    test_size = 1.0 - train_size - val_size
    if test_size <= 0:
        raise ValueError("train_size + val_size must be < 1.0")

    stratify = df["stratify_key"] if df["stratify_key"].value_counts().min() >= 2 else None
    if stratify is None:
        print("Warning: insufficient class counts for stratified split; falling back to random split.")

    train_df, temp_df = train_test_split(
        df, test_size=(1.0 - train_size), random_state=seed, stratify=stratify
    )
    val_relative = val_size / (val_size + test_size)

    temp_stratify = (
        temp_df["stratify_key"]
        if temp_df["stratify_key"].value_counts().min() >= 2
        else None
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_relative),
        random_state=seed,
        stratify=temp_stratify,
    )
    return train_df, val_df, test_df


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_table(Path(args.input))
    unified = make_unified(raw, args)
    train_df, val_df, test_df = split_dataset(
        unified, train_size=args.train_size, val_size=args.val_size, seed=args.seed
    )

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df.drop(columns=["stratify_key"]).to_parquet(output_dir / f"{split_name}.parquet", index=False)

    print(
        f"Saved splits to {output_dir}: "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )


if __name__ == "__main__":
    main()

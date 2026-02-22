# Multimodal Transformer-Based Detection of Sarcasm and Hate Speech

## Coursework implementation status
This repository now includes concrete implementation scaffolding for **steps 1-4** of the coursework development plan:
1. project framing and success criteria,
2. reproducible repository setup,
3. data acquisition + unified schema + stratified splits,
4. text/image preprocessing pipeline.

## Project layout
- `configs/project_scope.yaml`: task definitions, evaluation criteria.
- `configs/data_sources.yaml`: public dataset candidates and target schema.
- `src/preprocessing/text.py`: NFKC normalization, hashtag expansion, emoji mapping, sarcasm marker extraction.
- `src/preprocessing/image.py`: 224x224 letterboxing + constrained augmentation.
- `scripts/download_public_dataset.py`: helper to pull a public Hugging Face dataset.
- `scripts/prepare_dataset.py`: unify columns and build stratified train/val/test splits.
- `COURSEWORK_DEVELOPMENT_PLAN.md`: full 5-week development plan.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Download a public dataset (example)
```bash
python scripts/download_public_dataset.py \
  --dataset limjiayi/hateful_memes_expanded \
  --split train \
  --output data/raw/hateful_memes_train.parquet
```

### 2) Build unified schema and 70/15/15 splits
```bash
python scripts/prepare_dataset.py \
  --input data/raw/hateful_memes_train.parquet \
  --source hateful_memes_expanded \
  --text-col text \
  --image-col image_path \
  --hate-col label \
  --sarcasm-col label \
  --output-dir data/processed
```

> Note: If your source dataset has only one label, map the missing task label to 0 or merge with a second dataset before training multitask models.

## Next steps
Continue with model implementation/training/evaluation milestones in `COURSEWORK_DEVELOPMENT_PLAN.md`.

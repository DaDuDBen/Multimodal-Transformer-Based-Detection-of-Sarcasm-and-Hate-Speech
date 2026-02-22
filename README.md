# Multimodal Transformer-Based Detection of Sarcasm and Hate Speech

This repository now includes implementation scaffolding for:
- steps 1–4 (scope, setup, data, preprocessing), and
- model/training/evaluation milestones (steps 5–8).

## Implemented components
- `src/models/`
  - `fusion.py`: symmetric cross-attention + concat fusion baseline
  - `detector.py`: `MultimodalDetector` (CLIP-ViT + BERTweet, projection, fusion, dual heads)
- `src/training/`
  - `losses.py`: Focal loss, InfoNCE alignment loss, uncertainty weighting
  - `trainer.py`: 2-stage training (freeze encoders, then full fine-tuning)
- `src/evaluation/metrics.py`
  - Macro-F1, AUC-ROC, precision/recall/F1 for binary tasks
- `scripts/`
  - `download_public_dataset.py`: Hugging Face dataset download
  - `prepare_dataset.py`: unified schema + 70/15/15 split
  - `train.py`: stage-1/stage-2 training loop
  - `evaluate.py`: checkpoint evaluation on a parquet split
  - `run_baselines.py`: baseline/ablation experiment spec generation

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data prep
```bash
python scripts/download_public_dataset.py \
  --dataset limjiayi/hateful_memes_expanded \
  --split train \
  --output data/raw/hateful_memes_train.parquet

python scripts/prepare_dataset.py \
  --input data/raw/hateful_memes_train.parquet \
  --source hateful_memes_expanded \
  --text-col text \
  --image-col image_path \
  --hate-col label \
  --sarcasm-col label \
  --output-dir data/processed
```

## Train
```bash
python scripts/train.py \
  --config configs/train.yaml \
  --train-parquet data/processed/train.parquet \
  --val-parquet data/processed/val.parquet \
  --output reports/train_history.json
```

## Evaluate
```bash
python scripts/evaluate.py \
  --config configs/eval.yaml \
  --parquet data/processed/test.parquet \
  --checkpoint reports/model.pt \
  --output reports/eval_metrics.json
```

## Baselines
```bash
python scripts/run_baselines.py
cat reports/baselines.json
```

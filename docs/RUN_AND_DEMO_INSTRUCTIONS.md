# Run and Demonstration Instructions

## 1) Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Note: the default requirements install a large PyTorch stack. On constrained machines, install may take significant time.

## 2) Data Download (Public Example)

```bash
python scripts/download_public_dataset.py \
  --dataset limjiayi/hateful_memes_expanded \
  --split train \
  --output data/raw/hateful_memes_train.parquet
```

## 3) Data Preparation

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

Expected result:
- `data/processed/train.parquet`
- `data/processed/val.parquet`
- `data/processed/test.parquet`

## 4) Model Training

```bash
python scripts/train.py \
  --config configs/train.yaml \
  --train-parquet data/processed/train.parquet \
  --val-parquet data/processed/val.parquet \
  --output reports/train_history.json
```

Expected outputs:
- `reports/train_history.json`
- `reports/model.pt` (from `checkpoint_path` in config)

## 5) Evaluation

```bash
python scripts/evaluate.py \
  --config configs/eval.yaml \
  --parquet data/processed/test.parquet \
  --checkpoint reports/model.pt \
  --output reports/eval_metrics.json
```

Expected output:
- printed JSON metrics and `reports/eval_metrics.json`.

## 6) Demonstration Flow (Recommended)

For a short demo (without full training):
1. Run baseline generation:
   ```bash
   python scripts/run_baselines.py
   cat reports/baselines.json
   ```
2. Run metric smoke test:
   ```bash
   python scripts/smoke_test_metrics.py
   ```
3. Explain architecture from `src/models/detector.py` and fusion options from `src/models/fusion.py`.

This gives a fast, reliable demo of:
- experiment specification,
- metric computation,
- code organization and model design.

# Working Examples (Executed)

This file documents concrete command examples that were executed successfully in this environment.

## Example 1: Metric Pipeline Smoke Test

Command:
```bash
python scripts/smoke_test_metrics.py
```

Observed output:
```python
{'sarcasm': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'macro_f1': 1.0, 'auc_roc': 1.0}, 'hate': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'macro_f1': 1.0, 'auc_roc': 1.0}}
```

## Example 2: Baseline Spec Generation

Command:
```bash
python scripts/run_baselines.py
cat reports/baselines.json
```

Observed output snippet:
```json
{
  "text_only": {
    "fusion_type": "concat",
    "mask_image": true,
    "mask_text": false
  },
  "image_only": {
    "fusion_type": "concat",
    "mask_image": false,
    "mask_text": true
  },
  "late_fusion_concat": {
    "fusion_type": "concat",
    "mask_image": false,
    "mask_text": false
  },
  "proposed_cross_attention": {
    "fusion_type": "cross_attention",
    "mask_image": false,
    "mask_text": false
  }
}
```

## Example 3: Minimal End-to-End Command Sequence

```bash
python scripts/download_public_dataset.py --dataset limjiayi/hateful_memes_expanded --split train --output data/raw/hateful_memes_train.parquet
python scripts/prepare_dataset.py --input data/raw/hateful_memes_train.parquet --source hateful_memes_expanded --text-col text --image-col image_path --hate-col label --sarcasm-col label --output-dir data/processed
python scripts/train.py --config configs/train.yaml --train-parquet data/processed/train.parquet --val-parquet data/processed/val.parquet --output reports/train_history.json
python scripts/evaluate.py --config configs/eval.yaml --parquet data/processed/test.parquet --checkpoint reports/model.pt --output reports/eval_metrics.json
```

Use this sequence as the canonical runnable demonstration for the full pipeline.

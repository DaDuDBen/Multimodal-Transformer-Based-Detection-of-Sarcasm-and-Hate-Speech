"""Evaluation helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "auc_roc": float(auc),
    }


def evaluate_multitask(
    sarcasm_true: np.ndarray,
    sarcasm_prob: np.ndarray,
    hate_true: np.ndarray,
    hate_prob: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    return {
        "sarcasm": binary_metrics(sarcasm_true, sarcasm_prob),
        "hate": binary_metrics(hate_true, hate_prob),
    }

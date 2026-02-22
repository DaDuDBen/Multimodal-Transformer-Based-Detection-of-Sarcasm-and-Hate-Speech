#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.evaluation.metrics import evaluate_multitask

sarcasm_true = np.array([0, 1, 1, 0, 1, 0])
sarcasm_prob = np.array([0.2, 0.7, 0.8, 0.4, 0.6, 0.1])
hate_true = np.array([0, 0, 1, 1, 0, 1])
hate_prob = np.array([0.1, 0.3, 0.9, 0.8, 0.2, 0.7])

print(evaluate_multitask(sarcasm_true, sarcasm_prob, hate_true, hate_prob))

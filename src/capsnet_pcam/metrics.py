"""Binary classification metrics reported for every run."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """ROC AUC plus threshold-based metrics, with tumour (label 1) as the positive class."""
    y_true = np.asarray(y_true).astype(bool)
    pred = np.asarray(scores) >= threshold
    tp = np.sum(pred & y_true)
    tn = np.sum(~pred & ~y_true)
    fp = np.sum(pred & ~y_true)
    fn = np.sum(~pred & y_true)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "accuracy": float((tp + tn) / len(y_true)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(2 * precision * sensitivity / max(precision + sensitivity, 1e-12)),
    }

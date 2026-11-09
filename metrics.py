"""Classification metrics: sens / spec / AUROC / MCC / F1 / confusion matrix."""
from __future__ import annotations
from typing import Dict

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob_pos: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Binary metrics. Class 1 = PNEUMONIA = positive."""
    y_true = np.asarray(y_true).astype(int)
    y_prob_pos = np.asarray(y_prob_pos).astype(float)
    y_pred = (y_prob_pos >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    try:
        auroc = roc_auc_score(y_true, y_prob_pos)
    except ValueError:
        auroc = float("nan")

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "accuracy": float(acc),
        "auroc": float(auroc),
        "mcc": float(mcc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": float(f1),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "threshold": float(threshold),
    }


def find_threshold_for_sensitivity(
    y_true: np.ndarray, y_prob_pos: np.ndarray, target_sens: float = 0.95
) -> float:
    """Highest threshold on a 0.01-0.99 grid whose sensitivity still meets target_sens.

    Scans from high to low and returns the first threshold that reaches the target,
    i.e. the most specific operating point that satisfies the sensitivity constraint.
    Falls back to the lowest grid value (0.01) if the target is never reached.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob_pos = np.asarray(y_prob_pos).astype(float)
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = thresholds[0]
    for t in thresholds[::-1]:  # high → low; first to clear target wins
        y_pred = (y_prob_pos >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        if sens >= target_sens:
            best_t = float(t)
            return best_t
    return float(thresholds[0])

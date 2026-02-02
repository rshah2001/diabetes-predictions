from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


def eval_classifier(model, X_test, y_test) -> Dict[str, float]:
    preds = model.predict(X_test)

    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
    }

    # ROC-AUC requires predict_proba
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        out["roc_auc"] = float(roc_auc_score(y_test, probs))
    else:
        out["roc_auc"] = float("nan")

    return out


def build_leaderboard(results: List[Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    # nice ordering
    cols = ["Model", "roc_auc", "accuracy", "precision", "recall", "f1"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()

    # sort by roc_auc if available
    if "roc_auc" in df.columns and df["roc_auc"].notna().any():
        df = df.sort_values("roc_auc", ascending=False)
    else:
        df = df.sort_values("accuracy", ascending=False)

    return df.reset_index(drop=True)


def roc_curve_points(model, X_test, y_test) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not hasattr(model, "predict_proba"):
        # fallback: degenerate line
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.5])

    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thr = roc_curve(y_test, probs)
    return fpr, tpr, thr


def confusion_from_threshold(model, X_test, y_test, threshold: float = 0.5) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        preds = model.predict(X_test)
        return confusion_matrix(y_test, preds)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    return confusion_matrix(y_test, preds)


def metrics_from_confusion(cm: np.ndarray) -> Dict[str, float]:
    # cm layout: [[TN, FP],[FN, TP]]
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "false_negative_rate": float(1 - recall),
        "false_positive_rate": float(1 - specificity),
    }

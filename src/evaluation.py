"""Statistical evaluation utilities for publication-grade reporting.

Bootstrap confidence intervals on classification metrics, DeLong test for
comparing two correlated ROC-AUCs, and calibration (reliability) computations.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

RNG = np.random.default_rng(42)


def _metric_at(y, p, t, fn) -> float:
    return float(fn(y, (p >= t).astype(int)))


def point_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
        "accuracy": _metric_at(y_true, probs, threshold, accuracy_score),
        "precision": _metric_at(y_true, probs, threshold, lambda a, b: precision_score(a, b, zero_division=0)),
        "recall": _metric_at(y_true, probs, threshold, lambda a, b: recall_score(a, b, zero_division=0)),
        "f1": _metric_at(y_true, probs, threshold, lambda a, b: f1_score(a, b, zero_division=0)),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator = RNG,
) -> Dict[str, Dict[str, float]]:
    """Stratified bootstrap percentile CIs for all point metrics."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    n = len(y_true)
    samples: Dict[str, list] = {}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y_true[idx], probs[idx]
        if len(np.unique(yb)) < 2:  # need both classes for AUC
            continue
        for k, v in point_metrics(yb, pb, threshold).items():
            samples.setdefault(k, []).append(v)

    point = point_metrics(y_true, probs, threshold)
    out = {}
    for k, vals in samples.items():
        arr = np.asarray(vals)
        out[k] = {
            "estimate": point[k],
            "ci_low": float(np.percentile(arr, 100 * alpha / 2)),
            "ci_high": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        }
    return out


# --- DeLong test for two correlated ROC curves ----------------------------
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x_sorted = x[order]
    n = len(x)
    tieval = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        tieval[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = tieval
    return out


def _fast_delong(predictions_sorted_transposed: np.ndarray, m: int):
    n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> Dict[str, float]:
    """Two-sided DeLong test that AUC_a == AUC_b on the same samples.

    Returns the two AUCs and the p-value for their difference.
    """
    y_true = np.asarray(y_true)
    order = (-y_true).argsort(kind="mergesort")
    label_1_count = int(y_true.sum())
    preds = np.vstack((prob_a, prob_b))[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)

    l = np.array([[1, -1]])
    var = l @ cov @ l.T
    z = (aucs[0] - aucs[1]) / np.sqrt(var[0, 0]) if var[0, 0] > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": float(z), "p_value": float(p)}


def reliability_curve(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10):
    """Equal-width calibration bins. Returns (mean_pred, frac_pos, bin_count)."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    mean_pred, frac_pos, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        mean_pred.append(float(probs[mask].mean()))
        frac_pos.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))
    return np.array(mean_pred), np.array(frac_pos), np.array(counts)


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    mp, fp, counts = reliability_curve(y_true, probs, n_bins)
    if len(counts) == 0:
        return float("nan")
    weights = counts / counts.sum()
    return float(np.sum(weights * np.abs(mp - fp)))

"""Train the best-of-class diabetes prediction model.

Usage:
    python -m src.train [path/to/diabetes.csv]

Selects the best pipeline by repeated stratified cross-validated ROC-AUC,
calibrates its probabilities, tunes the decision threshold on out-of-fold
predictions, reports held-out test metrics, then refits on all data and
saves a single reusable artifact to models/diabetes_model.joblib.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import PimaPreprocessor
from src.models import build_classification_models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "diabetes.csv"
ARTIFACT_PATH = PROJECT_ROOT / "models" / "diabetes_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
TARGET_COL = "Outcome"
RANDOM_STATE = 42


def make_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("prep", PimaPreprocessor()),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def tune_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Picks the threshold that maximizes F1 on out-of-fold predictions."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.20, 0.81, 0.01):
        f1 = f1_score(y_true, (probs >= t).astype(int))
        if f1 > best_f1:
            best_t, best_f1 = float(round(t, 2)), f1
    return best_t


def classification_report_at(y_true, probs, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "threshold": float(threshold),
    }


def main(data_path: Path = DEFAULT_DATA) -> None:
    df = pd.read_csv(data_path)
    if TARGET_COL not in df.columns:
        raise SystemExit(f"Expected target column '{TARGET_COL}' in {data_path}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    print(f"Loaded {len(df)} rows, {X.shape[1]} features, positives={int(y.sum())}")

    # --- 1) Model selection by repeated stratified CV ROC-AUC -------------
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    scores = {}
    for name, model in build_classification_models().items():
        pipe = make_pipeline(model)
        auc = cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
        scores[name] = (float(auc.mean()), float(auc.std()))
        print(f"  CV ROC-AUC {auc.mean():.4f} ± {auc.std():.4f}  {name}")

    best_name = max(scores, key=lambda k: scores[k][0])
    print(f"\nBest by CV ROC-AUC: {best_name} ({scores[best_name][0]:.4f})")

    # --- 2) Honest held-out evaluation with calibration + threshold -------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    def calibrated_best():
        best_model = build_classification_models()[best_name]
        return CalibratedClassifierCV(
            make_pipeline(best_model), method="sigmoid", cv=5
        )

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = cross_val_predict(
        calibrated_best(), X_tr, y_tr, cv=inner_cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold = tune_threshold(y_tr.values, oof_probs)
    print(f"Tuned decision threshold (max F1 on train OOF): {threshold:.2f}")

    holdout_clf = calibrated_best().fit(X_tr, y_tr)
    test_probs = holdout_clf.predict_proba(X_te)[:, 1]
    test_metrics = classification_report_at(y_te.values, test_probs, threshold)
    print("Held-out test metrics:", json.dumps(test_metrics, indent=2))

    # --- 3) Final fit on all data and save artifact ------------------------
    final_clf = calibrated_best().fit(X, y)
    medians = final_clf.calibrated_classifiers_[0].estimator["prep"].medians_

    ARTIFACT_PATH.parent.mkdir(exist_ok=True)
    artifact = {
        "pipeline": final_clf,
        "threshold": threshold,
        "feature_cols": list(X.columns),
        "feature_medians": {k: float(v) for k, v in medians.items()},
        "model_name": best_name,
        "cv_roc_auc": scores[best_name][0],
        "test_metrics": test_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(artifact, ARTIFACT_PATH)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "model": best_name,
                "cv_scores": {k: {"mean": m, "std": s} for k, (m, s) in scores.items()},
                "threshold": threshold,
                "test_metrics": test_metrics,
            },
            indent=2,
        )
    )
    print(f"\nSaved model artifact -> {ARTIFACT_PATH}")
    print(f"Saved metrics        -> {METRICS_PATH}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA)

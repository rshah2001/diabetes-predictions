from __future__ import annotations
from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)


def _try_xgb():
    try:
        from xgboost import XGBClassifier  # type: ignore

        return XGBClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.1,
            reg_lambda=1.5,
            reg_alpha=0.1,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )
    except Exception:
        return None


def _try_lgbm():
    try:
        from lightgbm import LGBMClassifier  # type: ignore

        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=20,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        return None


def build_classification_models(include_xgb: bool = True) -> Dict[str, object]:
    models: Dict[str, object] = {
        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            C=0.5,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=42,
        ),
    }

    xgb = _try_xgb() if include_xgb else None
    if xgb is not None:
        models["XGBoost"] = xgb

    lgbm = _try_lgbm()
    if lgbm is not None:
        models["LightGBM"] = lgbm

    models["Stacked Ensemble"] = build_stacked_ensemble(include_xgb=include_xgb)
    return models


def build_stacked_ensemble(include_xgb: bool = True) -> StackingClassifier:
    """Best-of-class model: stacks diverse base learners with a logistic
    meta-learner trained on out-of-fold probabilities."""
    fresh = build_base_estimators(include_xgb=include_xgb)
    return StackingClassifier(
        estimators=list(fresh.items()),
        final_estimator=LogisticRegression(max_iter=5000, C=1.0),
        stack_method="predict_proba",
        cv=5,
        n_jobs=-1,
    )


def build_base_estimators(include_xgb: bool = True) -> Dict[str, object]:
    base: Dict[str, object] = {
        "lr": LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced"),
        "rf": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "gb": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=42,
        ),
    }

    xgb = _try_xgb() if include_xgb else None
    if xgb is not None:
        base["xgb"] = xgb

    lgbm = _try_lgbm()
    if lgbm is not None:
        base["lgbm"] = lgbm

    return base

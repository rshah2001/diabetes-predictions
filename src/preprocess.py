from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def prepare_classification_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
    zero_as_missing_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    df_raw = df.copy()
    feature_cols = [c for c in df.columns if c != target_col]

    # Coerce numeric (common for this dataset)
    df_raw = _coerce_numeric(df_raw, feature_cols + [target_col])

    # Handle zeros-as-missing
    zero_as_missing_cols = zero_as_missing_cols or []
    for c in zero_as_missing_cols:
        if c in df_raw.columns:
            df_raw.loc[df_raw[c] == 0, c] = np.nan

    # Drop rows where target missing
    df_raw = df_raw.dropna(subset=[target_col])

    # Basic imputation: fill features with median
    feature_medians = df_raw[feature_cols].median(numeric_only=True).to_dict()
    X = df_raw[feature_cols].copy()
    X = X.fillna(value=feature_medians)

    y = df_raw[target_col].astype(int)

    # Train/test split (stratify for class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values

    return {
        "df_raw": df_raw,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "feature_medians": feature_medians,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }

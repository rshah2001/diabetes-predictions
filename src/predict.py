"""Load the trained diabetes model artifact and run predictions on raw data."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = PROJECT_ROOT / "models" / "diabetes_model.joblib"


def load_artifact(path: Path = ARTIFACT_PATH) -> Optional[dict]:
    """Returns the saved model artifact, or None if it hasn't been trained yet."""
    if not Path(path).exists():
        return None
    return joblib.load(path)


def predict_proba(artifact: dict, df: pd.DataFrame) -> np.ndarray:
    """Probability of diabetes for each row of raw (unscaled) feature data."""
    X = df[artifact["feature_cols"]]
    return artifact["pipeline"].predict_proba(X)[:, 1]


def predict(artifact: dict, df: pd.DataFrame, threshold: Optional[float] = None) -> pd.DataFrame:
    """Returns df with probability and class columns appended."""
    t = artifact["threshold"] if threshold is None else threshold
    probs = predict_proba(artifact, df)
    out = df.copy()
    out["diabetes_probability"] = probs
    out["diabetes_prediction"] = (probs >= t).astype(int)
    return out

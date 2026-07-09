"""Load the trained diabetes model artifact and run predictions on raw data."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = PROJECT_ROOT / "models" / "diabetes_model.joblib"

# Probability floor for the "high" risk band. Predicted positives below this
# are "moderate"; see risk_band() for the full scheme and invariants.
HIGH_BAND_FLOOR = 0.60


def risk_band(prob: float, threshold: float) -> str:
    """Map a diabetes probability to a risk band, relative to the decision threshold.

    The model's decision threshold (0.28 in the saved artifact) is tuned for
    screening: it maximizes F1 on out-of-fold predictions and deliberately
    favors recall, so it sits well below 0.5. Risk bands must therefore be
    defined *relative to the threshold* — fixed cut-offs like 0.40/0.70 would
    label a predicted-positive patient (e.g. prob 0.30) as "low" risk.

    Scheme:
        low:      prob < threshold                      (predicted negative)
        moderate: threshold <= prob < max(0.60, threshold)
        high:     prob >= max(0.60, threshold)

    Invariants (hold for any threshold in (0, 1)):
        * prediction = 1 (prob >= threshold) is never banded "low".
        * prediction = 0 (prob < threshold) is never banded "high"
          (it is always "low", since the low band spans exactly the
          predicted-negative range).

    The 0.60 floor splits predicted positives into "flagged by a
    recall-favoring screen" (moderate) versus "more likely diabetic than not
    by a clear margin" (high). max(0.60, threshold) keeps the bands ordered
    if a caller supplies a stricter threshold than 0.60.
    """
    if prob < threshold:
        return "low"
    if prob < max(HIGH_BAND_FLOOR, threshold):
        return "moderate"
    return "high"


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

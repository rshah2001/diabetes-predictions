from __future__ import annotations
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Pima dataset encodes missing measurements as 0 in these columns
PIMA_ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


class PimaPreprocessor(BaseEstimator, TransformerMixin):
    """Cleans Pima-style diabetes data and adds derived clinical features.

    - Treats 0 as missing in measurement columns and imputes with the
      training-set median.
    - Adds interaction/ratio features that are well known to help on this
      dataset (glucose x BMI, insulin/glucose, etc.).

    Accepts and returns pandas DataFrames so it can sit at the front of an
    sklearn Pipeline operating on raw uploaded data.
    """

    def __init__(
        self,
        zero_as_missing_cols: Optional[List[str]] = None,
        add_derived: bool = True,
    ):
        self.zero_as_missing_cols = zero_as_missing_cols
        self.add_derived = add_derived

    def _zero_to_nan(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        cols = self.zero_as_missing_cols
        if cols is None:
            cols = PIMA_ZERO_AS_MISSING
        for c in cols:
            if c in out.columns:
                out.loc[out[c] == 0, c] = np.nan
        return out

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
        X = self._zero_to_nan(X)
        self.input_cols_ = list(X.columns)
        self.medians_ = X.median(numeric_only=True).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        # Tolerate extra columns / different order; require the trained ones
        missing = [c for c in self.input_cols_ if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        X = X[self.input_cols_].apply(pd.to_numeric, errors="coerce")
        X = self._zero_to_nan(X)
        X = X.fillna(value=self.medians_)
        # getattr keeps artifacts pickled before add_derived existed working
        return add_derived_features(X) if getattr(self, "add_derived", True) else X

    def get_feature_names_out(self, input_features=None):
        sample = pd.DataFrame([self.medians_], columns=self.input_cols_)
        out = add_derived_features(sample) if getattr(self, "add_derived", True) else sample
        return np.asarray(out.columns, dtype=object)


def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """Adds ratio/interaction features when their source columns exist."""
    out = X.copy()
    have = out.columns

    if {"Glucose", "BMI"} <= set(have):
        out["Glucose_x_BMI"] = out["Glucose"] * out["BMI"]
    if {"Glucose", "Age"} <= set(have):
        out["Glucose_div_Age"] = out["Glucose"] / out["Age"].clip(lower=1)
    if {"Insulin", "Glucose"} <= set(have):
        out["Insulin_div_Glucose"] = out["Insulin"] / out["Glucose"].clip(lower=1)
    if {"SkinThickness", "BMI"} <= set(have):
        out["SkinThickness_div_BMI"] = out["SkinThickness"] / out["BMI"].clip(lower=1)
    if {"Pregnancies", "Age"} <= set(have):
        out["Pregnancies_div_Age"] = out["Pregnancies"] / out["Age"].clip(lower=1)
    if {"BMI", "Age"} <= set(have):
        out["BMI_x_Age"] = out["BMI"] * out["Age"]

    return out

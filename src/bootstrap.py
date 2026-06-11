"""Auto-loads the bundled dataset and a trained model into the Streamlit session.

Lets every page work immediately without uploading a CSV: pages call
ensure_data_pack() / ensure_best_model() instead of stopping when the
session is empty. Uploading a custom dataset on the Upload page still
overrides the bundled one.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.features import PIMA_ZERO_AS_MISSING
from src.preprocess import prepare_classification_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_CSV = PROJECT_ROOT / "diabetes.csv"


def load_bundled_dataset() -> pd.DataFrame:
    return pd.read_csv(BUNDLED_CSV)


def ensure_data_pack() -> dict:
    """Returns the session data pack, preparing the bundled dataset if needed."""
    if "data_pack" not in st.session_state:
        df = load_bundled_dataset()
        st.session_state["data_pack"] = prepare_classification_data(
            df=df,
            target_col="Outcome",
            test_size=0.2,
            random_state=42,
            scale=True,
            zero_as_missing_cols=PIMA_ZERO_AS_MISSING,
        )
        st.session_state["data_source"] = "bundled diabetes.csv"
    return st.session_state["data_pack"]


def ensure_best_model(pack: dict):
    """Returns (model, name), auto-training the CV winner if none is selected.

    Extra Trees won model selection in src.train (CV ROC-AUC 0.844), so it is
    the default when the user hasn't run Model Compare in this session.
    """
    if "best_model" not in st.session_state:
        from src.models import build_classification_models

        with st.spinner("Training default model (Extra Trees)..."):
            model = build_classification_models()["Extra Trees"]
            model.fit(pack["X_train"], pack["y_train"])
        st.session_state["best_model"] = model
        st.session_state["best_model_name"] = "Extra Trees (auto-trained)"
    return (
        st.session_state["best_model"],
        st.session_state.get("best_model_name", "Best Model"),
    )


def data_source_caption() -> None:
    source = st.session_state.get("data_source", "bundled diabetes.csv")
    st.caption(f"Dataset in use: **{source}** — replace it any time on the Upload page.")

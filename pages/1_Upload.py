import streamlit as st
import pandas as pd

from src.preprocess import prepare_classification_data
from src.bootstrap import load_bundled_dataset

st.title("1) Data (Diabetes Classification)")

st.info(
    "The app comes with the Pima diabetes dataset built in — every page already works "
    "without uploading anything. Upload a CSV here only if you want to use your own data."
)

uploaded = st.file_uploader("Upload CSV (optional — overrides the bundled dataset)", type=["csv"])
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    data_source = f"uploaded file: {uploaded.name}"
else:
    raw = load_bundled_dataset()
    data_source = "bundled diabetes.csv"

st.caption(f"Currently previewing: **{data_source}**")

st.subheader("Preview")
st.dataframe(raw.head(25), use_container_width=True)

cols = list(raw.columns)

# Suggest typical target column
default_target = "Outcome" if "Outcome" in cols else cols[-1]
target_col = st.selectbox("Target column (0/1)", cols, index=cols.index(default_target))

feature_cols = [c for c in cols if c != target_col]
st.caption("All other columns will be treated as features.")
st.write("Features:", feature_cols)

st.divider()
st.subheader("Split + Preprocessing")

test_size = st.slider("Test size (%)", 10, 40, 20, step=5) / 100.0
random_state = st.number_input("Random state", value=42, step=1)

scale_features = st.checkbox("Standardize features (recommended)", value=True)

# Handle zeros-as-missing (Pima dataset often uses 0 for missing)
zero_as_missing_default = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
zero_as_missing = st.multiselect(
    "Treat 0 as missing for these columns (common for Pima dataset)",
    options=feature_cols,
    default=[c for c in zero_as_missing_default if c in feature_cols],
)

if st.button("Prepare dataset", type="primary"):
    try:
        pack = prepare_classification_data(
            df=raw,
            target_col=target_col,
            test_size=test_size,
            random_state=int(random_state),
            scale=scale_features,
            zero_as_missing_cols=zero_as_missing,
        )
        st.subheader("Target unique values")
        st.write(sorted(pack["df_raw"][pack["target_col"]].dropna().unique().tolist()))
        
    except Exception as e:
        st.error(f"Preparation failed: {e}")
        st.stop()

    st.session_state["data_pack"] = pack
    st.session_state["data_source"] = data_source
    # New dataset invalidates any previously trained session model
    for key in ("best_model", "best_model_name", "leaderboard", "trained_models"):
        st.session_state.pop(key, None)
    st.success("Saved! Go to Insights → Model Compare.")

    st.subheader("Prepared shapes")
    st.write(
        {
            "X_train": pack["X_train"].shape,
            "X_test": pack["X_test"].shape,
            "y_train": pack["y_train"].shape,
            "y_test": pack["y_test"].shape,
        }
    )

    st.subheader("Target balance (train)")
    st.write(pack["y_train"].value_counts(dropna=False).to_dict())

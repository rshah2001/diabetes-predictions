import streamlit as st
import pandas as pd

from src.predict import load_artifact, predict_proba

st.title("5) Predict Diabetes Risk")

st.caption(
    "This page lets you generate diabetes predictions. You can either enter one patient "
    "manually or upload a CSV to predict multiple patients at once."
)

# Two possible model sources:
# 1) The pre-trained best-of-class pipeline saved by `python -m src.train`
# 2) The best model selected interactively on the Model Compare page
artifact = load_artifact()
have_session = "best_model" in st.session_state and "data_pack" in st.session_state

if artifact is None and not have_session:
    st.warning(
        "No model available yet. Either run Model Compare first, or train the "
        "best-of-class model with: `python -m src.train`"
    )
    st.stop()

options = []
if artifact is not None:
    options.append("Pre-trained best-of-class model (recommended)")
if have_session:
    options.append("Best model from Model Compare session")

source = options[0] if len(options) == 1 else st.radio("Model source", options)
use_artifact = source.startswith("Pre-trained")

if use_artifact:
    feature_cols = artifact["feature_cols"]
    feature_medians = artifact["feature_medians"]
    default_threshold = float(artifact["threshold"])
    st.subheader(f"Using model: {artifact['model_name']} (calibrated)")
    tm = artifact["test_metrics"]
    st.caption(
        f"Held-out test performance — ROC-AUC: {tm['roc_auc']:.3f}, "
        f"Recall: {tm['recall']:.3f}, Precision: {tm['precision']:.3f}, F1: {tm['f1']:.3f}"
    )

    def get_probs(df: pd.DataFrame):
        return predict_proba(artifact, df)

else:
    pack = st.session_state["data_pack"]
    model = st.session_state["best_model"]
    best_name = st.session_state.get("best_model_name", "Best Model")
    feature_cols = pack["feature_cols"]
    feature_medians = pack["feature_medians"]
    scaler = pack.get("scaler", None)
    default_threshold = 0.50
    st.subheader(f"Using model: {best_name}")

    def get_probs(df: pd.DataFrame):
        X = df[feature_cols]
        X_scaled = scaler.transform(X) if scaler is not None else X.values
        return model.predict_proba(X_scaled)[:, 1]

st.caption(
    "The model outputs a probability (risk score) between 0 and 1:\n"
    "• Closer to 0 → model thinks diabetes is unlikely\n"
    "• Closer to 1 → model thinks diabetes is likely"
)

# Threshold slider here so prediction matches decision settings
threshold = st.slider(
    "Decision threshold (predict diabetes if probability ≥ threshold)",
    0.05, 0.95, default_threshold, 0.01
)
st.caption(
    "How to interpret the threshold:\n"
    "• Higher threshold (e.g., 0.70) = stricter → fewer positives → fewer false alarms, more missed cases.\n"
    "• Lower threshold (e.g., 0.35–0.45) = more sensitive → more positives → fewer missed cases, more false alarms."
)

mode = st.radio(
    "Prediction mode",
    ["Single patient", "Batch CSV"],
    horizontal=True
)

if mode == "Single patient":
    st.subheader("Enter patient features")
    st.caption(
        "Enter values for one patient. If you're unsure what to type, you can leave the default values "
        "which are based on the dataset’s median (typical) values."
    )

    # Build an input form
    user_input = {}
    cols = st.columns(2)

    for i, feat in enumerate(feature_cols):
        with cols[i % 2]:
            default_val = float(feature_medians.get(feat, 0.0))
            user_input[feat] = st.number_input(
                label=f"{feat}",
                value=default_val
            )

    if st.button("Predict risk", type="primary"):
        x = pd.DataFrame([user_input])

        # Probability of class 1 (diabetes)
        prob = float(get_probs(x)[0])
        pred = int(prob >= threshold)
        st.session_state["last_prediction"] = {
            "probability": prob,
            "features": user_input,
        }

        st.metric("Predicted diabetes probability", f"{prob:.3f}")
        st.write(f"Class prediction @ threshold {threshold:.2f}:", pred)

        # Human-readable risk bands (these are just UI labels)
        if prob >= 0.70:
            st.warning("High risk (probability ≥ 0.70).")
        elif prob >= 0.40:
            st.info("Moderate risk (0.40–0.69).")
        else:
            st.success("Lower risk (< 0.40).")

        st.caption(
            "Note: This is a model prediction based on patterns in the dataset. "
            "It is meant for demonstration / screening-style insight, not medical diagnosis."
        )
        st.info("Next: open **Geo Context** to see this risk in the context of any US county.")

else:
    st.subheader("Batch prediction (CSV)")
    st.caption(
        "Upload a CSV containing the same feature columns. The app will generate a probability and a predicted class "
        f"using your chosen threshold ({threshold:.2f})."
    )

    up = st.file_uploader("Upload a CSV with the same feature columns", type=["csv"], key="batch")
    st.caption("Your CSV should contain these columns: " + ", ".join(feature_cols))

    if up is None:
        st.stop()

    df = pd.read_csv(up)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required feature columns: {missing}")
        st.stop()

    probs = get_probs(df)
    preds = (probs >= threshold).astype(int)

    out = df.copy()
    out["diabetes_probability"] = probs
    out[f"diabetes_prediction_at_{threshold:.2f}"] = preds

    st.subheader("Results preview")
    st.caption(
        "This preview shows your original rows plus the model’s diabetes probability and final prediction."
    )
    st.dataframe(out.head(25), use_container_width=True)

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions CSV",
        data=csv_bytes,
        file_name="diabetes_predictions.csv"
    )

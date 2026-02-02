import streamlit as st
import pandas as pd

st.title("5) Predict Diabetes Risk")

st.caption(
    "This page lets you generate diabetes predictions using the best model selected earlier. "
    "You can either enter one patient manually or upload a CSV to predict multiple patients at once."
)

if "best_model" not in st.session_state or "data_pack" not in st.session_state:
    st.warning("Run Model Compare first so a best model is selected.")
    st.stop()

pack = st.session_state["data_pack"]
model = st.session_state["best_model"]
best_name = st.session_state.get("best_model_name", "Best Model")

feature_cols = pack["feature_cols"]
scaler = pack.get("scaler", None)

st.subheader(f"Using model: {best_name}")

st.caption(
    "The model outputs a probability (risk score) between 0 and 1:\n"
    "• Closer to 0 → model thinks diabetes is unlikely\n"
    "• Closer to 1 → model thinks diabetes is likely"
)

# Threshold slider here so prediction matches decision settings
threshold = st.slider(
    "Decision threshold (predict diabetes if probability ≥ threshold)",
    0.05, 0.95, 0.50, 0.01
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
            default_val = float(pack["feature_medians"].get(feat, 0.0))
            user_input[feat] = st.number_input(
                label=f"{feat}",
                value=default_val
            )

    if st.button("Predict risk", type="primary"):
        x = pd.DataFrame([user_input])

        if scaler is not None:
            x_scaled = scaler.transform(x)
        else:
            x_scaled = x.values

        # Probability of class 1 (diabetes)
        prob = float(model.predict_proba(x_scaled)[0][1])
        pred = int(prob >= threshold)

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

    X = df[feature_cols].copy()

    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values

    probs = model.predict_proba(X_scaled)[:, 1]
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

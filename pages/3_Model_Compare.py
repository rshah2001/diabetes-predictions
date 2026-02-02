import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.models import build_classification_models
from src.metrics import eval_classifier, build_leaderboard, roc_curve_points

st.title("3) Model Compare")

st.caption(
    "This page trains multiple machine learning models and compares how well they predict diabetes on unseen test data. "
    "You can think of it as a 'competition' where each model tries to correctly label patients as diabetic (1) or not (0)."
)

if "data_pack" not in st.session_state:
    st.warning("Upload and prepare data first in the Upload page.")
    st.stop()

pack = st.session_state["data_pack"]
X_train = pack["X_train"]
X_test = pack["X_test"]
y_train = pack["y_train"]
y_test = pack["y_test"]

# Determine if target is binary (ROC curves are only standard for binary classification)
unique_classes = pd.Series(y_test).dropna().unique()
is_binary = len(unique_classes) == 2

st.subheader("Models")
st.caption(
    "Choose which models to include before training. Logistic Regression is a strong baseline, "
    "while tree-based models (Random Forest / Gradient Boosting / XGBoost) can capture non-linear patterns."
)

use_xgb = st.checkbox("Include XGBoost (recommended)", value=True)
models = build_classification_models(include_xgb=use_xgb)

st.info(
    "What happens when you click **Train + Evaluate**:\n"
    "• Each model trains on the training set.\n"
    "• It is then tested on the test set (data the model has never seen).\n"
    "• We compute metrics like Accuracy and Recall to compare models fairly."
)

results = []
trained = {}

if st.button("Train + Evaluate", type="primary"):
    with st.spinner("Training models..."):
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained[name] = model

            metrics = eval_classifier(model, X_test, y_test)
            metrics["Model"] = name
            results.append(metrics)

    leaderboard = build_leaderboard(results)
    st.session_state["leaderboard"] = leaderboard
    st.session_state["trained_models"] = trained

    # Pick best by ROC-AUC (fallback to Accuracy if ROC-AUC missing)
    sort_col = "roc_auc" if "roc_auc" in leaderboard.columns else "accuracy"
    best_name = leaderboard.sort_values(sort_col, ascending=False).iloc[0]["Model"]
    st.session_state["best_model_name"] = best_name
    st.session_state["best_model"] = trained[best_name]

    st.success(f"Done. Best model: {best_name}")

if "leaderboard" in st.session_state:
    st.subheader("Leaderboard (higher is better)")
    st.caption(
        "How to interpret these metrics:\n"
        "• **Accuracy**: Overall percent correct.\n"
        "• **Precision**: When the model predicts diabetes, how often it is correct.\n"
        "• **Recall (Sensitivity)**: Out of all diabetics, how many the model correctly catches (important for screening).\n"
        "• **F1**: A balance between Precision and Recall.\n"
        "• **ROC-AUC**: Measures how well the model separates diabetics vs non-diabetics across all thresholds (higher is better)."
    )
    st.dataframe(st.session_state["leaderboard"], use_container_width=True)

    st.divider()

    if is_binary:
        st.subheader("ROC Curves")
        st.caption(
            "ROC curves show model performance across different decision thresholds.\n"
            "• The **x-axis** is the False Positive Rate: healthy patients incorrectly flagged as diabetic.\n"
            "• The **y-axis** is the True Positive Rate (Recall): diabetics correctly identified.\n"
            "Curves closer to the top-left corner indicate better performance."
        )
        st.caption(
            "The dashed diagonal line represents random guessing. Any model curve above it is doing better than random."
        )

        trained = st.session_state["trained_models"]

        fig = plt.figure()
        for name, model in trained.items():
            fpr, tpr, _ = roc_curve_points(model, X_test, y_test)
            plt.plot(fpr, tpr, label=name)

        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate (healthy flagged as diabetic)")
        plt.ylabel("True Positive Rate / Recall (diabetics caught)")
        plt.legend()
        st.pyplot(fig, clear_figure=True)

    else:
        st.info(
            "ROC curves are only shown for binary targets (0/1). "
            "Your selected target currently has more than two classes."
        )

    st.info("Next: go to **Decision Impact** to adjust the threshold and see how false positives/negatives change.")

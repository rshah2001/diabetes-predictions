import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.metrics import confusion_from_threshold, metrics_from_confusion

st.title("4) Decision Impact (Threshold + Confusion Matrix)")

st.caption(
    "This page shows how changing the decision threshold affects real outcomes. "
    "In other words, it helps you control the tradeoff between catching more diabetics "
    "and accidentally flagging some healthy patients."
)

if "best_model" not in st.session_state or "data_pack" not in st.session_state:
    st.warning("Run Model Compare first so a best model is selected.")
    st.stop()

pack = st.session_state["data_pack"]
model = st.session_state["best_model"]
best_name = st.session_state.get("best_model_name", "Best Model")

X_test = pack["X_test"]
y_test = pack["y_test"]

unique_classes = pd.Series(y_test).dropna().unique()
is_binary = len(unique_classes) == 2

st.subheader(f"Selected model: {best_name}")

if not is_binary:
    st.info(
        "This threshold + confusion matrix view is designed for binary targets (0/1). "
        "Your target currently has more than two classes."
    )
    st.stop()

st.caption(
    "The model produces a probability (risk score) between 0 and 1 for each patient. "
    "The **threshold** is the cutoff used to turn that probability into a final yes/no prediction."
)

threshold = st.slider(
    "Decision threshold (predict diabetes if probability ≥ threshold)",
    0.05, 0.95, 0.50, 0.01
)

st.caption(
    "How to interpret the threshold:\n"
    "• **Higher threshold (e.g., 0.70)** = stricter → fewer people predicted diabetic → fewer false positives, but more missed diabetics.\n"
    "• **Lower threshold (e.g., 0.35)** = more sensitive → more people predicted diabetic → fewer missed diabetics, but more false positives."
)

st.divider()

cm = confusion_from_threshold(model, X_test, y_test, threshold=threshold)
summary = metrics_from_confusion(cm)

# cm layout: [[TN, FP],[FN, TP]]
tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

st.subheader("Confusion Matrix")
st.caption(
    "The confusion matrix compares the model’s predictions to the true labels.\n"
    "• Rows = what is actually true (True 0 / True 1)\n"
    "• Columns = what the model predicted (Pred 0 / Pred 1)\n"
    "0 = no diabetes, 1 = diabetes"
)

st.write(
    {
        "True Negatives (TN) — correctly predicted no diabetes": tn,
        "False Positives (FP) — predicted diabetes but actually no": fp,
        "False Negatives (FN) — predicted no diabetes but actually yes (missed cases)": fn,
        "True Positives (TP) — correctly predicted diabetes": tp,
    }
)

fig = plt.figure()
plt.imshow(cm, aspect="equal")
plt.xticks([0, 1], ["Pred 0 (No diabetes)", "Pred 1 (Diabetes)"])
plt.yticks([0, 1], ["True 0 (No diabetes)", "True 1 (Diabetes)"])
plt.colorbar()

for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, int(v), ha="center", va="center")

plt.xlabel("Model prediction")
plt.ylabel("Actual label")
st.pyplot(fig, clear_figure=True)

st.divider()

st.subheader("Key metrics at this threshold")
st.caption(
    "These numbers summarize performance at the chosen threshold:\n"
    "• **Recall / Sensitivity**: Out of all diabetics, how many the model correctly catches.\n"
    "• **Precision**: Out of everyone predicted diabetic, how many truly are diabetic.\n"
    "• **Specificity**: Out of all non-diabetics, how many the model correctly identifies.\n"
    "• **False Negative Rate**: Portion of diabetics the model misses."
)
st.write(summary)

st.divider()

st.subheader("How to choose a threshold (rule of thumb)")
st.info(
    "If this is a **screening** tool (catching cases early):\n"
    "• Prioritize **Recall/Sensitivity** → consider a lower threshold (often ~0.35–0.45).\n\n"
    "If this is a **confirmation** tool (avoid false alarms):\n"
    "• Prioritize **Precision** → consider a higher threshold (often ~0.55–0.70)."
)

st.caption(
    "In healthcare screening, missing true cases (false negatives) can be more costly than false positives, "
    "so many systems intentionally choose a threshold below 0.50."
)

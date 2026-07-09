import pandas as pd
import streamlit as st

from src.features import PIMA_ZERO_AS_MISSING
from src.predict import load_artifact, predict_proba, risk_band

st.title("5) Predict Diabetes Risk")

st.caption(
    "Estimate diabetes risk for one patient, or score a CSV of patients. "
    "This is an educational screening demo, not a medical diagnosis."
)

# Per-field metadata driving the single-patient form and the batch column help.
# Defaults are healthy-adult values (or empty/required) — never cohort medians,
# which would make an untouched form score like the study population.
FIELD_SPECS = {
    "Pregnancies": {
        "label": "Pregnancies (count)",
        "short": "Pregnancies",
        "min": 0, "max": 20, "step": 1, "default": 0,
        "help": "Number of times pregnant. Enter 0 if none or not applicable.",
        "unit": "count", "note": "Number of times pregnant",
    },
    "Glucose": {
        "label": "Glucose (mg/dL, 2-hr OGTT)",
        "short": "Glucose",
        "min": 40, "max": 300, "step": 1, "default": None,
        "placeholder": "e.g. 95",
        "help": (
            "Plasma glucose 2 hours into an oral glucose tolerance test. "
            "Normal < 140, prediabetic range 140–199, diabetic range ≥ 200."
        ),
        "unit": "mg/dL", "note": "2-hr OGTT plasma glucose — use 0 if not measured",
    },
    "BloodPressure": {
        "label": "Diastolic blood pressure (mm Hg)",
        "short": "Diastolic blood pressure",
        "min": 30, "max": 140, "step": 1, "default": 70,
        "help": "The lower number in a blood-pressure reading. Typical healthy adult: 60–80 mm Hg.",
        "unit": "mm Hg", "note": "Diastolic blood pressure — use 0 if not measured",
    },
    "SkinThickness": {
        "label": "Triceps skinfold thickness (mm)",
        "short": "Triceps skinfold thickness",
        "min": 5, "max": 80, "step": 1, "default": None,
        "unknown_default": True,
        "help": (
            "Caliper measurement of the skinfold at the back of the upper arm. "
            "Rarely measured outside studies — leave 'Not measured' checked if unknown."
        ),
        "unit": "mm", "note": "Triceps skinfold — use 0 if not measured",
    },
    "Insulin": {
        "label": "2-hour serum insulin (µU/mL)",
        "short": "2-hour serum insulin",
        "min": 10, "max": 900, "step": 5, "default": None,
        "unknown_default": True,
        "help": "Serum insulin 2 hours into the OGTT. Rarely available — leave 'Not measured' checked if unknown.",
        "unit": "µU/mL", "note": "2-hour serum insulin — use 0 if not measured",
    },
    "BMI": {
        "label": "BMI (kg/m², body mass index)",
        "short": "BMI",
        "min": 15.0, "max": 70.0, "step": 0.1, "default": None,
        "placeholder": "e.g. 24.5",
        "help": "Weight (kg) divided by height (m) squared. Healthy 18.5–24.9, overweight 25–29.9, obese ≥ 30.",
        "unit": "kg/m²", "note": "Body mass index — use 0 if not measured",
    },
    "DiabetesPedigreeFunction": {
        "label": "Diabetes pedigree function (family-history score)",
        "short": "Diabetes pedigree function",
        "min": 0.05, "max": 2.50, "step": 0.01, "default": 0.20,
        "help": (
            "Dataset-specific score summarizing diabetes in relatives. Typical range 0.1–1.0; "
            "higher = stronger family history. 0.20 ≈ little family history."
        ),
        "unit": "score", "note": "Family-history score; typical 0.1–1.0",
    },
    "Age": {
        "label": "Age (years)",
        "short": "Age",
        "min": 21, "max": 100, "step": 1, "default": None,
        "placeholder": "e.g. 35",
        "help": "Age in years. The training cohort is women aged 21+; predictions outside that group are extrapolation.",
        "unit": "years", "note": "Age in years",
    },
}

# Glucose and BMI can alternatively be marked "Not measured"; Age cannot.
REQUIRED_FIELDS = ["Glucose", "BMI", "Age"]

PIMA_SENTENCE = (
    "Trained on the Pima Indians Diabetes dataset: 768 women aged 21+, ~35% with diabetes — "
    "a much higher rate than the general population, which is why raw scores run high."
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
    default_threshold = float(artifact["threshold"])

    with st.expander("About this model", expanded=False):
        st.write(f"**{artifact['model_name']}** (calibrated)")
        tm = artifact["test_metrics"]
        st.caption(
            f"Held-out test performance — ROC-AUC: {tm['roc_auc']:.3f}, "
            f"Recall: {tm['recall']:.3f}, Precision: {tm['precision']:.3f}, F1: {tm['f1']:.3f}"
        )
        st.caption(PIMA_SENTENCE)

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

    with st.expander("About this model", expanded=False):
        st.write(f"**{best_name}** (from this session's Model Compare run)")
        st.caption(PIMA_SENTENCE)

    def get_probs(df: pd.DataFrame):
        X = df[feature_cols].copy()
        # Session models have no imputation step in front of the scaler, so map
        # the 0 = "not measured" sentinel to the training medians here (the
        # pre-trained artifact's pipeline does this internally).
        for c in PIMA_ZERO_AS_MISSING:
            if c in X.columns:
                X.loc[X[c] == 0, c] = feature_medians.get(c, 0.0)
        X_scaled = scaler.transform(X) if scaler is not None else X.values
        return model.predict_proba(X_scaled)[:, 1]


def threshold_control() -> float:
    """Collapsed expander with screening/balanced presets and a custom slider.

    The chosen value is kept in session state so it survives reruns and
    mode switches; the expander label always shows the active setting.
    """
    if "predict_threshold" not in st.session_state:
        st.session_state["predict_threshold"] = default_threshold
    current = float(st.session_state["predict_threshold"])

    with st.expander(f"Decision threshold — currently {current:.0%}", expanded=False):
        if use_artifact:
            presets = [
                f"Screening — {default_threshold:.2f} (recommended)",
                "Balanced — 0.50",
                "Custom",
            ]
        else:
            presets = ["Balanced — 0.50 (recommended)", "Custom"]

        choice = st.radio("Preset", presets, horizontal=True, key="threshold_preset")
        if choice.startswith("Screening"):
            threshold = default_threshold
        elif choice.startswith("Balanced"):
            threshold = 0.50
        else:
            threshold = st.slider("Custom threshold", 0.05, 0.95, current, 0.01, key="threshold_custom")

        caption = "Lower threshold = catches more true cases but more false alarms; higher = the reverse."
        if use_artifact:
            caption += (
                f" The {default_threshold:.2f} default was tuned for screening-style recall — "
                "see the Decision Impact page for the full tradeoff."
            )
        st.caption(caption)

    st.session_state["predict_threshold"] = threshold
    return threshold


mode = st.radio(
    "Prediction mode",
    ["Single patient", "Batch CSV"],
    horizontal=True
)

if mode == "Single patient":
    st.subheader("Enter patient features")
    st.caption(
        "Fields marked 'Not measured' are filled in with typical values from the training data "
        "(median imputation). The more real measurements you enter, the more the estimate reflects "
        "this patient rather than the cohort average."
    )

    user_input = {}
    not_measured = {}
    cols = st.columns(2)

    for i, feat in enumerate(feature_cols):
        spec = FIELD_SPECS.get(feat)
        with cols[i % 2]:
            if spec is None:
                # Feature outside the known Pima schema (defensive; normally
                # feature_cols matches FIELD_SPECS exactly).
                user_input[feat] = st.number_input(feat, value=0.0)
                not_measured[feat] = False
                continue

            has_unknown = feat in PIMA_ZERO_AS_MISSING
            unk_default = spec.get("unknown_default", False)
            # Checkbox state from the previous rerun controls the disabled flag
            # even though the checkbox is rendered below its input.
            unk = st.session_state.get(f"unk_{feat}", unk_default)

            user_input[feat] = st.number_input(
                spec["label"],
                min_value=spec["min"],
                max_value=spec["max"],
                step=spec["step"],
                value=spec["default"],
                placeholder=spec.get("placeholder"),
                help=spec["help"],
                disabled=has_unknown and unk,
            )
            if has_unknown:
                unk = st.checkbox("Not measured", key=f"unk_{feat}", value=unk_default)
            not_measured[feat] = has_unknown and unk

    threshold = threshold_control()

    if st.button("Estimate risk", type="primary"):
        missing = [
            f for f in REQUIRED_FIELDS
            if f in user_input and user_input[f] is None and not not_measured.get(f, False)
        ]
        if missing:
            st.error(
                "Please fill in the required fields (or mark them 'Not measured'): "
                + ", ".join(missing)
            )
        else:
            row = {}
            imputed_labels = []
            for feat, val in user_input.items():
                if feat in PIMA_ZERO_AS_MISSING and (not_measured.get(feat) or val is None):
                    row[feat] = 0.0  # missing sentinel -> median-imputed downstream
                    imputed_labels.append(FIELD_SPECS.get(feat, {}).get("short", feat))
                else:
                    row[feat] = float(val)

            x = pd.DataFrame([row])
            prob = float(get_probs(x)[0])
            flagged = prob >= threshold
            band = risk_band(prob, threshold)
            st.session_state["last_prediction"] = {
                "probability": prob,
                "features": row,
            }

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Estimated probability", f"{prob:.0%}",
                help=(
                    "The model's estimated probability of diabetes, relative to its "
                    "training cohort (a high-prevalence study population)."
                ),
            )
            m2.metric(
                "Screening threshold", f"{threshold:.0%}",
                help="Probabilities at or above this cutoff are flagged. Change it under 'Decision threshold' above.",
            )
            m3.metric("Screen result", "Flagged" if flagged else "Not flagged")

            # Exactly one status box, chosen by the shared risk_band(); band and
            # screen result agree by construction.
            if band == "low":
                st.success(
                    f"**Below the screening threshold.** At the current threshold ({threshold:.0%}), "
                    "this patient would **not be flagged** for follow-up."
                )
            elif band == "moderate":
                st.warning(
                    f"**Flagged by the screen.** The estimated risk ({prob:.0%}) is above the screening "
                    f"threshold ({threshold:.0%}), so this patient **would be flagged for follow-up testing**. "
                    "At this deliberately sensitive threshold, many flagged patients turn out not to have "
                    "diabetes — that is by design."
                )
            else:
                st.error(
                    f"**High estimated risk.** The model is substantially more confident here ({prob:.0%}); "
                    "this patient would be flagged for follow-up testing."
                )

            if imputed_labels:
                st.caption(
                    "Not measured (filled with cohort-typical values): "
                    + ", ".join(imputed_labels)
                    + ". Imputed values pull the estimate toward the training cohort's average, "
                    "so treat it as a rougher guess."
                )

            st.caption(
                f"A flag at a low threshold (like {threshold:.0%}) is a screening result, not a diagnosis: "
                "the threshold is set low on purpose to catch as many true cases as possible, at the cost "
                "of extra false alarms. Real screening programs confirm flags with diagnostic testing "
                "(fasting glucose, HbA1c). This demo is trained on a small 1990s research cohort and is "
                "for education only."
            )
            st.caption("Next: open **Geo Context** to rescale this score to a real US county's prevalence.")

else:
    st.subheader("Batch prediction (CSV)")
    st.caption("Upload a CSV with one row per patient to score many patients at once.")

    with st.expander("Expected columns and units"):
        st.table(
            pd.DataFrame(
                [
                    (f, FIELD_SPECS[f]["unit"], FIELD_SPECS[f]["note"])
                    for f in feature_cols if f in FIELD_SPECS
                ],
                columns=["Column", "Unit", "Notes"],
            )
        )
        st.caption(
            "Use 0 in " + ", ".join(PIMA_ZERO_AS_MISSING[:-1]) + ", or " + PIMA_ZERO_AS_MISSING[-1]
            + " to mean 'not measured'; those cells are median-imputed."
        )

    threshold = threshold_control()

    up = st.file_uploader("Upload a CSV with the same feature columns", type=["csv"], key="batch")

    if up is None:
        st.stop()

    df = pd.read_csv(up)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.error(
            "This CSV is missing required columns: " + ", ".join(missing) + ". "
            f"Expected exactly these {len(feature_cols)}: " + ", ".join(feature_cols) + "."
        )
        st.stop()

    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    bad_cols = [c for c in feature_cols if df[c].isna().any()]
    if bad_cols:
        st.error("Non-numeric values found in: " + ", ".join(bad_cols) + ". Fix these cells and re-upload.")
        st.stop()

    probs = get_probs(df)
    preds = (probs >= threshold).astype(int)

    out = df.copy()
    out["diabetes_probability"] = probs
    out[f"diabetes_prediction_at_{threshold:.2f}"] = preds
    out["screen_result"] = ["Flagged" if p else "Not flagged" for p in preds]

    st.subheader("Results preview")
    st.caption(f"Each row gets a probability and a screen result at your chosen threshold ({threshold:.0%}).")
    st.dataframe(out.head(25), use_container_width=True)

    zero_cols = [c for c in PIMA_ZERO_AS_MISSING if c in df.columns]
    n_zero_rows = int((df[zero_cols] == 0).any(axis=1).sum()) if zero_cols else 0
    if n_zero_rows:
        st.caption(
            f"{n_zero_rows} rows have 0 ('not measured') in one or more of "
            + ", ".join(zero_cols)
            + " — those values were imputed with cohort medians before scoring."
        )

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions CSV",
        data=csv_bytes,
        file_name="diabetes_predictions.csv"
    )

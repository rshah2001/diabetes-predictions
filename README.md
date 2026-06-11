# Diabetes Prediction

An end-to-end **machine learning application** for predicting diabetes risk based on patient health indicators.  
The app supports **data upload, preprocessing, model training, evaluation, and prediction** through an interactive interface.

---

## Project Overview

This project applies supervised machine learning to classify whether a patient is likely to have diabetes using clinical and demographic features such as glucose level, BMI, age, and blood pressure.

The goal is to demonstrate:
- Practical ML preprocessing
- Model comparison and evaluation
- Clear separation of data, models, and metrics
- Reproducible ML workflows

---

## Features

- Upload and validate structured health datasets
- Automatic data preprocessing:
  - Missing value handling
  - Feature scaling
  - Derived clinical interaction features
- Multiple ML models for comparison (including a stacked ensemble)
- Model evaluation using standard classification metrics
- Prediction on new/unseen data
- **Geographic risk context (GeoAI)**: combines the personal model score with
  CDC PLACES county-level data — adult diabetes prevalence, obesity, physical
  inactivity, blood pressure, and insurance access for all 3,144 US counties —
  including a national prevalence map and a prior-corrected risk estimate
  rescaled to the selected county's base rate
- REST API (FastAPI) serving predictions with optional county context
- **Publication-grade evaluation** (`python -m src.report`): ablation study,
  model comparison with DeLong significance tests, bootstrap confidence
  intervals, calibration (reliability diagram + ECE), SHAP explainability, and
  **external validation on NHANES 2017–2018** — all written to `reports/` as
  CSV tables and vector-PDF figures, and surfaced in the app's Chapter Results page

---

## Book chapter / reproducibility

This repository is the artifact for a Springer Nature book chapter; the GeoAI
geographic prior-correction is its novel contribution. For methodology and
headline results see **[METHODS.md](METHODS.md)** and **[MODEL_CARD.md](MODEL_CARD.md)**.

```bash
make install      # virtualenv + pinned dependencies
make reproduce    # train model + regenerate every table and figure in reports/
make app          # interactive Streamlit app
make api          # FastAPI prediction service
```

---

## Models Used

- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost
- LightGBM
- **Stacked Ensemble** (all of the above combined with a logistic meta-learner)

The training pipeline (`python -m src.train`) selects the best model by
repeated stratified 5-fold cross-validated ROC-AUC, calibrates its
probabilities, tunes the decision threshold on out-of-fold predictions, and
saves a single reusable artifact to `models/diabetes_model.joblib`.

Models are evaluated and compared using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC / PR-AUC

---

## Dataset

The app is compatible with datasets structured similarly to the **Pima Indians Diabetes Dataset**, containing features such as:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (target)

---

## How to Run

### 1) Clone the repository
```bash
git clone https://github.com/rshah2001/diabetes-predictions.git
cd diabetes-predictions
```

### 2) Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Train the best-of-class model
```bash
python -m src.train
```
This prints a cross-validated leaderboard, held-out test metrics, and saves
the calibrated best pipeline to `models/diabetes_model.joblib`.

### 4) Run the Streamlit app
```bash
streamlit run app.py
```
The **Predict** page can use the pre-trained model directly, or you can walk
the full Upload → Insights → Model Compare flow interactively.

### 5) (Optional) Serve predictions over a REST API
```bash
uvicorn backend.api:app --reload
```
Then `POST /predict` with patient features, or check `GET /health`.

Add `?county_fips=04013` (any US county FIPS) to `POST /predict` to get
geographic context — county prevalence, national percentiles, and the risk
score rescaled to that county's base rate. `GET /counties?state_abbr=AZ`
lists counties. County data (CDC PLACES, 2024 release) downloads and caches
automatically on first use.

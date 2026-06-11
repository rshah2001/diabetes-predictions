# Model Card — Diabetes Risk Prediction with Geographic Context

## Model details
- **Type:** Calibrated tree-ensemble classifier (Extra Trees selected by repeated
  stratified cross-validated ROC-AUC; a stacked ensemble of LR/RF/GB/XGBoost/LightGBM
  is also available).
- **Probability calibration:** Platt scaling (sigmoid) via 5-fold `CalibratedClassifierCV`.
- **Decision threshold:** Tuned on out-of-fold predictions to maximize F1 (≈0.28),
  reflecting a screening setting where recall is prioritized.
- **Inputs:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
  DiabetesPedigreeFunction, Age, plus derived interaction/ratio features.
- **Output:** Calibrated probability of diabetes, a class label at the chosen threshold,
  and (optionally) a county-prevalence–adjusted probability.

## Intended use
- **Use:** Educational and screening-style risk illustration; a book-chapter artifact
  demonstrating a reproducible ML + GeoAI workflow.
- **Not for:** Clinical diagnosis or individual medical decisions. Outputs are not a
  substitute for laboratory testing or a clinician's judgment.

## Training data
- **Pima Indians Diabetes Dataset** (768 adult women, ≥21 years). Target prevalence ≈35%.
- Measurement zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI are treated
  as missing and median-imputed.

### Ethical note on data provenance
This dataset originates from research on the **Akimel O'odham (Pima) community** of
Arizona, who have been the subject of extensive, and at times ethically criticized,
diabetes research. We use it here because it is a long-standing public ML benchmark,
but readers should be aware of this history and avoid drawing population-level
conclusions about any community from a model trained on it.

## Evaluation
- **Internal:** Repeated stratified 5-fold CV and a held-out 20% test set, reported with
  95% bootstrap confidence intervals; model differences assessed with the DeLong test.
- **Calibration:** Reliability diagram and Expected Calibration Error (ECE), before/after
  Platt scaling.
- **External validation:** Evaluated on **NHANES 2017–2018** women aged ≥21 (n≈1,181,
  prevalence ≈16%) using the four shared features (Glucose, BloodPressure, BMI, Age).
  A performance drop from internal to external is expected and reported honestly.

## Geographic context (GeoAI)
- County-level adult prevalence (diabetes, obesity, inactivity, blood pressure, insurance)
  from **CDC PLACES 2024** for all 3,144 US counties.
- The calibrated probability is re-expressed to a county's base rate via odds-ratio
  prior correction. This assumes the *relative* discriminative signal transports across
  populations while the *base rate* differs — an approximation, not a causal claim.

## Limitations
- Small, single-cohort training data (adult women only); limited demographic coverage.
- Self-reported diabetes status in NHANES; clinical thresholds differ across datasets/eras.
- The geographic adjustment shifts base rates, not individual biology, and should not be
  read as a place-based diagnosis.

## Reproducibility
- `make reproduce` regenerates the model artifact and every table/figure in `reports/`.
- Dependencies pinned in `requirements.txt`; random seeds fixed at 42 throughout.

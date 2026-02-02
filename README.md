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
- Multiple ML models for comparison
- Model evaluation using standard classification metrics
- Prediction on new/unseen data

---

## Models Used

- Logistic Regression
- Random Forest
- Gradient Boosting
- (Optional) XGBoost

Models are evaluated and compared using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

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

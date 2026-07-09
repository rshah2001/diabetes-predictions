"""Shared fixtures for the diabetes-prediction test suite.

Run from the project root:  .venv/bin/python -m pytest tests/ -v
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Make `src`, `backend`, and `pages` importable regardless of how pytest
# inserted paths (tests/ has no __init__.py, so only tests/ gets added).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Canonical patient profiles (raw feature order: Pregnancies, Glucose,
# BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

HEALTHY = dict(zip(FEATURE_COLS, [0, 85, 70, 20, 80, 21, 0.2, 25]))
DIABETIC = dict(zip(FEATURE_COLS, [6, 190, 90, 35, 300, 36, 1.2, 55]))
# Lands in the moderate band at the artifact threshold 0.28 (prob ~0.536,
# i.e. 0.28 <= prob < 0.60) with SkinThickness/Insulin sent as 0 sentinels.
MODERATE = dict(zip(FEATURE_COLS, [2, 130, 70, 0, 0, 30, 0.4, 40]))

# Known-good reference probabilities from before the refactor. If these move,
# the saved artifact was retrained (it must not be — published chapter asset).
REF_PROB_HEALTHY = 0.041
REF_PROB_DIABETIC = 0.837
REF_PROB_MEDIANS = 0.30


@pytest.fixture(scope="session")
def artifact():
    from src.predict import load_artifact

    art = load_artifact()
    assert art is not None, (
        "models/diabetes_model.joblib missing - tests need the saved artifact"
    )
    return art

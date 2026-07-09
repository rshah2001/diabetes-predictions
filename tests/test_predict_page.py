"""Streamlit page tests for pages/5_Predict.py via streamlit.testing.v1.AppTest.

Widget map for the single-patient view (artifact-only, so no model-source radio):
  radio[0]                 -> "Prediction mode"
  number_input             -> NOT in feature order: st.columns(2) renders all of
                              column 0 before column 1 in the element tree, so
                              inputs are addressed by label prefix (see ninput()).
  checkbox(key="unk_<f>")  -> "Not measured" for the 5 zero-as-missing fields
  radio(key="threshold_preset") -> threshold presets (default: Screening 0.28)
  button[0]                -> "Estimate risk"
"""
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.predict import predict_proba
from tests.conftest import PROJECT_ROOT

PAGE = str(PROJECT_ROOT / "pages" / "5_Predict.py")

# Feature name -> unique label prefix of its st.number_input
LABELS = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose",
    "BloodPressure": "Diastolic blood pressure",
    "SkinThickness": "Triceps skinfold",
    "Insulin": "2-hour serum insulin",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Diabetes pedigree function",
    "Age": "Age",
}


def ninput(at: AppTest, feat: str):
    """Find a feature's number_input by its label (order-independent)."""
    prefix = LABELS[feat]
    matches = [n for n in at.number_input if n.label.startswith(prefix)]
    assert len(matches) == 1, f"{feat}: expected 1 input labeled '{prefix}*', got {len(matches)}"
    return matches[0]


def run_page() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=60)
    at.run()
    assert not at.exception, f"page raised: {at.exception}"
    return at


class TestPageLayout:
    def test_initial_render(self):
        at = run_page()
        # 8 feature inputs, 5 "Not measured" checkboxes, one Estimate button
        assert len(at.number_input) == 8
        assert len(at.checkbox) == 5
        assert len(at.button) >= 1
        assert at.button[0].label == "Estimate risk"
        # Required fields start empty (Streamlit 1.58 value=None)
        for f in ("Glucose", "BMI", "Age"):
            assert ninput(at, f).value is None, f"{f} should start empty"
        # Rarely-measured fields default to "Not measured"
        assert at.checkbox(key="unk_SkinThickness").value is True
        assert at.checkbox(key="unk_Insulin").value is True
        assert at.checkbox(key="unk_Glucose").value is False
        assert at.checkbox(key="unk_BMI").value is False
        # No result shown before any click
        assert len(at.metric) == 0

    def test_default_threshold_is_artifact_screening(self, artifact):
        at = run_page()
        assert at.session_state["predict_threshold"] == pytest.approx(
            artifact["threshold"]
        )


class TestUntouchedForm:
    def test_click_without_input_shows_error_not_prediction(self):
        at = run_page()
        at.button[0].click().run()
        assert not at.exception
        assert len(at.error) >= 1, "expected an error for empty required fields"
        joined = " ".join(e.value for e in at.error)
        for f in ("Glucose", "BMI", "Age"):
            assert f in joined, f"error should name missing field {f}"
        # No prediction of any kind rendered
        assert len(at.metric) == 0
        assert len(at.success) == 0
        assert len(at.warning) == 0


class TestHealthyProfile:
    def test_low_risk_exactly_one_success(self, artifact):
        at = run_page()
        ninput(at, "Glucose").set_value(85)
        ninput(at, "BMI").set_value(21.0)
        ninput(at, "Age").set_value(25)
        at.button[0].click().run()
        assert not at.exception

        assert len(at.success) == 1, "healthy profile must show exactly one success box"
        assert len(at.error) == 0
        assert len(at.warning) == 0

        # Metrics coherent with the model: Skin/Insulin sent as 0 sentinels
        row = {"Pregnancies": 0, "Glucose": 85, "BloodPressure": 70,
               "SkinThickness": 0, "Insulin": 0, "BMI": 21,
               "DiabetesPedigreeFunction": 0.2, "Age": 25}
        prob = float(predict_proba(artifact, pd.DataFrame([row]))[0])
        assert prob < artifact["threshold"]
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Estimated probability"] == f"{prob:.0%}"
        assert metrics["Screen result"] == "Not flagged"

        # Imputation disclosure names the unmeasured fields
        captions = " ".join(c.value for c in at.caption)
        assert "Triceps skinfold thickness" in captions
        assert "2-hour serum insulin" in captions


class TestModerateProfile:
    def test_over_threshold_warning_no_success(self, artifact):
        """Profile landing in the moderate band (prob ~0.54 at t=0.28):
        must render a warning and NO success box (the old contradiction)."""
        at = run_page()
        ninput(at, "Pregnancies").set_value(2)
        ninput(at, "Glucose").set_value(130)
        ninput(at, "BMI").set_value(30.0)
        ninput(at, "DiabetesPedigreeFunction").set_value(0.4)
        ninput(at, "Age").set_value(40)
        at.button[0].click().run()
        assert not at.exception

        row = {"Pregnancies": 2, "Glucose": 130, "BloodPressure": 70,
               "SkinThickness": 0, "Insulin": 0, "BMI": 30,
               "DiabetesPedigreeFunction": 0.4, "Age": 40}
        prob = float(predict_proba(artifact, pd.DataFrame([row]))[0])
        assert artifact["threshold"] <= prob < 0.60, (
            f"test profile drifted out of moderate band: prob={prob}"
        )

        assert len(at.warning) >= 1, "moderate band must show a warning box"
        assert len(at.success) == 0, "flagged patient must NOT get a success box"
        assert len(at.error) == 0
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Screen result"] == "Flagged"
        flagged_text = " ".join(w.value for w in at.warning)
        assert "flagged" in flagged_text.lower()


class TestHighProfile:
    def test_high_band_error_box(self, artifact):
        """Clearly diabetic profile (prob ~0.84) -> the 'high' st.error box,
        and still no success box."""
        at = run_page()
        at.checkbox(key="unk_SkinThickness").uncheck().run()
        at.checkbox(key="unk_Insulin").uncheck().run()
        ninput(at, "Pregnancies").set_value(6)
        ninput(at, "Glucose").set_value(190)
        ninput(at, "BloodPressure").set_value(90)
        ninput(at, "SkinThickness").set_value(35)
        ninput(at, "Insulin").set_value(300)
        ninput(at, "BMI").set_value(36.0)
        ninput(at, "DiabetesPedigreeFunction").set_value(1.2)
        ninput(at, "Age").set_value(55)
        at.button[0].click().run()
        assert not at.exception

        assert len(at.error) == 1, "high band must show exactly one error box"
        assert len(at.success) == 0
        assert len(at.warning) == 0
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Screen result"] == "Flagged"
        assert "High estimated risk" in at.error[0].value

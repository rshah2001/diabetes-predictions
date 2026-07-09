"""Unit + regression tests for src.predict: risk_band, artifact scoring, parity.

Also contains the repo guardrail test (no modifications under reports/ or
models/ - those are published book-chapter assets).
"""
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.predict import HIGH_BAND_FLOOR, predict, predict_proba, risk_band
from tests.conftest import (
    DIABETIC,
    FEATURE_COLS,
    HEALTHY,
    MODERATE,
    PROJECT_ROOT,
    REF_PROB_DIABETIC,
    REF_PROB_HEALTHY,
    REF_PROB_MEDIANS,
)

BAND_ORDER = {"low": 0, "moderate": 1, "high": 2}


# ---------------------------------------------------------------------------
# risk_band unit tests
# ---------------------------------------------------------------------------
class TestRiskBand:
    def test_high_band_floor_constant(self):
        assert HIGH_BAND_FLOOR == 0.60

    def test_exhaustive_invariant_sweep(self):
        """Thresholds 0.01..0.99 x probs 0..1 (step 0.005): core invariants."""
        thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
        probs = np.round(np.arange(0.0, 1.0 + 1e-9, 0.005), 3)
        for t in thresholds:
            t = float(t)
            floor = max(HIGH_BAND_FLOOR, t)
            for p in probs:
                p = float(p)
                band = risk_band(p, t)
                assert band in ("low", "moderate", "high"), (p, t, band)
                if p >= t:
                    # prediction = 1 must never band "low"
                    assert band != "low", f"prob={p} t={t} predicted 1 but banded low"
                else:
                    # prediction = 0 must band exactly "low" (never mod/high)
                    assert band == "low", f"prob={p} t={t} predicted 0 but banded {band}"
                # scheme itself
                if p >= floor:
                    assert band == "high", (p, t, band)
                elif p >= t:
                    assert band == "moderate", (p, t, band)

    def test_band_monotone_in_prob(self):
        """For a fixed threshold, band never decreases as prob increases."""
        probs = np.round(np.arange(0.0, 1.0 + 1e-9, 0.005), 3)
        for t in (0.05, 0.28, 0.5, 0.6, 0.7, 0.95):
            ranks = [BAND_ORDER[risk_band(float(p), t)] for p in probs]
            assert ranks == sorted(ranks), f"bands not monotone at t={t}"

    def test_boundary_prob_equals_threshold_is_moderate(self):
        # prob == t means prediction 1 -> must not be "low"
        assert risk_band(0.28, 0.28) == "moderate"
        assert risk_band(0.5, 0.5) == "moderate"

    def test_boundary_prob_at_floor_is_high(self):
        assert risk_band(0.60, 0.28) == "high"
        assert risk_band(0.5999999, 0.28) == "moderate"

    def test_just_below_threshold_is_low(self):
        assert risk_band(0.2799999, 0.28) == "low"

    def test_threshold_above_floor(self):
        """t=0.7 > 0.60: floor becomes t, moderate band is empty, ordering holds."""
        t = 0.7
        assert risk_band(0.69, t) == "low"
        assert risk_band(0.70, t) == "high"   # prob == t, predicted 1, not "low"
        assert risk_band(0.99, t) == "high"
        # no prob in [0,1] can band "moderate" at t=0.7
        probs = np.round(np.arange(0.0, 1.0 + 1e-9, 0.005), 3)
        assert all(risk_band(float(p), t) != "moderate" for p in probs)

    def test_extreme_probs(self):
        assert risk_band(0.0, 0.28) == "low"
        assert risk_band(1.0, 0.28) == "high"
        assert risk_band(1.0, 0.99) == "high"
        assert risk_band(0.0, 0.01) == "low"


# ---------------------------------------------------------------------------
# Model regression against the saved artifact (must NOT have been retrained)
# ---------------------------------------------------------------------------
class TestModelRegression:
    def test_artifact_metadata(self, artifact):
        assert artifact["model_name"] == "Extra Trees"
        assert artifact["threshold"] == pytest.approx(0.28)
        assert artifact["feature_cols"] == FEATURE_COLS

    def test_healthy_profile_low(self, artifact):
        prob = float(predict_proba(artifact, pd.DataFrame([HEALTHY]))[0])
        assert prob < 0.10
        assert risk_band(prob, artifact["threshold"]) == "low"

    def test_diabetic_profile_high(self, artifact):
        prob = float(predict_proba(artifact, pd.DataFrame([DIABETIC]))[0])
        assert prob > 0.70
        assert risk_band(prob, artifact["threshold"]) == "high"

    def test_reference_probs_artifact_not_retrained(self, artifact):
        """Known-good probs from before the refactor - detects a retrain."""
        df = pd.DataFrame([HEALTHY, DIABETIC, dict(artifact["feature_medians"])])
        probs = predict_proba(artifact, df)
        assert probs[0] == pytest.approx(REF_PROB_HEALTHY, abs=0.005)
        assert probs[1] == pytest.approx(REF_PROB_DIABETIC, abs=0.005)
        assert probs[2] == pytest.approx(REF_PROB_MEDIANS, abs=0.02)

    def test_glucose_monotone(self, artifact):
        """Raising Glucose 85 -> 190 (others at healthy profile) strictly raises prob."""
        lo = dict(HEALTHY)
        hi = dict(HEALTHY, Glucose=190)
        probs = predict_proba(artifact, pd.DataFrame([lo, hi]))
        assert probs[1] > probs[0]


# ---------------------------------------------------------------------------
# predict() vs predict_proba parity, batch-vs-single parity
# ---------------------------------------------------------------------------
class TestParity:
    def test_predict_matches_predict_proba(self, artifact):
        df = pd.DataFrame([HEALTHY, MODERATE, DIABETIC])
        probs = predict_proba(artifact, df)
        out = predict(artifact, df)  # default threshold from artifact
        np.testing.assert_allclose(
            out["diabetes_probability"].to_numpy(), probs, atol=1e-12
        )
        t = artifact["threshold"]
        expected_pred = (probs >= t).astype(int)
        np.testing.assert_array_equal(
            out["diabetes_prediction"].to_numpy(), expected_pred
        )

    def test_predict_threshold_override(self, artifact):
        df = pd.DataFrame([HEALTHY, MODERATE, DIABETIC])
        out = predict(artifact, df, threshold=0.7)
        probs = out["diabetes_probability"].to_numpy()
        np.testing.assert_array_equal(
            out["diabetes_prediction"].to_numpy(), (probs >= 0.7).astype(int)
        )

    def test_batch_vs_single_parity(self, artifact):
        rows = [HEALTHY, MODERATE, DIABETIC]
        batch = predict(artifact, pd.DataFrame(rows))["diabetes_probability"].to_numpy()
        singles = np.array([
            float(predict(artifact, pd.DataFrame([r]))["diabetes_probability"].iloc[0])
            for r in rows
        ])
        np.testing.assert_allclose(batch, singles, atol=1e-12)

    def test_predict_does_not_mutate_input(self, artifact):
        df = pd.DataFrame([HEALTHY])
        cols_before = list(df.columns)
        predict(artifact, df)
        assert list(df.columns) == cols_before


# ---------------------------------------------------------------------------
# Guardrail: published assets untouched
# ---------------------------------------------------------------------------
class TestGuardrail:
    def test_reports_and_models_unmodified(self):
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        offenders = []
        for line in res.stdout.splitlines():
            # porcelain format: XY <path> (or XY <old> -> <new> for renames)
            path_part = line[3:]
            for p in path_part.split(" -> "):
                p = p.strip().strip('"')
                if p.startswith("reports/") or p.startswith("models/"):
                    offenders.append(line)
        assert offenders == [], (
            "Published assets modified:\n" + "\n".join(offenders)
        )

"""API tests for backend/api.py using the real saved artifact.

Uses the lifespan-aware TestClient context manager: the model loads in the
FastAPI lifespan, so all requests go through `with TestClient(app) as c`.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api import app
from src.features import PIMA_ZERO_AS_MISSING
from src.predict import predict_proba, risk_band
from tests.conftest import DIABETIC, FEATURE_COLS, HEALTHY, MODERATE

PATIENTS = [HEALTHY, MODERATE, DIABETIC]
THRESHOLDS = [0.1, 0.28, 0.5, 0.7]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_name"] == "Extra Trees"
        assert body["trained_at"]


class TestModelEndpoint:
    def test_model_schema(self, client, artifact):
        r = client.get("/model")
        assert r.status_code == 200
        body = r.json()
        assert set(body.keys()) == {
            "model_name", "threshold", "test_metrics",
            "feature_cols", "feature_medians", "trained_at",
        }
        assert body["model_name"] == "Extra Trees"
        assert body["threshold"] == pytest.approx(0.28)
        assert body["feature_cols"] == FEATURE_COLS
        assert set(body["feature_medians"].keys()) == set(FEATURE_COLS)
        assert set(body["test_metrics"]) >= {"roc_auc", "recall", "precision", "f1"}
        assert body["trained_at"] == artifact["trained_at"]


class TestPredict:
    @pytest.mark.parametrize("threshold", THRESHOLDS)
    @pytest.mark.parametrize("patient", PATIENTS, ids=["healthy", "moderate", "diabetic"])
    def test_predict_coherence(self, client, artifact, patient, threshold):
        """prediction == 1 <=> band != 'low'; prob matches src.predict; band matches risk_band."""
        r = client.post("/predict", params={"threshold": threshold}, json=patient)
        assert r.status_code == 200
        body = r.json()

        expected_prob = float(predict_proba(artifact, pd.DataFrame([patient]))[0])
        assert body["probability"] == pytest.approx(expected_prob, abs=1e-9)
        assert body["threshold"] == pytest.approx(threshold)
        assert body["prediction"] == int(body["probability"] >= threshold)
        assert body["risk_band"] == risk_band(body["probability"], threshold)
        # coherence invariant
        assert (body["prediction"] == 1) == (body["risk_band"] != "low")

    def test_predict_default_threshold_is_artifact(self, client, artifact):
        r = client.post("/predict", json=HEALTHY)
        assert r.status_code == 200
        assert r.json()["threshold"] == pytest.approx(artifact["threshold"])

    def test_imputed_fields_exactly_the_zero_fields(self, client):
        payload = dict(HEALTHY, Insulin=0, SkinThickness=0)
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        assert sorted(r.json()["imputed_fields"]) == ["Insulin", "SkinThickness"]

    def test_imputed_fields_empty_when_all_measured(self, client):
        # HEALTHY has Pregnancies=0, but Pregnancies is not a zero-as-missing field
        r = client.post("/predict", json=HEALTHY)
        assert r.status_code == 200
        assert r.json()["imputed_fields"] == []

    def test_imputed_fields_only_from_pima_list(self, client):
        payload = dict(HEALTHY, Glucose=0, BloodPressure=0, SkinThickness=0,
                       Insulin=0, BMI=0)
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        assert sorted(r.json()["imputed_fields"]) == sorted(PIMA_ZERO_AS_MISSING)

    def test_predict_validation_rejects_missing_required(self, client):
        r = client.post("/predict", json={"Pregnancies": 1})  # no Glucose/BMI/Age
        assert r.status_code == 422


class TestPredictBatch:
    @pytest.mark.parametrize("threshold", [0.28, 0.5])
    def test_batch_parity_with_single(self, client, threshold):
        rb = client.post("/predict_batch", params={"threshold": threshold}, json=PATIENTS)
        assert rb.status_code == 200
        batch = rb.json()
        assert batch["threshold"] == pytest.approx(threshold)
        assert len(batch["results"]) == len(PATIENTS)

        for patient, res in zip(PATIENTS, batch["results"]):
            rs = client.post("/predict", params={"threshold": threshold}, json=patient)
            single = rs.json()
            assert res["probability"] == pytest.approx(single["probability"], abs=1e-9)
            assert res["prediction"] == single["prediction"]
            assert res["risk_band"] == single["risk_band"]
            assert sorted(res["imputed_fields"]) == sorted(single["imputed_fields"])

    def test_batch_coherence(self, client):
        rb = client.post("/predict_batch", json=PATIENTS)
        assert rb.status_code == 200
        for res in rb.json()["results"]:
            assert (res["prediction"] == 1) == (res["risk_band"] != "low")

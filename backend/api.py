"""FastAPI service that serves the trained diabetes prediction model.

Run from the project root (after `python -m src.train`):
    uvicorn backend.api:app --reload
"""
from contextlib import asynccontextmanager
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import PIMA_ZERO_AS_MISSING
from src.geo import geo_risk_summary, load_places
from src.predict import load_artifact, predict_proba, risk_band

state = {"artifact": None, "places": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["artifact"] = load_artifact()
    try:
        state["places"] = load_places()
    except Exception:
        # Geo context is optional; predictions still work without it
        state["places"] = None
    yield


app = FastAPI(title="Diabetes Prediction API", lifespan=lifespan)


class Patient(BaseModel):
    Pregnancies: float = Field(0, ge=0)
    Glucose: float = Field(..., ge=0)
    BloodPressure: float = Field(0, ge=0)
    SkinThickness: float = Field(0, ge=0)
    Insulin: float = Field(0, ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(0.0, ge=0)
    Age: float = Field(..., ge=1)


class Prediction(BaseModel):
    probability: float
    prediction: int
    threshold: float
    risk_band: str
    # Zero-as-missing fields (Glucose, BloodPressure, SkinThickness, Insulin,
    # BMI) that were sent as 0 and therefore median-imputed by the pipeline.
    imputed_fields: Optional[List[str]] = None
    geo_context: Optional[dict] = None


def _require_artifact() -> dict:
    artifact = state["artifact"]
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run `python -m src.train` first.",
        )
    return artifact


def _imputed_fields(record: dict) -> List[str]:
    """Zero-as-missing fields that were 0 in the request (imputed downstream)."""
    return [c for c in PIMA_ZERO_AS_MISSING if record.get(c, 0) == 0]


@app.get("/health")
def health():
    artifact = state["artifact"]
    return {
        "status": "ok",
        "model_loaded": artifact is not None,
        "model_name": artifact["model_name"] if artifact else None,
        "trained_at": artifact["trained_at"] if artifact else None,
    }


@app.get("/model")
def model_info():
    """Metadata about the loaded model artifact (for clients and reproducibility)."""
    artifact = _require_artifact()
    return {
        "model_name": artifact["model_name"],
        "threshold": artifact["threshold"],
        "test_metrics": artifact["test_metrics"],
        "feature_cols": artifact["feature_cols"],
        "feature_medians": artifact["feature_medians"],
        "trained_at": artifact["trained_at"],
    }


@app.post("/predict", response_model=Prediction)
def predict_one(
    patient: Patient,
    threshold: Optional[float] = None,
    county_fips: Optional[str] = None,
):
    artifact = _require_artifact()
    t = artifact["threshold"] if threshold is None else threshold
    record = patient.model_dump()
    df = pd.DataFrame([record])
    prob = float(predict_proba(artifact, df)[0])

    geo = None
    if county_fips is not None:
        if state["places"] is None:
            raise HTTPException(status_code=503, detail="Geo data unavailable.")
        geo = geo_risk_summary(state["places"], county_fips, prob)
        if geo is None:
            raise HTTPException(
                status_code=404, detail=f"Unknown county FIPS '{county_fips}'."
            )

    return Prediction(
        probability=prob,
        prediction=int(prob >= t),
        threshold=t,
        risk_band=risk_band(prob, t),
        imputed_fields=_imputed_fields(record),
        geo_context=geo,
    )


@app.get("/counties")
def list_counties(state_abbr: Optional[str] = None):
    if state["places"] is None:
        raise HTTPException(status_code=503, detail="Geo data unavailable.")
    df = state["places"]
    if state_abbr:
        df = df[df["state"] == state_abbr.upper()]
    cols = ["county_fips", "county", "state", "diabetes_pct"]
    return df[cols].to_dict(orient="records")


@app.post("/predict_batch")
def predict_batch(patients: List[Patient], threshold: Optional[float] = None):
    artifact = _require_artifact()
    t = artifact["threshold"] if threshold is None else threshold
    records = [p.model_dump() for p in patients]
    df = pd.DataFrame(records)
    probs = predict_proba(artifact, df)
    return {
        "threshold": t,
        "results": [
            {
                "probability": float(p),
                "prediction": int(p >= t),
                "risk_band": risk_band(float(p), t),
                "imputed_fields": _imputed_fields(rec),
            }
            for p, rec in zip(probs, records)
        ],
    }

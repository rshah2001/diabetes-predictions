"""FastAPI service that serves the trained diabetes prediction model.

Run from the project root (after `python -m src.train`):
    uvicorn backend.api:app --reload
"""
from contextlib import asynccontextmanager
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.geo import geo_risk_summary, load_places
from src.predict import load_artifact, predict_proba

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
    geo_context: Optional[dict] = None


def _require_artifact() -> dict:
    artifact = state["artifact"]
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run `python -m src.train` first.",
        )
    return artifact


def _risk_band(prob: float) -> str:
    if prob >= 0.70:
        return "high"
    if prob >= 0.40:
        return "moderate"
    return "low"


@app.get("/health")
def health():
    artifact = state["artifact"]
    return {
        "status": "ok",
        "model_loaded": artifact is not None,
        "model_name": artifact["model_name"] if artifact else None,
        "trained_at": artifact["trained_at"] if artifact else None,
    }


@app.post("/predict", response_model=Prediction)
def predict_one(
    patient: Patient,
    threshold: Optional[float] = None,
    county_fips: Optional[str] = None,
):
    artifact = _require_artifact()
    t = artifact["threshold"] if threshold is None else threshold
    df = pd.DataFrame([patient.model_dump()])
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
        risk_band=_risk_band(prob),
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
    df = pd.DataFrame([p.model_dump() for p in patients])
    probs = predict_proba(artifact, df)
    return {
        "threshold": t,
        "results": [
            {
                "probability": float(p),
                "prediction": int(p >= t),
                "risk_band": _risk_band(float(p)),
            }
            for p in probs
        ],
    }

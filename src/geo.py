"""County-level health context from CDC PLACES (2024 release).

Downloads adult diabetes / obesity / inactivity / blood-pressure / insurance
prevalence for every US county, caches it locally, and combines it with the
model's personal risk score so a prediction can be read in the context of
where the patient lives.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = PROJECT_ROOT / "data" / "cdc_places_county.csv"

# PLACES: County Data (GIS Friendly Format), 2024 release
PLACES_DATASET_ID = "d3i6-k6z5"
PLACES_COLUMNS = {
    "stateabbr": "state",
    "statedesc": "state_name",
    "countyname": "county",
    "countyfips": "county_fips",
    "totalpopulation": "population",
    "diabetes_crudeprev": "diabetes_pct",
    "obesity_crudeprev": "obesity_pct",
    "lpa_crudeprev": "physical_inactivity_pct",
    "bphigh_crudeprev": "high_blood_pressure_pct",
    "access2_crudeprev": "no_insurance_pct",
    "geolocation": "geolocation",
}

# Prevalence of diabetes in the training cohort (Pima dataset: 268/768).
# The model's probabilities are calibrated to this base rate.
TRAIN_PREVALENCE = 268 / 768

MEASURE_LABELS = {
    "diabetes_pct": "Diagnosed diabetes (adults)",
    "obesity_pct": "Obesity (adults)",
    "physical_inactivity_pct": "No leisure-time physical activity",
    "high_blood_pressure_pct": "High blood pressure (adults)",
    "no_insurance_pct": "No health insurance (18-64)",
}


def _places_url() -> str:
    select = ",".join(PLACES_COLUMNS)
    return (
        f"https://data.cdc.gov/resource/{PLACES_DATASET_ID}.csv"
        f"?$select={select}&$limit=4000"
    )


def _parse_point(value: str) -> tuple:
    m = re.search(r"POINT \((-?[\d.]+) (-?[\d.]+)\)", str(value))
    if not m:
        return (np.nan, np.nan)
    return (float(m.group(2)), float(m.group(1)))  # (lat, lon)


def load_places(force_download: bool = False) -> pd.DataFrame:
    """Returns county PLACES data, downloading and caching on first use."""
    if CACHE_PATH.exists() and not force_download:
        return pd.read_csv(CACHE_PATH, dtype={"county_fips": str})

    df = pd.read_csv(_places_url())
    df = df.rename(columns=PLACES_COLUMNS)
    latlon = df["geolocation"].map(_parse_point)
    df["lat"] = [p[0] for p in latlon]
    df["lon"] = [p[1] for p in latlon]
    df = df.drop(columns=["geolocation"])
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    df = df.dropna(subset=["diabetes_pct"])

    CACHE_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    return df


def get_county(df: pd.DataFrame, county_fips: str) -> Optional[pd.Series]:
    rows = df[df["county_fips"] == str(county_fips).zfill(5)]
    return rows.iloc[0] if len(rows) else None


def national_percentile(df: pd.DataFrame, measure: str, value: float) -> float:
    """Percent of counties with a value below this one (0-100)."""
    col = df[measure].dropna()
    return float((col < value).mean() * 100)


def county_context(df: pd.DataFrame, county_fips: str) -> Optional[dict]:
    """County prevalence measures with national medians and percentiles."""
    row = get_county(df, county_fips)
    if row is None:
        return None

    measures = {}
    for col, label in MEASURE_LABELS.items():
        val = float(row[col])
        measures[col] = {
            "label": label,
            "county_pct": val,
            "national_median_pct": float(df[col].median()),
            "percentile": national_percentile(df, col, val),
        }

    return {
        "county_fips": row["county_fips"],
        "county": row["county"],
        "state": row["state"],
        "state_name": row["state_name"],
        "population": int(row["population"]),
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "measures": measures,
    }


def adjust_probability_for_prevalence(
    prob: float,
    local_prevalence: float,
    train_prevalence: float = TRAIN_PREVALENCE,
) -> float:
    """Shifts a calibrated probability from the training base rate to a local one.

    Standard prior-correction: multiply the model's odds by the ratio of the
    local prevalence odds to the training-cohort prevalence odds. Useful for
    reading a model trained on a high-prevalence cohort (Pima, ~35%) against
    a general adult population (county prevalence, typically 8-15%).
    """
    prob = float(np.clip(prob, 1e-6, 1 - 1e-6))
    model_odds = prob / (1 - prob)
    ratio = (local_prevalence / (1 - local_prevalence)) / (
        train_prevalence / (1 - train_prevalence)
    )
    adj_odds = model_odds * ratio
    return float(adj_odds / (1 + adj_odds))


def geo_risk_summary(df: pd.DataFrame, county_fips: str, prob: float) -> Optional[dict]:
    """Combines a model risk score with county context."""
    ctx = county_context(df, county_fips)
    if ctx is None:
        return None

    county_prev = ctx["measures"]["diabetes_pct"]["county_pct"] / 100.0
    adjusted = adjust_probability_for_prevalence(prob, county_prev)
    return {
        **ctx,
        "model_probability": float(prob),
        "county_prevalence": county_prev,
        "adjusted_probability": adjusted,
        "relative_risk_vs_county": float(adjusted / county_prev) if county_prev else None,
    }

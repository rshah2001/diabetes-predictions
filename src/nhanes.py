"""NHANES 2017-2018 loader for external validation of the Pima-trained model.

The Pima dataset and NHANES were collected decades apart on different
populations, so only a subset of features is comparable. We map the four
that align cleanly and build a diabetes outcome from the diabetes
questionnaire, then validate a Pima-trained model on NHANES women aged 21+
(matching the Pima inclusion criteria) — a genuine cross-cohort test.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = PROJECT_ROOT / "data" / "nhanes_2017_2018.csv"
XPT_CACHE = PROJECT_ROOT / "data" / "nhanes_xpt"

NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"

# Features shared between Pima and NHANES (the transportable subset)
SHARED_FEATURES = ["Glucose", "BloodPressure", "BMI", "Age"]

FILES = {
    "DEMO_J": ["SEQN", "RIAGENDR", "RIDAGEYR"],
    "GLU_J": ["SEQN", "LBXGLU"],
    "BMX_J": ["SEQN", "BMXBMI"],
    "BPX_J": ["SEQN", "BPXDI1"],
    "DIQ_J": ["SEQN", "DIQ010"],
}


def _read_xpt(name: str) -> pd.DataFrame:
    XPT_CACHE.mkdir(parents=True, exist_ok=True)
    local = XPT_CACHE / f"{name}.XPT"
    if not local.exists():
        req = urllib.request.Request(
            f"{NHANES_BASE}/{name}.XPT", headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req) as r, open(local, "wb") as f:
            f.write(r.read())
    return pd.read_sas(local, format="xport")


def load_nhanes(force_download: bool = False) -> pd.DataFrame:
    """Returns NHANES women 21+ with Pima-aligned columns, cached locally.

    Columns: Glucose, BloodPressure, BMI, Age, Outcome (1 = diagnosed diabetes).
    """
    if CACHE_PATH.exists() and not force_download:
        return pd.read_csv(CACHE_PATH)

    frames = {name: _read_xpt(name)[cols] for name, cols in FILES.items()}

    df = frames["DEMO_J"]
    for name in ["GLU_J", "BMX_J", "BPX_J", "DIQ_J"]:
        df = df.merge(frames[name], on="SEQN", how="left")

    df = df.rename(
        columns={
            "LBXGLU": "Glucose",
            "BPXDI1": "BloodPressure",
            "BMXBMI": "BMI",
            "RIDAGEYR": "Age",
        }
    )

    # Match Pima inclusion criteria: female, age >= 21
    df = df[(df["RIAGENDR"] == 2) & (df["Age"] >= 21)]

    # Diabetes outcome from DIQ010: 1 = yes, 2 = no (drop borderline/refused/unknown)
    df = df[df["DIQ010"].isin([1, 2])]
    df["Outcome"] = (df["DIQ010"] == 1).astype(int)

    # Diastolic blood pressure recorded as 0 means a failed reading, not real data
    df.loc[df["BloodPressure"] == 0, "BloodPressure"] = pd.NA

    out = df[SHARED_FEATURES + ["Outcome"]].dropna(subset=["Glucose", "BMI", "Age"])
    out = out.reset_index(drop=True)

    CACHE_PATH.parent.mkdir(exist_ok=True)
    out.to_csv(CACHE_PATH, index=False)
    return out


def load_pima_shared() -> pd.DataFrame:
    """Pima dataset restricted to the NHANES-shared feature subset."""
    pima = pd.read_csv(PROJECT_ROOT / "diabetes.csv")
    return pima[SHARED_FEATURES + ["Outcome"]]

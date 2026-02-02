from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
import pandas as pd

from src.models import ets_forecast, seasonal_naive_forecast, ml_forecast


DB_PATH = "data.db"
TABLE = "demand"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            ds TEXT PRIMARY KEY,
            y REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def read_series() -> pd.Series:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT ds, y FROM {TABLE} ORDER BY ds", conn)
    conn.close()
    if df.empty:
        return pd.Series(dtype=float)
    df["ds"] = pd.to_datetime(df["ds"])
    return pd.Series(df["y"].values, index=df["ds"])


class IngestPoint(BaseModel):
    ds: str  # ISO datetime string
    y: float


app = FastAPI(title="Demand Forecasting Backend")


@app.on_event("startup")
def startup():
    init_db()


@app.post("/ingest")
def ingest(point: IngestPoint):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        f"INSERT OR REPLACE INTO {TABLE} (ds, y) VALUES (?, ?)",
        (point.ds, float(point.y)),
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "ingested": {"ds": point.ds, "y": point.y}}


@app.get("/forecast")
def forecast(model: str = "ML (Gradient Boosting)", horizon: int = 14, season_length: int = 7):
    y = read_series()
    if len(y) < 20:
        return {"error": "Not enough data yet. Ingest more points."}

    if model == "Seasonal Naive":
        fc = seasonal_naive_forecast(y, horizon, season_length)
    elif model == "ETS (Holt-Winters)":
        fc = ets_forecast(y, horizon, season_length)
    elif model == "ML (Gradient Boosting)":
        fc = ml_forecast(y, horizon, season_length)
    else:
        return {"error": f"Unknown model '{model}'"}

    future_index = pd.date_range(start=y.index[-1], periods=horizon + 1, freq=pd.infer_freq(y.index) or "D")[1:]
    out = [{"ds": str(d), "y_pred": float(v)} for d, v in zip(future_index, fc)]
    return {"model": model, "horizon": horizon, "forecast": out}

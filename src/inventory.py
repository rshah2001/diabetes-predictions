import numpy as np
from math import sqrt

# Simple z-values for common service levels (good enough for MVP)
Z_TABLE = {
    0.90: 1.282,
    0.95: 1.645,
    0.97: 1.881,
    0.98: 2.054,
    0.99: 2.326,
}

def z_value(service_level: float) -> float:
    # Snap to closest key to avoid requiring scipy
    keys = np.array(sorted(Z_TABLE.keys()))
    idx = int(np.argmin(np.abs(keys - service_level)))
    return float(Z_TABLE[float(keys[idx])])

def safety_stock(rmse_value: float, lead_time_days: int, service_level: float) -> float:
    z = z_value(service_level)
    # Assume independent daily errors: sigma_LT = RMSE * sqrt(L)
    return z * rmse_value * sqrt(max(1, lead_time_days))

def reorder_point(avg_daily_demand: float, lead_time_days: int, safety_stock_units: float) -> float:
    return avg_daily_demand * max(1, lead_time_days) + safety_stock_units

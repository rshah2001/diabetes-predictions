**Current version:** v2.0.0  

# Demand Forecasting Studio

An end-to-end **time series forecasting + decision support** app built with **Streamlit**.  
Upload a dataset → clean/standardize it into a `ds/y` time series → run **walk-forward backtests** across models → generate a **forward forecast** → translate forecast error into **inventory safety stock + reorder point**.

---

## What this app does

### ✅ Workflow
1. **Upload & Prepare (Page 1)**
   - Choose date column + target column
   - Converts data into a standardized time series with columns:
     - `ds` = datetime
     - `y`  = numeric target (e.g., demand, sales, revenue)
   - Resamples to your chosen frequency (default: Daily)

2. **Insights (Page 2)**
   - Quick exploratory views (trend + smoothed signal)

3. **Model Compare (Page 3)**
   - Runs **walk-forward backtesting** to evaluate models realistically (no future leakage)
   - Shows leaderboard metrics: MAE, RMSE, sMAPE, WAPE
   - Stores the best model + error stats for downstream pages

4. **Decision Impact (Page 4)**
   - Converts backtest error (RMSE) into:
     - **Safety Stock**
     - **Reorder Point (ROP)**
   - If a forward forecast exists, uses **forecasted lead-time demand** (sum of next L days)

5. **Forecast (Page 5)**
   - Trains the selected/best model on the full series
   - Generates a forward forecast horizon
   - Optional uncertainty band (RMSE-based)
   - Cleaner plotting (zoom + rolling mean overlay)

---

## Models included
- **Seasonal Naive** (baseline)
- **ETS / Holt-Winters**
- **XGBoost (GBDT)** (recommended for speed + scalability)
- **ML Gradient Boosting** (educational; can be slow for large horizons)

> Note: ML Gradient Boosting can be expensive because it may retrain repeatedly across forecast steps, which can freeze laptops on large horizons.

---

## How to run

### 1) Install dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

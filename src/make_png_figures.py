"""Regenerate the report figures as high-resolution PNGs for document embedding.

Uses cached data, so it's fast (no cross-validation / bootstrap-heavy compute
beyond a single model fit). Run after `python -m src.report`.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from src.evaluation import expected_calibration_error, reliability_curve
from src.features import PimaPreprocessor
from src.geo import TRAIN_PREVALENCE, adjust_probability_for_prevalence, load_places
from src.report import _pipe

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PNG = PROJECT_ROOT / "reports" / "png"
SEED = 42
plt.rcParams.update({"figure.dpi": 200, "font.size": 10, "savefig.bbox": "tight"})


def _et():
    return ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=3, max_features="sqrt",
        random_state=SEED, n_jobs=-1, class_weight="balanced")


def main():
    PNG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROJECT_ROOT / "diabetes.csv")
    X, y = df.drop(columns=["Outcome"]), df["Outcome"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    base = _pipe(_et()).fit(X_tr, y_tr)
    p_uncal = base.predict_proba(X_te)[:, 1]
    cal = CalibratedClassifierCV(_pipe(_et()), method="sigmoid", cv=5).fit(X_tr, y_tr)
    p_cal = cal.predict_proba(X_te)[:, 1]

    # Calibration
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for p, lab in [(p_uncal, "Uncalibrated"), (p_cal, "Calibrated (Platt)")]:
        mp, fp, _ = reliability_curve(y_te.values, p, 10)
        ax.plot(mp, fp, "o-", label=f"{lab} (ECE={expected_calibration_error(y_te.values, p):.3f})")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraction positive")
    ax.set_title("Reliability diagram"); ax.legend(loc="upper left")
    fig.savefig(PNG / "calibration.png"); plt.close(fig)

    # ROC band
    rng = np.random.default_rng(SEED); grid = np.linspace(0, 1, 100); tprs = []
    for _ in range(1000):
        idx = rng.integers(0, len(y_te), len(y_te))
        if len(np.unique(y_te.values[idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_te.values[idx], p_cal[idx])
        tprs.append(np.interp(grid, fpr, tpr))
    tprs = np.array(tprs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(grid, tprs.mean(0), label=f"Extra Trees (AUC={roc_auc_score(y_te, p_cal):.3f})")
    ax.fill_between(grid, np.percentile(tprs, 2.5, 0), np.percentile(tprs, 97.5, 0),
                    alpha=0.25, label="95% bootstrap CI")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve with bootstrap confidence band"); ax.legend(loc="lower right")
    fig.savefig(PNG / "roc_curves.png"); plt.close(fig)

    # SHAP
    import shap
    prep = PimaPreprocessor(add_derived=True).fit(X, y)
    Xt = prep.transform(X); names = list(Xt.columns)
    model = _et().fit(Xt, y)
    sv = shap.TreeExplainer(model).shap_values(Xt)
    sv_pos = sv[..., 1] if (isinstance(sv, np.ndarray) and sv.ndim == 3) else (sv[1] if isinstance(sv, list) else sv)
    fig = plt.figure()
    shap.summary_plot(sv_pos, Xt, feature_names=names, show=False)
    plt.title("SHAP global feature importance")
    plt.savefig(PNG / "shap_summary.png", bbox_inches="tight"); plt.close(fig)

    # Geo map
    places = load_places(); m = places.dropna(subset=["lat", "lon"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sc = ax.scatter(m["lon"], m["lat"], c=m["diabetes_pct"], s=6, cmap="RdYlGn_r", alpha=0.8)
    ax.set_xlim(-125, -66); ax.set_ylim(24, 50)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Adult diagnosed diabetes prevalence by US county (CDC PLACES 2024)")
    plt.colorbar(sc, ax=ax, label="Diabetes prevalence (%)")
    fig.savefig(PNG / "geo_map.png"); plt.close(fig)

    # Geo prior correction
    prevs = places["diabetes_pct"].dropna() / 100.0
    lo_c, med_c, hi_c = prevs.quantile([0.05, 0.5, 0.95]); grid = np.linspace(0.01, 0.99, 99)
    fig, ax = plt.subplots(figsize=(6, 5))
    for cp, lab in [(lo_c, f"Low-prevalence county ({lo_c:.1%})"),
                    (med_c, f"Median county ({med_c:.1%})"),
                    (hi_c, f"High-prevalence county ({hi_c:.1%})")]:
        ax.plot(grid, [adjust_probability_for_prevalence(p, cp) for p in grid], label=lab)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="No adjustment")
    ax.set_xlabel(f"Model probability (trained at {TRAIN_PREVALENCE:.0%} prevalence)")
    ax.set_ylabel("Prevalence-adjusted probability")
    ax.set_title("GeoAI prior correction across county base rates"); ax.legend()
    fig.savefig(PNG / "geo_prior_correction.png"); plt.close(fig)

    print(f"PNG figures written to {PNG}/")


if __name__ == "__main__":
    main()

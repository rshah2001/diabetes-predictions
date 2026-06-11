"""Generates all publication assets for the book chapter into reports/.

Run from the project root:
    python -m src.report

Produces (CSV tables + vector PDF figures):
  - ablation_table.csv          component-by-component CV ROC-AUC
  - bootstrap_cis.csv           held-out metrics with 95% bootstrap CIs
  - model_comparison.csv        repeated-CV AUC per model + DeLong vs best
  - external_validation.csv     Pima-trained model evaluated on NHANES
  - calibration.pdf             reliability diagram (before/after calibration)
  - roc_curves.pdf              ROC with bootstrap CI band
  - shap_summary.pdf            global SHAP feature importance
  - shap_waterfall.pdf          per-patient SHAP explanation
  - geo_map.pdf                 US county diabetes prevalence (GeoAI)
  - geo_prior_correction.pdf    prior-correction curves across counties
  - metrics_summary.json        machine-readable roll-up of headline numbers
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import (
    bootstrap_ci,
    delong_roc_test,
    expected_calibration_error,
    point_metrics,
    reliability_curve,
)
from src.features import PimaPreprocessor
from src.geo import (
    MEASURE_LABELS,
    TRAIN_PREVALENCE,
    adjust_probability_for_prevalence,
    load_places,
)
from src.models import build_classification_models
from src.nhanes import SHARED_FEATURES, load_nhanes, load_pima_shared

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "reports"
DATA = PROJECT_ROOT / "diabetes.csv"
TARGET = "Outcome"
SEED = 42
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "savefig.bbox": "tight"})


def _pipe(model, add_derived=True, scale=True):
    steps = [("prep", PimaPreprocessor(add_derived=add_derived))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def load_data():
    df = pd.read_csv(DATA)
    return df.drop(columns=[TARGET]), df[TARGET].astype(int)


# --------------------------------------------------------------------------
# 1) Ablation study
# --------------------------------------------------------------------------
def ablation_study(X, y) -> pd.DataFrame:
    print("\n[1/6] Ablation study...")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    from sklearn.ensemble import ExtraTreesClassifier

    def et():
        return ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=3, max_features="sqrt",
            random_state=SEED, n_jobs=-1, class_weight="balanced",
        )

    configs = {
        "Raw features (no cleaning)": _pipe_raw(et()),
        "+ zeros-as-missing imputation": _pipe(et(), add_derived=False, scale=True),
        "+ derived clinical features": _pipe(et(), add_derived=True, scale=True),
    }
    rows = []
    for name, pipe in configs.items():
        auc = cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
        rows.append({"Configuration": name, "CV ROC-AUC": auc.mean(), "Std": auc.std()})
        print(f"   {auc.mean():.4f} ± {auc.std():.4f}  {name}")

    # Calibration + stacking effect on the final model
    stack = build_classification_models()["Stacked Ensemble"]
    auc = cross_val_score(_pipe(stack), X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
    rows.append({"Configuration": "Stacked ensemble (full pipeline)",
                 "CV ROC-AUC": auc.mean(), "Std": auc.std()})
    print(f"   {auc.mean():.4f} ± {auc.std():.4f}  Stacked ensemble (full pipeline)")

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "ablation_table.csv", index=False)
    return df


def _pipe_raw(model):
    # No zeros-as-missing handling, median impute only, no derived features
    return Pipeline([
        ("prep", PimaPreprocessor(zero_as_missing_cols=[], add_derived=False)),
        ("scale", StandardScaler()),
        ("model", model),
    ])


# --------------------------------------------------------------------------
# 2) Model comparison with DeLong test
# --------------------------------------------------------------------------
def model_comparison(X, y) -> pd.DataFrame:
    print("\n[2/6] Model comparison + DeLong test...")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    models = build_classification_models()
    rows, oof = [], {}
    for name, model in models.items():
        pipe = _pipe(model)
        auc = cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
        rows.append({"Model": name, "CV ROC-AUC": auc.mean(), "Std": auc.std()})
        # single-split OOF probs for DeLong
        from sklearn.model_selection import StratifiedKFold
        oof[name] = cross_val_predict(
            _pipe(model), X, y,
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
            method="predict_proba", n_jobs=-1,
        )[:, 1]
        print(f"   {auc.mean():.4f} ± {auc.std():.4f}  {name}")

    df = pd.DataFrame(rows).sort_values("CV ROC-AUC", ascending=False).reset_index(drop=True)
    best = df.iloc[0]["Model"]
    df["DeLong p vs best"] = [
        np.nan if m == best else delong_roc_test(y.values, oof[best], oof[m])["p_value"]
        for m in df["Model"]
    ]
    df.to_csv(REPORTS / "model_comparison.csv", index=False)
    return df


# --------------------------------------------------------------------------
# 3) Bootstrap CIs + ROC band + calibration
# --------------------------------------------------------------------------
def holdout_analysis(X, y) -> dict:
    print("\n[3/6] Bootstrap CIs, ROC band, calibration...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    from sklearn.ensemble import ExtraTreesClassifier
    base = _pipe(ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=3, max_features="sqrt",
        random_state=SEED, n_jobs=-1, class_weight="balanced"))

    # Uncalibrated vs calibrated probabilities
    base.fit(X_tr, y_tr)
    p_uncal = base.predict_proba(X_te)[:, 1]

    cal = CalibratedClassifierCV(_pipe(ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=3, max_features="sqrt",
        random_state=SEED, n_jobs=-1, class_weight="balanced")),
        method="sigmoid", cv=5).fit(X_tr, y_tr)
    p_cal = cal.predict_proba(X_te)[:, 1]

    # threshold tuned on training OOF for F1
    from sklearn.model_selection import StratifiedKFold
    oof = cross_val_predict(
        CalibratedClassifierCV(_pipe(ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=3, max_features="sqrt",
            random_state=SEED, n_jobs=-1, class_weight="balanced")),
            method="sigmoid", cv=5),
        X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        method="predict_proba", n_jobs=-1)[:, 1]
    from sklearn.metrics import f1_score
    thr = max(np.arange(0.2, 0.81, 0.01),
              key=lambda t: f1_score(y_tr, (oof >= t).astype(int)))

    cis = bootstrap_ci(y_te.values, p_cal, float(thr))
    pd.DataFrame(
        [{"metric": k, **v} for k, v in cis.items()]
    ).to_csv(REPORTS / "bootstrap_cis.csv", index=False)
    for k, v in cis.items():
        print(f"   {k:10s} {v['estimate']:.3f}  [{v['ci_low']:.3f}, {v['ci_high']:.3f}]")

    _plot_calibration(y_te.values, p_uncal, p_cal)
    _plot_roc_band(y_te.values, p_cal)

    return {"threshold": float(thr), "test_metrics_ci": cis,
            "ece_uncalibrated": expected_calibration_error(y_te.values, p_uncal),
            "ece_calibrated": expected_calibration_error(y_te.values, p_cal)}


def _plot_calibration(y, p_uncal, p_cal):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for p, lab in [(p_uncal, "Uncalibrated"), (p_cal, "Calibrated (Platt)")]:
        mp, fp, _ = reliability_curve(y, p, n_bins=10)
        ece = expected_calibration_error(y, p)
        ax.plot(mp, fp, "o-", label=f"{lab} (ECE={ece:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left")
    fig.savefig(REPORTS / "calibration.pdf")
    plt.close(fig)


def _plot_roc_band(y, p, n_boot=1000):
    from sklearn.metrics import roc_curve, roc_auc_score
    rng = np.random.default_rng(SEED)
    grid = np.linspace(0, 1, 100)
    tprs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[idx], p[idx])
        tprs.append(np.interp(grid, fpr, tpr))
    tprs = np.array(tprs)
    mean_tpr, lo, hi = tprs.mean(0), np.percentile(tprs, 2.5, 0), np.percentile(tprs, 97.5, 0)
    auc = roc_auc_score(y, p)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(grid, mean_tpr, label=f"Extra Trees (AUC={auc:.3f})")
    ax.fill_between(grid, lo, hi, alpha=0.25, label="95% bootstrap CI")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve with bootstrap confidence band")
    ax.legend(loc="lower right")
    fig.savefig(REPORTS / "roc_curves.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# 4) SHAP explainability
# --------------------------------------------------------------------------
def shap_analysis(X, y):
    print("\n[4/6] SHAP explainability...")
    import shap
    from sklearn.ensemble import ExtraTreesClassifier

    prep = PimaPreprocessor(add_derived=True).fit(X, y)
    X_t = prep.transform(X)
    names = list(X_t.columns)
    model = ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=3, max_features="sqrt",
        random_state=SEED, n_jobs=-1, class_weight="balanced").fit(X_t, y)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_t)
    sv_pos = sv[..., 1] if isinstance(sv, np.ndarray) and sv.ndim == 3 else (
        sv[1] if isinstance(sv, list) else sv)

    fig = plt.figure()
    shap.summary_plot(sv_pos, X_t, feature_names=names, show=False)
    plt.title("SHAP global feature importance")
    plt.savefig(REPORTS / "shap_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    # per-patient waterfall for a high-risk case
    base_val = explainer.expected_value
    base_val = base_val[1] if np.ndim(base_val) else base_val
    i = int(np.argmax(model.predict_proba(X_t)[:, 1]))
    expl = shap.Explanation(values=sv_pos[i], base_values=base_val,
                            data=X_t.iloc[i].values, feature_names=names)
    fig = plt.figure()
    shap.plots.waterfall(expl, show=False)
    plt.title(f"SHAP explanation — patient #{i} (high risk)")
    plt.savefig(REPORTS / "shap_waterfall.pdf", bbox_inches="tight")
    plt.close(fig)

    imp = pd.DataFrame({"feature": names,
                        "mean_abs_shap": np.abs(sv_pos).mean(0)}
                       ).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(REPORTS / "shap_importance.csv", index=False)
    print("   top features:", ", ".join(imp["feature"].head(4)))
    return imp


# --------------------------------------------------------------------------
# 5) External validation on NHANES
# --------------------------------------------------------------------------
def external_validation() -> dict:
    print("\n[5/6] External validation (Pima -> NHANES)...")
    pima = load_pima_shared()
    nh = load_nhanes()
    Xp, yp = pima[SHARED_FEATURES], pima[TARGET].astype(int)
    Xn, yn = nh[SHARED_FEATURES], nh[TARGET].astype(int)

    from sklearn.ensemble import ExtraTreesClassifier
    # Shared-feature model; no derived features (some need Insulin etc. absent here)
    clf = CalibratedClassifierCV(
        _pipe(ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=3, max_features="sqrt",
            random_state=SEED, n_jobs=-1, class_weight="balanced"),
            add_derived=False),
        method="sigmoid", cv=5).fit(Xp, yp)

    # Internal (Pima holdout) vs external (NHANES) at threshold 0.5
    X_tr, X_te, y_tr, y_te = train_test_split(Xp, yp, test_size=0.2, stratify=yp, random_state=SEED)
    clf_int = CalibratedClassifierCV(
        _pipe(ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=3, max_features="sqrt",
            random_state=SEED, n_jobs=-1, class_weight="balanced"),
            add_derived=False), method="sigmoid", cv=5).fit(X_tr, y_tr)

    p_int = clf_int.predict_proba(X_te)[:, 1]
    p_ext = clf.predict_proba(Xn)[:, 1]
    m_int = point_metrics(y_te.values, p_int, 0.5)
    m_ext = point_metrics(yn.values, p_ext, 0.5)
    ext_ci = bootstrap_ci(yn.values, p_ext, 0.5)

    rows = []
    for k in m_int:
        rows.append({"metric": k, "internal_pima": m_int[k], "external_nhanes": m_ext[k],
                     "external_ci_low": ext_ci.get(k, {}).get("ci_low"),
                     "external_ci_high": ext_ci.get(k, {}).get("ci_high")})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "external_validation.csv", index=False)
    print(f"   Internal AUC {m_int['roc_auc']:.3f}  ->  External AUC {m_ext['roc_auc']:.3f}"
          f"  [{ext_ci['roc_auc']['ci_low']:.3f}, {ext_ci['roc_auc']['ci_high']:.3f}]")
    return {"internal": m_int, "external": m_ext,
            "external_ci": {k: ext_ci[k] for k in ext_ci}}


# --------------------------------------------------------------------------
# 6) GeoAI figures
# --------------------------------------------------------------------------
def geo_figures() -> dict:
    print("\n[6/6] GeoAI figures...")
    places = load_places()

    # County diabetes prevalence map
    fig, ax = plt.subplots(figsize=(9, 5.5))
    m = places.dropna(subset=["lat", "lon"])
    sc = ax.scatter(m["lon"], m["lat"], c=m["diabetes_pct"], s=6,
                    cmap="RdYlGn_r", alpha=0.8)
    ax.set_xlim(-125, -66); ax.set_ylim(24, 50)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Adult diagnosed diabetes prevalence by US county (CDC PLACES 2024)")
    plt.colorbar(sc, ax=ax, label="Diabetes prevalence (%)")
    fig.savefig(REPORTS / "geo_map.pdf")
    plt.close(fig)

    # Prior-correction curves: how a model score maps to adjusted risk
    # across the observed range of county prevalences
    prevs = places["diabetes_pct"].dropna() / 100.0
    lo_c, med_c, hi_c = prevs.quantile([0.05, 0.5, 0.95])
    grid = np.linspace(0.01, 0.99, 99)
    fig, ax = plt.subplots(figsize=(6, 5))
    for cp, lab in [(lo_c, f"Low-prevalence county ({lo_c:.1%})"),
                    (med_c, f"Median county ({med_c:.1%})"),
                    (hi_c, f"High-prevalence county ({hi_c:.1%})")]:
        adj = [adjust_probability_for_prevalence(p, cp) for p in grid]
        ax.plot(grid, adj, label=lab)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="No adjustment")
    ax.set_xlabel(f"Model probability (trained at {TRAIN_PREVALENCE:.0%} prevalence)")
    ax.set_ylabel("Prevalence-adjusted probability")
    ax.set_title("GeoAI prior correction across county base rates")
    ax.legend()
    fig.savefig(REPORTS / "geo_prior_correction.pdf")
    plt.close(fig)

    # Validation: adjusting at the *training* prevalence is the identity map
    ident_err = max(abs(adjust_probability_for_prevalence(p, TRAIN_PREVALENCE) - p)
                    for p in grid)
    print(f"   counties mapped: {len(m)}; identity-map max error: {ident_err:.2e}")
    summary = {
        "n_counties": int(len(m)),
        "prevalence_min": float(prevs.min()),
        "prevalence_median": float(med_c),
        "prevalence_max": float(prevs.max()),
        "identity_map_max_error": float(ident_err),
    }
    pd.DataFrame([{"measure": k, **{
        "county_pct": places[k].median(),
        "label": v}} for k, v in MEASURE_LABELS.items()]
    ).to_csv(REPORTS / "geo_measures_national.csv", index=False)
    return summary


def main():
    REPORTS.mkdir(exist_ok=True)
    X, y = load_data()
    summary = {}
    summary["ablation"] = ablation_study(X, y).to_dict(orient="records")
    summary["model_comparison"] = model_comparison(X, y).to_dict(orient="records")
    summary["holdout"] = holdout_analysis(X, y)
    summary["shap_top"] = shap_analysis(X, y).head(6).to_dict(orient="records")
    summary["external_validation"] = external_validation()
    summary["geo"] = geo_figures()

    (REPORTS / "metrics_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nAll assets written to {REPORTS}/")


if __name__ == "__main__":
    main()

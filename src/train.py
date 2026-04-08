"""
train.py

EBM training pipeline for rail station ridership prediction.

Validation strategy: 5-fold stratified cross-validation, stratified on
log-ridership quantiles so every fold sees the full range of station sizes.
After CV, a final model is fit on all data and saved.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_loader import (
    ALL_FEATURES,
    TARGET,
    get_feature_matrix,
    load_stations,
)

MODELS_DIR = ROOT / "models"
SHAPE_DIR = ROOT / "results" / "shape_functions"
METRICS_DIR = ROOT / "results" / "metrics"
N_FOLDS = 5

for d in [MODELS_DIR, SHAPE_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Training ───────────────────────────────────────────────────────────────────

def make_ebm(feature_names: list) -> ExplainableBoostingRegressor:
    return ExplainableBoostingRegressor(
        feature_names=feature_names,
        interactions=3,
        max_bins=256,
        learning_rate=0.01,
        min_samples_leaf=2,
        random_state=42,
    )


# ── Shape function plots ───────────────────────────────────────────────────────

def save_shape_plots(ebm: ExplainableBoostingRegressor, feature_names: list):
    explanation = ebm.explain_global(name="EBM v1")

    for i, name in enumerate(feature_names):
        try:
            feat_data = explanation.data(i)
        except Exception:
            continue

        names = feat_data.get("names", [])
        scores = feat_data.get("scores", [])
        if len(names) == 0 or len(scores) == 0:
            continue
        if isinstance(scores, np.ndarray) and scores.ndim == 2:
            continue  # skip interaction terms

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(names), y=list(scores),
                                 mode="lines+markers", name=name))
        fig.update_layout(
            title=f"Shape function: {name}",
            xaxis_title=name,
            yaxis_title="Effect on log(ridership)",
            template="plotly_white",
        )
        fig.write_image(str(SHAPE_DIR / f"{name.replace('/', '_')}.png"))

    print(f"  Shape plots saved to {SHAPE_DIR}/")


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray,
                    label: str = "") -> dict:
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    r2 = r2_score(y_true_log, y_pred_log)
    mape = np.mean(
        np.abs((np.expm1(y_true_log) - np.expm1(y_pred_log))
               / np.clip(np.expm1(y_true_log), 1, None))
    ) * 100

    print(f"\n{label} metrics:")
    print(f"  RMSE (log scale): {rmse_log:.4f}")
    print(f"  R²:               {r2:.4f}")
    print(f"  MAPE:             {mape:.1f}%")

    return {"label": label, "rmse_log": rmse_log, "r2": r2, "mape": mape}


# ── Cross-validation ───────────────────────────────────────────────────────────

def run_cv(X: pd.DataFrame, y_log: np.ndarray,
           name_col: pd.Series) -> tuple[np.ndarray, list]:
    """
    5-fold stratified CV on log-ridership quantiles.
    Returns out-of-fold predictions (same length as X) and per-fold metrics.
    """
    # Bin log-ridership into N_FOLDS quantiles for stratification
    quantile_bins = pd.qcut(y_log, q=N_FOLDS, labels=False, duplicates="drop")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, quantile_bins), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_log[train_idx], y_log[val_idx]

        ebm = make_ebm(X.columns.tolist())
        ebm.fit(X_tr, y_tr)

        preds = ebm.predict(X_val)
        oof_preds[val_idx] = preds

        m = compute_metrics(y_val, preds, label=f"Fold {fold}")
        fold_metrics.append(m)

        # Show a few representative predictions per fold
        sample = pd.DataFrame({
            "station": name_col.iloc[val_idx].values,
            "actual": np.expm1(y_val).astype(int),
            "predicted": np.expm1(preds).astype(int),
        }).nlargest(3, "actual")
        print(f"  Top 3 by ridership in fold {fold}:")
        print(sample.to_string(index=False))

    return oof_preds, fold_metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    df = load_stations()
    df = df[df[TARGET].notna()].reset_index(drop=True)
    print(f"  {len(df)} stations with ridership")

    X = get_feature_matrix(df)
    y_log = df["log_ridership"].values
    name_col = df["station_name"] if "station_name" in df.columns else df["map_station"]

    print(f"\nFeatures ({len(X.columns)}): {X.columns.tolist()}")

    # ── 5-fold stratified CV ──
    print(f"\nRunning {N_FOLDS}-fold stratified CV …")
    oof_preds, fold_metrics = run_cv(X, y_log, name_col)

    # Aggregate OOF metrics (every station predicted exactly once)
    print("\n── Overall cross-validation (out-of-fold) ──")
    oof_metrics = compute_metrics(y_log, oof_preds, label="CV (OOF)")

    fold_df = pd.DataFrame(fold_metrics)
    print(f"\nPer-fold summary:")
    print(fold_df[["label", "rmse_log", "r2", "mape"]].to_string(index=False))
    print(f"\nMean ± std:  RMSE={fold_df['rmse_log'].mean():.4f} ± {fold_df['rmse_log'].std():.4f}"
          f"  R²={fold_df['r2'].mean():.4f} ± {fold_df['r2'].std():.4f}")

    # ── Final model on all data ──
    print("\nFitting final model on all data …")
    final_ebm = make_ebm(X.columns.tolist())
    final_ebm.fit(X, y_log)
    print("  Done.")

    # ── Save model ──
    model_path = MODELS_DIR / "ebm_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"ebm": final_ebm, "features": X.columns.tolist()}, f)
    print(f"Model saved to {model_path}")

    # ── Shape plots ──
    print("\nGenerating shape function plots …")
    try:
        save_shape_plots(final_ebm, X.columns.tolist())
    except Exception as e:
        print(f"  Warning: shape plot generation failed: {e}")

    # ── Save metrics ──
    all_metrics = fold_df.copy()
    all_metrics = pd.concat([all_metrics, pd.DataFrame([oof_metrics])], ignore_index=True)
    all_metrics.to_csv(METRICS_DIR / "training_metrics_v1.csv", index=False)

    # Save OOF predictions for inspection
    oof_df = pd.DataFrame({
        "station": name_col.values,
        "actual_ridership": np.expm1(y_log).astype(int),
        "oof_predicted_ridership": np.expm1(oof_preds).astype(int),
        "pct_error": ((np.expm1(oof_preds) - np.expm1(y_log))
                      / np.expm1(y_log) * 100).round(1),
    })
    oof_df.to_csv(METRICS_DIR / "oof_predictions_v1.csv", index=False)
    print(f"Metrics saved to {METRICS_DIR}/")

    # ── Compare vs v0 ──
    v0_path = METRICS_DIR / "training_metrics.csv"
    if v0_path.exists():
        v0 = pd.read_csv(v0_path)
        v0_oof = v0[v0["label"] == "CV (OOF)"].iloc[0]
        print("\n── v0 vs v1 comparison (OOF) ──")
        print(f"  {'Metric':<18} {'v0':>10} {'v1':>10} {'delta':>10}")
        print(f"  {'RMSE (log)':18} {v0_oof['rmse_log']:10.4f} {oof_metrics['rmse_log']:10.4f}"
              f" {oof_metrics['rmse_log'] - v0_oof['rmse_log']:+10.4f}")
        print(f"  {'R²':18} {v0_oof['r2']:10.4f} {oof_metrics['r2']:10.4f}"
              f" {oof_metrics['r2'] - v0_oof['r2']:+10.4f}")
        print(f"  {'MAPE':18} {v0_oof['mape']:10.1f} {oof_metrics['mape']:10.1f}"
              f" {oof_metrics['mape'] - v0_oof['mape']:+10.1f}")

    return final_ebm


if __name__ == "__main__":
    main()
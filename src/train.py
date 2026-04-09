"""
train.py

EBM training pipeline for rail station ridership prediction.

Validation strategy: 5-fold stratified group CV. Stations are first clustered
into 20 geographic groups via k-means on raw lat/lon; groups are passed to
StratifiedGroupKFold so that geographically proximate stations (e.g. Trenton
and Princeton Junction) always appear in the same fold, preventing leakage.
Stratification is on log-ridership quantiles so every fold sees the full range
of station sizes. After CV, a final model is fit on all data and saved.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold

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

    skipped = 0
    for i, name in enumerate(feature_names):
        plot_path = SHAPE_DIR / f"{name.replace('/', '_')}.png"
        if plot_path.exists():
            skipped += 1
            continue

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
        fig.write_image(str(plot_path))

    if skipped:
        print(f"  Skipped {skipped} existing plots (delete {SHAPE_DIR}/ to regenerate)")
    print(f"  Shape plots saved to {SHAPE_DIR}/")


def save_shape_data(ebm: ExplainableBoostingRegressor, feature_names: list):
    """
    Save shape function values to CSV for programmatic analysis.

    Outputs two files:
    - shape_functions.csv: one row per (feature, bin) with x value and score
    - feature_importance.csv: one row per feature, ranked by mean |score|
    """
    explanation = ebm.explain_global(name="EBM v1")

    rows = []
    importance_rows = []

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

        scores_arr = np.array(scores)
        mean_abs_score = float(np.mean(np.abs(scores_arr)))
        score_range = float(scores_arr.max() - scores_arr.min())

        importance_rows.append({
            "feature": name,
            "mean_abs_score": round(mean_abs_score, 5),
            "score_range": round(score_range, 5),
            "n_bins": len(names),
        })

        for x_val, score in zip(names, scores_arr):
            rows.append({
                "feature": name,
                "x": x_val,
                "score": round(float(score), 5),
            })

    shape_df = pd.DataFrame(rows)
    shape_df.to_csv(METRICS_DIR / "shape_functions.csv", index=False)

    imp_df = pd.DataFrame(importance_rows).sort_values("mean_abs_score", ascending=False)
    imp_df.to_csv(METRICS_DIR / "feature_importance.csv", index=False)

    print(f"  Shape data saved to {METRICS_DIR}/shape_functions.csv")
    print(f"\nFeature importance (mean |score|):")
    print(imp_df.to_string(index=False))


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray,
                    label: str = "") -> dict:
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    r2 = r2_score(y_true_log, y_pred_log)

    print(f"\n{label} metrics:")
    print(f"  RMSE (log scale): {rmse_log:.4f}")
    print(f"  R²:               {r2:.4f}")

    return {"label": label, "rmse_log": rmse_log, "r2": r2}


# ── Geographic clustering ─────────────────────────────────────────────────────

N_GEO_CLUSTERS = 20

def make_geo_groups(df: pd.DataFrame) -> np.ndarray:
    """
    Cluster stations into N_GEO_CLUSTERS geographic groups via k-means on
    raw lat/lon.  Stations with missing coordinates are assigned cluster -1
    (treated as their own singleton group by StratifiedGroupKFold).
    """
    has_coords = df["lat"].notna() & df["lon"].notna()
    coords = df.loc[has_coords, ["lat", "lon"]].values

    km = KMeans(n_clusters=N_GEO_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(coords)

    groups = np.full(len(df), -1, dtype=int)
    groups[has_coords.values] = labels

    # Diagnostic: cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Geographic clusters (k={N_GEO_CLUSTERS}): "
          f"min={counts.min()}, median={int(np.median(counts))}, max={counts.max()} stations/cluster")

    return groups


# ── Cross-validation ───────────────────────────────────────────────────────────

def run_cv(X: pd.DataFrame, y_log: np.ndarray,
           name_col: pd.Series, groups: np.ndarray) -> tuple[np.ndarray, list]:
    """
    5-fold stratified group CV on log-ridership quantiles.
    Groups are geographic clusters so nearby stations stay in the same fold.
    Returns out-of-fold predictions (same length as X) and per-fold metrics.
    """
    # Bin log-ridership into N_FOLDS quantiles for stratification
    quantile_bins = pd.qcut(y_log, q=N_FOLDS, labels=False, duplicates="drop")

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS)

    oof_preds = np.zeros(len(X))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, quantile_bins, groups=groups), 1):
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

    # ── Geographic grouping for CV ──
    print(f"\nClustering stations into geographic groups …")
    groups = make_geo_groups(df)

    # ── 5-fold stratified group CV ──
    print(f"\nRunning {N_FOLDS}-fold stratified group CV …")
    oof_preds, fold_metrics = run_cv(X, y_log, name_col, groups)

    # Aggregate OOF metrics (every station predicted exactly once)
    print("\n── Overall cross-validation (out-of-fold) ──")
    oof_metrics = compute_metrics(y_log, oof_preds, label="CV (OOF)")

    fold_df = pd.DataFrame(fold_metrics)
    print(f"\nPer-fold summary:")
    print(fold_df[["label", "rmse_log", "r2"]].to_string(index=False))
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

    # ── Shape plots + data ──
    print("\nGenerating shape function plots …")
    try:
        save_shape_plots(final_ebm, X.columns.tolist())
        save_shape_data(final_ebm, X.columns.tolist())
    except Exception as e:
        print(f"  Warning: shape plot/data generation failed: {e}")

    # ── Save metrics ──
    all_metrics = fold_df.copy()
    all_metrics = pd.concat([all_metrics, pd.DataFrame([oof_metrics])], ignore_index=True)
    all_metrics.to_csv(METRICS_DIR / "training_metrics_v1.csv", index=False)

    # Save OOF predictions for inspection
    code_col = df["code"] if "code" in df.columns else pd.Series([""] * len(df))
    oof_df = pd.DataFrame({
        "code": code_col.values,
        "station": name_col.values,
        "actual_ridership": np.expm1(y_log).astype(int),
        "oof_predicted_ridership": np.expm1(oof_preds).astype(int),
        "pct_error": ((np.expm1(oof_preds) - np.expm1(y_log))
                      / np.expm1(y_log) * 100).round(1),
    })
    oof_df.to_csv(METRICS_DIR / "oof_predictions_v1.csv", index=False)
    print(f"Metrics saved to {METRICS_DIR}/")

    return final_ebm


if __name__ == "__main__":
    main()
"""Shared inference logic for pipelines and API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loader import add_active_flag, load_raw_data, validate_date_index
from src.drift.inference import pick_best_model_name
from src.features.assembly import assemble_feature_matrix, prepare_supervised
from src.models.baseline import predict_naive, train_naive_baseline
from src.models.multi_output import DualTargetModels, predict_dual_target
from src.models.tabular import predict_tabular
from src.selection.feature_selector import load_feature_sets
from src.utils.config import load_config, load_json, load_model


def _latest_prediction_date(feature_df: pd.DataFrame) -> str:
    if "is_active" in feature_df.columns:
        return str(feature_df.loc[feature_df["is_active"], "Date"].iloc[-1].date())
    return str(feature_df["Date"].iloc[-1].date())


def predict_next_day(
    config_path: str = "config/model_config.yaml",
    model_name: str | None = None,
    metrics_summary: dict[str, Any] | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    """Return next-day forecast with Income, Outcome, and Balance."""
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    artifacts = config.get("artifacts", {})
    target_col = data_cfg.get("target_column", "Balance")

    df = load_raw_data(data_cfg.get("path", "data.csv"), data_cfg.get("date_column", "Date"))
    df = validate_date_index(df, data_cfg.get("date_column", "Date"))
    df = add_active_flag(df)
    feature_df = assemble_feature_matrix(df, config)

    if metrics_summary is None:
        metrics_path = Path(artifacts.get("metrics", "artifacts/metrics.json"))
        metrics_summary = load_json(metrics_path) if metrics_path.exists() else {}

    if model_name is None:
        model_name = pick_best_model_name(metrics_summary.get("model_metrics", {}))
    if model_name is None:
        model_name = "dual_target_naive"

    if model is None:
        models_dir = Path(artifacts.get("models_dir", "artifacts/models"))
        model_path = models_dir / f"{model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        model = load_model(model_path)

    feature_sets = load_feature_sets(artifacts.get("feature_sets", "artifacts/feature_sets.json"))
    pred_date = _latest_prediction_date(feature_df)

    if isinstance(model, DualTargetModels) or model_name.startswith("dual_target"):
        work = feature_df[feature_df.get("is_active", True)].copy()
        feature_names = [
            c for c in work.columns if c not in {"Date", "Income", "Outcome", "Balance", "is_active"}
        ]
        valid = work[feature_names].notna().all(axis=1)
        X_latest = work.loc[valid, feature_names].iloc[[-1]]
        preds = predict_dual_target(model, X_latest)
        return {
            "model": model_name,
            "date": pred_date,
            "Income": float(preds["Income"][0]),
            "Outcome": float(preds["Outcome"][0]),
            "Balance": float(preds["Balance"][0]),
            "partial": False,
        }

    X, y, feature_names = prepare_supervised(feature_df, target_col=target_col, active_only=True)
    if hasattr(model, "pipeline"):
        features = getattr(model, "features", feature_sets.get("spearman", feature_names))
        balance = float(predict_tabular(model, X.iloc[[-1]])[0])
    elif hasattr(model, "forecast"):
        exog_cols = getattr(model, "exog_columns", None)
        if exog_cols:
            balance = float(model.forecast(steps=1, exog=X.iloc[[-1]])[0])
        else:
            balance = float(model.forecast(steps=1)[0])
    else:
        naive = train_naive_baseline(y)
        balance = float(predict_naive(naive, 1)[0])

    return {
        "model": model_name,
        "date": pred_date,
        "Income": None,
        "Outcome": None,
        "Balance": balance,
        "partial": True,
    }

"""Hold-out prediction helpers for drift baseline setup."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.baseline import predict_naive, train_naive_baseline
from src.models.multi_output import DualTargetModels, predict_dual_target
from src.models.tabular import predict_tabular


def compute_holdout_predictions(
    model_name: str,
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    feature_sets: dict[str, list[str]] | None = None,
) -> np.ndarray:
    """Generate hold-out predictions for the selected model."""
    feature_sets = feature_sets or {}

    if model_name == "naive" or (hasattr(model, "last_value") and not hasattr(model, "forecast")):
        naive = model if hasattr(model, "last_value") else train_naive_baseline(y_train)
        return np.asarray(predict_naive(naive, len(X_test)), dtype=float)

    if hasattr(model, "forecast"):
        exog_cols = getattr(model, "exog_columns", None)
        if exog_cols:
            return np.asarray(model.forecast(steps=len(X_test), exog=X_test), dtype=float)
        return np.asarray(model.forecast(steps=len(X_test)), dtype=float)

    if isinstance(model, DualTargetModels) or model_name.startswith("dual_target"):
        preds = predict_dual_target(model, X_test)
        return np.asarray(preds["Balance"], dtype=float)

    if hasattr(model, "pipeline"):
        features = getattr(model, "features", feature_sets.get("spearman", X_test.columns.tolist()))
        return np.asarray(predict_tabular(model, X_test[features]), dtype=float)

    raise ValueError(f"Unsupported model type for hold-out predictions: {model_name}")


def pick_best_model_name(metrics: dict[str, Any], target: str = "Balance") -> str | None:
    best_name = None
    best_mae = float("inf")
    for name, model_metrics in metrics.items():
        if isinstance(model_metrics, dict) and target in model_metrics:
            mae = model_metrics[target].get("mae", float("inf"))
        elif isinstance(model_metrics, dict):
            mae = model_metrics.get("mae", float("inf"))
        else:
            continue
        if mae < best_mae:
            best_mae = mae
            best_name = name
    return best_name

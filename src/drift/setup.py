"""Initialize drift baseline and residual history after training."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.drift.detector import DriftBaseline, DriftDetector
from src.drift.history import build_holdout_records, save_residual_history
from src.drift.inference import compute_holdout_predictions, pick_best_model_name
from src.utils.config import load_config


def setup_drift_artifacts(
    model_metrics: dict[str, Any],
    models: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    split_idx: int,
    feature_sets: dict[str, list[str]],
    monitoring_config_path: str = "config/monitoring_config.yaml",
) -> dict[str, Any] | None:
    """Save drift baseline and hold-out residual history for the best model."""
    monitoring_config = load_config(monitoring_config_path)
    if not monitoring_config.get("drift", {}).get("enabled", True):
        return None

    best_name = pick_best_model_name(model_metrics)
    if best_name is None or best_name not in models:
        return None

    n = min(len(X), len(y))
    split_idx = min(split_idx, n - 1)
    if split_idx < 1:
        return None

    X = X.iloc[:n]
    y = y.iloc[:n]
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    y_pred = compute_holdout_predictions(
        best_name,
        models[best_name],
        X_train,
        X_test,
        y_train,
        feature_sets=feature_sets,
    )
    residuals = np.asarray(y_test, dtype=float) - y_pred
    baseline = DriftDetector.fit_baseline(residuals, model_name=best_name)

    artifact_cfg = monitoring_config.get("artifacts", {})
    baseline_path = artifact_cfg.get("drift_baseline", "artifacts/drift_baseline.json")
    history_path = artifact_cfg.get("residual_history", "artifacts/residual_history.json")

    DriftDetector.save_baseline(baseline, baseline_path)
    test_dates = y_test.index if isinstance(y_test.index, pd.DatetimeIndex) else pd.RangeIndex(len(y_test))
    records = build_holdout_records(test_dates, y_test.values, y_pred, best_name)
    save_residual_history(history_path, records)

    return baseline.to_dict()

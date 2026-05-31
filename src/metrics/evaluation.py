"""Hold-out evaluation of saved model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.loader import add_active_flag, load_raw_data, validate_date_index
from src.drift.inference import compute_holdout_predictions, pick_best_model_name
from src.features.assembly import assemble_feature_matrix, prepare_supervised
from src.metrics.business_metrics import business_report, passes_quality_gate, rank_models_by_business_score
from src.models.multi_output import (
    DualTargetModels,
    evaluate_dual_target,
    prepare_aligned_dual_data,
)
from src.selection.feature_selector import load_feature_sets
from src.utils.config import load_config, load_json, load_model


def _get_split_idx(config: dict[str, Any], metrics_summary: dict[str, Any] | None, n_rows: int) -> int:
    if metrics_summary and "n_train" in metrics_summary:
        return min(int(metrics_summary["n_train"]), n_rows - 1)
    holdout_ratio = config.get("split", {}).get("holdout_ratio", 0.8)
    return int(n_rows * holdout_ratio)


def _index_to_dates(index: pd.Index) -> list[str]:
    dates: list[str] = []
    for idx in index:
        if isinstance(idx, pd.Timestamp):
            dates.append(str(idx.date()))
        else:
            dates.append(str(idx))
    return dates


def collect_holdout_predictions(
    config_path: str = "config/model_config.yaml",
    models_dir: str | Path | None = None,
    *,
    max_models: int | None = None,
    model_names: list[str] | None = None,
) -> dict[str, Any]:
    """Collect hold-out Balance predictions for visualization."""
    config = load_config(config_path)
    artifacts = config.get("artifacts", {})
    metrics_path = Path(artifacts.get("metrics", "artifacts/metrics.json"))
    eval_path = Path(artifacts.get("eval_metrics", "artifacts/eval_metrics.json"))
    train_summary = load_json(metrics_path) if metrics_path.exists() else {}
    eval_summary = load_json(eval_path) if eval_path.exists() else {}
    training_mode = train_summary.get("training_mode", config.get("training", {}).get("mode", "dual_target"))

    data_cfg = config.get("data", {})
    df = load_raw_data(data_cfg.get("path", "data.csv"), data_cfg.get("date_column", "Date"))
    df = validate_date_index(df, data_cfg.get("date_column", "Date"))
    df = add_active_flag(df)
    feature_df = assemble_feature_matrix(df, config)
    feature_sets = load_feature_sets(artifacts.get("feature_sets", "artifacts/feature_sets.json"))

    models_path = Path(models_dir or artifacts.get("models_dir", "artifacts/models"))
    model_files = sorted(models_path.glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No model artifacts found in {models_path}")

    if training_mode == "dual_target":
        X, y_income, y_outcome, y_balance, _ = prepare_aligned_dual_data(
            feature_df,
            data_cfg.get("income_column", "Income"),
            data_cfg.get("outcome_column", "Outcome"),
            data_cfg.get("derived_target", "Balance"),
        )
        split_idx = _get_split_idx(config, train_summary, len(X))
        split_idx = min(split_idx, len(X) - 1)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_test = y_balance.iloc[split_idx:]
        y_train_balance = y_balance.iloc[:split_idx]
        test_index = y_test.index
    else:
        target_col = data_cfg.get("target_column", "Balance")
        X, y, _ = prepare_supervised(feature_df, target_col=target_col, active_only=True)
        split_idx = _get_split_idx(config, train_summary, len(X))
        split_idx = min(split_idx, len(X) - 1)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        y_train_balance = y.iloc[:split_idx]
        test_index = y_test.index

    if model_names is None:
        model_names = [p.stem for p in model_files]
        if max_models is not None:
            ranked = pick_best_model_name(eval_summary.get("model_metrics", {}))
            preferred = [n for n in model_names if n.startswith("dual_target") or n in {"arima", "sarima", "arimax"}]
            if ranked and ranked in model_names:
                preferred = [ranked] + [n for n in preferred if n != ranked]
            model_names = preferred[:max_models]

    arima_names = frozenset({"arima", "sarima", "arimax"})
    predictions: dict[str, Any] = {}
    for model_name in model_names:
        model_path = models_path / f"{model_name}.joblib"
        if not model_path.exists():
            continue
        model = load_model(model_path)
        if isinstance(model, DualTargetModels) or model_name.startswith("dual_target"):
            preds = compute_holdout_predictions(model_name, model, X_train, X_test, y_train_balance, feature_sets)
        elif model_name in arima_names or training_mode != "dual_target":
            preds = compute_holdout_predictions(model_name, model, X_train, X_test, y_train_balance, feature_sets)
        else:
            continue
        predictions[model_name] = {"y_pred": np.asarray(preds, dtype=float).tolist()}

    return {
        "training_mode": training_mode,
        "n_train": split_idx,
        "n_test": len(y_test),
        "dates": _index_to_dates(test_index),
        "y_true": np.asarray(y_test, dtype=float).tolist(),
        "models": predictions,
    }


def evaluate_saved_models(
    config_path: str = "config/model_config.yaml",
    models_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute hold-out metrics for all saved models."""
    config = load_config(config_path)
    artifacts = config.get("artifacts", {})
    metrics_path = Path(artifacts.get("metrics", "artifacts/metrics.json"))
    train_summary = load_json(metrics_path) if metrics_path.exists() else {}
    training_mode = train_summary.get("training_mode", config.get("training", {}).get("mode", "dual_target"))
    threshold = config.get("metrics", {}).get("balance_mae_threshold", 0.42)

    data_cfg = config.get("data", {})
    df = load_raw_data(data_cfg.get("path", "data.csv"), data_cfg.get("date_column", "Date"))
    df = validate_date_index(df, data_cfg.get("date_column", "Date"))
    df = add_active_flag(df)
    feature_df = assemble_feature_matrix(df, config)
    feature_sets = load_feature_sets(artifacts.get("feature_sets", "artifacts/feature_sets.json"))

    models_path = Path(models_dir or artifacts.get("models_dir", "artifacts/models"))
    model_files = sorted(models_path.glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No model artifacts found in {models_path}")

    if training_mode == "dual_target":
        X, y_income, y_outcome, y_balance, _ = prepare_aligned_dual_data(
            feature_df,
            data_cfg.get("income_column", "Income"),
            data_cfg.get("outcome_column", "Outcome"),
            data_cfg.get("derived_target", "Balance"),
        )
        split_idx = _get_split_idx(config, train_summary, len(X))
        split_idx = min(split_idx, len(X) - 1)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_income_test = y_income.iloc[split_idx:]
        y_outcome_test = y_outcome.iloc[split_idx:]
        y_balance_test = y_balance.iloc[split_idx:]
        y_test = y_balance.iloc[split_idx:]
    else:
        target_col = data_cfg.get("target_column", "Balance")
        X, y, _ = prepare_supervised(feature_df, target_col=target_col, active_only=True)
        split_idx = _get_split_idx(config, train_summary, len(X))
        split_idx = min(split_idx, len(X) - 1)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

    arima_names = frozenset({"arima", "sarima", "arimax"})
    eval_metrics: dict[str, Any] = {}
    quality_gate: dict[str, bool] = {}

    for model_file in model_files:
        model_name = model_file.stem
        model = load_model(model_file)
        if isinstance(model, DualTargetModels) or model_name.startswith("dual_target"):
            eval_metrics[model_name] = evaluate_dual_target(
                model,
                X_test,
                y_income_test,
                y_outcome_test,
                y_balance_test,
                threshold=threshold,
            )
            quality_gate[model_name] = passes_quality_gate(eval_metrics[model_name]["Balance"], threshold)
        elif model_name in arima_names:
            y_train_balance = y_balance.iloc[:split_idx] if training_mode == "dual_target" else y.iloc[:split_idx]
            preds = compute_holdout_predictions(model_name, model, X_train, X_test, y_train_balance, feature_sets)
            eval_metrics[model_name] = business_report(y_test, preds, threshold=threshold)
            quality_gate[model_name] = passes_quality_gate(eval_metrics[model_name], threshold)
        else:
            if training_mode == "dual_target":
                continue
            y_train = y.iloc[:split_idx]
            preds = compute_holdout_predictions(model_name, model, X_train, X_test, y_train, feature_sets)
            eval_metrics[model_name] = business_report(y_test, preds, threshold=threshold)
            quality_gate[model_name] = passes_quality_gate(eval_metrics[model_name], threshold)

    business_ranking = rank_models_by_business_score(eval_metrics)

    return {
        "training_mode": training_mode,
        "n_train": split_idx,
        "n_test": len(X_test),
        "model_metrics": eval_metrics,
        "quality_gate_passed": quality_gate,
        "business_ranking": [{"model": name, "asymmetric_cost": cost} for name, cost in business_ranking],
        "train_metrics": train_summary.get("model_metrics", {}),
    }

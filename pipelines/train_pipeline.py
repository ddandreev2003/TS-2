"""End-to-end training workflow for liquidity forecasting.

ПОЧЕМУ: единый CLI-пайплайн закрывает требование автоматизированного прогноза D+1
        с FS, HPO, ARIMA-benchmark, calibration study и drift baseline.
КАК: load → features → FS (auto winner) → dual_target + ARIMA → save artifacts.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from src.calibration.recalibrator import run_calibration_study, select_calibration_policy
from src.data.loader import add_active_flag, build_data_quality_report, load_raw_data, validate_date_index
from src.features.assembly import assemble_feature_matrix, prepare_supervised
from src.metrics.business_metrics import business_report, passes_quality_gate, rank_models_by_business_score
from src.models.arima_family import (
    fit_arima_model,
    fit_arimax_model,
    fit_sarima_model,
    grid_search_arima,
    grid_search_arimax,
    grid_search_sarima,
)
from src.models.baseline import predict_naive, train_naive_baseline
from src.models.multi_output import (
    dual_target_artifact_name,
    evaluate_dual_target,
    prepare_aligned_dual_data,
    train_dual_target_pair,
)
from src.models.tabular import predict_tabular, train_linear_model, train_nonlinear_model
from src.selection.feature_selector import resolve_fs_features, run_feature_selection_cv, save_feature_sets, select_best_fs_method
from src.drift.setup import setup_drift_artifacts
from src.mlops.experiment_tracker import log_training_run
from src.utils.config import ensure_artifact_dirs, load_config, save_json, save_model


def temporal_split(n_rows: int, holdout_ratio: float) -> int:
    return int(n_rows * holdout_ratio)


def _evaluate_predictions(y_true, y_pred, threshold: float) -> dict[str, float]:
    return business_report(y_true, y_pred, threshold=threshold)


def train_arima_benchmark(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any],
    feature_sets: dict[str, list[str]],
    split_idx: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """ПОЧЕМУ: заказчик требует исследовать ARIMA/SARIMA/ARIMAX в едином сравнении с ML."""
    threshold = config.get("metrics", {}).get("balance_mae_threshold", 0.42)
    arimax_k = config.get("selection", {}).get("arimax_exog_count", 12)
    fs_method = config.get("selection", {}).get("selected_method", "spearman")
    linear_features = feature_sets.get(fs_method, X.columns.tolist()) or X.columns.tolist()
    arimax_features = linear_features[:arimax_k]

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    metrics: dict[str, Any] = {}
    models: dict[str, Any] = {}

    arima_order, _ = grid_search_arima(y_train, config)
    arima_model = fit_arima_model(y_train, arima_order)
    arima_pred = arima_model.forecast(steps=len(y_test))
    metrics["arima"] = _evaluate_predictions(y_test, arima_pred, threshold)
    models["arima"] = arima_model

    sarima_order, seasonal_order, _ = grid_search_sarima(y_train, config)
    sarima_model = fit_sarima_model(y_train, sarima_order, seasonal_order)
    sarima_pred = sarima_model.forecast(steps=len(y_test))
    metrics["sarima"] = _evaluate_predictions(y_test, sarima_pred, threshold)
    models["sarima"] = sarima_model

    if arimax_features:
        arimax_order, _ = grid_search_arimax(y_train, X_train[arimax_features], config)
        arimax_model = fit_arimax_model(y_train, X_train, arimax_order, arimax_features)
        arimax_pred = arimax_model.forecast(steps=len(y_test), exog=X_test)
        metrics["arimax"] = _evaluate_predictions(y_test, arimax_pred, threshold)
        models["arimax"] = arimax_model

    return metrics, models


def train_balance_models(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any],
    feature_sets: dict[str, list[str]],
    split_idx: int,
    fs_method: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train all Balance-target models on temporal hold-out split."""
    threshold = config.get("metrics", {}).get("balance_mae_threshold", 0.42)
    linear_features = feature_sets.get(fs_method, X.columns.tolist()) or X.columns.tolist()

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    metrics: dict[str, Any] = {}
    models: dict[str, Any] = {}

    naive = train_naive_baseline(y_train)
    naive_pred = np.full(len(y_test), naive.last_value)
    metrics["naive"] = _evaluate_predictions(y_test, naive_pred, threshold)
    models["naive"] = naive

    arima_metrics, arima_models = train_arima_benchmark(X, y, config, feature_sets, split_idx)
    metrics.update(arima_metrics)
    models.update(arima_models)

    for name in ("lasso", "ridge", "elasticnet"):
        model = train_linear_model(X_train, y_train, name, config, features=linear_features)
        preds = predict_tabular(model, X_test)
        metrics[name] = _evaluate_predictions(y_test, preds, threshold)
        models[name] = model

    for name in ("random_forest", "extra_trees", "gradient_boosting"):
        model = train_nonlinear_model(X_train, y_train, name, config, features=X.columns.tolist())
        preds = predict_tabular(model, X_test)
        metrics[name] = _evaluate_predictions(y_test, preds, threshold)
        models[name] = model

    return metrics, models


def train_dual_target_pipeline(
    feature_df: pd.DataFrame,
    config: dict[str, Any],
    feature_sets: dict[str, list[str]],
    split_idx: int,
    fs_method: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train Income/Outcome model pairs and evaluate reconstructed Balance."""
    threshold = config.get("metrics", {}).get("balance_mae_threshold", 0.42)
    data_cfg = config.get("data", {})
    income_col = data_cfg.get("income_column", "Income")
    outcome_col = data_cfg.get("outcome_column", "Outcome")
    balance_col = data_cfg.get("derived_target", "Balance")
    model_names = config.get("training", {}).get(
        "dual_target_models",
        ["naive", "lasso", "ridge", "elasticnet", "random_forest", "extra_trees", "gradient_boosting"],
    )

    X, y_income, y_outcome, y_balance, feature_names = prepare_aligned_dual_data(
        feature_df, income_col, outcome_col, balance_col
    )

    income_fs_result = run_feature_selection_cv(X, y_income, config)
    outcome_fs_result = run_feature_selection_cv(X, y_outcome, config)
    _, income_features = resolve_fs_features(income_fs_result, config, feature_names)
    _, outcome_features = resolve_fs_features(outcome_fs_result, config, feature_names)
    income_method = select_best_fs_method(income_fs_result, config)
    outcome_method = select_best_fs_method(outcome_fs_result, config)

    split_idx = min(split_idx, len(X) - 1)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_income_train = y_income.iloc[:split_idx]
    y_income_test = y_income.iloc[split_idx:]
    y_outcome_train = y_outcome.iloc[:split_idx]
    y_outcome_test = y_outcome.iloc[split_idx:]
    y_balance_test = y_balance.iloc[split_idx:]

    metrics: dict[str, Any] = {}
    models: dict[str, Any] = {}
    for model_name in model_names:
        dual_models = train_dual_target_pair(
            model_name,
            config,
            X_train,
            y_income_train,
            y_outcome_train,
            income_features,
            outcome_features,
        )
        artifact_name = dual_target_artifact_name(model_name)
        metrics[artifact_name] = evaluate_dual_target(
            dual_models,
            X_test,
            y_income_test,
            y_outcome_test,
            y_balance_test,
            threshold=threshold,
        )
        models[artifact_name] = dual_models

    if config.get("training", {}).get("include_arima_benchmark", True):
        arima_metrics, arima_models = train_arima_benchmark(
            X, y_balance, config, feature_sets, split_idx
        )
        metrics.update(arima_metrics)
        models.update(arima_models)

    extended_feature_sets = {
        f"income_{income_method}": income_features,
        f"outcome_{outcome_method}": outcome_features,
        **feature_sets,
    }
    save_feature_sets(
        extended_feature_sets,
        config.get("artifacts", {}).get("feature_sets", "artifacts/feature_sets.json"),
    )
    return metrics, models


def run_training(config_path: str, mode_override: str | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    if mode_override:
        config.setdefault("training", {})["mode"] = mode_override
    paths = ensure_artifact_dirs(config)
    data_cfg = config.get("data", {})
    target_col = data_cfg.get("target_column", "Balance")
    training_mode = config.get("training", {}).get("mode", "dual_target")

    df = load_raw_data(data_cfg.get("path", "data.csv"), data_cfg.get("date_column", "Date"))
    df = validate_date_index(df, data_cfg.get("date_column", "Date"))
    df = add_active_flag(df)
    quality_report = build_data_quality_report(df, data_cfg.get("date_column", "Date"))

    feature_df = assemble_feature_matrix(df, config)
    X, y, _ = prepare_supervised(feature_df, target_col=target_col, active_only=True)

    fs_result = run_feature_selection_cv(X, y, config)
    fs_method, _ = resolve_fs_features(fs_result, config, X.columns.tolist())
    config.setdefault("selection", {})["selected_method"] = fs_method
    feature_sets = fs_result.consensus_sets
    save_feature_sets(feature_sets, config.get("artifacts", {}).get("feature_sets", "artifacts/feature_sets.json"))

    cal_output = config.get("artifacts", {}).get("calibration_policy", "artifacts/calibration_policy.json")
    calibration_summary = run_calibration_study(y, config, output_path=cal_output)
    calibration_policy = select_calibration_policy(y, config)

    holdout_ratio = config.get("split", {}).get("holdout_ratio", 0.8)
    split_idx = temporal_split(len(X), holdout_ratio)

    if training_mode == "dual_target":
        metrics, models = train_dual_target_pipeline(feature_df, config, feature_sets, split_idx, fs_method)
        X_for_drift, y_for_drift, _, _, _ = prepare_aligned_dual_data(
            feature_df,
            data_cfg.get("income_column", "Income"),
            data_cfg.get("outcome_column", "Outcome"),
            data_cfg.get("derived_target", "Balance"),
        )
        split_idx = min(split_idx, len(X_for_drift) - 1)
    else:
        metrics, models = train_balance_models(X, y, config, feature_sets, split_idx, fs_method)
        X_for_drift, y_for_drift = X, y

    drift_baseline = setup_drift_artifacts(
        model_metrics=metrics,
        models=models,
        X=X_for_drift,
        y=y_for_drift,
        split_idx=split_idx,
        feature_sets=feature_sets,
    )

    threshold = config.get("metrics", {}).get("balance_mae_threshold", 0.42)
    business_ranking = rank_models_by_business_score(metrics)
    summary = {
        "quality_report": quality_report,
        "training_mode": training_mode,
        "n_train": split_idx,
        "n_test": len(X) - split_idx,
        "feature_selection_method": fs_method,
        "feature_selection_ranking": fs_result.method_ranking.to_dict(orient="records") if fs_result.method_ranking is not None else [],
        "stability_metrics": fs_result.stability_metrics,
        "calibration_policy": calibration_policy.to_dict(),
        "calibration_study": calibration_summary,
        "model_metrics": metrics,
        "business_ranking": [{"model": name, "asymmetric_cost": cost} for name, cost in business_ranking],
        "quality_gate_passed": {},
    }
    if drift_baseline is not None:
        summary["drift_baseline"] = drift_baseline

    for model_name, model_metrics in metrics.items():
        if isinstance(model_metrics, dict) and "Balance" in model_metrics:
            summary["quality_gate_passed"][model_name] = passes_quality_gate(model_metrics["Balance"], threshold)
        elif isinstance(model_metrics, dict) and "mae" in model_metrics:
            summary["quality_gate_passed"][model_name] = passes_quality_gate(model_metrics, threshold)

    models_dir = paths["models_dir"]
    for name, model in models.items():
        save_model(model, models_dir / f"{name}.joblib")

    metrics_path = config.get("artifacts", {}).get("metrics", "artifacts/metrics.json")
    save_json(summary, metrics_path)

    try:
        log_training_run(summary, config_path=config_path)
    except Exception:
        pass

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train liquidity forecasting models")
    parser.add_argument("--config", default="config/model_config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    summary = run_training(args.config)
    print(f"Training complete. Models trained: {list(summary['model_metrics'].keys())}")
    if summary.get("feature_selection_method"):
        print(f"Selected FS method: {summary['feature_selection_method']}")


if __name__ == "__main__":
    main()

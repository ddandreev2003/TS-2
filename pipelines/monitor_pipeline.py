"""Drift monitoring pipeline: update residual history and run CUSUM / SR detection.

ПОЧЕМУ: онлайн-мониторинг остатков прогноза для раннего обнаружения разладки.
КАК: дополняем residual_history → detect → при auto_retrain вызываем retrain_pipeline.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loader import add_active_flag, load_raw_data, validate_date_index
from src.drift.detector import DriftDetector
from src.drift.history import (
    fill_pending_with_actuals,
    load_residual_history,
    records_to_arrays,
    save_residual_history,
)
from src.drift.inference import compute_holdout_predictions, pick_best_model_name
from src.features.assembly import assemble_feature_matrix, prepare_supervised
from src.mlops.auto_retrain import check_drift_trigger, get_retrain_recommendation
from src.selection.feature_selector import load_feature_sets
from src.utils.config import load_config, load_json, load_model, save_json


def _actuals_from_supervised(y: pd.Series) -> dict[str, float]:
    actuals: dict[str, float] = {}
    for idx, value in y.items():
        if isinstance(idx, pd.Timestamp):
            date_str = str(idx.date())
        else:
            date_str = str(idx)
        actuals[date_str] = float(value)
    return actuals


def _resolve_pending_predictions(
    records: list[dict[str, Any]],
    actuals: dict[str, float],
) -> list[dict[str, Any]]:
    return fill_pending_with_actuals(records, actuals)


def run_monitoring(
    config_path: str = "config/model_config.yaml",
    monitoring_config_path: str = "config/monitoring_config.yaml",
) -> dict[str, Any]:
    config = load_config(config_path)
    monitoring_config = load_config(monitoring_config_path)
    drift_cfg = monitoring_config.get("drift", {})
    artifact_cfg = monitoring_config.get("artifacts", {})

    if not drift_cfg.get("enabled", True):
        return {"status": "disabled", "message": "Drift monitoring is disabled"}

    baseline_path = Path(artifact_cfg.get("drift_baseline", "artifacts/drift_baseline.json"))
    history_path = Path(artifact_cfg.get("residual_history", "artifacts/residual_history.json"))
    status_path = Path(artifact_cfg.get("drift_status", "artifacts/drift_status.json"))
    alerts_dir = Path(artifact_cfg.get("drift_alerts_dir", "drift_alerts"))

    if not baseline_path.exists():
        raise FileNotFoundError(f"Drift baseline not found: {baseline_path}. Run training first.")

    baseline = DriftDetector.load_baseline(baseline_path)
    records = load_residual_history(history_path)

    data_cfg = config.get("data", {})
    target_col = data_cfg.get("target_column", "Balance")
    df = load_raw_data(data_cfg.get("path", "data.csv"), data_cfg.get("date_column", "Date"))
    df = validate_date_index(df, data_cfg.get("date_column", "Date"))
    df = add_active_flag(df)
    feature_df = assemble_feature_matrix(df, config)
    X, y, _ = prepare_supervised(feature_df, target_col=target_col, active_only=True)
    actuals = _actuals_from_supervised(y)
    records = _resolve_pending_predictions(records, actuals)

    metrics_path = Path(config.get("artifacts", {}).get("metrics", "artifacts/metrics.json"))
    metrics_summary = load_json(metrics_path) if metrics_path.exists() else {}
    model_name = baseline.model_name or pick_best_model_name(metrics_summary.get("model_metrics", {}))
    if model_name is None:
        model_name = "naive"

    holdout_ratio = config.get("split", {}).get("holdout_ratio", 0.8)
    split_idx = int(len(X) * holdout_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]

    models_dir = Path(config.get("artifacts", {}).get("models_dir", "artifacts/models"))
    model_path = models_dir / f"{model_name}.joblib"
    if model_path.exists():
        model = load_model(model_path)
        feature_sets = load_feature_sets(config.get("artifacts", {}).get("feature_sets", "artifacts/feature_sets.json"))
        existing_dates = {str(r["date"]) for r in records}
        test_dates = y.iloc[split_idx:].index
        y_test = y.iloc[split_idx:]
        y_pred = compute_holdout_predictions(
            model_name, model, X_train, X_test, y_train, feature_sets=feature_sets
        )
        for i in range(len(y_test)):
            if isinstance(test_dates[i], pd.Timestamp):
                date_str = str(test_dates[i].date())
            else:
                date_str = str(test_dates[i])
            if date_str in existing_dates:
                continue
            yt = float(y_test.iloc[i])
            yp = float(y_pred[i])
            records.append(
                {
                    "date": date_str,
                    "y_true": yt,
                    "y_pred": yp,
                    "residual": yt - yp,
                    "status": "monitor",
                    "model": model_name,
                }
            )

    save_residual_history(history_path, records)

    dates, _, _, residuals = records_to_arrays(records)
    detector = DriftDetector(monitoring_config)
    detector.set_baseline(baseline)
    alert = detector.detect(residuals, dates=dates)

    status_payload = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": "alarm" if alert.detected else "ok",
        "alert": alert.to_dict(),
        "baseline": baseline.to_dict(),
        "n_residuals": len(residuals),
    }
    retrain_recommendation = get_retrain_recommendation(status_payload, config)
    status_payload["retrain_recommendation"] = retrain_recommendation.to_dict()
    if check_drift_trigger(status_payload):
        status_payload["drift_trigger"] = True

    retrain_cfg = config.get("retraining", {})
    if alert.detected:
        from src.serving.prometheus_metrics import record_drift_alarm

        record_drift_alarm()
        alerts_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        alert_path = alerts_dir / f"alert_{ts}.json"
        save_json(status_payload, alert_path)
        print(f"DRIFT ALARM: {alert.message}", file=sys.stderr)
        print(f"Recommended actions: {', '.join(alert.recommended_actions)}", file=sys.stderr)

    if retrain_recommendation.auto_retrain and retrain_cfg.get("auto_retrain", True):
        from pipelines.retrain_pipeline import run_retraining

        retrain_result = run_retraining(
            config_path,
            reason=retrain_recommendation.reason or "auto",
            force=False,
            monitoring_config_path=monitoring_config_path,
        )
        status_payload["retrain_result"] = retrain_result
        print(f"Automatic retraining triggered: {retrain_result.get('status')}", file=sys.stderr)
    elif alert.detected:
        print("Consider manual control or unplanned retrain.", file=sys.stderr)

    save_json(status_payload, status_path)

    return status_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor forecast residuals for drift")
    parser.add_argument("--config", default="config/model_config.yaml", help="Path to model YAML config")
    parser.add_argument(
        "--monitoring",
        default="config/monitoring_config.yaml",
        help="Path to monitoring YAML config",
    )
    args = parser.parse_args()
    result = run_monitoring(args.config, args.monitoring)
    print(f"Drift monitoring complete. Status: {result.get('status')}")


if __name__ == "__main__":
    main()

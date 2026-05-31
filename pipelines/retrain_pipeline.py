"""Automatic retraining pipeline with before/after metric logging.

ПОЧЕМУ: заказчик требует автоматическое дообучение с логированием периода и метрик.
КАК: snapshot metrics → calibration study → run_training → log_retrain_event.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from src.calibration.recalibrator import infer_retraining_window, run_calibration_study, select_calibration_policy
from src.features.assembly import assemble_feature_matrix, prepare_supervised
from src.mlops.auto_retrain import (
    build_retrain_summary,
    extract_before_metrics,
    get_retrain_recommendation,
    log_retrain_event,
)
from src.utils.config import load_config, load_json


def run_retraining(
    config_path: str = "config/model_config.yaml",
    *,
    reason: str = "manual",
    force: bool = False,
    monitoring_config_path: str = "config/monitoring_config.yaml",
) -> dict[str, Any]:
    """ПОЧЕМУ: дообучение должно быть полностью автоматическим при drift или по расписанию."""
    config = load_config(config_path)
    retrain_cfg = config.get("retraining", {})

    if not force and not retrain_cfg.get("enabled", True):
        return {"status": "skipped", "reason": "retraining disabled in config"}

    drift_status: dict[str, Any] = {}
    status_path = config.get("artifacts", {}).get("drift_status", "artifacts/drift_status.json")
    from pathlib import Path

    if Path(status_path).exists():
        drift_status = load_json(status_path)

    recommendation = get_retrain_recommendation(drift_status, config)
    if not force and not recommendation.auto_retrain and reason == "auto":
        return {
            "status": "skipped",
            "reason": "no retrain trigger",
            "recommendation": recommendation.to_dict(),
        }

    before = extract_before_metrics(config)

    from src.data.loader import add_active_flag, load_raw_data, validate_date_index

    data_cfg = config.get("data", {})
    df = load_raw_data(data_cfg.get("path", "data.csv"), data_cfg.get("date_column", "Date"))
    df = validate_date_index(df, data_cfg.get("date_column", "Date"))
    df = add_active_flag(df)
    feature_df = assemble_feature_matrix(df, config)
    _, y, _ = prepare_supervised(feature_df, target_col=data_cfg.get("target_column", "Balance"), active_only=True)

    cal_output = config.get("artifacts", {}).get("calibration_policy", "artifacts/calibration_policy.json")
    run_calibration_study(y, config, output_path=cal_output)
    policy = select_calibration_policy(y, config)
    window, window_reason = infer_retraining_window(len(y), config, policy)

    from pipelines.train_pipeline import run_training

    effective_reason = reason if force else (recommendation.reason or reason)
    summary = run_training(config_path)
    after = summary

    event = build_retrain_summary(effective_reason, before, after, window_reason)
    event["training_window"] = window
    log_dir = retrain_cfg.get("log_dir", "retraining")
    event_path = log_retrain_event(event, log_dir)

    return {
        "status": "completed",
        "reason": effective_reason,
        "event_path": str(event_path),
        "event": event,
        "recommendation": recommendation.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain liquidity forecasting models")
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--monitoring", default="config/monitoring_config.yaml")
    parser.add_argument("--force", action="store_true", help="Force retrain regardless of triggers")
    parser.add_argument("--reason", default="manual", help="Retrain reason label")
    args = parser.parse_args()

    result = run_retraining(
        args.config,
        reason=args.reason,
        force=args.force,
        monitoring_config_path=args.monitoring,
    )
    print(f"Retrain status: {result.get('status')}")
    if result.get("status") == "completed":
        event = result.get("event", {})
        before_mae = event.get("metrics_before", {}).get("best_balance_mae")
        after_mae = event.get("metrics_after", {}).get("best_balance_mae")
        print(f"Best Balance MAE: before={before_mae} after={after_mae} improved={event.get('improved')}")
    elif result.get("status") == "skipped":
        print(result.get("reason", ""), file=sys.stderr)


if __name__ == "__main__":
    main()

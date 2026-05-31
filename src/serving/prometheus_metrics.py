"""Prometheus metrics for the serving layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prometheus_client import Counter, Gauge

MODEL_BALANCE_MAE = Gauge("model_balance_mae", "Best model Balance MAE on hold-out")
MODEL_BALANCE_MAE_BY_NAME = Gauge(
    "model_balance_mae_by_name",
    "Balance MAE per model on hold-out",
    ["model"],
)
FEATURE_SELECTION_KUNCHEVA = Gauge(
    "feature_selection_kuncheva",
    "Kuncheva stability index per FS method",
    ["method"],
)
DRIFT_ALARM_ACTIVE = Gauge("drift_alarm_active", "1 when drift alarm is active")
CALIBRATION_CADENCE_DAYS = Gauge("calibration_cadence_days", "Recommended recalibration cadence in days")
RETRAIN_EVENTS_COUNT = Gauge("retrain_events_total", "Number of logged retrain events")
MODEL_LAST_TRAIN_TIMESTAMP = Gauge("model_last_train_timestamp", "Unix timestamp of last training run")
MODEL_DRIFT_ALERTS_TOTAL = Counter("model_drift_alerts_total", "Drift detector alarm count")


def _balance_mae(metrics: dict[str, Any]) -> float:
    if isinstance(metrics, dict) and "Balance" in metrics:
        return float(metrics["Balance"].get("mae", float("inf")))
    if isinstance(metrics, dict):
        return float(metrics.get("mae", float("inf")))
    return float("inf")


def update_from_eval_summary(eval_summary: dict) -> None:
    model_metrics = eval_summary.get("model_metrics", {})
    best_mae = float("inf")
    for name, metrics in model_metrics.items():
        mae = _balance_mae(metrics)
        if mae < float("inf"):
            MODEL_BALANCE_MAE_BY_NAME.labels(model=name).set(mae)
        best_mae = min(best_mae, mae)
    if best_mae < float("inf"):
        MODEL_BALANCE_MAE.set(best_mae)


def update_from_train_summary(metrics_summary: dict) -> None:
    ranking = metrics_summary.get("feature_selection_ranking", [])
    for row in ranking:
        method = row.get("method")
        kuncheva = row.get("kuncheva")
        if method is not None and kuncheva is not None:
            FEATURE_SELECTION_KUNCHEVA.labels(method=str(method)).set(float(kuncheva))

    policy = metrics_summary.get("calibration_policy", {})
    cadence = policy.get("recommended_cadence_days")
    if cadence is not None:
        CALIBRATION_CADENCE_DAYS.set(float(cadence))

    drift_baseline = metrics_summary.get("drift_baseline", {})
    created_at = drift_baseline.get("created_at")
    if created_at:
        try:
            from datetime import datetime

            ts = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
            MODEL_LAST_TRAIN_TIMESTAMP.set(ts.timestamp())
        except ValueError:
            pass


def update_from_drift_status(drift_status: dict) -> None:
    active = 1.0 if drift_status.get("status") == "alarm" else 0.0
    alert = drift_status.get("alert", {})
    if alert.get("detected"):
        active = 1.0
    DRIFT_ALARM_ACTIVE.set(active)


def update_from_retraining(retraining_dir: Path) -> None:
    if not retraining_dir.exists():
        RETRAIN_EVENTS_COUNT.set(0)
        return
    count = len(list(retraining_dir.glob("retrain_*.json")))
    RETRAIN_EVENTS_COUNT.set(count)


def update_all_from_artifacts(config: dict[str, Any]) -> None:
    artifacts = config.get("artifacts", {})
    root = Path(artifacts.get("dir", "artifacts"))

    eval_path = root / "eval_metrics.json"
    if eval_path.exists():
        from src.utils.config import load_json

        update_from_eval_summary(load_json(eval_path))

    metrics_path = Path(artifacts.get("metrics", "artifacts/metrics.json"))
    if metrics_path.exists():
        from src.utils.config import load_json

        update_from_train_summary(load_json(metrics_path))

    drift_path = Path(artifacts.get("drift_status", "artifacts/drift_status.json"))
    if drift_path.exists():
        from src.utils.config import load_json

        update_from_drift_status(load_json(drift_path))

    retrain_dir = Path(config.get("retraining", {}).get("log_dir", "retraining"))
    update_from_retraining(retrain_dir)


def record_drift_alarm() -> None:
    MODEL_DRIFT_ALERTS_TOTAL.inc()
    DRIFT_ALARM_ACTIVE.set(1.0)

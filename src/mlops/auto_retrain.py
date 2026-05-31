"""Auto-retraining trigger and execution helpers.

ПОЧЕМУ: дообучение должно запускаться без ручного вмешательства при drift
        или по cadence из calibration policy.
КАК: check_drift_trigger / check_scheduled_retrain → RetrainRecommendation;
     retrain_pipeline выполняет train и пишет метрики до/после в retraining/.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import load_json, save_json


@dataclass
class RetrainRecommendation:
    auto_retrain: bool
    reason: str | None
    recommended_actions: list[str]
    training_window: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_drift_trigger(drift_status: dict[str, Any]) -> bool:
    """Return True when drift status indicates an alarm."""
    if drift_status.get("status") == "alarm":
        return True
    alert = drift_status.get("alert", {})
    return bool(alert.get("detected", False))


def check_scheduled_retrain(
    config: dict[str, Any],
    last_retrain_path: str | Path = "retraining/last_retrain.json",
) -> bool:
    """ПОЧЕМУ: плановое дообучение по cadence из calibration policy."""
    retrain_cfg = config.get("retraining", {})
    if not retrain_cfg.get("enabled", True):
        return False

    path = Path(last_retrain_path)
    if not path.exists():
        return True

    last = load_json(path)
    last_at = last.get("completed_at")
    if not last_at:
        return True

    cadence_days = retrain_cfg.get("cadence_days")
    if cadence_days is None:
        cal_path = Path(config.get("artifacts", {}).get("calibration_policy", "artifacts/calibration_policy.json"))
        if cal_path.exists():
            cal = load_json(cal_path)
            cadence_days = cal.get("policy", {}).get("recommended_cadence_days", 20)
        else:
            cadence_days = 20

    last_dt = datetime.fromisoformat(last_at.replace("Z", "+00:00"))
    elapsed = (datetime.now(timezone.utc) - last_dt).days
    return elapsed >= cadence_days


def get_retrain_recommendation(
    drift_status: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> RetrainRecommendation:
    """Return retrain recommendation; auto_retrain=True when config allows."""
    retrain_cfg = (config or {}).get("retraining", {})
    auto_enabled = retrain_cfg.get("auto_retrain", True)

    if check_drift_trigger(drift_status):
        alert = drift_status.get("alert", {})
        return RetrainRecommendation(
            auto_retrain=auto_enabled and retrain_cfg.get("retrain_on_drift", True),
            reason="drift_detected",
            recommended_actions=list(alert.get("recommended_actions", ["manual_control", "unplanned_retrain"])),
        )

    if config and check_scheduled_retrain(config):
        return RetrainRecommendation(
            auto_retrain=auto_enabled and retrain_cfg.get("retrain_on_schedule", True),
            reason="scheduled_cadence",
            recommended_actions=["scheduled_retrain"],
        )

    return RetrainRecommendation(
        auto_retrain=False,
        reason=None,
        recommended_actions=[],
    )


def extract_before_metrics(config: dict[str, Any]) -> dict[str, Any]:
    """Snapshot current metrics before retraining."""
    metrics_path = Path(config.get("artifacts", {}).get("metrics", "artifacts/metrics.json"))
    if metrics_path.exists():
        return load_json(metrics_path)
    return {}


def _best_balance_mae(model_metrics: dict[str, Any]) -> float | None:
    best: float | None = None
    for metrics in model_metrics.values():
        if isinstance(metrics, dict) and "Balance" in metrics:
            mae = metrics["Balance"].get("mae")
        elif isinstance(metrics, dict) and "mae" in metrics:
            mae = metrics.get("mae")
        else:
            continue
        if mae is not None and (best is None or mae < best):
            best = float(mae)
    return best


def log_retrain_event(
    event: dict[str, Any],
    log_dir: str | Path = "retraining",
) -> Path:
    """КАК: каждое дообучение пишется в retraining/ с метриками до/после."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    event_path = log_path / f"retrain_{ts}.json"
    save_json(event, event_path)

    save_json(
        {
            "completed_at": event.get("completed_at"),
            "reason": event.get("reason"),
            "best_balance_mae_before": event.get("metrics_before", {}).get("best_balance_mae"),
            "best_balance_mae_after": event.get("metrics_after", {}).get("best_balance_mae"),
        },
        log_path / "last_retrain.json",
    )
    return event_path


def build_retrain_summary(
    reason: str,
    before: dict[str, Any],
    after: dict[str, Any],
    window_reason: str,
) -> dict[str, Any]:
    before_metrics = before.get("model_metrics", {})
    after_metrics = after.get("model_metrics", {})
    return {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "training_window_reason": window_reason,
        "metrics_before": {
            "best_balance_mae": _best_balance_mae(before_metrics),
            "n_train": before.get("n_train"),
            "model_count": len(before_metrics),
        },
        "metrics_after": {
            "best_balance_mae": _best_balance_mae(after_metrics),
            "n_train": after.get("n_train"),
            "model_count": len(after_metrics),
        },
        "improved": (
            _best_balance_mae(after_metrics) is not None
            and _best_balance_mae(before_metrics) is not None
            and _best_balance_mae(after_metrics) <= _best_balance_mae(before_metrics)
        ),
    }

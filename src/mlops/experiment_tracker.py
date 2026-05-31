"""MLflow experiment tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.config import load_config


def log_training_run(
    summary: dict[str, Any],
    config_path: str = "config/model_config.yaml",
    mlflow_config_path: str = "config/mlflow_config.yaml",
) -> str | None:
    """Log training summary to MLflow. Returns run_id or None if logging fails."""
    try:
        import mlflow
    except ImportError:
        return None

    mlflow_cfg = load_config(mlflow_config_path)
    model_cfg = load_config(config_path)
    tracking_uri = mlflow_cfg.get("tracking_uri", "file:./mlflow")
    experiment_name = mlflow_cfg.get("experiment_name", "liquidity_forecast")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"train_{summary.get('training_mode', 'unknown')}") as run:
        mlflow.log_param("training_mode", summary.get("training_mode"))
        mlflow.log_param("n_train", summary.get("n_train"))
        mlflow.log_param("n_test", summary.get("n_test"))

        for model_name, metrics in summary.get("model_metrics", {}).items():
            if isinstance(metrics, dict) and "Balance" in metrics:
                mlflow.log_metric(f"{model_name}_balance_mae", metrics["Balance"].get("mae", 0.0))
            elif isinstance(metrics, dict) and "mae" in metrics:
                mlflow.log_metric(f"{model_name}_balance_mae", metrics.get("mae", 0.0))

        metrics_path = model_cfg.get("artifacts", {}).get("metrics", "artifacts/metrics.json")
        if Path(metrics_path).exists():
            mlflow.log_artifact(metrics_path)
        return run.info.run_id

"""Model artifact loading for serving."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from src.drift.inference import pick_best_model_name
from src.utils.config import load_config, load_json, load_model


@lru_cache(maxsize=1)
def _cached_config(config_path: str) -> dict[str, Any]:
    return load_config(config_path)


def load_best_model(config_path: str = "config/model_config.yaml") -> tuple[str, Any]:
    """Load the best model artifact by hold-out Balance MAE."""
    config = _cached_config(config_path)
    artifacts = config.get("artifacts", {})
    metrics_path = Path(artifacts.get("metrics", "artifacts/metrics.json"))
    metrics_summary = load_json(metrics_path) if metrics_path.exists() else {}
    model_name = pick_best_model_name(metrics_summary.get("model_metrics", {}))
    if model_name is None:
        model_name = "dual_target_naive"
    models_dir = Path(artifacts.get("models_dir", "artifacts/models"))
    model_path = models_dir / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return model_name, load_model(model_path)


def clear_model_cache() -> None:
    _cached_config.cache_clear()

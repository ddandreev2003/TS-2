"""Model quality gate tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.drift.inference import pick_best_model_name
from src.metrics.business_metrics import passes_quality_gate
from src.utils.config import load_config, load_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "model_config.yaml"
EVAL_METRICS = ROOT / "artifacts" / "eval_metrics.json"


@pytest.fixture(scope="module")
def eval_summary() -> dict:
    if not EVAL_METRICS.exists():
        from pipelines.eval_pipeline import run_evaluation

        run_evaluation(str(CONFIG))
    return load_json(EVAL_METRICS)


def test_best_dual_target_passes_quality_gate(eval_summary: dict) -> None:
    config = load_config(CONFIG)
    threshold = config.get("metrics", {}).get("balance_mae_threshold", 0.42)
    model_metrics = eval_summary.get("model_metrics", {})
    best_name = pick_best_model_name(model_metrics)
    assert best_name is not None, "No models in eval metrics"

    metrics = model_metrics[best_name]
    if "Balance" in metrics:
        balance_metrics = metrics["Balance"]
    else:
        balance_metrics = metrics

    mae = balance_metrics.get("mae")
    assert mae is not None, f"No MAE for best model {best_name}"
    assert passes_quality_gate(balance_metrics, threshold), (
        f"Best model {best_name} failed quality gate: Balance MAE={mae:.4f} > {threshold}"
    )

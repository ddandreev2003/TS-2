"""End-to-end pipeline integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "model_config.yaml"
DATA = ROOT / "data.csv"
METRICS = ROOT / "artifacts" / "metrics.json"
EVAL_METRICS = ROOT / "artifacts" / "eval_metrics.json"


@pytest.mark.slow
@pytest.mark.skipif(not DATA.exists(), reason="data.csv missing")
def test_train_eval_predict_flow() -> None:
    from pipelines.eval_pipeline import run_evaluation
    from pipelines.predict_pipeline import run_prediction
    from pipelines.train_pipeline import run_training

    if not METRICS.exists():
        run_training(str(CONFIG))

    ranking = run_evaluation(str(CONFIG))
    assert not ranking.empty

    assert EVAL_METRICS.exists()
    result = run_prediction(str(CONFIG))
    assert "Income" in result
    assert "Outcome" in result
    assert "Balance" in result

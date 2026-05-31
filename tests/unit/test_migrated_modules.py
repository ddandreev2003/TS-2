"""Smoke tests for migrated notebook modules."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import add_active_flag, load_raw_data, validate_date_index
from src.features.assembly import assemble_feature_matrix, prepare_supervised
from src.metrics.business_metrics import compute_balance_metrics, passes_quality_gate
from src.models.baseline import predict_naive, train_naive_baseline
from src.selection.stability import jaccard, stability


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data.csv"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    df = load_raw_data(DATA_PATH)
    df = validate_date_index(df)
    return add_active_flag(df)


def test_load_data(sample_df: pd.DataFrame) -> None:
    assert len(sample_df) > 100
    assert "Balance" in sample_df.columns
    assert "is_active" in sample_df.columns


def test_feature_assembly(sample_df: pd.DataFrame) -> None:
    from src.utils.config import load_config

    config = load_config(ROOT / "config" / "model_config.yaml")
    features = assemble_feature_matrix(sample_df, config)
    X, y, names = prepare_supervised(features, target_col="Balance")
    assert len(X) > 50
    assert len(names) > 10
    assert len(y) == len(X)


def test_baseline_and_metrics() -> None:
    y_train = pd.Series([0.1, 0.2, -0.1, 0.3])
    model = train_naive_baseline(y_train)
    pred = predict_naive(model, 2)
    metrics = compute_balance_metrics(y_train.iloc[-2:], pred)
    assert "mae" in metrics
    assert passes_quality_gate({"mae": 0.1}, threshold=0.42)


def test_stability_metrics() -> None:
    sets = [{"a", "b", "c"}, {"a", "b", "d"}, {"a", "b", "c"}]
    result = stability(sets, p=10)
    assert 0.0 <= result["jaccard"] <= 1.0
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0

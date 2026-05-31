"""Smoke tests for visualization report generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.viz.drift import plot_residuals
from src.viz.eda import plot_balance_timeseries
from src.viz.features import plot_stability_bars
from src.viz.models import plot_all_forecasts
from src.viz.report import generate_all_plots


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    balance = np.cumsum(rng.normal(0, 0.1, len(dates)))
    return pd.DataFrame(
        {
            "Date": dates,
            "Income": np.abs(balance) + rng.normal(0.1, 0.05, len(dates)),
            "Outcome": np.abs(balance) * 0.8 + rng.normal(0.05, 0.03, len(dates)),
            "Balance": balance,
            "is_active": 1,
        }
    )


def test_eda_plot_smoke(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    path = plot_balance_timeseries(sample_df, tmp_path)
    assert Path(path).exists()


def test_fs_plot_smoke(tmp_path: Path) -> None:
    ranking = [
        {"method": "lasso", "jaccard": 0.9, "dice": 0.95, "kuncheva": 0.92, "mae_mean": 0.03},
        {"method": "spearman", "jaccard": 0.6, "dice": 0.7, "kuncheva": 0.68, "mae_mean": 0.028},
    ]
    path = plot_stability_bars(ranking, tmp_path)
    assert Path(path).exists()


def test_forecast_plot_smoke(tmp_path: Path) -> None:
    y_true = np.linspace(-0.2, 0.3, 20)
    preds = {
        "model_a": {"y_pred": y_true + 0.01},
        "model_b": {"y_pred": y_true - 0.02},
    }
    path = plot_all_forecasts(y_true, preds, tmp_path, max_models=2)
    assert Path(path).exists()


def test_drift_plot_smoke(tmp_path: Path) -> None:
    records = [
        {"date": "2020-01-01", "y_true": 0.1, "y_pred": 0.08, "residual": 0.02},
        {"date": "2020-01-02", "y_true": -0.1, "y_pred": -0.05, "residual": -0.05},
    ]
    path = plot_residuals(records, tmp_path)
    assert Path(path).exists()


def test_generate_all_plots_with_existing_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[2]
    if not (root / "artifacts" / "metrics.json").exists():
        pytest.skip("trained artifacts required")
    config = {
        "data": {"path": str(root / "data.csv"), "date_column": "Date", "target_column": "Balance"},
        "artifacts": {
            "dir": str(root / "artifacts"),
            "metrics": str(root / "artifacts" / "metrics.json"),
            "eval_metrics": str(root / "artifacts" / "eval_metrics.json"),
            "holdout_predictions": str(root / "artifacts" / "holdout_predictions.json"),
            "report_manifest": str(tmp_path / "report_manifest.json"),
        },
    }
    manifest = generate_all_plots(config)
    assert manifest.get("n_plots", 0) >= 1

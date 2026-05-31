"""Smoke tests for Streamlit dashboard data loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not (ROOT / "artifacts" / "metrics.json").exists(), reason="artifacts required")
def test_dashboard_loaders_with_artifacts() -> None:
    from src.dashboard import data_loaders as dl

    metrics = dl.load_metrics()
    assert metrics
    manifest = dl.load_manifest()
    ranking = dl.model_ranking_df()
    assert ranking is not None


def test_dashboard_loaders_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.dashboard import data_loaders as dl

    monkeypatch.setattr(dl, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(dl, "PLOTS", tmp_path / "plots")
    assert dl.load_metrics() == {}
    assert dl.model_ranking_df().empty

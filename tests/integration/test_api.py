"""API integration tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.serving.api import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload
    assert "training_mode" in payload


def test_metrics_endpoint() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "model_balance_mae" in response.text or response.text.startswith("#")


def test_predict_endpoint_with_or_without_model() -> None:
    response = client.post("/predict")
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        payload = response.json()
        assert "Income" in payload
        assert "Outcome" in payload
        assert "Balance" in payload

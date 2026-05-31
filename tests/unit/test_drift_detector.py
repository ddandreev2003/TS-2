"""Unit tests for drift detection (CUSUM / Shiryayev-Roberts)."""

from __future__ import annotations

import numpy as np
import pytest

from src.drift.detector import DriftBaseline, DriftDetector
from src.drift.statistics import cusum_two_sided, shiryayev_roberts


def _monitoring_config(method: str = "cusum") -> dict:
    return {
        "drift": {
            "enabled": True,
            "method": method,
            "min_observations": 30,
            "cusum": {"k": 0.5, "h": 4.0},
            "shiryayev_roberts": {"alpha": 0.01, "delta": 1.0, "threshold": 20.0},
            "actions": {"on_alarm": ["manual_control", "unplanned_retrain"]},
        }
    }


def test_cusum_no_drift() -> None:
    rng = np.random.default_rng(42)
    z = rng.normal(0, 1, 200)
    result = cusum_two_sided(z, k=0.5, h=5.0)
    assert result.alarm is False


def test_cusum_detects_mean_shift() -> None:
    z = np.concatenate([np.zeros(150), np.full(50, 4.0)])
    result = cusum_two_sided(z, k=0.5, h=4.0)
    assert result.alarm is True
    assert result.alarm_index is not None
    assert result.alarm_index >= 150


def test_shiryayev_roberts_detects_mean_shift() -> None:
    z = np.concatenate([np.zeros(150), np.full(50, 4.0)])
    result = shiryayev_roberts(z, delta=1.0, threshold=20.0)
    assert result.alarm is True
    assert result.alarm_index is not None
    assert result.alarm_index >= 150


def test_detector_no_drift_on_holdout_like_residuals() -> None:
    rng = np.random.default_rng(7)
    residuals = rng.normal(0.0, 0.05, 80)
    baseline = DriftDetector.fit_baseline(residuals, model_name="test_model")
    cfg = _monitoring_config("cusum")
    cfg["drift"]["cusum"]["h"] = 8.0
    detector = DriftDetector(cfg)
    detector.set_baseline(baseline)
    alert = detector.detect(residuals)
    assert alert.detected is False


def test_detector_alarm_on_shifted_residuals() -> None:
    rng = np.random.default_rng(8)
    in_control = rng.normal(0.0, 0.05, 60)
    baseline = DriftDetector.fit_baseline(in_control, model_name="test_model")
    shifted = rng.normal(0.5, 0.05, 40)
    residuals = np.concatenate([in_control, shifted])
    detector = DriftDetector(_monitoring_config("cusum"))
    detector.set_baseline(baseline)
    alert = detector.detect(residuals)
    assert alert.detected is True
    assert "manual_control" in alert.recommended_actions


def test_detector_shiryayev_roberts_method() -> None:
    rng = np.random.default_rng(9)
    in_control = rng.normal(0.0, 0.05, 60)
    baseline = DriftDetector.fit_baseline(in_control, model_name="test_model")
    shifted = rng.normal(0.5, 0.05, 40)
    residuals = np.concatenate([in_control, shifted])
    detector = DriftDetector(_monitoring_config("shiryayev_roberts"))
    detector.set_baseline(baseline)
    alert = detector.detect(residuals)
    assert alert.detected is True
    assert alert.method == "shiryayev_roberts"


def test_baseline_handles_zero_sigma() -> None:
    baseline = DriftDetector.fit_baseline(np.zeros(10), model_name="flat")
    assert baseline.sigma > 0
    detector = DriftDetector(_monitoring_config())
    detector.set_baseline(baseline)
    alert = detector.detect(np.zeros(40))
    assert alert.n_observations == 40


def test_insufficient_observations() -> None:
    baseline = DriftDetector.fit_baseline(np.array([0.1, -0.1, 0.2]), model_name="test")
    detector = DriftDetector(_monitoring_config())
    detector.set_baseline(baseline)
    alert = detector.detect(np.array([0.1, -0.1]))
    assert alert.detected is False
    assert "Insufficient" in alert.message

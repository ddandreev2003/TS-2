"""Tests for compliance modules: calibration, auto-retrain, business metrics, FS winner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.calibration.recalibrator import (
    infer_retraining_window,
    select_calibration_policy,
    sweep_calibration_cadence,
)
from src.metrics.business_metrics import compute_asymmetric_cost, rank_models_by_business_score
from src.mlops.auto_retrain import (
    build_retrain_summary,
    check_drift_trigger,
    get_retrain_recommendation,
)
from src.selection.feature_selector import FeatureSelectionResult, select_best_fs_method

ROOT = Path(__file__).resolve().parents[2]


def test_asymmetric_cost_penalizes_sign_mismatch() -> None:
    y_true = np.array([1.0, -1.0, 0.5, -0.5])
    y_pred_match = np.array([0.9, -0.9, 0.4, -0.4])
    y_pred_mismatch = np.array([-0.9, 0.9, -0.4, 0.4])
    assert compute_asymmetric_cost(y_true, y_pred_match) < compute_asymmetric_cost(y_true, y_pred_mismatch)


def test_rank_models_by_business_score() -> None:
    metrics = {
        "model_a": {"Balance": {"asymmetric_cost": 0.1, "mae": 0.2}},
        "model_b": {"Balance": {"asymmetric_cost": 0.05, "mae": 0.3}},
    }
    ranked = rank_models_by_business_score(metrics)
    assert ranked[0][0] == "model_b"


def test_calibration_cadence_sweep() -> None:
    rng = np.random.default_rng(42)
    y = pd.Series(np.cumsum(rng.normal(0, 0.1, 200)))
    results = sweep_calibration_cadence(y, cadence_days=[5, 20], min_train_size=40, eval_horizon=5)
    assert len(results) >= 1
    assert all(r.cadence_days in (5, 20) for r in results)


def test_select_calibration_policy() -> None:
    rng = np.random.default_rng(0)
    y = pd.Series(np.cumsum(rng.normal(0, 0.1, 150)))
    config = {
        "calibration": {
            "cadence_candidates": [5, 10, 20],
            "min_training_window": 40,
            "max_mae_degradation": 0.5,
            "eval_horizon_days": 5,
            "default_cadence_days": 10,
        }
    }
    policy = select_calibration_policy(y, config)
    assert policy.recommended_cadence_days in (5, 10, 20)
    assert policy.rationale


def test_infer_retraining_window() -> None:
    config = {"split": {"holdout_ratio": 0.8}, "calibration": {"min_training_window": 60, "max_training_window": 300}}
    window, reason = infer_retraining_window(400, config)
    assert 60 <= window <= 300
    assert reason


def test_drift_trigger_recommendation() -> None:
    drift_status = {"status": "alarm", "alert": {"detected": True, "recommended_actions": ["manual_control"]}}
    assert check_drift_trigger(drift_status)
    config = {"retraining": {"auto_retrain": True, "retrain_on_drift": True}}
    rec = get_retrain_recommendation(drift_status, config)
    assert rec.auto_retrain is True
    assert rec.reason == "drift_detected"


def test_build_retrain_summary() -> None:
    before = {"model_metrics": {"m1": {"Balance": {"mae": 0.5}}}, "n_train": 100}
    after = {"model_metrics": {"m1": {"Balance": {"mae": 0.3}}}, "n_train": 110}
    summary = build_retrain_summary("drift_detected", before, after, "window=110")
    assert summary["improved"] is True
    assert summary["metrics_after"]["best_balance_mae"] == 0.3


def test_select_best_fs_method() -> None:
    ranking = pd.DataFrame(
        [
            {"method": "lasso", "loss_gamma_0.5": 0.05},
            {"method": "spearman", "loss_gamma_0.5": 0.16},
        ]
    )
    result = FeatureSelectionResult(method_ranking=ranking, consensus_sets={"lasso": ["a"], "spearman": ["b"]})
    method = select_best_fs_method(result, {"selection": {"ranking_gamma": 0.5}})
    assert method == "lasso"

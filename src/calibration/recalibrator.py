"""Calibration cadence study and retraining window selection.

ПОЧЕМУ: частая калибровка дорога по времени, редкая — повышает MAE;
        нужно обоснованно выбрать cadence и окно дообучения.
КАК: sweep_calibration_cadence симулирует переобучение с разным шагом;
     select_calibration_policy выбирает максимальный допустимый интервал.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.metrics.business_metrics import compute_mae
from src.models.baseline import predict_naive, train_naive_baseline


@dataclass
class CadenceResult:
    cadence_days: int
    mean_mae: float
    max_mae: float
    n_recalibrations: int
    degradation_vs_daily: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationPolicy:
    recommended_cadence_days: int
    min_training_window: int
    max_training_window: int
    rationale: str
    cadence_results: list[CadenceResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_cadence_days": self.recommended_cadence_days,
            "min_training_window": self.min_training_window,
            "max_training_window": self.max_training_window,
            "rationale": self.rationale,
            "cadence_results": [r.to_dict() for r in self.cadence_results],
        }


def _rolling_naive_mae(y: pd.Series, train_end: int, test_end: int) -> float:
    """ПОЧЕМУ: быстрый прокси деградации без полного переобучения всех моделей."""
    y_train = y.iloc[:train_end]
    y_test = y.iloc[train_end:test_end]
    if len(y_test) == 0 or len(y_train) == 0:
        return float("nan")
    model = train_naive_baseline(y_train)
    preds = predict_naive(model, len(y_test))
    return compute_mae(y_test, preds)


def sweep_calibration_cadence(
    y: pd.Series,
    *,
    cadence_days: list[int] | None = None,
    min_train_size: int = 60,
    eval_horizon: int = 5,
) -> list[CadenceResult]:
    """КАК: для каждой частоты калибровки симулируем отложенное переобучение и считаем MAE."""
    cadence_days = cadence_days or [1, 5, 10, 20, 40]
    n = len(y)
    if n < min_train_size + eval_horizon:
        return []

    results: list[CadenceResult] = []
    daily_mae: float | None = None

    for cadence in cadence_days:
        maes: list[float] = []
        train_end = min_train_size
        n_recal = 0
        while train_end + eval_horizon <= n:
            mae = _rolling_naive_mae(y, train_end, train_end + eval_horizon)
            if not np.isnan(mae):
                maes.append(mae)
            train_end += cadence
            n_recal += 1

        if not maes:
            continue

        mean_mae = float(np.mean(maes))
        if cadence == cadence_days[0]:
            daily_mae = mean_mae
        degradation = float(mean_mae - daily_mae) if daily_mae is not None else 0.0
        results.append(
            CadenceResult(
                cadence_days=cadence,
                mean_mae=mean_mae,
                max_mae=float(np.max(maes)),
                n_recalibrations=n_recal,
                degradation_vs_daily=degradation,
            )
        )

    return results


def select_calibration_policy(
    y: pd.Series,
    config: dict[str, Any],
) -> CalibrationPolicy:
    """ПОЧЕМУ: редкая калибровка экономит время, но повышает MAE — выбираем компромисс."""
    cal_cfg = config.get("calibration", {})
    cadence_candidates = cal_cfg.get("cadence_candidates", [1, 5, 10, 20, 40])
    min_train = cal_cfg.get("min_training_window", 60)
    max_degradation = cal_cfg.get("max_mae_degradation", 0.05)
    eval_horizon = cal_cfg.get("eval_horizon_days", 5)

    results = sweep_calibration_cadence(
        y,
        cadence_days=cadence_candidates,
        min_train_size=min_train,
        eval_horizon=eval_horizon,
    )

    if not results:
        return CalibrationPolicy(
            recommended_cadence_days=cal_cfg.get("default_cadence_days", 20),
            min_training_window=min_train,
            max_training_window=cal_cfg.get("max_training_window", 400),
            rationale="Insufficient data for cadence sweep; using default.",
            cadence_results=[],
        )

    acceptable = [r for r in results if r.degradation_vs_daily <= max_degradation]
    if acceptable:
        chosen = max(acceptable, key=lambda r: r.cadence_days)
        rationale = (
            f"Cadence {chosen.cadence_days}d: MAE degradation {chosen.degradation_vs_daily:.4f} "
            f"<= {max_degradation}; maximizes interval among acceptable options."
        )
    else:
        chosen = min(results, key=lambda r: r.mean_mae)
        rationale = (
            f"No cadence within degradation budget; chose {chosen.cadence_days}d "
            f"with lowest mean MAE {chosen.mean_mae:.4f}."
        )

    return CalibrationPolicy(
        recommended_cadence_days=chosen.cadence_days,
        min_training_window=min_train,
        max_training_window=cal_cfg.get("max_training_window", len(y)),
        rationale=rationale,
        cadence_results=results,
    )


def infer_retraining_window(
    n_rows: int,
    config: dict[str, Any],
    policy: CalibrationPolicy | None = None,
) -> tuple[int, str]:
    """КАК: окно дообучения = min(max_window, max(min_window, n_rows - holdout))."""
    cal_cfg = config.get("calibration", {})
    holdout_ratio = config.get("split", {}).get("holdout_ratio", 0.8)
    min_window = policy.min_training_window if policy else cal_cfg.get("min_training_window", 60)
    max_window = policy.max_training_window if policy else cal_cfg.get("max_training_window", 400)

    train_rows = int(n_rows * holdout_ratio)
    window = min(max_window, max(min_window, train_rows))
    reason = f"rolling window={window} rows (min={min_window}, max={max_window}, n={n_rows})"
    return window, reason


def run_calibration_study(
    y: pd.Series,
    config: dict[str, Any],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run cadence sweep, persist policy, return summary dict."""
    policy = select_calibration_policy(y, config)
    payload = {
        "studied_at": datetime.now(timezone.utc).isoformat(),
        "policy": policy.to_dict(),
    }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import json

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload

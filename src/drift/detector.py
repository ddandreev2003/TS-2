"""Drift detection on forecast residuals using CUSUM or Shiryayev-Roberts.

ПОЧЕМУ: при смене режима ликвидности прогнозная ошибка растёт — позиционеру нужен
        сигнал перейти на ручное управление или внеплановое дообучение.
КАК: стандартизируем hold-out остатки (z = (r - mu) / sigma), прогоняем CUSUM или
     Shiryayev-Roberts; при превышении порога h/threshold возвращаем DriftAlert.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.drift.statistics import ControlChartResult, cusum_two_sided, shiryayev_roberts
from src.utils.config import load_json, save_json

SIGMA_EPS = 1e-9


@dataclass
class DriftBaseline:
    mu: float
    sigma: float
    method: str
    n_observations: int
    model_name: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DriftBaseline:
        return cls(**data)


@dataclass
class DriftAlert:
    detected: bool
    method: str
    alarm_index: int | None
    alarm_date: str | None
    statistic_value: float
    recommended_actions: list[str]
    message: str
    n_observations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DriftDetector:
    """Detect drift in standardized forecast residuals."""

    def __init__(self, monitoring_config: dict[str, Any]) -> None:
        self.config = monitoring_config
        self.drift_cfg = monitoring_config.get("drift", {})
        self.baseline: DriftBaseline | None = None

    @staticmethod
    def fit_baseline(residuals: np.ndarray, model_name: str, method: str = "holdout") -> DriftBaseline:
        residuals = np.asarray(residuals, dtype=float)
        sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        if sigma < SIGMA_EPS:
            sigma = SIGMA_EPS
        return DriftBaseline(
            mu=float(np.mean(residuals)),
            sigma=sigma,
            method=method,
            n_observations=len(residuals),
            model_name=model_name,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def set_baseline(self, baseline: DriftBaseline) -> None:
        self.baseline = baseline

    def standardize(self, residuals: np.ndarray) -> np.ndarray:
        if self.baseline is None:
            raise ValueError("Drift baseline is not set")
        return (np.asarray(residuals, dtype=float) - self.baseline.mu) / self.baseline.sigma

    def _run_chart(self, z: np.ndarray) -> tuple[str, ControlChartResult]:
        method = self.drift_cfg.get("method", "cusum")
        if method == "shiryayev_roberts":
            sr_cfg = self.drift_cfg.get("shiryayev_roberts", {})
            result = shiryayev_roberts(
                z,
                delta=float(sr_cfg.get("delta", 1.0)),
                threshold=float(sr_cfg.get("threshold", 50.0)),
                alpha=float(sr_cfg.get("alpha", 0.01)),
            )
            return "shiryayev_roberts", result

        cusum_cfg = self.drift_cfg.get("cusum", {})
        result = cusum_two_sided(
            z,
            k=float(cusum_cfg.get("k", 0.5)),
            h=float(cusum_cfg.get("h", 5.0)),
        )
        return "cusum", result

    def detect(self, residuals: np.ndarray, dates: list[str] | None = None) -> DriftAlert:
        if self.baseline is None:
            raise ValueError("Drift baseline is not set")

        residuals = np.asarray(residuals, dtype=float)
        min_obs = int(self.drift_cfg.get("min_observations", 30))
        if len(residuals) < min_obs:
            return DriftAlert(
                detected=False,
                method=str(self.drift_cfg.get("method", "cusum")),
                alarm_index=None,
                alarm_date=None,
                statistic_value=0.0,
                recommended_actions=[],
                message=f"Insufficient observations for drift detection ({len(residuals)} < {min_obs})",
                n_observations=len(residuals),
            )

        z = self.standardize(residuals)
        method_name, chart = self._run_chart(z)
        actions = list(self.drift_cfg.get("actions", {}).get("on_alarm", ["manual_control", "unplanned_retrain"]))

        alarm_date = None
        if chart.alarm_index is not None and dates is not None and chart.alarm_index < len(dates):
            alarm_date = dates[chart.alarm_index]

        if chart.alarm:
            message = (
                f"Drift detected via {method_name} at index {chart.alarm_index}"
                + (f" (date={alarm_date})" if alarm_date else "")
                + f"; statistic={chart.max_statistic:.4f}"
            )
            return DriftAlert(
                detected=True,
                method=method_name,
                alarm_index=chart.alarm_index,
                alarm_date=alarm_date,
                statistic_value=float(chart.max_statistic),
                recommended_actions=actions,
                message=message,
                n_observations=len(residuals),
            )

        return DriftAlert(
            detected=False,
            method=method_name,
            alarm_index=None,
            alarm_date=None,
            statistic_value=float(chart.max_statistic),
            recommended_actions=[],
            message="No drift detected",
            n_observations=len(residuals),
        )

    @staticmethod
    def save_baseline(baseline: DriftBaseline, path: str | Path) -> None:
        save_json(baseline.to_dict(), path)

    @staticmethod
    def load_baseline(path: str | Path) -> DriftBaseline:
        return DriftBaseline.from_dict(load_json(path))

    def save_state(self, path: str | Path, alert: DriftAlert, chart_stats: np.ndarray | None = None) -> None:
        payload: dict[str, Any] = {
            "status": "alarm" if alert.detected else "ok",
            "alert": alert.to_dict(),
            "baseline": self.baseline.to_dict() if self.baseline else None,
        }
        if chart_stats is not None:
            payload["last_statistics"] = chart_stats.tolist()
        save_json(payload, path)

    @staticmethod
    def load_state(path: str | Path) -> dict[str, Any]:
        return load_json(path)

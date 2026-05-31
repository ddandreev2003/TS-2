"""CUSUM and Shiryayev-Roberts control chart statistics for drift detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ControlChartResult:
    """Result of a sequential control chart run."""

    alarm: bool
    alarm_index: int | None
    max_statistic: float
    statistics: np.ndarray
    upper_statistics: np.ndarray | None = None
    lower_statistics: np.ndarray | None = None


def cusum_two_sided(z: np.ndarray, k: float = 0.5, h: float = 5.0) -> ControlChartResult:
    """Two-sided Page CUSUM on standardized observations."""
    z = np.asarray(z, dtype=float)
    n = len(z)
    if n == 0:
        return ControlChartResult(
            alarm=False,
            alarm_index=None,
            max_statistic=0.0,
            statistics=np.array([]),
            upper_statistics=np.array([]),
            lower_statistics=np.array([]),
        )

    s_plus = np.zeros(n)
    s_minus = np.zeros(n)
    for t in range(n):
        prev_plus = s_plus[t - 1] if t > 0 else 0.0
        prev_minus = s_minus[t - 1] if t > 0 else 0.0
        s_plus[t] = max(0.0, prev_plus + z[t] - k)
        s_minus[t] = max(0.0, prev_minus - z[t] - k)

    combined = np.maximum(s_plus, s_minus)
    alarm_mask = (s_plus > h) | (s_minus > h)
    alarm_index = int(np.argmax(alarm_mask)) if alarm_mask.any() else None
    return ControlChartResult(
        alarm=bool(alarm_mask.any()),
        alarm_index=alarm_index if alarm_mask.any() else None,
        max_statistic=float(np.max(combined)),
        statistics=combined,
        upper_statistics=s_plus,
        lower_statistics=s_minus,
    )


def shiryayev_roberts(
    z: np.ndarray,
    delta: float = 1.0,
    threshold: float = 50.0,
    alpha: float = 0.01,
) -> ControlChartResult:
    """Two-sided Shiryayev-Roberts statistic (Pollak recursion) on standardized observations."""
    del alpha  # reserved for future online reset policy
    z = np.asarray(z, dtype=float)
    n = len(z)
    if n == 0:
        return ControlChartResult(
            alarm=False,
            alarm_index=None,
            max_statistic=0.0,
            statistics=np.array([]),
        )

    r_up = np.zeros(n)
    r_down = np.zeros(n)
    for t in range(n):
        prev_up = r_up[t - 1] if t > 0 else 0.0
        prev_down = r_down[t - 1] if t > 0 else 0.0
        r_up[t] = (1.0 + prev_up) * np.exp(delta * z[t] - 0.5 * delta**2)
        r_down[t] = (1.0 + prev_down) * np.exp(-delta * z[t] - 0.5 * delta**2)

    combined = np.maximum(r_up, r_down)
    alarm_mask = combined > threshold
    alarm_index = int(np.argmax(alarm_mask)) if alarm_mask.any() else None
    return ControlChartResult(
        alarm=bool(alarm_mask.any()),
        alarm_index=alarm_index if alarm_mask.any() else None,
        max_statistic=float(np.max(combined)),
        statistics=combined,
        upper_statistics=r_up,
        lower_statistics=r_down,
    )

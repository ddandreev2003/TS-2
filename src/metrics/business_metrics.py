"""Regression and business metrics for liquidity forecasting."""

from __future__ import annotations

import numpy as np

# ПОЧЕМУ: позиционер несёт разную стоимость ошибки при профиците (Key−0.9%) и дефиците (Key+1%).
# КАК: веса задают относительную «цену» единицы ошибки по знаку фактического сальдо.
DEFAULT_SURPLUS_SPREAD = 0.009  # Key − 0.9% (размещение в ЦБ)
DEFAULT_DEFICIT_SPREAD = 0.010  # Key + 1% (overnight-займ)
DEFAULT_DERIVATIVES_SPREAD = 0.005  # Key + 0.5% (размещение на деривативах)


def compute_mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100)


def compute_directional_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return 0.0
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(true_dir == pred_dir))


def compute_asymmetric_cost(
    y_true,
    y_pred,
    *,
    surplus_spread: float = DEFAULT_SURPLUS_SPREAD,
    deficit_spread: float = DEFAULT_DEFICIT_SPREAD,
    sign_mismatch_multiplier: float = 2.0,
) -> float:
    """Business-weighted forecast error cost (higher weight for deficit-side mistakes)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_pred - y_true)
    sign_mismatch = np.sign(y_true) != np.sign(y_pred)
    weights = np.where(y_true >= 0, surplus_spread, deficit_spread)
    weights = np.where(sign_mismatch, deficit_spread * sign_mismatch_multiplier, weights)
    norm = (surplus_spread + deficit_spread) / 2.0
    return float(np.mean(abs_err * weights / norm))


def compute_regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": compute_mae(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "smape": compute_smape(y_true, y_pred),
        "directional_accuracy": compute_directional_accuracy(y_true, y_pred),
    }


def compute_balance_metrics(y_true, y_pred, threshold: float = 0.42) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_errors = np.abs(y_true - y_pred)
    metrics = compute_regression_metrics(y_true, y_pred)
    metrics["within_threshold_share"] = float(np.mean(abs_errors <= threshold))
    metrics["max_abs_error"] = float(np.max(abs_errors)) if len(abs_errors) else 0.0
    metrics["sign_accuracy"] = float(np.mean(np.sign(y_true) == np.sign(y_pred))) if len(y_true) else 0.0
    metrics["asymmetric_cost"] = compute_asymmetric_cost(y_true, y_pred)
    return metrics


def business_report(y_true, y_pred, threshold: float = 0.42) -> dict[str, float]:
    """Notebook-compatible business metric summary."""
    return compute_balance_metrics(y_true, y_pred, threshold=threshold)


def passes_quality_gate(metrics: dict[str, float], threshold: float = 0.42) -> bool:
    return metrics.get("mae", float("inf")) <= threshold


def rank_models_by_business_score(model_metrics: dict[str, dict]) -> list[tuple[str, float]]:
    """Rank models by asymmetric cost on Balance (lower is better)."""
    scores: list[tuple[str, float]] = []
    for name, metrics in model_metrics.items():
        if isinstance(metrics, dict) and "Balance" in metrics:
            cost = metrics["Balance"].get("asymmetric_cost", metrics["Balance"].get("mae", float("inf")))
        elif isinstance(metrics, dict) and "asymmetric_cost" in metrics:
            cost = metrics["asymmetric_cost"]
        elif isinstance(metrics, dict) and "mae" in metrics:
            cost = metrics["mae"]
        else:
            continue
        scores.append((name, float(cost)))
    return sorted(scores, key=lambda item: item[1])

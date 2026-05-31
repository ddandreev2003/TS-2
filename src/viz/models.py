"""Model comparison and forecast visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics.business_metrics import business_report
from src.viz._style import TARGET_ABS_ERROR, apply_style, save_figure


def _balance_mae(metrics: dict[str, Any]) -> float:
    if "Balance" in metrics:
        return float(metrics["Balance"].get("mae", float("inf")))
    return float(metrics.get("mae", float("inf")))


def _balance_cost(metrics: dict[str, Any]) -> float:
    if "Balance" in metrics:
        return float(metrics["Balance"].get("asymmetric_cost", metrics["Balance"].get("mae", float("inf"))))
    return float(metrics.get("asymmetric_cost", metrics.get("mae", float("inf"))))


def plot_model_mae_bar(eval_summary: dict[str, Any], output_dir: Path) -> str:
    apply_style()
    model_metrics = eval_summary.get("model_metrics", {})
    rows = [(name, _balance_mae(m)) for name, m in model_metrics.items()]
    rows.sort(key=lambda item: item[1])
    if not rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No model metrics", ha="center")
        return save_figure(fig, output_dir / "models_mae_bar.png")

    names, maes = zip(*rows)
    colors = ["#2ecc71" if m <= TARGET_ABS_ERROR else "#e74c3c" for m in maes]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names, maes, color=colors)
    ax.axvline(TARGET_ABS_ERROR, color="black", ls="--", label="threshold 0.42")
    ax.set_xlabel("Balance MAE")
    ax.set_title("Model comparison (hold-out Balance MAE)")
    ax.legend()
    return save_figure(fig, output_dir / "models_mae_bar.png")


def plot_asymmetric_cost_bar(eval_summary: dict[str, Any], output_dir: Path) -> str:
    apply_style()
    model_metrics = eval_summary.get("model_metrics", {})
    rows = [(name, _balance_cost(m)) for name, m in model_metrics.items()]
    rows.sort(key=lambda item: item[1])
    if not rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No model metrics", ha="center")
        return save_figure(fig, output_dir / "models_asymmetric_cost_bar.png")

    names, costs = zip(*rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names, costs, color="#8e44ad")
    ax.set_xlabel("Asymmetric cost")
    ax.set_title("Business-weighted model ranking")
    return save_figure(fig, output_dir / "models_asymmetric_cost_bar.png")


def plot_all_forecasts(
    y_true: np.ndarray | pd.Series,
    predictions_dict: dict[str, dict[str, Any]],
    output_dir: Path,
    target_error: float = TARGET_ABS_ERROR,
    max_models: int = 6,
) -> str:
    """Notebook-style grid of actual vs predicted on test set."""
    apply_style()
    items = list(predictions_dict.items())[:max_models]
    n = len(items)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No hold-out predictions", ha="center")
        return save_figure(fig, output_dir / "models_forecast_grid.png")

    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    y_true = np.asarray(y_true, dtype=float)
    x = np.arange(len(y_true))

    for ax, (name, payload) in zip(axes, items):
        y_pred = np.asarray(payload["y_pred"], dtype=float)
        report = business_report(y_true, y_pred, threshold=target_error)
        ax.plot(x, y_true, label="actual", linewidth=1.5)
        ax.plot(x, y_pred, label="forecast", linewidth=1.2, alpha=0.85)
        ax.fill_between(x, y_true - target_error, y_true + target_error, alpha=0.15, color="green")
        status = "PASS" if report["mae"] <= target_error else "FAIL"
        ax.set_title(f"{name} | MAE={report['mae']:.4f} | {status}")
        ax.legend(loc="upper right", fontsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Hold-out forecasts (Balance)", y=1.01)
    fig.tight_layout()
    return save_figure(fig, output_dir / "models_forecast_grid.png")


def plot_single_forecast(
    model_name: str,
    dates: list[str],
    y_true: list[float],
    y_pred: list[float],
    output_dir: Path,
    target_error: float = TARGET_ABS_ERROR,
) -> str:
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    x = range(len(y_true))
    ax.plot(x, y_true, label="actual", marker="o", markersize=3)
    ax.plot(x, y_pred, label="forecast", marker="x", markersize=3)
    ax.fill_between(x, np.array(y_true) - target_error, np.array(y_true) + target_error, alpha=0.15, color="green")
    report = business_report(y_true, y_pred, threshold=target_error)
    ax.set_title(f"{model_name} — MAE={report['mae']:.4f}")
    ax.set_xlabel("Test index")
    ax.legend()
    safe_name = model_name.replace("/", "_")
    return save_figure(fig, output_dir / f"forecast_{safe_name}.png")


def generate_model_plots(
    eval_summary: dict[str, Any],
    holdout_predictions: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    paths = [
        plot_model_mae_bar(eval_summary, output_dir),
        plot_asymmetric_cost_bar(eval_summary, output_dir),
    ]
    if holdout_predictions.get("models"):
        y_true = holdout_predictions.get("y_true", [])
        preds = {
            name: {"y_pred": payload["y_pred"]}
            for name, payload in holdout_predictions["models"].items()
        }
        paths.append(plot_all_forecasts(y_true, preds, output_dir))
        for name, payload in list(holdout_predictions["models"].items())[:4]:
            paths.append(
                plot_single_forecast(
                    name,
                    holdout_predictions.get("dates", []),
                    holdout_predictions.get("y_true", []),
                    payload["y_pred"],
                    output_dir,
                )
            )
    return paths

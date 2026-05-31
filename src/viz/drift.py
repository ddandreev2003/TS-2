"""Drift and anomaly visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.drift.detector import DriftBaseline
from src.drift.statistics import cusum_two_sided, shiryayev_roberts
from src.viz._style import apply_style, save_figure


def _load_residual_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["date", "residual", "y_true", "y_pred"])
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def plot_residuals(records: list[dict[str, Any]], output_dir: Path) -> str:
    apply_style()
    df = _load_residual_df(records)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    if df.empty:
        axes[0].text(0.5, 0.5, "No residual history", ha="center")
        return save_figure(fig, output_dir / "drift_residuals.png")

    axes[0].plot(df["date"], df["y_true"], label="actual", linewidth=1.2)
    axes[0].plot(df["date"], df["y_pred"], label="forecast", linewidth=1.0, alpha=0.8)
    axes[0].set_title("Actual vs forecast (monitoring window)")
    axes[0].legend()

    axes[1].bar(df["date"], df["residual"], width=1.0, color="#3498db", alpha=0.7)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("Forecast residuals")
    fig.autofmt_xdate()
    fig.tight_layout()
    return save_figure(fig, output_dir / "drift_residuals.png")


def plot_control_chart(
    records: list[dict[str, Any]],
    baseline: dict[str, Any],
    monitoring_config: dict[str, Any],
    output_dir: Path,
) -> str:
    apply_style()
    df = _load_residual_df(records)
    if df.empty or not baseline:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Insufficient data for control chart", ha="center")
        return save_figure(fig, output_dir / "drift_control_chart.png")

    bl = DriftBaseline.from_dict(baseline)
    sigma = max(bl.sigma, 1e-9)
    z = (df["residual"].to_numpy() - bl.mu) / sigma

    drift_cfg = monitoring_config.get("drift", {})
    method = drift_cfg.get("method", "cusum")
    if method == "shiryayev_roberts":
        sr_cfg = drift_cfg.get("shiryayev_roberts", {})
        result = shiryayev_roberts(
            z,
            alpha=float(sr_cfg.get("alpha", 0.01)),
            delta=float(sr_cfg.get("delta", 1.0)),
            threshold=float(sr_cfg.get("threshold", 50.0)),
        )
        threshold = float(sr_cfg.get("threshold", 50.0))
        title = "Shiryayev-Roberts control chart"
    else:
        cusum_cfg = drift_cfg.get("cusum", {})
        result = cusum_two_sided(z, k=float(cusum_cfg.get("k", 0.5)), h=float(cusum_cfg.get("h", 5.0)))
        threshold = float(cusum_cfg.get("h", 5.0))
        title = "CUSUM control chart"

    fig, ax = plt.subplots(figsize=(12, 4))
    stats = result.statistics
    ax.plot(df["date"], stats, label="statistic", linewidth=1.2)
    ax.axhline(threshold, color="red", ls="--", label="threshold")
    if result.alarm and result.alarm_index is not None:
        ax.axvline(df["date"].iloc[result.alarm_index], color="red", alpha=0.5, label="alarm")
    ax.set_title(title)
    ax.legend()
    fig.autofmt_xdate()
    return save_figure(fig, output_dir / "drift_control_chart.png")


def plot_standardized_residuals(
    records: list[dict[str, Any]],
    baseline: dict[str, Any],
    output_dir: Path,
) -> str:
    apply_style()
    df = _load_residual_df(records)
    if df.empty or not baseline:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No standardized residuals", ha="center")
        return save_figure(fig, output_dir / "drift_standardized_residuals.png")

    bl = DriftBaseline.from_dict(baseline)
    sigma = max(bl.sigma, 1e-9)
    z = (df["residual"].to_numpy() - bl.mu) / sigma
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["date"], z, marker="o", markersize=2, linewidth=1)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(2, color="orange", ls="--", alpha=0.7)
    ax.axhline(-2, color="orange", ls="--", alpha=0.7)
    ax.set_title("Standardized forecast residuals (z-scores)")
    fig.autofmt_xdate()
    return save_figure(fig, output_dir / "drift_standardized_residuals.png")


def generate_drift_plots(
    residual_history: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
    monitoring_config: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    baseline = baseline or {}
    return [
        plot_residuals(residual_history, output_dir),
        plot_standardized_residuals(residual_history, baseline, output_dir),
        plot_control_chart(residual_history, baseline, monitoring_config, output_dir),
    ]

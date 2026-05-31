"""EDA plots mirroring the research notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.viz._style import TARGET_ABS_ERROR, apply_style, save_figure


def _date_series(df: pd.DataFrame) -> pd.Series:
    if "Date" in df.columns:
        return pd.to_datetime(df["Date"])
    return pd.to_datetime(df.index)


def plot_balance_timeseries(df: pd.DataFrame, output_dir: Path) -> str:
    apply_style()
    active = df[df.get("is_active", 1) == 1].copy()
    dates = _date_series(active)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, active["Balance"], linewidth=1.2, label="Balance")
    ax.axhline(TARGET_ABS_ERROR, color="green", ls="--", alpha=0.6, label="+0.42")
    ax.axhline(-TARGET_ABS_ERROR, color="green", ls="--", alpha=0.6, label="-0.42")
    ax.set_title("Balance (active days)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right")
    return save_figure(fig, output_dir / "eda_balance_timeseries.png")


def plot_income_outcome(df: pd.DataFrame, output_dir: Path) -> str:
    apply_style()
    active = df[df.get("is_active", 1) == 1].copy()
    dates = _date_series(active)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, active["Income"], label="Income", alpha=0.8)
    ax.plot(dates, active["Outcome"], label="Outcome", alpha=0.8)
    ax.set_title("Income vs Outcome")
    ax.legend()
    return save_figure(fig, output_dir / "eda_income_outcome.png")


def plot_balance_distribution(df: pd.DataFrame, output_dir: Path) -> str:
    apply_style()
    balance = df.loc[df.get("is_active", 1) == 1, "Balance"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(balance, bins=40, edgecolor="white")
    axes[0].axvline(TARGET_ABS_ERROR, color="green", ls="--")
    axes[0].axvline(-TARGET_ABS_ERROR, color="green", ls="--")
    axes[0].set_title("Balance distribution")
    share = (balance.abs() <= TARGET_ABS_ERROR).mean()
    axes[1].bar(["within 0.42", "outside 0.42"], [share, 1 - share], color=["#2ecc71", "#e74c3c"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Share within tolerance")
    fig.tight_layout()
    return save_figure(fig, output_dir / "eda_balance_distribution.png")


def plot_seasonality(df: pd.DataFrame, output_dir: Path) -> str:
    apply_style()
    active = df[df.get("is_active", 1) == 1].copy()
    dates = _date_series(active)
    active = active.assign(dow=dates.dt.dayofweek, month=dates.dt.month)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=active, x="dow", y="Balance", ax=axes[0])
    axes[0].set_title("Balance by day of week")
    sns.boxplot(data=active, x="month", y="Balance", ax=axes[1])
    axes[1].set_title("Balance by month")
    fig.tight_layout()
    return save_figure(fig, output_dir / "eda_seasonality.png")


def plot_acf_pacf(df: pd.DataFrame, output_dir: Path) -> str:
    apply_style()
    balance = df.loc[df.get("is_active", 1) == 1, "Balance"].dropna()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(balance, lags=40, ax=axes[0])
    plot_pacf(balance, lags=40, ax=axes[1], method="ywm")
    axes[0].set_title("ACF — Balance")
    axes[1].set_title("PACF — Balance")
    fig.tight_layout()
    return save_figure(fig, output_dir / "eda_acf_pacf.png")


def plot_year_month_heatmap(df: pd.DataFrame, output_dir: Path) -> str:
    apply_style()
    active = df[df.get("is_active", 1) == 1].copy()
    dates = _date_series(active)
    active = active.assign(year=dates.dt.year, month=dates.dt.month)
    pivot = active.pivot_table(values="Balance", index="year", columns="month", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r", center=0, ax=ax)
    ax.set_title("Mean Balance (year x month)")
    return save_figure(fig, output_dir / "eda_year_month_heatmap.png")


def generate_eda_plots(df: pd.DataFrame, output_dir: Path, config: dict[str, Any] | None = None) -> list[str]:
    _ = config
    paths = [
        plot_balance_timeseries(df, output_dir),
        plot_income_outcome(df, output_dir),
        plot_balance_distribution(df, output_dir),
        plot_seasonality(df, output_dir),
        plot_acf_pacf(df, output_dir),
        plot_year_month_heatmap(df, output_dir),
    ]
    return paths

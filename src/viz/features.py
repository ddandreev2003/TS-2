"""Feature selection visualization plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.viz._style import apply_style, save_figure


def plot_stability_bars(ranking: list[dict[str, Any]], output_dir: Path) -> str:
    apply_style()
    if not ranking:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No feature selection ranking", ha="center")
        return save_figure(fig, output_dir / "fs_stability_bars.png")

    df = pd.DataFrame(ranking)
    methods = df["method"].tolist()
    metrics = ["jaccard", "dice", "kuncheva"]
    x = range(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            ax.bar([xi + i * width for xi in x], df[metric], width=width, label=metric)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_title("Feature selection stability")
    ax.legend()
    return save_figure(fig, output_dir / "fs_stability_bars.png")


def plot_stability_mae_tradeoff(ranking: list[dict[str, Any]], output_dir: Path) -> str:
    apply_style()
    if not ranking:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No FS ranking data", ha="center")
        return save_figure(fig, output_dir / "fs_stability_mae_tradeoff.png")

    df = pd.DataFrame(ranking)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["kuncheva"], df["mae_mean"], s=80)
    for _, row in df.iterrows():
        ax.annotate(row["method"], (row["kuncheva"], row["mae_mean"]), fontsize=9, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Kuncheva stability")
    ax.set_ylabel("Mean probe MAE")
    ax.set_title("Stability vs error trade-off")
    return save_figure(fig, output_dir / "fs_stability_mae_tradeoff.png")


def plot_top_correlations(X: pd.DataFrame, y: pd.Series, output_dir: Path, top_k: int = 10) -> str:
    apply_style()
    corrs = X.corrwith(y).abs().sort_values(ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(8, 5))
    corrs.sort_values().plot(kind="barh", ax=ax, color="#3498db")
    ax.set_title(f"Top {top_k} feature correlations with Balance")
    ax.set_xlabel("|correlation|")
    return save_figure(fig, output_dir / "fs_top_correlations.png")


def generate_feature_plots(
    metrics_summary: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
) -> list[str]:
    ranking = metrics_summary.get("feature_selection_ranking", [])
    return [
        plot_stability_bars(ranking, output_dir),
        plot_stability_mae_tradeoff(ranking, output_dir),
        plot_top_correlations(X, y, output_dir),
    ]

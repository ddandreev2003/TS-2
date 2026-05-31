"""Shared matplotlib styling for report plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TARGET_ABS_ERROR = 0.42


def apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 5),
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "font.size": 10,
        }
    )


def save_figure(fig: plt.Figure, path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)

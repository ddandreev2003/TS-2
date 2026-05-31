"""Residual history persistence for drift monitoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import load_json, save_json


def load_residual_history(path: str | Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    data = load_json(path)
    return data.get("records", [])


def save_residual_history(path: str | Path, records: list[dict[str, Any]]) -> None:
    save_json({"records": records}, path)


def records_to_arrays(
    records: list[dict[str, Any]],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Extract dates and arrays for rows with computed residuals."""
    dates: list[str] = []
    y_true: list[float] = []
    y_pred: list[float] = []
    residuals: list[float] = []
    for row in records:
        if row.get("residual") is None:
            continue
        dates.append(str(row["date"]))
        y_true.append(float(row["y_true"]))
        y_pred.append(float(row["y_pred"]))
        residuals.append(float(row["residual"]))
    return dates, np.asarray(y_true), np.asarray(y_pred), np.asarray(residuals)


def append_pending_prediction(
    path: str | Path,
    date: str,
    y_pred: float,
    model_name: str,
) -> None:
    records = load_residual_history(path)
    if any(r.get("date") == date for r in records):
        return
    records.append(
        {
            "date": date,
            "y_true": None,
            "y_pred": y_pred,
            "residual": None,
            "status": "pending",
            "model": model_name,
        }
    )
    save_residual_history(path, records)


def fill_pending_with_actuals(
    records: list[dict[str, Any]],
    actuals: dict[str, float],
) -> list[dict[str, Any]]:
    """Replace pending rows when actual values become available."""
    updated: list[dict[str, Any]] = []
    for row in records:
        row = dict(row)
        date = str(row.get("date"))
        if row.get("status") == "pending" and date in actuals:
            y_true = float(actuals[date])
            y_pred = float(row["y_pred"])
            row["y_true"] = y_true
            row["residual"] = y_true - y_pred
            row["status"] = "resolved"
        updated.append(row)
    return updated


def build_holdout_records(
    dates: pd.Index | pd.Series,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for i in range(len(y_true)):
        date_val = dates[i]
        if isinstance(date_val, pd.Timestamp):
            date_str = str(date_val.date())
        else:
            date_str = str(date_val)
        yt = float(y_true[i])
        yp = float(y_pred[i])
        records.append(
            {
                "date": date_str,
                "y_true": yt,
                "y_pred": yp,
                "residual": yt - yp,
                "status": "holdout",
                "model": model_name,
            }
        )
    return records

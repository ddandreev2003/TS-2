"""Autoregressive and target-derived features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_autoregressive_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Build lag, rolling, EWM, diff, and ratio features for target columns."""
    feat_cfg = config.get("features", config)
    lags = feat_cfg.get("lags", [1, 2, 3, 5, 10, 20])
    rolling_windows = feat_cfg.get("rolling_windows", [5, 10, 20])
    ewm_spans = feat_cfg.get("ewm_spans", [5, 20])
    diff_lags = feat_cfg.get("diff_lags", [1, 5])
    target_cols = feat_cfg.get("target_columns_for_ar", ["Balance", "Income", "Outcome"])

    out = df[["Date"]].copy()
    work = df.set_index("Date")

    for col in target_cols:
        if col not in work.columns:
            continue
        series = work[col]

        for lag in lags:
            out[f"{col}_lag_{lag}"] = series.shift(lag).values

        for window in rolling_windows:
            rolled = series.shift(1).rolling(window)
            out[f"{col}_roll_mean_{window}"] = rolled.mean().values
            out[f"{col}_roll_std_{window}"] = rolled.std().values
            out[f"{col}_roll_min_{window}"] = rolled.min().values
            out[f"{col}_roll_max_{window}"] = rolled.max().values

        for span in ewm_spans:
            out[f"{col}_ewm_{span}"] = series.shift(1).ewm(span=span, adjust=False).mean().values

        for dlag in diff_lags:
            out[f"{col}_diff_{dlag}"] = series.diff(dlag).values

        out[f"{col}_sign"] = np.sign(series.values)
        out[f"{col}_abs"] = np.abs(series.values)

    if "Income" in work.columns and "Outcome" in work.columns:
        income = work["Income"]
        outcome = work["Outcome"]
        out["io_ratio"] = (income / outcome.replace(0, np.nan)).values
        out["io_diff"] = (income - outcome).values
        out["io_sum"] = (income + outcome).values
        out["balance_volatility_5"] = work["Balance"].shift(1).rolling(5).std().values
        out["balance_volatility_20"] = work["Balance"].shift(1).rolling(20).std().values

    return out

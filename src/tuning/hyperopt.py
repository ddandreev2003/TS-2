"""Hyperparameter search spaces migrated from the notebook."""

from __future__ import annotations

from typing import Any

import numpy as np


def get_arima_search_space(config: dict[str, Any]) -> dict[str, list[int]]:
    model_cfg = config.get("models", {}).get("arima", {})
    return {
        "p": model_cfg.get("p_range", [0, 1, 2, 3]),
        "d": model_cfg.get("d_range", [0, 1]),
        "q": model_cfg.get("q_range", [0, 1, 2, 3]),
    }


def get_sarima_search_space(config: dict[str, Any]) -> dict[str, Any]:
    arima = get_arima_search_space(config)
    sarima_cfg = config.get("models", {}).get("sarima", {})
    arima.update(
        {
            "P": sarima_cfg.get("P_range", [0, 1]),
            "D": sarima_cfg.get("D_range", [0, 1]),
            "Q": sarima_cfg.get("Q_range", [0, 1]),
            "s": config.get("models", {}).get("seasonal_period", 5),
        }
    )
    return arima


def get_arimax_search_space(config: dict[str, Any]) -> dict[str, list[int]]:
    space = get_arima_search_space(config)
    return {
        "p": [v for v in space["p"] if v <= 2],
        "d": space["d"],
        "q": [v for v in space["q"] if v <= 2],
    }


def get_linear_search_space(model_name: str, config: dict[str, Any]) -> dict[str, list[Any]]:
    linear_cfg = config.get("models", {}).get("linear", {})
    alpha = np.logspace(
        np.log10(linear_cfg.get("alpha_min", 1e-4)),
        np.log10(linear_cfg.get("alpha_max", 10.0)),
        linear_cfg.get("alpha_steps", 20),
    ).tolist()
    if model_name.lower() == "elasticnet":
        return {"model__alpha": alpha, "model__l1_ratio": linear_cfg.get("l1_ratio", [0.1, 0.5, 0.9])}
    return {"model__alpha": alpha}


def get_nonlinear_search_space(model_name: str, config: dict[str, Any]) -> dict[str, list[Any]]:
    _ = model_name
    nl_cfg = config.get("models", {}).get("nonlinear", {})
    return {
        "model__n_estimators": nl_cfg.get("n_estimators", [100, 200]),
        "model__max_depth": nl_cfg.get("max_depth", [4, 8, None]),
        "model__min_samples_leaf": nl_cfg.get("min_samples_leaf", [1, 3, 5]),
        "model__max_features": nl_cfg.get("max_features", ["sqrt", 0.5]),
    }

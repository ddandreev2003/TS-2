"""Dual-target training wrapper for Income and Outcome.

ПОЧЕМУ: притоки и оттоки имеют разную динамику; Balance = Income − Outcome
        даёт бизнес-метрику для позиционера без прямого моделирования сальдо.
КАК: обучаем пару моделей на общем X, прогноз Balance реконструируем как разность.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.metrics.business_metrics import compute_balance_metrics, compute_regression_metrics
from src.models.baseline import NaiveModel, predict_naive, train_naive_baseline
from src.models.tabular import (
    FittedTabularModel,
    predict_tabular,
    train_linear_model,
    train_nonlinear_model,
)

LINEAR_MODELS = frozenset({"lasso", "ridge", "elasticnet"})
NONLINEAR_MODELS = frozenset({"random_forest", "extra_trees", "gradient_boosting"})
DUAL_TARGET_MODELS = frozenset({"naive", *LINEAR_MODELS, *NONLINEAR_MODELS})


@dataclass
class DualTargetModels:
    income_model: Any
    outcome_model: Any
    model_type: str = "tabular"
    model_name: str = "random_forest"
    target_columns: list[str] = field(default_factory=lambda: ["Income", "Outcome"])


def dual_target_artifact_name(model_name: str) -> str:
    return f"dual_target_{model_name}"


def prepare_aligned_dual_data(
    feature_df: pd.DataFrame,
    income_col: str,
    outcome_col: str,
    balance_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    """Return aligned X, y_income, y_outcome, y_balance on shared valid rows."""
    from src.features.assembly import prepare_supervised

    X_income, y_income, feature_names = prepare_supervised(feature_df, target_col=income_col, active_only=True)
    X_outcome, y_outcome, _ = prepare_supervised(feature_df, target_col=outcome_col, active_only=True)
    _, y_balance, _ = prepare_supervised(feature_df, target_col=balance_col, active_only=True)

    common_index = X_income.index.intersection(X_outcome.index).intersection(y_balance.index)
    return (
        X_income.loc[common_index],
        y_income.loc[common_index],
        y_outcome.loc[common_index],
        y_balance.loc[common_index],
        feature_names,
    )


def train_dual_target_pair(
    model_name: str,
    config: dict[str, Any],
    X_train: pd.DataFrame,
    y_income: pd.Series,
    y_outcome: pd.Series,
    income_features: list[str],
    outcome_features: list[str],
) -> DualTargetModels:
    """Train separate Income and Outcome models for one model family."""
    if model_name == "naive":
        return DualTargetModels(
            income_model=train_naive_baseline(y_income),
            outcome_model=train_naive_baseline(y_outcome),
            model_type="naive",
            model_name=model_name,
        )
    if model_name in LINEAR_MODELS:
        return DualTargetModels(
            income_model=train_linear_model(X_train, y_income, model_name, config, features=income_features),
            outcome_model=train_linear_model(X_train, y_outcome, model_name, config, features=outcome_features),
            model_type="tabular",
            model_name=model_name,
        )
    if model_name in NONLINEAR_MODELS:
        return DualTargetModels(
            income_model=train_nonlinear_model(X_train, y_income, model_name, config, features=income_features),
            outcome_model=train_nonlinear_model(X_train, y_outcome, model_name, config, features=outcome_features),
            model_type="tabular",
            model_name=model_name,
        )
    raise ValueError(f"Unsupported dual-target model: {model_name}")


def train_dual_target_models(
    X_train: pd.DataFrame,
    y_income: pd.Series,
    y_outcome: pd.Series,
    model_name: str,
    config: dict[str, Any],
    features: list[str] | None = None,
) -> DualTargetModels:
    """Backward-compatible wrapper for a single feature set."""
    features = features or X_train.columns.tolist()
    return train_dual_target_pair(
        model_name,
        config,
        X_train,
        y_income,
        y_outcome,
        income_features=features,
        outcome_features=features,
    )


def predict_dual_target(models: DualTargetModels, X: pd.DataFrame) -> dict[str, np.ndarray]:
    n_steps = len(X)
    if models.model_type == "naive":
        income_pred = predict_naive(models.income_model, n_steps)
        outcome_pred = predict_naive(models.outcome_model, n_steps)
    else:
        income_pred = predict_tabular(models.income_model, X)
        outcome_pred = predict_tabular(models.outcome_model, X)
    balance_pred = income_pred - outcome_pred
    return {"Income": income_pred, "Outcome": outcome_pred, "Balance": balance_pred}


def evaluate_dual_target(
    models: DualTargetModels,
    X_test: pd.DataFrame,
    y_income: pd.Series,
    y_outcome: pd.Series,
    y_balance: pd.Series,
    threshold: float = 0.42,
) -> dict[str, dict[str, float]]:
    preds = predict_dual_target(models, X_test)
    return {
        "Income": compute_regression_metrics(y_income, preds["Income"]),
        "Outcome": compute_regression_metrics(y_outcome, preds["Outcome"]),
        "Balance": compute_balance_metrics(y_balance, preds["Balance"], threshold=threshold),
    }

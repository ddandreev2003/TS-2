"""Model package exports."""

from src.models.arima_family import (
    FittedTSModel,
    cv_mae,
    fit_and_forecast,
    fit_arima_model,
    fit_arimax_model,
    fit_sarima_model,
    grid_search_arima,
    grid_search_arimax,
    grid_search_sarima,
)
from src.models.baseline import NaiveModel, predict_naive, train_naive_baseline
from src.models.multi_output import (
    DUAL_TARGET_MODELS,
    DualTargetModels,
    dual_target_artifact_name,
    evaluate_dual_target,
    predict_dual_target,
    prepare_aligned_dual_data,
    train_dual_target_pair,
    train_dual_target_models,
)
from src.models.tabular import FittedTabularModel, predict_tabular, train_linear_model, train_nonlinear_model

__all__ = [
    "NaiveModel",
    "train_naive_baseline",
    "predict_naive",
    "FittedTSModel",
    "cv_mae",
    "grid_search_arima",
    "grid_search_sarima",
    "grid_search_arimax",
    "fit_arima_model",
    "fit_sarima_model",
    "fit_arimax_model",
    "fit_and_forecast",
    "FittedTabularModel",
    "train_linear_model",
    "train_nonlinear_model",
    "predict_tabular",
    "DualTargetModels",
    "train_dual_target_models",
    "predict_dual_target",
    "evaluate_dual_target",
]

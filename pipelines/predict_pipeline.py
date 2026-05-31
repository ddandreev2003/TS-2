"""Next-day prediction workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.drift.history import fill_pending_with_actuals, load_residual_history, save_residual_history
from src.features.assembly import prepare_supervised
from src.serving.predictor import predict_next_day
from src.utils.config import load_config


def _update_residual_history(
    monitoring_config_path: str,
    pred_date: str,
    balance_pred: float,
    model_name: str,
    actuals: dict[str, float] | None = None,
) -> None:
    monitoring_config = load_config(monitoring_config_path)
    if not monitoring_config.get("drift", {}).get("enabled", True):
        return
    history_path = monitoring_config.get("artifacts", {}).get(
        "residual_history", "artifacts/residual_history.json"
    )
    records = load_residual_history(history_path)
    if actuals:
        records = fill_pending_with_actuals(records, actuals)
    if not any(str(r.get("date")) == pred_date for r in records):
        records.append(
            {
                "date": pred_date,
                "y_true": None,
                "y_pred": balance_pred,
                "residual": None,
                "status": "pending",
                "model": model_name,
            }
        )
    save_residual_history(history_path, records)


def run_prediction(
    config_path: str = "config/model_config.yaml",
    model_name: str | None = None,
    monitoring_config_path: str = "config/monitoring_config.yaml",
) -> dict:
    result = predict_next_day(config_path, model_name=model_name)

    config = load_config(config_path)
    data_cfg = config.get("data", {})
    target_col = data_cfg.get("target_column", "Balance")
    from src.data.loader import add_active_flag, load_raw_data, validate_date_index
    from src.features.assembly import assemble_feature_matrix

    df = load_raw_data(data_cfg.get("path", "data.csv"), data_cfg.get("date_column", "Date"))
    df = validate_date_index(df, data_cfg.get("date_column", "Date"))
    df = add_active_flag(df)
    feature_df = assemble_feature_matrix(df, config)
    _, y, _ = prepare_supervised(feature_df, target_col=target_col, active_only=True)

    actuals: dict[str, float] = {}
    for idx, value in y.items():
        date_key = str(idx.date()) if isinstance(idx, pd.Timestamp) else str(idx)
        actuals[date_key] = float(value)

    if result.get("Balance") is not None:
        _update_residual_history(
            monitoring_config_path,
            result["date"],
            float(result["Balance"]),
            result["model"],
            actuals=actuals,
        )

    artifacts = config.get("artifacts", {})
    output_path = Path(artifacts.get("dir", "artifacts")) / "latest_prediction.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict next-day liquidity balance")
    parser.add_argument("--config", default="config/model_config.yaml", help="Path to YAML config")
    parser.add_argument("--model", default=None, help="Model name override")
    args = parser.parse_args()
    result = run_prediction(args.config, model_name=args.model)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

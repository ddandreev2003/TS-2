"""Evaluation workflow: recompute hold-out metrics for saved models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.metrics.business_metrics import passes_quality_gate
from src.metrics.evaluation import evaluate_saved_models
from src.serving.prometheus_metrics import update_from_eval_summary
from src.utils.config import load_config, save_json


def _flatten_metrics(model_metrics: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    rows = []
    for model_name, metrics in model_metrics.items():
        if isinstance(metrics, dict) and "Balance" in metrics:
            balance = metrics["Balance"]
            rows.append(
                {
                    "model": model_name,
                    "target": "Balance",
                    "mae": balance.get("mae"),
                    "rmse": balance.get("rmse"),
                    "within_threshold_share": balance.get("within_threshold_share"),
                    "max_abs_error": balance.get("max_abs_error"),
                    "asymmetric_cost": balance.get("asymmetric_cost"),
                    "quality_gate_passed": passes_quality_gate(balance, threshold),
                }
            )
            for target in ("Income", "Outcome"):
                if target in metrics:
                    rows.append(
                        {
                            "model": model_name,
                            "target": target,
                            "mae": metrics[target].get("mae"),
                            "rmse": metrics[target].get("rmse"),
                            "within_threshold_share": None,
                            "max_abs_error": None,
                            "quality_gate_passed": None,
                        }
                    )
        elif isinstance(metrics, dict) and "mae" in metrics:
            rows.append(
                {
                    "model": model_name,
                    "target": "Balance",
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "within_threshold_share": metrics.get("within_threshold_share"),
                    "max_abs_error": metrics.get("max_abs_error"),
                    "asymmetric_cost": metrics.get("asymmetric_cost"),
                    "quality_gate_passed": passes_quality_gate(metrics, threshold),
                }
            )
    return rows


def run_evaluation(config_path: str) -> pd.DataFrame:
    config = load_config(config_path)
    artifacts = config.get("artifacts", {})
    threshold = config.get("metrics", {}).get("balance_mae_threshold", 0.42)

    eval_summary = evaluate_saved_models(config_path)
    eval_path = Path(artifacts.get("dir", "artifacts")) / "eval_metrics.json"
    save_json(eval_summary, eval_path)
    update_from_eval_summary(eval_summary)

    rows = _flatten_metrics(eval_summary.get("model_metrics", {}), threshold)
    ranking = pd.DataFrame(rows)
    if not ranking.empty and "mae" in ranking.columns:
        balance_rows = ranking[ranking["target"] == "Balance"].sort_values("mae")
        ranking_path = artifacts.get("model_ranking", "artifacts/model_ranking.csv")
        balance_rows.to_csv(ranking_path, index=False)

    print("Hold-out evaluation (fresh recompute):")
    if not ranking.empty:
        print(ranking[ranking["target"] == "Balance"][["model", "mae", "quality_gate_passed"]])

    train_metrics = eval_summary.get("train_metrics", {})
    if train_metrics:
        print("\nTrain vs eval Balance MAE diff:")
        for model_name, metrics in eval_summary.get("model_metrics", {}).items():
            if isinstance(metrics, dict) and "Balance" in metrics and model_name in train_metrics:
                train_mae = train_metrics[model_name].get("Balance", {}).get("mae")
                eval_mae = metrics["Balance"].get("mae")
                if train_mae is not None and eval_mae is not None:
                    print(f"  {model_name}: train={train_mae:.4f} eval={eval_mae:.4f} diff={eval_mae - train_mae:+.4f}")

    return ranking


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained liquidity forecasting models")
    parser.add_argument("--config", default="config/model_config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    run_evaluation(args.config)


if __name__ == "__main__":
    main()

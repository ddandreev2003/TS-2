"""Report generation pipeline: plots + hold-out predictions manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.metrics.evaluation import collect_holdout_predictions
from src.utils.config import load_config, save_json
from src.viz.report import generate_all_plots


def run_report(
    config_path: str = "config/model_config.yaml",
    monitoring_config_path: str = "config/monitoring_config.yaml",
    *,
    max_models: int = 8,
) -> dict[str, Any]:
    config = load_config(config_path)
    monitoring_config = load_config(monitoring_config_path)
    artifacts = config.setdefault("artifacts", {})
    artifacts.setdefault("holdout_predictions", "artifacts/holdout_predictions.json")
    artifacts.setdefault("report_manifest", "artifacts/report_manifest.json")

    holdout = collect_holdout_predictions(config_path, max_models=max_models)
    holdout_path = Path(artifacts["holdout_predictions"])
    save_json(holdout, holdout_path)

    manifest = generate_all_plots(
        config,
        holdout_predictions=holdout,
        monitoring_config=monitoring_config,
    )
    return {"holdout_predictions_path": str(holdout_path), "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visualization report artifacts")
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--monitoring", default="config/monitoring_config.yaml")
    parser.add_argument("--max-models", type=int, default=8)
    args = parser.parse_args()
    result = run_report(args.config, args.monitoring, max_models=args.max_models)
    manifest = result["manifest"]
    print(f"Report complete: {manifest.get('n_plots', 0)} plots in {manifest.get('plots_dir')}")


if __name__ == "__main__":
    main()

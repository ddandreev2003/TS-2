"""Dual-target training entry point (Income + Outcome -> derived Balance)."""

from __future__ import annotations

import argparse

from pipelines.train_pipeline import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dual-target Income/Outcome models")
    parser.add_argument("--config", default="config/model_config.yaml")
    args = parser.parse_args()
    summary = run_training(args.config, mode_override="dual_target")
    print("Dual-target training complete.")
    print(summary.get("model_metrics", {}))


if __name__ == "__main__":
    main()

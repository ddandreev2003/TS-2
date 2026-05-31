#!/bin/bash
set -euo pipefail

cd /app

if [ "${SKIP_PIPELINE:-0}" != "1" ]; then
  echo "==> Running ML pipelines (train → eval → monitor → report)"
  python -m pipelines.train_pipeline --config config/model_config.yaml
  python -m pipelines.eval_pipeline --config config/model_config.yaml
  python -m pipelines.monitor_pipeline --config config/model_config.yaml --monitoring config/monitoring_config.yaml
  python -m pipelines.report_pipeline --config config/model_config.yaml --monitoring config/monitoring_config.yaml
  echo "==> Pipelines complete"
else
  echo "==> SKIP_PIPELINE=1 — using mounted artifacts"
fi

exec "$@"

# Plan: Autonomous Liquidity Forecasting App

## 1. Goal

Build an autonomous, container-ready forecasting application for the daily time series in `data.csv`.

The modeling objective is two-step:

1. Predict `Income` for the next day.
2. Predict `Outcome` for the next day.

`Balance` is not the primary target. It is a derived control target computed as:

`Balance = Income - Outcome`

The final business evaluation must be done on the reconstructed `Balance`, while model training and model comparison are done on `Income` and `Outcome` separately.

This is the correct compromise for the dataset we have: the raw file already contains all three columns, but the future application should not depend on direct balance prediction only.

---

## 2. Feasibility Assessment

This project is realistic.

What is already good:

- The data is a daily time series.
- The dataset already contains target values and exogenous signals.
- Calendar effects are explicitly encoded.
- The COVID regime is marked.
- Macro variables are present.

What makes the project non-trivial:

- The series is temporal, so leakage must be avoided.
- `Income` and `Outcome` should be modeled separately or with a shared multi-output design.
- `Balance` must be computed after prediction, not treated as the only modeling target.
- Enterprise MLOps is possible, but it should be a second phase, not the first milestone.

Practical conclusion:

- MVP is fully feasible.
- A Dockerized local app is feasible.
- Full MLflow + DVC + CI/CD + monitoring is feasible as a later phase.

---

## 3. Repository Structure to Build

The current workspace should evolve into the following structure.

### Root-level folders

- `config/` - runtime and experiment configuration.
- `data/raw/` - source data snapshots.
- `data/processed/` - cleaned and feature-ready datasets.
- `data/external/` - auxiliary macro or calendar data.
- `pipelines/` - entry points for train, evaluate, predict, and backtest workflows.
- `src/` - reusable application logic.
- `tests/unit/` - unit tests.
- `tests/integration/` - end-to-end tests.
- `tests/model_quality/` - forecast quality checks.
- `monitoring/` - Prometheus, Grafana, and alert rule assets.
- `logs/` - runtime logs and experiment outputs.
- `artifacts/` - saved models, plots, metrics, and reports.
- `mlflow/` - local tracking artifacts when MLOps is added.
- `retraining/` - retraining outputs and schedules.
- `drift_alerts/` - drift alerts and snapshots.

### Source tree

- `src/features/` - feature generation.
- `src/selection/` - feature selection.
- `src/models/` - baselines and predictive models.
- `src/tuning/` - hyperparameter search.
- `src/calibration/` - recalibration and retraining rules.
- `src/drift/` - drift detection.
- `src/metrics/` - metric calculations and business scoring.
- `src/mlops/` - MLflow, registry, retraining orchestration.
- `src/serving/` - API serving, model loading, shadow mode.

---

## 4. Entry Points

The repository should have a small number of clear entry points.

### Training entry point

Location: `pipelines/train_pipeline` in the future.

Responsibilities:

- load `data.csv`
- validate the time series
- build features
- split by time
- train candidate models for `Income` and `Outcome`
- evaluate by time-based hold-out or rolling window CV
- compute `Balance` from predictions
- save final metrics and selected models

### Evaluation entry point

Location: `pipelines/eval_pipeline` in the future.

Responsibilities:

- load stored predictions or trained models
- compare model families on the same split
- produce a ranking table
- verify the MAE threshold on the reconstructed `Balance`

### Prediction entry point

Location: `pipelines/predict_pipeline` in the future.

Responsibilities:

- take the latest available history window
- build the same lag features used in training
- predict next-day `Income` and `Outcome`
- derive `Balance`
- export the forecast in a machine-readable format

### Serving entry point

Location: `src/serving/`.

Responsibilities:

- expose `/health`
- expose `/predict`
- expose `/metrics`
- load the latest approved model artifact
- support shadow mode for future production validation

### MLOps entry point

Location: `src/mlops/`.

Responsibilities:

- track experiments
- register models
- manage promotion rules
- start retraining on drift or schedule

---

## 5. Implementation Phases

## Phase 1. Data layer

Purpose: make sure the time series is clean, ordered, and safe for modeling.

Tasks:

- load CSV with explicit column names
- parse the `Date` column
- sort rows by date
- verify there are no duplicate dates
- identify gaps in the calendar
- mark weekends and holidays correctly
- inspect missing values in macro variables
- decide whether to forward-fill, back-fill, or leave as missing with indicators
- create a validation report

Functions to implement later:

- `load_raw_data`
- `normalize_columns`
- `validate_date_index`
- `detect_duplicates`
- `summarize_missingness`
- `validate_business_days`
- `build_data_quality_report`

Acceptance criteria:

- the dataset can be loaded without manual cleanup
- the row order is deterministic
- the validation step can stop the pipeline if quality is too low

## Phase 2. Feature engineering

Purpose: transform raw daily history into a supervised learning table.

Feature groups:

- autoregressive lags for `Income` and `Outcome`
- rolling mean, rolling std, min, max
- day-of-week and month features
- `tax_day`
- `IsDayOff_Status_Workalendar_RU`
- `covid`
- `IMICEX`
- `TransRUB1M`
- optional differences and percent changes of macro variables

Functions to implement later:

- `build_autoregressive_features`
- `build_calendar_features`
- `build_macro_features`
- `build_target_lags`
- `build_rolling_windows`
- `assemble_feature_matrix`
- `drop_leaky_rows`

Rules:

- only use information available before the forecast date
- do not use future balance values in features
- keep identical feature generation for both targets

## Phase 3. Modeling

Purpose: compare baseline and non-baseline models for each target.

Model families to include:

- naive persistence baseline
- ARIMA or SARIMA
- ARIMAX with external regressors
- VAR if joint dynamics justify it
- tabular ML models such as Ridge, Lasso, RandomForest, CatBoost, LightGBM, or XGBoost

Decision rule:

- start with two independent target models for `Income` and `Outcome`
- optionally move to a shared multi-output design if it improves stability

Functions to implement later:

- `train_naive_baseline`
- `train_arima_family`
- `train_exogenous_model`
- `train_tabular_regressor`
- `fit_multi_output_forecaster`
- `predict_next_day`

## Phase 4. Feature selection and tuning

Purpose: avoid unstable inputs and tune models systematically.

Selection strategy:

- permutation importance as the main method
- correlation filter as a simple baseline
- SHAP as an explanatory diagnostic layer

Hyperparameter tuning:

- Optuna for ML models
- grid search for ARIMA family orders
- time-series cross-validation only

Functions to implement later:

- `select_features`
- `rank_features`
- `run_optuna_search`
- `search_arima_orders`
- `store_search_results`

## Phase 5. Evaluation

Purpose: compare all models on one fair temporal protocol.

Evaluation rule:

- use one fixed temporal hold-out or rolling window CV
- do not mix random split with time series
- report metrics per target and on reconstructed balance

Metrics per target:

- MAE
- RMSE
- SMAPE
- directional accuracy

Metrics on reconstructed `Balance`:

- MAE on balance
- RMSE on balance
- sign accuracy of balance
- business loss if asymmetry matters

Final selection rule:

- choose the model that gives the best operational balance between target-level accuracy and balance-level business performance
- use `MAE <= 0.42` on `Balance` as a quality gate, not as the only criterion

Functions to implement later:

- `compute_mae`
- `compute_rmse`
- `compute_smape`
- `compute_directional_accuracy`
- `compute_balance_metrics`
- `rank_models`

## Phase 6. Packaging and serving

Purpose: make the system usable as a containerized app.

Package goals:

- one Docker image for local run
- one API process for inference
- one optional batch pipeline for training

Serving behavior:

- load the latest approved model artifact
- receive the latest feature window
- return the next-day predictions for `Income`, `Outcome`, and derived `Balance`

## Phase 7. MLOps extension

Purpose: add experiment tracking, registry, retraining, and monitoring after the MVP works.

Later additions:

- MLflow tracking
- model registry
- DVC for data versioning
- GitLab CI/CD
- retraining scheduler
- drift detector
- Prometheus and Grafana dashboards
- alert rules

---

## 6. Detailed Module Plan

### `src/features/`

Submodules to plan:

- `autoregressive` - lag features and rolling statistics
- `calendar` - day, week, month, holiday, and tax-day features
- `macro` - exogenous macro signals and transformations

### `src/selection/`

Responsibilities:

- remove unstable features
- compare selection methods
- save feature rankings and chosen subsets

### `src/models/`

Responsibilities:

- house baselines
- house classical time-series models
- house tabular ML models
- optionally house multi-output wrappers

### `src/tuning/`

Responsibilities:

- run hyperparameter search
- store search history
- expose reusable search space definitions

### `src/calibration/`

Responsibilities:

- define retraining or recalibration frequency
- compare fresh vs stale performance
- decide whether calibration is needed

### `src/drift/`

Responsibilities:

- detect drift in inputs and residuals
- emit retraining or alert signals
- maintain thresholds and statistics

### `src/metrics/`

Responsibilities:

- compute standard regression metrics
- compute business metrics
- compute derived balance metrics from `Income` and `Outcome`

### `src/mlops/`

Responsibilities:

- experiment tracker
- registry manager
- feature store abstraction
- auto retrain trigger logic
- pipeline orchestration interface

### `src/serving/`

Responsibilities:

- API application
- model loading abstraction
- shadow mode validation

---

## 7. File-by-File Implementation Order

This is the exact order a programmer should follow from top to bottom.

### Step 1. Root bootstrap files

1. `plan.md` - keep this document as the implementation guide.
2. `README.md` - short project overview and run instructions.
3. `requirements.txt` - dependency list.
4. `Dockerfile` - base runtime image.
5. `docker-compose.yml` - local orchestration template.
6. `.gitlab-ci.yml` - CI template.

### Step 2. Configuration files

7. `config/model_config.yaml` - data, split, feature, and metric settings.
8. `config/mlflow_config.yaml` - MLOps tracking template.
9. `config/monitoring_config.yaml` - drift and alert thresholds.

### Step 3. Pipeline entry points

10. `pipelines/__init__.py` - package marker.
11. `pipelines/train_pipeline.py` - training workflow entry point.
12. `pipelines/eval_pipeline.py` - evaluation workflow entry point.
13. `pipelines/predict_pipeline.py` - inference workflow entry point.

### Step 4. Source package bootstrap

14. `src/__init__.py` - top-level package marker.
15. `src/features/__init__.py` - feature package marker.
16. `src/selection/__init__.py` - selection package marker.
17. `src/models/__init__.py` - model package marker.
18. `src/tuning/__init__.py` - tuning package marker.
19. `src/calibration/__init__.py` - calibration package marker.
20. `src/drift/__init__.py` - drift package marker.
21. `src/metrics/__init__.py` - metrics package marker.
22. `src/mlops/__init__.py` - MLOps package marker.
23. `src/serving/__init__.py` - serving package marker.

### Step 5. Data and feature layer

24. `src/features/autoregressive.py` - lags and rolling features.
25. `src/features/calendar.py` - calendar and business-day features.
26. `src/features/macro.py` - macroeconomic features.

### Step 6. Selection and metrics foundation

27. `src/selection/feature_selector.py` - feature filtering and ranking.
28. `src/metrics/business_metrics.py` - MAE, RMSE, SMAPE, directional accuracy, balance cost.

### Step 7. Baseline and model files

29. `src/models/baseline.py` - persistence baseline.
30. `src/models/arima_family.py` - ARIMA, SARIMA, ARIMAX.
31. `src/models/tabular.py` - regression and boosting models.
32. `src/models/multi_output.py` - shared-feature dual-target strategy.

### Step 8. Tuning, calibration, drift

33. `src/tuning/hyperopt.py` - hyperparameter search.
34. `src/calibration/recalibrator.py` - recalibration policy.
35. `src/drift/detector.py` - drift detection.

### Step 9. MLOps layer

36. `src/mlops/experiment_tracker.py` - experiment logging.
37. `src/mlops/model_registry.py` - model registration and promotion.
38. `src/mlops/auto_retrain.py` - retraining orchestration.
39. `src/mlops/pipeline_orchestrator.py` - future workflow orchestration.

### Step 10. Serving layer

40. `src/serving/model_loader.py` - model artifact loading.
41. `src/serving/api.py` - FastAPI application.
42. `src/serving/shadow_mode.py` - shadow-mode comparison.

### Step 11. Tests and monitoring

43. `tests/__init__.py` - test package marker.
44. `tests/unit/test_placeholder.py` - unit test scaffold.
45. `tests/integration/test_placeholder.py` - integration test scaffold.
46. `tests/model_quality/test_placeholder.py` - quality test scaffold.
47. `monitoring/prometheus/prometheus.yml` - scrape template.
48. `monitoring/alerts/alert_rules.yml` - alert template.

### Step 12. Optional runtime folders

49. `artifacts/` - saved outputs.
50. `logs/` - runtime logs.
51. `mlflow/` - tracking artifacts.
52. `retraining/` - retraining outputs.
53. `drift_alerts/` - drift snapshots.

Rule:

- do not implement a later file until the previous layer is at least stubbed and understood
- keep the order strict to minimize dependency confusion
- start with configuration and pipelines before touching model internals

---

## 8. File-by-File Checklist

Use this as the practical checklist during implementation.

### Root files

- `plan.md`: keep the implementation sequence, update it only when scope changes.
- `README.md`: add how to run, what files exist, and what the MVP does.
- `requirements.txt`: pin runtime dependencies used by the first MVP.
- `Dockerfile`: define the base image, workdir, dependency install, and default command.
- `docker-compose.yml`: connect app, local database, object storage, and future monitoring.
- `.gitlab-ci.yml`: define validate, train, evaluate, and deploy steps.

### Config files

- `config/model_config.yaml`: add target columns, lag windows, rolling windows, split sizes, and thresholds.
- `config/mlflow_config.yaml`: add tracking URI and experiment name for later MLOps.
- `config/monitoring_config.yaml`: add drift thresholds and alert switches.

### Pipeline entry points

- `pipelines/__init__.py`: keep package importable.
- `pipelines/train_pipeline.py`: wire the full training flow from loading to saved artifacts.
- `pipelines/eval_pipeline.py`: compare models on a fixed temporal split.
- `pipelines/predict_pipeline.py`: prepare inference from the latest history window.

### Source package bootstrap

- `src/__init__.py`: mark the package root.
- `src/features/__init__.py`: expose feature builders later.
- `src/selection/__init__.py`: expose feature selection later.
- `src/models/__init__.py`: expose model families later.
- `src/tuning/__init__.py`: expose tuning helpers later.
- `src/calibration/__init__.py`: expose calibration helpers later.
- `src/drift/__init__.py`: expose drift tools later.
- `src/metrics/__init__.py`: expose metric helpers later.
- `src/mlops/__init__.py`: expose tracking and registry tools later.
- `src/serving/__init__.py`: expose API and model loader later.

### Data and feature layer

- `src/features/autoregressive.py`: start with lags for `Income` and `Outcome`, then rolling stats.
- `src/features/calendar.py`: add weekday, month, holiday, tax-day, and workday features.
- `src/features/macro.py`: add `IMICEX`, `TransRUB1M`, and their lagged changes.

### Selection and metrics

- `src/selection/feature_selector.py`: start with correlation filter, then permutation importance.
- `src/metrics/business_metrics.py`: implement MAE, RMSE, SMAPE, directional accuracy, and balance metrics.

### Model files

- `src/models/baseline.py`: add naive persistence for both targets.
- `src/models/arima_family.py`: add ARIMA, SARIMA, and ARIMAX wrappers.
- `src/models/tabular.py`: add ridge, lasso, tree-based, and boosting regressors.
- `src/models/multi_output.py`: add a shared-feature two-target training wrapper.

### Tuning, calibration, drift

- `src/tuning/hyperopt.py`: define Optuna or grid-search entry points.
- `src/calibration/recalibrator.py`: define recalibration frequency and retrain windows.
- `src/drift/detector.py`: define drift statistics and alert thresholds.

### MLOps layer

- `src/mlops/experiment_tracker.py`: define experiment start, log metrics, and log artifacts.
- `src/mlops/model_registry.py`: define registration and promotion rules.
- `src/mlops/auto_retrain.py`: define trigger logic for retraining.
- `src/mlops/pipeline_orchestrator.py`: define the orchestration contract.

### Serving layer

- `src/serving/model_loader.py`: load the latest approved model artifact.
- `src/serving/api.py`: expose `/health`, `/predict`, and `/metrics`.
- `src/serving/shadow_mode.py`: compare candidate and current models in parallel.

### Tests and monitoring

- `tests/__init__.py`: keep tests importable.
- `tests/unit/test_placeholder.py`: add validation, feature, and metric unit tests.
- `tests/integration/test_placeholder.py`: add end-to-end flow tests.
- `tests/model_quality/test_placeholder.py`: add hold-out quality gates.
- `monitoring/prometheus/prometheus.yml`: add app scrape targets later.
- `monitoring/alerts/alert_rules.yml`: add alert rules later.

Checklist rule:

- complete the minimal scaffold for one file before moving to the next file
- do not jump to MLOps before the data, features, and baseline models exist
- keep balance evaluation separate from raw target training

---

## 9. Recommended Model Order

Use this order when implementation starts:

1. Naive baseline on `Income` and `Outcome`.
2. ARIMA or SARIMA for each target.
3. ARIMAX with exogenous features.
4. Tabular regression with lagged features.
5. Multi-output model or ensembling if needed.

This order is important because it gives a clean baseline ladder.

---

## 10. Metrics Strategy

### Primary metrics

- MAE on `Income`
- MAE on `Outcome`
- MAE on reconstructed `Balance`

### Secondary metrics

- RMSE
- SMAPE
- directional accuracy
- sign accuracy of `Balance`

### Business metric

Use a derived cost metric on `Balance` to reflect business asymmetry if needed.

### Acceptance gate

- `Balance` MAE must be below the agreed threshold for the model to be considered ready

---

## 11. Manual Implementation Order

1. Keep the directory structure in place.
2. Write data validation logic.
3. Write feature generation logic.
4. Add baseline models.
5. Add evaluation logic.
6. Add model comparison.
7. Add prediction export.
8. Add Docker packaging.
9. Add serving.
10. Add MLOps only after the MVP is stable.

---

## 12. Definition of Done

The first implementation stage is done when:

- the repository has a clear directory skeleton
- the plan is documented in this file
- the modeling strategy is fixed around `Income` and `Outcome`
- `Balance` is derived from the two forecasts
- evaluation metrics are defined
- entry points for pipelines and serving are separated
- the project is ready for manual implementation without redesign

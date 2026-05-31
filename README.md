# TS-2 — Прогноз сальдо ликвидности

Ежедневный прогноз Income, Outcome и Balance (Balance = Income − Outcome) с отбором признаков, сравнением моделей, детекцией разладки и автоматическим дообучением. Проект включает REST API, Streamlit-дашборд и мониторинг Prometheus/Grafana.

**Полный отчёт по методологии и результатам:** [report.md](report.md)

---

## Сервисы (Docker)

| Сервис | URL | Назначение |
|--------|-----|------------|
| API | http://localhost:8000 | `/health`, `/predict`, `/metrics` |
| Streamlit | http://localhost:8501 | Дашборд всех этапов пайплайна |
| Prometheus | http://localhost:9090 | Сбор метрик |
| Grafana | http://localhost:3000 | Дашборд `Liquidity ML Monitoring` (логин: `admin` / `admin`) |

---

## Требования к окружению

- **Python** 3.11+ и `pip`
- **Docker** + Docker Compose (опционально, для полного стека)
- **ОС:** Linux, WSL2, macOS
- **RAM:** ~2 GB для обучения
- **Данные:** файл `data.csv` в корне проекта (уже включён в репозиторий)

---

## Быстрый старт

| Сценарий | Команда | Когда использовать |
|----------|---------|-------------------|
| Локально, полный цикл | см. [Пошаговая инструкция](#пошаговая-инструкция-локально) | Первый запуск, разработка |
| Docker, с обучением | `docker compose up --build` | Демо «всё в одном» |
| Docker, без обучения | `SKIP_PIPELINE=1 docker compose up` | Повторный запуск, после Ctrl+C |

---

## Пошаговая инструкция (локально)

### 1. Клонирование и окружение

```bash
cd /path/to/TS-2
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Обучение (~1–3 мин)

```bash
PYTHONPATH=. python -m pipelines.train_pipeline --config config/model_config.yaml
```

Что происходит:
1. Загрузка и валидация `data.csv`
2. Построение признаков (лаги, календарь, налоги, макро)
3. Отбор признаков (4 метода → автовыбор победителя)
4. Temporal hold-out split (80% train / 20% test)
5. Обучение 7 dual-target пар + ARIMA/SARIMA/ARIMAX
6. Сохранение моделей, метрик, drift baseline

### 3. Оценка на hold-out

```bash
PYTHONPATH=. python -m pipelines.eval_pipeline --config config/model_config.yaml
```

Пересчитывает метрики из сохранённых моделей → `artifacts/eval_metrics.json`, `artifacts/model_ranking.csv`.

### 4. Мониторинг разладки

```bash
PYTHONPATH=. python -m pipelines.monitor_pipeline \
  --config config/model_config.yaml \
  --monitoring config/monitoring_config.yaml
```

Проверяет CUSUM/Shiryayev-Roberts на остатках. При alarm может запустить auto-retrain (если включено в config).

### 5. Генерация графиков

```bash
PYTHONPATH=. python -m pipelines.report_pipeline --config config/model_config.yaml
```

Создаёт 19 PNG в `artifacts/plots/` и манифест для дашборда.

### 6. Streamlit-дашборд

```bash
streamlit run src/dashboard/app.py
```

Откройте http://localhost:8501.

### 7. Прогноз на следующий день (опционально)

```bash
PYTHONPATH=. python -m pipelines.predict_pipeline --config config/model_config.yaml
```

Возвращает JSON: `{Income, Outcome, Balance}`.

### Ожидаемые артефакты после полного цикла

```
artifacts/
├── models/*.joblib          # обученные модели
├── metrics.json             # метрики train
├── eval_metrics.json        # hold-out метрики
├── model_ranking.csv        # ранжирование
├── feature_sets.json        # выбранные признаки
├── calibration_policy.json  # политика калибровки
├── drift_baseline.json      # baseline для drift
├── drift_status.json        # текущий статус
├── holdout_predictions.json # прогнозы на test
├── report_manifest.json     # индекс графиков
└── plots/*.png              # 19 визуализаций
```

---

## Docker: подробно

### Полный стек с обучением

```bash
docker compose up --build
```

Сервис **init** запускает цепочку train → eval → monitor → report ([`docker/entrypoint.sh`](docker/entrypoint.sh)), затем поднимаются app, dashboard, prometheus, grafana.

### Запуск без переобучения

```bash
SKIP_PIPELINE=1 docker compose up
```

Init пропускает пайплайны и использует артефакты из `./artifacts` (volume mount). Подходит для:
- повторного запуска после `Ctrl+C`;
- демо с уже обученными моделями.

**Ctrl+C** останавливает контейнеры, но **не удаляет** артефакты на диске.

### Остановка

```bash
docker compose down        # корректная остановка всех сервисов
```

### Только API

```bash
SKIP_PIPELINE=1 docker compose up app
```

### Первый запуск Grafana

При первом старте Grafana выполняет миграции БД (~500+ migrations) — это нормально, не связано с ML-обучением.

---

## Пайплайны

| Pipeline | Команда | Назначение | Артефакты |
|----------|---------|------------|-----------|
| train | `python -m pipelines.train_pipeline` | Обучение моделей | `models/`, `metrics.json` |
| eval | `python -m pipelines.eval_pipeline` | Hold-out метрики | `eval_metrics.json`, `model_ranking.csv` |
| predict | `python -m pipelines.predict_pipeline` | Прогноз на завтра | stdout JSON |
| monitor | `python -m pipelines.monitor_pipeline --monitoring config/monitoring_config.yaml` | Drift + auto-retrain | `drift_status.json`, `drift_alerts/` |
| retrain | `python -m pipelines.retrain_pipeline --force` | Принудительное дообучение | `retraining/*.json` |
| report | `python -m pipelines.report_pipeline` | Графики для dashboard | `plots/`, `report_manifest.json` |

Все команды запускайте из корня проекта с `PYTHONPATH=.` (или активированным venv в корне).

---

## REST API

Запуск (локально, после обучения):

```bash
PYTHONPATH=. uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
```

### Эндпоинты

| Метод | URL | Описание |
|-------|-----|----------|
| GET | `/health` | Статус сервиса и имя лучшей модели |
| POST | `/predict` | Прогноз Income, Outcome, Balance на следующий день |
| GET | `/metrics` | Prometheus-метрики |

### Примеры

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict

curl http://localhost:8000/metrics
```

---

## Streamlit Dashboard

7 страниц ([`src/dashboard/app.py`](src/dashboard/app.py)):

| Страница | Содержание |
|----------|------------|
| **Overview** | Сводка: train/test rows, лучшая модель, FS method, calibration, ranking |
| **EDA** | 6 графиков исследовательского анализа |
| **Feature Selection** | Таблица сравнения 4 методов + графики стабильности |
| **Models** | MAE, asymmetric cost, forecast grid |
| **Forecasts** | Интерактивный график hold-out прогнозов по моделям |
| **Drift & Anomalies** | Статус drift, control charts |
| **Retrain** | История событий дообучения |

Если страницы пустые — сначала запустите `report_pipeline`.

---

## Конфигурация

### [`config/model_config.yaml`](config/model_config.yaml)

| Параметр | Значение по умолчанию | Описание |
|----------|----------------------|----------|
| `training.mode` | `dual_target` | Режим: dual_target или balance (legacy) |
| `training.include_arima_benchmark` | `true` | Обучать ARIMA/SARIMA/ARIMAX |
| `split.holdout_ratio` | `0.8` | Доля train |
| `metrics.balance_mae_threshold` | `0.42` | Quality gate |
| `retraining.auto_retrain` | `true` | Авто-retrain при drift/schedule |
| `selection.top_k_features` | `20` | Число признаков после FS |

### [`config/monitoring_config.yaml`](config/monitoring_config.yaml)

| Параметр | Значение | Описание |
|----------|----------|----------|
| `drift.method` | `cusum` | cusum или shiryayev_roberts |
| `drift.cusum.h` | `5.0` | Порог CUSUM |
| `metrics.balance_mae_threshold` | `0.42` | Порог для алертов |

### Legacy: прямой прогноз Balance

Измените в `config/model_config.yaml`:

```yaml
training:
  mode: balance
```

Затем перезапустите train_pipeline.

---

## Переобучение

### Ручное (принудительное)

```bash
PYTHONPATH=. python -m pipelines.retrain_pipeline \
  --config config/model_config.yaml \
  --force
```

### Автоматическое

Monitor запускает retrain при:
- обнаружении drift (`retrain_on_drift: true`);
- расписании (`retrain_on_schedule: true`).

Логи: `retraining/retrain_*.json`, `retraining/last_retrain.json`.

### После retrain

Re-evaluate и обновите графики вручную:

```bash
PYTHONPATH=. python -m pipelines.eval_pipeline --config config/model_config.yaml
PYTHONPATH=. python -m pipelines.report_pipeline --config config/model_config.yaml
```

---

## Тесты

```bash
# Быстрые unit + API
pytest tests/unit tests/integration/test_api.py -q

# Quality gate (нужны обученные артефакты)
pytest tests/model_quality -q

# Полный integration (медленно)
pytest tests/integration -m slow -q
```

Quality gate: лучшая модель Balance MAE ≤ 0.42 на hold-out.

---

## Структура проекта

```
TS-2/
├── config/                  # YAML-конфигурации
├── data.csv                 # исходные данные
├── pipelines/               # entry points (train, eval, predict, ...)
├── src/
│   ├── data/                # загрузка и валидация
│   ├── features/            # feature engineering
│   ├── selection/           # feature selection (4 метода)
│   ├── models/              # baseline, ARIMA, tabular ML, dual-target
│   ├── calibration/         # политика калибровки
│   ├── drift/               # CUSUM / Shiryayev-Roberts
│   ├── metrics/             # метрики и business scoring
│   ├── mlops/               # auto-retrain, MLflow
│   ├── serving/             # FastAPI, Prometheus
│   ├── viz/                 # генерация графиков
│   └── dashboard/           # Streamlit app
├── artifacts/               # модели, метрики, графики
├── monitoring/              # Prometheus, Grafana configs
├── docker/                  # entrypoint, supervisord
├── tests/                   # unit, integration, model_quality
├── docker-compose.yml
├── Dockerfile
├── report.md                # формальный отчёт
└── README.md                # этот файл
```

---

## Troubleshooting

| Проблема | Решение |
|----------|---------|
| `ModuleNotFoundError: No module named 'src'` | Запускайте из корня с `PYTHONPATH=.` или используйте Docker |
| Dashboard пустой, «No plots found» | Запустите `report_pipeline` |
| API возвращает `no_model` | Сначала выполните `train_pipeline` |
| Grafana долго стартует с migrations | Нормально при первом запуске |
| Хочу перезапустить без обучения | `SKIP_PIPELINE=1 docker compose up` |
| Ctrl+C остановил Docker | Артефакты сохранены; перезапустите с `SKIP_PIPELINE=1` |

---

## Ссылки

- [report.md](report.md) — формальный отчёт: методология, метрики, соответствие требованиям
- [Итоговый_TS_Project_FS.ipynb](Итоговый_TS_Project_FS.ipynb) — research reference (notebook); production logic в `src/` и `pipelines/`
- [plan.md](plan.md) — план реализации проекта

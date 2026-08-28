# 🏬 Retail Demand Forecasting System

An end-to-end MLOps platform designed to optimize retail inventory planning and prevent stockouts. The system delivers 28-day store and item-level demand projections using LightGBM, governed with reproducible data versioning, automated model evaluation gates, and an interactive Plotly Dash decision dashboard.

---

## 🎯 Business Value

* **Stockout Prevention & Waste Reduction:** Accurate intermittent demand forecasts help store managers maintain optimal inventory levels across departments.
* **Production Reliability:** Shared feature engineering between training and inference eliminates data discrepancies and ensures dependable forecasts.
* **Automated Model Governance:** Continuous validation prevents performance regressions by blocking models that fail to outperform historical baselines.

---


---
📊 Model Performance & BenchmarksModels are evaluated using Weighted Absolute Percentage Error (WAPE) and Root Mean Squared Error (RMSE) across time-series validation splits.Model PipelineStrategyWAPE (%)RMSEStatusBaseline BenchmarkNaïve Seasonal / Lag-724.12%3.14ArchivedLightGBM ForecasterTuned Hyperparameters17.85%2.08


## Repository Structure

```
.
├── app.py                              # Root WSGI (Dash server = app.server) for Render
├── Procfile                            # web: gunicorn app:server --workers 2 --timeout 120
├── pyproject.toml / requirements.txt   # uv / pip — kedro, pyspark, lightgbm, mlflow, fastapi, dash
├── dvc.yaml / dvc.lock / data.dvc      # Pipeline + data versioning
├── conf/
│   ├── base/
│   │   ├── catalog.yml                 # 6 datasets — filepath authority
│   │   ├── parameters.yml              # lag_days, rolling_windows, model_params, optuna, metrics, drift
│   │   └── mlflow.yml                  # sqlite:///mlflow.db
│   └── logging.yml
├── data/                               # DVC-tracked (not in git)
│   ├── 01_raw/                         # calendar.csv, sell_prices.csv, sales_train_validation.csv
│   ├── 02_intermediate/                # sales_melted.parquet
│   ├── 03_features/                    # model_input.parquet (377K rows)
│   └── 06_metrics/                     # model_metrics.json
├── models/
│   └── forecast_model.pkl              # LightGBM 38-feature, DVC persisted
├── metrics.json                        # Production WAPE 29.74 / R2 0.828
├── metrics_baseline.json / metrics_tuned.json
├── mlflow.db / mlruns/                 # Experiment store
├── drift_report.json                   # PSI / KS per feature
├── src/
│   ├── data_prep.py                    # DVC shim -> kedro pipelines or placeholder
│   ├── train.py                        # pandas fallback + train/tune + catalog resolution
│   └── retail_demand_forecasting/
│       ├── pipeline_registry.py
│       ├── nodes/
│       │   ├── constants.py            # FEATURE_COLS (38), TARGET_COL, RANDOM_STATE
│       │   ├── data_engineering.py     # unpivot wide d_1..d_1913 + calendar/price joins
│       │   ├── feature_engineering.py  # lags, rolling, calendar, hierarchical, clean
│       │   └── data_science.py         # train_model, tune_and_train, compute_metrics
│       ├── pipelines/
│       │   ├── data_engineering/
│       │   ├── feature_engineering/
│       │   └── data_science/
│       ├── utils/
│       │   └── catalog.py              # get_catalog_filepath() — config authority
│       ├── api/
│       │   └── app.py                  # FastAPI /health, /predict, /predict/batch
│       └── dash_app/
│           └── app.py                  # Dash 5 KPIs, recursive forecast, 2-decimal table
├── scripts/
│   ├── evaluate_gate.py                # WAPE/R2 promotion gate
│   ├── check_catalog_authority.py      # no hard-coded data/ paths
│   └── export_gh_pages.py              # docs/index.html for GitHub Pages
├── tests/
│   ├── test_api.py                     # 38-field Pydantic + batch + float
│   ├── test_drift.py                   # PSI/KS + zero-rate + payload bounds
│   ├── test_nodes.py                   # metrics, split, train
│   ├── test_transforms.py              # rowsBetween(-w,-1) audit
│   └── test_training_serving_parity.py # Spark vs pandas vs API parity
├── docs/
│   └── index.html                      # Static demo (Plotly CDN)
├── .github/workflows/ci_cd.yml         # ruff → pytest → gate → docker
└── start_all.bat / start_all.py        # Launch API 8000 + Dash 8050
```

---

## Deployment

**Render / Railway (free)**
1. Push to GitHub; ensure `app.py` exposes `server`, `Procfile` and `requirements.txt` list `gunicorn, dash, lightgbm`
2. Render: New Web Service → Build `uv sync` or `pip install -r requirements.txt` → Start `gunicorn app:server --workers 2 --timeout 120`
3. Env: `PYTHONPATH=src`, `API_BASE` if API separate

**GitHub Pages (static demo)**
```bash
uv run python scripts/export_gh_pages.py
git add docs/index.html && git commit -m "gh-pages" && git push
# Settings → Pages → Source: main / docs
```

**Docker**
```bash
docker build -t retail-demand-forecasting:prod .
docker run -p 8000:8000 -p 8050:8050 retail-demand-forecasting:prod
```

---

## Quality

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run pytest -q --cov=src
uv run python scripts/check_catalog_authority.py
uv run python scripts/evaluate_gate.py
```

CI `.github/workflows/ci_cd.yml` — lint → test → gates → `dvc repro --dry` → docker build. Coverage target 80%+ on `api/`, `nodes/`.

---

## License

MIT



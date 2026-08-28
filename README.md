PowerShellcd "D:\Retail Demand Forecasting"

@'
# 🏬 Retail Demand Forecasting System (Production MLOps)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![DVC Tracked](https://img.shields.io/badge/data-DVC-9cf.svg)](https://dvc.org/)
[![MLflow Tracking](https://img.shields.io/badge/experiments-MLflow-0194E2.svg)](https://mlflow.org/)
[![Dashboard](https://img.shields.io/badge/UI-Plotly_Dash-blue.svg)](https://plotly.com/dash/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

An end-to-end Machine Learning Operations (MLOps) platform designed to optimize retail inventory planning and forecast intermittent sales. The system integrates Kedro modular pipelines, reproducible DVC data versioning, MLflow experiment tracking with automated promotion gates, and an interactive Plotly Dash decision dashboard.

---

## 🎯 Business Value & Core Capabilities

* **Stockout Reduction & Demand Planning:** Delivers store- and item-level demand projections across flexible forecast horizons (7 to 28 days).
* **Zero Training-Serving Skew:** Unified feature engineering pipeline shared across batch DAGs and real-time inference endpoints.
* **Automated Governance Gates:** Continuous validation blocks model deployment unless candidate runs beat baseline thresholds ($WAPE$ and $R^2$).
* **Statistical Drift Monitoring:** Production-ready Kolmogorov-Smirnov (KS) tests and Population Stability Index (PSI) to detect feature distribution shifts.

---

## Dash app 

<img width="1920" height="1080" alt="retail_dashbord1" src="https://github.com/user-attachments/assets/fc029543-cc9e-4955-be9a-6443633bfe70" />




📊 Model Performance & BenchmarksModels are evaluated across time-series validation splits. Candidates are promoted only if they outperform the naïve benchmark across both error and goodness-of-fit metrics:Model PipelineStrategyWAPE (%)RMSER2 ScoreRegistry StatusBaseline BenchmarkNaïve Seasonal (Lag-7)34.80%3.420.710ArchivedLightGBM (Tuned)38-Feature Pipeline29.74%2.150.828Production🚀 



📁 Repository ArchitecturePlaintext├── app.py                              # Root WSGI entry point (Dash server) for cloud deployment

├── app.py                              # Root WSGI entry point (Dash server) for cloud deployment
├── Procfile                            # Gunicorn production worker configuration
├── pyproject.toml / requirements.txt   # Dependency definitions (uv / pip)
├── dvc.yaml / dvc.lock                 # DVC pipeline DAG and reproducible stages
├── conf/
│   ├── base/
│   │   ├── catalog.yml                 # Dataset registry & centralized filepath authority
│   │   ├── parameters.yml              # Model hyperparameters, lag windows, & drift configs
│   │   └── mlflow.yml                  # MLflow tracking URI & experiment metadata
│   └── logging.yml
├── data/                               # DVC-tracked data layers (excluded from git)
│   ├── 01_raw/                         # Raw retail sales, calendar, and price tables
│   ├── 02_intermediate/                # Unpivoted sales data (sales_melted.parquet)
│   └── 03_features/                    # 38-feature engineered dataset (model_input.parquet)
├── models/
│   └── forecast_model.pkl              # Production LightGBM model artifact
├── metrics.json                        # Final model metrics (WAPE, RMSE, R²)
├── drift_report.json                   # Statistical drift monitoring results (KS-test / PSI)
├── src/
│   ├── data_prep.py                    # Data preparation pipeline runner
│   ├── train.py                        # Training and hyperparameter tuning runner
│   └── retail_demand_forecasting/
│       ├── nodes/                      # Modular Kedro node functions
│       │   ├── constants.py            # Feature definitions (38 cols) & random seeds
│       │   ├── data_engineering.py     # Unpivoting & relational calendar/price joins
│       │   ├── feature_engineering.py  # Lag features, rolling windows, & calendar encodings
│       │   └── data_science.py         # LightGBM training, tuning, & evaluation metrics
│       ├── pipelines/                  # Data and modeling DAG execution logic
│       ├── utils/
│       │   └── catalog.py              # Centralized configuration resolver
│       ├── api/
│       │   └── app.py                  # FastAPI service (/health, /predict, /predict/batch)
│       └── dash_app/
│           └── app.py                  # Plotly Dash interactive executive dashboard
├── scripts/
│   ├── evaluate_gate.py                # Automated WAPE/R² model promotion gate
│   └── check_catalog_authority.py      # Linter checking for hardcoded data paths
├── tests/
│   ├── test_api.py                     # Pydantic schema & endpoint latency validation
│   ├── test_drift.py                   # Data and concept drift statistical tests
│   ├── test_nodes.py                   # Unit tests for data transformation nodes
│   ├── test_transforms.py              # Windowing and rolling calculation tests
│   └── test_training_serving_parity.py # Training vs. inference feature parity tests
├── .github/workflows/ci_cd.yml         # CI/CD pipeline (Ruff -> Pytest -> Gate -> Build)
└── start_all.bat                       # Local one-click runner for API & Dashboard

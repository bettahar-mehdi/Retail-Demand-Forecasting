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




# Retail Demand Forecasting Constitution

## Core Principles

### I. Zero Training-Serving Skew (NON-NEGOTIABLE)
Training and serving MUST use an identical, leakage-free transformation. No feature may be computed differently offline vs online.

- **Single source of feature logic**: All lags, rolling aggregates, calendar/cyclical, and hierarchical features defined in `src/retail_demand_forecasting/nodes/feature_engineering.py` are authoritative. `src/retail_demand_forecasting/api/app.py::_prepare_X` and `src/retail_demand_forecasting/dash_app/app.py` MUST align inputs to `src/retail_demand_forecasting/nodes/constants.py:6` `FEATURE_COLS` (32 cols) without re-implementing transformations.
- **Leakage-free contract**: Lags use `F.lag(..., n).over(Window.partitionBy("store_id","item_id").orderBy("day_id"))`; rolling aggregates use `rowsBetween(-w, -1)` (equivalent to `shift(1).rolling(w)`) — enforced and audited by `tests/test_transforms.py`. `lead()`, forward-fill, or `rowsBetween(...,0)` are FORBIDDEN. `clean_features()` MUST `dropna` first `max(lag_days)` rows per series.
- **Float preservation**: Sparse intermittent demand (e.g., `HOBBIES_1_003`) predicts rates like `0.15`. Serving MUST preserve `float` end-to-end — no `int()` cast in `PredictionRequest`, `PredictionResponse`, or aggregation (Dash 7/30/60-day horizons sum floats). Clip only negatives: `max(0.0, pred)`.
- **Verification**: Any change to feature engineering REQUIRES `tests/test_transforms.py::test_feature_engineering_source_uses_trailing_window` + `test_pandas_trailing_rolling_no_leakage` to pass and a review demonstrating training features == serving features (column-for-column, order via `FEATURE_COLS`).

### II. Configuration-Driven Datasets & Parameters (conf/ Authority)
File paths and hyperparameters MUST NOT be hard-coded.

- **Catalog authority**: All dataset paths MUST be declared in `conf/base/catalog.yml` and resolved via Kedro `DataCatalog` / `catalog.yml` entries (`calendar_raw`, `sell_prices_raw`, `sales_train_raw`, `intermediate_sales_melted`, `model_input_features`, `model_metrics`). Code that directly opens `data/01_raw/...` outside the catalog is a violation.
- **Parameters authority**: All tunable values ( `lag_days: [1,2,3,7,14,21,28]`, `rolling_window_days: [7,28]`, `model_params.tweedie_variance_power: 1.3`, `optuna.n_trials`, `metrics.primary: wape` ) MUST be read from `conf/base/parameters.yml` (and `conf/base/mlflow.yml` for tracking URI). Nodes receive `params: dict` injected by Kedro, not inline literals.
- **Environment overlay**: Secrets and URIs (e.g., `MLFLOW_TRACKING_URI=sqlite:///mlflow.db` in `src/retail_demand_forecasting/nodes/constants.py:58`) MAY be overridden via `conf/local/` or env vars, but defaults must be declared in `conf/base/`.
- **Verification**: `grep -r "data/0" src/ --include="*.py"` outside catalog loading, or any literal filepath assigned to `filepath:` in Python code, fails review. CI MUST run `kedro catalog list` / `dvc repro --dry` without path drift.

### III. MLflow Parameter/Metric Logging with Evaluation Gates (NON-NEGOTIABLE)
Every training execution MUST be tracked and gated.

- **Mandatory logging**: `src/retail_demand_forecasting/nodes/data_science.py:train_model` and `tune_and_train` MUST `mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)` (`sqlite:///mlflow.db`), `mlflow.start_run`, `mlflow.log_param` for every key in `model_params` + `random_state`, `test_size`, `features` (JSON), `train_rows`/`test_rows`, and `mlflow.log_metric` for `wape`, `rmse`, `mae`, `mape`, `r2` via `compute_metrics()`. Flavor-specific logging (`mlflow.lightgbm.log_model` / `mlflow.xgboost.log_model` / `mlflow.sklearn.log_model`) is required.
- **Optuna provenance**: `optuna_objective` optimizes mean `WAPE` under `TimeSeriesSplit(n_splits=cv_splits)` with `TPESampler(seed=42)`. All trials log `best_cv_wape`, `trial_{i}_wape`, and `best_{param}` to the parent run; best model artifact is `model_best`.
- **Evaluation gates**: No model may be promoted to `models/forecast_model.pkl` or `metrics.json` (DVC-tracked via `dvc.yaml`) unless:
  1. `WAPE` (primary) is computed on a chronological holdout (`time_series_train_test_split` — last `test_size` fraction, no shuffling, `seed_everything(42)`).
  2. Metrics are written to `metrics.json` and `data/06_metrics/model_metrics.json` and compared to baseline (e.g., `metrics_baseline.json`). Regression gate: `wape` must not degrade >2pp vs baseline and `r2 >= 0.80` on 377k-row sample; otherwise PR is blocked.
  3. DVC stage `dvc repro` reproduces deterministically.
- **Verification**: CI checks that an MLflow run exists with all `model_params` + all `metrics.track: [wape, rmse, mae, r2, mape]`, and that `metrics.json` is updated. Missing params/metrics fails the gate.

### IV. Pydantic Input Validation for Dash/API Serving (NON-NEGOTIABLE)
All serving boundaries MUST validate with Pydantic v2.

- **Schemas**: `src/retail_demand_forecasting/api/app.py:84` `PredictionRequest` is the single validation boundary. Every feature has `Field(ge, le)` bounds (e.g., lags/rollings `ge=0 le=10000`, `day_of_week 1..7`, `is_weekend 0..1`, cyclical `sin/cos -1..1`, `sell_price ge=0`). `model_config = {"extra": "forbid"}` — unknown fields are rejected. `BatchPredictionRequest` enforces `min_length=1 max_length=1000`.
- **Behavior**: Validation errors return `422 Unprocessable Entity`; missing model returns `503`; inference exceptions return `500` via `HTTPException`. Dash callbacks (`src/retail_demand_forecasting/dash_app/app.py`) MUST reuse the same Pydantic models or delegate to `/predict` — no raw `request.json` parsing bypassing validation.
- **Dataframe alignment**: `_prepare_X` MUST construct `pd.DataFrame([row])[FEATURE_COLS].fillna(0.0)` — never trust caller-provided column order. Hierarchical features default to `0.0` if absent.
- **Verification**: `tests/test_api.py` MUST cover boundary violations (negative lag, extra field, batch >1000) and float preservation. Ruff `C408`/`SIM105` exceptions notwithstanding, Pydantic fields MUST remain typed `float` for sparse items.

### V. Automated Distribution Drift Tests (NON-NEGOTIABLE)
Data and concept drift MUST be tested automatically.

- **Pipeline tests**: Distribution checks live in `tests/` (`test_transforms.py`, `test_nodes.py`) and/or a dedicated `tests/test_drift.py`. They MUST run on every PR and nightly.
- **Required checks**:
  1. **Feature distribution drift**: Kolmogorov-Smirnov or Population Stability Index (PSI) per `FEATURE_COLS` vs training snapshot (`data/03_features/model_input.parquet` or reference sample). Alert if `p < 0.05` or `PSI > 0.2`.
  2. **Target drift**: Zero-rate and mean sales per `cat_id`/`dept_id` (critical for `HOBBIES_*` sparse) compared to baseline.
  3. **Serving payload drift**: Validate live `PredictionRequest` batches against training feature ranges (bounds in Pydantic double as drift sentinels).
- **Integration**: Drift tests are independent of `pytest -q --cov` quality gates and MUST produce a machine-readable report (JSON) consumed by DVC/MLflow or CI artifacts. Failures block promotion but not local dev — flagged as `warning` outside `main`.
- **Verification**: PRs that add new features MUST add corresponding drift assertions and update the reference statistics artifact.

## Additional Constraints

**Technology Stack (locked)**:
- Orchestration: Kedro 1.3.1 + PySpark 4.1.1 (`spark.SparkDataset`, `kedro-datasets[spark]`), DVC 3.62.0 (`dvc.yaml`, `dvc.lock`, `data.dvc`)
- Experiment tracking: MLflow 3.15.1 (`sqlite:///mlflow.db`, `mlruns/`, `kedro-mlflow 2.0.3`), Optuna 4.4.0 (`TimeSeriesSplit`)
- Modeling: scikit-learn 1.5.2, LightGBM 4.6.0 (`objective=tweedie`, `tweedie_variance_power 1.1-1.5`), XGBoost 3.0.2; `RANDOM_STATE=42`
- Serving: FastAPI 0.136.1 + Uvicorn + Pydantic 2.13.3, Dash 4.4.1 + Plotly 5.24.1, Gunicorn 21.2.0 (`app.py:server`, `Procfile`)
- Quality: Ruff 0.15.22 (`line-length 100`, `target-version py310`), Pytest 9.1.1 + pytest-cov, mypy (optional)

**DVC & Reproducibility**:
- Pipeline flow `data/01_raw` → `data/02_intermediate` → `data/03_features` → `models/forecast_model.pkl` → `metrics.json` is DVC-tracked. `dvc repro` MUST be reproducible with fixed seeds (`seed_everything`). No manual copy of artifacts.

**Security & Secrets**:
- No secrets in `conf/base/`; use `conf/local/` (gitignored) + `.env.example`. `KAGGLE_*` via env only.

**API Contracts**:
- `GET /health` → `{status, model_loaded}`, `POST /predict` → `PredictionResponse{prediction: float, features: dict}`, `POST /predict/batch` → `BatchPredictionResponse`. Model loading prefers local pickle then MLflow `runs:/{run_id}/{model_best|model}`.

## Development Workflow

1. **Spec-first**: Feature starts from a `spec.md` under `.specify/` instantiated from `.specify/templates/spec-template.md`. Constitution takes precedence over spec if conflict.
2. **Plan**: `.specify/templates/plan-template.md` documents stack choices and gates (WAPE primary, leakage audit).
3. **Tasks**: `.specify/templates/tasks-template.md` splits work; each task references a principle (I–V).
4. **Implementation order**: Catalog/params → Feature engineering (tests/test_transforms) → Data science (MLflow) → API/Dash (Pydantic) → Drift tests → DVC repro → Docs (`docs/index.html` via `scripts/export_gh_pages.py`).
5. **Quality gates per PR** (`.github/workflows/ci_cd.yml`):
   ```
   ruff check src/ tests/ ; ruff format --check src/ tests/
   pytest tests/ -v --cov=src --cov-report=term-missing
   dvc repro --dry   # validate pipeline
   # MLflow gate + drift report artifacts
   ```
   All gates must pass; evaluation gate (WAPE regression) blocks merge to `main`.
6. **Review checklist** (from `.specify/templates/checklist-template.md`): reviewer verifies I–V compliance, `conf/` authority, MLflow logging completeness, Pydantic bounds, drift tests.

## Governance

This constitution supersedes all other practices, templates, and informal decisions. All PRs, pipeline runs, and serving changes MUST verify compliance with Principles I–V.

- **Amendments**: Require a PR updating this file with: (a) rationale, (b) impact on `conf/`, `src/`, `tests/`, `dvc.yaml`, (c) migration plan for existing MLflow runs / `models/forecast_model.pkl`. Approvals from tech lead + MLOps owner required.
- **Versioning**: Follow semantic versioning: MAJOR = breaking principle change, MINOR = new constraint/gate, PATCH = clarification.
- **Compliance**: Use `constitution.md` as runtime guidance for agents and reviewers. Complexity or exceptions must be justified in the PR description and recorded as an ADR in `docs/`.
- **Reference enforcement**: File paths with line numbers (e.g., `src/retail_demand_forecasting/api/app.py:84`, `src/retail_demand_forecasting/nodes/constants.py:6`) are binding anchors — refactors MUST update the constitution.

**Version**: 1.0.0 | **Ratified**: 2026-08-27 | **Last Amended**: 2026-08-27

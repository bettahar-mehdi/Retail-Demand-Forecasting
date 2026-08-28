# Feature Specification: Audit and Enforce All 5 Constitution Principles

**Feature Branch**: `001-audit-and-enforce`

**Created**: 2026-08-27

**Status**: Draft

**Input**: User description: "Audit and enforce all 5 principles in constitution.md across our existing code."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Zero Training-Serving Skew Enforcement (Priority: P1)

As an MLOps engineer, I need training and serving to use identical, leakage-free transformations so that offline metrics match online predictions and sparse items (e.g., `HOBBIES_1_003`) return float rates like `0.15` instead of collapsed zeros.

**Why this priority**: P1 - Constitution Principle I is NON-NEGOTIABLE. Skew causes silent production accuracy loss. Current risk: `src/train.py:_add_pandas_phase1_features` duplicates Spark logic; serving `_build_forecast` in Dash does recursive updates that must exactly match `F.lag` / `rowsBetween(-w,-1)` contract.

**Independent Test**: Run `pytest tests/test_transforms.py -v` and a new parity test `tests/test_training_serving_parity.py` that generates same input via Spark `feature_engineering.create_features` vs pandas shim vs `PredictionRequest` payload builder and asserts column-for-column equality (32 `FEATURE_COLS` ordered) and `lead()` absent. Manually verify Dash 7/30/60-day horizon sums floats (no `int()` cast).

**Acceptance Scenarios**:

1. **Given** raw sales with known lags `[sales_t-1, sales_t-7]`, **When** Spark `create_lag_features` and pandas `_add_pandas_phase1_features` run on same sorted data, **Then** resulting `lag_1..lag_28` values are identical (±1e-6) and no future row influences current row.
2. **Given** a valid `PredictionRequest` with `lag_1=0.15` (float sparse rate), **When** `POST /predict` is called, **Then** response `prediction` is `float` (e.g., `0.15`) with `max(0.0, pred)` only, never rounded to `0`, and `features` echo matches input.
3. **Given** change to `feature_engineering.py`, **When** CI runs `test_feature_engineering_source_uses_trailing_window` + `test_pandas_trailing_rolling_no_leakage`, **Then** PR fails if `rowsBetween(...,0)` or `lead(` is introduced or `dropna` cutoff removed.

---

### User Story 2 - Strict conf/ Authority for Dataset Paths (Priority: P1)

As a platform engineer, I need every dataset filepath to be declared in `conf/base/catalog.yml` and resolved via Kedro DataCatalog so that path changes require only config edits and hard-coded `data/01_raw/...` cannot drift.

**Why this priority**: P1 - Constitution Principle II. Audit found direct FS access violating authority: `src/retail_demand_forecasting/dash_app/app.py:41` `PROJECT_ROOT / "data/01_raw/..."` via `pd.read_csv`; `src/train.py:140` `Path("data/01_raw/...")`; `src/data_prep.py:13` `Path("data/01_raw/calendar.csv")`. These bypass catalog versioning and break DVC/Kedro lineage.

**Independent Test**: Run `rg -n "data/0" src/ --glob '*.py'` outside catalog loader; must return zero violations after fix. Run `python -m kedro catalog list` and `dvc repro --dry` successfully with empty overrides. Dash starts without raw CSVs (graceful empty state) and `PROJECT_ROOT` env still supported.

**Acceptance Scenarios**:

1. **Given** `conf/base/catalog.yml` declares `calendar_raw: filepath: data/01_raw/calendar.csv`, **When** any Python code needs calendar data, **Then** it MUST load via catalog (`DataCatalog.load("calendar_raw")` or `catalog.yml` lookup), not `Path(...)`/`open(...)`.
2. **Given** `src/train.py` fallback `_build_real_features_pandas` needs raw CSV, **When** it executes, **Then** it resolves paths via `yaml.safe_load(conf/base/catalog.yml)["calendar_raw"]["filepath"]` or via injected catalog, never hard-coded strings.
3. **Given** `src/retail_demand_forecasting/dash_app/app.py` starts with missing `data/01_raw/`, **When** `_load_sample_data()` runs, **Then** it either uses catalog resolution or logs `"Sales data not found"` and returns `None` without hard-coded `Path("data/...")` literal outside a catalog helper.

---

### User Story 3 - MLflow Parameter/Metric Logging with Evaluation Gates (Priority: P1)

As a data scientist, I need every training run to log all `model_params` + metrics to `sqlite:///mlflow.db` and block promotion if `WAPE` regresses >2pp vs baseline or `r2 < 0.80`, so we never ship a worse model.

**Why this priority**: P1 - Constitution Principle III NON-NEGOTIABLE. Current `train_model` logs params/metrics correctly (`src/retail_demand_forecasting/nodes/data_science.py:250`), but no automated gate compares `metrics.json` vs `metrics_baseline.json` and CI does not fail on missing MLflow runs.

**Independent Test**: Trigger `python src/train.py --no-tune` twice with same seed; verify `mlflow.db` has runs with `log_param("tweedie_variance_power")`, `log_metric("wape")`, `log_metric("r2")`, flavor `model` artifact. Run new script `scripts/evaluate_gate.py` that loads `metrics.json` + `metrics_baseline.json` and asserts `wape <= baseline_wape + 2.0` and `r2 >= 0.80`; CI job fails otherwise. Run `pytest tests/test_nodes.py::TestTrainModel::test_returns_expected_keys` to confirm metric keys.

**Acceptance Scenarios**:

1. **Given** `train_model` completes with `WAPE=29.7%`, `R2=0.828`, **When** run finishes, **Then** MLflow run (name `retail_demand_forecast`) exists with params `model_type, objective, tweedie_variance_power, n_estimators, max_depth, num_leaves, learning_rate, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda, random_state, test_size, train_rows, test_rows, features` and metrics `wape, rmse, mae, mape, r2`.
2. **Given** `tune_and_train` with `n_trials=30`, **When** it finishes, **Then** parent run `optuna_tuning` logs `best_cv_wape`, `trial_{i}_wape`, `best_{param}`, holdout `wape`, and artifact `model_best`, plus `metrics.json` with `best_params`.
3. **Given** new `metrics.json` has `wape=35.0` while baseline `29.7`, **When** `scripts/evaluate_gate.py` runs in CI, **Then** it exits non-zero (`WAPE regression 5.3pp > 2pp gate`) and PR is blocked.

---

### User Story 4 - Pydantic Input Validation for Dash/API Serving (Priority: P1)

As an API consumer, I need all Dash and API inputs validated by Pydantic v2 with `extra="forbid"` and bounds, so invalid `day_of_week=8` returns `422` and hierarchical features default safely.

**Why this priority**: P1 - Constitution Principle IV NON-NEGOTIABLE. `src/retail_demand_forecasting/api/app.py:84` already implements `PredictionRequest` correctly (32 fields, `ge/le`, `extra="forbid"`), but `src/retail_demand_forecasting/dash_app/app.py:487` builds `payload` dict manually and POSTs without local Pydantic validation (no import of `PredictionRequest`), and `tests/test_api.py:84` `_valid_payload` only supplies 13 fields (incomplete) yet test passes due to missing required fields not enforced in test fixture's model (would fail if all 32 required). Dash should reuse same schema.

**Independent Test**: Send `POST /predict` with `{"lag_1": -1}` → `422`; with `{"extra_field": 1}` → `422`; with `{"requests": []}` batch → `422`; with batch 1001 → `422`. Verify Dash `generate_forecast` calls `PredictionRequest(**payload)` locally before HTTP, and `tests/test_api.py` covers all boundaries + float preservation.

**Acceptance Scenarios**:

1. **Given** payload missing `lag_14` or `day_of_week=8` or `snap_CA=2`, **When** `POST /predict`, **Then** FastAPI returns `422 Unprocessable Entity` with Pydantic error detail, never `200`.
2. **Given** Dash `_build_forecast` constructs `payload` for `i=1..forecast_days`, **When** it prepares each recursive step, **Then** it validates via `PredictionRequest(**payload).model_dump()` (or imports same schema) and `_prepare_X` aligns via `[FEATURE_COLS].fillna(0.0)` before `model.predict`.
3. **Given** batch request with 2 valid items, **When** `POST /predict/batch`, **Then** response `count==2` and each `prediction` is `float >=0.0` (e.g., `0.15`), not `int`.

---

### User Story 5 - Automated Distribution Drift Tests (Priority: P2)

As an MLOps operator, I need automated drift tests (KS/PSI per feature, zero-rate drift, serving payload validation) running on every PR/nightly, so we catch data and concept drift before model promotion.

**Why this priority**: P2 - Constitution Principle V is NON-NEGOTIABLE but currently unimplemented (no `tests/test_drift.py`). Without it, distribution shift in `HOBBIES_*` sparse categories goes undetected.

**Independent Test**: Create `tests/test_drift.py` with 3 test classes; run `pytest tests/test_drift.py -v` → produces `drift_report.json` artifact. In PR, intentionally perturb `rolling_mean_7` distribution (e.g., +50% shift) → test fails with `PSI > 0.2` / `p < 0.05`. Serving payload drift test validates live `PredictionRequest` batches vs training stats.

**Acceptance Scenarios**:

1. **Given** reference stats from `data/03_features/model_input.parquet` (or cached `_SALES_RAW` sample), **When** drift suite runs, **Then** it computes KS p-value and PSI per `FEATURE_COLS` and fails if `PSI>0.2` or `p<0.05` for any feature, writing `drift_report.json` with `{feature, psi, ks_p, status}`.
2. **Given** target drift check per `cat_id` (zero-rate = `mean(sales==0)`), **When** current zero-rate deviates >10pp vs baseline, **Then** test reports `target_drift` warning and blocks promotion on `main` (but allows `warning` on feature branches).
3. **Given** new feature added to `FEATURE_COLS` without drift assertion, **When** PR is opened, **Then** reviewer checklist flags missing drift coverage and CI fails until `test_drift.py` updated and reference artifact refreshed.

---

### Edge Cases

- **Training-serving parity with missing data**: When `data/01_raw` missing (CI/Docker), `load_features()` falls back to synthetic but still must pass drift threshold baselines (synthetic is excluded from drift gate via `CI_SYNTHETIC=1` flag).
- **Empty batch**: `POST /predict/batch` with `requests=[]` must return `422`, not `500` or empty `predictions`.
- **Model not loaded**: `GET /health` returns `model_loaded=false`; `POST /predict` returns `503` with detail `"Model not loaded — train pipeline or check MLflow/DVC artifacts"`.
- **Catalog path override**: If `conf/local/catalog.yml` overrides `calendar_raw.filepath`, hard-coded path detection must not trigger; only literal `data/` strings outside catalog loader count.
- **MLflow DB missing**: If `mlflow.db` absent, `lifespan` logs warning and still serves health; training still creates new DB via `sqlite:///mlflow.db`.
- **Pydantic extra fields**: Payload with `extra_field` or typo like `lag_1x` must be rejected (`extra="forbid"`), not silently ignored.
- **Drift small sample**: If reference sample <30 rows, KS test is skipped (insufficient power) and PSI uses quantile binning with fallback.
- **Float edge**: Prediction `0.001` must not be rounded to `0` or `0.00` in `PredictionResponse`; Dash table shows `".2f"` display but internal `df["predicted"]` retains full float.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST keep single source of feature logic in `src/retail_demand_forecasting/nodes/feature_engineering.py` (Spark) mirrored exactly by `src/train.py:_add_pandas_phase1_features` (pandas) using `shift(lag)`, `shift(1).rolling(w).mean/min/max/std`, and `rowsBetween(-w,-1)` semantics. Deviation MUST fail `tests/test_transforms.py`.
- **FR-002**: System MUST strictly read all dataset filepaths from `conf/base/catalog.yml` via Kedro `DataCatalog` or `yaml.safe_load(catalog.yml)` helper; direct `Path("data/01_raw/...")` or `Path("data/02_intermediate")` literals in `src/` (except in `src/data_prep.py` placeholder CI branch and `src/train.py` catalog-loaded fallback) MUST be removed/refactored. Dash `_load_sample_data()` MUST resolve via catalog or `PROJECT_ROOT` + catalog filepath, not hard-coded string.
- **FR-003**: System MUST read all hyperparameters (`lag_days`, `rolling_window_days`, `model_params.*`, `optuna.*`, `metrics.*`) from `conf/base/parameters.yml` injected as `params: dict`; no inline literals like `LAGS = [1,2,3...]` outside defaults fallback.
- **FR-004**: System MUST log every training run to `sqlite:///mlflow.db` via `mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)` with `log_param` for all `model_params` keys + `random_state`, `test_size`, `features` (JSON), `train_rows`, `test_rows` and `log_metric` for `wape`, `rmse`, `mae`, `mape`, `r2` plus flavor-specific `log_model` (`model` and `model_best`). (`src/retail_demand_forecasting/nodes/data_science.py:250`).
- **FR-005**: System MUST enforce evaluation gates: script `scripts/evaluate_gate.py` compares `metrics.json` vs `metrics_baseline.json` (or `data/06_metrics/model_metrics.json`) and fails CI if `wape > baseline_wape + 2.0` or `r2 < 0.80` or any tracked metric missing.
- **FR-006**: System MUST validate all serving inputs with Pydantic v2 `PredictionRequest` (`src/retail_demand_forecasting/api/app.py:84`) with `Field(ge,le)` bounds for all 32 `FEATURE_COLS` + `extra="forbid"` and `BatchPredictionRequest(min_length=1,max_length=1000)`. Dash `dash_app/app.py` MUST import and use `PredictionRequest` for local validation before HTTP call.
- **FR-007**: System MUST align serving DataFrame via `pd.DataFrame([row])[FEATURE_COLS].fillna(0.0)` (`src/retail_demand_forecasting/api/app.py:161` `_prepare_X`) and preserve `float` (`max(0.0,float(pred))`, no `int()`), including Dash `generate_forecast` recursive `recent.append(pred_float)`.
- **FR-008**: System MUST provide automated drift tests in `tests/test_drift.py` computing KS (`scipy.stats.ks_2samp`) and PSI per `FEATURE_COLS` vs reference snapshot (`data/03_features/model_input.parquet` or bundled sample), target zero-rate drift per `cat_id`/`dept_id`, and serving payload bounds validation. Output `drift_report.json` artifact; fail on `PSI>0.2` or `p<0.05` (configurable via `conf/base/parameters.yml` `drift:` section).
- **FR-009**: System MUST update `conf/base/parameters.yml` to add `drift:` config (`psi_threshold: 0.2`, `ks_alpha: 0.05`, `zero_rate_delta: 0.10`) and `dvc.yaml`/`pyproject.toml` to include drift gate in pipeline if applicable.
- **FR-010**: System MUST extend test coverage: `tests/test_api.py` must supply full 32-field `_valid_payload` and test `negative lag →422`, `extra_field→422`, `batch empty→422`, `batch 1001→422`, `float preservation`; `tests/test_transforms.py` must assert no `lead(` and `rowsBetween(-w,-1)`.
- **FR-011**: System MUST ensure `dvc repro --dry` and `pytest -q --cov=src` pass after refactor, with `ruff check src/ tests/` clean.

### Key Entities

- **Feature Columns (FEATURE_COLS)**: 32-column list in `src/retail_demand_forecasting/nodes/constants.py:6` (lags, rolling 7/28 mean/min/max/std, hierarchical store/dept/cat 7/28, calendar, cyclical sin/cos, SNAP/events/price). Single ordering authority for training and serving.
- **Catalog Entry**: `conf/base/catalog.yml` entry with `type: spark.SparkDataset`, `filepath`, `file_format`, `load_args`. Only allowed source of dataset paths.
- **MLflow Run**: Tracking entry in `sqlite:///mlflow.db` with params/metrics/artifacts (`model`, `model_best`), experiment `retail_demand_forecast` or `optuna_tuning`.
- **Evaluation Gate Report**: JSON produced by `scripts/evaluate_gate.py` with `{metric, baseline, current, delta, status, threshold}`; CI artifact.
- **Drift Report**: `drift_report.json` with per-feature `{psi, ks_p, status}` and target `{zero_rate_delta, status}`; nightly artifact.
- **PredictionRequest Payload**: Pydantic model instance with 32 validated fields; rejected if out-of-bounds or extra.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Zero `data/0` hard-coded literals in `src/` outside catalog loader (verified via `rg "data/0" src/ --glob '*.py'` returns 0 hits); `kedro catalog list` resolves all 6 datasets.
- **SC-002**: `pytest tests/test_transforms.py tests/test_training_serving_parity.py` passes 100% — training/serving feature values match ±1e-6 for all 32 cols, no `lead(` leakage.
- **SC-003**: Every `python src/train.py` run creates MLflow run with 15+ params and 5 metrics (`wape, rmse, mae, mape, r2`) and model artifact; `scripts/evaluate_gate.py` blocks PR when WAPE degrades >2pp (tested via synthetic bad metrics).
- **SC-004**: `POST /predict` with invalid payload returns `422` in <100ms; valid payload returns `200` with `prediction: float >=0.0` and `features` echo; batch limits enforced (empty→422, 1001→422, 1000→200).
- **SC-005**: `tests/test_drift.py` runs in <30s, produces `drift_report.json`, and correctly fails when synthetic distribution shift PSI>0.2; nightly run uploads artifact.
- **SC-006**: `pytest --cov=src` achieves ≥80% coverage on `api/`, `nodes/`; `ruff check src/ tests/` reports 0 errors; `dvc repro --dry` validates pipeline graph.

## Assumptions

- M5 raw data (`data/01_raw/*.csv`) is present locally for full `train.py` real feature build; CI fallback uses synthetic data but drift gates are skipped/marked `skipped` when synthetic flag set.
- `sqlite:///mlflow.db` at repo root is the tracking URI (via `conf/base/mlflow.yml` and `src/retail_demand_forecasting/nodes/constants.py:58`); no remote tracking server needed for this audit.
- `FEATURE_COLS` (32 cols) is frozen for this spec; adding a new feature is out of scope but drift test must handle future additions gracefully (detect missing drift assertion).
- `scipy` (for `ks_2samp`) is allowed as new dependency if not already pinned; otherwise implement PSI-only fallback.
- Dash `PROJECT_ROOT` env override pattern remains supported for Docker/Render deployments.
- `metrics_baseline.json` exists at repo root as gate baseline; if absent, gate creates it from first `metrics.json` and warns.

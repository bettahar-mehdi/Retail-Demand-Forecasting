# Implementation Plan: Audit and Enforce All 5 Constitution Principles

**Branch**: `001-audit-and-enforce` | **Date**: 2026-08-27 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-audit-and-enforce/spec.md` — audit and enforce 5 governing principles (zero skew, conf authority, MLflow gates, Pydantic validation, drift tests) across existing code.

## Summary

Enforce constitution v1.0.0 across the Retail Demand Forecasting MLOps codebase. Audit revealed 3 hard-coded `data/01_raw` violations bypassing `conf/base/catalog.yml`, incomplete `PredictionRequest` coverage in tests, missing evaluation gate, and absent drift suite. Approach: (1) introduce catalog-resolution helper (`src/retail_demand_forecasting/utils/catalog.py`) and refactor `dash_app/app.py:41`, `train.py:140`, `data_prep.py:13` to read paths from catalog; (2) add `scripts/evaluate_gate.py` blocking CI on WAPE regression >2pp / r2<0.80; (3) make Dash reuse `PredictionRequest` for local validation; (4) add `tests/test_training_serving_parity.py` and expand `tests/test_api.py` to 32-field payloads; (5) add `tests/test_drift.py` (KS + PSI per `FEATURE_COLS`, zero-rate drift) and `drift:` config in `parameters.yml`, plus `drift_report.json` artifact.

## Technical Context

**Language/Version**: Python 3.10–3.12 (project `requires-python >=3.10,<3.13`, `pyproject.toml:10`), pinned via `pyvenv.cfg`.

**Primary Dependencies**: Kedro 1.3.1 + `kedro-datasets[spark] 9.3.0` + `kedro-mlflow 2.0.3`, PySpark 4.1.1, pandas 2.3.3, numpy 2.2.6, MLflow 3.15.1 (`sqlite:///mlflow.db` via `conf/base/mlflow.yml:2`, `src/retail_demand_forecasting/nodes/constants.py:58`), LightGBM 4.6.0 (`objective=tweedie`, `tweedie_variance_power 1.1-1.5`), XGBoost 3.0.2, Optuna 4.4.0 (`TimeSeriesSplit` + `TPESampler(seed=42)`), FastAPI 0.136.1 + Uvicorn 0.46.0 + Pydantic 2.13.3 (`src/retail_demand_forecasting/api/app.py:84`), Dash 4.4.1 + Plotly 5.24.1, DVC 3.62.0 (`dvc.yaml:1`), scipy (new, for `ks_2samp` if absent → PSI fallback), PyYAML 6.0.1.

**Storage**: Local Parquet (`data/02_intermediate/sales_melted.parquet`, `data/03_features/model_input.parquet`), JSON metrics (`metrics.json`, `data/06_metrics/model_metrics.json`, `mlflow.db` + `mlruns/`), DVC-tracked `models/forecast_model.pkl`. Catalog `conf/base/catalog.yml:1` declares 6 datasets (3 `spark.SparkDataset`, 2 intermediate Parquet, 1 `JSONDataset`).

**Testing**: pytest 9.1.1 + pytest-cov 7.1.0 (`pyproject.toml:80` `testpaths=["tests"]`), ruff 0.15.22 (`line-length 100`, `target-version py310`), existing `tests/test_transforms.py`, `test_nodes.py`, `test_api.py`. New tests: `test_training_serving_parity.py`, `test_drift.py`. Run `pytest -q --cov=src`, `ruff check/format`, `dvc repro --dry`.

**Target Platform**: Linux (Render/Railway) + local Windows/macOS dev; Docker (`Dockerfile`) builds `gunicorn app:server --workers 2 --timeout 120` (`Procfile`), Dash `app.server`, API on 8000 via `start_all.py`.

**Project Type**: MLOps web-service + batch pipeline (single project, Kedro pipeline + FastAPI serving + Dash frontend).

**Performance Goals**: `POST /predict` p95 <100ms (in-memory `MODEL` load), batch 1000 in <2s; `test_drift.py` <30s; `dvc repro` data_prep <60s (synthetic) / <5min (real M5 1913×~30k); training 377k rows `WAPE 29.7%`, `R² 0.828` baseline must not regress.

**Constraints**: Principle I & III–V NON-NEGOTIABLE — no `lead()`/forward-fill, `rowsBetween(-w,-1)` only; `extra="forbid"`; MLflow gates block merge; hard-coded `data/` literals fail CI. `FEATURE_COLS` 32 frozen; `RANDOM_STATE=42` deterministic. `line-length 100`, no secrets in `conf/base/`.

**Scale/Scope**: 5 user stories (4×P1, 1×P2), 11 FRs, 6 SCs, ~30 tasks touching 10 files + 3 new files. Risk low—refactor, not new model architecture.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status Before | Enforcement in This Plan | Gate |
|-----------|---------------|--------------------------|------|
| **I. Zero Training-Serving Skew** | Spark `feature_engineering.py:57` correct; `train.py:_add_pandas_phase1_features:47` mirrors via `shift`+`rolling`—needs parity test; Dash `dash_app/app.py:422` recursive `recent` correct but untested | Add `tests/test_training_serving_parity.py` comparing Spark vs pandas vs `PredictionRequest` payload builder column-for-column 32 `FEATURE_COLS`; add assertion `lead(` absent; keep `max(0.0,float)` | **PASS** — no transform divergence introduced |
| **II. conf/ Authority** | **FAIL**: `dash_app/app.py:41` hard-codes `PROJECT_ROOT/"data/01_raw/..."`; `train.py:140` `Path("data/01_raw/...")`; `data_prep.py:13` direct Path | Introduce `utils/catalog.py:get_catalog_filepath(name)` reading `conf/base/catalog.yml` (+ `conf/local/` overlay) and refactor all three call sites; `rg "data/0" src/` must be 0 outside helper | **PASS** after refactor — helper is single exception |
| **III. MLflow + Gates** | `data_science.py:250` logs correctly; **missing** gate (`metrics.json` vs `metrics_baseline.json`) | Add `scripts/evaluate_gate.py` (`wape ≤ baseline+2.0`, `r2≥0.80`, all `metrics.track` present); CI job `evaluate-gate` fails PR; log `best_cv_wape`/`trial_{i}_wape` already | **PASS** — no extra tracking DB needed |
| **IV. Pydantic Validation** | `api/app.py:84` correct (32 fields, `ge/le`, `extra="forbid"`); **gap**: Dash does not reuse schema, `test_api.py:84` payload 13 fields | Dash `dash_app/app.py` imports `PredictionRequest` and validates `PredictionRequest(**payload).model_dump()` before POST; expand `test_api.py` to 32-field `_valid_payload` + 5 boundary tests | **PASS** — no API contract change |
| **V. Drift Tests** | **FAIL**: no `tests/test_drift.py` | Add `tests/test_drift.py` (KS `scipy.stats.ks_2samp` + PSI per feature, zero-rate per `cat_id`, serving bounds), `drift:` config in `parameters.yml:30`, `drift_report.json` artifact; graceful `CI_SYNTHETIC` skip | **PASS** |

Post-design re-check: no new datasets outside catalog, no hard-coded hyperparams, no new service dependencies. Complexity tracking not required (no violations).

## Project Structure

### Documentation (this feature)

```text
specs/001-audit-and-enforce/
├── spec.md              # Feature spec (5 stories, 11 FRs)
├── plan.md              # This file
├── research.md          # Phase 0 — decision log (generated below, inline)
├── data-model.md        # Phase 1 — entity definitions (inline)
├── quickstart.md        # Phase 1 — reproduce & verify (inline)
├── contracts/           # Phase 1 — OpenAPI fragment for /predict
│   └── predict.openapi.yaml
└── tasks.md             # Phase 2 — created by /speckit.tasks
```

### Source Code (repository root)

```text
conf/base/
├── catalog.yml          # 6 datasets — AUTHORITY (no code literals)
├── parameters.yml       # add drift: {psi_threshold, ks_alpha, zero_rate_delta}
└── mlflow.yml           # mlflow_tracking_uri: sqlite:///mlflow.db

src/retail_demand_forecasting/
├── utils/
│   └── catalog.py       # NEW: get_catalog_filepath(name) -> Path
├── nodes/
│   ├── constants.py     # FEATURE_COLS 32, RANDOM_STATE 42, MLFLOW_TRACKING_URI
│   ├── feature_engineering.py # Spark lags/rolling/calendar/hierarchical (authoritative)
│   ├── data_engineering.py    # unpivot + joins (unchanged)
│   └── data_science.py  # train_model / tune_and_train + mlflow logging
├── pipelines/
│   ├── data_engineering/pipeline.py
│   ├── feature_engineering/pipeline.py
│   └── data_science/pipeline.py
├── api/app.py           # PredictionRequest (32) + _prepare_X[FEATURE_COLS].fillna(0)
└── dash_app/app.py      # REFACTORED: _load_sample_data via catalog helper + PredictionRequest validation

src/
├── train.py             # REFACTORED: _build_real_features_pandas resolves via catalog helper
└── data_prep.py         # REFACTORED: uses catalog helper for RAW_EXISTS check

scripts/
└── evaluate_gate.py     # NEW: compare metrics.json vs metrics_baseline.json — exit 1 on gate fail

tests/
├── test_transforms.py           # existing — assert rowsBetween(-w,-1), no lead
├── test_training_serving_parity.py # NEW: Spark vs pandas vs API parity ±1e-6
├── test_api.py                  # EXPANDED: 32-field payload, 5 boundary tests
├── test_nodes.py                # existing — metrics & split
└── test_drift.py                # NEW: KS/PSI + zero-rate + payload drift → drift_report.json

dvc.yaml                # train_and_tune metrics: metrics.json + data/06_metrics/model_metrics.json
pyproject.toml          # add scipy if missing, ruff/pytest config
```

**Structure Decision**: Single project (DEFAULT) — Kedro pipeline + serving in one repo. No backend/frontend split; `src/retail_demand_forecasting/` is package root (`pyproject.toml:53` `where=["src"]`), `tests/` at root. New `utils/catalog.py` keeps catalog resolution DRY for both training and serving; `scripts/` for CI gates.

## Phase 0: Research (Decisions)

### R0 — Catalog helper resolution

**Decision**: `utils/catalog.py` reads `conf/base/catalog.yml` via `yaml.safe_load` + merges `conf/local/catalog.yml` if present (Kedro overlay). Exposes `get_catalog_filepath(name: str) -> Path` returning `Path(repo_root / entry["filepath"])` resolved.

**Alternatives**: Kedro `DataCatalog.from_config` (requires Kedro context, heavy for Dash); direct `open(catalog.yml)` per call site (DRY violation). Chosen helper is lightweight, no Kedro dependency in Dash.

**Risk**: `PROJECT_ROOT` env override for Dash must still work — helper resolves `Path(__file__).parents[3]` as repo root fallback, then `catalog.yml`.

### R1 — Hard-coded literal detection

**Decision**: `rg -n "data/0[123]" src/ --glob '*.py'` fails CI if match outside `utils/catalog.py` and `src/data_prep.py` placeholder branch (`if not RAW_EXISTS: create .gitkeep`). Allowlist is explicit comment `# catalog-allowlist: util`.

**Alternative**: Pre-commit hook — deferred, CI check sufficient.

### R2 — Evaluation gate implementation

**Decision**: `scripts/evaluate_gate.py` loads `metrics.json` (required) and `metrics_baseline.json` (or `data/06_metrics/model_metrics.json` if baseline missing, warning). Thresholds from `parameters.yml` `metrics:` and `drift:` or defaults `wape_delta=2.0`, `r2_min=0.80`. Outputs JSON report `{metric, baseline, current, delta, status}` and exits 1 if any gate fails. CI runs after `python src/train.py`.

**Alternative**: MLflow-based gate (query `MlflowClient.search_runs`) — more robust but requires DB; file gate is simpler and matches DVC metrics.

### R3 — Dash Pydantic reuse

**Decision**: `dash_app/app.py` imports `from retail_demand_forecasting.api.app import PredictionRequest` and calls `PredictionRequest(**payload).model_dump(mode="json")` before `requests.post`. Keeps float types; validation error → logs and skips that forecast day (fallback `rolling_mean_7`). No circular import: `api.app` does not import `dash_app`.

**Alternative**: Duplicate schema in Dash — rejected (violates single source).

### R4 — Drift test design

**Decision**: `tests/test_drift.py` implements:
- PSI: 10 quantile bins from reference, smoothing `0.5` counts, `psi = sum((a-b)*log(a/b))`, fail if `> psi_threshold 0.2`.
- KS: `scipy.stats.ks_2samp` if `scipy` installed else skip KS (PSI only), `p<0.05` fails; skip if `n<30`.
- Reference: `data/03_features/model_input.parquet` if exists else cached small sample `tests/fixtures/model_input_sample.parquet` (1000 rows) or synthetic fallback (marked skipped when `CI_SYNTHETIC=1`).
- Target drift: zero-rate `mean(sales==0)` global + per `cat_id` (if `cat_id` col present), fail if `abs(delta)>0.10`.

**Alternative**: `evidently` library — heavy, adds dependency; manual PSI/KS suffices.

### R5 — Parity test

**Decision**: `tests/test_training_serving_parity.py` creates small pandas DF (100 rows, 2 store×2 item series), runs Spark `create_features` if Spark session available else simulates via pandas helper, then feeds same `day_id` slice to `PredictionRequest` payload builder and asserts all 32 values equal ±1e-6. Also asserts `feature_engineering.py` source contains `rowsBetween(-` and not `lead(`.

## Phase 1: Design

### Data Model (entities)

- **CatalogEntry**: `name: str` (e.g., `calendar_raw`), `filepath: Path` (e.g., `data/01_raw/calendar.csv`), `type: str` (`spark.SparkDataset`), `file_format: str`, `load_args: dict`. Authority in `conf/base/catalog.yml`.
- **FeatureRow**: 32 cols from `FEATURE_COLS` + keys `day_id, date, store_id, item_id, sales, sell_price, ...`. Produced by `feature_engineering.create_features` (leakage-free) and aligned in `api._prepare_X`. Stored as Parquet `model_input_features`.
- **MLflowRun**: `run_name: str` (`retail_demand_forecast`|`optuna_tuning`), `params: dict` (15+), `metrics: dict` (`wape, rmse, mae, mape, r2, best_cv_wape`), `artifact_uri: str` (`model`, `model_best`). Logged to `sqlite:///mlflow.db`.
- **EvaluationGateReport**: `baseline_path: Path`, `current_path: Path`, `checks: list[{metric, baseline, current, delta, threshold, status: pass|fail}]`, `overall: pass|fail`. Produced by `scripts/evaluate_gate.py`.
- **DriftReport**: `generated_at: isoformat`, `reference: Path`, `psi_threshold: float`, `ks_alpha: float`, `features: list[{feature, psi, ks_p, status}]`, `target: {global_zero_rate_delta, per_cat: [{cat_id, delta}], status}`. Written to `drift_report.json`.
- **PredictionRequest**: Pydantic v2, 32 fields with `Field(ge,le)`, `model_config={"extra":"forbid"}` (`src/retail_demand_forecasting/api/app.py:84`). Single validation boundary.

### Contracts

#### API Contract fragment (`contracts/predict.openapi.yaml`)

```yaml
openapi: 3.1.0
paths:
  /predict:
    post:
      requestBody:
        content:
          application/json:
            schema: { $ref: '#/components/schemas/PredictionRequest' }
      responses:
        '200': { description: float prediction }
        '422': { description: validation error }
        '503': { description: model not loaded }
  /predict/batch:
    post:
      requestBody:
        content:
          application/json:
            schema: { $ref: '#/components/schemas/BatchPredictionRequest' }
      responses:
        '200': { description: batch predictions }
        '422': { description: batch size 0 or 1001 or validation }
components:
  schemas:
    PredictionRequest:
      type: object
      additionalProperties: false
      required: [lag_1, lag_2, lag_3, lag_7, lag_14, lag_21, lag_28, rolling_mean_7, ... , sell_price]
      properties:
        lag_1: { type: number, minimum: 0, maximum: 10000 }
        day_of_week: { type: integer, minimum: 1, maximum: 7 }
        day_of_week_sin: { type: number, minimum: -1, maximum: 1 }
        # ... 32 fields as in api/app.py:84
```

No new endpoints; existing `GET /health` unchanged.

### Quickstart (reproduce)

```bash
# 1. Setup
python -m venv .venv && .venv\Scripts\activate
pip install -e .  # or pip install -r requirements.txt
# 2. Verify catalog authority (should be 0)
rg -n "data/0" src/ --glob '*.py'  # only utils/catalog.py allowed
kedro catalog list
# 3. Train baseline (synthetic if raw missing)
.venv\Scripts\python src/train.py --no-tune
cat metrics.json
# 4. Gate
.venv\Scripts\python scripts/evaluate_gate.py --baseline metrics_baseline.json --current metrics.json
# 5. API validation
.venv\Scripts\python -m uvicorn retail_demand_forecasting.api.app:app --port 8000 &
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @tests/fixtures/valid_payload.json
# invalid → 422
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"lag_1":-1,...}'
# 6. Drift
.venv\Scripts\pytest tests/test_drift.py -v --tb=short
cat drift_report.json
# 7. Full quality
ruff check src/ tests/ && ruff format --check src/ tests/
.venv\Scripts\pytest -q --cov=src
dvc repro --dry
```

## Complexity Tracking

> No constitution violations requiring justification. `utils/catalog.py` is the single allowed exception for `data/` literals (explicit allowlist). All other logic stays within existing pipelines.

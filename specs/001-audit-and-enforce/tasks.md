---
description: "Task list for 001-audit-and-enforce — enforce 5 constitution principles"
---

# Tasks: Audit and Enforce All 5 Constitution Principles

**Input**: Design documents from `/specs/001-audit-and-enforce/` (spec.md, plan.md)
**Prerequisites**: plan.md (done), spec.md (done)
**Branch**: `001-audit-and-enforce`

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Parallelizable (different files, no dependencies)
- **[Story]**: US1–US5 mapping to spec user stories
- All paths are absolute from repo root.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Validate branch, dependencies, and baseline metrics before changes.

- [ ] T001 Verify branch `001-audit-and-enforce` and feature.json `specs/001-audit-and-enforce` alignment
- [ ] T002 [P] Install and verify dependencies: `.venv\Scripts\python -m pip install -e .` + check `scipy` availability (`python -c "import scipy.stats"` → fallback if missing)
- [ ] T003 [P] Create `specs/001-audit-and-enforce/contracts/predict.openapi.yaml` fragment per plan.md contracts section (no logic change)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared config and CI gate scaffolding that all user stories depend on.

**⚠️ CRITICAL**: No user-story work until this phase complete.

- [ ] T004 [P] Add `drift:` config to `conf/base/parameters.yml` — `psi_threshold: 0.2`, `ks_alpha: 0.05`, `zero_rate_delta: 0.10` (per FR-009)
- [ ] T005 Create `src/retail_demand_forecasting/utils/catalog.py` with `get_catalog_filepath(name: str) -> Path` reading `conf/base/catalog.yml` + `conf/local/catalog.yml` overlay, resolving via repo root (`Path(__file__).parents[3]`)
- [ ] T006 [P] Add `scipy` to `pyproject.toml:dependencies` and `requirements.txt` if not present (needed for KS); fallback PSI-only logic if install fails
- [ ] T007 Verify foundational: `ruff check src/ tests/` clean, `python -m kedro catalog list` resolves 6 datasets

**Checkpoint**: Foundation ready — US1–US5 can proceed (in parallel if staffed, else P1→P2 order).

---

## Phase 3: User Story 1 — Zero Training-Serving Skew Enforcement (Priority: P1) 🎯 MVP

**Goal**: Prove Spark `feature_engineering.py` and pandas `train.py` shim and serving payload produce identical leakage-free features (32 `FEATURE_COLS`).

**Independent Test**: `pytest tests/test_transforms.py tests/test_training_serving_parity.py -v` — 32 cols equal ±1e-6, no `lead(`, `rowsBetween(-w,-1)` only, `POST /predict` float `0.15` preserved.

### Tests for US1 (write FIRST, ensure FAIL)

- [ ] T008 [P] [US1] Create `tests/test_training_serving_parity.py` — generate 100-row pandas DF (2 store×2 item series), compare Spark `create_features` vs pandas `_add_pandas_phase1_features` vs `PredictionRequest` payload builder values; assert `lead(` absent in `feature_engineering.py` source
- [ ] T009 [P] [US1] Expand `tests/test_transforms.py` to assert `rowsBetween(-` present and `lead(` not present (already partially) — add explicit `test_noLead` and `test_trailingWindow`

### Implementation for US1

- [ ] T010 [US1] Audit and align `src/train.py:_add_pandas_phase1_features` semantics to exactly `shift(lag)` / `shift(1).rolling(w)` + hierarchical `groupby(...).shift(1).rolling` matching `src/retail_demand_forecasting/nodes/feature_engineering.py:57` (verify 7/28 windows, std `fillna(0)`, `dropna` cutoff 28)
- [ ] T011 [US1] Verify `src/retail_demand_forecasting/api/app.py:161` `_prepare_X` uses `[FEATURE_COLS].fillna(0.0)` and `max(0.0,float)` float preservation; verify `dash_app/app.py:422` `recent.append(pred_float)` keeps float (no `int()`)

**Checkpoint**: US1 independently functional — skew tests green.

---

## Phase 4: User Story 2 — Strict conf/ Authority for Dataset Paths (Priority: P1)

**Goal**: Zero hard-coded `data/01_raw` literals outside `utils/catalog.py`.

**Independent Test**: `rg -n "data/0" src/ --glob '*.py'` → 0 hits outside allowlist; `kedro catalog list` and `dvc repro --dry` succeed.

### Tests for US2

- [ ] T012 [P] [US2] Add CI check `scripts/check_catalog_authority.py` (or inline `rg` assertion in `tests/test_catalog_authority.py`) that scans `src/` for `data/0` literals outside `utils/catalog.py` and fails

### Implementation for US2

- [ ] T013 [US2] Refactor `src/retail_demand_forecasting/dash_app/app.py:41` `_load_sample_data()` to resolve `sales_train_validation.csv`/`calendar.csv` via `get_catalog_filepath("sales_train_raw")` / `get_catalog_filepath("calendar_raw")` instead of `PROJECT_ROOT/"data/01_raw/..."`
- [ ] T014 [US2] Refactor `src/train.py:140` `_build_real_features_pandas()` to resolve `raw_sales/raw_cal/raw_prices` via `get_catalog_filepath` (fallback to hard-coded only if catalog file missing with warning)
- [ ] T015 [US2] Refactor `src/data_prep.py:13` `RAW_EXISTS` check to use `get_catalog_filepath` paths; keep placeholder `data/processed/.gitkeep` branch allowed for CI with comment `# catalog-allowlist: placeholder`

**Checkpoint**: US1+US2 both green — authority enforced.

---

## Phase 5: User Story 3 — MLflow Parameter/Metric Logging with Evaluation Gates (Priority: P1)

**Goal**: Every training run logged; `WAPE +2pp` / `r2<0.80` blocks promotion.

**Independent Test**: Run `python src/train.py --no-tune` → `mlflow.db` run with 15 params + 5 metrics + artifact; run `python scripts/evaluate_gate.py --baseline metrics_baseline.json --current metrics.json` → exit 1 when `wape=35` vs `29.7`.

### Tests for US3

- [ ] T016 [P] [US3] Expand `tests/test_nodes.py::TestTrainModel` to assert MLflow run exists with required params/metrics (mock `MLFLOW_TRACKING_URI` to temp DB)

### Implementation for US3

- [ ] T017 [US3] Create `scripts/evaluate_gate.py` — loads `metrics.json` + `metrics_baseline.json` (fallback `data/06_metrics/model_metrics.json`), thresholds `wape_delta=2.0`, `r2_min=0.80` (from `parameters.yml` if present), prints gate report JSON, exits 1 on fail; handles missing baseline by warning+pass
- [ ] T018 [US3] Add CI job snippet to `.github/workflows/ci_cd.yml` or document manual gate: `python scripts/evaluate_gate.py` after `python src/train.py`; ensure `dvc.yaml:metrics` still tracks `metrics.json`
- [ ] T019 [US3] Verify `src/retail_demand_forecasting/nodes/data_science.py:250` logs all `model_params` + `features` JSON + `train_rows/test_rows` and flavor `log_model` (audit only, fix if gap)

**Checkpoint**: US3 gate blocks regression.

---

## Phase 6: User Story 4 — Pydantic Input Validation for Dash/API Serving (Priority: P1)

**Goal**: All serving inputs validated via `PredictionRequest(extra="forbid")`, batch 1–1000, float preserved.

**Independent Test**: `curl POST /predict` with `lag_1=-1`→422, `extra_field→422`, `requests=[]→422`, `requests` 1001→422, valid 32-field→200 with `prediction: float`.

### Tests for US4

- [ ] T020 [P] [US4] Rewrite `tests/test_api.py:84` `_valid_payload()` to supply full 32-field valid payload (all lags, rolling, hierarchical, calendar, cyclical, SNAP/events/price) and add tests: `test_negative_lag_422`, `test_extra_field_422`, `test_batch_1001_422`, `test_float_preservation` (mock model returns 0.15, assert `prediction==0.15`)
- [ ] T021 [US4] Add `tests/test_dash_validation.py` or extend `test_api` to import `PredictionRequest` and validate Dash payload builder

### Implementation for US4

- [ ] T022 [US4] Update `src/retail_demand_forecasting/dash_app/app.py:487` `_build_forecast` to `from retail_demand_forecasting.api.app import PredictionRequest` and validate `PredictionRequest(**payload).model_dump()` before `requests.post`; on `ValidationError` log and skip
- [ ] T023 [US4] Verify `src/retail_demand_forecasting/api/app.py:84` bounds unchanged and `_prepare_X` alignment still `fillna(0.0)`; add comment linking to constitution IV

**Checkpoint**: US4 validation enforced end-to-end.

---

## Phase 7: User Story 5 — Automated Distribution Drift Tests (Priority: P2)

**Goal**: KS/PSI per feature + zero-rate drift produce `drift_report.json` on every PR/nightly.

**Independent Test**: `pytest tests/test_drift.py -v` → `drift_report.json` with 32 features; perturb `rolling_mean_7` +50% → PSI>0.2 fails.

### Tests for US5 (core deliverable)

- [ ] T024 [P] [US5] Create `tests/test_drift.py` with helpers `_psi(a,b, bins=10)` and `_ks_p(a,b)` (scipy or skip), classes `TestFeatureDrift` (PSI/KS per `FEATURE_COLS` vs reference `data/03_features/model_input.parquet` or `tests/fixtures/model_input_sample.parquet`), `TestTargetDrift` (zero-rate global + per `cat_id`), `TestServingPayloadDrift` (bounds check via `PredictionRequest` FieldInfo)
- [ ] T025 [US5] Add fixture `tests/fixtures/model_input_sample.parquet` (or generate synthetic 1000-row reference in test setup) and ensure `CI_SYNTHETIC=1` marks drift as skipped not failed

### Implementation for US5

- [ ] T026 [US5] Implement drift thresholds reading `conf/base/parameters.yml` `drift:` (`psi_threshold`, `ks_alpha`, `zero_rate_delta`) with defaults
- [ ] T027 [US5] Ensure `tests/test_drift.py` writes `drift_report.json` at repo root with `{feature, psi, ks_p, status}` and exits non-zero on threshold breach (fail on `main`, warn on feature branch via `GITHUB_REF` check — import `os`)

**Checkpoint**: US5 nightly drift green, PR gate integrated.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Quality gates and docs.

- [ ] T028 [P] Run `ruff check src/ tests/ --fix` and `ruff format src/ tests/` — ensure 0 errors
- [ ] T029 [P] Run `pytest -q --cov=src --cov-report=term-missing` — achieve ≥80% on `api/`, `nodes/`, fix gaps
- [ ] T030 [P] Run `dvc repro --dry` and `kedro catalog list` — validate pipeline graph
- [ ] T031 [P] Update `docs/` or `README.md` drift/gate section if needed; add `contracts/predict.openapi.yaml` reference
- [ ] T032 Run `quickstart.md` validation (plan.md quickstart steps) end-to-end
- [ ] T033 Final `scripts/evaluate_gate.py` manual fail test: write temp `metrics.json` with `wape=35` and confirm exit 1

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (T001–T003)**: No dependencies — start immediately
- **Foundational (T004–T007)**: Depends on Setup — BLOCKS all US
- **User Stories (US1–US5)**: All depend on Foundational; can run in parallel by different owners, else sequential P1 (US1→US2→US3→US4) → P2 (US5)
- **Polish (T028–T033)**: Depends on all desired US complete

### Within Each US

- Tests FIRST (marked [P] parallel) → ensure FAIL before impl
- Implementation after tests
- Verify story checkpoint independently

### Parallel Opportunities (marked [P])

- T002, T003 parallel
- T004, T006 parallel
- T008, T009 parallel
- T012 parallel file
- T016 alone
- T020, T021 parallel (if split)
- T024, T025 parallel

---

## Implementation Strategy

### MVP First (US1 only)
1. T001–T007 Foundation
2. T008–T011 US1
3. Validate `pytest tests/test_training_serving_parity.py`

### Incremental
Add US2 → US3 → US4 → US5, each independently testable.

### Task Count
33 tasks (7 foundational + 19 US + 7 polish), 10 parallelizable.

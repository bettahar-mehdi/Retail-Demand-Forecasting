# Retail Demand Forecasting

Production-grade MLOps pipeline for the **Walmart M5 Forecasting** dataset, built with **Kedro**, **PySpark**, **MLflow**, and **FastAPI**.

---

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Raw CSVs   │───▶│  Data Engineering │───▶│    Feature Eng.   │───▶│  ML Training    │
│ (Kaggle M5) │    │  unpivot + join   │    │  lag, rolling,    │    │  RandomForest   │
└─────────────┘    └──────────────────┘    │  calendar, SNAP   │    │  + MLflow       │
                                           └──────────────────┘    └────────┬────────┘
                                                                           │
                                              ┌────────────────────────────┘
                                              ▼
                                     ┌─────────────────┐
                                     │   FastAPI        │
                                     │  /predict        │
                                     │  /predict/batch  │
                                     │  /health         │
                                     └─────────────────┘
```

## Tech Stack

| Layer | Tool |
|-------|------|
| Orchestration | Kedro 1.3 |
| Distributed compute | PySpark 4.1 |
| Experiment tracking | MLflow (SQLite) |
| Serving | FastAPI + Uvicorn |
| ML | scikit-learn RandomForest |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data (M5 dataset)
#    Place calendar.csv, sell_prices.csv, sales_train_validation.csv
#    into data/01_raw/

# 3. Run the full pipeline
kedro run

# 4. Start the API + Dashboard
python start_all.py
#    API:     http://localhost:8000
#    Dashboard: http://localhost:8050
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health check |
| `POST` | `/predict` | Single-item demand forecast |
| `POST` | `/predict/batch` | Batch demand forecast |

### Example Request

```json
POST /predict
{
  "lag_7": 5,
  "lag_28": 3,
  "rolling_mean_7": 4.5,
  "rolling_mean_28": 3.2,
  "day_of_week": 6,
  "month": 4,
  "year": 2016,
  "snap_CA": 1,
  "snap_TX": 0,
  "snap_WI": 0,
  "has_event_1": 0,
  "has_event_2": 0,
  "sell_price": 1.25
}
```

## Features

| Feature | Description |
|---------|-------------|
| `lag_7`, `lag_28` | Sales from 7/28 days ago |
| `rolling_mean_7`, `rolling_mean_28` | 7/28-day trailing average |
| `day_of_week`, `month`, `year` | Calendar components |
| `snap_CA/TX/WI` | SNAP benefit flags |
| `has_event_1/2` | Holiday / event indicator |
| `sell_price` | Unit sell price |

## Configuration

All parameters live in `conf/base/parameters.yml`. Override per-environment via `conf/local/`.

```yaml
feature_engineering:
  lag_days: [7, 28]
  rolling_window_days: [7, 28]

model_params:
  max_depth: 5
  num_trees: 50
  test_size: 0.2
  random_state: 42
```

## Project Structure

```
├── conf/                          # Kedro configuration
│   ├── base/
│   │   ├── catalog.yml            # Dataset definitions
│   │   ├── parameters.yml         # Feature & model params
│   │   └── mlflow.yml             # MLflow tracking URI
│   └── logging.yml
├── src/retail_demand_forecasting/
│   ├── nodes/
│   │   ├── constants.py           # Shared feature columns & defaults
│   │   ├── data_engineering.py    # Unpivot + join logic
│   │   ├── feature_engineering.py # PySpark Window functions
│   │   └── data_science.py        # Model training + MLflow
│   ├── pipelines/
│   │   ├── data_engineering/
│   │   ├── feature_engineering/
│   │   └── data_science/
│   ├── api/app.py                 # FastAPI serving layer
│   ├── dash_app/app.py            # Dash interactive dashboard
│   ├── pipeline_registry.py       # Pipeline wiring
│   └── settings.py                # Kedro config loader
├── notebooks/                     # Phase-by-phase verification
├── tests/                         # Unit tests (6 passing)
├── start_all.py                   # Launch API + Dashboard
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

## License

MIT

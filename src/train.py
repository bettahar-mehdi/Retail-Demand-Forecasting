"""Train shim for DVC stage — `python src/train.py`.

Reads feature data from catalog or parquet fallback, or builds real M5
features via pandas if raw CSVs exist (no Spark needed), runs baseline or
Optuna-tuned training, saves model to models/forecast_model.pkl and metrics.
"""

import json
import sys
from pathlib import Path

# Ensure src is on path when called as `python src/train.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import yaml

from retail_demand_forecasting.nodes.constants import RANDOM_STATE
from retail_demand_forecasting.nodes.data_science import (
    seed_everything,
    train_model,
    tune_and_train,
)


def load_params() -> dict:
    p = Path("conf/base/parameters.yml")
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


def _add_pandas_phase1_features(df: pd.DataFrame, group_cols: list | None = None) -> pd.DataFrame:
    """Apply Phase-1 transformations to a sorted pandas DataFrame (leakage-free).

    Adds lags [1,2,3,7,14,21,28], rolling 7/28 mean/min/max/std shifted by 1,
    calendar + cyclical sin/cos + is_weekend, plus hierarchical store/dept/cat
    rolling means for sparse items. If group_cols provided,
    computes per-group via groupby; otherwise assumes single series.
    """
    import numpy as np

    LAGS = [1, 2, 3, 7, 14, 21, 28]
    WINDOWS = [7, 28]

    def _apply(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("day_id")
        for lag_n in LAGS:
            g[f"lag_{lag_n}"] = g["sales"].shift(lag_n)
        for win in WINDOWS:
            shifted = g["sales"].shift(1)
            g[f"rolling_mean_{win}"] = shifted.rolling(win, min_periods=1).mean()
            g[f"rolling_min_{win}"] = shifted.rolling(win, min_periods=1).min()
            g[f"rolling_max_{win}"] = shifted.rolling(win, min_periods=1).max()
            g[f"rolling_std_{win}"] = shifted.rolling(win, min_periods=1).std().fillna(0)
        # Calendar
        g["day_of_week"] = g["date"].dt.dayofweek + 1
        g["day_of_month"] = g["date"].dt.day
        g["month"] = g["date"].dt.month
        g["year"] = g["date"].dt.year
        g["is_weekend"] = g["day_of_week"].isin([6, 7]).astype(int)
        # Cyclical
        g["day_of_week_sin"] = np.sin(2 * np.pi * g["day_of_week"] / 7)
        g["day_of_week_cos"] = np.cos(2 * np.pi * g["day_of_week"] / 7)
        g["day_of_month_sin"] = np.sin(2 * np.pi * g["day_of_month"] / 31)
        g["day_of_month_cos"] = np.cos(2 * np.pi * g["day_of_month"] / 31)
        g["month_sin"] = np.sin(2 * np.pi * g["month"] / 12)
        g["month_cos"] = np.cos(2 * np.pi * g["month"] / 12)
        return g

    if group_cols:
        df = df.groupby(group_cols, group_keys=False).apply(_apply)
        # Drop rows where max lag not available (first 28 days per group)
        df = df.dropna(subset=[f"lag_{n}" for n in LAGS] + [f"rolling_mean_{w}" for w in WINDOWS])
    else:
        df = _apply(df)
        df = df.dropna(subset=[f"lag_{n}" for n in LAGS] + [f"rolling_mean_{w}" for w in WINDOWS])

    # Hierarchical store/dept/cat rolling means (sparse HOBBIES_1_003 inherits store trend)
    # Computed as shifted rolling per hierarchy, filled 0 for first days
    for win in WINDOWS:
        if "store_id" in df.columns:
            # per store
            df[f"store_rolling_mean_{win}"] = (
                df.sort_values(["store_id", "day_id"])
                .groupby("store_id", group_keys=False)["sales"]
                .apply(lambda x, win=win: x.shift(1).rolling(win, min_periods=1).mean())
            )
            df[f"store_rolling_mean_{win}"] = df[f"store_rolling_mean_{win}"].fillna(0)
        else:
            df[f"store_rolling_mean_{win}"] = df[f"rolling_mean_{win}"]
        if "dept_id" in df.columns:
            df[f"dept_rolling_mean_{win}"] = (
                df.sort_values(["dept_id", "day_id"])
                .groupby("dept_id", group_keys=False)["sales"]
                .apply(lambda x, win=win: x.shift(1).rolling(win, min_periods=1).mean())
            )
            df[f"dept_rolling_mean_{win}"] = df[f"dept_rolling_mean_{win}"].fillna(0)
        else:
            df[f"dept_rolling_mean_{win}"] = df[f"store_rolling_mean_{win}"]
        if "cat_id" in df.columns:
            df[f"cat_rolling_mean_{win}"] = (
                df.sort_values(["cat_id", "day_id"])
                .groupby("cat_id", group_keys=False)["sales"]
                .apply(lambda x, win=win: x.shift(1).rolling(win, min_periods=1).mean())
            )
            df[f"cat_rolling_mean_{win}"] = df[f"cat_rolling_mean_{win}"].fillna(0)
        else:
            df[f"cat_rolling_mean_{win}"] = df[f"store_rolling_mean_{win}"]
    return df


def _synthetic_df(n: int = 500) -> pd.DataFrame:
    """Deterministic synthetic with weekly seasonality (Phase-1 features)."""
    print("Using improved synthetic fallback (Phase-1 weekly seasonality)")
    import numpy as np

    rng = np.random.default_rng(RANDOM_STATE)
    dates = pd.date_range("2016-01-01", periods=n, freq="D")
    base = 5
    weekly = 3 * np.sin(2 * np.pi * dates.dayofweek / 7)
    sales = np.maximum(0, base + weekly + rng.normal(0, 1.5, n)).astype(int)
    df = pd.DataFrame({"date": dates, "day_id": range(1, n + 1), "sales": sales})
    df["item_id"] = "FOODS_1_001"
    df["store_id"] = "CA_1"
    df["snap_CA"] = (df["date"].dt.dayofweek == 5).astype(int)
    df["snap_TX"] = 0
    df["snap_WI"] = 0
    df["has_event_1"] = (df["date"].dt.month == 12).astype(int)
    df["has_event_2"] = 0
    df["sell_price"] = rng.uniform(1, 5, n)
    # Apply full Phase-1 feature set leakage-free
    df = _add_pandas_phase1_features(df, group_cols=None)
    return df


def _resolve_catalog_path(name: str, fallback: str) -> Path:
    """Resolve dataset path via catalog authority (Principle II), fallback to literal."""
    try:
        from retail_demand_forecasting.utils.catalog import get_catalog_filepath

        return get_catalog_filepath(name)
    except Exception:
        return Path(fallback)


def _build_real_features_pandas(sample_rows: int = 200) -> pd.DataFrame | None:
    """Build real M5 features via pandas without Spark. Returns None if raw missing."""
    raw_sales = _resolve_catalog_path("sales_train_raw", "data/01_raw/sales_train_validation.csv")
    raw_cal = _resolve_catalog_path("calendar_raw", "data/01_raw/calendar.csv")
    raw_prices = _resolve_catalog_path("sell_prices_raw", "data/01_raw/sell_prices.csv")
    if not (raw_sales.exists() and raw_cal.exists()):
        return None
    try:
        print(f"Building real features from raw CSVs (sample {sample_rows} items)...")

        # 1. Calendar
        cal = pd.read_csv(raw_cal)
        # cal has columns: date, wm_yr_wk, weekday, wday, month, year, d, event_name_1, event_type_1, event_name_2, event_type_2, snap_CA/TX/WI
        # Map d -> day_id
        cal["day_id"] = cal["d"].str.replace("d_", "").astype(int)
        cal["date"] = pd.to_datetime(cal["date"])
        cal["has_event_1"] = cal["event_name_1"].notna().astype(int)
        cal["has_event_2"] = cal["event_name_2"].notna().astype(int)
        # day_of_week: wday is 1=Sat? Use date dt.dayofweek+1 for ISO (Mon=1)
        cal["day_of_week"] = cal["date"].dt.dayofweek + 1

        # 2. Sales - sample top + random to capture variance
        # Read only needed columns: id, item_id, store_id, plus d_ columns
        # For memory, read full then sample
        sales = pd.read_csv(raw_sales)
        day_cols = [c for c in sales.columns if c.startswith("d_")]
        # Sample: take top 100 by total and 100 random low
        sales["total"] = sales[day_cols].sum(axis=1)
        top = sales.nlargest(100, "total")
        rest = sales[~sales.index.isin(top.index)].sample(
            n=min(100, len(sales) - 100), random_state=RANDOM_STATE
        )
        sampled = pd.concat([top, rest]).drop(columns=["total"])
        print(f"Sampled {len(sampled)} series, melting {len(sampled) * len(day_cols)} rows...")
        # Melt
        melted = sampled.melt(
            id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
            value_vars=day_cols,
            var_name="d",
            value_name="sales",
        )
        melted["day_id"] = melted["d"].str.replace("d_", "").astype(int)
        melted = melted.drop(columns=["d"])
        # Join calendar
        merged = melted.merge(cal, on="day_id", how="left", suffixes=("", "_cal"))
        # Join sell_prices on store_id, item_id, wm_yr_wk
        if raw_prices.exists():
            prices = pd.read_csv(raw_prices)
            merged = merged.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
            merged["sell_price"] = (
                merged["sell_price"]
                .fillna(merged.groupby("item_id")["sell_price"].transform("median"))
                .fillna(2.0)
            )
        else:
            merged["sell_price"] = 1.25
        # Sort for time-series features
        merged = merged.sort_values(["store_id", "item_id", "day_id"])
        # 3. Lag / rolling / calendar via Phase-1 helper (leakage-free)
        merged = merged.sort_values(["store_id", "item_id", "day_id"])
        # Use shared Phase-1 pandas logic per group
        merged = _add_pandas_phase1_features(merged, group_cols=["store_id", "item_id"])
        # Select final columns
        # Ensure snap columns exist
        for col in ["snap_CA", "snap_TX", "snap_WI"]:
            if col not in merged.columns:
                merged[col] = 0
        merged["snap_CA"] = merged["snap_CA"].fillna(0).astype(int)
        merged["snap_TX"] = merged["snap_TX"].fillna(0).astype(int)
        merged["snap_WI"] = merged["snap_WI"].fillna(0).astype(int)
        # Final dataframe with Phase-1 FEATURE_COLS + keys (no leakage, cutoff already applied)
        from retail_demand_forecasting.nodes.constants import FEATURE_COLS as _FC

        keep = ["day_id", "date", "item_id", "store_id", "sales"] + _FC
        # Ensure all keep columns exist (some cyclical may be missing if date invalid)
        for c in keep:
            if c not in merged.columns:
                merged[c] = 0
        final = merged[keep].copy()
        final = final.dropna()
        print(
            f"Built real features: {len(final)} rows, {final['sales'].min()}-{final['sales'].max()} sales, mean {final['sales'].mean():.2f}"
        )
        print(final.head(2).to_string())
        return final
    except Exception as e:
        print(f"Failed to build real features: {e}")
        import traceback

        traceback.print_exc()
        return None


def load_features() -> pd.DataFrame:
    from retail_demand_forecasting.nodes.constants import FEATURE_COLS

    # 0. Try real pandas build from raw CSVs (no Spark) - best fidelity
    real = _build_real_features_pandas(sample_rows=200)
    if real is not None and not real.empty and any(c in real.columns for c in FEATURE_COLS):
        # Cache for DVC outs — resolve via catalog authority
        try:
            out = _resolve_catalog_path(
                "model_input_features",
                "data/03_features/model_input.parquet",  # catalog-allowlist: fallback
            ).parent
            out.mkdir(parents=True, exist_ok=True)
            # Save parquet for future loads (optional)
            real.head(1000).to_parquet(out / "model_input_sample.parquet", index=False)
        except Exception:
            pass
        return real

    # 1. Try Kedro intermediate parquet — catalog authority
    candidates = [
        _resolve_catalog_path(
            "model_input_features",
            "data/03_features/model_input.parquet",  # catalog-allowlist: fallback
        ),
        _resolve_catalog_path(
            "model_input_features",
            "data/03_features/model_input.parquet",  # catalog-allowlist: fallback
        ).parent,
        Path(
            "data/processed/model_input.parquet"
        ),  # catalog-allowlist: fallback for CI placeholder
    ]
    for c in candidates:
        if c.exists():
            try:
                if c.is_dir():
                    parquet_files = list(c.glob("*.parquet")) + list(c.rglob("*.parquet"))
                    if not parquet_files:
                        continue
                    df = pd.read_parquet(c)
                else:
                    df = pd.read_parquet(c)
                if df.empty or not any(col in df.columns for col in FEATURE_COLS):
                    print(f"Candidate {c} has no valid features, skipping")
                    continue
                print(f"Loaded features from {c}: {len(df)} rows")
                return df
            except Exception as e:
                print(f"Failed to load {c}: {e}, trying next")
                continue
    # 2. Improved synthetic with seasonality
    return _synthetic_df(n=1000)


def main():
    seed_everything(RANDOM_STATE)
    params = load_params()
    feature_params = params.get("feature_engineering", {})
    model_params_raw = params.get("model_params", {})
    optuna_cfg = params.get("optuna", {})

    flat_params = {
        "model_params": model_params_raw,
        "target_col": model_params_raw.get("target_col", "sales"),
        "test_size": model_params_raw.get("test_size", 0.2),
        "random_state": model_params_raw.get("random_state", RANDOM_STATE),
        "feature_engineering": feature_params,
    }

    df = load_features()
    print(f"Loaded features: {len(df)} rows, cols={list(df.columns)[:5]}...")

    use_optuna = optuna_cfg.get("enabled", False)
    if "--tune" in sys.argv:
        use_optuna = True
    if "--no-tune" in sys.argv or "--baseline" in sys.argv:
        use_optuna = False

    if use_optuna:
        print("Running Optuna tuning...")
        try:
            result = tune_and_train(
                df,
                flat_params,
                n_trials=optuna_cfg.get("n_trials", 20),
                cv_splits=optuna_cfg.get("cv_splits", 3),
                timeout=optuna_cfg.get("timeout"),
            )
        except ImportError as e:
            print(f"Optuna not available ({e}) — falling back to baseline")
            result = train_model(df, flat_params)
    else:
        print("Running baseline training (time-series split)...")
        result = train_model(df, flat_params)

    metrics_path = Path("metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Metrics saved to {metrics_path}: {result}")

    kedro_metrics = _resolve_catalog_path("model_metrics", "data/06_metrics/model_metrics.json")
    kedro_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(kedro_metrics, "w") as f:
        json.dump(result, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()

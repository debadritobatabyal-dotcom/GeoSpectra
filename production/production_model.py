import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

LOGGER = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "production", "synthetic_production_dataset_fixed.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "production_xgboost_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "production_metrics.json")

BASE_OPERATIONAL_FEATURES = [

    "total_trucks", "available_trucks", "active_trucks",
    "total_shovels", "available_shovels", "active_shovels",
    "truck_availability_pct", "shovel_availability_pct",
    "truck_utilization_pct", "shovel_utilization_pct",
    "truck_downtime_hours", "shovel_downtime_hours",
    "truck_breakdown_events", "shovel_breakdown_events", "maintenance_delay_hours",

    "haul_distance_km", "average_loaded_travel_time_min", "average_empty_travel_time_min",
    "average_loading_time_min", "average_dumping_time_min", "average_cycle_time_min",
    "truck_cycles", "queue_delay_min", "haulage_delay_hours",

    "operating_hours", "effective_operating_hours", "overall_utilization_pct",
    "loading_efficiency_pct", "haulage_efficiency_pct", "blasting_delay_hours",
    "shift_delay_hours", "fragmentation_index", "weather_disruption_hours",

    "rainfall_mm", "temperature_c", "soil_moisture_index", "ground_condition_index",

    "ore_feed_tonnes", "ore_grade_pct", "waste_to_ore_ratio",

    "month", "quarter", "day_of_week_num"
]

ALL_MINE_IDS = [
    "MINE_01", "MINE_02", "MINE_03", "MINE_04",
    "MINE_05", "MINE_06", "MINE_07", "MINE_08"
]

def load_and_preprocess_dataset(csv_path: str = DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Production dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["day_of_week_num"] = df["date"].dt.dayofweek

    mine_encoded = pd.DataFrame(index=df.index)
    for m in ALL_MINE_IDS:
        mine_encoded[f"mine_{m}"] = (df["mine_id"] == m).astype(float)

    X = pd.concat([df[BASE_OPERATIONAL_FEATURES], mine_encoded], axis=1)
    y = df["actual_production_tpd"]

    feature_names = list(X.columns)
    return df, X, y, feature_names

def train_and_evaluate(
    csv_path: str = DATA_PATH,
    split_date: str = "2024-01-01"
) -> dict:
    df, X, y, feature_names = load_and_preprocess_dataset(csv_path)

    train_mask = df["date"] < split_date
    test_mask = df["date"] >= split_date

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    train_start = df.loc[train_mask, "date"].min().strftime("%Y-%m-%d")
    train_end = df.loc[train_mask, "date"].max().strftime("%Y-%m-%d")
    test_start = df.loc[test_mask, "date"].min().strftime("%Y-%m-%d")
    test_end = df.loc[test_mask, "date"].max().strftime("%Y-%m-%d")

    LOGGER.info("Chronological split: Train %s to %s (%d rows), Test %s to %s (%d rows)",
                train_start, train_end, len(X_train), test_start, test_end, len(X_test))

    xgb_params = {
        "n_estimators": 250,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1
    }

    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    mae_xgb = float(mean_absolute_error(y_test, y_pred_xgb))
    rmse_xgb = float(root_mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = float(r2_score(y_test, y_pred_xgb))

    rf_params = {
        "n_estimators": 150,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1
    }

    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    mae_rf = float(mean_absolute_error(y_test, y_pred_rf))
    rmse_rf = float(root_mean_squared_error(y_test, y_pred_rf))
    r2_rf = float(r2_score(y_test, y_pred_rf))

    baseline_stats = {}
    for col in BASE_OPERATIONAL_FEATURES:
        baseline_stats[col] = {
            "mean": float(X_train[col].mean()),
            "std": float(X_train[col].std()),
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max()),
            "median": float(X_train[col].median()),
        }

    metrics_payload = {
        "dataset": os.path.basename(csv_path),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "split_type": "chronological_holdout",
        "train_date_range": [train_start, train_end],
        "test_date_range": [test_start, test_end],
        "target_variable": "actual_production_tpd",
        "models": {
            "xgboost": {
                "model_type": "XGBRegressor",
                "hyperparameters": xgb_params,
                "metrics": {
                    "mae": round(mae_xgb, 2),
                    "rmse": round(rmse_xgb, 2),
                    "r2": round(r2_xgb, 4)
                },
                "status": "production_primary"
            },
            "random_forest_benchmark": {
                "model_type": "RandomForestRegressor",
                "hyperparameters": rf_params,
                "metrics": {
                    "mae": round(mae_rf, 2),
                    "rmse": round(rmse_rf, 2),
                    "r2": round(r2_rf, 4)
                },
                "status": "benchmark_comparison"
            }
        },
        "features_count": len(feature_names),
        "features_list": feature_names
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {
        "model": xgb_model,
        "feature_names": feature_names,
        "base_features": BASE_OPERATIONAL_FEATURES,
        "mine_ids": ALL_MINE_IDS,
        "baseline_stats": baseline_stats,
        "metrics": metrics_payload["models"]["xgboost"]["metrics"],
        "benchmark_metrics": metrics_payload["models"]["random_forest_benchmark"]["metrics"],
        "temporal_holdout": {
            "train": [train_start, train_end],
            "test": [test_start, test_end]
        }
    }
    joblib.dump(bundle, MODEL_PATH)
    LOGGER.info("Saved XGBoost production model bundle to: %s", MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    LOGGER.info("Saved production metrics to: %s", METRICS_PATH)

    return {
        "xgboost": metrics_payload["models"]["xgboost"]["metrics"],
        "random_forest": metrics_payload["models"]["random_forest_benchmark"]["metrics"],
        "bundle_path": MODEL_PATH,
        "metrics_path": METRICS_PATH
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=" * 70)
    print("GeoSpectra Production Intelligence: Training XGBoost & Benchmark")
    print("=" * 70)
    res = train_and_evaluate()
    print("\n--- RESULTS ---")
    print(f"XGBoost Regressor:      MAE = {res['xgboost']['mae']} t, RMSE = {res['xgboost']['rmse']} t, R² = {res['xgboost']['r2']}")
    print(f"Random Forest Benchmark: MAE = {res['random_forest']['mae']} t, RMSE = {res['random_forest']['rmse']} t, R² = {res['random_forest']['r2']}")
    print(f"\nArtifacts saved:")
    print(f"  Model:   {res['bundle_path']}")
    print(f"  Metrics: {res['metrics_path']}")

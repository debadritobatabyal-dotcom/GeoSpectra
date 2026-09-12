import os
import joblib
import numpy as np
import pandas as pd
from typing import Any

from .production_explain import explain_production_prediction
from .recommendations import generate_recommendations

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIMARY_MODEL_PATH = os.path.join(BASE_DIR, "models", "production", "production_xgboost_model.pkl")
FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, "models", "production_xgboost_model.pkl")

_MODEL_BUNDLE = None

RISK_THRESHOLDS = {
    "low_max_pct": 5.0,
    "medium_max_pct": 15.0
}

def load_production_model_bundle(model_path: str = None) -> dict:
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        target_path = model_path or PRIMARY_MODEL_PATH
        if not os.path.exists(target_path):
            if os.path.exists(FALLBACK_MODEL_PATH):
                target_path = FALLBACK_MODEL_PATH
            else:
                raise FileNotFoundError(
                    f"Production model bundle not found at {target_path}. "
                    f"Run 'python3 production/production_model.py' first."
                )
        _MODEL_BUNDLE = joblib.load(target_path)
    return _MODEL_BUNDLE

def classify_risk(shortfall_pct: float, thresholds: dict = None) -> tuple[str, str, str]:
    t = thresholds or RISK_THRESHOLDS
    if shortfall_pct < t["low_max_pct"]:
        return "LOW", "#16a34a", "Production on or near planned target"
    elif shortfall_pct < t["medium_max_pct"]:
        return "MEDIUM", "#d97706", "Moderate target shortfall projected"
    else:
        return "HIGH", "#dc2626", "Significant target shortfall projected"

def prepare_feature_row(input_data: dict[str, Any], bundle: dict) -> tuple[pd.DataFrame, float]:
    feature_names = bundle["feature_names"]
    base_features = bundle["base_features"]
    baseline_stats = bundle.get("baseline_stats", {})
    mine_ids = bundle.get("mine_ids", [])

    target_tpd = float(input_data.get("planned_production_tpd", 2400.0))

    mine_id = str(input_data.get("mine_id", "MINE_01")).upper()

    row_dict = {}

    for col in base_features:
        if col in input_data and input_data[col] is not None:
            try:
                row_dict[col] = float(input_data[col])
            except (ValueError, TypeError):
                row_dict[col] = float(baseline_stats.get(col, {}).get("median", 0.0))
        else:
            row_dict[col] = float(baseline_stats.get(col, {}).get("median", 0.0))

    for m in mine_ids:
        col_name = f"mine_{m}"
        row_dict[col_name] = 1.0 if mine_id == m else 0.0

    df_row = pd.DataFrame([row_dict], columns=feature_names)
    return df_row, target_tpd

def production_predict(
    input_data: dict[str, Any] | pd.Series | pd.DataFrame,
    bundle: dict = None,
    include_explain: bool = True
) -> dict[str, Any]:
    b = bundle or load_production_model_bundle()
    model = b["model"]
    feature_names = b["feature_names"]

    if isinstance(input_data, pd.Series):
        input_dict = input_data.to_dict()
    elif isinstance(input_data, pd.DataFrame):
        input_dict = input_data.iloc[0].to_dict()
    else:
        input_dict = dict(input_data)

    df_features, target_tpd = prepare_feature_row(input_dict, b)

    raw_pred = float(model.predict(df_features)[0])
    expected_tpd = round(max(0.0, raw_pred), 1)

    target_variance_tpd = round(expected_tpd - target_tpd, 1)

    shortfall_tonnes = round(max(0.0, target_tpd - expected_tpd), 1)
    shortfall_pct = round((shortfall_tonnes / max(1.0, target_tpd)) * 100.0, 2)
    on_target = target_variance_tpd >= 0.0

    risk_level, risk_color, risk_summary = classify_risk(shortfall_pct)

    shap_drivers = {}
    negative_drivers = []
    positive_drivers = []
    if include_explain:
        try:
            shap_drivers = explain_production_prediction(
                model=model,
                feature_row=df_features,
                feature_names=feature_names,
                top_k=8
            )
            negative_drivers = shap_drivers.get("top_negative_drivers", [])
            positive_drivers = shap_drivers.get("top_positive_drivers", [])
        except Exception as e:
            shap_drivers = {
                "error": f"SHAP explanation unavailable: {str(e)}",
                "top_negative_drivers": [],
                "top_positive_drivers": []
            }

    recommendations = generate_recommendations(
        inputs=input_dict,
        negative_drivers=negative_drivers,
        risk_level=risk_level
    )

    return {
        "expected_production_tpd": expected_tpd,
        "planned_production_tpd": round(target_tpd, 1),
        "target_variance_tpd": target_variance_tpd,
        "shortfall_tonnes": shortfall_tonnes,
        "shortfall_pct": shortfall_pct,
        "on_target": on_target,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_summary": risk_summary,
        "shap_drivers": shap_drivers,
        "negative_drivers": negative_drivers,
        "positive_drivers": positive_drivers,
        "recommendations": recommendations,
        "model_info": {
            "model_type": "XGBRegressor",
            "mae_holdout": b.get("metrics", {}).get("mae"),
            "rmse_holdout": b.get("metrics", {}).get("rmse"),
            "r2_holdout": b.get("metrics", {}).get("r2"),
            "temporal_holdout": b.get("temporal_holdout", {})
        }
    }

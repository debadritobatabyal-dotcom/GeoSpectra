from __future__ import annotations
import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import shap

import feature_pipeline
from feature_pipeline import SatelliteUnavailableError

LOGGER = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_BUNDLE_PATH = os.path.join(BASE_DIR, "manganese_prospectivity_model.pkl")
SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")

_CACHED_BUNDLE = None
_CACHED_SCHEMA = None
_CACHED_EXPLAINER = None

def load_feature_schema():
    global _CACHED_SCHEMA
    if _CACHED_SCHEMA is None:
        with open(SCHEMA_PATH, "r") as f:
            _CACHED_SCHEMA = json.load(f)
    return _CACHED_SCHEMA

def load_model_bundle(model_path: str = None):
    global _CACHED_BUNDLE, _CACHED_EXPLAINER
    if _CACHED_BUNDLE is None:
        target_path = model_path or MODEL_BUNDLE_PATH
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Model bundle not found at {target_path}. Run train_model.py first.")
        _CACHED_BUNDLE = joblib.load(target_path)
        try:
            _CACHED_EXPLAINER = shap.TreeExplainer(_CACHED_BUNDLE["model"])
        except Exception as e:
            LOGGER.warning("Could not initialize TreeExplainer: %s", e)
            _CACHED_EXPLAINER = None
    return _CACHED_BUNDLE

def check_domain_bounds(lat: float, lon: float, bbox: dict = None) -> bool:
    if bbox is None:
        schema = load_feature_schema()
        bbox = schema["study_domain"]
    return (bbox["lat_min"] <= lat <= bbox["lat_max"]) and (bbox["lon_min"] <= lon <= bbox["lon_max"])

def encode_vector_for_inference(row_dict: dict, feature_cols: list, cat_cols: list, cat_mappings: dict) -> pd.DataFrame:
    row_data = {}
    for col in feature_cols:
        val = row_dict.get(col)
        if col in cat_cols:
            if pd.isna(val) or val is None:
                row_data[col] = -1.0
            else:
                mapping = cat_mappings.get(col, {})
                encoded_val = mapping.get(str(val), -1.0)
                row_data[col] = float(encoded_val)
        else:
            if pd.isna(val) or val is None:
                row_data[col] = np.nan
            else:
                try:
                    row_data[col] = float(val)
                except (ValueError, TypeError):
                    row_data[col] = np.nan

    return pd.DataFrame([row_data], columns=feature_cols)

def compute_shap_drivers(explainer, model, X_row: pd.DataFrame, feature_cols: list, top_n: int = 5) -> list[dict]:
    if explainer is None:
        return []
    try:
        shap_vals = explainer.shap_values(X_row)
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            vals = shap_vals[1][0]
        elif isinstance(shap_vals, np.ndarray):
            vals = shap_vals[0] if shap_vals.ndim > 1 else shap_vals
        else:
            return []

        indices = np.argsort(-np.abs(vals))[:top_n]
        schema = load_feature_schema()
        specs = schema.get("feature_specifications", {})

        drivers = []
        for idx in indices:
            feat_name = feature_cols[idx]
            impact = float(vals[idx])
            feat_val = X_row.iloc[0, idx]
            direction = "increased prospectivity score" if impact > 0 else "decreased prospectivity score"
            domain = specs.get(feat_name, {}).get("source", "Feature")

            drivers.append({
                "feature": feat_name,
                "domain": domain,
                "value": round(feat_val, 4) if isinstance(feat_val, (float, np.floating)) and np.isfinite(feat_val) else feat_val,
                "shap_impact": round(impact, 4),
                "direction": direction,
                "description": f"Feature '{feat_name}' {direction} (impact: {impact:+.3f})",
            })
        return drivers
    except Exception as e:
        LOGGER.warning("SHAP calculation error: %s", e)
        return []

def predict_single_location(features: dict, bundle: dict = None) -> dict:
    schema = load_feature_schema()
    bbox = schema["study_domain"]

    try:
        lat = float(features.get("latitude"))
        lon = float(features.get("longitude"))
    except (TypeError, ValueError):
        return {
            "status": "INVALID_INPUT",
            "message": "Latitude and longitude must be finite numeric values.",
            "prospectivity_score": None,
            "prospectivity_probability": None,
            "prospectivity_percentage": None,
            "prospectivity_category": None,
            "classification": "INVALID INPUT",
        }

    if not check_domain_bounds(lat, lon, bbox):
        return {
            "status": "OUT_OF_STUDY_DOMAIN",
            "message": f"Coordinates ({lat:.4f}N, {lon:.4f}E) fall outside the Sausar Manganese Belt domain ({bbox['lat_min']}–{bbox['lat_max']}N, {bbox['lon_min']}–{bbox['lon_max']}E).",
            "latitude": lat,
            "longitude": lon,
            "study_domain": bbox,
            "prospectivity_score": None,
            "prospectivity_probability": None,
            "prospectivity_percentage": None,
            "prospectivity_category": None,
            "classification": "OUT OF STUDY DOMAIN",
        }

    primary_set = set(schema["primary_features"])
    provided_set = set(features.keys())
    missing_keys = primary_set - provided_set

    if len(missing_keys) > 5:
        try:
            full_vector = feature_pipeline.extract_features(lat, lon)
        except SatelliteUnavailableError as e:
            return {
                "status": "SATELLITE_UNAVAILABLE",
                "message": f"Satellite data retrieval failed: {str(e)}. The system refuses to fabricate synthetic satellite observations.",
                "latitude": lat,
                "longitude": lon,
                "prospectivity_score": None,
                "prospectivity_probability": None,
                "prospectivity_percentage": None,
                "prospectivity_category": None,
                "classification": "SATELLITE UNAVAILABLE",
            }
    else:
        full_vector = features.copy()
        if "geology_source" not in full_vector:
            full_vector["geology_source"] = "provided_in_input"
        if "satellite_source" not in full_vector:
            full_vector["satellite_source"] = "provided_in_input"

    b = bundle or load_model_bundle()
    model = b["model"]
    feature_cols = b["feature_cols"]
    cat_cols = b["cat_cols"]
    cat_mappings = b["cat_mappings"]
    thresholds = b["probability_thresholds"]
    app_info = b.get("applicability", {})

    X_row = encode_vector_for_inference(full_vector, feature_cols, cat_cols, cat_mappings)

    prob = float(model.predict_proba(X_row)[0, 1])

    ref_probs = b.get("reference_predictions")
    if ref_probs is not None and len(ref_probs) > 0:
        idx_left = np.searchsorted(ref_probs, prob, side='left')
        idx_right = np.searchsorted(ref_probs, prob, side='right')
        prospectivity_score = round(float((idx_left + idx_right) / (2.0 * len(ref_probs)) * 100.0), 1)
    else:
        prospectivity_score = round(prob * 100.0, 1)

    if prospectivity_score >= 80.0:
        cat = "VERY HIGH"
    elif prospectivity_score >= 60.0:
        cat = "HIGH"
    elif prospectivity_score >= 40.0:
        cat = "MODERATE"
    elif prospectivity_score >= 20.0:
        cat = "LOW"
    else:
        cat = "VERY LOW"

    mod_cut = thresholds.get("moderate_cutoff", 0.05)
    high_cut = thresholds.get("high_cutoff", 0.75)

    iso = app_info.get("model")
    if iso is not None:
        try:
            X_iso = X_row.fillna(0.0)
            app_score = float(iso.decision_function(X_iso)[0])
            th_mod = app_info["threshold_moderate"]
            th_low = app_info["threshold_low"]
            if app_score >= th_mod:
                app_status = "HIGH APPLICABILITY"
            elif app_score >= th_low:
                app_status = "MODERATE APPLICABILITY"
            else:
                app_status = "LOW APPLICABILITY"
        except Exception as e:
            LOGGER.warning("IsolationForest scoring failed: %s", e)
            app_score = 0.0
            app_status = "MODERATE APPLICABILITY"
    else:
        app_score = 0.0
        app_status = "HIGH APPLICABILITY"

    global _CACHED_EXPLAINER
    if _CACHED_EXPLAINER is None:
        try:
            _CACHED_EXPLAINER = shap.TreeExplainer(model)
        except Exception:
            _CACHED_EXPLAINER = None

    top_drivers = compute_shap_drivers(_CACHED_EXPLAINER, model, X_row, feature_cols, top_n=5)

    nearby_mine_info = feature_pipeline.match_known_occurrence(lat, lon)
    if nearby_mine_info is not None:
        known_mine_nearby = True
        mine_name = nearby_mine_info.get("mine_name", "Known MOIL Deposit")
        mine_dist_km = nearby_mine_info.get("distance_km", 0.0)
    else:
        known_mine_nearby = False
        mine_name = None
        mine_dist_km = None

    warnings = [
        "Spatially validated continuous prospectivity score (0–100) trained on satellite, terrain, and geological features.",
        "Scores represent relative exploration prospectivity, not confirmed manganese concentration or calibrated probability.",
        "Screening tool only — does not estimate reserves, prove subsurface ore, or replace core drilling.",
    ]
    if full_vector.get("geology_source") == "geological_grid_lookup":
        geo_ctx = full_vector.get("geology_context", {})
        lookup_dist = geo_ctx.get("lookup_distance_km")
        if lookup_dist and lookup_dist > 5.0:
            warnings.append(f"Geological features assigned from nearest grid station ({lookup_dist:.1f} km away). Accuracy may decrease with distance.")

    return {
        "status": "SUCCESS",
        "latitude": lat,
        "longitude": lon,

        "prospectivity_score": prospectivity_score,
        "prospectivity_class": cat,
        "prospectivity_category": cat,
        "classification": cat,

        "raw_model_score": round(prob, 4),
        "raw_model_probability": round(prob, 4),
        "percentile_rank": prospectivity_score,

        "prospectivity_probability": prob,
        "prospectivity_percentage": prospectivity_score,

        "probability_thresholds": {
            "moderate_cutoff": round(mod_cut, 4),
            "high_cutoff": round(high_cut, 4),
            "type": "Empirically-derived percentile thresholds (p80, p95)",
        },
        "applicability_status": app_status,
        "applicability_score": round(app_score, 4),
        "geology_source": full_vector.get("geology_source"),
        "geology_context": full_vector.get("geology_context", {}),
        "satellite_source": full_vector.get("satellite_source"),
        "top_shap_drivers": top_drivers,
        "key_drivers": [d["description"] for d in top_drivers],
        "known_occurrence_nearby": known_mine_nearby,
        "nearest_mine_name": mine_name,
        "nearest_mine_distance_km": round(mine_dist_km, 2) if mine_dist_km is not None else None,
        "features": full_vector,
        "warnings": warnings,
        "model_version": b.get("schema_version", "4.0.0"),
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, default=21.8333, help="Latitude")
    parser.add_argument("--lon", type=float, default=80.2333, help="Longitude")
    args = parser.parse_args()
    res = predict_single_location({"latitude": args.lat, "longitude": args.lon})
    print(json.dumps(res, indent=2))

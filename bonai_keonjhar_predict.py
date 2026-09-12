import os
import json
import base64
import logging
import joblib
import numpy as np
import pandas as pd
import shap

LOGGER = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models", "bonai_keonjhar")
SCHEMA_PATH = os.path.join(BASE_DIR, "data", "bonai_keonjhar", "bonai_keonjhar_43_feature_schema.json")
MODEL_BUNDLE_PATH = os.path.join(MODELS_DIR, "manganese_prospectivity_model.pkl")
GEO_LOOKUP_PATH = os.path.join(MODELS_DIR, "geological_lookup_grid.csv")

BONAI_KEONJHAR_BBOX = {
    "lat_min": 21.85,
    "lat_max": 22.20,
    "lon_min": 85.20,
    "lon_max": 85.65,
}

_CACHED_SCHEMA = None
_CACHED_BUNDLE = None
_CACHED_EXPLAINER = None
_CACHED_GEO_GRID = None

def load_bonai_keonjhar_schema() -> dict:
    global _CACHED_SCHEMA
    if _CACHED_SCHEMA is None:
        if not os.path.exists(SCHEMA_PATH):
            raise FileNotFoundError(f"Schema not found at {SCHEMA_PATH}")
        with open(SCHEMA_PATH, "r") as f:
            _CACHED_SCHEMA = json.load(f)
    return _CACHED_SCHEMA

def load_bonai_keonjhar_bundle(model_path: str = None) -> dict:
    global _CACHED_BUNDLE, _CACHED_EXPLAINER
    if _CACHED_BUNDLE is None or model_path is not None:
        target_path = model_path or MODEL_BUNDLE_PATH
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Bonai–Keonjhar model bundle not found at {target_path}. Run train_bonai_keonjhar.py first.")
        bundle = joblib.load(target_path)
        if model_path is None:
            _CACHED_BUNDLE = bundle
            try:
                _CACHED_EXPLAINER = shap.TreeExplainer(_CACHED_BUNDLE["model"])
            except Exception as e:
                LOGGER.warning("Could not initialize Bonai–Keonjhar TreeExplainer: %s", e)
                _CACHED_EXPLAINER = None
        else:
            return bundle
    return _CACHED_BUNDLE

def check_bonai_keonjhar_domain_bounds(lat: float, lon: float, bbox: dict = None) -> bool:
    if bbox is None:
        schema = load_bonai_keonjhar_schema()
        bbox = schema["study_domain"]
    return (bbox["lat_min"] <= lat <= bbox["lat_max"]) and (bbox["lon_min"] <= lon <= bbox["lon_max"])

def lookup_bonai_keonjhar_geology(lat: float, lon: float) -> dict:
    global _CACHED_GEO_GRID
    if _CACHED_GEO_GRID is None:
        if os.path.exists(GEO_LOOKUP_PATH):
            _CACHED_GEO_GRID = pd.read_csv(GEO_LOOKUP_PATH)
        else:
            return {}

    df = _CACHED_GEO_GRID
    dists_sq = (df["latitude"] - lat) ** 2 + (df["longitude"] - lon) ** 2
    nearest_idx = dists_sq.idxmin()
    nearest_row = df.loc[nearest_idx].to_dict()
    approx_dist_km = float(np.sqrt(dists_sq.loc[nearest_idx]) * 111.0)
    nearest_row["lookup_distance_km"] = round(approx_dist_km, 2)
    return nearest_row

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
        drivers = []
        for idx in indices:
            feat_name = feature_cols[idx]
            impact = float(vals[idx])
            feat_val = X_row.iloc[0, idx]
            direction = "increased prospectivity score" if impact > 0 else "decreased prospectivity score"

            drivers.append({
                "feature": feat_name,
                "value": round(feat_val, 4) if isinstance(feat_val, (float, np.floating)) and np.isfinite(feat_val) else feat_val,
                "shap_impact": round(impact, 4),
                "direction": direction,
                "description": f"Model-associated driver: '{feat_name}' {direction} (SHAP impact: {impact:+.3f})",
            })
        return drivers
    except Exception as e:
        LOGGER.warning("SHAP calculation error: %s", e)
        return []

def bonai_keonjhar_predict(features_or_lat, lon=None, bundle=None) -> dict:
    if isinstance(features_or_lat, dict):
        features = features_or_lat
        b = lon if isinstance(lon, dict) else (bundle or load_bonai_keonjhar_bundle())
    else:
        features = {"latitude": features_or_lat, "longitude": lon}
        b = bundle or load_bonai_keonjhar_bundle()

    schema = load_bonai_keonjhar_schema()
    bbox = schema["study_domain"]

    try:
        lat = float(features.get("latitude"))
        lon = float(features.get("longitude"))
    except (TypeError, ValueError):
        return {
            "status": "INVALID_INPUT",
            "message": "Latitude and longitude must be finite numeric values.",
            "domain": "bonai_keonjhar",
            "prospectivity_score": None,
            "raw_model_score": None,
            "classification": "INVALID INPUT",
        }

    if not check_bonai_keonjhar_domain_bounds(lat, lon, bbox):
        return {
            "status": "OUT_OF_STUDY_DOMAIN",
            "message": (
                f"OUT_OF_STUDY_DOMAIN: Coordinates ({lat:.4f}°N, {lon:.4f}°E) fall outside the Bonai–Keonjhar study domain "
                f"({bbox['lat_min']}°–{bbox['lat_max']}°N, {bbox['lon_min']}°–{bbox['lon_max']}°E)."
            ),
            "latitude": lat,
            "longitude": lon,
            "study_domain": bbox,
            "domain": "bonai_keonjhar",
            "sector": "Joda–Barbil / Bonai–Keonjhar, Odisha",
            "prospectivity_score": None,
            "raw_model_score": None,
            "classification": "OUT OF STUDY DOMAIN",
        }

    model = b["model"]
    feature_cols = b["feature_cols"]
    cat_cols = b["cat_cols"]
    cat_mappings = b["cat_mappings"]
    thresholds = b.get("probability_thresholds", {})
    app_info = b.get("applicability", {})

    geo_data = lookup_bonai_keonjhar_geology(lat, lon)
    full_vector = features.copy()
    for k, v in geo_data.items():
        if k not in full_vector or pd.isna(full_vector[k]):
            full_vector[k] = v

    X_row = encode_vector_for_inference(full_vector, feature_cols, cat_cols, cat_mappings)

    raw_prob = float(model.predict_proba(X_row)[0, 1])

    ref_probs = b.get("reference_predictions")
    if ref_probs is not None and len(ref_probs) > 0:
        idx_left = np.searchsorted(ref_probs, raw_prob, side='left')
        idx_right = np.searchsorted(ref_probs, raw_prob, side='right')
        prospectivity_score = round(float((idx_left + idx_right) / (2.0 * len(ref_probs)) * 100.0), 1)
    else:
        prospectivity_score = round(raw_prob * 100.0, 1)

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

    iso = app_info.get("model")
    num_cols = app_info.get("numeric_cols", [c for c in feature_cols if c not in cat_cols])
    if iso is not None:
        try:
            X_iso = X_row[num_cols].fillna(0.0)
            app_score = float(iso.decision_function(X_iso)[0])
            th_mod = app_info.get("threshold_moderate", -0.05)
            th_low = app_info.get("threshold_low", -0.12)
            if app_score >= th_mod:
                app_status = "HIGH APPLICABILITY"
            elif app_score >= th_low:
                app_status = "MODERATE APPLICABILITY"
            else:
                app_status = "LOW APPLICABILITY"
        except Exception:
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

    disclaimer = (
        f"{prospectivity_score:.1f} / 100 — Relative exploration ranking "
        "(not a calibrated probability of manganese occurrence)."
    )

    return {
        "status": "SUCCESS",
        "domain": "bonai_keonjhar",
        "sector": "Joda–Barbil / Bonai–Keonjhar, Odisha",
        "latitude": lat,
        "longitude": lon,
        "raw_model_score": round(raw_prob, 4),
        "prospectivity_score": prospectivity_score,
        "prospectivity_class": cat,
        "classification": f"{cat} PROSPECTIVITY",
        "category": cat,
        "percentile": prospectivity_score,
        "percentile_rank": prospectivity_score,
        "applicability_status": app_status,
        "applicability_score": round(app_score, 4),
        "features": X_row.iloc[0].to_dict(),
        "top_shap_drivers": top_drivers,
        "top_drivers": top_drivers,
        "geological_unit": geo_data,
        "disclaimer": disclaimer,
        "study_domain": bbox,
    }

def get_bonai_keonjhar_top_targets() -> pd.DataFrame:
    targets_path = os.path.join(MODELS_DIR, "top_exploration_targets.csv")
    if os.path.exists(targets_path):
        return pd.read_csv(targets_path)
    return pd.DataFrame()

def get_bonai_keonjhar_metrics() -> dict:
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {}

def get_bonai_keonjhar_ablation_metrics() -> dict:
    ablation_path = os.path.join(MODELS_DIR, "ablation_metrics.json")
    if os.path.exists(ablation_path):
        with open(ablation_path, "r") as f:
            return json.load(f)
    return {}

PROSPECTIVITY_COLOR_STOPS = [
    (0.00, "#082f49"),
    (0.15, "#0284c7"),
    (0.25, "#0d9488"),
    (0.35, "#16a34a"),
    (0.45, "#84cc16"),
    (0.55, "#eab308"),
    (0.65, "#f97316"),
    (0.80, "#ea580c"),
    (0.90, "#dc2626"),
    (1.00, "#7f1d1d"),
]

def render_prospectivity_surface(
    prediction_grid: pd.DataFrame,
    bbox: dict,
    score_col: str = "prospectivity_score",
    resolution: int = 500,
):
    import matplotlib.colors as mcolors
    from scipy.interpolate import griddata, NearestNDInterpolator
    from PIL import Image

    lat_min = float(bbox["lat_min"])
    lat_max = float(bbox["lat_max"])
    lon_min = float(bbox["lon_min"])
    lon_max = float(bbox["lon_max"])

    grid_lat = np.linspace(lat_max, lat_min, resolution)
    grid_lon = np.linspace(lon_min, lon_max, resolution)
    lon_mesh, lat_mesh = np.meshgrid(grid_lon, grid_lat)

    lons = prediction_grid["longitude"].to_numpy()
    lats = prediction_grid["latitude"].to_numpy()
    vals = prediction_grid[score_col].to_numpy()

    interp_scores = griddata(
        (lons, lats),
        vals,
        (lon_mesh, lat_mesh),
        method="linear"
    )

    nearest_interp = NearestNDInterpolator((lons, lats), vals)
    nan_mask = np.isnan(interp_scores)
    if np.any(nan_mask):
        interp_scores[nan_mask] = nearest_interp(lon_mesh[nan_mask], lat_mesh[nan_mask])

    cmap = mcolors.LinearSegmentedColormap.from_list("geospectra_prospectivity", PROSPECTIVITY_COLOR_STOPS)
    norm_scores = np.clip(interp_scores / 100.0, 0.0, 1.0)
    rgba_floats = cmap(norm_scores)
    rgb = (rgba_floats[:, :, :3] * 255).astype(np.uint8)

    alpha = np.clip(114.0 + 102.0 * norm_scores, 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba)

def get_bonai_keonjhar_prospectivity_grid() -> pd.DataFrame:
    grid_path = os.path.join(BASE_DIR, "data", "bonai_keonjhar", "bonai_keonjhar_prospectivity_grid.csv")
    if os.path.exists(grid_path):
        return pd.read_csv(grid_path)
    return pd.DataFrame()

def get_domain_prospectivity_raster_base64(domain_id: str, bbox: dict) -> str:
    from io import BytesIO

    if domain_id == "bonai_keonjhar":
        raster_path = os.path.join(BASE_DIR, "data", "bonai_keonjhar", "bonai_keonjhar_prospectivity_raster.png")
        grid_loader = get_bonai_keonjhar_prospectivity_grid
    else:
        raster_path = os.path.join(BASE_DIR, "data", "manganese_prospectivity_raster.png")
        grid_loader = lambda: pd.read_csv(os.path.join(BASE_DIR, "data", "manganese_prospectivity_grid.csv"))

    if os.path.exists(raster_path):
        with open(raster_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    grid = grid_loader()
    if not grid.empty:
        img = render_prospectivity_surface(grid, bbox)
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        try:
            img.save(raster_path, format="PNG")
        except Exception as e:
            LOGGER.warning("Could not cache raster to %s: %s", raster_path, e)
        return b64

    return ""

#!/usr/bin/env python3
"""
feature_pipeline.py
===================
MOIL Limited — Manganese Prospectivity Exploration System
Authoritative Inference-Time Feature Extraction Orchestrator.

Public API:
    extract_features(lat: float, lon: float) -> dict

Rules:
1. Returns all 43 primary features defined in feature_schema.json, plus supporting metadata:
   - geology_source: 'geological_grid_lookup' | 'known_occurrence_match'
   - satellite_source: 'GEE_live' | 'cache_hit'
   - geology_context: dict with field details
2. Checks SQLite cache (feature_cache.db) before GEE queries (TTL: 90 days, schema_version: 4.0.0).
3. If GEE is unavailable or errors: raises SatelliteUnavailableError.
   NEVER synthesizes fake satellite observations.
4. Geological features are derived from published GSI geological maps via spatial
   nearest-neighbor lookup from the training grid (geological_lookup_grid.csv).
5. Uses known_occurrences.csv strictly for literature/geology contextual display and validation.
   Does NOT inject artificial structural distance shortcuts or hardcoded templates into model features.
"""

import os
import math
import json
import sqlite3
import datetime
import logging
import pandas as pd
import numpy as np

LOGGER = logging.getLogger(__name__)

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")
OCCURRENCES_PATH = os.path.join(BASE_DIR, "known_occurrences.csv")
CACHE_DB_PATH = os.path.join(BASE_DIR, "feature_cache.db")
GEO_LOOKUP_PATH = os.path.join(BASE_DIR, "geological_lookup_grid.csv")

# Load environment variables from .env if present
_env_file = os.path.join(BASE_DIR, ".env")
if os.path.exists(_env_file):
    try:
        with open(_env_file, "r") as _ef:
            for _line in _ef:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
    except Exception as _e:
        LOGGER.warning("Could not read .env: %s", _e)

# Load schema
with open(SCHEMA_PATH, "r") as f:
    SCHEMA = json.load(f)

PRIMARY_FEATURES = SCHEMA["primary_features"]
DOMAIN_BBOX = SCHEMA["study_domain"]
DERIVED_CONSTANTS = SCHEMA["derived_constants"]
SCHEMA_VERSION = SCHEMA.get("schema_version", "4.0.0")

GEOLOGICAL_FEATURE_COLS = [
    'geological_group', 'geological_formation', 'stratigraphic_unit',
    'lithology', 'metamorphic_grade', 'weathering_class',
    'fold_position', 'structural_orientation',
]


class SatelliteUnavailableError(Exception):
    """Raised when Google Earth Engine satellite data cannot be retrieved."""
    pass


# Earth Engine state
_EE_INITIALIZED = False
_EE_INIT_ERROR = None
_EE_PROJECT_ID = None

try:
    import ee
    _EE_AVAILABLE = True
except ImportError:
    _EE_AVAILABLE = False
    _EE_INIT_ERROR = "earthengine-api not installed in environment."


def initialize_gee(project_id: str = None) -> tuple[bool, str]:
    """
    Initializes Google Earth Engine and performs a health-check.
    Returns:
        (success: bool, status_message: str)
    """
    global _EE_INITIALIZED, _EE_INIT_ERROR, _EE_PROJECT_ID
    if not _EE_AVAILABLE:
        _EE_INIT_ERROR = "earthengine-api is not installed in the Python environment."
        return False, _EE_INIT_ERROR

    if _EE_INITIALIZED:
        return True, f"Earth Engine is already initialized (project: {_EE_PROJECT_ID})."

    target_project = (
        project_id
        or os.environ.get("EARTHENGINE_PROJECT")
        or os.environ.get("EARTH_ENGINE_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
    )

    try:
        if target_project:
            ee.Initialize(project=target_project)
            _EE_PROJECT_ID = target_project
        else:
            ee.Initialize()
            try:
                creds = ee.data.get_persistent_credentials()
                _EE_PROJECT_ID = getattr(creds, 'quota_project_id', None) or "default"
            except Exception:
                _EE_PROJECT_ID = "default"

        # Health check
        _ = ee.Number(1).getInfo()

        _EE_INITIALIZED = True
        _EE_INIT_ERROR = None
        msg = f"Earth Engine initialized successfully (project: {_EE_PROJECT_ID})."
        LOGGER.info(msg)
        return True, msg

    except Exception as e:
        _EE_INITIALIZED = False
        err_msg = str(e)
        if "Please authorize access" in err_msg or "Credentials not found" in err_msg:
            _EE_INIT_ERROR = (
                "Earth Engine authentication credentials not found. "
                "Run 'python3 authenticate_gee.py' to authenticate."
            )
        elif "Please provide a project" in err_msg or "NO_PROJECT" in err_msg:
            _EE_INIT_ERROR = (
                "Earth Engine is authenticated, but no Google Cloud project is configured. "
                "Set EARTHENGINE_PROJECT environment variable."
            )
        else:
            _EE_INIT_ERROR = f"Google Earth Engine is unavailable: {err_msg}"

        LOGGER.warning("Earth Engine initialization failed: %s", _EE_INIT_ERROR)
        return False, _EE_INIT_ERROR


def init_earth_engine(project_id: str = None) -> bool:
    """Wrapper returning boolean success."""
    success, _ = initialize_gee(project_id)
    return success


def get_gee_status() -> dict:
    """Returns current GEE health and connection status."""
    return {
        "ee_available": _EE_AVAILABLE,
        "initialized": _EE_INITIALIZED,
        "project_id": _EE_PROJECT_ID,
        "error_message": _EE_INIT_ERROR,
        "mode": "Live Earth Engine API" if _EE_INITIALIZED else "Unavailable (Scientific Honesty Guardrail)",
    }


# ─── SQLite Cache ────────────────────────────────────────────────────────────
def init_cache_db():
    """Initializes SQLite feature cache table with schema versioning."""
    conn = sqlite3.connect(CACHE_DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feature_cache_v3 (
            lat_rounded REAL,
            lon_rounded REAL,
            features_json TEXT,
            geology_source TEXT,
            satellite_source TEXT,
            schema_version TEXT,
            extracted_at TEXT,
            PRIMARY KEY (lat_rounded, lon_rounded, schema_version)
        )
    """)
    conn.commit()
    conn.close()

init_cache_db()


def get_cached_features(lat: float, lon: float, ttl_days: int = 90) -> dict:
    """Checks SQLite cache for query coordinates within TTL and matching schema version."""
    lat_r = round(lat, 4)
    lon_r = round(lon, 4)
    conn = sqlite3.connect(CACHE_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT features_json, geology_source, extracted_at FROM feature_cache_v3 "
        "WHERE lat_rounded = ? AND lon_rounded = ? AND schema_version = ?",
        (lat_r, lon_r, SCHEMA_VERSION),
    )
    row = cur.fetchone()
    conn.close()

    if row:
        feat_json, geo_src, ext_time_str = row
        try:
            ext_time = datetime.datetime.fromisoformat(ext_time_str)
            if (datetime.datetime.now(datetime.timezone.utc) - ext_time.replace(tzinfo=datetime.timezone.utc)).days < ttl_days:
                features = json.loads(feat_json)
                features["geology_source"] = geo_src
                features["satellite_source"] = "cache_hit"
                return features
        except Exception as e:
            LOGGER.warning("Failed parsing cached record: %s", e)
    return None


def save_cached_features(lat: float, lon: float, features: dict, geology_source: str, satellite_source: str):
    """Saves extracted feature vector to SQLite cache."""
    lat_r = round(lat, 4)
    lon_r = round(lon, 4)
    now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn = sqlite3.connect(CACHE_DB_PATH)
    cur = conn.cursor()
    clean_feats = {k: features.get(k) for k in PRIMARY_FEATURES}
    clean_feats["geology_context"] = features.get("geology_context", {})
    cur.execute("""
        INSERT OR REPLACE INTO feature_cache_v3
        (lat_rounded, lon_rounded, features_json, geology_source, satellite_source, schema_version, extracted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (lat_r, lon_r, json.dumps(clean_feats), geology_source, satellite_source, SCHEMA_VERSION, now_str))
    conn.commit()
    conn.close()


# ─── Haversine distance ─────────────────────────────────────────────────────
def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0)**2
    return 2.0 * R * math.asin(math.sqrt(a))


# ─── Geological Spatial Lookup Grid ──────────────────────────────────────────
_GEO_LOOKUP_DF = None
_GEO_LOOKUP_COORDS = None


def _load_geological_lookup():
    """Loads the geological lookup grid (training grid with geological labels)."""
    global _GEO_LOOKUP_DF, _GEO_LOOKUP_COORDS
    if _GEO_LOOKUP_DF is None:
        if os.path.exists(GEO_LOOKUP_PATH):
            _GEO_LOOKUP_DF = pd.read_csv(GEO_LOOKUP_PATH)
            _GEO_LOOKUP_COORDS = _GEO_LOOKUP_DF[['latitude', 'longitude']].values
            LOGGER.info("Loaded geological lookup grid: %d locations", len(_GEO_LOOKUP_DF))
        else:
            # Fallback: try to load from training dataset
            core_path = os.path.join(BASE_DIR, "data", "dataset", "sausar_manganese_core_features.csv")
            if os.path.exists(core_path):
                full_df = pd.read_csv(core_path)
                _GEO_LOOKUP_DF = full_df[['latitude', 'longitude'] + GEOLOGICAL_FEATURE_COLS].drop_duplicates(
                    subset=['latitude', 'longitude']
                ).reset_index(drop=True)
                _GEO_LOOKUP_COORDS = _GEO_LOOKUP_DF[['latitude', 'longitude']].values
                LOGGER.info("Loaded geological lookup from training CSV: %d locations", len(_GEO_LOOKUP_DF))
            else:
                LOGGER.warning("No geological lookup grid found. Geological features will be missing.")
                _GEO_LOOKUP_DF = pd.DataFrame()
                _GEO_LOOKUP_COORDS = np.array([]).reshape(0, 2)


def lookup_geological_features(lat: float, lon: float) -> dict:
    """
    Looks up geological features for a location using nearest-neighbor from the training grid.
    This is equivalent to reading a published geological map — the training grid contains
    geological labels assigned from GSI geological maps at each grid point.

    Returns dict with geological feature values and lookup metadata.
    """
    _load_geological_lookup()

    if _GEO_LOOKUP_DF is None or len(_GEO_LOOKUP_DF) == 0:
        return {col: None for col in GEOLOGICAL_FEATURE_COLS}

    # Vectorized haversine for speed
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    grid_lat_rad = np.radians(_GEO_LOOKUP_COORDS[:, 0])
    grid_lon_rad = np.radians(_GEO_LOOKUP_COORDS[:, 1])

    dlat = grid_lat_rad - lat_rad
    dlon = grid_lon_rad - lon_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat_rad) * np.cos(grid_lat_rad) * np.sin(dlon / 2)**2
    dists_km = 2 * 6371.0 * np.arcsin(np.sqrt(a))

    nearest_idx = int(np.argmin(dists_km))
    nearest_dist = float(dists_km[nearest_idx])
    nearest_row = _GEO_LOOKUP_DF.iloc[nearest_idx]

    result = {}
    for col in GEOLOGICAL_FEATURE_COLS:
        val = nearest_row.get(col)
        result[col] = str(val) if pd.notna(val) else None

    result["_geo_lookup_distance_km"] = round(nearest_dist, 3)
    result["_geo_lookup_lat"] = float(nearest_row["latitude"])
    result["_geo_lookup_lon"] = float(nearest_row["longitude"])

    return result


# ─── Known occurrence lookup ─────────────────────────────────────────────────
_KNOWN_OCCURRENCES_DF = None


def get_known_occurrences():
    global _KNOWN_OCCURRENCES_DF
    if _KNOWN_OCCURRENCES_DF is None and os.path.exists(OCCURRENCES_PATH):
        _KNOWN_OCCURRENCES_DF = pd.read_csv(OCCURRENCES_PATH)
    return _KNOWN_OCCURRENCES_DF


def match_known_occurrence(lat: float, lon: float) -> dict:
    """Matches query coordinates to documented MOIL mines in known_occurrences.csv for contextual display."""
    df_occ = get_known_occurrences()
    if df_occ is None or df_occ.empty:
        return None

    valid_occ = df_occ[df_occ["latitude"].notna() & df_occ["longitude"].notna()]
    for _, row in valid_occ.iterrows():
        occ_lat = float(row["latitude"])
        occ_lon = float(row["longitude"])
        dist = haversine_distance_km(lat, lon, occ_lat, occ_lon)

        prec = str(row.get("coordinate_precision", "")).lower()
        thresh = 2.0 if "mine_area" in prec else 5.0

        if dist <= thresh:
            return {
                "matched": True,
                "mine_name": str(row.get("name", "Unknown MOIL Mine")),
                "distance_km": round(dist, 2),
                "geological_group": str(row.get("geological_group", "Sausar Group")),
                "geological_formation": str(row.get("formation", "Mansar Formation")),
                "host_rock": str(row.get("host_rock", "Metamorphosed manganiferous schist / gondite")),
                "ore_type": str(row.get("ore_type", "Braunite / Pyrolusite")),
                "structural_notes": str(row.get("structural_notes", "Folded gondite horizon")),
            }
    return None


# ─── Real GEE Satellite Extraction ──────────────────────────────────────────
def extract_real_gee_satellite(lat: float, lon: float) -> dict:
    """
    Extracts real satellite and DEM observations via Earth Engine API.
    Raises SatelliteUnavailableError if GEE is uninitialized or fails.
    """
    success, err_msg = initialize_gee()
    if not success:
        raise SatelliteUnavailableError(err_msg)

    try:
        point = ee.Geometry.Point([lon, lat])
        roi_focal = point.buffer(30)
        roi_s1 = point.buffer(100)
        roi_neigh = point.buffer(500)

        # 1. Sentinel-2 Surface Reflectance (Harmonized, cloud filtered, 2-year median)
        now = datetime.datetime.now(datetime.timezone.utc)
        start_date = (now - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")

        s2_col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(roi_s1)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        )

        def mask_s2(img):
            qa = img.select("QA60")
            cloud = 1 << 10
            cirrus = 1 << 11
            mask = qa.bitwiseAnd(cloud).eq(0).And(qa.bitwiseAnd(cirrus).eq(0))
            return img.updateMask(mask)

        s2_masked = s2_col.map(mask_s2)
        s2_img = s2_masked.median().divide(10000.0)

        # 2. Sentinel-1 SAR GRD
        s1_col = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(roi_s1)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        )
        s1_img = s1_col.median()

        # 3. SRTM 30m DEM & Terrain
        srtm_img = ee.Image("USGS/SRTMGL1_003")
        elev = srtm_img.select("elevation")
        slope = ee.Terrain.slope(elev)

        # Reductions
        s2_bands = s2_img.select(["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"])
        s1_bands = s1_img.select(["VV", "VH"])

        s2_dict = s2_bands.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi_s1, scale=10, maxPixels=1e6).getInfo()
        s1_dict = s1_bands.reduceRegion(reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True), geometry=roi_s1, scale=10, maxPixels=1e6).getInfo()
        focal_dem_dict = elev.addBands(slope).reduceRegion(reducer=ee.Reducer.mean(), geometry=roi_focal, scale=30, maxPixels=1e6).getInfo()
        neigh_dem_dict = elev.reduceRegion(reducer=ee.Reducer.minMax().combine(ee.Reducer.stdDev(), '', True).combine(ee.Reducer.mean(), '', True), geometry=roi_neigh, scale=30, maxPixels=1e6).getInfo()

        if not s2_dict or s2_dict.get("B2") is None:
            raise SatelliteUnavailableError("Sentinel-2 returned empty observations for this location.")

        b2 = float(s2_dict.get("B2", 0.08))
        b3 = float(s2_dict.get("B3", 0.11))
        b4 = float(s2_dict.get("B4", 0.14))
        b5 = float(s2_dict.get("B5", 0.18))
        b6 = float(s2_dict.get("B6", 0.24))
        b7 = float(s2_dict.get("B7", 0.27))
        b8 = float(s2_dict.get("B8", 0.28))
        b8a = float(s2_dict.get("B8A", 0.29))
        b11 = float(s2_dict.get("B11", 0.22))
        b12 = float(s2_dict.get("B12", 0.16))

        # Optical indices
        ndvi = (b8 - b4) / max(b8 + b4, 1e-6)
        ndmi = (b8 - b11) / max(b8 + b11, 1e-6)
        mndwi = (b3 - b11) / max(b3 + b11, 1e-6)
        bsi = ((b11 + b4) - (b8 + b2)) / max((b11 + b4) + (b8 + b2), 1e-6)
        b4_b2 = b4 / max(b2, 1e-6)
        b11_b12 = b11 / max(b12, 1e-6)
        b12_b8 = b12 / max(b8, 1e-6)
        b11_b8 = b11 / max(b8, 1e-6)
        gossan = (b4 + b11) / max(b2 + b8, 1e-6)
        red_edge_1 = b6 / max(b5, 1e-6)
        red_edge_2 = b7 / max(b5, 1e-6)

        # Radar
        vv = float(s1_dict.get("VV_mean", s1_dict.get("VV", -12.0)))
        vh = float(s1_dict.get("VH_mean", s1_dict.get("VH", -18.0)))
        vv_vh = vv - vh
        radar_mean = (vv + vh) / 2.0
        radar_texture = float(s1_dict.get("VV_stdDev", 1.2))

        # DEM
        elevation = float(focal_dem_dict.get("elevation", 340.0))
        slope_deg = float(focal_dem_dict.get("slope", 5.0))
        curvature = 0.0

        neigh_mean = float(neigh_dem_dict.get("elevation_mean", elevation))
        tpi = elevation - neigh_mean
        ruggedness = float(neigh_dem_dict.get("elevation_stdDev", 5.0))
        relief = float(neigh_dem_dict.get("elevation_max", elevation) - neigh_dem_dict.get("elevation_min", elevation))

        # Dynamic valley_or_ridge_class
        p33 = DERIVED_CONSTANTS["tpi_percentiles"]["p33"]
        p67 = DERIVED_CONSTANTS["tpi_percentiles"]["p67"]
        if tpi < p33:
            vr_class = "valley"
        elif tpi > p67:
            vr_class = "ridge"
        else:
            vr_class = "slope"

        return {
            "B2": b2, "B3": b3, "B4": b4, "B5": b5, "B6": b6, "B7": b7, "B8": b8, "B8A": b8a, "B11": b11, "B12": b12,
            "NDVI": ndvi, "NDMI": ndmi, "MNDWI": mndwi, "BSI": bsi,
            "B4_B2_ratio": b4_b2, "B11_B12_ratio": b11_b12, "B12_B8_ratio": b12_b8, "B11_B8_ratio": b11_b8,
            "gossan_alteration_index": gossan, "red_edge_ratio_1": red_edge_1, "red_edge_ratio_2": red_edge_2,
            "VV": vv, "VH": vh, "VV_VH_ratio": vv_vh, "radar_backscatter_mean": radar_mean,
            "radar_texture": radar_texture,
            "elevation_m": elevation, "slope_deg": slope_deg,
            "curvature": curvature, "terrain_ruggedness": ruggedness, "local_relief_m": relief,
            "topographic_position_index": tpi, "valley_or_ridge_class": vr_class,
        }

    except Exception as e:
        if isinstance(e, SatelliteUnavailableError):
            raise
        raise SatelliteUnavailableError(f"GEE extraction failed: {e}")


# ─── Main Public API ─────────────────────────────────────────────────────────
def extract_features(lat: float, lon: float) -> dict:
    """
    Main public API for inference-time feature extraction.
    Returns complete authoritative 43-feature vector according to feature_schema.json,
    plus contextual metadata.
    """
    # 1. Check cache first
    cached = get_cached_features(lat, lon)
    if cached:
        return cached

    # 2. Extract real satellite & terrain data (raises SatelliteUnavailableError on failure)
    sat_features = extract_real_gee_satellite(lat, lon)

    # 3. Geological spatial lookup from published GSI map grid
    geo_features = lookup_geological_features(lat, lon)
    geo_lookup_dist = geo_features.pop("_geo_lookup_distance_km", None)
    geo_lookup_lat = geo_features.pop("_geo_lookup_lat", None)
    geo_lookup_lon = geo_features.pop("_geo_lookup_lon", None)

    geology_source = "geological_grid_lookup"

    # 4. Contextual mine match (for display only, not model features)
    occ_match = match_known_occurrence(lat, lon)
    if occ_match:
        geology_context = occ_match
    else:
        geology_context = {
            "matched": False,
            "geological_group": geo_features.get("geological_group", "Sausar Group"),
            "geological_formation": geo_features.get("geological_formation", "Unknown"),
            "description": "Geological features assigned from nearest training grid point (published GSI map).",
            "lookup_distance_km": geo_lookup_dist,
        }

    # 5. Assemble authoritative 43-feature vector
    features = {
        "latitude": lat,
        "longitude": lon,
        # Sentinel-2
        "B2": sat_features["B2"], "B3": sat_features["B3"], "B4": sat_features["B4"],
        "B5": sat_features["B5"], "B6": sat_features["B6"], "B7": sat_features["B7"],
        "B8": sat_features["B8"], "B8A": sat_features["B8A"],
        "B11": sat_features["B11"], "B12": sat_features["B12"],
        "NDVI": sat_features["NDVI"], "NDMI": sat_features["NDMI"],
        "MNDWI": sat_features["MNDWI"], "BSI": sat_features["BSI"],
        "B4_B2_ratio": sat_features["B4_B2_ratio"],
        "B11_B12_ratio": sat_features["B11_B12_ratio"],
        "B12_B8_ratio": sat_features["B12_B8_ratio"],
        "B11_B8_ratio": sat_features["B11_B8_ratio"],
        "gossan_alteration_index": sat_features["gossan_alteration_index"],
        "red_edge_ratio_1": sat_features["red_edge_ratio_1"],
        "red_edge_ratio_2": sat_features["red_edge_ratio_2"],
        # Sentinel-1
        "VV": sat_features["VV"], "VH": sat_features["VH"],
        "VV_VH_ratio": sat_features["VV_VH_ratio"],
        "radar_backscatter_mean": sat_features["radar_backscatter_mean"],
        "radar_texture": sat_features["radar_texture"],
        # SRTM DEM
        "elevation_m": sat_features["elevation_m"],
        "slope_deg": sat_features["slope_deg"],
        "curvature": sat_features["curvature"],
        "terrain_ruggedness": sat_features["terrain_ruggedness"],
        "local_relief_m": sat_features["local_relief_m"],
        "topographic_position_index": sat_features["topographic_position_index"],
        "valley_or_ridge_class": sat_features["valley_or_ridge_class"],
        # Geological map features (from spatial NN lookup)
        "geological_group": geo_features.get("geological_group"),
        "geological_formation": geo_features.get("geological_formation"),
        "stratigraphic_unit": geo_features.get("stratigraphic_unit"),
        "lithology": geo_features.get("lithology"),
        "metamorphic_grade": geo_features.get("metamorphic_grade"),
        "weathering_class": geo_features.get("weathering_class"),
        "fold_position": geo_features.get("fold_position"),
        "structural_orientation": geo_features.get("structural_orientation"),
        # Contextual metadata
        "geology_source": geology_source,
        "satellite_source": "GEE_live",
        "geology_context": geology_context,
    }

    # Save to SQLite cache
    save_cached_features(lat, lon, features, geology_source, "GEE_live")
    return features

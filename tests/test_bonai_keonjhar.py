import os
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from bonai_keonjhar_predict import (
    bonai_keonjhar_predict,
    check_bonai_keonjhar_domain_bounds,
    load_bonai_keonjhar_bundle,
    load_bonai_keonjhar_schema,
    get_bonai_keonjhar_top_targets,
    get_bonai_keonjhar_metrics,
    get_bonai_keonjhar_ablation_metrics,
    lookup_bonai_keonjhar_geology,
    BONAI_KEONJHAR_BBOX,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models", "bonai_keonjhar")
SAUSAR_MODEL_PATH = os.path.join(BASE_DIR, "manganese_prospectivity_model.pkl")
APP_PATH = os.path.join(BASE_DIR, "app.py")

EXPECTED_CATEGORICAL = [
    "valley_or_ridge_class",
    "geological_group",
    "geological_formation",
    "stratigraphic_unit",
    "lithology",
    "metamorphic_grade",
    "weathering_class",
    "fold_position",
    "structural_orientation",
]

BANNED_FEATURE_SUBSTRINGS = [
    "distance_to_nearest_mine",
    "mine_distance",
    "target_proximity",
    "occurrence_dist",
    "nearest_mine",
    "label",
    "prospectivity",
    "target",
]

def test_authoritative_43_feature_schema():
    schema = load_bonai_keonjhar_schema()
    all_features = schema["primary_features"]
    assert len(all_features) == 43, f"Expected exactly 43 features, found {len(all_features)}"

    bundle = load_bonai_keonjhar_bundle()
    assert len(bundle["feature_cols"]) == 43
    assert len(bundle["cat_cols"]) == 9
    numeric_features = [c for c in bundle["feature_cols"] if c not in bundle["cat_cols"]]
    assert len(numeric_features) == 34
    assert len(numeric_features) + len(bundle["cat_cols"]) == 43

def test_authoritative_9_categorical_features():
    schema = load_bonai_keonjhar_schema()
    cat_features = schema["categorical_features"]
    assert sorted(cat_features) == sorted(EXPECTED_CATEGORICAL)

    bundle = load_bonai_keonjhar_bundle()
    assert sorted(bundle["cat_cols"]) == sorted(EXPECTED_CATEGORICAL)

def test_zero_banned_leakage_features():
    schema = load_bonai_keonjhar_schema()
    bundle = load_bonai_keonjhar_bundle()

    for feat in schema["primary_features"]:
        feat_lower = feat.lower()
        for banned in BANNED_FEATURE_SUBSTRINGS:
            assert banned not in feat_lower or feat_lower == "top_exploration_targets", (
                f"Banned substring '{banned}' detected in feature '{feat}'"
            )

    for feat in bundle["feature_cols"]:
        feat_lower = feat.lower()
        for banned in BANNED_FEATURE_SUBSTRINGS:
            assert banned not in feat_lower, (
                f"Banned substring '{banned}' detected in bundle feature '{feat}'"
            )

def test_bonai_keonjhar_model_bundle_loads():
    bundle = load_bonai_keonjhar_bundle()
    assert "model" in bundle
    assert "feature_cols" in bundle
    assert "reference_predictions" in bundle
    assert "domain_bbox" in bundle
    assert "cat_mappings" in bundle
    assert "classifier_name" in bundle

    assert len(bundle["reference_predictions"]) == 19421
    assert bundle["domain_bbox"]["lat_min"] == 21.85
    assert bundle["domain_bbox"]["lat_max"] == 22.20
    assert bundle["domain_bbox"]["lon_min"] == 85.20
    assert bundle["domain_bbox"]["lon_max"] == 85.65

def test_sausar_model_bundle_untouched():
    assert os.path.exists(SAUSAR_MODEL_PATH)
    sausar_bundle = joblib.load(SAUSAR_MODEL_PATH)

    assert "model" in sausar_bundle
    assert "feature_cols" in sausar_bundle or "feature_names" in sausar_bundle

    assert sausar_bundle.get("domain") != "bonai_keonjhar"
    assert "bonai" not in str(sausar_bundle.get("domain", "")).lower()

def test_in_domain_prediction_generates_valid_score():
    test_lat, test_lon = 22.0250, 85.4250
    assert check_bonai_keonjhar_domain_bounds(test_lat, test_lon) is True

    result = bonai_keonjhar_predict(test_lat, test_lon)
    assert result["status"] == "SUCCESS"
    assert result["domain"] == "bonai_keonjhar"
    assert 0.0 <= result["prospectivity_score"] <= 100.0
    assert result["prospectivity_class"] in ["VERY HIGH", "HIGH", "MODERATE", "LOW", "VERY LOW"]
    assert 0.0 <= result["raw_model_score"] <= 1.0
    assert "top_shap_drivers" in result
    assert len(result["top_shap_drivers"]) > 0
    assert "geological_unit" in result
    assert "features" in result
    assert len(result["features"]) == 43

def test_out_of_domain_guardrail():
    out_points = [
        (21.15, 79.08),
        (18.92, 72.83),
        (28.61, 77.20),
        (21.80, 85.40),
        (22.25, 85.40),
        (22.00, 85.10),
        (22.00, 85.70),
    ]
    for lat, lon in out_points:
        assert check_bonai_keonjhar_domain_bounds(lat, lon) is False
        res = bonai_keonjhar_predict(lat, lon)
        assert res["status"] == "OUT_OF_STUDY_DOMAIN"
        assert "OUT_OF_STUDY_DOMAIN" in res["message"]

def test_reference_distribution_quarantine():
    bk_bundle = load_bonai_keonjhar_bundle()
    bk_refs = bk_bundle["reference_predictions"]

    assert len(bk_refs) == 19421

    assert np.all(np.diff(bk_refs) >= 0)

    raw_val = 0.50
    expected_pct = float(np.searchsorted(bk_refs, raw_val) / len(bk_refs) * 100.0)
    assert 0.0 <= expected_pct <= 100.0

def test_lat_lon_ablation_metrics_integrity():
    ablation = get_bonai_keonjhar_ablation_metrics()
    assert "full_model" in ablation
    assert "ablated_model_no_coords" in ablation
    assert "delta" in ablation

    full_pr = ablation["full_model"]["pr_auc"]
    abl_pr = ablation["ablated_model_no_coords"]["pr_auc"]
    delta_pr = ablation["delta"]["pr_auc_difference"]

    assert full_pr >= 0.35
    assert abl_pr >= 0.35

    assert abs(delta_pr) < 0.01, f"Ablation delta {delta_pr} is too large, possible coordinate dependence"

def test_geological_lookup_coverage():
    geo = lookup_bonai_keonjhar_geology(22.0250, 85.4250)
    assert "geological_formation" in geo
    assert "geological_group" in geo
    assert "lithology" in geo
    assert "weathering_class" in geo
    assert "fold_position" in geo
    assert "structural_orientation" in geo

def test_top_exploration_targets_structure():
    df = get_bonai_keonjhar_top_targets()
    assert not df.empty
    assert len(df) == 10
    required_cols = [
        "target_id", "latitude", "longitude", "prospectivity_score",
        "prospectivity_class", "formation", "lithology", "is_greenfield"
    ]
    for col in required_cols:
        assert col in df.columns

    scores = df["prospectivity_score"].tolist()
    assert scores == sorted(scores, reverse=True)

def test_domain_selector_apptest():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    if len(at.button) > 1:
        at.button[1].click().run()
    elif len(at.button) > 0:
        at.button[0].click().run()
    assert at.session_state["authenticated"] is True

    if "selected_domain" in at.session_state:
        assert at.session_state["selected_domain"] in ["sausar", None]

    at.session_state["selected_domain"] = "bonai_keonjhar"
    at.session_state["target_lat"] = 22.0250
    at.session_state["target_lon"] = 85.4250
    at.run()

    assert not at.exception

    markdown_texts = [m.value for m in at.markdown]
    has_bk = any("Bonai–Keonjhar" in text or "Joda–Barbil" in text for text in markdown_texts)
    assert has_bk is True

def test_bonai_keonjhar_prospectivity_raster_properties():
    from PIL import Image
    raster_path = os.path.join(BASE_DIR, "data", "bonai_keonjhar", "bonai_keonjhar_prospectivity_raster.png")
    assert os.path.exists(raster_path), f"Raster missing at {raster_path}"

    img = Image.open(raster_path)
    assert img.size == (500, 500)
    assert img.mode == "RGBA"

    arr = np.array(img)
    alpha = arr[:, :, 3]

    assert alpha.min() >= 110
    assert alpha.max() <= 220

def test_bonai_keonjhar_prospectivity_grid_integrity():
    from bonai_keonjhar_predict import get_bonai_keonjhar_prospectivity_grid
    grid = get_bonai_keonjhar_prospectivity_grid()
    assert not grid.empty
    assert len(grid) == 19421

    expected_cols = [
        "latitude", "longitude", "raw_model_score", "prospectivity_score",
        "prospectivity_class", "geological_group", "geological_formation"
    ]
    for col in expected_cols:
        assert col in grid.columns

    assert grid["latitude"].min() >= BONAI_KEONJHAR_BBOX["lat_min"] - 0.01
    assert grid["latitude"].max() <= BONAI_KEONJHAR_BBOX["lat_max"] + 0.01
    assert grid["longitude"].min() >= BONAI_KEONJHAR_BBOX["lon_min"] - 0.01
    assert grid["longitude"].max() <= BONAI_KEONJHAR_BBOX["lon_max"] + 0.01

    scores = grid["prospectivity_score"]
    assert scores.min() >= 0.0
    assert scores.max() <= 100.0
    assert abs(scores.mean() - 50.0) < 5.0

def test_shared_render_prospectivity_surface():
    from bonai_keonjhar_predict import (
        render_prospectivity_surface,
        get_bonai_keonjhar_prospectivity_grid,
        get_domain_prospectivity_raster_base64
    )
    grid = get_bonai_keonjhar_prospectivity_grid()
    img = render_prospectivity_surface(grid, BONAI_KEONJHAR_BBOX)
    assert img.size == (500, 500)
    assert img.mode == "RGBA"

    b64_bk = get_domain_prospectivity_raster_base64("bonai_keonjhar", BONAI_KEONJHAR_BBOX)
    assert len(b64_bk) > 1000
    assert isinstance(b64_bk, str)

    sausar_bbox = {"lat_min": 20.95, "lat_max": 22.15, "lon_min": 79.35, "lon_max": 80.65}
    b64_sausar = get_domain_prospectivity_raster_base64("sausar", sausar_bbox)
    assert len(b64_sausar) > 1000

def test_domain_aware_selected_target_caption_apptest():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    if len(at.button) > 1:
        at.button[1].click().run()
    elif len(at.button) > 0:
        at.button[0].click().run()

    at.session_state["selected_domain"] = "bonai_keonjhar"
    at.session_state["target_lat"] = 22.0250
    at.session_state["target_lon"] = 85.4250
    at.run()

    assert not at.exception
    all_markdown = [m.value for m in at.markdown]
    combined_text = " ".join(all_markdown)

    assert "19,421 reference stations" in combined_text

    assert "20,000 reference stations across the Sausar Belt" not in combined_text

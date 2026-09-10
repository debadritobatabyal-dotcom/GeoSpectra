import os
from streamlit.testing.v1 import AppTest
import feature_pipeline
from feature_pipeline import SatelliteUnavailableError

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

def test_login_rejects_empty_credentials():
    """Verify that clicking Sign In with empty credentials rejects login."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    # Find text inputs
    at.text_input[0].input("").run()
    at.text_input[1].input("").run()
    at.button[0].click().run()
    assert at.session_state["authenticated"] is False
    assert len(at.error) > 0

def test_login_accepts_demo_access():
    """Verify that clicking Demo Access authenticates successfully."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button[1].click().run()  # Demo Access button
    assert at.session_state["authenticated"] is True

def test_out_of_domain_card_never_shows_percentage():
    """Verify that out of domain coordinates show guardrail and no percentage."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button[1].click().run()  # Authenticate
    # Set coordinates to Mumbai (18.92N, 72.83E) directly in session state
    at.session_state.target_lat = 18.9220
    at.session_state.target_lon = 72.8347
    at.run()
    
    # Verify OUT OF DOMAIN appears in markdown and no prospectivity % score
    markdown_texts = [m.value for m in at.markdown]
    has_ood = any("OUT OF STUDY DOMAIN" in text or "OUT OF DOMAIN" in text for text in markdown_texts)
    assert has_ood is True
    # Verify no prospectivity score like "83.5%" appears for out of domain
    has_pct_score = any("Manganese Prospectivity Score" in text for text in markdown_texts)
    assert has_pct_score is False

def test_prediction_card_renders_for_in_domain_point(monkeypatch):
    """Verify prediction card renders with prospectivity score when in domain."""
    test_lat, test_lon = 21.8333, 80.2333
    test_feats = {
        "latitude": test_lat, "longitude": test_lon,
        "B2": 0.065, "B3": 0.088, "B4": 0.099, "B5": 0.137, "B6": 0.180, "B7": 0.197, "B8": 0.206, "B8A": 0.216,
        "B11": 0.204, "B12": 0.145, "NDVI": 0.35, "NDMI": 0.004, "MNDWI": -0.40, "BSI": 0.057,
        "B4_B2_ratio": 1.53, "B11_B12_ratio": 1.41, "B12_B8_ratio": 0.70, "B11_B8_ratio": 0.99,
        "gossan_alteration_index": 1.12, "red_edge_ratio_1": 1.32, "red_edge_ratio_2": 1.44,
        "VV": -10.59, "VH": -18.32, "VV_VH_ratio": 7.73, "radar_backscatter_mean": -14.45,
        "radar_texture": 1.55,
        "elevation_m": 304.9, "slope_deg": 1.45, "curvature": 0.0,
        "terrain_ruggedness": 1.16, "local_relief_m": 6.55,
        "topographic_position_index": -1.51, "valley_or_ridge_class": "slope"
    }
    feature_pipeline.save_cached_features(test_lat, test_lon, test_feats, "known_occurrence_match", "cache_hit")

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button[1].click().run()  # Authenticate
    at.session_state.target_lat = test_lat
    at.session_state.target_lon = test_lon
    at.run()

    markdown_texts = [m.value for m in at.markdown]
    has_score_card = any("Manganese Prospectivity" in text or "Prospectivity Score" in text for text in markdown_texts)
    assert has_score_card is True

def test_tabs_and_top_targets_render():
    """Verify that all tabs, KPIs, and top exploration targets render without error."""
    test_lat, test_lon = 21.8333, 80.2333
    test_feats = {
        "latitude": test_lat, "longitude": test_lon,
        "B2": 0.065, "B3": 0.088, "B4": 0.099, "B5": 0.137, "B6": 0.180, "B7": 0.197, "B8": 0.206, "B8A": 0.216,
        "B11": 0.204, "B12": 0.145, "NDVI": 0.35, "NDMI": 0.004, "MNDWI": -0.40, "BSI": 0.057,
        "B4_B2_ratio": 1.53, "B11_B12_ratio": 1.41, "B12_B8_ratio": 0.70, "B11_B8_ratio": 0.99,
        "gossan_alteration_index": 1.12, "red_edge_ratio_1": 1.32, "red_edge_ratio_2": 1.44,
        "VV": -10.59, "VH": -18.32, "VV_VH_ratio": 7.73, "radar_backscatter_mean": -14.45,
        "radar_texture": 1.55,
        "elevation_m": 304.9, "slope_deg": 1.45, "curvature": 0.0,
        "terrain_ruggedness": 1.16, "local_relief_m": 6.55,
        "topographic_position_index": -1.51, "valley_or_ridge_class": "slope"
    }
    feature_pipeline.save_cached_features(test_lat, test_lon, test_feats, "known_occurrence_match", "cache_hit")

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button[1].click().run()  # Demo Access
    
    # Check KPIs
    markdown_texts = [m.value for m in at.markdown]
    assert any("Sausar Manganese Belt" in text for text in markdown_texts)
    assert any("HistGradientBoosting" in text for text in markdown_texts)
    
    # Check Top Exploration Targets dataset is loaded
    assert any("Top Exploration Targets" in text or "TARGET" in text for text in markdown_texts)
    
    # Check Why This Target Is Prospective bars render
    assert any("Why This Target is Prospective" in text or "WHY THIS TARGET" in text.upper() for text in markdown_texts)
    assert any("GEOLOGY" in text for text in markdown_texts)
    assert any("SPECTRAL SIGNATURE" in text for text in markdown_texts)



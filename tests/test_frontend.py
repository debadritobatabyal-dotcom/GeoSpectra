import os
from streamlit.testing.v1 import AppTest
import feature_pipeline
from feature_pipeline import SatelliteUnavailableError

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

def test_login_rejects_empty_credentials():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    at.text_input[0].input("").run()
    at.text_input[1].input("").run()
    at.button[0].click().run()
    assert at.session_state["authenticated"] is False
    assert len(at.error) > 0

def test_login_accepts_demo_access():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button[1].click().run()
    assert at.session_state["authenticated"] is True

def test_out_of_domain_card_never_shows_percentage():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button[1].click().run()

    at.session_state.target_lat = 18.9220
    at.session_state.target_lon = 72.8347
    at.run()

    markdown_texts = [m.value for m in at.markdown]
    has_ood = any("OUT OF STUDY DOMAIN" in text or "OUT OF DOMAIN" in text for text in markdown_texts)
    assert has_ood is True

    has_pct_score = any("Manganese Prospectivity Score" in text for text in markdown_texts)
    assert has_pct_score is False

def test_prediction_card_renders_for_in_domain_point(monkeypatch):
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
    at.button[1].click().run()
    at.session_state.target_lat = test_lat
    at.session_state.target_lon = test_lon
    at.run()

    markdown_texts = [m.value for m in at.markdown]
    has_score_card = any("Manganese Prospectivity" in text or "Prospectivity Score" in text for text in markdown_texts)
    assert has_score_card is True

def test_tabs_and_top_targets_render():
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
    at.button[1].click().run()

    markdown_texts = [m.value for m in at.markdown]
    assert any("Sausar Manganese Belt" in text for text in markdown_texts)
    assert any("HistGradientBoosting" in text for text in markdown_texts)

    assert any("Top Exploration Targets" in text or "TARGET" in text for text in markdown_texts)

    assert any("Why This Target is Prospective" in text or "WHY THIS TARGET" in text.upper() for text in markdown_texts)
    assert any("GEOLOGY" in text for text in markdown_texts)
    assert any("SPECTRAL SIGNATURE" in text for text in markdown_texts)

def test_editorial_redesign_components():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button[1].click().run()

    markdown_texts = [m.value for m in at.markdown]

    assert any("GEOSPECTRA" in text and "NATIONAL MINERAL INTELLIGENCE" in text for text in markdown_texts)
    assert any("From Earth observation to operational intelligence." in text for text in markdown_texts)

    assert any("Target Priority" in text or "TARGET PRIORITY" in text.upper() for text in markdown_texts)
    assert any("01" in text for text in markdown_texts)

    assert any("0.892" in text for text in markdown_texts)
    assert any("80.58%" in text for text in markdown_texts)
    assert any("99.4" in text for text in markdown_texts)

    assert any("01" in text and "EARTH OBSERVATION" in text for text in markdown_texts)
    assert any("02" in text and "GEOLOGICAL DATA" in text for text in markdown_texts)
    assert any("03" in text and "EXPLORATION MODEL" in text for text in markdown_texts)
    assert any("04" in text and "PRODUCTION MODEL" in text for text in markdown_texts)
    assert any("05" in text and "EXPLAINABILITY" in text for text in markdown_texts)
    assert any("06" in text and "DECISION SUPPORT" in text for text in markdown_texts)
    assert any("07" in text and "SCENARIO SIMULATION" in text for text in markdown_texts)

def test_production_intelligence_tab_renders():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button[1].click().run()

    markdown_texts = [m.value for m in at.markdown]
    assert any("Production Intelligence" in text for text in markdown_texts)
    assert any("XGBoost Expected Production" in text or "XGBoost" in text for text in markdown_texts)
    assert any("DEMO / SYNTHETIC OPERATIONAL DATA" in text.upper() for text in markdown_texts)
    assert any("HOW DO WE RECOVER" in text.upper() or "RECOVERY SIMULATOR" in text.upper() for text in markdown_texts)

def test_login_rejects_invalid_email():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.text_input[0].input("invalid-email-address").run()
    at.button[0].click().run()
    assert at.session_state["authenticated"] is False
    assert at.session_state["auth_step"] == "email"
    assert len(at.error) > 0

def test_login_otp_flow_and_verification():
    import auth_service
    test_email = "test.officer@geology.gov.in"
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    at.text_input[0].input(test_email).run()
    at.button[0].click().run()
    assert at.session_state["auth_step"] == "otp"
    assert at.session_state["auth_email"] == test_email

    at.text_input[0].input("000000").run()
    at.button[0].click().run()
    assert at.session_state["authenticated"] is False
    assert len(at.error) > 0

    active_code = auth_service.get_active_code_hint(test_email)
    assert active_code is not None
    at.text_input[0].input(active_code).run()
    at.button[0].click().run()
    assert at.session_state["authenticated"] is True
    assert at.session_state["user_email"] == test_email

def test_login_otp_change_email():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.text_input[0].input("officer@geology.gov.in").run()
    at.button[0].click().run()
    assert at.session_state["auth_step"] == "otp"

    change_btn = [b for b in at.button if "CHANGE" in b.label.upper() or "बदलें" in b.label][0]
    change_btn.click().run()
    assert at.session_state["auth_step"] == "email"

def test_login_page_theme_and_language_toggles():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert at.session_state["theme"] == "light"
    assert at.session_state["language"] == "en"

    thm_radio = [r for r in at.radio if r.key == "login_theme_selector"][0]
    thm_radio.set_value("🌙 Dark").run()
    assert at.session_state["theme"] == "dark"

    lng_radio = [r for r in at.radio if r.key == "login_lang_selector"][0]
    lng_radio.set_value("हिन्दी").run()
    assert at.session_state["language"] == "hi"

    markdowns = [m.value for m in at.markdown]
    assert any("जियोस्पेक्ट्रा" in m for m in markdowns)

    lng_radio = [r for r in at.radio if r.key == "login_lang_selector"][0]
    lng_radio.set_value("EN").run()
    assert at.session_state["language"] == "en"

    thm_radio = [r for r in at.radio if r.key == "login_theme_selector"][0]
    thm_radio.set_value("☀️ Light").run()
    assert at.session_state["theme"] == "light"

def test_study_domain_selectbox_switching():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button[1].click().run()
    assert at.session_state["authenticated"] is True
    assert at.session_state["selected_domain"] == "sausar"

    domain_select = [s for s in at.selectbox if s.key == "top_domain_selector"][0]
    assert domain_select.value == "sausar"

    domain_select.select("bonai_keonjhar").run()
    assert at.session_state["selected_domain"] == "bonai_keonjhar"
    assert at.session_state["target_lat"] == 22.0250
    assert at.session_state["target_lon"] == 85.4250

    markdowns_bonai = [m.value for m in at.markdown]
    assert any("Bonai–Keonjhar" in m for m in markdowns_bonai)
    assert any("19,421 Stations" in m or "19,421 संदर्भ स्थल" in m for m in markdowns_bonai)

    domain_select = [s for s in at.selectbox if s.key == "top_domain_selector"][0]
    domain_select.select("sausar").run()
    assert at.session_state["selected_domain"] == "sausar"
    assert at.session_state["target_lat"] == 21.8333
    assert at.session_state["target_lon"] == 80.2333

    markdowns_sausar = [m.value for m in at.markdown]
    assert any("Sausar Belt" in m or "सॉसार बेल्ट" in m for m in markdowns_sausar)
    assert any("11 MOIL Localities" in m or "11 प्रमाणित स्थल" in m for m in markdowns_sausar)

def test_study_domain_css_and_single_source_of_truth():
    import theme
    css_dark = theme.generate_css("dark")
    css_light = theme.generate_css("light")

    assert "geospectra-domain-anchor" in css_dark
    assert "geospectra-domain-anchor" in css_light
    assert 'div[data-baseweb="popover"]' in css_dark
    assert 'div[data-baseweb="popover"]' in css_light
    assert 'border-radius: 9999px !important;' in css_dark
    assert 'max-width: 350px !important;' in css_dark
    assert '✓' in css_dark
    assert '✓' in css_light
    # Verify no position: relative on option items (which causes react-window gap bugs)
    assert 'position: relative !important;' not in css_dark
    assert 'position: relative !important;' not in css_light
    assert 'height: 40px !important;' in css_dark
    assert 'padding: 0 !important;' in css_dark



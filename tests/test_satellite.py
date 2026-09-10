import os
import pytest
from feature_pipeline import (
    SatelliteUnavailableError,
    get_cached_features,
    save_cached_features,
    extract_features,
    initialize_gee
)
from predict import predict_single_location

def test_no_satellite_simulator_reachable():
    """Verify that no synthetic satellite simulator function exists in the authoritative feature pipeline."""
    import feature_pipeline
    assert not hasattr(feature_pipeline, "extract_demo_simulator_features"), "Fake satellite simulator must not exist!"
    assert not hasattr(feature_pipeline, "synthetic_simulator"), "Fake satellite simulator must not exist!"

def test_cache_hit_avoids_second_gee_call():
    """Verify that cached feature vectors return satellite_source='cache_hit'."""
    test_lat, test_lon = 21.85, 80.20
    test_feats = {
        "latitude": test_lat,
        "longitude": test_lon,
        "B2": 0.08,
        "VV": -12.0
    }
    save_cached_features(test_lat, test_lon, test_feats, "test_geo", "GEE_live")
    cached = get_cached_features(test_lat, test_lon)
    assert cached is not None
    assert cached["satellite_source"] == "cache_hit"
    assert cached["B2"] == 0.08

def test_gee_unavailable_raises_satellite_unavailable_error(monkeypatch):
    """Verify that when GEE is unavailable, SatelliteUnavailableError is raised (no fake fallback)."""
    import feature_pipeline
    monkeypatch.setattr(feature_pipeline, "initialize_gee", lambda project_id=None: (False, "Earth Engine credentials not found"))
    with pytest.raises(SatelliteUnavailableError) as exc_info:
        feature_pipeline.extract_real_gee_satellite(21.8333, 80.2333)
    assert "Earth Engine credentials not found" in str(exc_info.value)

def test_initialize_gee_missing_credentials_returns_clear_error(monkeypatch):
    """Verify that initialize_gee returns clean instructions when credentials do not exist."""
    import feature_pipeline
    import ee
    feature_pipeline._EE_INITIALIZED = False

    def mock_init(*args, **kwargs):
        raise ee.EEException("Please authorize access to your Earth Engine account by running ee.Authenticate()")

    monkeypatch.setattr(ee, "Initialize", mock_init)
    success, msg = feature_pipeline.initialize_gee()
    assert success is False
    assert "ee.Authenticate()" in msg or "authenticate_gee.py" in msg

def test_initialize_gee_success_when_mocked(monkeypatch):
    """Verify that initialize_gee succeeds and updates state when API calls succeed."""
    import feature_pipeline
    import ee
    feature_pipeline._EE_INITIALIZED = False

    monkeypatch.setattr(ee, "Initialize", lambda *args, **kwargs: None)
    
    class MockNumber:
        def __init__(self, val): self.val = val
        def getInfo(self): return self.val

    monkeypatch.setattr(ee, "Number", lambda val: MockNumber(val))

    success, msg = feature_pipeline.initialize_gee("test-project-123")
    assert success is True
    assert "test-project-123" in msg
    assert feature_pipeline._EE_INITIALIZED is True

def test_satellite_failure_never_calls_model(monkeypatch):
    """Verify that when satellite data fails to extract, predict_single_location returns SATELLITE_UNAVAILABLE."""
    import feature_pipeline
    def mock_extract(lat, lon):
        raise SatelliteUnavailableError("Simulated GEE offline failure")

    monkeypatch.setattr(feature_pipeline, "extract_features", mock_extract)
    # Clear cache for this test coord
    test_lat, test_lon = 21.7777, 80.1111
    monkeypatch.setattr(feature_pipeline, "get_cached_features", lambda lat, lon: None)

    res = predict_single_location({"latitude": test_lat, "longitude": test_lon})
    assert res["status"] == "SATELLITE_UNAVAILABLE"
    assert res["prospectivity_probability"] is None
    assert res["classification"] == "SATELLITE UNAVAILABLE"

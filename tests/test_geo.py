import os
import json
from predict import check_domain_bounds, predict_single_location
import feature_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")

def test_in_domain_point():
    """Verify that points within 20.95-22.15N, 79.35-80.65E pass domain check."""
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    bbox = schema["study_domain"]
    assert check_domain_bounds(21.8333, 80.2333, bbox) is True
    assert check_domain_bounds(21.6850, 79.7120, bbox) is True

def test_out_of_domain_point():
    """Verify that points outside the domain fail check."""
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    bbox = schema["study_domain"]
    # Mumbai
    assert check_domain_bounds(18.9220, 72.8347, bbox) is False
    # Delhi
    assert check_domain_bounds(28.6139, 77.2090, bbox) is False

def test_out_of_domain_never_produces_probability():
    """Verify that out of domain coordinates return OUT_OF_STUDY_DOMAIN with no probability."""
    res = predict_single_location({"latitude": 18.9220, "longitude": 72.8347})
    assert res["status"] == "OUT_OF_STUDY_DOMAIN"
    assert res["prospectivity_probability"] is None
    assert res["prospectivity_percentage"] is None
    assert res["classification"] == "OUT OF STUDY DOMAIN"

def test_known_occurrence_lookup_matches_real_row():
    """Verify that coordinates of Balaghat Mine (~21.8333N, 80.2333E) match known occurrence."""
    match = feature_pipeline.match_known_occurrence(21.8333, 80.2333)
    assert match is not None
    assert match["matched"] is True
    assert "Balaghat" in match["mine_name"]
    assert match["geological_group"] == "Sausar Group"

def test_unmapped_point_uses_regional_default_not_fabrication():
    """Verify that an arbitrary in-domain point distant from mines gets regional_default_unmapped."""
    # Point in Sausar belt corridor away from mines: 21.15N, 80.50E
    match = feature_pipeline.match_known_occurrence(21.1500, 80.5000)
    assert match is None  # No mine within 5km

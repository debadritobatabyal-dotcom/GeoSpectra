import os
import json
from predict import check_domain_bounds, predict_single_location
import feature_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")

def test_in_domain_point():
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    bbox = schema["study_domain"]
    assert check_domain_bounds(21.8333, 80.2333, bbox) is True
    assert check_domain_bounds(21.6850, 79.7120, bbox) is True

def test_out_of_domain_point():
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    bbox = schema["study_domain"]

    assert check_domain_bounds(18.9220, 72.8347, bbox) is False

    assert check_domain_bounds(28.6139, 77.2090, bbox) is False

def test_out_of_domain_never_produces_probability():
    res = predict_single_location({"latitude": 18.9220, "longitude": 72.8347})
    assert res["status"] == "OUT_OF_STUDY_DOMAIN"
    assert res["prospectivity_probability"] is None
    assert res["prospectivity_percentage"] is None
    assert res["classification"] == "OUT OF STUDY DOMAIN"

def test_known_occurrence_lookup_matches_real_row():
    match = feature_pipeline.match_known_occurrence(21.8333, 80.2333)
    assert match is not None
    assert match["matched"] is True
    assert "Balaghat" in match["mine_name"]
    assert match["geological_group"] == "Sausar Group"

def test_unmapped_point_uses_regional_default_not_fabrication():

    match = feature_pipeline.match_known_occurrence(21.1500, 80.5000)
    assert match is None

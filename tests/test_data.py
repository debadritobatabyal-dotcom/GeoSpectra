import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset", "sausar_manganese_core_features.csv")
SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")

def test_core_dataset_loads():
    assert os.path.exists(DATA_PATH), f"Missing dataset at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    assert len(df) == 20000, f"Expected 20,000 rows, got {len(df)}"
    assert "label_status" in df.columns

def test_label_distribution():
    df = pd.read_csv(DATA_PATH)
    counts = df["label_status"].value_counts().to_dict()
    assert counts.get("POSITIVE", 0) == 1273
    assert counts.get("UNLABELLED", 0) == 17610
    assert counts.get("UNCERTAIN", 0) == 1117

def test_no_banned_columns_in_feature_schema():
    assert os.path.exists(SCHEMA_PATH)
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)

    features = schema["primary_features"]
    assert len(features) == 43, f"Expected 43 authoritative features, got {len(features)}"

    assert "latitude" in features
    assert "longitude" in features

    banned = [
        "distance_to_nearest_known_manganese_occurrence_km",
        "manganese_bearing_horizon_indicator",
        "legacy_report_manganese_mention",
        "legacy_report_evidence_score",
        "sample_id",
        "spatial_block_id",
        "regional_zone_id",
        "label_status",
        "manganese_associated",
        "data_quality_flag",
        "feature_distribution_distance",
        "temporary_binary_label",
        "synthetic_ground_truth_probability",
        "synthetic_ood_score",
        "aspect_deg",
        "aspect_sin",
        "aspect_cos",
        "radar_roughness_proxy",
        "aster_vnir_swir_proxy",
        "aster_thermal_lithology_proxy",
        "multisensor_spectral_anomaly_score",
        "drainage_density",
        "drainage_anomaly_score",
        "distance_to_fault_km",
        "distance_to_lineament_km",
        "shear_zone_proximity_km",
        "fold_hinge_proximity_km",
        "alteration_intensity",
        "geological_complexity_score",
        "structural_complexity_score",
        "ore_host_compatibility_score",
        "formation_contact_distance_km",
        "geological_age_ma",
        "distance_to_major_geological_contact_km",
        "host_rock",
    ]
    for b in banned:
        assert b not in features, f"Banned/unreproducible column {b} found in primary features!"

def test_real_dataset_never_enters_training():
    complete_path = os.path.join(BASE_DIR, "data", "sausar_manganese_prospectivity_dataset_complete.csv")
    assert os.path.exists(complete_path)
    comp_df = pd.read_csv(complete_path)
    assert len(comp_df) == 351
    df = pd.read_csv(DATA_PATH)
    assert len(df) == 20000

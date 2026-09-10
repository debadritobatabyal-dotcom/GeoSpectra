import os
import joblib
import pandas as pd
import numpy as np
from predict import load_model_bundle, predict_single_location

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "manganese_prospectivity_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset", "sausar_manganese_core_features.csv")

def test_model_serializes_and_reloads():
    """Verify that model bundle exists, reloads cleanly, and has required components."""
    assert os.path.exists(MODEL_PATH)
    bundle = load_model_bundle(MODEL_PATH)
    assert "model" in bundle
    assert "feature_cols" in bundle
    assert len(bundle["feature_cols"]) == 43
    assert "cat_mappings" in bundle
    assert "probability_thresholds" in bundle
    assert "applicability" in bundle
    assert "s2_positive_mean" in bundle
    assert "s2_unlabelled_mean" in bundle
    assert "radar_positive_mean" in bundle

def test_no_banned_feature_enters_primary_model():
    """Verify that primary model feature list contains zero leakage, proxy, or aspect features."""
    bundle = load_model_bundle(MODEL_PATH)
    cols = bundle["feature_cols"]
    assert "distance_to_nearest_known_manganese_occurrence_km" not in cols
    assert "manganese_bearing_horizon_indicator" not in cols
    assert "synthetic_ground_truth_probability" not in cols
    assert "aspect_sin" not in cols
    assert "aspect_cos" not in cols
    assert "aspect_deg" not in cols
    assert "radar_roughness_proxy" not in cols
    assert "aster_vnir_swir_proxy" not in cols
    assert "distance_to_fault_km" not in cols
    assert "latitude" in cols
    assert "longitude" in cols

def test_predict_positive_and_unlabelled_samples():
    """Verify that known positive samples score significantly higher on average than unlabelled background."""
    bundle = load_model_bundle(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    pos_df = df[df["label_status"] == "POSITIVE"].sample(20, random_state=42)
    unl_df = df[df["label_status"] == "UNLABELLED"].sample(20, random_state=42)

    pos_probs = []
    for _, row in pos_df.iterrows():
        res = predict_single_location(row.to_dict(), bundle=bundle)
        assert res["status"] == "SUCCESS"
        pos_probs.append(res["prospectivity_probability"])

    unl_probs = []
    for _, row in unl_df.iterrows():
        res = predict_single_location(row.to_dict(), bundle=bundle)
        assert res["status"] == "SUCCESS"
        unl_probs.append(res["prospectivity_probability"])

    mean_pos = np.mean(pos_probs)
    mean_unl = np.mean(unl_probs)
    assert mean_pos > mean_unl, f"Expected POSITIVE mean ({mean_pos:.3f}) > UNLABELLED mean ({mean_unl:.3f})"

def test_aspect_invariance():
    """Verify that slope aspect direction has zero impact on prospectivity score (aspect invariance)."""
    bundle = load_model_bundle(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    base_sample = df.iloc[0].to_dict()

    # Modified aspect (North vs South facing) - even if passed in input, aspect is excluded from model
    sample_north = base_sample.copy()
    sample_north["aspect_deg"] = 0.0
    sample_north["aspect_cos"] = 1.0

    sample_south = base_sample.copy()
    sample_south["aspect_deg"] = 180.0
    sample_south["aspect_cos"] = -1.0

    res_north = predict_single_location(sample_north, bundle=bundle)
    res_south = predict_single_location(sample_south, bundle=bundle)

    assert res_north["status"] == "SUCCESS"
    assert res_south["status"] == "SUCCESS"
    assert abs(res_north["prospectivity_probability"] - res_south["prospectivity_probability"]) < 1e-6


def test_applicability_scoring():
    """Verify that IsolationForest applicability model assigns valid scores and status categories."""
    bundle = load_model_bundle(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    sample = df.iloc[0].to_dict()

    res = predict_single_location(sample, bundle=bundle)
    assert res["status"] == "SUCCESS"
    assert res["applicability_status"] in ["HIGH APPLICABILITY", "MODERATE APPLICABILITY", "LOW APPLICABILITY"]
    assert isinstance(res["applicability_score"], float)


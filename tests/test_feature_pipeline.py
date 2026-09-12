import os
import json
import numpy as np
import pandas as pd
from predict import encode_vector_for_inference

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")

def test_train_infer_schema_identical():
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    assert len(schema["primary_features"]) == 43
    assert len(schema["categorical_features"]) == 9
    assert len(schema["numeric_features"]) == 34
    assert "valley_or_ridge_class" in schema["categorical_features"]
    assert "geological_group" in schema["categorical_features"]

def test_missing_feature_becomes_nan_not_silently_filled():
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    feature_cols = schema["primary_features"]
    cat_cols = schema["categorical_features"]
    cat_mappings = {c: {"ridge": 0, "slope": 1, "valley": 2} for c in cat_cols}

    sample_dict = {"latitude": 21.89, "longitude": 80.22}
    row_df = encode_vector_for_inference(sample_dict, feature_cols, cat_cols, cat_mappings)

    assert len(row_df.columns) == 43
    assert pd.isna(row_df.loc[0, "elevation_m"])
    assert pd.isna(row_df.loc[0, "B2"])
    assert pd.isna(row_df.loc[0, "VV"])
    assert not np.isfinite(row_df.loc[0, "elevation_m"])

def test_saved_category_mappings_are_reused():
    cat_cols = ["valley_or_ridge_class"]
    cat_mappings = {"valley_or_ridge_class": {"ridge": 2, "slope": 1, "valley": 0}}
    sample_dict = {"valley_or_ridge_class": "ridge"}
    row_df = encode_vector_for_inference(sample_dict, ["valley_or_ridge_class"], cat_cols, cat_mappings)
    assert row_df.loc[0, "valley_or_ridge_class"] == 2.0

def test_unknown_category_maps_to_minus_one():
    cat_cols = ["valley_or_ridge_class"]
    cat_mappings = {"valley_or_ridge_class": {"ridge": 0, "slope": 1, "valley": 2}}
    sample_dict = {"valley_or_ridge_class": "plateau_canyon"}
    row_df = encode_vector_for_inference(sample_dict, ["valley_or_ridge_class"], cat_cols, cat_mappings)
    assert row_df.loc[0, "valley_or_ridge_class"] == -1.0

def test_real_vs_synthetic_band_ranges_compatible():
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    for band in ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]:
        rng = schema["feature_specifications"][band]["valid_range"]
        assert rng["min"] >= 0.0
        assert rng["max"] <= 1.0

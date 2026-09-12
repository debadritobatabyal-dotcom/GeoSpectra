import os
import json
import pytest
import pandas as pd
import numpy as np

from production.production_predict import (
    production_predict,
    load_production_model_bundle,
    classify_risk,
    RISK_THRESHOLDS,
)
from production.scenarios import (
    simulate_scenario,
    list_scenarios,
    SCENARIOS
)
from production.recovery import (
    simulate_recovery_scenario,
    list_recovery_scenarios,
    RECOVERY_SCENARIOS
)
from production.recommendations import generate_recommendations

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "production", "production_xgboost_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "production", "production_metrics.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "production", "synthetic_production_dataset_fixed.csv")

@pytest.fixture(scope="module")
def sample_dataset():
    assert os.path.exists(DATA_PATH), f"Dataset missing at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    return df

@pytest.fixture(scope="module")
def sample_record(sample_dataset):
    return sample_dataset.iloc[120].to_dict()

def test_model_bundle_integrity():
    assert os.path.exists(MODEL_PATH), f"Model bundle missing at {MODEL_PATH}"
    assert os.path.exists(METRICS_PATH), f"Metrics JSON missing at {METRICS_PATH}"

    bundle = load_production_model_bundle(MODEL_PATH)
    assert "model" in bundle
    assert "feature_names" in bundle
    assert "metrics" in bundle
    assert "temporal_holdout" in bundle

    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    assert "xgboost" in metrics["models"]
    assert "random_forest_benchmark" in metrics["models"]
    assert metrics["models"]["xgboost"]["metrics"]["r2"] > 0.92
    assert metrics["models"]["xgboost"]["metrics"]["mae"] < 130.0

def test_zero_target_leakage():
    bundle = load_production_model_bundle(MODEL_PATH)
    feature_names = bundle["feature_names"]

    forbidden_terms = [
        "actual_production", "actual_production_tpd",
        "planned_production", "planned_production_tpd",
        "shortfall", "shortfall_tonnes", "shortfall_pct",
        "risk", "risk_level", "target_variance"
    ]
    for term in forbidden_terms:
        assert term not in feature_names, f"Target leakage detected! '{term}' found in model features."

def test_single_source_of_truth_baseline(sample_record):

    main_pred = production_predict(sample_record, include_explain=True)
    baseline_expected = main_pred["expected_production_tpd"]

    hazard_res = simulate_scenario(
        sample_record, "heavy_rainfall",
        lambda x: production_predict(x, include_explain=False),
        baseline_prediction=main_pred
    )
    assert hazard_res["baseline_prediction"]["expected_production_tpd"] == baseline_expected

    recovery_res = simulate_recovery_scenario(
        sample_record, "improve_shovel_availability",
        lambda x: production_predict(x, include_explain=False),
        baseline_prediction=main_pred
    )
    assert recovery_res["baseline_expected_tpd"] == baseline_expected

def test_target_variance_terminology(sample_record):

    low_input = dict(sample_record)
    low_input["planned_production_tpd"] = 3500.0
    res_low = production_predict(low_input, include_explain=False)
    assert res_low["target_variance_tpd"] < 0.0
    assert res_low["target_variance_tpd"] == pytest.approx(
        res_low["expected_production_tpd"] - 3500.0, abs=0.1
    )
    assert res_low["shortfall_tonnes"] == pytest.approx(-res_low["target_variance_tpd"], abs=0.1)
    assert res_low["on_target"] is False

    high_input = dict(sample_record)
    high_input["planned_production_tpd"] = 500.0
    res_high = production_predict(high_input, include_explain=False)
    assert res_high["target_variance_tpd"] > 0.0
    assert res_high["shortfall_tonnes"] == 0.0
    assert res_high["shortfall_pct"] == 0.0
    assert res_high["on_target"] is True
    assert res_high["risk_level"] == "LOW"

def test_risk_classification_prototype_thresholds():
    low_level, _, _ = classify_risk(3.0)
    assert low_level == "LOW"

    med_level, _, _ = classify_risk(9.5)
    assert med_level == "MEDIUM"

    high_level, _, _ = classify_risk(16.0)
    assert high_level == "HIGH"

def test_shap_explainability_and_cautious_language(sample_record):
    result = production_predict(sample_record, include_explain=True)
    shap_data = result["shap_drivers"]

    assert "base_value_tpd" in shap_data
    assert "top_negative_drivers" in shap_data
    assert "top_positive_drivers" in shap_data
    assert "disclaimer" in shap_data

    disclaimer = shap_data["disclaimer"]
    assert "Model-associated" in disclaimer or "model-associated" in disclaimer
    assert "causal" in disclaimer.lower()

def test_controllable_vs_external_driver_categorization(sample_record):
    result = production_predict(sample_record, include_explain=True)
    shap_data = result["shap_drivers"]

    assert "controllable_negative_drivers" in shap_data
    assert "external_negative_drivers" in shap_data

    for d in shap_data["all_contributions"]:
        if d["feature"] in ["active_shovels", "queue_delay_min", "haulage_efficiency_pct"]:
            assert d["is_controllable"] is True
        elif d["feature"] in ["rainfall_mm", "weather_disruption_hours", "ground_condition_index"]:
            assert d["is_controllable"] is False

def test_shap_informed_recommendations(sample_record):
    result = production_predict(sample_record, include_explain=True)
    recs = result["recommendations"]
    assert len(recs) > 0

    all_actions = " ".join(r["action"] for r in recs)
    assert any(word in all_actions for word in ["Consider", "Evaluate", "Review", "Prioritize"])
    assert "guarantee" not in all_actions.lower()

def test_hazard_scenarios_rerun_model(sample_record):
    scenarios = list_scenarios()
    assert len(scenarios) == 4

    main_pred = production_predict(sample_record, include_explain=False)
    base_tonnes = main_pred["expected_production_tpd"]

    sim_rain = simulate_scenario(
        sample_record, "heavy_rainfall",
        lambda x: production_predict(x, include_explain=False),
        baseline_prediction=main_pred
    )
    assert sim_rain["scenario_prediction"]["expected_production_tpd"] < base_tonnes
    assert sim_rain["delta_tonnes"] < 0.0

    sim_shovel = simulate_scenario(
        sample_record, "shovel_breakdown",
        lambda x: production_predict(x, include_explain=False),
        baseline_prediction=main_pred
    )
    assert sim_shovel["scenario_prediction"]["expected_production_tpd"] < base_tonnes

def test_recovery_scenarios_rerun_model_and_bound_limits(sample_record):
    recovery_scenarios = list_recovery_scenarios()
    assert len(recovery_scenarios) == 5

    sub_target_record = dict(sample_record)
    sub_target_record["planned_production_tpd"] = 3200.0
    sub_target_record["active_shovels"] = 2
    sub_target_record["available_shovels"] = 2
    sub_target_record["total_shovels"] = 4
    sub_target_record["queue_delay_min"] = 4.2

    main_pred = production_predict(sub_target_record, include_explain=False)
    base_tonnes = main_pred["expected_production_tpd"]
    base_shortfall = main_pred["shortfall_tonnes"]
    assert base_shortfall > 0.0

    rec_shovel = simulate_recovery_scenario(
        sub_target_record, "improve_shovel_availability",
        lambda x: production_predict(x, include_explain=False),
        baseline_prediction=main_pred
    )
    assert rec_shovel["scenario_expected_tpd"] > base_tonnes
    assert rec_shovel["recovery_tonnes"] > 0.0
    assert rec_shovel["remaining_shortfall_tonnes"] < base_shortfall

    for mod in rec_shovel["applied_modifications"]:
        if mod["parameter"] == "active_shovels":
            assert mod["modified_value"] <= sub_target_record["total_shovels"]

    rec_queue = simulate_recovery_scenario(
        sub_target_record, "reduce_queue_delay",
        lambda x: production_predict(x, include_explain=False),
        baseline_prediction=main_pred
    )
    assert rec_queue["recovery_tonnes"] > 0.0
    assert rec_queue["remaining_shortfall_tonnes"] <= base_shortfall

    rec_comb = simulate_recovery_scenario(
        sub_target_record, "combined_recovery",
        lambda x: production_predict(x, include_explain=False),
        baseline_prediction=main_pred
    )
    assert rec_comb["recovery_tonnes"] > 0.0
    assert rec_comb["scenario_expected_tpd"] > base_tonnes

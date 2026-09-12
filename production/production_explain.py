import logging
import numpy as np
import pandas as pd
import shap

LOGGER = logging.getLogger(__name__)

_EXPLAINER = None
_EXPLAINER_MODEL_ID = None

FEATURE_METADATA = {

    "active_shovels": {
        "label": "Active Shovel Count", "unit": "units", "category": "Equipment Fleet",
        "is_controllable": True, "actionable_lever": "Shovel face allocation & deployment"
    },
    "available_shovels": {
        "label": "Available Shovels", "unit": "units", "category": "Equipment Fleet",
        "is_controllable": True, "actionable_lever": "Preventative maintenance release"
    },
    "shovel_availability_pct": {
        "label": "Shovel Availability", "unit": "%", "category": "Equipment Fleet",
        "is_controllable": True, "actionable_lever": "Workshop turn-around prioritization"
    },
    "shovel_downtime_hours": {
        "label": "Shovel Downtime Duration", "unit": "hrs", "category": "Equipment Fleet",
        "is_controllable": True, "actionable_lever": "Maintenance response acceleration"
    },
    "shovel_breakdown_events": {
        "label": "Shovel Breakdown Incidents", "unit": "events", "category": "Equipment Fleet",
        "is_controllable": True, "actionable_lever": "Hydraulic / electrical pre-shift checks"
    },
    "shovel_utilization_pct": {
        "label": "Shovel Utilization", "unit": "%", "category": "Equipment Fleet",
        "is_controllable": True, "actionable_lever": "Face readiness and bench preparation"
    },
    "active_trucks": {
        "label": "Active Haul Truck Count", "unit": "units", "category": "Haulage Fleet",
        "is_controllable": True, "actionable_lever": "Fleet dispatch allocation"
    },
    "available_trucks": {
        "label": "Available Haul Trucks", "unit": "units", "category": "Haulage Fleet",
        "is_controllable": True, "actionable_lever": "Standby truck readiness"
    },
    "truck_availability_pct": {
        "label": "Truck Fleet Availability", "unit": "%", "category": "Haulage Fleet",
        "is_controllable": True, "actionable_lever": "Workshop servicing scheduling"
    },
    "truck_downtime_hours": {
        "label": "Truck Downtime Duration", "unit": "hrs", "category": "Haulage Fleet",
        "is_controllable": True, "actionable_lever": "Tyre / engine servicing queue"
    },
    "truck_utilization_pct": {
        "label": "Truck Utilization", "unit": "%", "category": "Haulage Fleet",
        "is_controllable": True, "actionable_lever": "Haul route traffic balancing"
    },
    "queue_delay_min": {
        "label": "Queue & Dispatch Bottleneck", "unit": "min", "category": "Dispatch & Traffic",
        "is_controllable": True, "actionable_lever": "Dynamic shovel-truck dispatch reallocation"
    },
    "average_cycle_time_min": {
        "label": "Truck Cycle Duration", "unit": "min", "category": "Haulage Dynamics",
        "is_controllable": True, "actionable_lever": "Haul route speed & ramp gradient optimization"
    },
    "haulage_efficiency_pct": {
        "label": "Haulage Operational Efficiency", "unit": "%", "category": "Haulage Dynamics",
        "is_controllable": True, "actionable_lever": "Haul road surface grading & rolling resistance"
    },
    "loading_efficiency_pct": {
        "label": "Shovel Loading Efficiency", "unit": "%", "category": "Operations",
        "is_controllable": True, "actionable_lever": "Bench floor grading & spot positioning"
    },
    "effective_operating_hours": {
        "label": "Effective Productive Hours", "unit": "hrs", "category": "Operations",
        "is_controllable": True, "actionable_lever": "Shift handover and lunch relay management"
    },
    "overall_utilization_pct": {
        "label": "Overall Fleet Utilization", "unit": "%", "category": "Operations",
        "is_controllable": True, "actionable_lever": "Synchronized fleet scheduling"
    },
    "blasting_delay_hours": {
        "label": "Blasting Clear & Prep Delay", "unit": "hrs", "category": "Operations",
        "is_controllable": True, "actionable_lever": "Align blast clearance with shift boundaries"
    },
    "maintenance_delay_hours": {
        "label": "Unscheduled Maintenance Delay", "unit": "hrs", "category": "Operations",
        "is_controllable": True, "actionable_lever": "Preventative parts staging & triage"
    },
    "truck_cycles": {
        "label": "Completed Truck Haul Cycles", "unit": "cycles", "category": "Haulage Dynamics",
        "is_controllable": True, "actionable_lever": "Shift cycle pacing and queue reduction"
    },

    "weather_disruption_hours": {
        "label": "Weather Disruption Duration", "unit": "hrs", "category": "Environmental Hazards",
        "is_controllable": False, "actionable_lever": "Wet-weather operating & safety protocols"
    },
    "rainfall_mm": {
        "label": "Rainfall Intensity", "unit": "mm", "category": "Environmental Hazards",
        "is_controllable": False, "actionable_lever": "Pit sump pumping & drainage channels"
    },
    "ground_condition_index": {
        "label": "Ground Degradation Index", "unit": "pts", "category": "Environmental Hazards",
        "is_controllable": False, "actionable_lever": "Gravel dressing and road maintenance"
    },
    "soil_moisture_index": {
        "label": "Soil Moisture Saturation", "unit": "index", "category": "Environmental Hazards",
        "is_controllable": False, "actionable_lever": "Slope stability monitoring"
    },
    "temperature_c": {
        "label": "Ambient Temperature", "unit": "°C", "category": "Environmental Hazards",
        "is_controllable": False, "actionable_lever": "Heat-stress shift break rotation"
    },
    "haul_distance_km": {
        "label": "Haul Route Distance", "unit": "km", "category": "Mine Geography",
        "is_controllable": False, "actionable_lever": "In-pit dump planning (medium-term)"
    },
    "fragmentation_index": {
        "label": "Rock Fragmentation Index", "unit": "pts", "category": "Material Characteristics",
        "is_controllable": False, "actionable_lever": "Drill & blast pattern design (medium-term)"
    },
    "ore_feed_tonnes": {
        "label": "ROM Ore Feed Volume", "unit": "t", "category": "Material Characteristics",
        "is_controllable": False, "actionable_lever": "Crusher feed bin blending"
    },
    "ore_grade_pct": {
        "label": "Ore Feed Grade", "unit": "%", "category": "Material Characteristics",
        "is_controllable": False, "actionable_lever": "Selective mining face extraction"
    },
    "waste_to_ore_ratio": {
        "label": "Waste-to-Ore Stripping Ratio", "unit": "ratio", "category": "Material Characteristics",
        "is_controllable": False, "actionable_lever": "Pit pushback sequencing"
    },
}

def get_tree_explainer(model) -> shap.TreeExplainer:
    global _EXPLAINER, _EXPLAINER_MODEL_ID
    model_id = id(model)
    if _EXPLAINER is None or _EXPLAINER_MODEL_ID != model_id:
        _EXPLAINER = shap.TreeExplainer(model)
        _EXPLAINER_MODEL_ID = model_id
    return _EXPLAINER

def explain_production_prediction(
    model,
    feature_row: pd.DataFrame | np.ndarray,
    feature_names: list[str],
    top_k: int = 8
) -> dict:
    explainer = get_tree_explainer(model)

    if isinstance(feature_row, pd.DataFrame):
        df_row = feature_row[feature_names].copy()
    else:
        df_row = pd.DataFrame(np.array(feature_row).reshape(1, -1), columns=feature_names)

    shap_result = explainer(df_row)
    shap_vals = shap_result.values[0]
    base_val = float(explainer.expected_value)
    pred_val = float(base_val + np.sum(shap_vals))

    contributions = []
    for idx, col in enumerate(feature_names):
        val = float(shap_vals[idx])
        raw_val = float(df_row.iloc[0, idx])
        meta = FEATURE_METADATA.get(col, {
            "label": col.replace("_", " ").title(),
            "unit": "",
            "category": "Site Indicator" if col.startswith("mine_") else "General",
            "is_controllable": False,
            "actionable_lever": "N/A"
        })
        contributions.append({
            "feature": col,
            "label": meta["label"],
            "category": meta["category"],
            "unit": meta["unit"],
            "is_controllable": meta["is_controllable"],
            "actionable_lever": meta["actionable_lever"],
            "feature_value": raw_val,
            "shap_impact_tonnes": round(val, 2),
            "direction": "downward" if val < 0 else "upward"
        })

    sorted_all = sorted(contributions, key=lambda x: abs(x["shap_impact_tonnes"]), reverse=True)

    negative_drivers = [d for d in sorted_all if d["shap_impact_tonnes"] < -2.0][:top_k]
    positive_drivers = [d for d in sorted_all if d["shap_impact_tonnes"] > 2.0][:top_k]

    controllable_negatives = [d for d in negative_drivers if d["is_controllable"]]
    external_negatives = [d for d in negative_drivers if not d["is_controllable"]]

    return {
        "base_value_tpd": round(base_val, 1),
        "prediction_tpd": round(pred_val, 1),
        "top_negative_drivers": negative_drivers,
        "top_positive_drivers": positive_drivers,
        "controllable_negative_drivers": controllable_negatives,
        "external_negative_drivers": external_negatives,
        "all_contributions": sorted_all,
        "methodology": "TreeSHAP Local Attribution",
        "disclaimer": (
            "Model-associated drivers indicate relative statistical contributions to the "
            "XGBoost production forecast. These represent model-associated factors rather "
            "than proven direct causal mechanics."
        )
    }

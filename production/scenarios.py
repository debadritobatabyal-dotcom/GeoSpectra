from typing import Any, Callable

SCENARIOS = {
    "baseline": {
        "name": "Baseline Shift",
        "icon": "🟢",
        "description": "Standard operating baseline under current pit telemetry.",
    },
    "heavy_rainfall": {
        "name": "Heavy Monsoonal Rainfall",
        "icon": "🌧️",
        "description": "Severe storm cell: saturated haul roads, increased rolling resistance, reduced travel speeds, and active drainage delays.",
    },
    "shovel_breakdown": {
        "name": "Primary Shovel Breakdown",
        "icon": "🚜",
        "description": "Sudden mechanical failure of a primary digging unit, forcing truck queueing at remaining faces.",
    },
    "multi_hazard": {
        "name": "Multi-Hazard Compound Event",
        "icon": "⚠️",
        "description": "Compounded event: severe rainfall concurrent with primary shovel hydraulic breakdown and haul route congestion.",
    },
}

def list_scenarios() -> list[dict]:
    return [
        {"key": k, **v} for k, v in SCENARIOS.items()
    ]

def apply_scenario_modifications(inputs: dict[str, Any], scenario_key: str) -> tuple[dict[str, Any], list[dict]]:
    data = dict(inputs)
    mods = []

    def _set(key, new_val, reason):
        old_val = data.get(key)
        data[key] = new_val
        mods.append({
            "parameter": key,
            "original_value": old_val,
            "modified_value": new_val,
            "reason": reason
        })

    def _get(key, default=0.0):
        val = data.get(key, default)
        try:
            return float(val) if val is not None else default
        except (ValueError, TypeError):
            return default

    if scenario_key == "baseline":
        return data, mods

    elif scenario_key == "heavy_rainfall":
        rain = max(38.0, _get("rainfall_mm") + 32.0)
        _set("rainfall_mm", round(rain, 1), "Intense monsoonal downpour")

        weather_disrupt = max(4.0, _get("weather_disruption_hours") + 3.5)
        _set("weather_disruption_hours", round(weather_disrupt, 1), "Suspended operations for safety & lightning")

        soil_m = min(92.0, max(75.0, _get("soil_moisture_index") + 45.0))
        _set("soil_moisture_index", round(soil_m, 1), "Ramp surface water saturation")

        ground_c = min(88.0, max(70.0, _get("ground_condition_index") + 40.0))
        _set("ground_condition_index", round(ground_c, 1), "Degraded road traction & rutting")

        loaded_t = _get("average_loaded_travel_time_min", 8.0) + 3.0
        _set("average_loaded_travel_time_min", round(loaded_t, 1), "Reduced ramp speed on slick wet roads")

        empty_t = _get("average_empty_travel_time_min", 7.0) + 2.0
        _set("average_empty_travel_time_min", round(empty_t, 1), "Cautionary descent travel speeds")

        cycle_t = _get("average_cycle_time_min", 24.0) + 6.0
        _set("average_cycle_time_min", round(cycle_t, 1), "Cumulative travel delay")

        eff_hours = max(12.0, _get("effective_operating_hours", 20.0) - 4.5)
        _set("effective_operating_hours", round(eff_hours, 1), "Lost production hours")

        haul_eff = max(45.0, _get("haulage_efficiency_pct", 80.0) - 22.0)
        _set("haulage_efficiency_pct", round(haul_eff, 1), "Speed and payload derating")

    elif scenario_key == "shovel_breakdown":
        tot_shovels = int(_get("total_shovels", 4))
        cur_active = int(_get("active_shovels", 3))
        new_active = max(1, cur_active - 1)
        _set("active_shovels", new_active, "Hydraulic pump failure on primary excavator")

        cur_avail = int(_get("available_shovels", 3))
        _set("available_shovels", max(1, cur_avail - 1), "Unit transferred to emergency maintenance")

        new_avail_pct = round((max(1, cur_avail - 1) / max(1, tot_shovels)) * 100.0, 1)
        _set("shovel_availability_pct", new_avail_pct, "Fleet availability drop")

        downtime = _get("shovel_downtime_hours", 0.0) + 8.5
        _set("shovel_downtime_hours", round(downtime, 1), "Unscheduled maintenance duration")

        breakdowns = int(_get("shovel_breakdown_events", 0)) + 1
        _set("shovel_breakdown_events", breakdowns, "Breakdown event recorded")

        queue = _get("queue_delay_min", 2.0) + 4.8
        _set("queue_delay_min", round(queue, 1), "Truck bunching at remaining loading faces")

        cycle_t = _get("average_cycle_time_min", 24.0) + 5.0
        _set("average_cycle_time_min", round(cycle_t, 1), "Extended cycle duration due to queueing")

        load_eff = max(40.0, _get("loading_efficiency_pct", 85.0) - 28.0)
        _set("loading_efficiency_pct", round(load_eff, 1), "Loss of loading fleet capacity")

    elif scenario_key == "multi_hazard":

        tot_shovels = int(_get("total_shovels", 4))
        cur_active = int(_get("active_shovels", 3))
        _set("active_shovels", max(1, cur_active - 1), "Emergency shovel breakdown")
        _set("available_shovels", max(1, int(_get("available_shovels", 3)) - 1), "Unit in maintenance bay")
        _set("shovel_downtime_hours", round(_get("shovel_downtime_hours", 0.0) + 9.0, 1), "Shovel repairs")
        _set("shovel_breakdown_events", int(_get("shovel_breakdown_events", 0)) + 1, "Breakdown incident")

        rain = max(42.0, _get("rainfall_mm") + 35.0)
        _set("rainfall_mm", round(rain, 1), "Torrential rainfall")

        weather_disrupt = max(5.0, _get("weather_disruption_hours") + 4.5)
        _set("weather_disruption_hours", round(weather_disrupt, 1), "Storm shutdown")

        _set("soil_moisture_index", 90.0, "Total waterlogging")
        _set("ground_condition_index", 85.0, "Severe mud & slip hazard")

        _set("queue_delay_min", round(_get("queue_delay_min", 2.0) + 5.5, 1), "Queue bottleneck + slick pads")
        _set("average_cycle_time_min", round(_get("average_cycle_time_min", 24.0) + 9.0, 1), "Severe cycle delay")
        _set("effective_operating_hours", max(10.0, round(_get("effective_operating_hours", 20.0) - 6.0, 1)), "Truncated productive hours")
        _set("haulage_efficiency_pct", max(35.0, round(_get("haulage_efficiency_pct", 80.0) - 35.0, 1)), "Compound efficiency loss")
        _set("loading_efficiency_pct", max(35.0, round(_get("loading_efficiency_pct", 85.0) - 35.0, 1)), "Compound loading derate")

    return data, mods

def simulate_scenario(
    base_inputs: dict[str, Any],
    scenario_key: str,
    predict_fn: Callable[[dict[str, Any]], dict[str, Any]],
    baseline_prediction: dict[str, Any] = None
) -> dict[str, Any]:
    meta = SCENARIOS.get(scenario_key, SCENARIOS["baseline"])
    mod_inputs, applied_mods = apply_scenario_modifications(base_inputs, scenario_key)

    baseline_result = baseline_prediction or predict_fn(base_inputs)

    scenario_result = predict_fn(mod_inputs)

    base_expected = baseline_result["expected_production_tpd"]
    scen_expected = scenario_result["expected_production_tpd"]
    delta_tonnes = round(scen_expected - base_expected, 1)
    delta_pct = round((delta_tonnes / max(1.0, base_expected)) * 100.0, 2)

    return {
        "scenario_key": scenario_key,
        "scenario_name": meta["name"],
        "scenario_icon": meta["icon"],
        "scenario_description": meta["description"],
        "baseline_prediction": baseline_result,
        "scenario_prediction": scenario_result,
        "delta_tonnes": delta_tonnes,
        "delta_pct": delta_pct,
        "modifications": applied_mods,
        "modified_inputs": mod_inputs
    }
